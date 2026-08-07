"""Conditional finite-horizon loiter DP with explicit primal and dual bounds."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.analysis.replay_comparison import ConstraintEncounter
from src.models.atmosphere import atmosphere
from src.simulation.simulator import Aircraft, TimeStep

__all__ = [
    "FiniteHorizonDPBounds",
    "FiniteHorizonDPProblem",
    "FiniteHorizonDPResult",
    "clear_transition_kernel_cache",
    "plot_finite_horizon_policy",
    "run_finite_horizon_scenarios",
    "solve_finite_horizon_dp",
    "write_finite_horizon_dp_csv",
]

_POWER_TOLERANCE_KW = 1.0e-7
_KERNEL_CACHE: dict[str, "_TransitionKernel"] = {}


@dataclass(frozen=True)
class FiniteHorizonDPProblem:
    """Exogenous conditional-loiter trajectory and discrete policy grids."""

    steps: tuple[TimeStep, ...]
    aircraft: Aircraft
    initial_soc: float
    target_energy_change_kwh: float
    restart_fuel_kg: float
    minimum_on_time_s: float
    minimum_off_time_s: float
    soc_grid_points: int
    action_grid_points: int
    initial_engine_on: bool = True
    initial_remaining_dwell_s: float = 0.0
    dwell_semantics: str = "hard"
    max_backward_inductions: int = 32
    scenario_name: str = "unnamed"

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("steps must not be empty")
        nominal = self.steps[0].dt_s
        if any(abs(step.dt_s - nominal) > 1.0e-10 for step in self.steps[:-1]):
            raise ValueError("only the final replay interval may be shorter")
        if not 0.0 < self.steps[-1].dt_s <= nominal + 1.0e-10:
            raise ValueError("final replay interval must be positive and no longer than nominal")
        if not self.aircraft.engine.allow_shutdown:
            raise ValueError("finite-horizon DP requires genuine engine shutdown")
        if self.aircraft.battery.battery_mode.value != "legacy":
            raise ValueError("primary finite-horizon DP uses the verified legacy battery")
        if not self.aircraft.battery.soc_min <= self.initial_soc <= 1.0:
            raise ValueError("initial_soc lies outside the battery bounds")
        if self.restart_fuel_kg < 0.0:
            raise ValueError("restart_fuel_kg must be non-negative")
        if self.minimum_on_time_s < 0.0 or self.minimum_off_time_s < 0.0:
            raise ValueError("minimum dwell times must be non-negative")
        if self.initial_remaining_dwell_s < 0.0:
            raise ValueError("initial remaining dwell must be non-negative")
        if self.soc_grid_points < 11 or self.action_grid_points < 3:
            raise ValueError("DP grids are too small")
        if self.dwell_semantics != "hard":
            raise ValueError("finite-horizon DP currently supports hard dwell only")
        if self.max_backward_inductions < 4:
            raise ValueError("max_backward_inductions must be at least four")
        start = float(self.aircraft.battery.stored_energy_kwh(self.initial_soc))
        target = start + self.target_energy_change_kwh
        lower = float(
            self.aircraft.battery.stored_energy_kwh(self.aircraft.battery.soc_min)
        )
        upper = float(self.aircraft.battery.stored_energy_kwh(1.0))
        if not lower <= target <= upper:
            raise ValueError("terminal energy target lies outside battery bounds")

    @property
    def timestep_s(self) -> float:
        return self.steps[0].dt_s


@dataclass(frozen=True)
class FiniteHorizonDPResult:
    """One discrete DP policy and its fixed-action continuous replay."""

    fuel_consumed_kg: float
    average_fuel_rate_kg_h: float
    engine_fuel_kg: float
    restart_fuel_kg: float
    initial_soc: float
    terminal_soc: float
    discrete_terminal_soc: float
    minimum_soc: float
    maximum_soc: float
    endpoint_energy_change_kwh: float
    discrete_endpoint_energy_change_kwh: float
    integrated_energy_change_kwh: float
    ledger_residual_kwh: float
    target_energy_change_kwh: float
    terminal_target_residual_kwh: float
    discrete_terminal_target_residual_kwh: float
    terminal_shadow_price_kg_kwh: float | None
    lagrangian_value_kg: float | None
    engine_off_fraction: float
    restart_count: int
    dwell_violation_count: int
    on_durations_s: tuple[float, ...]
    off_durations_s: tuple[float, ...]
    engine_on_power_mean_kw: float
    battery_charge_kwh: float
    battery_discharge_kwh: float
    battery_ohmic_loss_kwh: float
    constraint_encounters: tuple[ConstraintEncounter, ...]
    soc_trajectory: tuple[float, ...]
    discrete_soc_trajectory: tuple[float, ...]
    engine_power_trajectory_kw: tuple[float, ...]
    depletion_before_final_tenth_fraction: float
    continuous_constraints_satisfied: bool
    continuous_constraint_violations: tuple[str, ...]
    policy_hash: str
    kernel_build_runtime_s: float
    policy_solve_runtime_s: float
    runtime_s: float
    policy_memory_bytes: int
    problem: FiniteHorizonDPProblem


@dataclass(frozen=True)
class FiniteHorizonDPBounds:
    """Supported endpoint policies and valid bounds for the discrete DP model."""

    lower_energy_policy: FiniteHorizonDPResult
    upper_energy_policy: FiniteHorizonDPResult
    feasible_upper_bound_policy: FiniteHorizonDPResult | None
    target_energy_change_kwh: float
    endpoint_energy_interval_kwh: tuple[float, float]
    endpoint_energy_interval_width_kwh: float
    policy_fuel_values_kg: tuple[float, float]
    endpoint_target_bracketed: bool
    dual_lower_bound_kg: float
    feasible_upper_bound_kg: float | None
    optimality_gap_kg: float | None
    bound_scope: str
    effective_soc_grid_points: int
    backward_inductions: int
    termination_reason: str
    kernel_cache_hit: bool
    kernel_build_runtime_s: float
    policy_solve_runtime_s: float


@dataclass(frozen=True)
class _TransitionKernel:
    soc_grid: np.ndarray
    energy_grid: np.ndarray
    initial_soc_index: int
    target_soc_index: int
    action_power_kw: np.ndarray
    next_soc: np.ndarray
    next_soc_index: np.ndarray
    feasible: np.ndarray
    fuel_kg: np.ndarray
    battery_power_kw: np.ndarray
    max_dwell_steps: int
    on_dwell_steps: int
    off_dwell_steps: int
    build_runtime_s: float
    memory_bytes: int
    cache_key: str


@dataclass(frozen=True)
class _DiscretePolicy:
    action_indices: tuple[int, ...]
    soc_indices: tuple[int, ...]
    engine_states: tuple[bool, ...]
    engine_fuel_kg: float
    restart_count: int
    restart_fuel_kg: float
    policy_hash: str
    policy_memory_bytes: int


def _soc_for_energy(aircraft: Aircraft, target_kwh: float) -> float:
    battery = aircraft.battery
    lower = battery.soc_min
    upper = 1.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if float(battery.stored_energy_kwh(middle)) < target_kwh:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _anchored_soc_grid(problem: FiniteHorizonDPProblem) -> np.ndarray:
    battery = problem.aircraft.battery
    start_energy = float(battery.stored_energy_kwh(problem.initial_soc))
    target_soc = _soc_for_energy(
        problem.aircraft, start_energy + problem.target_energy_change_kwh
    )
    base = np.linspace(battery.soc_min, 1.0, problem.soc_grid_points)
    return np.unique(np.concatenate((base, (problem.initial_soc, target_soc))))


def _kernel_key(problem: FiniteHorizonDPProblem) -> str:
    step_data = tuple(
        (step.dt_s, step.altitude_m, step.bus_demand_kw) for step in problem.steps
    )
    payload = repr(
        (
            step_data,
            problem.aircraft.engine,
            problem.aircraft.battery,
            problem.aircraft.powertrain,
            problem.initial_soc,
            problem.target_energy_change_kwh,
            problem.minimum_on_time_s,
            problem.minimum_off_time_s,
            problem.soc_grid_points,
            problem.action_grid_points,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clear_transition_kernel_cache() -> None:
    """Drop the in-process scenario kernel cache."""
    _KERNEL_CACHE.clear()


def _build_kernel(problem: FiniteHorizonDPProblem) -> _TransitionKernel:
    started = time.perf_counter()
    battery = problem.aircraft.battery
    engine = problem.aircraft.engine
    powertrain = problem.aircraft.powertrain
    soc_grid = _anchored_soc_grid(problem)
    energy_grid = np.asarray(battery.stored_energy_kwh(soc_grid), dtype=float)
    initial_index = int(np.argmin(abs(soc_grid - problem.initial_soc)))
    target_energy = energy_grid[initial_index] + problem.target_energy_change_kwh
    target_index = int(np.argmin(abs(energy_grid - target_energy)))
    action_count = problem.action_grid_points
    shape = (len(problem.steps), action_count, len(soc_grid))
    action_power = np.zeros((len(problem.steps), action_count), dtype=float)
    next_soc = np.empty(shape, dtype=np.float64)
    next_index = np.full(shape, -1, dtype=np.int32)
    feasible = np.zeros(shape, dtype=bool)
    fuel = np.zeros((len(problem.steps), action_count), dtype=float)
    battery_power = np.zeros((len(problem.steps), action_count), dtype=float)

    for time_index, step in enumerate(problem.steps):
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        maximum = engine.max_power_kw(sigma)
        on_actions = np.linspace(engine.idle_power_kw, maximum, action_count - 1)
        if action_count >= 5:
            load_following = float(powertrain.engine_power_for_bus(step.bus_demand_kw))
            charge_boundary = float(
                powertrain.engine_power_for_bus(
                    step.bus_demand_kw
                    + battery.available_charge_kw(problem.initial_soc, step.dt_s)
                )
            )
            on_actions[-3] = min(max(load_following, engine.idle_power_kw), maximum)
            on_actions[-2] = min(max(charge_boundary, engine.idle_power_kw), maximum)
            on_actions.sort()
        action_power[time_index, 1:] = on_actions
        for action_index, command in enumerate(action_power[time_index]):
            engine_state = engine.operate(float(command), sigma)
            actual = 0.0 if action_index == 0 else engine_state.delivered_kw
            bus = step.bus_demand_kw - float(powertrain.bus_power_from_engine(actual))
            battery_power[time_index, action_index] = bus
            fuel[time_index, action_index] = engine_state.fuel_flow_kg_s * step.dt_s
            for soc_index, soc in enumerate(soc_grid):
                state = battery.step(float(soc), bus, step.dt_s)
                reproduced = abs(state.power_kw - bus) <= _POWER_TOLERANCE_KW
                feasible[time_index, action_index, soc_index] = reproduced
                next_soc[time_index, action_index, soc_index] = state.soc
                if reproduced:
                    next_index[time_index, action_index, soc_index] = int(
                        np.argmin(abs(soc_grid - state.soc))
                    )

    on_dwell = math.ceil(problem.minimum_on_time_s / problem.timestep_s)
    off_dwell = math.ceil(problem.minimum_off_time_s / problem.timestep_s)
    arrays = action_power, next_soc, next_index, feasible, fuel, battery_power
    return _TransitionKernel(
        soc_grid=soc_grid,
        energy_grid=energy_grid,
        initial_soc_index=initial_index,
        target_soc_index=target_index,
        action_power_kw=action_power,
        next_soc=next_soc,
        next_soc_index=next_index,
        feasible=feasible,
        fuel_kg=fuel,
        battery_power_kw=battery_power,
        max_dwell_steps=max(on_dwell, off_dwell, 1),
        on_dwell_steps=on_dwell,
        off_dwell_steps=off_dwell,
        build_runtime_s=time.perf_counter() - started,
        memory_bytes=sum(array.nbytes for array in arrays),
        cache_key=_kernel_key(problem),
    )


def _get_kernel(problem: FiniteHorizonDPProblem) -> tuple[_TransitionKernel, bool]:
    key = _kernel_key(problem)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key], True
    kernel = _build_kernel(problem)
    _KERNEL_CACHE[key] = kernel
    return kernel, False


def _backward_policy(
    problem: FiniteHorizonDPProblem,
    kernel: _TransitionKernel,
    terminal_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    state_count = kernel.max_dwell_steps + 1
    soc_count = len(kernel.soc_grid)
    action_count = problem.action_grid_points
    value = np.broadcast_to(terminal_cost, (2, state_count, soc_count)).copy()
    policy = np.zeros((len(problem.steps), 2, state_count, soc_count), dtype=np.uint16)
    for time_index in range(len(problem.steps) - 1, -1, -1):
        next_value = value
        value = np.full_like(next_value, np.inf)
        for current_on in (0, 1):
            for remaining in range(state_count):
                candidates = np.full((action_count, soc_count), np.inf)
                for action_index in range(action_count):
                    next_on = int(action_index > 0)
                    if next_on == current_on:
                        next_remaining = max(remaining - 1, 0)
                    elif remaining == 0:
                        dwell = (
                            kernel.on_dwell_steps if next_on else kernel.off_dwell_steps
                        )
                        next_remaining = max(dwell - 1, 0)
                    else:
                        continue
                    valid = kernel.feasible[time_index, action_index]
                    if not np.any(valid):
                        continue
                    soc_indices = np.flatnonzero(valid)
                    successors = kernel.next_soc_index[
                        time_index, action_index, soc_indices
                    ]
                    restart = (
                        problem.restart_fuel_kg if not current_on and next_on else 0.0
                    )
                    candidates[action_index, soc_indices] = (
                        kernel.fuel_kg[time_index, action_index]
                        + restart
                        + next_value[next_on, next_remaining, successors]
                    )
                policy[time_index, current_on, remaining] = np.argmin(candidates, axis=0)
                value[current_on, remaining] = np.min(candidates, axis=0)
    return policy, value, time.perf_counter() - started


def _extract_policy(
    problem: FiniteHorizonDPProblem,
    kernel: _TransitionKernel,
    policy: np.ndarray,
) -> _DiscretePolicy:
    soc_index = kernel.initial_soc_index
    current_on = int(problem.initial_engine_on)
    remaining = min(
        math.ceil(problem.initial_remaining_dwell_s / problem.timestep_s),
        kernel.max_dwell_steps,
    )
    actions: list[int] = []
    states: list[bool] = []
    soc_indices = [soc_index]
    engine_fuel = 0.0
    restart_count = 0
    for time_index in range(len(problem.steps)):
        action_index = int(policy[time_index, current_on, remaining, soc_index])
        next_on = int(action_index > 0)
        if next_on != current_on and remaining > 0:
            raise RuntimeError("DP policy violates hard dwell")
        if not kernel.feasible[time_index, action_index, soc_index]:
            raise RuntimeError("DP policy selected an infeasible grid transition")
        if next_on == current_on:
            remaining = max(remaining - 1, 0)
        else:
            dwell = kernel.on_dwell_steps if next_on else kernel.off_dwell_steps
            remaining = max(dwell - 1, 0)
        restarted = int(not current_on and next_on)
        restart_count += restarted
        engine_fuel += kernel.fuel_kg[time_index, action_index]
        soc_index = int(kernel.next_soc_index[time_index, action_index, soc_index])
        current_on = next_on
        actions.append(action_index)
        states.append(bool(current_on))
        soc_indices.append(soc_index)
    payload = bytes(np.asarray(actions, dtype=np.uint16))
    return _DiscretePolicy(
        action_indices=tuple(actions),
        soc_indices=tuple(soc_indices),
        engine_states=tuple(states),
        engine_fuel_kg=engine_fuel,
        restart_count=restart_count,
        restart_fuel_kg=restart_count * problem.restart_fuel_kg,
        policy_hash=hashlib.sha256(payload).hexdigest()[:16],
        policy_memory_bytes=policy.nbytes,
    )


def _dwell_runs(
    states: Sequence[bool], steps: Sequence[TimeStep], initial_on: bool
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    on: list[float] = []
    off: list[float] = []
    current = initial_on
    elapsed = 0.0
    for state, step in zip(states, steps):
        if state != current:
            if elapsed > 0.0:
                (on if current else off).append(elapsed)
            current = state
            elapsed = 0.0
        elapsed += step.dt_s
    if elapsed > 0.0:
        (on if current else off).append(elapsed)
    return tuple(on), tuple(off)


def _encounters(
    observed: dict[tuple[str, str], list[tuple[float, float, float]]]
) -> tuple[ConstraintEncounter, ...]:
    return tuple(
        ConstraintEncounter(
            direction=direction,
            limit=limit,
            timestep_count=len(values),
            first_time_s=min(value[0] for value in values),
            minimum_soc=min(value[1] for value in values),
            maximum_soc=max(value[1] for value in values),
            minimum_power_kw=min(value[2] for value in values),
            maximum_power_kw=max(value[2] for value in values),
        )
        for (direction, limit), values in sorted(observed.items())
    )


def _replay_policy(
    problem: FiniteHorizonDPProblem,
    kernel: _TransitionKernel,
    discrete: _DiscretePolicy,
    *,
    shadow_price: float | None,
    solve_runtime_s: float,
    kernel_build_runtime_s: float,
) -> FiniteHorizonDPResult:
    started = time.perf_counter()
    battery = problem.aircraft.battery
    engine = problem.aircraft.engine
    powertrain = problem.aircraft.powertrain
    soc = problem.initial_soc
    internal_change = charge = discharge = ohmic = 0.0
    soc_path = [soc]
    powers: list[float] = []
    energy_changes: list[float] = []
    feasible = True
    violations: list[str] = []
    observed: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for time_index, (step, action_index) in enumerate(
        zip(problem.steps, discrete.action_indices)
    ):
        pre_soc = soc
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        command = float(kernel.action_power_kw[time_index, action_index])
        engine_state = engine.operate(command, sigma)
        actual = 0.0 if action_index == 0 else engine_state.delivered_kw
        bus = step.bus_demand_kw - float(powertrain.bus_power_from_engine(actual))
        battery_state = battery.step(soc, bus, step.dt_s)
        reproduced = abs(battery_state.power_kw - bus) <= _POWER_TOLERANCE_KW
        feasible = feasible and reproduced
        if not reproduced:
            violations.append(f"battery_power_limit@step_{time_index}")
        if engine_state.power_limited:
            feasible = False
            violations.append(f"engine_power_limit@step_{time_index}")
        scale_h = step.dt_s / 3600.0
        internal_kw = battery_state.open_circuit_voltage_v * battery_state.current_a / 1000.0
        internal_change -= internal_kw * scale_h
        charge += max(-battery_state.power_kw, 0.0) * scale_h
        discharge += max(battery_state.power_kw, 0.0) * scale_h
        ohmic += battery_state.ohmic_loss_kw * scale_h
        change = float(battery.stored_energy_kwh(battery_state.soc)) - float(
            battery.stored_energy_kwh(soc)
        )
        energy_changes.append(change)
        if battery_state.active_limit != "none" and abs(bus) > 1.0e-10:
            direction = "charge" if bus < 0.0 else "discharge"
            for limit in battery_state.active_limit.split("_and_"):
                observed.setdefault((direction, limit), []).append(
                    (step.time_s, pre_soc, abs(bus))
                )
        soc = battery_state.soc
        soc_path.append(soc)
        powers.append(actual)

    start_energy = float(battery.stored_energy_kwh(problem.initial_soc))
    endpoint = float(battery.stored_energy_kwh(soc)) - start_energy
    discrete_start = kernel.energy_grid[discrete.soc_indices[0]]
    discrete_endpoint = kernel.energy_grid[discrete.soc_indices[-1]] - discrete_start
    total_fuel = discrete.engine_fuel_kg + discrete.restart_fuel_kg
    target = problem.target_energy_change_kwh
    lagrangian = (
        total_fuel + shadow_price * (discrete_endpoint - target)
        if shadow_price is not None
        else None
    )
    on_durations, off_durations = _dwell_runs(
        discrete.engine_states, problem.steps, problem.initial_engine_on
    )
    depletion = np.maximum(-np.asarray(energy_changes), 0.0)
    split = math.floor(0.9 * len(depletion))
    total_depletion = float(np.sum(depletion))
    early = (
        float(np.sum(depletion[:split])) / total_depletion
        if total_depletion > 0.0
        else 0.0
    )
    duration_h = sum(step.dt_s for step in problem.steps) / 3600.0
    on_powers = [power for power in powers if power > 0.0]
    replay_runtime = time.perf_counter() - started
    return FiniteHorizonDPResult(
        fuel_consumed_kg=total_fuel,
        average_fuel_rate_kg_h=total_fuel / duration_h,
        engine_fuel_kg=discrete.engine_fuel_kg,
        restart_fuel_kg=discrete.restart_fuel_kg,
        initial_soc=problem.initial_soc,
        terminal_soc=soc,
        discrete_terminal_soc=float(kernel.soc_grid[discrete.soc_indices[-1]]),
        minimum_soc=min(soc_path),
        maximum_soc=max(soc_path),
        endpoint_energy_change_kwh=endpoint,
        discrete_endpoint_energy_change_kwh=float(discrete_endpoint),
        integrated_energy_change_kwh=internal_change,
        ledger_residual_kwh=endpoint - internal_change,
        target_energy_change_kwh=target,
        terminal_target_residual_kwh=endpoint - target,
        discrete_terminal_target_residual_kwh=float(discrete_endpoint - target),
        terminal_shadow_price_kg_kwh=shadow_price,
        lagrangian_value_kg=lagrangian,
        engine_off_fraction=discrete.engine_states.count(False) / len(discrete.engine_states),
        restart_count=discrete.restart_count,
        dwell_violation_count=0,
        on_durations_s=on_durations,
        off_durations_s=off_durations,
        engine_on_power_mean_kw=float(np.mean(on_powers)) if on_powers else 0.0,
        battery_charge_kwh=charge,
        battery_discharge_kwh=discharge,
        battery_ohmic_loss_kwh=ohmic,
        constraint_encounters=_encounters(observed),
        soc_trajectory=tuple(soc_path),
        discrete_soc_trajectory=tuple(
            float(kernel.soc_grid[index]) for index in discrete.soc_indices
        ),
        engine_power_trajectory_kw=tuple(powers),
        depletion_before_final_tenth_fraction=early,
        continuous_constraints_satisfied=feasible,
        continuous_constraint_violations=tuple(violations),
        policy_hash=discrete.policy_hash,
        kernel_build_runtime_s=kernel_build_runtime_s,
        policy_solve_runtime_s=solve_runtime_s,
        runtime_s=kernel_build_runtime_s + solve_runtime_s + replay_runtime,
        policy_memory_bytes=discrete.policy_memory_bytes + kernel.memory_bytes,
        problem=problem,
    )


def _solve_terminal_cost(
    problem: FiniteHorizonDPProblem,
    kernel: _TransitionKernel,
    terminal_cost: np.ndarray,
    *,
    shadow_price: float | None,
    kernel_build_runtime_s: float,
) -> FiniteHorizonDPResult:
    policy, value, solve_runtime = _backward_policy(problem, kernel, terminal_cost)
    initial_remaining = min(
        math.ceil(problem.initial_remaining_dwell_s / problem.timestep_s),
        kernel.max_dwell_steps,
    )
    initial_value = value[
        int(problem.initial_engine_on), initial_remaining, kernel.initial_soc_index
    ]
    if not math.isfinite(float(initial_value)):
        raise RuntimeError("terminal DP state is unreachable from the initial state")
    discrete = _extract_policy(problem, kernel, policy)
    return _replay_policy(
        problem,
        kernel,
        discrete,
        shadow_price=shadow_price,
        solve_runtime_s=solve_runtime,
        kernel_build_runtime_s=kernel_build_runtime_s,
    )


def _solve_shadow_price(
    problem: FiniteHorizonDPProblem,
    kernel: _TransitionKernel,
    shadow_price: float,
    kernel_build_runtime_s: float = 0.0,
) -> FiniteHorizonDPResult:
    target_absolute = (
        kernel.energy_grid[kernel.initial_soc_index] + problem.target_energy_change_kwh
    )
    terminal_cost = shadow_price * (kernel.energy_grid - target_absolute)
    return _solve_terminal_cost(
        problem,
        kernel,
        terminal_cost,
        shadow_price=shadow_price,
        kernel_build_runtime_s=kernel_build_runtime_s,
    )


def _solve_exact_discrete_target(
    problem: FiniteHorizonDPProblem,
    kernel: _TransitionKernel,
) -> FiniteHorizonDPResult | None:
    terminal_cost = np.full(len(kernel.soc_grid), np.inf)
    terminal_cost[kernel.target_soc_index] = 0.0
    try:
        return _solve_terminal_cost(
            problem,
            kernel,
            terminal_cost,
            shadow_price=None,
            kernel_build_runtime_s=0.0,
        )
    except RuntimeError:
        return None


def solve_finite_horizon_dp(
    problem: FiniteHorizonDPProblem,
    *,
    progress: Callable[[str], None] | None = print,
) -> FiniteHorizonDPBounds:
    """Maximise a valid discrete-model dual and construct an exact grid target."""
    started = time.perf_counter()
    kernel, cache_hit = _get_kernel(problem)
    build_runtime = 0.0 if cache_hit else kernel.build_runtime_s
    results: dict[float, FiniteHorizonDPResult] = {}
    inductions = 0

    def solve(shadow: float) -> FiniteHorizonDPResult:
        nonlocal inductions
        key = float(shadow)
        if key in results:
            return results[key]
        if inductions >= problem.max_backward_inductions:
            raise RuntimeError("maximum backward inductions reached")
        inductions += 1
        if progress is not None:
            progress(
                f"scenario={problem.scenario_name} grid={len(kernel.soc_grid)}x"
                f"{problem.action_grid_points} multiplier_iteration={inductions} "
                f"lambda={key:.9g} elapsed_s={time.perf_counter() - started:.2f}"
            )
        results[key] = _solve_shadow_price(problem, kernel, key)
        return results[key]

    target = problem.target_energy_change_kwh
    scale = problem.aircraft.engine.willans_a / max(
        problem.aircraft.powertrain.source_chain_efficiency, 1.0e-12
    )
    lower_shadow = -scale
    upper_shadow = scale
    upper_energy = solve(lower_shadow)
    lower_energy = solve(upper_shadow)
    while not (
        lower_energy.discrete_endpoint_energy_change_kwh
        <= target
        <= upper_energy.discrete_endpoint_energy_change_kwh
    ):
        if inductions >= problem.max_backward_inductions - 1:
            raise RuntimeError("requested terminal energy is outside the searched DP range")
        if lower_energy.discrete_endpoint_energy_change_kwh > target:
            upper_shadow *= 2.0
            lower_energy = solve(upper_shadow)
        else:
            lower_shadow *= 2.0
            upper_energy = solve(lower_shadow)

    termination = "maximum_backward_inductions"
    while inductions < problem.max_backward_inductions - 1:
        if lower_energy.policy_hash == upper_energy.policy_hash:
            termination = "identical_supported_policy"
            break
        middle_shadow = 0.5 * (lower_shadow + upper_shadow)
        middle = solve(middle_shadow)
        previous_pair = (lower_energy.policy_hash, upper_energy.policy_hash)
        if middle.discrete_endpoint_energy_change_kwh > target:
            lower_shadow, upper_energy = middle_shadow, middle
        else:
            upper_shadow, lower_energy = middle_shadow, middle
        current_pair = (lower_energy.policy_hash, upper_energy.policy_hash)
        if current_pair == previous_pair:
            termination = "repeated_adjacent_policy_pair"
            break
        if abs(lower_energy.discrete_terminal_target_residual_kwh) <= 1.0e-12:
            upper_energy = lower_energy
            termination = "exact_shadow_supported_target"
            break
        if abs(upper_energy.discrete_terminal_target_residual_kwh) <= 1.0e-12:
            lower_energy = upper_energy
            termination = "exact_shadow_supported_target"
            break

    upper_bound = _solve_exact_discrete_target(problem, kernel)
    if upper_bound is not None:
        inductions += 1
    dual = max(
        float(result.lagrangian_value_kg)
        for result in results.values()
        if result.lagrangian_value_kg is not None
    )
    upper_value = upper_bound.fuel_consumed_kg if upper_bound is not None else None
    endpoints = (
        lower_energy.discrete_endpoint_energy_change_kwh,
        upper_energy.discrete_endpoint_energy_change_kwh,
    )
    if progress is not None:
        progress(
            f"scenario={problem.scenario_name} complete inductions={inductions} "
            f"termination={termination} elapsed_s={time.perf_counter() - started:.2f}"
        )
    return FiniteHorizonDPBounds(
        lower_energy_policy=lower_energy,
        upper_energy_policy=upper_energy,
        feasible_upper_bound_policy=upper_bound,
        target_energy_change_kwh=target,
        endpoint_energy_interval_kwh=(min(endpoints), max(endpoints)),
        endpoint_energy_interval_width_kwh=abs(endpoints[1] - endpoints[0]),
        policy_fuel_values_kg=(
            lower_energy.fuel_consumed_kg,
            upper_energy.fuel_consumed_kg,
        ),
        endpoint_target_bracketed=min(endpoints) <= target <= max(endpoints),
        dual_lower_bound_kg=dual,
        feasible_upper_bound_kg=upper_value,
        optimality_gap_kg=upper_value - dual if upper_value is not None else None,
        bound_scope=(
            "discrete SoC/action DP with nearest-grid transitions; continuous replay "
            "is diagnostic and is not covered by these bounds"
        ),
        effective_soc_grid_points=len(kernel.soc_grid),
        backward_inductions=inductions,
        termination_reason=termination,
        kernel_cache_hit=cache_hit,
        kernel_build_runtime_s=build_runtime,
        policy_solve_runtime_s=sum(
            result.policy_solve_runtime_s for result in results.values()
        )
        + (upper_bound.policy_solve_runtime_s if upper_bound is not None else 0.0),
    )


def _csv_records(
    rows: Sequence[tuple[str, FiniteHorizonDPBounds]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for scenario, bounds in rows:
        policies = (
            ("lower_energy_policy", bounds.lower_energy_policy),
            ("upper_energy_policy", bounds.upper_energy_policy),
        )
        if bounds.feasible_upper_bound_policy is not None:
            policies += (("feasible_upper_bound_policy", bounds.feasible_upper_bound_policy),)
        for role, result in policies:
            records.append(
                {
                    "scenario": scenario,
                    "policy_role": role,
                    "status": "exploratory_discretised_conditional_loiter",
                    "timestep_s": result.problem.timestep_s,
                    "requested_soc_grid_points": result.problem.soc_grid_points,
                    "effective_soc_grid_points": bounds.effective_soc_grid_points,
                    "action_grid_points": result.problem.action_grid_points,
                    "dwell_semantics": result.problem.dwell_semantics,
                    "minimum_on_time_s": result.problem.minimum_on_time_s,
                    "minimum_off_time_s": result.problem.minimum_off_time_s,
                    "restart_fuel_per_start_kg": result.problem.restart_fuel_kg,
                    "target_energy_change_kwh": bounds.target_energy_change_kwh,
                    "continuous_endpoint_energy_change_kwh": result.endpoint_energy_change_kwh,
                    "discrete_endpoint_energy_change_kwh": result.discrete_endpoint_energy_change_kwh,
                    "ledger_residual_kwh": result.ledger_residual_kwh,
                    "terminal_target_residual_kwh": result.terminal_target_residual_kwh,
                    "discrete_terminal_target_residual_kwh": result.discrete_terminal_target_residual_kwh,
                    "endpoint_energy_interval_width_kwh": bounds.endpoint_energy_interval_width_kwh,
                    "endpoint_target_bracketed": bounds.endpoint_target_bracketed,
                    "policy_fuel_kg": result.fuel_consumed_kg,
                    "dual_lower_bound_kg": bounds.dual_lower_bound_kg,
                    "feasible_upper_bound_kg": bounds.feasible_upper_bound_kg,
                    "optimality_gap_kg": bounds.optimality_gap_kg,
                    "terminal_shadow_price_kg_kwh": result.terminal_shadow_price_kg_kwh,
                    "lagrangian_value_kg": result.lagrangian_value_kg,
                    "policy_hash": result.policy_hash,
                    "continuous_constraints_satisfied": (
                        result.continuous_constraints_satisfied
                    ),
                    "continuous_constraint_violations": json.dumps(
                        result.continuous_constraint_violations
                    ),
                    "fuel_consumed_kg": result.fuel_consumed_kg,
                    "restart_count": result.restart_count,
                    "dwell_violation_count": result.dwell_violation_count,
                    "engine_off_fraction": result.engine_off_fraction,
                    "engine_on_power_mean_kw": result.engine_on_power_mean_kw,
                    "initial_soc": result.initial_soc,
                    "terminal_soc": result.terminal_soc,
                    "discrete_terminal_soc": result.discrete_terminal_soc,
                    "kernel_build_runtime_s": bounds.kernel_build_runtime_s,
                    "policy_solve_runtime_s": bounds.policy_solve_runtime_s,
                    "backward_inductions": bounds.backward_inductions,
                    "termination_reason": bounds.termination_reason,
                    "kernel_cache_hit": bounds.kernel_cache_hit,
                    "on_durations_s": json.dumps(result.on_durations_s),
                    "off_durations_s": json.dumps(result.off_durations_s),
                    "constraint_encounters": json.dumps(
                        [encounter.__dict__ for encounter in result.constraint_encounters],
                        sort_keys=True,
                    ),
                    "soc_trajectory": json.dumps(result.soc_trajectory),
                    "discrete_soc_trajectory": json.dumps(result.discrete_soc_trajectory),
                    "engine_power_trajectory_kw": json.dumps(
                        result.engine_power_trajectory_kw
                    ),
                    "bound_scope": bounds.bound_scope,
                }
            )
    return records


def write_finite_horizon_dp_csv(
    rows: Sequence[tuple[str, FiniteHorizonDPBounds]], output_path: str | Path
) -> Path:
    """Write supported policies and valid discrete-model bounds."""
    records = _csv_records(rows)
    if not records:
        raise ValueError("rows must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def run_finite_horizon_scenarios(
    scenarios: Sequence[tuple[str, FiniteHorizonDPProblem]],
    output_path: str | Path,
    *,
    resume: bool = True,
    progress: Callable[[str], None] | None = print,
) -> tuple[tuple[str, FiniteHorizonDPBounds], ...]:
    """Solve scenarios with an immediate append-only CSV checkpoint."""
    path = Path(output_path)
    completed: set[str] = set()
    if resume and path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            completed = {row["scenario"] for row in csv.DictReader(stream)}
    solved: list[tuple[str, FiniteHorizonDPBounds]] = []
    for scenario, problem in scenarios:
        if scenario in completed:
            if progress is not None:
                progress(f"scenario={scenario} resume=skipped")
            continue
        named = problem if problem.scenario_name == scenario else dataclass_replace(
            problem, scenario_name=scenario
        )
        bounds = solve_finite_horizon_dp(named, progress=progress)
        records = _csv_records(((scenario, bounds),))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(records)
            stream.flush()
        solved.append((scenario, bounds))
    return tuple(solved)


def dataclass_replace(problem: FiniteHorizonDPProblem, **changes: object) -> FiniteHorizonDPProblem:
    """Local typed replacement without exposing mutable scenario state."""
    values = {field: getattr(problem, field) for field in problem.__dataclass_fields__}
    values.update(changes)
    return FiniteHorizonDPProblem(**values)


def plot_finite_horizon_policy(
    bounds: FiniteHorizonDPBounds, output_path: str | Path
) -> Path:
    """Plot the two unequal-energy supported policies without implying fuel bounds."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 5.5), sharex=True)
    for label, result in (
        ("lower-energy policy", bounds.lower_energy_policy),
        ("upper-energy policy", bounds.upper_energy_policy),
    ):
        time_h = np.arange(len(result.soc_trajectory)) * result.problem.timestep_s / 3600.0
        axes[0].plot(time_h, result.soc_trajectory, label=label)
        axes[1].step(
            time_h[:-1], result.engine_power_trajectory_kw, where="post", label=label
        )
    axes[0].set_ylabel("SoC [-]")
    axes[1].set_ylabel("Engine shaft power [kW]")
    axes[1].set_xlabel("Conditional loiter time [h]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
