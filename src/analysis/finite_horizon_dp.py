"""Conditional finite-horizon loiter dynamic programming."""

from __future__ import annotations

import math
import time
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.analysis.replay_comparison import ConstraintEncounter
from src.models.atmosphere import atmosphere
from src.simulation.simulator import Aircraft, TimeStep

__all__ = [
    "FiniteHorizonDPBracket",
    "FiniteHorizonDPProblem",
    "FiniteHorizonDPResult",
    "plot_finite_horizon_policy",
    "solve_finite_horizon_dp",
    "write_finite_horizon_dp_csv",
]


@dataclass(frozen=True)
class FiniteHorizonDPProblem:
    """Exogenous loiter trajectory and independently refinable DP grids."""

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

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("steps must not be empty")
        nominal = self.steps[0].dt_s
        if any(
            abs(step.dt_s - nominal) > 1.0e-10
            for step in self.steps[:-1]
        ) or not 0.0 < self.steps[-1].dt_s <= nominal + 1.0e-10:
            raise ValueError("only the final replay interval may be shorter")
        if not self.aircraft.engine.allow_shutdown:
            raise ValueError("finite-horizon DP requires genuine engine shutdown")
        if not self.aircraft.battery.battery_mode.value == "legacy":
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
    """One grid-terminal DP policy replayed through the battery model."""

    fuel_consumed_kg: float
    average_fuel_rate_kg_h: float
    engine_fuel_kg: float
    restart_fuel_kg: float
    initial_soc: float
    terminal_soc: float
    minimum_soc: float
    maximum_soc: float
    endpoint_energy_change_kwh: float
    integrated_energy_change_kwh: float
    target_energy_change_kwh: float
    requested_target_energy_change_kwh: float
    target_residual_kwh: float
    terminal_shadow_price_kg_kwh: float
    engine_off_fraction: float
    restart_count: int
    on_durations_s: tuple[float, ...]
    off_durations_s: tuple[float, ...]
    engine_on_power_mean_kw: float
    battery_charge_kwh: float
    battery_discharge_kwh: float
    battery_ohmic_loss_kwh: float
    constraint_encounters: tuple[ConstraintEncounter, ...]
    soc_trajectory: tuple[float, ...]
    engine_power_trajectory_kw: tuple[float, ...]
    depletion_before_final_tenth_fraction: float
    runtime_s: float
    policy_memory_bytes: int
    problem: FiniteHorizonDPProblem


@dataclass(frozen=True)
class FiniteHorizonDPBracket:
    """Policies ending at the adjacent grid energies around one target."""

    lower_energy_result: FiniteHorizonDPResult
    upper_energy_result: FiniteHorizonDPResult
    target_energy_change_kwh: float
    target_grid_width_kwh: float
    endpoint_bracketed: bool

    @property
    def fuel_interval_kg(self) -> tuple[float, float]:
        values = (
            self.lower_energy_result.fuel_consumed_kg,
            self.upper_energy_result.fuel_consumed_kg,
        )
        return min(values), max(values)

    @property
    def endpoint_energy_interval_kwh(self) -> tuple[float, float]:
        values = (
            self.lower_energy_result.endpoint_energy_change_kwh,
            self.upper_energy_result.endpoint_energy_change_kwh,
        )
        return min(values), max(values)


@dataclass(frozen=True)
class _TransitionKernel:
    soc_grid: np.ndarray
    energy_grid: np.ndarray
    action_power_kw: np.ndarray
    next_soc: np.ndarray
    feasible: np.ndarray
    fuel_kg: np.ndarray
    internal_change_kwh: np.ndarray
    battery_power_kw: np.ndarray
    ohmic_loss_kwh: np.ndarray
    max_dwell_steps: int
    on_dwell_steps: int
    off_dwell_steps: int
    build_runtime_s: float
    memory_bytes: int


def _dwell_runs(
    states: Sequence[bool], timestep_s: float, initial_engine_on: bool
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    on: list[float] = []
    off: list[float] = []
    current = initial_engine_on
    elapsed = 0.0
    for state in states:
        if state != current:
            if elapsed > 0.0:
                (on if current else off).append(elapsed)
            current = state
            elapsed = 0.0
        elapsed += timestep_s
    if elapsed > 0.0:
        (on if current else off).append(elapsed)
    return tuple(on), tuple(off)


def _encounters(
    observed: dict[tuple[str, str], list[tuple[float, float, float]]],
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


def _build_kernel(problem: FiniteHorizonDPProblem) -> _TransitionKernel:
    started = time.perf_counter()
    steps = problem.steps
    battery = problem.aircraft.battery
    engine = problem.aircraft.engine
    powertrain = problem.aircraft.powertrain
    timestep_s = problem.timestep_s
    soc_grid = np.linspace(battery.soc_min, 1.0, problem.soc_grid_points)
    energy_grid = np.asarray(battery.stored_energy_kwh(soc_grid), dtype=float)
    action_count = problem.action_grid_points
    shape = (len(steps), action_count, problem.soc_grid_points)
    action_power = np.zeros((len(steps), action_count), dtype=float)
    next_soc = np.empty(shape, dtype=np.float64)
    feasible = np.zeros(shape, dtype=bool)
    fuel = np.zeros((len(steps), action_count), dtype=float)
    internal = np.zeros(shape, dtype=np.float64)
    battery_power = np.zeros((len(steps), action_count), dtype=float)
    ohmic = np.zeros(shape, dtype=np.float64)

    for time_index, step in enumerate(steps):
        step_h = step.dt_s / 3600.0
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        maximum = engine.max_power_kw(sigma)
        on_actions = np.linspace(engine.idle_power_kw, maximum, action_count - 1)
        if action_count >= 5:
            load_following = float(
                powertrain.engine_power_for_bus(step.bus_demand_kw)
            )
            charge_boundary = float(
                powertrain.engine_power_for_bus(
                    step.bus_demand_kw + battery.available_charge_kw(problem.initial_soc)
                )
            )
            on_actions[-3] = min(max(load_following, engine.idle_power_kw), maximum)
            on_actions[-2] = min(max(charge_boundary, engine.idle_power_kw), maximum)
            on_actions.sort()
        action_power[time_index, 1:] = on_actions
        for action_index, command in enumerate(action_power[time_index]):
            engine_state = engine.operate(float(command), sigma)
            actual_power = 0.0 if action_index == 0 else engine_state.delivered_kw
            bus_from_engine = float(powertrain.bus_power_from_engine(actual_power))
            requested_battery = step.bus_demand_kw - bus_from_engine
            battery_power[time_index, action_index] = requested_battery
            fuel[time_index, action_index] = engine_state.fuel_flow_kg_s * step.dt_s
            for soc_index, soc in enumerate(soc_grid):
                state = battery.step(float(soc), requested_battery, step.dt_s)
                reproduced = abs(state.power_kw - requested_battery) <= 1.0e-7
                feasible[time_index, action_index, soc_index] = reproduced
                next_soc[time_index, action_index, soc_index] = state.soc
                internal_kw = state.open_circuit_voltage_v * state.current_a / 1000.0
                internal[time_index, action_index, soc_index] = -internal_kw * step_h
                ohmic[time_index, action_index, soc_index] = state.ohmic_loss_kw * step_h

    on_dwell = math.ceil(problem.minimum_on_time_s / timestep_s)
    off_dwell = math.ceil(problem.minimum_off_time_s / timestep_s)
    maximum_dwell = max(on_dwell, off_dwell, 1)
    arrays = (
        action_power,
        next_soc,
        feasible,
        fuel,
        internal,
        battery_power,
        ohmic,
    )
    return _TransitionKernel(
        soc_grid=soc_grid,
        energy_grid=energy_grid,
        action_power_kw=action_power,
        next_soc=next_soc,
        feasible=feasible,
        fuel_kg=fuel,
        internal_change_kwh=internal,
        battery_power_kw=battery_power,
        ohmic_loss_kwh=ohmic,
        max_dwell_steps=maximum_dwell,
        on_dwell_steps=on_dwell,
        off_dwell_steps=off_dwell,
        build_runtime_s=time.perf_counter() - started,
        memory_bytes=sum(array.nbytes for array in arrays),
    )


def _solve_shadow_price(
    problem: FiniteHorizonDPProblem,
    kernel: _TransitionKernel,
    terminal_shadow_price_kg_kwh: float,
) -> FiniteHorizonDPResult:
    started = time.perf_counter()
    state_count = kernel.max_dwell_steps + 1
    soc_count = problem.soc_grid_points
    action_count = problem.action_grid_points
    step_count = len(problem.steps)
    terminal_cost = terminal_shadow_price_kg_kwh * kernel.energy_grid
    value = np.broadcast_to(
        terminal_cost, (2, state_count, soc_count)
    ).copy()
    policy = np.zeros(
        (step_count, 2, state_count, soc_count), dtype=np.uint16
    )

    for time_index in range(step_count - 1, -1, -1):
        next_value = value
        value = np.full_like(next_value, np.inf)
        for current_on in (0, 1):
            for remaining in range(state_count):
                candidates = np.full((action_count, soc_count), np.inf)
                off_feasible = kernel.feasible[time_index, 0]
                for action_index in range(action_count):
                    next_on = int(action_index > 0)
                    if next_on == current_on:
                        next_remaining = max(remaining - 1, 0)
                        allowed = np.ones(soc_count, dtype=bool)
                    elif current_on:
                        next_remaining = max(kernel.off_dwell_steps - 1, 0)
                        allowed = np.full(soc_count, remaining == 0)
                    else:
                        next_remaining = max(kernel.on_dwell_steps - 1, 0)
                        allowed = np.full(soc_count, remaining == 0) | ~off_feasible
                    valid = allowed & kernel.feasible[time_index, action_index]
                    if not np.any(valid):
                        continue
                    restart = (
                        problem.restart_fuel_kg
                        if not current_on and next_on
                        else 0.0
                    )
                    candidates[action_index, valid] = (
                        kernel.fuel_kg[time_index, action_index]
                        + restart
                        + np.interp(
                            kernel.next_soc[time_index, action_index, valid],
                            kernel.soc_grid,
                            next_value[next_on, next_remaining],
                        )
                    )
                policy[time_index, current_on, remaining] = np.argmin(
                    candidates, axis=0
                )
                value[current_on, remaining] = np.min(candidates, axis=0)

    initial_index = int(np.argmin(abs(kernel.soc_grid - problem.initial_soc)))
    initial_remaining = min(
        math.ceil(problem.initial_remaining_dwell_s / problem.timestep_s),
        kernel.max_dwell_steps,
    )
    initial_on = int(problem.initial_engine_on)
    if not math.isfinite(value[initial_on, initial_remaining, initial_index]):
        raise RuntimeError("terminal grid state is unreachable from the initial state")

    battery = problem.aircraft.battery
    engine = problem.aircraft.engine
    powertrain = problem.aircraft.powertrain
    soc = problem.initial_soc
    current_on = initial_on
    remaining = initial_remaining
    engine_fuel = 0.0
    restart_fuel = 0.0
    internal_change = 0.0
    charge = 0.0
    discharge = 0.0
    ohmic = 0.0
    restart_count = 0
    states: list[bool] = []
    soc_path = [soc]
    power_path: list[float] = []
    energy_changes: list[float] = []
    observed: dict[tuple[str, str], list[tuple[float, float, float]]] = {}

    for time_index, step in enumerate(problem.steps):
        soc_index = int(np.argmin(abs(kernel.soc_grid - soc)))
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        selected = int(policy[time_index, current_on, remaining, soc_index])
        off_bus = step.bus_demand_kw
        off_state = battery.step(soc, off_bus, step.dt_s)
        off_feasible = abs(off_state.power_kw - off_bus) <= 1.0e-6
        chosen = None
        for action_index in sorted(
            range(problem.action_grid_points), key=lambda index: abs(index - selected)
        ):
            next_on = int(action_index > 0)
            if next_on != current_on and remaining > 0:
                safety_restart = not current_on and next_on and not off_feasible
                if not safety_restart:
                    continue
            power = float(kernel.action_power_kw[time_index, action_index])
            candidate_engine = engine.operate(power, sigma)
            candidate_bus = step.bus_demand_kw - float(
                powertrain.bus_power_from_engine(candidate_engine.delivered_kw)
            )
            candidate_battery = battery.step(soc, candidate_bus, step.dt_s)
            if abs(candidate_battery.power_kw - candidate_bus) <= 1.0e-6:
                chosen = (
                    action_index,
                    next_on,
                    candidate_engine,
                    candidate_bus,
                    candidate_battery,
                )
                break
        if chosen is None:
            raise RuntimeError("DP policy has no continuously feasible action")
        action_index, next_on, engine_state, bus, battery_state = chosen
        scale_h = step.dt_s / 3600.0
        engine_fuel += engine_state.fuel_flow_kg_s * step.dt_s
        if not current_on and next_on:
            restart_count += 1
            restart_fuel += problem.restart_fuel_kg
        internal_kw = (
            battery_state.open_circuit_voltage_v * battery_state.current_a / 1000.0
        )
        change = float(battery.stored_energy_kwh(battery_state.soc)) - float(
            battery.stored_energy_kwh(soc)
        )
        energy_changes.append(change)
        internal_change -= internal_kw * scale_h
        charge += max(-battery_state.power_kw, 0.0) * scale_h
        discharge += max(battery_state.power_kw, 0.0) * scale_h
        ohmic += battery_state.ohmic_loss_kw * scale_h
        if battery_state.active_limit != "none" and abs(bus) > 1.0e-10:
            direction = "charge" if bus < 0.0 else "discharge"
            for limit in battery_state.active_limit.split("_and_"):
                observed.setdefault((direction, limit), []).append(
                    (step.time_s, soc, abs(bus))
                )
        if next_on == current_on:
            remaining = max(remaining - 1, 0)
        elif current_on:
            remaining = max(kernel.off_dwell_steps - 1, 0)
        else:
            remaining = max(kernel.on_dwell_steps - 1, 0)
        current_on = next_on
        soc = battery_state.soc
        states.append(bool(current_on))
        power_path.append(engine_state.delivered_kw)
        soc_path.append(soc)

    start_energy = float(battery.stored_energy_kwh(problem.initial_soc))
    endpoint = float(battery.stored_energy_kwh(soc)) - start_energy
    total_fuel = engine_fuel + restart_fuel
    on_durations, off_durations = _dwell_runs(
        states, problem.timestep_s, problem.initial_engine_on
    )
    discharge_magnitudes = np.maximum(-np.asarray(energy_changes), 0.0)
    final_tenth_start = math.floor(0.9 * len(discharge_magnitudes))
    total_depletion = float(np.sum(discharge_magnitudes))
    early_fraction = (
        float(np.sum(discharge_magnitudes[:final_tenth_start])) / total_depletion
        if total_depletion > 0.0
        else 0.0
    )
    duration_h = sum(step.dt_s for step in problem.steps) / 3600.0
    on_powers = [power for power in power_path if power > 0.0]
    return FiniteHorizonDPResult(
        fuel_consumed_kg=total_fuel,
        average_fuel_rate_kg_h=total_fuel / duration_h,
        engine_fuel_kg=engine_fuel,
        restart_fuel_kg=restart_fuel,
        initial_soc=problem.initial_soc,
        terminal_soc=soc,
        minimum_soc=min(soc_path),
        maximum_soc=max(soc_path),
        endpoint_energy_change_kwh=endpoint,
        integrated_energy_change_kwh=internal_change,
        target_energy_change_kwh=problem.target_energy_change_kwh,
        requested_target_energy_change_kwh=problem.target_energy_change_kwh,
        target_residual_kwh=endpoint - problem.target_energy_change_kwh,
        terminal_shadow_price_kg_kwh=terminal_shadow_price_kg_kwh,
        engine_off_fraction=states.count(False) / len(states),
        restart_count=restart_count,
        on_durations_s=on_durations,
        off_durations_s=off_durations,
        engine_on_power_mean_kw=float(np.mean(on_powers)) if on_powers else 0.0,
        battery_charge_kwh=charge,
        battery_discharge_kwh=discharge,
        battery_ohmic_loss_kwh=ohmic,
        constraint_encounters=_encounters(observed),
        soc_trajectory=tuple(soc_path),
        engine_power_trajectory_kw=tuple(power_path),
        depletion_before_final_tenth_fraction=early_fraction,
        runtime_s=time.perf_counter() - started + kernel.build_runtime_s,
        policy_memory_bytes=policy.nbytes + kernel.memory_bytes,
        problem=problem,
    )


def solve_finite_horizon_dp(
    problem: FiniteHorizonDPProblem,
) -> FiniteHorizonDPBracket:
    """Bracket the endpoint with adjacent terminal shadow-price policies."""
    kernel = _build_kernel(problem)
    cache: dict[float, FiniteHorizonDPResult] = {}

    def solve(shadow: float) -> FiniteHorizonDPResult:
        if shadow not in cache:
            cache[shadow] = _solve_shadow_price(problem, kernel, shadow)
        return cache[shadow]

    target = problem.target_energy_change_kwh
    scale = problem.aircraft.engine.willans_a / max(
        problem.aircraft.powertrain.source_chain_efficiency, 1.0e-12
    )
    lower_shadow = -scale
    upper_shadow = scale
    upper_energy = solve(lower_shadow)
    lower_energy = solve(upper_shadow)
    for _ in range(24):
        if (
            lower_energy.endpoint_energy_change_kwh
            <= target
            <= upper_energy.endpoint_energy_change_kwh
        ):
            break
        if lower_energy.endpoint_energy_change_kwh > target:
            upper_shadow *= 2.0
            lower_energy = solve(upper_shadow)
        else:
            lower_shadow *= 2.0
            upper_energy = solve(lower_shadow)
    else:
        raise RuntimeError("requested terminal energy is outside the reachable DP range")

    for _ in range(24):
        middle_shadow = 0.5 * (lower_shadow + upper_shadow)
        middle = solve(middle_shadow)
        if middle.endpoint_energy_change_kwh > target:
            lower_shadow, upper_energy = middle_shadow, middle
        else:
            upper_shadow, lower_energy = middle_shadow, middle
        if (
            lower_energy.endpoint_energy_change_kwh
            == upper_energy.endpoint_energy_change_kwh
        ):
            break
    endpoint_values = (
        lower_energy.endpoint_energy_change_kwh,
        upper_energy.endpoint_energy_change_kwh,
    )
    bracketed = min(endpoint_values) <= problem.target_energy_change_kwh <= max(
        endpoint_values
    )
    return FiniteHorizonDPBracket(
        lower_energy_result=lower_energy,
        upper_energy_result=upper_energy,
        target_energy_change_kwh=problem.target_energy_change_kwh,
        target_grid_width_kwh=abs(endpoint_values[1] - endpoint_values[0]),
        endpoint_bracketed=bracketed,
    )


def write_finite_horizon_dp_csv(
    rows: Sequence[tuple[str, FiniteHorizonDPBracket]], output_path: str | Path
) -> Path:
    """Write both endpoint edges for each explicitly labelled DP scenario."""
    records = []
    for scenario, bracket in rows:
        for edge, result in (
            ("lower_energy", bracket.lower_energy_result),
            ("upper_energy", bracket.upper_energy_result),
        ):
            records.append(
                {
                    "scenario": scenario,
                    "edge": edge,
                    "timestep_s": result.problem.timestep_s,
                    "soc_grid_points": result.problem.soc_grid_points,
                    "action_grid_points": result.problem.action_grid_points,
                    "minimum_on_time_s": result.problem.minimum_on_time_s,
                    "minimum_off_time_s": result.problem.minimum_off_time_s,
                    "restart_fuel_per_start_kg": result.problem.restart_fuel_kg,
                    "target_energy_change_kwh": bracket.target_energy_change_kwh,
                    "endpoint_energy_change_kwh": result.endpoint_energy_change_kwh,
                    "integrated_energy_change_kwh": result.integrated_energy_change_kwh,
                    "endpoint_bracket_width_kwh": bracket.target_grid_width_kwh,
                    "endpoint_bracketed": bracket.endpoint_bracketed,
                    "fuel_consumed_kg": result.fuel_consumed_kg,
                    "average_fuel_rate_kg_h": result.average_fuel_rate_kg_h,
                    "engine_fuel_kg": result.engine_fuel_kg,
                    "restart_fuel_kg": result.restart_fuel_kg,
                    "restart_count": result.restart_count,
                    "engine_off_fraction": result.engine_off_fraction,
                    "engine_on_power_mean_kw": result.engine_on_power_mean_kw,
                    "initial_soc": result.initial_soc,
                    "terminal_soc": result.terminal_soc,
                    "minimum_soc": result.minimum_soc,
                    "maximum_soc": result.maximum_soc,
                    "battery_charge_kwh": result.battery_charge_kwh,
                    "battery_discharge_kwh": result.battery_discharge_kwh,
                    "battery_ohmic_loss_kwh": result.battery_ohmic_loss_kwh,
                    "depletion_before_final_tenth_fraction": (
                        result.depletion_before_final_tenth_fraction
                    ),
                    "terminal_shadow_price_kg_kwh": (
                        result.terminal_shadow_price_kg_kwh
                    ),
                    "runtime_s": result.runtime_s,
                    "policy_memory_bytes": result.policy_memory_bytes,
                    "on_durations_s": json.dumps(result.on_durations_s),
                    "off_durations_s": json.dumps(result.off_durations_s),
                    "constraint_encounters": json.dumps(
                        [encounter.__dict__ for encounter in result.constraint_encounters],
                        sort_keys=True,
                    ),
                    "soc_trajectory": json.dumps(result.soc_trajectory),
                    "engine_power_trajectory_kw": json.dumps(
                        result.engine_power_trajectory_kw
                    ),
                    "scope": "finite-horizon optimum conditional on exogenous trajectory",
                }
            )
    if not records:
        raise ValueError("rows must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def plot_finite_horizon_policy(
    bracket: FiniteHorizonDPBracket, output_path: str | Path
) -> Path:
    """Plot the two endpoint-bracketing SoC and engine-power policies."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 5.5), sharex=True)
    for label, result in (
        ("lower endpoint", bracket.lower_energy_result),
        ("upper endpoint", bracket.upper_energy_result),
    ):
        time_h = np.arange(len(result.soc_trajectory)) * result.problem.timestep_s / 3600.0
        axes[0].plot(time_h, result.soc_trajectory, label=label)
        axes[1].step(
            time_h[:-1],
            result.engine_power_trajectory_kw,
            where="post",
            label=label,
        )
    axes[0].set_ylabel("SoC [-]")
    axes[1].set_ylabel("Engine shaft power [kW]")
    axes[1].set_xlabel("Post-crossing time [h]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
