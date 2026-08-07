"""Matched periodic and conditional finite-horizon dynamic-programming tools."""

from __future__ import annotations

import math
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.analysis.cycle_model import CycleOptimum, optimal_engine_on_power

__all__ = [
    "PeriodicDPProblem",
    "PeriodicDPResult",
    "solve_periodic_dp",
    "write_periodic_dp_csv",
]


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class PeriodicDPProblem:
    """Constant matched analytical problem and independent DP grids."""

    demand_bus_kw: float
    engine_max_kw: float
    charge_limit_bus_kw: float
    willans_a_kg_kwh: float
    willans_b_kg_h: float
    source_efficiency: float
    eta_charge: float
    eta_discharge: float
    energy_capacity_kwh: float
    initial_energy_kwh: float
    horizon_s: float
    timestep_s: float
    energy_grid_points: int
    action_grid_points: int

    def __post_init__(self) -> None:
        for name in (
            "demand_bus_kw",
            "engine_max_kw",
            "willans_a_kg_kwh",
            "source_efficiency",
            "eta_charge",
            "eta_discharge",
            "energy_capacity_kwh",
            "horizon_s",
            "timestep_s",
        ):
            _positive(name, getattr(self, name))
        if self.charge_limit_bus_kw < 0.0 or self.willans_b_kg_h < 0.0:
            raise ValueError("charge limit and Willans intercept must be non-negative")
        if not 0.0 < self.source_efficiency <= 1.0:
            raise ValueError("source_efficiency must not exceed one")
        if not 0.0 < self.eta_charge <= 1.0 or not 0.0 < self.eta_discharge <= 1.0:
            raise ValueError("battery efficiencies must not exceed one")
        if not 0.0 < self.initial_energy_kwh < self.energy_capacity_kwh:
            raise ValueError("initial energy must lie inside the DP energy grid")
        if self.energy_grid_points < 11 or self.action_grid_points < 2:
            raise ValueError("DP grids are too small")
        steps = self.horizon_s / self.timestep_s
        if abs(steps - round(steps)) > 1.0e-12:
            raise ValueError("horizon_s must be an integer number of timesteps")


@dataclass(frozen=True)
class PeriodicDPResult:
    """Periodic DP policy metrics and its matched analytical reference."""

    average_fuel_kg_h: float
    total_fuel_kg: float
    mean_engine_power_kw: float
    engine_off_fraction: float
    terminal_energy_error_kwh: float
    terminal_energy_tolerance_kwh: float
    engine_on_power_mean_kw: float
    actions_kw: tuple[float, ...]
    energy_trajectory_kwh: tuple[float, ...]
    analytical: CycleOptimum
    fuel_error_fraction: float
    duty_error: float
    runtime_s: float
    policy_memory_bytes: int
    problem: PeriodicDPProblem


def _energy_rate_kw(problem: PeriodicDPProblem, engine_kw: float) -> float:
    if engine_kw == 0.0:
        return -problem.demand_bus_kw / problem.eta_discharge
    bus = problem.source_efficiency * engine_kw
    difference = bus - problem.demand_bus_kw
    if difference >= 0.0:
        return problem.eta_charge * difference
    return difference / problem.eta_discharge


def solve_periodic_dp(problem: PeriodicDPProblem) -> PeriodicDPResult:
    """Solve the zero-transition-cost periodic problem by backward induction."""
    started = time.perf_counter()
    analytical = optimal_engine_on_power(
        problem.demand_bus_kw,
        problem.engine_max_kw,
        problem.charge_limit_bus_kw,
        problem.willans_a_kg_kwh,
        problem.willans_b_kg_h,
        problem.source_efficiency,
        problem.eta_charge,
        problem.eta_discharge,
    )
    lower = problem.demand_bus_kw / problem.source_efficiency
    upper = analytical.upper_bound_kw
    on_actions = np.linspace(lower, upper, problem.action_grid_points)
    actions = np.concatenate(([0.0], on_actions))
    energy = np.linspace(0.0, problem.energy_capacity_kwh, problem.energy_grid_points)
    energy_step = float(energy[1] - energy[0])
    terminal_tolerance = 0.5 * energy_step
    step_h = problem.timestep_s / 3600.0
    action_cost = np.where(
        actions == 0.0,
        0.0,
        problem.willans_a_kg_kwh * actions + problem.willans_b_kg_h,
    ) * step_h
    deltas = np.asarray([_energy_rate_kw(problem, action) for action in actions]) * step_h
    slopes = []
    for first in range(len(actions)):
        for second in range(first + 1, len(actions)):
            energy_difference = abs(deltas[first] - deltas[second])
            if energy_difference > 0.0:
                slopes.append(
                    abs(action_cost[first] - action_cost[second]) / energy_difference
                )
    terminal_penalty = math.nextafter(max(slopes), math.inf)
    value = terminal_penalty * abs(energy - problem.initial_energy_kwh)
    step_count = round(problem.horizon_s / problem.timestep_s)
    policy = np.zeros((step_count, problem.energy_grid_points), dtype=np.uint16)

    for index in range(step_count - 1, -1, -1):
        candidates = np.full((len(actions), len(energy)), np.inf)
        for action_index, (delta, cost) in enumerate(zip(deltas, action_cost)):
            next_energy = energy + delta
            valid = (next_energy >= 0.0) & (next_energy <= problem.energy_capacity_kwh)
            if np.any(valid):
                candidates[action_index, valid] = cost + np.interp(
                    next_energy[valid], energy, value
                )
        policy[index] = np.argmin(candidates, axis=0)
        value = np.min(candidates, axis=0)

    current_energy = problem.initial_energy_kwh
    selected_actions = []
    energy_path = [current_energy]
    total_fuel = 0.0
    for index in range(step_count):
        state_index = int(np.argmin(abs(energy - current_energy)))
        action = float(actions[int(policy[index, state_index])])
        selected_actions.append(action)
        total_fuel += (
            0.0
            if action == 0.0
            else (problem.willans_a_kg_kwh * action + problem.willans_b_kg_h) * step_h
        )
        current_energy += _energy_rate_kw(problem, action) * step_h
        energy_path.append(current_energy)
    average_fuel = total_fuel / (problem.horizon_s / 3600.0)
    mean_engine = float(np.mean(selected_actions))
    on = [action for action in selected_actions if action > 0.0]
    off_fraction = selected_actions.count(0.0) / len(selected_actions)
    runtime = time.perf_counter() - started
    return PeriodicDPResult(
        average_fuel_kg_h=average_fuel,
        total_fuel_kg=total_fuel,
        mean_engine_power_kw=mean_engine,
        engine_off_fraction=off_fraction,
        terminal_energy_error_kwh=current_energy - problem.initial_energy_kwh,
        terminal_energy_tolerance_kwh=terminal_tolerance,
        engine_on_power_mean_kw=float(np.mean(on)) if on else 0.0,
        actions_kw=tuple(selected_actions),
        energy_trajectory_kwh=tuple(energy_path),
        analytical=analytical,
        fuel_error_fraction=(average_fuel - analytical.cycle_fuel_kg_h)
        / analytical.cycle_fuel_kg_h,
        duty_error=(1.0 - off_fraction) - analytical.duty_cycle,
        runtime_s=runtime,
        policy_memory_bytes=policy.nbytes,
        problem=problem,
    )


def write_periodic_dp_csv(
    rows: Sequence[tuple[str, PeriodicDPResult]], output_path: str | Path
) -> Path:
    """Write matched-problem refinement metrics without assigning restart counts."""
    records = []
    for scenario, result in rows:
        records.append(
            {
                "scenario": scenario,
                "timestep_s": result.problem.timestep_s,
                "horizon_s": result.problem.horizon_s,
                "energy_grid_points": result.problem.energy_grid_points,
                "action_grid_points": result.problem.action_grid_points,
                "dp_average_fuel_kg_h": result.average_fuel_kg_h,
                "analytical_average_fuel_kg_h": result.analytical.cycle_fuel_kg_h,
                "fuel_error_fraction": result.fuel_error_fraction,
                "dp_mean_engine_power_kw": result.mean_engine_power_kw,
                "analytical_mean_engine_power_kw": (
                    result.analytical.duty_cycle * result.analytical.engine_on_kw
                ),
                "dp_engine_off_fraction": result.engine_off_fraction,
                "analytical_engine_off_fraction": 1.0 - result.analytical.duty_cycle,
                "duty_error": result.duty_error,
                "terminal_energy_error_kwh": result.terminal_energy_error_kwh,
                "terminal_energy_tolerance_kwh": result.terminal_energy_tolerance_kwh,
                "restart_count": "undefined",
                "runtime_s": result.runtime_s,
                "policy_memory_bytes": result.policy_memory_bytes,
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
