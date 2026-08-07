"""Extract and replay a SoC-threshold approximation to a DP policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.analysis.finite_horizon_dp import (
    FiniteHorizonDPBounds,
    FiniteHorizonDPResult,
)
from src.analysis.thermostat_comparison import (
    ThermostatEndpointPolicyInterval,
    tune_thermostat_endpoint_interval,
)
from src.control.thermostat import (
    TerminalStrategy,
    ThermostatParameters,
    ThermostatState,
)
from src.simulation.simulator import Aircraft, TimeStep

__all__ = ["ThermostatReduction", "reduce_dp_to_thermostat"]


@dataclass(frozen=True)
class ThermostatReduction:
    """DP transition statistics and unequal-energy reduced-policy replay."""

    extracted_soc_low: float
    extracted_soc_high: float
    restart_soc_range: tuple[float, float]
    shutdown_soc_range: tuple[float, float]
    restart_demand_range_kw: tuple[float, float]
    shutdown_demand_range_kw: tuple[float, float]
    early_late_restart_soc_shift: float
    early_late_shutdown_soc_shift: float
    thermostat_endpoint_policy_interval: ThermostatEndpointPolicyInterval | None
    thermostat_policy_fuel_values_kg: tuple[float, float] | None
    dp_policy_fuel_values_kg: tuple[float, float]
    endpoint_target_surrounded: bool
    fuel_gap_status: str
    preview_dependent: bool
    replay_status: str


def _transition_samples(
    result: FiniteHorizonDPResult,
    steps: Sequence[TimeStep],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    restarts: list[tuple[float, float, float]] = []
    shutdowns: list[tuple[float, float, float]] = []
    previous_on = result.problem.initial_engine_on
    duration_s = sum(step.dt_s for step in steps)
    elapsed_s = 0.0
    for index, (power, step) in enumerate(
        zip(result.engine_power_trajectory_kw, steps)
    ):
        current_on = power > 0.0
        sample = (
            result.soc_trajectory[index],
            step.bus_demand_kw,
            elapsed_s / duration_s,
        )
        if not previous_on and current_on:
            restarts.append(sample)
        elif previous_on and not current_on:
            shutdowns.append(sample)
        previous_on = current_on
        elapsed_s += step.dt_s
    return restarts, shutdowns


def _range(values: Sequence[float]) -> tuple[float, float]:
    return min(values), max(values)


def _early_late_shift(samples: Sequence[tuple[float, float, float]]) -> float:
    early = [soc for soc, _, fraction in samples if fraction < 0.5]
    late = [soc for soc, _, fraction in samples if fraction >= 0.5]
    if not early or not late:
        return 0.0
    return float(np.median(late) - np.median(early))


def reduce_dp_to_thermostat(
    dp: FiniteHorizonDPBounds,
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
) -> ThermostatReduction:
    """Fit independent thresholds and report their endpoint-policy interval."""
    entries = tuple(steps)
    restart_samples: list[tuple[float, float, float]] = []
    shutdown_samples: list[tuple[float, float, float]] = []
    for result in (dp.lower_energy_policy, dp.upper_energy_policy):
        restarts, shutdowns = _transition_samples(result, entries)
        restart_samples.extend(restarts)
        shutdown_samples.extend(shutdowns)
    if not restart_samples or not shutdown_samples:
        problem = dp.lower_energy_policy.problem
        missing = []
        if not restart_samples:
            missing.append("restart")
        if not shutdown_samples:
            missing.append("shutdown")
        return ThermostatReduction(
            extracted_soc_low=aircraft.battery.soc_min,
            extracted_soc_high=1.0,
            restart_soc_range=(math.nan, math.nan),
            shutdown_soc_range=(math.nan, math.nan),
            restart_demand_range_kw=(math.nan, math.nan),
            shutdown_demand_range_kw=(math.nan, math.nan),
            early_late_restart_soc_shift=0.0,
            early_late_shutdown_soc_shift=0.0,
            thermostat_endpoint_policy_interval=None,
            thermostat_policy_fuel_values_kg=None,
            dp_policy_fuel_values_kg=dp.policy_fuel_values_kg,
            endpoint_target_surrounded=False,
            fuel_gap_status=(
                "invalid across unequal terminal energy; policy fuels are descriptive only"
            ),
            preview_dependent=problem.target_energy_change_kwh < 0.0,
            replay_status="unresolved: missing " + " and ".join(missing) + " transitions",
        )
    restart_soc = [sample[0] for sample in restart_samples]
    shutdown_soc = [sample[0] for sample in shutdown_samples]
    soc_low = max(float(np.median(restart_soc)), aircraft.battery.soc_min)
    soc_high = float(np.median(shutdown_soc))
    if soc_low >= soc_high:
        soc_low = float(np.quantile(restart_soc, 0.25))
        soc_high = float(np.quantile(shutdown_soc, 0.75))
    if soc_low >= soc_high:
        raise RuntimeError("DP switching surface is not ordered as a thermostat")
    problem = dp.lower_energy_policy.problem
    preview = problem.target_energy_change_kwh < 0.0
    parameters = ThermostatParameters(
        soc_low=soc_low,
        soc_high=soc_high,
        minimum_on_time_s=problem.minimum_on_time_s,
        minimum_off_time_s=problem.minimum_off_time_s,
        restart_fuel_kg=problem.restart_fuel_kg,
        engine_on_power_kw=None,
        terminal_strategy=(
            TerminalStrategy.HORIZON_AWARE if preview else TerminalStrategy.CAUSAL
        ),
    )
    initial_state = ThermostatState(
        problem.initial_engine_on,
        max(problem.minimum_on_time_s, problem.minimum_off_time_s),
    )
    failures = []
    thermostat = None
    for parameter_name in ("soc_low", "soc_high"):
        try:
            thermostat = tune_thermostat_endpoint_interval(
                entries,
                aircraft,
                parameters,
                initial_soc=problem.initial_soc,
                initial_state=initial_state,
                target_energy_change_kwh=problem.target_energy_change_kwh,
                parameter_name=parameter_name,
            )
            break
        except RuntimeError as error:
            failures.append(str(error))
    thermostat_fuel = (
        thermostat.policy_fuel_values_kg if thermostat is not None else None
    )
    dp_fuel = dp.policy_fuel_values_kg
    matched = False
    if thermostat is not None:
        matched = (
            min(
                thermostat.lower_energy_policy.endpoint_energy_change_kwh,
                thermostat.upper_energy_policy.endpoint_energy_change_kwh,
            )
            <= problem.target_energy_change_kwh
            <= max(
                thermostat.lower_energy_policy.endpoint_energy_change_kwh,
                thermostat.upper_energy_policy.endpoint_energy_change_kwh,
            )
        )
    return ThermostatReduction(
        extracted_soc_low=soc_low,
        extracted_soc_high=soc_high,
        restart_soc_range=_range(restart_soc),
        shutdown_soc_range=_range(shutdown_soc),
        restart_demand_range_kw=_range([sample[1] for sample in restart_samples]),
        shutdown_demand_range_kw=_range([sample[1] for sample in shutdown_samples]),
        early_late_restart_soc_shift=_early_late_shift(restart_samples),
        early_late_shutdown_soc_shift=_early_late_shift(shutdown_samples),
        thermostat_endpoint_policy_interval=thermostat,
        thermostat_policy_fuel_values_kg=thermostat_fuel,
        dp_policy_fuel_values_kg=dp_fuel,
        endpoint_target_surrounded=matched,
        fuel_gap_status=(
            "invalid across unequal terminal energy; policy fuels are descriptive only"
        ),
        preview_dependent=preview,
        replay_status=(
            "endpoint_energy_interval_found"
            if matched
            else "unresolved: " + " | ".join(failures)
        ),
    )
