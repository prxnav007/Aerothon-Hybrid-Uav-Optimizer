"""Extract and replay a SoC-threshold approximation to a DP policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.analysis.finite_horizon_dp import (
    FiniteHorizonDPBracket,
    FiniteHorizonDPResult,
)
from src.analysis.thermostat_comparison import (
    ThermostatEnergyBracket,
    tune_thermostat_energy_bracket,
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
    """DP transition statistics and equal-energy reduced-policy replay."""

    extracted_soc_low: float
    extracted_soc_high: float
    restart_soc_range: tuple[float, float]
    shutdown_soc_range: tuple[float, float]
    restart_demand_range_kw: tuple[float, float]
    shutdown_demand_range_kw: tuple[float, float]
    early_late_restart_soc_shift: float
    early_late_shutdown_soc_shift: float
    thermostat_bracket: ThermostatEnergyBracket | None
    thermostat_fuel_interval_kg: tuple[float, float] | None
    dp_fuel_interval_kg: tuple[float, float]
    thermostat_gap_to_dp_kg: tuple[float, float] | None
    endpoint_energy_matched: bool
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
    dp: FiniteHorizonDPBracket,
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
) -> ThermostatReduction:
    """Fit independent thresholds, then enforce the same endpoint-energy target."""
    entries = tuple(steps)
    restart_samples: list[tuple[float, float, float]] = []
    shutdown_samples: list[tuple[float, float, float]] = []
    for result in (dp.lower_energy_result, dp.upper_energy_result):
        restarts, shutdowns = _transition_samples(result, entries)
        restart_samples.extend(restarts)
        shutdown_samples.extend(shutdowns)
    if not restart_samples or not shutdown_samples:
        raise RuntimeError("DP policy does not contain both switching directions")
    restart_soc = [sample[0] for sample in restart_samples]
    shutdown_soc = [sample[0] for sample in shutdown_samples]
    soc_low = float(np.median(restart_soc))
    soc_high = float(np.median(shutdown_soc))
    if soc_low >= soc_high:
        soc_low = float(np.quantile(restart_soc, 0.25))
        soc_high = float(np.quantile(shutdown_soc, 0.75))
    if soc_low >= soc_high:
        raise RuntimeError("DP switching surface is not ordered as a thermostat")
    problem = dp.lower_energy_result.problem
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
            thermostat = tune_thermostat_energy_bracket(
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
    thermostat_fuel = thermostat.fuel_interval_kg if thermostat is not None else None
    dp_fuel = dp.fuel_interval_kg
    gap = (
        (
            thermostat_fuel[0] - dp_fuel[1],
            thermostat_fuel[1] - dp_fuel[0],
        )
        if thermostat_fuel is not None
        else None
    )
    matched = False
    if thermostat is not None:
        matched = (
            min(
                thermostat.lower_result.endpoint_energy_change_kwh,
                thermostat.upper_result.endpoint_energy_change_kwh,
            )
            <= problem.target_energy_change_kwh
            <= max(
                thermostat.lower_result.endpoint_energy_change_kwh,
                thermostat.upper_result.endpoint_energy_change_kwh,
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
        thermostat_bracket=thermostat,
        thermostat_fuel_interval_kg=thermostat_fuel,
        dp_fuel_interval_kg=dp_fuel,
        thermostat_gap_to_dp_kg=gap,
        endpoint_energy_matched=matched,
        preview_dependent=preview,
        replay_status=(
            "endpoint_energy_bracketed"
            if matched
            else "unresolved: " + " | ".join(failures)
        ),
    )
