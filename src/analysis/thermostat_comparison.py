"""Thermostat replay, endpoint-policy intervals, and transition diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from src.analysis.replay_comparison import ConstraintEncounter
from src.control.thermostat import (
    TerminalStrategy,
    ThermostatParameters,
    ThermostatRegime,
    ThermostatState,
    thermostat_step,
)
from src.models.atmosphere import atmosphere
from src.simulation.simulator import Aircraft, TimeStep

__all__ = [
    "ThermostatEndpointPolicyInterval",
    "ThermostatOptimisation",
    "ThermostatReplay",
    "replay_thermostat",
    "optimise_thermostat_endpoint_interval",
    "tune_thermostat_endpoint_interval",
    "write_thermostat_replays_csv",
]

TERMINAL_TARGET_TOLERANCE_KWH = 1.0e-6


@dataclass(frozen=True)
class ThermostatReplay:
    """Integrated thermostat metrics over a fixed exogenous trajectory."""

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
    ledger_residual_kwh: float
    target_energy_change_kwh: float
    terminal_target_residual_kwh: float
    battery_bus_charge_kwh: float
    battery_bus_discharge_kwh: float
    battery_ohmic_loss_kwh: float
    engine_off_fraction: float
    restart_count: int
    restarts_per_flight_hour: float
    on_durations_s: tuple[float, ...]
    off_durations_s: tuple[float, ...]
    cycling_fraction: float
    continuous_engine_fraction: float
    battery_assisted_fraction: float
    terminal_depletion_fraction: float
    engine_on_power_mean_kw: float
    engine_on_power_minimum_kw: float
    engine_on_power_maximum_kw: float
    engine_on_power_samples_kw: tuple[float, ...]
    constraint_encounters: tuple[ConstraintEncounter, ...]
    timestep_s: float
    timing_quantisation_bound_s: float
    policy_hash: str
    dwell_violation_count: int
    scope: str
    battery_mode: str
    parameters: ThermostatParameters
    initial_state: ThermostatState
    terminal_state: ThermostatState


@dataclass(frozen=True)
class ThermostatEndpointPolicyInterval:
    """Unequal-energy threshold policies surrounding one target."""

    parameter_name: str
    lower_parameter: float
    upper_parameter: float
    lower_energy_policy: ThermostatReplay
    upper_energy_policy: ThermostatReplay
    exact_match: bool

    @property
    def parameter_width(self) -> float:
        return self.upper_parameter - self.lower_parameter

    @property
    def endpoint_energy_width_kwh(self) -> float:
        return abs(
            self.upper_energy_policy.endpoint_energy_change_kwh
            - self.lower_energy_policy.endpoint_energy_change_kwh
        )

    @property
    def policy_fuel_values_kg(self) -> tuple[float, float]:
        return (
            self.lower_energy_policy.fuel_consumed_kg,
            self.upper_energy_policy.fuel_consumed_kg,
        )


@dataclass(frozen=True)
class ThermostatOptimisation:
    """Smallest endpoint-energy interval from an upper-threshold sweep."""

    selected: ThermostatEndpointPolicyInterval
    candidate_soc_high: tuple[float, ...]
    unresolved_soc_high: tuple[float, ...]
    candidate_policy_fuel_values_kg: tuple[tuple[float, float], ...]
    selection_basis: str


def _summarise_encounters(
    observations: dict[tuple[str, str], list[tuple[float, float, float]]],
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
        for (direction, limit), values in sorted(observations.items())
    )


def _run_durations(
    modes: Sequence[tuple[bool, float]], initial_on: bool
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    on: list[float] = []
    off: list[float] = []
    current = initial_on
    elapsed = 0.0
    for engine_on, dt_s in modes:
        if engine_on != current and elapsed > 0.0:
            (on if current else off).append(elapsed)
            elapsed = 0.0
            current = engine_on
        elif engine_on != current:
            current = engine_on
        elapsed += dt_s
    if elapsed > 0.0:
        (on if current else off).append(elapsed)
    return tuple(on), tuple(off)


def _off_dwell_is_feasible(
    steps: Sequence[TimeStep],
    start_index: int,
    aircraft: Aircraft,
    soc: float,
    minimum_off_time_s: float,
) -> bool:
    """Check the complete prospective hard-OFF dwell against future demand."""
    elapsed = 0.0
    candidate_soc = soc
    for step in steps[start_index:]:
        if elapsed >= minimum_off_time_s - 1.0e-12:
            break
        state = aircraft.battery.step(candidate_soc, step.bus_demand_kw, step.dt_s)
        if abs(state.power_kw - step.bus_demand_kw) > 1.0e-7:
            return False
        candidate_soc = state.soc
        elapsed += step.dt_s
    return elapsed >= minimum_off_time_s - 1.0e-12


def replay_thermostat(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    parameters: ThermostatParameters,
    *,
    initial_soc: float,
    initial_state: ThermostatState,
    target_energy_change_kwh: float,
) -> ThermostatReplay:
    """Replay one explicit-state thermostat over logged demand and altitude."""
    entries = tuple(steps)
    if not entries:
        raise ValueError("steps must not be empty")
    duration_s = sum(step.dt_s for step in entries)
    duration_h = duration_s / 3600.0
    start_energy = float(aircraft.battery.stored_energy_kwh(initial_soc))
    target_absolute = start_energy + float(target_energy_change_kwh)
    soc = float(initial_soc)
    state = initial_state
    elapsed_s = 0.0
    engine_fuel = 0.0
    restart_fuel = 0.0
    internal_change = 0.0
    charge_bus = 0.0
    discharge_bus = 0.0
    ohmic_loss = 0.0
    min_soc = soc
    max_soc = soc
    modes: list[tuple[bool, float]] = []
    regime_time = {regime: 0.0 for regime in ThermostatRegime}
    on_power_samples: list[float] = []
    on_power_weighted = 0.0
    on_time_s = 0.0
    encountered: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    policy_signature: list[str] = []
    dwell_violations = 0

    for step_index, step in enumerate(entries):
        pre_soc = soc
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        horizon_aware = parameters.terminal_strategy is TerminalStrategy.HORIZON_AWARE
        decision = thermostat_step(
            parameters,
            state,
            demand_bus_kw=step.bus_demand_kw,
            soc=soc,
            sigma=sigma,
            dt_s=step.dt_s,
            engine=aircraft.engine,
            battery=aircraft.battery,
            powertrain=aircraft.powertrain,
            time_to_go_s=duration_s - elapsed_s if horizon_aware else None,
            terminal_energy_target_kwh=target_absolute if horizon_aware else None,
            off_dwell_feasible=(
                _off_dwell_is_feasible(
                    entries,
                    step_index,
                    aircraft,
                    soc,
                    parameters.minimum_off_time_s,
                )
                if horizon_aware
                else None
            ),
        )
        if not decision.feasible:
            raise RuntimeError("thermostat produced an infeasible battery split")
        battery_state = aircraft.battery.step(
            soc, decision.battery_bus_kw, step.dt_s
        )
        if abs(battery_state.power_kw - decision.battery_bus_kw) > 1.0e-6:
            raise RuntimeError("battery could not reproduce thermostat dispatch")
        scale_h = step.dt_s / 3600.0
        engine_fuel += decision.fuel_flow_kg_s * step.dt_s
        restart_fuel += decision.restart_fuel_kg
        internal_kw = (
            battery_state.open_circuit_voltage_v * battery_state.current_a / 1000.0
        )
        internal_change -= internal_kw * scale_h
        charge_bus += max(-battery_state.power_kw, 0.0) * scale_h
        discharge_bus += max(battery_state.power_kw, 0.0) * scale_h
        ohmic_loss += battery_state.ohmic_loss_kw * scale_h
        regime_time[decision.regime] += step.dt_s
        dwell_violations += int(decision.dwell_violation)
        policy_signature.append(
            f"{int(not decision.engine_off)}:{decision.engine_shaft_kw:.9f}"
        )
        modes.append((not decision.engine_off, step.dt_s))
        if not decision.engine_off:
            on_power_samples.append(decision.engine_shaft_kw)
            on_power_weighted += decision.engine_shaft_kw * step.dt_s
            on_time_s += step.dt_s
        if battery_state.active_limit != "none" and abs(battery_state.power_kw) > 1.0e-10:
            direction = "charge" if battery_state.power_kw < 0.0 else "discharge"
            for limit in battery_state.active_limit.split("_and_"):
                encountered.setdefault((direction, limit), []).append(
                    (step.time_s, pre_soc, abs(battery_state.power_kw))
                )
        if decision.active_constraint in {"engine_max", "battery_charge_limit"}:
            encountered.setdefault(("engine", decision.active_constraint), []).append(
                (step.time_s, pre_soc, decision.engine_shaft_kw)
            )
        soc = battery_state.soc
        state = decision.next_state
        elapsed_s += step.dt_s
        min_soc = min(min_soc, soc)
        max_soc = max(max_soc, soc)

    endpoint = float(aircraft.battery.stored_energy_kwh(soc)) - start_energy
    on_durations, off_durations = _run_durations(modes, initial_state.engine_on)
    total_fuel = engine_fuel + restart_fuel
    return ThermostatReplay(
        fuel_consumed_kg=total_fuel,
        average_fuel_rate_kg_h=total_fuel / duration_h,
        engine_fuel_kg=engine_fuel,
        restart_fuel_kg=restart_fuel,
        initial_soc=initial_soc,
        terminal_soc=soc,
        minimum_soc=min_soc,
        maximum_soc=max_soc,
        endpoint_energy_change_kwh=endpoint,
        integrated_energy_change_kwh=internal_change,
        ledger_residual_kwh=endpoint - internal_change,
        target_energy_change_kwh=target_energy_change_kwh,
        terminal_target_residual_kwh=endpoint - target_energy_change_kwh,
        battery_bus_charge_kwh=charge_bus,
        battery_bus_discharge_kwh=discharge_bus,
        battery_ohmic_loss_kwh=ohmic_loss,
        engine_off_fraction=sum(dt for on, dt in modes if not on) / duration_s,
        restart_count=state.restart_count - initial_state.restart_count,
        restarts_per_flight_hour=(
            state.restart_count - initial_state.restart_count
        )
        / duration_h,
        on_durations_s=on_durations,
        off_durations_s=off_durations,
        cycling_fraction=regime_time[ThermostatRegime.CYCLING] / duration_s,
        continuous_engine_fraction=(
            regime_time[ThermostatRegime.CONTINUOUS] / duration_s
        ),
        battery_assisted_fraction=(
            regime_time[ThermostatRegime.BATTERY_ASSISTED] / duration_s
        ),
        terminal_depletion_fraction=(
            regime_time[ThermostatRegime.TERMINAL_DEPLETION] / duration_s
        ),
        engine_on_power_mean_kw=(
            on_power_weighted / on_time_s if on_time_s > 0.0 else 0.0
        ),
        engine_on_power_minimum_kw=min(on_power_samples, default=0.0),
        engine_on_power_maximum_kw=max(on_power_samples, default=0.0),
        engine_on_power_samples_kw=tuple(on_power_samples),
        constraint_encounters=_summarise_encounters(encountered),
        timestep_s=max(step.dt_s for step in entries),
        timing_quantisation_bound_s=max(step.dt_s for step in entries),
        policy_hash=hashlib.sha256("|".join(policy_signature).encode("ascii")).hexdigest()[:16],
        dwell_violation_count=dwell_violations,
        scope="conditional loiter replay over an exogenous trajectory",
        battery_mode=aircraft.battery.battery_mode.value,
        parameters=parameters,
        initial_state=initial_state,
        terminal_state=state,
    )


def tune_thermostat_endpoint_interval(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    parameters: ThermostatParameters,
    *,
    initial_soc: float,
    initial_state: ThermostatState,
    target_energy_change_kwh: float,
    parameter_name: str = "soc_low",
    samples: int = 81,
    terminal_target_tolerance_kwh: float = TERMINAL_TARGET_TOLERANCE_KWH,
    max_bisections: int = 12,
) -> ThermostatEndpointPolicyInterval:
    """Find unequal-energy policies around a target without pricing the interval."""
    if parameter_name not in {"soc_low", "soc_high"}:
        raise ValueError("parameter_name must be soc_low or soc_high")
    if samples < 3:
        raise ValueError("samples must be at least three")
    if terminal_target_tolerance_kwh <= 0.0:
        raise ValueError("terminal_target_tolerance_kwh must be positive")
    if max_bisections < 1:
        raise ValueError("max_bisections must be positive")
    if parameters.soc_low < aircraft.battery.soc_min:
        raise ValueError("soc_low must not lie below the battery cutoff")
    if parameter_name == "soc_low":
        values = np.linspace(
            aircraft.battery.soc_min,
            parameters.soc_high - 1.0e-4,
            samples,
        )
    else:
        values = np.linspace(parameters.soc_low + 1.0e-4, 1.0, samples)
    results = []
    for value in values:
        candidate = replace(parameters, **{parameter_name: float(value)})
        result = replay_thermostat(
            steps,
            aircraft,
            candidate,
            initial_soc=initial_soc,
            initial_state=initial_state,
            target_energy_change_kwh=target_energy_change_kwh,
        )
        results.append((float(value), result))
        if abs(result.terminal_target_residual_kwh) <= terminal_target_tolerance_kwh:
            return ThermostatEndpointPolicyInterval(
                parameter_name, float(value), float(value), result, result, True
            )
    pairs = []
    for left, right in zip(results[:-1], results[1:]):
        if (
            left[1].terminal_target_residual_kwh
            * right[1].terminal_target_residual_kwh
            < 0.0
        ):
            pairs.append((left, right))
    if not pairs:
        nearest = min(
            results, key=lambda item: abs(item[1].terminal_target_residual_kwh)
        )
        raise RuntimeError(
            f"{parameter_name} did not bracket target; nearest residual "
            f"{nearest[1].terminal_target_residual_kwh:.9g} kWh"
        )
    (left_value, left), (right_value, right) = min(
        pairs, key=lambda pair: pair[1][0] - pair[0][0]
    )
    for _ in range(max_bisections):
        middle_value = 0.5 * (left_value + right_value)
        middle = replay_thermostat(
            steps,
            aircraft,
            replace(parameters, **{parameter_name: middle_value}),
            initial_soc=initial_soc,
            initial_state=initial_state,
            target_energy_change_kwh=target_energy_change_kwh,
        )
        if (
            abs(middle.terminal_target_residual_kwh)
            <= terminal_target_tolerance_kwh
        ):
            return ThermostatEndpointPolicyInterval(
                parameter_name,
                middle_value,
                middle_value,
                middle,
                middle,
                True,
            )
        previous_pair = (left.policy_hash, right.policy_hash)
        if (
            left.terminal_target_residual_kwh
            * middle.terminal_target_residual_kwh
            < 0.0
        ):
            right_value, right = middle_value, middle
        else:
            left_value, left = middle_value, middle
        if (left.policy_hash, right.policy_hash) == previous_pair:
            break
    lower = min(
        (left, right), key=lambda result: result.endpoint_energy_change_kwh
    )
    upper = max(
        (left, right), key=lambda result: result.endpoint_energy_change_kwh
    )
    return ThermostatEndpointPolicyInterval(
        parameter_name, left_value, right_value, lower, upper, False
    )


def _append_threshold_checkpoint(
    path: Path,
    soc_high: float,
    interval: ThermostatEndpointPolicyInterval | None,
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "soc_high": soc_high,
        "status": status,
        "lower_energy_kwh": (
            interval.lower_energy_policy.endpoint_energy_change_kwh
            if interval is not None
            else ""
        ),
        "upper_energy_kwh": (
            interval.upper_energy_policy.endpoint_energy_change_kwh
            if interval is not None
            else ""
        ),
        "lower_terminal_target_residual_kwh": (
            interval.lower_energy_policy.terminal_target_residual_kwh
            if interval is not None
            else ""
        ),
        "upper_terminal_target_residual_kwh": (
            interval.upper_energy_policy.terminal_target_residual_kwh
            if interval is not None
            else ""
        ),
        "lower_policy_fuel_kg": (
            interval.lower_energy_policy.fuel_consumed_kg
            if interval is not None
            else ""
        ),
        "upper_policy_fuel_kg": (
            interval.upper_energy_policy.fuel_consumed_kg
            if interval is not None
            else ""
        ),
        "endpoint_energy_interval_width_kwh": (
            interval.endpoint_energy_width_kwh if interval is not None else ""
        ),
    }
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def optimise_thermostat_endpoint_interval(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    parameters: ThermostatParameters,
    *,
    initial_soc: float,
    initial_state: ThermostatState,
    target_energy_change_kwh: float,
    soc_high_values: Sequence[float],
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> ThermostatOptimisation:
    """Sweep upper thresholds with immediate resumable scalar checkpoints."""
    values = tuple(float(value) for value in soc_high_values)
    if not values:
        raise ValueError("soc_high_values must not be empty")
    candidates: list[tuple[float, ThermostatEndpointPolicyInterval]] = []
    unresolved: list[float] = []
    completed: set[float] = set()
    checkpoint_rows: list[dict[str, str]] = []
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint is not None and checkpoint.exists():
        with checkpoint.open(newline="", encoding="utf-8") as stream:
            checkpoint_rows = list(csv.DictReader(stream))
        completed = {float(row["soc_high"]) for row in checkpoint_rows}
    for high in values:
        if not parameters.soc_low < high <= 1.0:
            raise ValueError("each upper threshold must exceed the supplied lower threshold")
        if high in completed:
            continue
        candidate = replace(parameters, soc_high=high)
        try:
            interval = tune_thermostat_endpoint_interval(
                steps,
                aircraft,
                candidate,
                initial_soc=initial_soc,
                initial_state=initial_state,
                target_energy_change_kwh=target_energy_change_kwh,
            )
        except RuntimeError:
            unresolved.append(high)
            if checkpoint is not None:
                _append_threshold_checkpoint(checkpoint, high, None, "unresolved")
            continue
        candidates.append((high, interval))
        if checkpoint is not None:
            _append_threshold_checkpoint(checkpoint, high, interval, "completed")
    if checkpoint is not None and checkpoint.exists():
        with checkpoint.open(newline="", encoding="utf-8") as stream:
            checkpoint_rows = list(csv.DictReader(stream))
        complete_rows = [row for row in checkpoint_rows if row["status"] == "completed"]
        unresolved = [
            float(row["soc_high"])
            for row in checkpoint_rows
            if row["status"] == "unresolved"
        ]
        if not complete_rows:
            raise RuntimeError("no upper threshold produced an endpoint-energy interval")
        best_row = min(
            complete_rows,
            key=lambda row: float(row["endpoint_energy_interval_width_kwh"]),
        )
        best_high = float(best_row["soc_high"])
        selected = next(
            (interval for high, interval in candidates if high == best_high),
            None,
        )
        if selected is None:
            selected = tune_thermostat_endpoint_interval(
                steps,
                aircraft,
                replace(parameters, soc_high=best_high),
                initial_soc=initial_soc,
                initial_state=initial_state,
                target_energy_change_kwh=target_energy_change_kwh,
            )
        candidate_highs = tuple(float(row["soc_high"]) for row in complete_rows)
        candidate_fuels = tuple(
            (
                float(row["lower_policy_fuel_kg"]),
                float(row["upper_policy_fuel_kg"]),
            )
            for row in complete_rows
        )
    else:
        if not candidates:
            raise RuntimeError("no upper threshold produced an endpoint-energy interval")
        _, selected = min(
            candidates,
            key=lambda item: (
                not item[1].exact_match,
                item[1].endpoint_energy_width_kwh,
                min(item[1].policy_fuel_values_kg),
            ),
        )
        candidate_highs = tuple(high for high, _ in candidates)
        candidate_fuels = tuple(
            interval.policy_fuel_values_kg for _, interval in candidates
        )
    return ThermostatOptimisation(
        selected=selected,
        candidate_soc_high=candidate_highs,
        unresolved_soc_high=tuple(unresolved),
        candidate_policy_fuel_values_kg=candidate_fuels,
        selection_basis="exact target first, then smallest endpoint-energy interval",
    )


def write_thermostat_replays_csv(
    rows: Sequence[tuple[str, ThermostatReplay]], output_path: str | Path
) -> Path:
    """Write thermostat point and unequal-energy policy results."""
    entries = []
    for label, result in rows:
        row = asdict(result)
        row["scenario"] = label
        for key in (
            "on_durations_s",
            "off_durations_s",
            "engine_on_power_samples_kw",
            "constraint_encounters",
            "parameters",
            "initial_state",
            "terminal_state",
        ):
            row[key] = json.dumps(row[key], sort_keys=True)
        entries.append(row)
    if not entries:
        raise ValueError("rows must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    return path
