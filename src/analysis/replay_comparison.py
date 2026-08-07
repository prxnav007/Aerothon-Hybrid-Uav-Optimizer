"""Energy-normalised replay of a logged demand and mass trajectory."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from src.analysis.cycle_feasibility import battery_efficiencies
from src.analysis.cycle_model import OperatingRegime, classify_regime
from src.control.base import ControlContext, EMSController
from src.control.power_split import solve_split
from src.models.atmosphere import atmosphere
from src.simulation.simulator import Aircraft, TimeStep

__all__ = [
    "ConstraintEncounter",
    "ENERGY_MATCH_TOLERANCE_KWH",
    "EnergyMismatchError",
    "EqualEnergyComparison",
    "PIEnergyBracket",
    "PIReplayTrace",
    "ReplayComparison",
    "StrategyReplay",
    "compare_replays",
    "compare_replays_at_initial_soc",
    "compare_equal_energy_replays",
    "derive_energy_match_tolerance_kwh",
    "replay_pi_ecms",
    "replay_pi_ecms_trace",
    "resample_replay_steps",
    "tune_pi_energy_bracket",
    "validated_fuel_gap",
    "write_equal_energy_comparisons_csv",
    "write_replay_comparison_csv",
    "write_replay_comparisons_csv",
]


# Largest physical-mode ledger residual in the Milestone-1b 60/30/15 s sweep
# was 8.793e-14 kWh. Two independently integrated endpoints can differ by the
# sum of their residuals; the next enclosing decimal is the numerical gate.
MILESTONE1B_MAX_PHYSICAL_RESIDUAL_KWH = 8.79296635503124e-14


def derive_energy_match_tolerance_kwh(residual_kwh: float) -> float:
    """Round a measured positive numerical residual up by one decimal bin."""
    value = abs(float(residual_kwh))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("residual_kwh must be finite and nonzero")
    return 10.0 ** math.ceil(math.log10(value))


ENERGY_MATCH_TOLERANCE_KWH = derive_energy_match_tolerance_kwh(
    2.0 * MILESTONE1B_MAX_PHYSICAL_RESIDUAL_KWH
)


class EnergyMismatchError(ValueError):
    """Raised when a fuel gap is requested across unmatched stored energy."""


@dataclass(frozen=True)
class ConstraintEncounter:
    """Repeated uses of one battery constraint during a replay."""

    direction: str
    limit: str
    timestep_count: int
    first_time_s: float
    minimum_soc: float
    maximum_soc: float
    minimum_power_kw: float
    maximum_power_kw: float


@dataclass(frozen=True)
class PIReplayTrace:
    """PI replay aggregate paired with reconstructed logged steps."""

    result: "StrategyReplay"
    steps: tuple[TimeStep, ...]


@dataclass(frozen=True)
class StrategyReplay:
    """Integrated result for one strategy over a fixed trajectory."""

    strategy: str
    fuel_consumed_kg: float
    average_fuel_rate_kg_h: float
    battery_ohmic_loss_kwh: float
    recirculated_round_trip_loss_kwh: float
    engine_off_fraction: float
    restart_count: int | None
    restart_count_status: str
    initial_soc: float
    terminal_soc: float
    minimum_soc: float
    maximum_soc: float
    battery_energy_change_kwh: float
    internal_ledger_energy_change_kwh: float
    euler_energy_residual_kwh: float
    target_battery_energy_change_kwh: float
    terminal_energy_shortfall_kwh: float
    battery_mode: str
    battery_capacity_kwh: float
    i_charge_max_a: float | None
    i_discharge_max_a: float | None
    terminal_voltage_min_v: float | None
    terminal_voltage_max_v: float | None
    q_nominal_ah: float | None
    timestep_s: float
    constraint_encounters: tuple[ConstraintEncounter, ...]
    calibration_parameter_value: float | None = None
    engine_shaft_power_mean_kw: float | None = None
    engine_shaft_power_minimum_kw: float | None = None
    engine_shaft_power_maximum_kw: float | None = None
    strategy_construction: str = ""


@dataclass(frozen=True)
class ReplayComparison:
    """Three replay results and the requested fuel-gap metrics."""

    strategies: tuple[StrategyReplay, ...]
    normalisation_method: str
    common_initial_soc: float
    target_battery_energy_change_kwh: float
    pi_target_residual_kwh: float
    pi_to_continuous_gap_kg: float
    pi_to_continuous_gap_fraction: float
    pi_to_ideal_cycle_gap_kg: float
    pi_to_ideal_cycle_gap_fraction: float
    pi_nearer_strategy: str
    energy_normalisation_achieved: bool
    fuel_gap_status: str


@dataclass(frozen=True)
class PIEnergyBracket:
    """Nearest sampled PI calibrations bracketing one endpoint-energy target."""

    lower_ratio: float
    upper_ratio: float
    lower_result: StrategyReplay
    upper_result: StrategyReplay
    target_kwh: float
    exact_match: bool

    @property
    def parameter_width(self) -> float:
        return self.upper_ratio - self.lower_ratio

    @property
    def energy_width_kwh(self) -> float:
        return abs(
            self.upper_result.battery_energy_change_kwh
            - self.lower_result.battery_energy_change_kwh
        )

    @property
    def fuel_interval_kg(self) -> tuple[float, float]:
        values = (
            self.lower_result.fuel_consumed_kg,
            self.upper_result.fuel_consumed_kg,
        )
        return min(values), max(values)


@dataclass(frozen=True)
class EqualEnergyComparison:
    """Equal-endpoint-energy comparison with a point or bracketed PI result."""

    comparison: str
    battery_mode: str
    timestep_s: float
    initial_soc: float
    target_battery_energy_change_kwh: float
    energy_match_tolerance_kwh: float
    continuous: StrategyReplay
    ideal_relaxed: StrategyReplay
    pi_bracket: PIEnergyBracket
    pi_to_continuous_gap_kg: tuple[float, float]
    pi_to_continuous_gap_fraction: tuple[float, float]
    pi_to_ideal_gap_kg: tuple[float, float]
    pi_to_ideal_gap_fraction: tuple[float, float]
    cycling_benefit_captured_fraction: tuple[float, float]
    fuel_gap_status: str


@dataclass(frozen=True)
class _BatteryLedger:
    bus_charge_kwh: float
    bus_discharge_kwh: float
    internal_charge_kwh: float
    internal_discharge_kwh: float
    ohmic_loss_kwh: float
    recirculated_loss_kwh: float


def _battery_ledger(
    bus_energy_kwh: Sequence[float],
    internal_energy_kwh: Sequence[float],
    ohmic_loss_kwh: Sequence[float],
) -> _BatteryLedger:
    bus_charge = -sum(min(value, 0.0) for value in bus_energy_kwh)
    bus_discharge = sum(max(value, 0.0) for value in bus_energy_kwh)
    internal_charge = -sum(min(value, 0.0) for value in internal_energy_kwh)
    internal_discharge = sum(max(value, 0.0) for value in internal_energy_kwh)
    recirculated = min(internal_charge, internal_discharge)
    recirculated_bus_in = (
        bus_charge * recirculated / internal_charge
        if internal_charge > 0.0
        else 0.0
    )
    recirculated_bus_out = (
        bus_discharge * recirculated / internal_discharge
        if internal_discharge > 0.0
        else 0.0
    )
    return _BatteryLedger(
        bus_charge,
        bus_discharge,
        internal_charge,
        internal_discharge,
        sum(ohmic_loss_kwh),
        recirculated_bus_in - recirculated_bus_out,
    )


def _duration_h(steps: Sequence[TimeStep]) -> float:
    duration = sum(step.dt_s for step in steps) / 3600.0
    if duration <= 0.0:
        raise ValueError("steps must span positive time")
    return duration


def _reported_timestep_s(steps: Sequence[TimeStep]) -> float:
    return max(step.dt_s for step in steps)


def _summarise_encounters(
    observations: dict[tuple[str, str], list[tuple[float, float, float]]],
) -> tuple[ConstraintEncounter, ...]:
    encounters = []
    for (direction, limit), values in sorted(observations.items()):
        times = [value[0] for value in values]
        socs = [value[1] for value in values]
        powers = [value[2] for value in values]
        encounters.append(
            ConstraintEncounter(
                direction=direction,
                limit=limit,
                timestep_count=len(values),
                first_time_s=min(times),
                minimum_soc=min(socs),
                maximum_soc=max(socs),
                minimum_power_kw=min(powers),
                maximum_power_kw=max(powers),
            )
        )
    return tuple(encounters)


def _merge_encounters(
    *groups: Sequence[ConstraintEncounter],
) -> tuple[ConstraintEncounter, ...]:
    merged: dict[tuple[str, str], list[ConstraintEncounter]] = {}
    for group in groups:
        for encounter in group:
            merged.setdefault((encounter.direction, encounter.limit), []).append(
                encounter
            )
    return tuple(
        ConstraintEncounter(
            direction=direction,
            limit=limit,
            timestep_count=sum(item.timestep_count for item in items),
            first_time_s=min(item.first_time_s for item in items),
            minimum_soc=min(item.minimum_soc for item in items),
            maximum_soc=max(item.maximum_soc for item in items),
            minimum_power_kw=min(item.minimum_power_kw for item in items),
            maximum_power_kw=max(item.maximum_power_kw for item in items),
        )
        for (direction, limit), items in sorted(merged.items())
    )


def validated_fuel_gap(
    reference: StrategyReplay,
    candidate: StrategyReplay,
    *,
    target_kwh: float,
    tolerance_kwh: float = ENERGY_MATCH_TOLERANCE_KWH,
) -> tuple[float, float]:
    """Return reference-minus-candidate fuel only after endpoint-energy checks."""
    for result in (reference, candidate):
        residual = result.battery_energy_change_kwh - target_kwh
        if abs(residual) > tolerance_kwh:
            raise EnergyMismatchError(
                f"{result.strategy} endpoint energy misses the target by "
                f"{residual:.12g} kWh (tolerance {tolerance_kwh:.12g} kWh)"
            )
    gap = reference.fuel_consumed_kg - candidate.fuel_consumed_kg
    return gap, gap / reference.fuel_consumed_kg


def resample_replay_steps(
    steps: Sequence[TimeStep], dt_s: float
) -> tuple[TimeStep, ...]:
    """Split logged intervals without changing their piecewise-constant inputs."""
    entries = tuple(steps)
    _duration_h(entries)
    target = float(dt_s)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    result: list[TimeStep] = []
    for step in entries:
        remaining = step.dt_s
        time_s = step.time_s - step.dt_s
        while remaining > 1.0e-12:
            duration = min(target, remaining)
            time_s += duration
            result.append(replace(step, time_s=time_s, dt_s=duration))
            remaining -= duration
    return tuple(result)


def replay_pi_ecms(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    controller: EMSController,
    *,
    initial_soc: float,
    initial_engine_shut_down: bool,
    target_battery_energy_change_kwh: float = 0.0,
) -> StrategyReplay:
    """Replay unchanged ECMS decisions against the logged exogenous trajectory."""
    return replay_pi_ecms_trace(
        steps,
        aircraft,
        controller,
        initial_soc=initial_soc,
        initial_engine_shut_down=initial_engine_shut_down,
        target_battery_energy_change_kwh=target_battery_energy_change_kwh,
    ).result


def replay_pi_ecms_trace(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    controller: EMSController,
    *,
    initial_soc: float,
    initial_engine_shut_down: bool,
    target_battery_energy_change_kwh: float = 0.0,
) -> PIReplayTrace:
    """Replay unchanged ECMS and retain a reconstructed battery-energy trace."""
    entries = tuple(steps)
    duration_h = _duration_h(entries)
    start_soc = float(initial_soc)
    if not aircraft.battery.soc_min <= start_soc <= 1.0:
        raise ValueError("initial_soc lies outside the modelled usable interval")
    soc = start_soc
    engine_was_off = bool(initial_engine_shut_down)
    fuel_kg = 0.0
    off_h = 0.0
    restarts = 0
    bus_energy: list[float] = []
    internal_energy: list[float] = []
    ohmic_energy: list[float] = []
    shaft_powers: list[float] = []
    replay_steps: list[TimeStep] = []
    encountered: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    minimum_soc = soc
    maximum_soc = soc

    for step in entries:
        pre_step_soc = soc
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        max_bus_kw = float(
            aircraft.powertrain.bus_power_from_engine(
                aircraft.engine.max_power_kw(sigma)
            )
        ) + aircraft.battery.available_discharge_kw(soc, step.dt_s)
        context = ControlContext(
            soc=soc,
            bus_demand_kw=step.bus_demand_kw,
            max_bus_kw=max_bus_kw,
            neutral_s=step.neutral_s,
            switching_s=step.switching_s,
            time_s=step.time_s - step.dt_s,
            phase=step.phase,
        )
        factor = controller.clamped_equivalence_factor(context)
        split = solve_split(
            bus_demand_kw=step.bus_demand_kw,
            engine=aircraft.engine,
            battery=aircraft.battery,
            powertrain=aircraft.powertrain,
            s=factor,
            soc=soc,
            sigma=sigma,
            dt_s=step.dt_s,
        )
        if not split.feasible:
            raise RuntimeError("PI-ECMS replay encountered an infeasible split")
        state = aircraft.battery.step(soc, split.battery_bus_kw, step.dt_s)
        if abs(state.power_kw - split.battery_bus_kw) > 1.0e-6:
            raise RuntimeError("PI-ECMS replay battery could not deliver its split")
        restart = engine_was_off and not split.engine_off
        restarts += int(restart)
        restart_fuel = aircraft.engine.restart_fuel_kg if restart else 0.0
        fuel_kg += split.fuel_flow_kg_s * step.dt_s + restart_fuel
        shaft_powers.append(split.engine_shaft_kw)
        off_h += step.dt_s / 3600.0 if split.engine_off else 0.0
        scale = step.dt_s / 3600.0
        bus_energy.append(state.power_kw * scale)
        internal_kw = state.open_circuit_voltage_v * state.current_a / 1000.0
        internal_energy.append(internal_kw * scale)
        ohmic_energy.append(state.ohmic_loss_kw * scale)
        stored_change = float(aircraft.battery.stored_energy_kwh(state.soc)) - float(
            aircraft.battery.stored_energy_kwh(pre_step_soc)
        )
        engine_state = aircraft.engine.operate(split.engine_shaft_kw, sigma)
        replay_steps.append(
            replace(
                step,
                equivalence_factor=factor,
                engine_shaft_kw=split.engine_shaft_kw,
                engine_load_fraction=engine_state.load_fraction,
                sfc_kg_kwh=engine_state.sfc_kg_kwh,
                fuel_flow_kg_s=split.fuel_flow_kg_s,
                restart_fuel_kg=restart_fuel,
                engine_shut_down=split.engine_off,
                battery_bus_kw=state.power_kw,
                soc=state.soc,
                bus_from_engine_kw=split.bus_from_engine_kw,
                battery_internal_kw=internal_kw,
                battery_ohmic_loss_kw=state.ohmic_loss_kw,
                battery_stored_energy_change_kwh=stored_change,
            )
        )
        if state.active_limit != "none" and abs(state.power_kw) > 1.0e-10:
            direction = "charge" if state.power_kw < 0.0 else "discharge"
            for limit in state.active_limit.split("_and_"):
                encountered.setdefault((direction, limit), []).append(
                    (step.time_s, pre_step_soc, abs(state.power_kw))
                )
        soc = state.soc
        minimum_soc = min(minimum_soc, soc)
        maximum_soc = max(maximum_soc, soc)
        engine_was_off = split.engine_off

    ledger = _battery_ledger(bus_energy, internal_energy, ohmic_energy)
    energy_change = float(aircraft.battery.stored_energy_kwh(soc)) - float(
        aircraft.battery.stored_energy_kwh(start_soc)
    )
    target = float(target_battery_energy_change_kwh)
    result = StrategyReplay(
        strategy="pi_ecms",
        fuel_consumed_kg=fuel_kg,
        average_fuel_rate_kg_h=fuel_kg / duration_h,
        battery_ohmic_loss_kwh=ledger.ohmic_loss_kwh,
        recirculated_round_trip_loss_kwh=ledger.recirculated_loss_kwh,
        engine_off_fraction=off_h / duration_h,
        restart_count=restarts,
        restart_count_status="defined from logged-step OFF-to-ON transitions",
        initial_soc=start_soc,
        terminal_soc=soc,
        minimum_soc=minimum_soc,
        maximum_soc=maximum_soc,
        battery_energy_change_kwh=energy_change,
        internal_ledger_energy_change_kwh=(
            ledger.internal_charge_kwh - ledger.internal_discharge_kwh
        ),
        euler_energy_residual_kwh=energy_change
        - (ledger.internal_charge_kwh - ledger.internal_discharge_kwh),
        target_battery_energy_change_kwh=target,
        terminal_energy_shortfall_kwh=energy_change - target,
        battery_mode=aircraft.battery.battery_mode.value,
        battery_capacity_kwh=aircraft.battery.capacity_kwh,
        i_charge_max_a=aircraft.battery.i_charge_max_a,
        i_discharge_max_a=aircraft.battery.i_discharge_max_a,
        terminal_voltage_min_v=aircraft.battery.terminal_voltage_min_v,
        terminal_voltage_max_v=aircraft.battery.terminal_voltage_max_v,
        q_nominal_ah=aircraft.battery.q_nominal_ah,
        timestep_s=_reported_timestep_s(entries),
        constraint_encounters=_summarise_encounters(encountered),
        calibration_parameter_value=float(getattr(controller, "s0_ratio", math.nan)),
        engine_shaft_power_mean_kw=sum(
            power * step.dt_s for power, step in zip(shaft_powers, entries)
        )
        / sum(step.dt_s for step in entries),
        engine_shaft_power_minimum_kw=min(shaft_powers),
        engine_shaft_power_maximum_kw=max(shaft_powers),
        strategy_construction="unchanged PI-ECMS replay",
    )
    return PIReplayTrace(result=result, steps=tuple(replay_steps))


def _continuous_replay(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    initial_soc: float,
    target_kwh: float,
) -> StrategyReplay:
    duration_h = _duration_h(steps)
    fuel_kg = 0.0
    shaft_energy_kw_s = 0.0
    shaft_powers = []
    for step in steps:
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        shaft_kw = float(
            aircraft.powertrain.engine_power_for_bus(step.bus_demand_kw)
        )
        engine_state = aircraft.engine.operate(shaft_kw, sigma)
        if engine_state.power_limited or engine_state.shut_down:
            raise RuntimeError("continuous replay cannot cover logged demand")
        fuel_kg += engine_state.fuel_flow_kg_s * step.dt_s
        shaft_energy_kw_s += shaft_kw * step.dt_s
        shaft_powers.append(shaft_kw)
    return StrategyReplay(
        strategy="continuous",
        fuel_consumed_kg=fuel_kg,
        average_fuel_rate_kg_h=fuel_kg / duration_h,
        battery_ohmic_loss_kwh=0.0,
        recirculated_round_trip_loss_kwh=0.0,
        engine_off_fraction=0.0,
        restart_count=0,
        restart_count_status="defined; engine remains running",
        initial_soc=initial_soc,
        terminal_soc=initial_soc,
        minimum_soc=initial_soc,
        maximum_soc=initial_soc,
        battery_energy_change_kwh=0.0,
        internal_ledger_energy_change_kwh=0.0,
        euler_energy_residual_kwh=0.0,
        target_battery_energy_change_kwh=target_kwh,
        terminal_energy_shortfall_kwh=-target_kwh,
        battery_mode=aircraft.battery.battery_mode.value,
        battery_capacity_kwh=aircraft.battery.capacity_kwh,
        i_charge_max_a=aircraft.battery.i_charge_max_a,
        i_discharge_max_a=aircraft.battery.i_discharge_max_a,
        terminal_voltage_min_v=aircraft.battery.terminal_voltage_min_v,
        terminal_voltage_max_v=aircraft.battery.terminal_voltage_max_v,
        q_nominal_ah=aircraft.battery.q_nominal_ah,
        timestep_s=_reported_timestep_s(steps),
        constraint_encounters=(),
        engine_shaft_power_mean_kw=shaft_energy_kw_s
        / sum(step.dt_s for step in steps),
        engine_shaft_power_minimum_kw=min(shaft_powers),
        engine_shaft_power_maximum_kw=max(shaft_powers),
        strategy_construction="engine load-following; battery idle",
    )


def _ideal_cycle_replay(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    initial_soc: float,
    target_kwh: float,
) -> StrategyReplay:
    duration_h = _duration_h(steps)
    source = aircraft.powertrain.source_chain_efficiency
    fuel_kg = 0.0
    off_h = 0.0
    bus_energy: list[float] = []
    internal_energy: list[float] = []
    ohmic_energy: list[float] = []
    has_fractional_cycles = False
    encountered: dict[tuple[str, str], list[tuple[float, float, float]]] = {}

    for step in steps:
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        engine_max_kw = aircraft.engine.max_power_kw(sigma)
        charge_limit_kw = aircraft.battery.available_charge_kw(initial_soc)
        discharge_limit_kw = aircraft.battery.available_discharge_kw(initial_soc)
        charge_kw = min(
            max(source * engine_max_kw - step.bus_demand_kw, 0.0),
            charge_limit_kw,
        )
        efficiencies = battery_efficiencies(
            aircraft.battery,
            charge_bus_kw=charge_kw,
            discharge_bus_kw=step.bus_demand_kw,
            soc=initial_soc,
        )
        classification = None
        for _ in range(4):
            classification = classify_regime(
                step.bus_demand_kw,
                engine_max_kw,
                charge_limit_kw,
                discharge_limit_kw,
                aircraft.engine.willans_a,
                aircraft.engine.willans_b,
                source,
                efficiencies.charge,
                efficiencies.discharge,
            )
            optimum = classification.cycle_optimum
            next_charge_kw = (
                max(source * optimum.engine_on_kw - step.bus_demand_kw, 0.0)
                if optimum is not None
                else 0.0
            )
            if abs(next_charge_kw - charge_kw) <= 1.0e-10:
                break
            charge_kw = next_charge_kw
            efficiencies = battery_efficiencies(
                aircraft.battery,
                charge_bus_kw=charge_kw,
                discharge_bus_kw=step.bus_demand_kw,
                soc=initial_soc,
            )
        if classification is None:
            raise RuntimeError("ideal-cycle classification did not converge")
        if classification.regime is not OperatingRegime.CYCLING_FEASIBLE:
            shaft_kw = float(
                aircraft.powertrain.engine_power_for_bus(step.bus_demand_kw)
            )
            engine_state = aircraft.engine.operate(shaft_kw, sigma)
            fuel_kg += engine_state.fuel_flow_kg_s * step.dt_s
            bus_energy.append(0.0)
            internal_energy.append(0.0)
            ohmic_energy.append(0.0)
            continue

        optimum = classification.cycle_optimum
        if optimum is None:
            raise RuntimeError("cycling regime has no constrained optimum")
        duty = optimum.duty_cycle
        has_fractional_cycles = has_fractional_cycles or duty < 1.0
        if "charge_ceiling" in optimum.active_bound:
            charge_binding = aircraft.battery.charge_availability(
                initial_soc
            ).binding_limit
            encountered.setdefault(("charge", charge_binding), []).append(
                (step.time_s, initial_soc, charge_kw)
            )
        if math.isclose(
            step.bus_demand_kw,
            discharge_limit_kw,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            discharge_binding = aircraft.battery.discharge_availability(
                initial_soc
            ).binding_limit
            encountered.setdefault(("discharge", discharge_binding), []).append(
                (step.time_s, initial_soc, step.bus_demand_kw)
            )
        scale_h = step.dt_s / 3600.0
        charge_bus_kwh = duty * charge_kw * scale_h
        discharge_bus_kwh = (1.0 - duty) * step.bus_demand_kw * scale_h
        charge_internal_kwh = -efficiencies.charge * charge_bus_kwh
        discharge_internal_kwh = (
            discharge_bus_kwh / efficiencies.discharge
        )
        bus_energy.extend((-charge_bus_kwh, discharge_bus_kwh))
        internal_energy.extend((charge_internal_kwh, discharge_internal_kwh))
        ohmic_energy.extend(
            (
                charge_bus_kwh + charge_internal_kwh,
                discharge_internal_kwh - discharge_bus_kwh,
            )
        )
        fuel_kg += optimum.cycle_fuel_kg_h * scale_h
        off_h += (1.0 - duty) * scale_h

    ledger = _battery_ledger(bus_energy, internal_energy, ohmic_energy)
    restart_count = None if has_fractional_cycles else 0
    restart_status = (
        "undefined; duty fraction has no cycle period or SoC band"
        if has_fractional_cycles
        else "defined; no cycling occurred"
    )
    return StrategyReplay(
        strategy="ideal_analytical_cycle",
        fuel_consumed_kg=fuel_kg,
        average_fuel_rate_kg_h=fuel_kg / duration_h,
        battery_ohmic_loss_kwh=ledger.ohmic_loss_kwh,
        recirculated_round_trip_loss_kwh=ledger.recirculated_loss_kwh,
        engine_off_fraction=off_h / duration_h,
        restart_count=restart_count,
        restart_count_status=restart_status,
        initial_soc=initial_soc,
        terminal_soc=initial_soc,
        minimum_soc=initial_soc,
        maximum_soc=initial_soc,
        battery_energy_change_kwh=0.0,
        internal_ledger_energy_change_kwh=(
            ledger.internal_charge_kwh - ledger.internal_discharge_kwh
        ),
        euler_energy_residual_kwh=-(
            ledger.internal_charge_kwh - ledger.internal_discharge_kwh
        ),
        target_battery_energy_change_kwh=target_kwh,
        terminal_energy_shortfall_kwh=-target_kwh,
        battery_mode=aircraft.battery.battery_mode.value,
        battery_capacity_kwh=aircraft.battery.capacity_kwh,
        i_charge_max_a=aircraft.battery.i_charge_max_a,
        i_discharge_max_a=aircraft.battery.i_discharge_max_a,
        terminal_voltage_min_v=aircraft.battery.terminal_voltage_min_v,
        terminal_voltage_max_v=aircraft.battery.terminal_voltage_max_v,
        q_nominal_ah=aircraft.battery.q_nominal_ah,
        timestep_s=_reported_timestep_s(steps),
        constraint_encounters=_summarise_encounters(encountered),
        strategy_construction=(
            "relaxed charge-sustaining duty fractions; cycle period and restart "
            "count undefined"
        ),
    )


def _continuous_assist_replay(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    initial_soc: float,
    target_kwh: float,
    battery_assist_kw: float,
) -> StrategyReplay:
    duration_h = _duration_h(steps)
    soc = initial_soc
    fuel_kg = 0.0
    bus_energy: list[float] = []
    internal_energy: list[float] = []
    ohmic_energy: list[float] = []
    shaft_powers: list[float] = []
    encountered: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    minimum_soc = soc
    maximum_soc = soc

    for step in steps:
        pre_step_soc = soc
        state = aircraft.battery.step(soc, battery_assist_kw, step.dt_s)
        if abs(state.power_kw - battery_assist_kw) > 1.0e-7:
            raise RuntimeError("continuous-assist battery command was constrained")
        engine_bus_kw = step.bus_demand_kw - state.power_kw
        shaft_kw = float(aircraft.powertrain.engine_power_for_bus(engine_bus_kw))
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        engine_state = aircraft.engine.operate(shaft_kw, sigma)
        if engine_state.power_limited or engine_state.shut_down:
            raise RuntimeError("continuous-assist engine cannot cover its command")
        fuel_kg += engine_state.fuel_flow_kg_s * step.dt_s
        shaft_powers.append(shaft_kw)
        scale_h = step.dt_s / 3600.0
        internal_kw = state.open_circuit_voltage_v * state.current_a / 1000.0
        bus_energy.append(state.power_kw * scale_h)
        internal_energy.append(internal_kw * scale_h)
        ohmic_energy.append(state.ohmic_loss_kw * scale_h)
        if state.active_limit != "none" and abs(state.power_kw) > 1.0e-10:
            for limit in state.active_limit.split("_and_"):
                encountered.setdefault(("discharge", limit), []).append(
                    (step.time_s, pre_step_soc, state.power_kw)
                )
        soc = state.soc
        minimum_soc = min(minimum_soc, soc)
        maximum_soc = max(maximum_soc, soc)

    ledger = _battery_ledger(bus_energy, internal_energy, ohmic_energy)
    energy_change = float(aircraft.battery.stored_energy_kwh(soc)) - float(
        aircraft.battery.stored_energy_kwh(initial_soc)
    )
    internal_change = ledger.internal_charge_kwh - ledger.internal_discharge_kwh
    return StrategyReplay(
        strategy="continuous",
        fuel_consumed_kg=fuel_kg,
        average_fuel_rate_kg_h=fuel_kg / duration_h,
        battery_ohmic_loss_kwh=ledger.ohmic_loss_kwh,
        recirculated_round_trip_loss_kwh=0.0,
        engine_off_fraction=0.0,
        restart_count=0,
        restart_count_status="defined; engine remains running",
        initial_soc=initial_soc,
        terminal_soc=soc,
        minimum_soc=minimum_soc,
        maximum_soc=maximum_soc,
        battery_energy_change_kwh=energy_change,
        internal_ledger_energy_change_kwh=internal_change,
        euler_energy_residual_kwh=energy_change - internal_change,
        target_battery_energy_change_kwh=target_kwh,
        terminal_energy_shortfall_kwh=energy_change - target_kwh,
        battery_mode=aircraft.battery.battery_mode.value,
        battery_capacity_kwh=aircraft.battery.capacity_kwh,
        i_charge_max_a=aircraft.battery.i_charge_max_a,
        i_discharge_max_a=aircraft.battery.i_discharge_max_a,
        terminal_voltage_min_v=aircraft.battery.terminal_voltage_min_v,
        terminal_voltage_max_v=aircraft.battery.terminal_voltage_max_v,
        q_nominal_ah=aircraft.battery.q_nominal_ah,
        timestep_s=_reported_timestep_s(steps),
        constraint_encounters=_summarise_encounters(encountered),
        calibration_parameter_value=battery_assist_kw,
        engine_shaft_power_mean_kw=sum(
            power * step.dt_s for power, step in zip(shaft_powers, steps)
        )
        / sum(step.dt_s for step in steps),
        engine_shaft_power_minimum_kw=min(shaft_powers),
        engine_shaft_power_maximum_kw=max(shaft_powers),
        strategy_construction=(
            "engine continuously supplies demand minus a solved constant battery "
            "bus-power assist"
        ),
    )


def _solve_continuous_assist_replay(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    initial_soc: float,
    target_kwh: float,
) -> StrategyReplay:
    if target_kwh == 0.0:
        return _continuous_replay(steps, aircraft, initial_soc, target_kwh)
    if target_kwh > 0.0:
        raise ValueError("continuous assist only supports neutral or depleting targets")
    lower_kw = 0.0
    lower = _continuous_assist_replay(
        steps, aircraft, initial_soc, target_kwh, lower_kw
    )
    upper_kw = abs(target_kwh) / _duration_h(steps)
    upper = _continuous_assist_replay(steps, aircraft, initial_soc, target_kwh, upper_kw)
    while upper.terminal_energy_shortfall_kwh > 0.0:
        upper_kw *= 1.1
        try:
            upper = _continuous_assist_replay(
                steps, aircraft, initial_soc, target_kwh, upper_kw
            )
        except RuntimeError as error:
            raise RuntimeError(
                "continuous assist reaches the running-engine floor before the "
                "depletion target"
            ) from error
    if upper.terminal_energy_shortfall_kwh > 0.0:
        raise RuntimeError("continuous assist cannot reach the depletion target")
    for _ in range(60):
        middle_kw = 0.5 * (lower_kw + upper_kw)
        middle = _continuous_assist_replay(
            steps, aircraft, initial_soc, target_kwh, middle_kw
        )
        if abs(middle.terminal_energy_shortfall_kwh) <= ENERGY_MATCH_TOLERANCE_KWH:
            return middle
        if middle.terminal_energy_shortfall_kwh > 0.0:
            lower_kw, lower = middle_kw, middle
        else:
            upper_kw, upper = middle_kw, middle
    return min((lower, upper), key=lambda item: abs(item.terminal_energy_shortfall_kwh))


@dataclass(frozen=True)
class _OffTail:
    terminal_soc: float
    energy_change_kwh: float
    internal_change_kwh: float
    ohmic_loss_kwh: float
    encountered: tuple[ConstraintEncounter, ...]


def _simulate_off_tail(
    steps: Sequence[TimeStep], aircraft: Aircraft, initial_soc: float
) -> _OffTail:
    soc = initial_soc
    internal_change = 0.0
    ohmic_loss = 0.0
    encountered: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for step in steps:
        pre_step_soc = soc
        state = aircraft.battery.step(soc, step.bus_demand_kw, step.dt_s)
        if abs(state.power_kw - step.bus_demand_kw) > 1.0e-7:
            raise RuntimeError("relaxed ideal depletion tail violates battery power")
        scale_h = step.dt_s / 3600.0
        internal_change -= (
            state.open_circuit_voltage_v * state.current_a / 1000.0 * scale_h
        )
        ohmic_loss += state.ohmic_loss_kw * scale_h
        if state.active_limit != "none":
            for limit in state.active_limit.split("_and_"):
                encountered.setdefault(("discharge", limit), []).append(
                    (step.time_s, pre_step_soc, state.power_kw)
                )
        soc = state.soc
    energy_change = float(aircraft.battery.stored_energy_kwh(soc)) - float(
        aircraft.battery.stored_energy_kwh(initial_soc)
    )
    return _OffTail(
        terminal_soc=soc,
        energy_change_kwh=energy_change,
        internal_change_kwh=internal_change,
        ohmic_loss_kwh=ohmic_loss,
        encountered=_summarise_encounters(encountered),
    )


def _ideal_depleting_replay(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    initial_soc: float,
    target_kwh: float,
) -> StrategyReplay:
    entries = tuple(steps)
    duration_h = _duration_h(entries)
    if target_kwh >= 0.0:
        raise ValueError("ideal depletion requires a negative target")
    selected_count = None
    for count in range(1, len(entries) + 1):
        try:
            tail = _simulate_off_tail(entries[-count:], aircraft, initial_soc)
        except RuntimeError:
            break
        if tail.energy_change_kwh <= target_kwh:
            selected_count = count
            break
    if selected_count is None:
        raise RuntimeError("relaxed ideal depletion tail cannot reach the target")

    boundary_index = len(entries) - selected_count
    boundary = entries[boundary_index]
    later = entries[boundary_index + 1 :]
    lower_s = 0.0
    upper_s = boundary.dt_s
    tail = _simulate_off_tail(later, aircraft, initial_soc)
    for _ in range(60):
        middle_s = 0.5 * (lower_s + upper_s)
        first = replace(boundary, dt_s=middle_s)
        candidate = _simulate_off_tail((first, *later), aircraft, initial_soc)
        tail = candidate
        if abs(candidate.energy_change_kwh - target_kwh) <= ENERGY_MATCH_TOLERANCE_KWH:
            break
        if candidate.energy_change_kwh > target_kwh:
            lower_s = middle_s
        else:
            upper_s = middle_s

    off_duration_s = sum(step.dt_s for step in later) + middle_s
    prefix_steps = list(entries[:boundary_index])
    prefix_duration_s = boundary.dt_s - middle_s
    if prefix_duration_s > 1.0e-12:
        prefix_steps.append(
            replace(
                boundary,
                dt_s=prefix_duration_s,
                time_s=boundary.time_s - middle_s,
            )
        )
    prefix = _ideal_cycle_replay(
        tuple(prefix_steps), aircraft, initial_soc, 0.0
    )
    internal_change = prefix.internal_ledger_energy_change_kwh + tail.internal_change_kwh
    energy_change = tail.energy_change_kwh
    fuel_kg = prefix.fuel_consumed_kg
    prefix_h = sum(step.dt_s for step in prefix_steps) / 3600.0
    off_h = prefix.engine_off_fraction * prefix_h + off_duration_s / 3600.0
    return StrategyReplay(
        strategy="ideal_analytical_cycle",
        fuel_consumed_kg=fuel_kg,
        average_fuel_rate_kg_h=fuel_kg / duration_h,
        battery_ohmic_loss_kwh=(
            prefix.battery_ohmic_loss_kwh + tail.ohmic_loss_kwh
        ),
        recirculated_round_trip_loss_kwh=(
            prefix.recirculated_round_trip_loss_kwh
        ),
        engine_off_fraction=off_h / duration_h,
        restart_count=None,
        restart_count_status=(
            "undefined; relaxed duty fractions and depletion tail specify no "
            "cycle period or SoC band"
        ),
        initial_soc=initial_soc,
        terminal_soc=tail.terminal_soc,
        minimum_soc=min(initial_soc, tail.terminal_soc),
        maximum_soc=initial_soc,
        battery_energy_change_kwh=energy_change,
        internal_ledger_energy_change_kwh=internal_change,
        euler_energy_residual_kwh=energy_change - internal_change,
        target_battery_energy_change_kwh=target_kwh,
        terminal_energy_shortfall_kwh=energy_change - target_kwh,
        battery_mode=aircraft.battery.battery_mode.value,
        battery_capacity_kwh=aircraft.battery.capacity_kwh,
        i_charge_max_a=aircraft.battery.i_charge_max_a,
        i_discharge_max_a=aircraft.battery.i_discharge_max_a,
        terminal_voltage_min_v=aircraft.battery.terminal_voltage_min_v,
        terminal_voltage_max_v=aircraft.battery.terminal_voltage_max_v,
        q_nominal_ah=aircraft.battery.q_nominal_ah,
        timestep_s=_reported_timestep_s(entries),
        constraint_encounters=_merge_encounters(
            prefix.constraint_encounters, tail.encountered
        ),
        strategy_construction=(
            "relaxed charge-sustaining duty fractions followed by a solved "
            "engine-OFF depletion tail within the same logged window"
        ),
    )


def tune_pi_energy_bracket(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    controller: EMSController,
    *,
    initial_soc: float,
    initial_engine_shut_down: bool,
    target_kwh: float,
    search_increment: float = 0.02,
) -> PIEnergyBracket:
    """Tune the exposed PI anchor and retain a discontinuous energy bracket."""
    if not hasattr(controller, "s0_ratio"):
        raise ValueError("controller must expose the existing s0_ratio calibration")
    base_ratio = float(getattr(controller, "s0_ratio"))
    if search_increment <= 0.0:
        raise ValueError("search_increment must be positive")
    results: dict[float, StrategyReplay] = {}

    def evaluate(ratio: float) -> StrategyReplay:
        key = round(ratio, 14)
        if key not in results:
            tuned = replace(controller, s0_ratio=ratio)
            results[key] = replay_pi_ecms(
                steps,
                aircraft,
                tuned,
                initial_soc=initial_soc,
                initial_engine_shut_down=initial_engine_shut_down,
                target_battery_energy_change_kwh=target_kwh,
            )
        return results[key]

    def exact_match() -> tuple[float, StrategyReplay] | None:
        matches = [
            (ratio, result)
            for ratio, result in results.items()
            if abs(result.terminal_energy_shortfall_kwh)
            <= ENERGY_MATCH_TOLERANCE_KWH
        ]
        return min(matches, key=lambda item: abs(item[0] - base_ratio)) if matches else None

    def straddling_pair() -> tuple[float, float, StrategyReplay, StrategyReplay] | None:
        ordered = sorted(results.items())
        pairs = []
        for (left_ratio, left), (right_ratio, right) in zip(
            ordered[:-1], ordered[1:]
        ):
            if (
                left.terminal_energy_shortfall_kwh
                * right.terminal_energy_shortfall_kwh
                < 0.0
            ):
                pairs.append((left_ratio, right_ratio, left, right))
        return min(pairs, key=lambda item: item[1] - item[0]) if pairs else None

    evaluate(base_ratio)
    match = exact_match()
    if match is not None:
        ratio, result = match
        return PIEnergyBracket(ratio, ratio, result, result, target_kwh, True)

    bracket = None
    for index in range(1, 76):
        lower_ratio = base_ratio - index * search_increment
        if lower_ratio > 0.0:
            evaluate(lower_ratio)
        evaluate(base_ratio + index * search_increment)
        match = exact_match()
        if match is not None:
            ratio, result = match
            return PIEnergyBracket(ratio, ratio, result, result, target_kwh, True)
        bracket = straddling_pair()
        if bracket is not None:
            break
    if bracket is None:
        nearest = min(
            results.items(),
            key=lambda item: abs(item[1].terminal_energy_shortfall_kwh),
        )
        raise RuntimeError(
            "PI s0_ratio search did not bracket the target; nearest residual is "
            f"{nearest[1].terminal_energy_shortfall_kwh:.9g} kWh at "
            f"s0_ratio={nearest[0]:.9g}"
        )

    left_ratio, right_ratio, left, right = bracket
    for _ in range(24):
        middle_ratio = 0.5 * (left_ratio + right_ratio)
        middle = evaluate(middle_ratio)
        if (
            abs(middle.terminal_energy_shortfall_kwh)
            <= ENERGY_MATCH_TOLERANCE_KWH
        ):
            return PIEnergyBracket(
                middle_ratio,
                middle_ratio,
                middle,
                middle,
                target_kwh,
                True,
            )
        if (
            left.terminal_energy_shortfall_kwh
            * middle.terminal_energy_shortfall_kwh
            < 0.0
        ):
            right_ratio, right = middle_ratio, middle
        else:
            left_ratio, left = middle_ratio, middle
    return PIEnergyBracket(
        left_ratio,
        right_ratio,
        left,
        right,
        target_kwh,
        False,
    )


def compare_equal_energy_replays(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    controller: EMSController,
    *,
    initial_soc: float,
    initial_engine_shut_down: bool,
    target_battery_energy_change_kwh: float,
) -> EqualEnergyComparison:
    """Compare three strategies at one endpoint stored-energy target."""
    entries = tuple(steps)
    _duration_h(entries)
    target = float(target_battery_energy_change_kwh)
    if target > 0.0 or not math.isfinite(target):
        raise ValueError("equal-energy replay target must be finite and non-positive")
    continuous = _solve_continuous_assist_replay(
        entries, aircraft, initial_soc, target
    )
    ideal = (
        _ideal_cycle_replay(entries, aircraft, initial_soc, target)
        if target == 0.0
        else _ideal_depleting_replay(entries, aircraft, initial_soc, target)
    )
    validated_fuel_gap(continuous, ideal, target_kwh=target)
    pi_bracket = tune_pi_energy_bracket(
        entries,
        aircraft,
        controller,
        initial_soc=initial_soc,
        initial_engine_shut_down=initial_engine_shut_down,
        target_kwh=target,
    )
    pi_fuel_low, pi_fuel_high = pi_bracket.fuel_interval_kg
    continuous_fuel = continuous.fuel_consumed_kg
    ideal_fuel = ideal.fuel_consumed_kg
    gap_cont_kg = (
        continuous_fuel - pi_fuel_high,
        continuous_fuel - pi_fuel_low,
    )
    gap_cont_fraction = tuple(value / continuous_fuel for value in gap_cont_kg)
    gap_ideal_kg = (pi_fuel_low - ideal_fuel, pi_fuel_high - ideal_fuel)
    gap_ideal_fraction = tuple(value / ideal_fuel for value in gap_ideal_kg)
    available_benefit = continuous_fuel - ideal_fuel
    captured = tuple(value / available_benefit for value in gap_cont_kg)
    if pi_bracket.exact_match:
        validated_fuel_gap(
            continuous,
            pi_bracket.lower_result,
            target_kwh=target,
        )
        status = (
            "valid point gap: endpoint stored energy passes the numerical gate"
        )
    else:
        low_residual = pi_bracket.lower_result.terminal_energy_shortfall_kwh
        high_residual = pi_bracket.upper_result.terminal_energy_shortfall_kwh
        if low_residual * high_residual >= 0.0:
            raise EnergyMismatchError("PI bracket does not straddle the target")
        status = (
            "valid bounded gap: the discontinuous PI endpoint energies straddle "
            "the target; neither bracket endpoint is reported as a point gap"
        )
    comparison = "charge_sustaining" if target == 0.0 else "mission_depleting"
    return EqualEnergyComparison(
        comparison=comparison,
        battery_mode=aircraft.battery.battery_mode.value,
        timestep_s=_reported_timestep_s(entries),
        initial_soc=initial_soc,
        target_battery_energy_change_kwh=target,
        energy_match_tolerance_kwh=ENERGY_MATCH_TOLERANCE_KWH,
        continuous=continuous,
        ideal_relaxed=ideal,
        pi_bracket=pi_bracket,
        pi_to_continuous_gap_kg=gap_cont_kg,
        pi_to_continuous_gap_fraction=gap_cont_fraction,
        pi_to_ideal_gap_kg=gap_ideal_kg,
        pi_to_ideal_gap_fraction=gap_ideal_fraction,
        cycling_benefit_captured_fraction=captured,
        fuel_gap_status=status,
    )


def _normalising_initial_soc(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    controller: EMSController,
    initial_engine_shut_down: bool,
    target_kwh: float,
) -> tuple[float, StrategyReplay]:
    lower = aircraft.battery.soc_min
    upper = 1.0

    def evaluate(soc: float) -> StrategyReplay:
        return replay_pi_ecms(
            steps,
            aircraft,
            controller,
            initial_soc=soc,
            initial_engine_shut_down=initial_engine_shut_down,
            target_battery_energy_change_kwh=target_kwh,
        )

    samples = np.linspace(lower, upper, 41)
    results = [evaluate(float(soc)) for soc in samples]
    best = min(results, key=lambda result: abs(result.terminal_energy_shortfall_kwh))
    bracket = None
    for left_soc, right_soc, left_result, right_result in zip(
        samples[:-1], samples[1:], results[:-1], results[1:]
    ):
        left = left_result.terminal_energy_shortfall_kwh
        right = right_result.terminal_energy_shortfall_kwh
        if left == 0.0:
            return float(left_soc), left_result
        if left * right <= 0.0:
            bracket = (float(left_soc), float(right_soc), left_result, right_result)
            break
    if bracket is None:
        return best.initial_soc, best

    left_soc, right_soc, left_result, right_result = bracket
    for _ in range(40):
        middle_soc = 0.5 * (left_soc + right_soc)
        middle_result = evaluate(middle_soc)
        if abs(middle_result.terminal_energy_shortfall_kwh) < abs(
            best.terminal_energy_shortfall_kwh
        ):
            best = middle_result
        left_value = left_result.terminal_energy_shortfall_kwh
        middle_value = middle_result.terminal_energy_shortfall_kwh
        if left_value * middle_value <= 0.0:
            right_soc, right_result = middle_soc, middle_result
        else:
            left_soc, left_result = middle_soc, middle_result
    return best.initial_soc, best


def compare_replays(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    controller: EMSController,
    *,
    initial_engine_shut_down: bool,
    target_battery_energy_change_kwh: float = 0.0,
) -> ReplayComparison:
    """Run the three strategies with one common energy-neutral initial SoC."""
    entries = tuple(steps)
    _duration_h(entries)
    target = float(target_battery_energy_change_kwh)
    if not math.isfinite(target):
        raise ValueError("target_battery_energy_change_kwh must be finite")
    initial_soc, pi_result = _normalising_initial_soc(
        entries,
        aircraft,
        controller,
        initial_engine_shut_down,
        target,
    )
    continuous = _continuous_replay(entries, aircraft, initial_soc, target)
    ideal = _ideal_cycle_replay(entries, aircraft, initial_soc, target)
    gap_cont_kg = continuous.fuel_consumed_kg - pi_result.fuel_consumed_kg
    gap_cycle_kg = pi_result.fuel_consumed_kg - ideal.fuel_consumed_kg
    distance_cont = abs(pi_result.fuel_consumed_kg - continuous.fuel_consumed_kg)
    distance_cycle = abs(pi_result.fuel_consumed_kg - ideal.fuel_consumed_kg)
    nearer = (
        "ideal_analytical_cycle"
        if distance_cycle < distance_cont
        else "continuous"
    )
    return ReplayComparison(
        strategies=(continuous, ideal, pi_result),
        normalisation_method=(
            "common initial SoC shooting; unchanged PI-ECMS targeted to "
            "zero endpoint battery-energy change"
        ),
        common_initial_soc=initial_soc,
        target_battery_energy_change_kwh=target,
        pi_target_residual_kwh=pi_result.terminal_energy_shortfall_kwh,
        pi_to_continuous_gap_kg=gap_cont_kg,
        pi_to_continuous_gap_fraction=gap_cont_kg / continuous.fuel_consumed_kg,
        pi_to_ideal_cycle_gap_kg=gap_cycle_kg,
        pi_to_ideal_cycle_gap_fraction=gap_cycle_kg / ideal.fuel_consumed_kg,
        pi_nearer_strategy=nearer,
        energy_normalisation_achieved=(
            max(
                abs(result.terminal_energy_shortfall_kwh)
                for result in (continuous, ideal, pi_result)
            )
            < 0.01
        ),
        fuel_gap_status=(
            "energy-normalised within the discrete-policy tolerance"
            if abs(target) < 0.01
            and abs(pi_result.terminal_energy_shortfall_kwh) < 0.01
            else "one or more strategies miss the target; fuel gaps include "
            "unequal battery energy"
        ),
    )


def compare_replays_at_initial_soc(
    steps: Sequence[TimeStep],
    aircraft: Aircraft,
    controller: EMSController,
    *,
    initial_soc: float,
    initial_engine_shut_down: bool,
    target_battery_energy_change_kwh: float,
) -> ReplayComparison:
    """Run fixed-policy replays from one representative initial SoC."""
    entries = tuple(steps)
    _duration_h(entries)
    start_soc = float(initial_soc)
    target = float(target_battery_energy_change_kwh)
    if not math.isfinite(target):
        raise ValueError("target_battery_energy_change_kwh must be finite")
    pi_result = replay_pi_ecms(
        entries,
        aircraft,
        controller,
        initial_soc=start_soc,
        initial_engine_shut_down=initial_engine_shut_down,
        target_battery_energy_change_kwh=target,
    )
    continuous = _continuous_replay(entries, aircraft, start_soc, target)
    ideal = _ideal_cycle_replay(entries, aircraft, start_soc, target)
    gap_cont_kg = continuous.fuel_consumed_kg - pi_result.fuel_consumed_kg
    gap_cycle_kg = pi_result.fuel_consumed_kg - ideal.fuel_consumed_kg
    distance_cont = abs(pi_result.fuel_consumed_kg - continuous.fuel_consumed_kg)
    distance_cycle = abs(pi_result.fuel_consumed_kg - ideal.fuel_consumed_kg)
    nearer = (
        "ideal_analytical_cycle"
        if distance_cycle < distance_cont
        else "continuous"
    )
    return ReplayComparison(
        strategies=(continuous, ideal, pi_result),
        normalisation_method=(
            "fixed actual post-crossing initial SoC; measured legacy 60 s "
            "battery-energy change is the common target; unchanged strategies "
            "report terminal-energy shortfall"
        ),
        common_initial_soc=start_soc,
        target_battery_energy_change_kwh=target,
        pi_target_residual_kwh=pi_result.terminal_energy_shortfall_kwh,
        pi_to_continuous_gap_kg=gap_cont_kg,
        pi_to_continuous_gap_fraction=gap_cont_kg / continuous.fuel_consumed_kg,
        pi_to_ideal_cycle_gap_kg=gap_cycle_kg,
        pi_to_ideal_cycle_gap_fraction=gap_cycle_kg / ideal.fuel_consumed_kg,
        pi_nearer_strategy=nearer,
        energy_normalisation_achieved=False,
        fuel_gap_status=(
            "not energy-normalised: continuous and ideal cycling keep the battery "
            "neutral and therefore miss the negative common target"
        ),
    )


def write_replay_comparison_csv(
    comparison: ReplayComparison, output_path: str | Path
) -> Path:
    """Write one row per strategy with repeated comparison metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(comparison)
    summary.pop("strategies")
    rows = []
    for result in comparison.strategies:
        row = asdict(result) | summary
        row["constraint_encounters"] = json.dumps(
            row["constraint_encounters"], sort_keys=True
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_replay_comparisons_csv(
    comparisons: Sequence[ReplayComparison], output_path: str | Path
) -> Path:
    """Write multiple timestep and battery-mode comparisons to one table."""
    rows = []
    for comparison in comparisons:
        summary = asdict(comparison)
        summary.pop("strategies")
        for result in comparison.strategies:
            row = asdict(result) | summary
            row["constraint_encounters"] = json.dumps(
                row["constraint_encounters"], sort_keys=True
            )
            rows.append(row)
    if not rows:
        raise ValueError("comparisons must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_equal_energy_comparisons_csv(
    comparisons: Sequence[EqualEnergyComparison], output_path: str | Path
) -> Path:
    """Write point strategies and both sides of every discontinuous PI bracket."""
    rows = []
    for comparison in comparisons:
        bracket = comparison.pi_bracket
        metadata = {
            "comparison": comparison.comparison,
            "comparison_battery_mode": comparison.battery_mode,
            "comparison_timestep_s": comparison.timestep_s,
            "comparison_initial_soc": comparison.initial_soc,
            "comparison_target_battery_energy_change_kwh": (
                comparison.target_battery_energy_change_kwh
            ),
            "energy_match_tolerance_kwh": comparison.energy_match_tolerance_kwh,
            "pi_s0_ratio_lower": bracket.lower_ratio,
            "pi_s0_ratio_upper": bracket.upper_ratio,
            "pi_s0_ratio_bracket_width": bracket.parameter_width,
            "pi_endpoint_energy_bracket_width_kwh": bracket.energy_width_kwh,
            "pi_exact_match": bracket.exact_match,
            "pi_to_continuous_gap_kg_low": comparison.pi_to_continuous_gap_kg[0],
            "pi_to_continuous_gap_kg_high": comparison.pi_to_continuous_gap_kg[1],
            "pi_to_continuous_gap_fraction_low": (
                comparison.pi_to_continuous_gap_fraction[0]
            ),
            "pi_to_continuous_gap_fraction_high": (
                comparison.pi_to_continuous_gap_fraction[1]
            ),
            "pi_to_ideal_gap_kg_low": comparison.pi_to_ideal_gap_kg[0],
            "pi_to_ideal_gap_kg_high": comparison.pi_to_ideal_gap_kg[1],
            "pi_to_ideal_gap_fraction_low": comparison.pi_to_ideal_gap_fraction[0],
            "pi_to_ideal_gap_fraction_high": comparison.pi_to_ideal_gap_fraction[1],
            "cycling_benefit_captured_fraction_low": (
                comparison.cycling_benefit_captured_fraction[0]
            ),
            "cycling_benefit_captured_fraction_high": (
                comparison.cycling_benefit_captured_fraction[1]
            ),
            "fuel_gap_status": comparison.fuel_gap_status,
        }
        strategy_rows = [
            ("point", comparison.continuous),
            ("relaxed_point", comparison.ideal_relaxed),
        ]
        if bracket.exact_match:
            strategy_rows.append(("point", bracket.lower_result))
        else:
            strategy_rows.extend(
                (("bracket_lower_ratio", bracket.lower_result),
                 ("bracket_upper_ratio", bracket.upper_result))
            )
        for bracket_role, result in strategy_rows:
            row = asdict(result) | metadata | {"bracket_role": bracket_role}
            row["constraint_encounters"] = json.dumps(
                row["constraint_encounters"], sort_keys=True
            )
            rows.append(row)
    if not rows:
        raise ValueError("comparisons must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
