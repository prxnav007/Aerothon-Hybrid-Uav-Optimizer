"""Deterministic time-marching mission simulation.

The simulator composes the immutable mission profile, atmosphere,
aerodynamics, component models, controller interface, and ECMS split solver.
It reports failures as data so an optimizer can retain a useful gradient toward
feasibility; malformed call arguments remain ordinary ``ValueError`` inputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

from src.control.base import ControlContext, EMSController, neutral_equivalence_factor
from src.control.power_split import (
    SplitDecision,
    solve_split,
    switching_equivalence_factor,
)
from src.models import aerodynamics
from src.models.atmosphere import atmosphere, g0
from src.models.battery import BatteryPack, SOC_EPS
from src.models.engine import LHV_KJ_KG, EngineState, Turboshaft
from src.models.mass import MassBreakdown
from src.models.powertrain import SeriesPowertrain
from src.simulation.mission import (
    AltitudeMode,
    MissionProfile,
    Phase,
    SpeedMode,
    Termination,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "Aircraft",
    "MissionEnergyBalance",
    "MissionResult",
    "TimeStep",
    "log_to_dataframe",
    "mission_energy_balance",
    "run_mission",
]

_POWER_TOLERANCE_KW = 1.0e-6
_STATE_EPS = 1.0e-10


def _finite_positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


@dataclass(frozen=True)
class Aircraft:
    """Aircraft data and component models consumed by the mission loop."""

    wing_area_m2: float
    aspect_ratio: float
    oswald_efficiency: float
    cd0: float
    cl_max: float
    propeller_efficiency: float
    engine: Turboshaft
    battery: BatteryPack
    powertrain: SeriesPowertrain
    masses: MassBreakdown

    def __post_init__(self) -> None:
        for name in ("wing_area_m2", "aspect_ratio", "oswald_efficiency", "cd0", "cl_max"):
            _finite_positive(name, getattr(self, name))
        efficiency = _finite_positive("propeller_efficiency", self.propeller_efficiency)
        if efficiency > 1.0:
            raise ValueError(
                "propeller_efficiency must lie in (0, 1], "
                f"got {self.propeller_efficiency!r}"
            )


@dataclass(frozen=True)
class TimeStep:
    """One completed integration step, logged only when explicitly requested."""

    time_s: float
    phase: str
    altitude_m: float
    speed_mps: float
    weight_n: float
    density_kg_m3: float
    lift_coefficient: float
    drag_n: float
    shaft_power_kw: float
    bus_demand_kw: float
    neutral_s: float
    switching_s: float
    equivalence_factor: float
    engine_shaft_kw: float
    engine_load_fraction: float
    sfc_kg_kwh: float
    fuel_flow_kg_s: float
    restart_fuel_kg: float
    engine_shut_down: bool
    battery_bus_kw: float
    soc: float
    fuel_remaining_kg: float
    system_efficiency: float
    power_off: bool
    power_limited: bool
    # Accounting fields make the conservation test auditable rather than
    # relying on hidden reconstruction from post-step state.
    dt_s: float
    bus_from_engine_kw: float
    battery_internal_kw: float
    battery_ohmic_loss_kw: float
    battery_stored_energy_change_kwh: float
    thrust_power_kw: float
    engine_thermal_loss_kw: float
    source_losses_kw: float
    demand_losses_kw: float
    propeller_losses_kw: float


@dataclass(frozen=True)
class MissionResult:
    """Aggregated mission outcome and optional immutable timestep log."""

    endurance_s: float
    mission_complete: bool
    termination_reason: str
    phase_durations_s: dict[str, float]
    fuel_used_kg: float
    fuel_remaining_kg: float
    final_soc: float
    min_soc: float
    peak_bus_kw: float
    peak_engine_kw: float
    mean_system_efficiency: float
    failure_flags: tuple[str, ...]
    log: tuple[TimeStep, ...] | None


@dataclass(frozen=True)
class MissionEnergyBalance:
    """Integrated fuel-to-thrust energy ledger for a recorded mission [kWh]."""

    fuel_chemical_in_kwh: float
    propulsive_work_out_kwh: float
    engine_thermal_loss_kwh: float
    source_chain_loss_kwh: float
    demand_chain_loss_kwh: float
    propeller_loss_kwh: float
    battery_ohmic_loss_kwh: float
    battery_stored_energy_change_kwh: float
    discrete_battery_stored_energy_change_kwh: float
    battery_integration_residual_kwh: float
    residual_kwh: float
    residual_fraction: float
    discrete_residual_fraction: float


@dataclass(frozen=True, slots=True)
class _Dispatch:
    atmospheric: Any
    speed_mps: float
    aerodynamic: Any
    shaft_power_kw: float
    bus_demand_kw: float
    neutral_s: float
    switching_s: float
    equivalence_factor: float
    max_bus_kw: float
    split: SplitDecision


def _speed_for_phase(
    phase: Phase,
    aircraft: Aircraft,
    weight_n: float,
    density_kg_m3: float,
) -> float:
    if phase.speed_mode is SpeedMode.FIXED:
        return float(phase.speed_mps)
    if phase.speed_mode is SpeedMode.MIN_POWER:
        speed_mps, _ = aerodynamics.loiter_speed(
            weight_n,
            density_kg_m3,
            aircraft.wing_area_m2,
            aircraft.cd0,
            aircraft.aspect_ratio,
            aircraft.oswald_efficiency,
            aircraft.cl_max,
        )
        return float(speed_mps)
    return float(
        aerodynamics.speed_best_ld(
            weight_n,
            density_kg_m3,
            aircraft.wing_area_m2,
            aircraft.cd0,
            aircraft.aspect_ratio,
            aircraft.oswald_efficiency,
        )
    )


def _source_efficiency_at(aircraft: Aircraft, engine_shaft_kw: float) -> float:
    if engine_shaft_kw <= 0.0:
        return aircraft.powertrain.source_chain_efficiency
    bus_kw = float(aircraft.powertrain.bus_power_from_engine(engine_shaft_kw))
    efficiency = bus_kw / engine_shaft_kw
    if efficiency <= 0.0:
        return aircraft.powertrain.source_chain_efficiency
    return min(efficiency, 1.0)


def _equivalence_references(
    aircraft: Aircraft,
    bus_demand_kw: float,
    sigma: float,
) -> tuple[float, float]:
    """Return average-cost neutral and marginal switching references.

    Actual engine SFC is split-dependent and therefore cannot be an input to
    the controller that selects the factor used to obtain that split.  The
    engine-only power that would cover current demand is deterministic and
    preserves load dependence without introducing a fixed-point iteration.
    """
    engine_max_kw = aircraft.engine.max_power_kw(sigma)
    required_kw = float(aircraft.powertrain.engine_power_for_bus(bus_demand_kw))
    reference_kw = min(max(required_kw, aircraft.engine.idle_power_kw), engine_max_kw)
    reference_kw = max(reference_kw, _STATE_EPS)
    sfc = float(aircraft.engine.sfc_kg_kwh(reference_kw))
    source_efficiency = _source_efficiency_at(aircraft, reference_kw)
    neutral_s = neutral_equivalence_factor(sfc, source_efficiency, LHV_KJ_KG)
    switching_s = switching_equivalence_factor(
        aircraft.engine.willans_a,
        source_efficiency,
        LHV_KJ_KG,
    )
    return neutral_s, switching_s


def _dispatch(
    aircraft: Aircraft,
    phase: Phase,
    controller: EMSController,
    split_solver: Callable[..., SplitDecision],
    *,
    time_s: float,
    altitude_m: float,
    mass_kg: float,
    soc: float,
    climb_rate_mps: float,
    dt_s: float,
) -> _Dispatch:
    atmospheric = atmosphere(altitude_m)
    # Aerodynamics consumes force in newtons, never mass in kilograms.
    weight_n = mass_kg * g0
    speed_mps = _speed_for_phase(
        phase, aircraft, weight_n, atmospheric.density_kg_m3
    )
    aerodynamic = aerodynamics.evaluate(
        weight_n,
        atmospheric.density_kg_m3,
        speed_mps,
        aircraft.wing_area_m2,
        aircraft.cd0,
        aircraft.aspect_ratio,
        aircraft.oswald_efficiency,
        aircraft.propeller_efficiency,
        climb_rate_mps=climb_rate_mps,
    )
    shaft_power_kw = float(aerodynamic.shaft_power_w) / 1000.0
    bus_demand_kw = float(aircraft.powertrain.bus_power_required(shaft_power_kw))
    neutral_s, switching_s = _equivalence_references(
        aircraft, bus_demand_kw, atmospheric.density_ratio
    )
    max_bus_kw = float(
        aircraft.powertrain.bus_power_from_engine(
            aircraft.engine.max_power_kw(atmospheric.density_ratio)
        )
    ) + aircraft.battery.available_discharge_kw(soc, dt_s)
    context = ControlContext(
        soc=soc,
        bus_demand_kw=bus_demand_kw,
        max_bus_kw=max_bus_kw,
        neutral_s=neutral_s,
        switching_s=switching_s,
        time_s=time_s,
        phase=phase.name,
    )
    factor = controller.clamped_equivalence_factor(context)
    split = split_solver(
        bus_demand_kw=bus_demand_kw,
        engine=aircraft.engine,
        battery=aircraft.battery,
        powertrain=aircraft.powertrain,
        s=factor,
        soc=soc,
        sigma=atmospheric.density_ratio,
        dt_s=dt_s,
    )
    return _Dispatch(
        atmospheric=atmospheric,
        speed_mps=speed_mps,
        aerodynamic=aerodynamic,
        shaft_power_kw=shaft_power_kw,
        bus_demand_kw=bus_demand_kw,
        neutral_s=neutral_s,
        switching_s=switching_s,
        equivalence_factor=factor,
        max_bus_kw=max_bus_kw,
        split=split,
    )


def _append_flag(flags: list[str], seen: set[str], flag: str) -> None:
    if flag not in seen:
        flags.append(flag)
        seen.add(flag)


def _battery_bound_flags(
    aircraft: Aircraft,
    soc: float,
    dt_s: float,
    split: SplitDecision,
    flags: list[str],
    seen: set[str],
) -> None:
    # A zero battery flow can coincide with a zero headroom bound (full pack on
    # charge or cutoff pack on discharge) without any requested power being
    # limited.  Do not turn that inactive equality into a failure diagnostic.
    if abs(split.battery_bus_kw) <= _POWER_TOLERANCE_KW:
        return
    if split.active_bound not in (
        "battery_discharge_limit",
        "battery_charge_limit",
    ):
        return
    if split.active_bound == "battery_discharge_limit":
        instantaneous = aircraft.battery.available_discharge_kw(soc)
        step_limit = aircraft.battery.available_discharge_kw(soc, dt_s)
    else:
        instantaneous = aircraft.battery.available_charge_kw(soc)
        step_limit = aircraft.battery.available_charge_kw(soc, dt_s)
    if step_limit < instantaneous - _POWER_TOLERANCE_KW:
        _append_flag(flags, seen, "battery_energy_limited")
    else:
        _append_flag(flags, seen, "battery_rate_limited")


def run_mission(
    aircraft: Aircraft,
    mission: MissionProfile,
    controller: EMSController,
    split_solver: Callable[..., SplitDecision] = solve_split,
    dt_s: float = 60.0,
    phase_dt_s: dict[str, float] | None = None,
    initial_soc: float = 1.0,
    record_log: bool = False,
) -> MissionResult:
    """Fly ``mission`` deterministically and return outcomes and diagnostics."""
    default_dt_s = _finite_positive("dt_s", dt_s)
    initial_soc = float(initial_soc)
    if not math.isfinite(initial_soc) or not 0.0 <= initial_soc <= 1.0:
        raise ValueError(f"initial_soc must lie in [0, 1], got {initial_soc!r}")

    phase_steps = {} if phase_dt_s is None else dict(phase_dt_s)
    unknown_phases = set(phase_steps).difference(mission.phase_names)
    if unknown_phases:
        raise ValueError(f"phase_dt_s contains unknown phases: {sorted(unknown_phases)!r}")
    for name, value in phase_steps.items():
        phase_steps[name] = _finite_positive(f"phase_dt_s[{name!r}]", value)

    initial_fuel_kg = float(aircraft.masses.fuel_kg)
    fuel_remaining_kg = max(initial_fuel_kg, 0.0)
    soc = initial_soc
    altitude_m = float(mission.initial_altitude_m)
    time_s = 0.0
    min_soc = soc
    peak_bus_kw = 0.0
    peak_engine_kw = 0.0
    efficiency_time_integral = 0.0
    integrated_time_s = 0.0
    phase_durations = {phase.name: 0.0 for phase in mission.phases}
    flags: list[str] = []
    seen_flags: set[str] = set()
    log_entries: list[TimeStep] | None = [] if record_log else None
    termination_reason = "mission_complete"
    resource_reason: str | None = None
    terminal_failure = False
    all_phases_completed = True
    engine_was_shut_down = False

    if initial_fuel_kg <= 0.0:
        _append_flag(flags, seen_flags, "fuel_exhausted")
        termination_reason = "fuel_exhausted"
        terminal_failure = True
        all_phases_completed = False

    for phase in mission.phases:
        if terminal_failure:
            break
        phase_elapsed_s = 0.0

        while True:
            if phase.termination is Termination.DURATION:
                remaining_phase_s = float(phase.duration_s) - phase_elapsed_s
                if remaining_phase_s <= _STATE_EPS:
                    break
            elif phase.termination is Termination.ALTITUDE:
                if abs(altitude_m - phase.target_altitude_m) <= _STATE_EPS:
                    altitude_m = float(phase.target_altitude_m)
                    break
            else:
                if fuel_remaining_kg <= mission.loiter_fuel_floor_kg + _STATE_EPS:
                    fuel_remaining_kg = max(
                        fuel_remaining_kg, mission.loiter_fuel_floor_kg
                    )
                    resource_reason = "fuel_reserve"
                    break
                if soc <= aircraft.battery.soc_min + SOC_EPS:
                    resource_reason = "soc_cutoff"
                    break

            if time_s >= mission.max_mission_time_s - _STATE_EPS:
                termination_reason = "max_mission_time"
                terminal_failure = True
                all_phases_completed = False
                break
            if fuel_remaining_kg <= _STATE_EPS:
                _append_flag(flags, seen_flags, "fuel_exhausted")
                termination_reason = "fuel_exhausted"
                terminal_failure = True
                all_phases_completed = False
                break

            step_dt_s = min(
                phase_steps.get(phase.name, default_dt_s),
                mission.max_mission_time_s - time_s,
            )
            if phase.termination is Termination.DURATION:
                step_dt_s = min(step_dt_s, remaining_phase_s)

            mass_kg = aircraft.masses.dry_kg + fuel_remaining_kg
            atmospheric = atmosphere(altitude_m)
            weight_n = mass_kg * g0
            speed_mps = _speed_for_phase(
                phase, aircraft, weight_n, atmospheric.density_kg_m3
            )

            climb_rate_mps = 0.0
            climb_was_limited = False
            if phase.altitude_mode is AltitudeMode.CLIMB_TO:
                target_rate_mps = float(phase.climb_rate_mps)
                engine_bus_max_kw = float(
                    aircraft.powertrain.bus_power_from_engine(
                        aircraft.engine.max_power_kw(atmospheric.density_ratio)
                    )
                )
                battery_bus_max_kw = aircraft.battery.available_discharge_kw(
                    soc, step_dt_s
                )
                shaft_available_kw = float(
                    aircraft.powertrain.shaft_power_from_bus(
                        engine_bus_max_kw + battery_bus_max_kw
                    )
                )
                max_rate_mps = float(
                    aerodynamics.rate_of_climb(
                        weight_n,
                        atmospheric.density_kg_m3,
                        speed_mps,
                        aircraft.wing_area_m2,
                        aircraft.cd0,
                        aircraft.aspect_ratio,
                        aircraft.oswald_efficiency,
                        aircraft.propeller_efficiency,
                        shaft_available_kw * 1000.0,
                    )
                )
                climb_rate_mps = min(target_rate_mps, max_rate_mps)
                if climb_rate_mps < target_rate_mps - _STATE_EPS:
                    climb_was_limited = True
                    _append_flag(flags, seen_flags, "climb_rate_unachievable")
                    _append_flag(flags, seen_flags, "engine_power_limited")
                    instantaneous = aircraft.battery.available_discharge_kw(soc)
                    if battery_bus_max_kw < instantaneous - _POWER_TOLERANCE_KW:
                        _append_flag(flags, seen_flags, "battery_energy_limited")
                    else:
                        _append_flag(flags, seen_flags, "battery_rate_limited")
                if climb_rate_mps <= 0.0:
                    _append_flag(flags, seen_flags, "altitude_unreachable")
                    termination_reason = "altitude_unreachable"
                    terminal_failure = True
                    all_phases_completed = False
                    break
            elif phase.altitude_mode is AltitudeMode.DESCEND_TO:
                climb_rate_mps = float(phase.climb_rate_mps)

            if phase.termination is Termination.ALTITUDE:
                altitude_delta_m = phase.target_altitude_m - altitude_m
                time_to_target_s = altitude_delta_m / climb_rate_mps
                if time_to_target_s <= 0.0:
                    _append_flag(flags, seen_flags, "altitude_unreachable")
                    termination_reason = "altitude_unreachable"
                    terminal_failure = True
                    all_phases_completed = False
                    break
                step_dt_s = min(step_dt_s, time_to_target_s)

            # Fuel-boundary shortening is normally exercised once, but repeat
            # because changing dt can change battery energy availability and
            # therefore the selected engine power.
            dispatch: _Dispatch | None = None
            engine_state: EngineState | None = None
            restart_fuel_kg = 0.0
            fuel_boundary_prevents_step = False
            for _ in range(6):
                dispatch = _dispatch(
                    aircraft,
                    phase,
                    controller,
                    split_solver,
                    time_s=time_s,
                    altitude_m=altitude_m,
                    mass_kg=mass_kg,
                    soc=soc,
                    climb_rate_mps=climb_rate_mps,
                    dt_s=step_dt_s,
                )
                if not dispatch.split.feasible:
                    break
                engine_state = aircraft.engine.operate(
                    dispatch.split.engine_shaft_kw,
                    dispatch.atmospheric.density_ratio,
                )
                restart_fuel_kg = (
                    aircraft.engine.restart_fuel_kg
                    if engine_was_shut_down and not engine_state.shut_down
                    else 0.0
                )
                fuel_floor_kg = (
                    mission.loiter_fuel_floor_kg
                    if phase.termination is Termination.RESOURCE
                    else 0.0
                )
                expendable_kg = max(fuel_remaining_kg - fuel_floor_kg, 0.0)
                burn_kg = engine_state.fuel_flow_kg_s * step_dt_s + restart_fuel_kg
                if burn_kg <= expendable_kg + _STATE_EPS or burn_kg <= 0.0:
                    break
                fuel_for_operation_kg = expendable_kg - restart_fuel_kg
                if fuel_for_operation_kg <= _STATE_EPS:
                    fuel_boundary_prevents_step = True
                    break
                shortened_dt_s = fuel_for_operation_kg / engine_state.fuel_flow_kg_s
                if shortened_dt_s <= _STATE_EPS:
                    fuel_boundary_prevents_step = True
                    break
                step_dt_s = shortened_dt_s

            assert dispatch is not None
            if fuel_boundary_prevents_step:
                if phase.termination is Termination.RESOURCE:
                    resource_reason = "fuel_reserve"
                else:
                    fuel_remaining_kg = 0.0
                    _append_flag(flags, seen_flags, "fuel_exhausted")
                    termination_reason = "fuel_exhausted"
                    terminal_failure = True
                    all_phases_completed = False
                break
            if not dispatch.split.feasible:
                _append_flag(flags, seen_flags, "power_shortfall")
                termination_reason = "power_shortfall"
                terminal_failure = True
                all_phases_completed = False
                break
            assert engine_state is not None

            balance_kw = (
                dispatch.split.bus_from_engine_kw
                + dispatch.split.battery_bus_kw
                - dispatch.bus_demand_kw
            )
            if abs(balance_kw) > _POWER_TOLERANCE_KW:
                _append_flag(flags, seen_flags, "power_shortfall")
                termination_reason = "power_shortfall"
                terminal_failure = True
                all_phases_completed = False
                break

            pre_step_soc = soc
            battery_state = aircraft.battery.step(
                pre_step_soc, dispatch.split.battery_bus_kw, step_dt_s
            )
            if abs(battery_state.power_kw - dispatch.split.battery_bus_kw) > _POWER_TOLERANCE_KW:
                if battery_state.rate_limited:
                    _append_flag(flags, seen_flags, "battery_rate_limited")
                if battery_state.energy_limited:
                    _append_flag(flags, seen_flags, "battery_energy_limited")
                _append_flag(flags, seen_flags, "power_shortfall")
                termination_reason = "power_shortfall"
                terminal_failure = True
                all_phases_completed = False
                break

            _battery_bound_flags(
                aircraft,
                soc,
                step_dt_s,
                dispatch.split,
                flags,
                seen_flags,
            )
            if dispatch.split.active_bound == "engine_max":
                _append_flag(flags, seen_flags, "engine_power_limited")
            battery_flow_is_material = (
                abs(dispatch.split.battery_bus_kw) > _POWER_TOLERANCE_KW
            )
            if battery_state.rate_limited and battery_flow_is_material:
                _append_flag(flags, seen_flags, "battery_rate_limited")
            if battery_state.energy_limited and battery_flow_is_material:
                _append_flag(flags, seen_flags, "battery_energy_limited")

            fuel_burned_kg = (
                engine_state.fuel_flow_kg_s * step_dt_s + restart_fuel_kg
            )
            fuel_remaining_kg -= fuel_burned_kg
            if (
                phase.termination is Termination.RESOURCE
                and abs(fuel_remaining_kg - mission.loiter_fuel_floor_kg) <= _STATE_EPS
            ):
                fuel_remaining_kg = mission.loiter_fuel_floor_kg
            soc = battery_state.soc
            engine_was_shut_down = engine_state.shut_down
            altitude_m += climb_rate_mps * step_dt_s
            if phase.termination is Termination.ALTITUDE:
                if (
                    climb_rate_mps > 0.0
                    and altitude_m >= phase.target_altitude_m - _STATE_EPS
                ) or (
                    climb_rate_mps < 0.0
                    and altitude_m <= phase.target_altitude_m + _STATE_EPS
                ):
                    altitude_m = float(phase.target_altitude_m)
            time_s += step_dt_s
            phase_elapsed_s += step_dt_s
            phase_durations[phase.name] = phase_elapsed_s
            min_soc = min(min_soc, soc)
            peak_bus_kw = max(peak_bus_kw, dispatch.bus_demand_kw)
            peak_engine_kw = max(peak_engine_kw, engine_state.delivered_kw)

            fuel_chemical_kw = (
                engine_state.fuel_flow_kg_s + restart_fuel_kg / step_dt_s
            ) * LHV_KJ_KG
            propulsive_kw = max(float(dispatch.aerodynamic.thrust_power_w), 0.0) / 1000.0
            system_efficiency = aircraft.powertrain.system_efficiency(
                propulsive_kw,
                fuel_chemical_kw,
                battery_state.power_kw,
            )
            efficiency_time_integral += system_efficiency * step_dt_s
            integrated_time_s += step_dt_s

            if log_entries is not None:
                post_weight_n = (aircraft.masses.dry_kg + fuel_remaining_kg) * g0
                bus_from_engine_kw = float(
                    aircraft.powertrain.bus_power_from_engine(engine_state.delivered_kw)
                )
                battery_internal_kw = (
                    battery_state.open_circuit_voltage_v * battery_state.current_a / 1000.0
                )
                battery_stored_energy_change_kwh = (
                    float(aircraft.battery.stored_energy_kwh(battery_state.soc))
                    - float(aircraft.battery.stored_energy_kwh(pre_step_soc))
                )
                engine_thermal_loss_kw = fuel_chemical_kw - engine_state.delivered_kw
                source_losses_kw = engine_state.delivered_kw - bus_from_engine_kw
                demand_losses_kw = dispatch.bus_demand_kw - dispatch.shaft_power_kw
                propeller_losses_kw = dispatch.shaft_power_kw - propulsive_kw
                power_limited = (
                    climb_was_limited
                    or dispatch.split.active_bound == "engine_max"
                    or (
                        battery_flow_is_material
                        and dispatch.split.active_bound
                        in ("battery_discharge_limit", "battery_charge_limit")
                    )
                )
                log_entries.append(
                    TimeStep(
                        time_s=time_s,
                        phase=phase.name,
                        altitude_m=altitude_m,
                        speed_mps=dispatch.speed_mps,
                        weight_n=post_weight_n,
                        density_kg_m3=dispatch.atmospheric.density_kg_m3,
                        lift_coefficient=dispatch.aerodynamic.lift_coefficient,
                        drag_n=dispatch.aerodynamic.drag_n,
                        shaft_power_kw=dispatch.shaft_power_kw,
                        bus_demand_kw=dispatch.bus_demand_kw,
                        neutral_s=dispatch.neutral_s,
                        switching_s=dispatch.switching_s,
                        equivalence_factor=dispatch.equivalence_factor,
                        engine_shaft_kw=engine_state.delivered_kw,
                        engine_load_fraction=engine_state.load_fraction,
                        sfc_kg_kwh=engine_state.sfc_kg_kwh,
                        fuel_flow_kg_s=engine_state.fuel_flow_kg_s,
                        restart_fuel_kg=restart_fuel_kg,
                        engine_shut_down=engine_state.shut_down,
                        battery_bus_kw=battery_state.power_kw,
                        soc=soc,
                        fuel_remaining_kg=fuel_remaining_kg,
                        system_efficiency=system_efficiency,
                        power_off=bool(dispatch.aerodynamic.power_off),
                        power_limited=power_limited,
                        dt_s=step_dt_s,
                        bus_from_engine_kw=bus_from_engine_kw,
                        battery_internal_kw=battery_internal_kw,
                        battery_ohmic_loss_kw=battery_state.ohmic_loss_kw,
                        battery_stored_energy_change_kwh=(
                            battery_stored_energy_change_kwh
                        ),
                        thrust_power_kw=propulsive_kw,
                        engine_thermal_loss_kw=engine_thermal_loss_kw,
                        source_losses_kw=source_losses_kw,
                        demand_losses_kw=demand_losses_kw,
                        propeller_losses_kw=propeller_losses_kw,
                    )
                )

            if fuel_remaining_kg <= _STATE_EPS:
                fuel_remaining_kg = max(fuel_remaining_kg, 0.0)
                _append_flag(flags, seen_flags, "fuel_exhausted")
                termination_reason = "fuel_exhausted"
                terminal_failure = True
                all_phases_completed = False
                break

        if terminal_failure:
            break

    if all_phases_completed and not terminal_failure:
        if fuel_remaining_kg < mission.fuel_reserve_kg - _STATE_EPS:
            _append_flag(flags, seen_flags, "fuel_reserve_shortfall")
            termination_reason = "fuel_reserve_shortfall"
            terminal_failure = True
            all_phases_completed = False
        else:
            termination_reason = resource_reason or "mission_complete"

    mean_efficiency = (
        efficiency_time_integral / integrated_time_s if integrated_time_s > 0.0 else 0.0
    )
    return MissionResult(
        endurance_s=time_s,
        mission_complete=all_phases_completed and not terminal_failure,
        termination_reason=termination_reason,
        phase_durations_s=phase_durations,
        fuel_used_kg=initial_fuel_kg - fuel_remaining_kg,
        fuel_remaining_kg=fuel_remaining_kg,
        final_soc=soc,
        min_soc=min_soc,
        peak_bus_kw=peak_bus_kw,
        peak_engine_kw=peak_engine_kw,
        mean_system_efficiency=mean_efficiency,
        failure_flags=tuple(flags),
        log=None if log_entries is None else tuple(log_entries),
    )


def mission_energy_balance(result: MissionResult) -> MissionEnergyBalance:
    """Integrate the per-step energy ledger for a recorded mission.

    The primary residual uses endpoint energy from the integrated SoC and the
    linear OCV curve.  The discrete residual separately uses
    ``delta_E_stored = -V_oc,start*I*dt`` and therefore isolates conversion
    accounting from the explicit-Euler OCV integration bias.
    """
    if result.log is None:
        raise ValueError("mission_energy_balance requires run_mission(record_log=True)")

    fuel_chemical_kwh = sum(
        (step.fuel_flow_kg_s * step.dt_s + step.restart_fuel_kg)
        * LHV_KJ_KG
        / 3600.0
        for step in result.log
    )
    propulsive_kwh = sum(
        step.thrust_power_kw * step.dt_s / 3600.0 for step in result.log
    )
    engine_thermal_kwh = sum(
        step.engine_thermal_loss_kw * step.dt_s / 3600.0
        for step in result.log
    )
    source_loss_kwh = sum(
        step.source_losses_kw * step.dt_s / 3600.0 for step in result.log
    )
    demand_loss_kwh = sum(
        step.demand_losses_kw * step.dt_s / 3600.0 for step in result.log
    )
    propeller_loss_kwh = sum(
        step.propeller_losses_kw * step.dt_s / 3600.0 for step in result.log
    )
    battery_loss_kwh = sum(
        step.battery_ohmic_loss_kw * step.dt_s / 3600.0
        for step in result.log
    )
    stored_change_kwh = sum(
        step.battery_stored_energy_change_kwh for step in result.log
    )
    discrete_stored_change_kwh = -sum(
        step.battery_internal_kw * step.dt_s / 3600.0 for step in result.log
    )
    accounted_kwh = (
        propulsive_kwh
        + engine_thermal_kwh
        + source_loss_kwh
        + demand_loss_kwh
        + propeller_loss_kwh
        + battery_loss_kwh
        + stored_change_kwh
    )
    residual_kwh = fuel_chemical_kwh - accounted_kwh
    discrete_accounted_kwh = (
        accounted_kwh - stored_change_kwh + discrete_stored_change_kwh
    )
    discrete_residual_kwh = fuel_chemical_kwh - discrete_accounted_kwh
    scale_kwh = max(abs(fuel_chemical_kwh), 1.0e-30)
    return MissionEnergyBalance(
        fuel_chemical_in_kwh=fuel_chemical_kwh,
        propulsive_work_out_kwh=propulsive_kwh,
        engine_thermal_loss_kwh=engine_thermal_kwh,
        source_chain_loss_kwh=source_loss_kwh,
        demand_chain_loss_kwh=demand_loss_kwh,
        propeller_loss_kwh=propeller_loss_kwh,
        battery_ohmic_loss_kwh=battery_loss_kwh,
        battery_stored_energy_change_kwh=stored_change_kwh,
        discrete_battery_stored_energy_change_kwh=discrete_stored_change_kwh,
        battery_integration_residual_kwh=(
            stored_change_kwh - discrete_stored_change_kwh
        ),
        residual_kwh=residual_kwh,
        residual_fraction=abs(residual_kwh) / scale_kwh,
        discrete_residual_fraction=abs(discrete_residual_kwh) / scale_kwh,
    )


def log_to_dataframe(result: MissionResult) -> "pd.DataFrame":
    """Convert a recorded log to a pandas DataFrame (empty when no log exists)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency failure is environmental.
        raise ImportError(
            "log_to_dataframe requires pandas; install the pinned requirements.txt"
        ) from exc

    columns = [field.name for field in fields(TimeStep)]
    if result.log is None:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records((asdict(step) for step in result.log), columns=columns)
