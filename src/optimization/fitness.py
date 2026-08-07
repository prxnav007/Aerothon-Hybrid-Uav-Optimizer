"""One-chromosome full-mission fitness evaluation for the later GA.

Static screening is performed before any simulator call. A screened candidate
is built into fresh immutable plant and thermostat inputs, flown exactly once,
and audited without assigning scalar endurance to an infeasible mission.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from statistics import fmean
from typing import Any, Literal

from src.control.thermostat import (
    DwellSemantics,
    TerminalStrategy,
    ThermostatParameters,
    ThermostatState,
)
from src.models.battery import SOC_SAFE_MIN, BatteryPack
from src.models.engine import (
    IDLE_FUEL_FRACTION,
    MIN_POWER_FRACTION,
    SFC_RATED_KG_KWH,
    Turboshaft,
)
from src.models.mass import MassBreakdown
from src.models.powertrain import SeriesPowertrain
from src.optimization.chromosome import (
    DecodedPlantThermostatDesign,
    NormalizedChromosome,
    PlantThermostatDesignSpace,
    decode_chromosome,
)
from src.optimization.feasibility import (
    ConstraintRecord,
    ResolvedPlantDesign,
    StaticFeasibilityResult,
    StaticFeasibilityScenario,
    evaluate_static_feasibility,
)
from src.simulation.mission import MissionProfile, ps1_mission
from src.simulation.simulator import (
    Aircraft,
    MissionResult,
    TimeStep,
    mission_energy_balance,
    run_mission,
)

__all__ = [
    "CandidateMissionInfeasibleError",
    "ConstructedMissionInputs",
    "ControllerBehavior",
    "DurationSummary",
    "DynamicConstraintRecord",
    "DYNAMIC_CONSTRAINT_NAMES",
    "FitnessDiagnostic",
    "FitnessResult",
    "FitnessScenario",
    "MissionResources",
    "MissionValidity",
    "PowerRange",
    "construct_mission_inputs",
    "evaluate_fitness",
    "evaluation_identity",
    "resolved_design_identity",
]

FITNESS_RESULT_SCHEMA_VERSION = 1
_FITNESS_SCENARIO_SCHEMA_VERSION = 1

DYNAMIC_CONSTRAINT_NAMES = (
    "mission_log_present",
    "six_phase_order",
    "takeoff_completed",
    "climb_completed",
    "cruise_completed",
    "loiter_entered",
    "loiter_completed",
    "descent_completed",
    "landing_completed",
    "mission_complete",
    "fuel_reserve",
    "soc_floor",
    "controller_feasible",
    "plant_feasible",
    "hard_dwell",
    "restart_accounting",
    "terminal_failure_flags",
    "candidate_mission_exception",
    "bus_power_ledger",
    "fuel_ledger",
    "discrete_energy_ledger",
)

_TERMINAL_FAILURE_FLAGS = frozenset(
    {
        "power_shortfall",
        "fuel_exhausted",
        "fuel_reserve_shortfall",
        "altitude_unreachable",
        "controller_infeasible",
        "hard_dwell_infeasible",
        "max_mission_time",
    }
)


def _finite(name: str, value: Any) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _report_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _report_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _report_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_report_value(item) for item in value]
    return value


def _digest(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class FitnessScenario:
    """All non-gene inputs and tolerances for one mission evaluation."""

    name: str
    classification: str
    static_scenario: StaticFeasibilityScenario
    mission: MissionProfile
    timestep_s: float
    initial_soc: float
    restart_fuel_kg: float
    minimum_on_time_s: float
    minimum_off_time_s: float
    initial_engine_on: bool
    initial_elapsed_in_state_s: float
    terminal_strategy: TerminalStrategy | str
    dwell_semantics: DwellSemantics | str
    engine_on_power_kw: float | None
    engine_sfc_rated_kg_kwh: float
    engine_idle_fuel_fraction: float
    engine_min_power_fraction: float
    battery_safe_soc_floor: float
    phase_completion_tolerance_s: float
    reserve_tolerance_kg: float
    soc_floor_tolerance: float
    power_balance_tolerance_kw: float
    fuel_ledger_tolerance_kg: float
    discrete_energy_ledger_tolerance_fraction: float
    restart_accounting_tolerance_count: float
    normalization_epsilon: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.classification, str) or not self.classification:
            raise ValueError("classification must be a non-empty string")
        if not isinstance(self.static_scenario, StaticFeasibilityScenario):
            raise ValueError("static_scenario must be a StaticFeasibilityScenario")
        if not isinstance(self.mission, MissionProfile):
            raise ValueError("mission must be a MissionProfile")
        if not isinstance(self.initial_engine_on, bool):
            raise ValueError("initial_engine_on must be boolean")
        numeric = (
            "timestep_s", "initial_soc", "restart_fuel_kg",
            "minimum_on_time_s", "minimum_off_time_s",
            "initial_elapsed_in_state_s", "engine_sfc_rated_kg_kwh",
            "engine_idle_fuel_fraction", "engine_min_power_fraction",
            "battery_safe_soc_floor", "phase_completion_tolerance_s",
            "reserve_tolerance_kg", "soc_floor_tolerance",
            "power_balance_tolerance_kw", "fuel_ledger_tolerance_kg",
            "discrete_energy_ledger_tolerance_fraction",
            "restart_accounting_tolerance_count", "normalization_epsilon",
        )
        for field_name in numeric:
            object.__setattr__(self, field_name, _finite(field_name, getattr(self, field_name)))
        if self.engine_on_power_kw is not None:
            value = _finite("engine_on_power_kw", self.engine_on_power_kw)
            if value <= 0.0:
                raise ValueError("engine_on_power_kw must be positive when supplied")
            object.__setattr__(self, "engine_on_power_kw", value)
        if self.timestep_s <= 0.0 or self.engine_sfc_rated_kg_kwh <= 0.0:
            raise ValueError("timestep and rated SFC must be positive")
        if self.restart_fuel_kg < 0.0:
            raise ValueError("restart_fuel_kg must be non-negative")
        if min(
            self.minimum_on_time_s,
            self.minimum_off_time_s,
            self.initial_elapsed_in_state_s,
        ) < 0.0:
            raise ValueError("dwell and initial elapsed times must be non-negative")
        if not self.static_scenario.battery_soc_floor <= self.initial_soc <= 1.0:
            raise ValueError("initial_soc must lie within the usable battery interval")
        if not 0.0 <= self.engine_idle_fuel_fraction < 1.0:
            raise ValueError("engine_idle_fuel_fraction must lie in [0, 1)")
        if not 0.0 < self.engine_min_power_fraction <= 1.0:
            raise ValueError("engine_min_power_fraction must lie in (0, 1]")
        if not self.static_scenario.battery_soc_floor <= self.battery_safe_soc_floor <= 1.0:
            raise ValueError("battery_safe_soc_floor must not be below the hard floor")
        for field_name in (
            "phase_completion_tolerance_s", "reserve_tolerance_kg",
            "soc_floor_tolerance", "power_balance_tolerance_kw",
            "fuel_ledger_tolerance_kg",
            "discrete_energy_ledger_tolerance_fraction",
            "restart_accounting_tolerance_count",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")
        try:
            strategy = TerminalStrategy(self.terminal_strategy)
            dwell = DwellSemantics(self.dwell_semantics)
        except (TypeError, ValueError) as error:
            raise ValueError("fitness supports causal thermostat scheduling with hard dwell") from error
        if strategy is not TerminalStrategy.CAUSAL or dwell is not DwellSemantics.HARD:
            raise ValueError("fitness supports causal thermostat scheduling with hard dwell")
        object.__setattr__(self, "terminal_strategy", strategy)
        object.__setattr__(self, "dwell_semantics", dwell)
        self._validate_mission_alignment()

    def _validate_mission_alignment(self) -> None:
        static = self.static_scenario
        mission = self.mission
        checks = (
            (mission.phase_by_name("cruise").target_altitude_m, static.cruise_altitude_m),
            (mission.phase_by_name("cruise").speed_mps, static.cruise_speed_mps),
            (mission.phase_by_name("climb").speed_mps, static.climb_speed_mps),
            (mission.phase_by_name("climb").climb_rate_mps, static.climb_rate_mps),
            (mission.phase_by_name("takeoff").speed_mps, static.takeoff_speed_mps),
            (mission.phase_by_name("landing").speed_mps, static.landing_speed_mps),
            (mission.min_usable_fuel_kg, static.minimum_usable_fuel_kg),
        )
        if any(float(left) != float(right) for left, right in checks):
            raise ValueError("fitness mission and static screen use different design conditions")

    @classmethod
    def nominal(cls) -> FitnessScenario:
        """Return the named uncalibrated 0.1 kg/start practical scenario."""
        static = StaticFeasibilityScenario.nominal()
        return cls(
            name="practical_3km_thermostat_restart_0_1",
            classification="uncalibrated practical restart-fuel sensitivity",
            static_scenario=static,
            mission=ps1_mission(cruise_altitude_m=static.cruise_altitude_m),
            timestep_s=static.timestep_s,
            initial_soc=1.0,
            restart_fuel_kg=0.1,
            minimum_on_time_s=60.0,
            minimum_off_time_s=60.0,
            initial_engine_on=True,
            initial_elapsed_in_state_s=60.0,
            terminal_strategy=TerminalStrategy.CAUSAL,
            dwell_semantics=DwellSemantics.HARD,
            engine_on_power_kw=None,
            engine_sfc_rated_kg_kwh=SFC_RATED_KG_KWH,
            engine_idle_fuel_fraction=IDLE_FUEL_FRACTION,
            engine_min_power_fraction=MIN_POWER_FRACTION,
            battery_safe_soc_floor=SOC_SAFE_MIN,
            phase_completion_tolerance_s=1.0e-9,
            reserve_tolerance_kg=1.0e-10,
            soc_floor_tolerance=1.0e-10,
            power_balance_tolerance_kw=1.0e-6,
            fuel_ledger_tolerance_kg=1.0e-9,
            discrete_energy_ledger_tolerance_fraction=1.0e-12,
            restart_accounting_tolerance_count=1.0e-9,
            normalization_epsilon=1.0e-12,
        )

    @property
    def identity(self) -> str:
        return _digest(
            f"fitness-scenario-v{_FITNESS_SCENARIO_SCHEMA_VERSION}",
            {"schema_version": _FITNESS_SCENARIO_SCHEMA_VERSION, **asdict(self)},
        )

    def to_dict(self) -> dict[str, Any]:
        return _report_value(
            {
                "schema_version": _FITNESS_SCENARIO_SCHEMA_VERSION,
                "fitness_scenario_id": self.identity,
                **asdict(self),
            }
        )


@dataclass(frozen=True)
class DurationSummary:
    count: int
    minimum_s: float
    mean_s: float
    maximum_s: float


@dataclass(frozen=True)
class PowerRange:
    minimum_kw: float
    maximum_kw: float


@dataclass(frozen=True)
class MissionResources:
    initial_fuel_kg: float
    final_fuel_kg: float
    total_fuel_consumed_kg: float
    running_fuel_consumed_kg: float
    restart_fuel_consumed_kg: float
    final_soc: float
    minimum_soc: float
    final_stored_battery_energy_kwh: float
    fuel_slack_above_reserve_kg: float
    usable_battery_energy_above_floor_kwh: float


@dataclass(frozen=True)
class ControllerBehavior:
    restart_count: int
    restarts_per_loiter_hour: float
    overall_engine_off_fraction: float
    loiter_engine_off_fraction: float
    on_run_durations: DurationSummary
    off_run_durations: DurationSummary
    requested_engine_power_range: PowerRange
    delivered_engine_power_range: PowerRange
    charge_limit_encounter_count: int
    discharge_limit_encounter_count: int


@dataclass(frozen=True)
class MissionValidity:
    completed_takeoff: bool
    completed_climb: bool
    completed_cruise: bool
    entered_loiter: bool
    completed_loiter_process: bool
    completed_descent: bool
    completed_landing: bool
    termination_reason: str
    reserve_shortfall_kg: float
    soc_floor_violation: float
    controller_infeasibility_count: int
    plant_infeasibility_count: int
    hard_dwell_violation_count: int
    hidden_or_unaccounted_restart_count: float
    maximum_bus_power_residual_kw: float | None
    fuel_ledger_residual_kg: float | None
    battery_energy_ledger_residual_kwh: float | None
    discrete_energy_ledger_residual_fraction: float | None
    battery_integration_residual_kwh: float | None
    failure_flags: tuple[str, ...]


@dataclass(frozen=True)
class DynamicConstraintRecord:
    name: str
    quantity: float
    required_or_allowed: float
    tolerance: float
    raw_violation: float
    normalization_scale: float
    normalized_violation: float
    unit: str
    satisfied: bool
    relationship: Literal["at_least", "at_most"]
    source: str

    @classmethod
    def at_least(
        cls, name: str, actual: float, required: float, tolerance: float,
        scale: float, unit: str, source: str, epsilon: float,
    ) -> DynamicConstraintRecord:
        raw = max(required - actual - tolerance, 0.0)
        denominator = max(abs(scale), epsilon)
        return cls(
            name, actual, required, tolerance, raw, denominator,
            raw / denominator, unit, raw == 0.0, "at_least", source,
        )

    @classmethod
    def at_most(
        cls, name: str, actual: float, allowed: float, tolerance: float,
        scale: float, unit: str, source: str, epsilon: float,
    ) -> DynamicConstraintRecord:
        raw = max(actual - allowed - tolerance, 0.0)
        denominator = max(abs(scale), epsilon)
        return cls(
            name, actual, allowed, tolerance, raw, denominator,
            raw / denominator, unit, raw == 0.0, "at_most", source,
        )


@dataclass(frozen=True)
class FitnessDiagnostic:
    name: str
    message: str
    source: str


@dataclass(frozen=True)
class ConstructedMissionInputs:
    aircraft: Aircraft
    mission: MissionProfile
    thermostat_parameters: ThermostatParameters
    initial_thermostat_state: ThermostatState
    timestep_s: float
    initial_soc: float
    resolved_design_id: str
    fitness_scenario_id: str


@dataclass(frozen=True)
class FitnessResult:
    schema_version: int
    chromosome_cache_key: str
    design_space_id: str
    static_scenario_id: str
    fitness_scenario_id: str
    evaluation_key: str
    decoded_design: DecodedPlantThermostatDesign
    resolved_design_id: str
    resolved_design: ResolvedPlantDesign
    static_feasible: bool
    static_normalized_total_violation: float
    static_constraints: tuple[ConstraintRecord, ...]
    run_mission_called: bool
    dynamically_feasible: bool
    objective_loiter_seconds: float | None
    objective_hours: float | None
    observed_loiter_seconds: float | None
    total_mission_seconds: float | None
    resources: MissionResources | None
    controller_behavior: ControllerBehavior | None
    validity: MissionValidity | None
    dynamic_constraint_names: tuple[str, ...]
    dynamic_constraints: tuple[DynamicConstraintRecord, ...]
    total_normalized_dynamic_violation: float
    combined_static_dynamic_violation: float
    warnings: tuple[FitnessDiagnostic, ...]
    failure_category: str | None
    failure_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return _report_value(asdict(self))


class CandidateMissionInfeasibleError(Exception):
    """Documented adapter signal for candidate-specific physical failure."""

    def __init__(
        self, category: str, message: str, *, normalized_violation: float = 1.0
    ) -> None:
        if not isinstance(category, str) or not category:
            raise ValueError("category must be a non-empty string")
        violation = _finite("normalized_violation", normalized_violation)
        if violation <= 0.0:
            raise ValueError("normalized_violation must be positive")
        super().__init__(message)
        self.category = category
        self.normalized_violation = violation


def resolved_design_identity(resolved: ResolvedPlantDesign) -> str:
    if not isinstance(resolved, ResolvedPlantDesign):
        raise ValueError("resolved must be a ResolvedPlantDesign")
    return _digest("resolved-plant-design-v1", resolved.to_dict())


def evaluation_identity(
    chromosome: NormalizedChromosome,
    *,
    bounds: PlantThermostatDesignSpace,
    scenario: FitnessScenario,
) -> str:
    if not isinstance(chromosome, NormalizedChromosome):
        raise ValueError("chromosome must be a NormalizedChromosome")
    if not isinstance(bounds, PlantThermostatDesignSpace):
        raise ValueError("bounds must be a PlantThermostatDesignSpace")
    if not isinstance(scenario, FitnessScenario):
        raise ValueError("scenario must be a FitnessScenario")
    payload = {
        "fitness_result_schema_version": FITNESS_RESULT_SCHEMA_VERSION,
        "chromosome_cache_key": chromosome.cache_key(bounds=bounds),
        "design_space_id": bounds.identifier,
        "static_scenario_id": scenario.static_scenario.identity,
        "fitness_scenario_id": scenario.identity,
    }
    return _digest(f"fitness-evaluation-v{FITNESS_RESULT_SCHEMA_VERSION}", payload)


def construct_mission_inputs(
    resolved: ResolvedPlantDesign,
    *,
    scenario: FitnessScenario,
) -> ConstructedMissionInputs:
    """Build fresh simulator inputs from one resolved static design."""
    if not isinstance(resolved, ResolvedPlantDesign):
        raise ValueError("resolved must be a ResolvedPlantDesign")
    if not isinstance(scenario, FitnessScenario):
        raise ValueError("scenario must be a FitnessScenario")
    static = scenario.static_scenario
    if resolved.scenario_id != static.identity:
        raise ValueError("resolved design belongs to a different static scenario")
    design = resolved.decoded
    mass = resolved.masses
    masses = MassBreakdown(
        fixed_kg=mass.fixed_group_kg,
        payload_kg=mass.payload_kg,
        wing_kg=mass.wing_kg,
        engine_kg=mass.engine_kg,
        generator_kg=mass.generator_kg,
        rectifier_kg=mass.rectifier_kg,
        inverter_kg=mass.inverter_kg,
        motor_kg=mass.motor_kg,
        cabling_cooling_kg=mass.cabling_cooling_kg,
        battery_kg=mass.battery_kg,
        fuel_system_kg=mass.fuel_system_kg,
        fuel_kg=mass.fuel_kg,
    )
    if abs(masses.total_kg - static.mtow_kg) > static.mass_closure_tolerance_kg:
        raise ValueError("resolved mass budget does not close to the static MTOW")
    engine = Turboshaft(
        rated_power_kw=design.engine_rating_kw,
        sfc_rated_kg_kwh=scenario.engine_sfc_rated_kg_kwh,
        idle_fuel_fraction=scenario.engine_idle_fuel_fraction,
        lapse_exponent=static.engine_lapse_exponent,
        min_power_fraction=scenario.engine_min_power_fraction,
        allow_shutdown=static.engine_allow_shutdown,
        restart_fuel_kg=scenario.restart_fuel_kg,
    )
    battery = BatteryPack(
        capacity_kwh=design.battery_capacity_kwh,
        v_min_v=static.battery_v_min_v,
        v_max_v=static.battery_v_max_v,
        r_ref_ohm=static.battery_r_ref_ohm,
        r_ref_capacity_kwh=static.battery_r_ref_capacity_kwh,
        scale_resistance=static.battery_scale_resistance,
        discharge_c_rate=static.battery_discharge_c_rate,
        charge_c_rate=static.battery_charge_c_rate,
        soc_min=static.battery_soc_floor,
        soc_safe_min=scenario.battery_safe_soc_floor,
        mode=static.battery_mode,
    )
    powertrain = SeriesPowertrain(
        eta_generator=static.eta_generator,
        eta_rectifier=static.eta_rectifier,
        eta_cabling=static.eta_cabling,
        eta_inverter=static.eta_inverter,
        eta_motor=static.eta_motor,
        load_dependent=static.powertrain_load_dependent,
        noload_loss_fraction=static.powertrain_noload_loss_fraction,
        rated_engine_kw=(
            design.engine_rating_kw if static.powertrain_load_dependent else None
        ),
        rated_bus_kw=(
            resolved.power.peak_bus_capability_kw
            if static.powertrain_load_dependent else None
        ),
    )
    aircraft = Aircraft(
        wing_area_m2=design.wing_area_m2,
        aspect_ratio=design.aspect_ratio,
        oswald_efficiency=resolved.wing.oswald_efficiency,
        cd0=resolved.wing.cd0,
        cl_max=static.cl_max,
        propeller_efficiency=static.propeller_efficiency,
        engine=engine,
        battery=battery,
        powertrain=powertrain,
        masses=masses,
    )
    parameters = ThermostatParameters(
        soc_low=resolved.soc_low,
        soc_high=resolved.soc_high,
        minimum_on_time_s=scenario.minimum_on_time_s,
        minimum_off_time_s=scenario.minimum_off_time_s,
        restart_fuel_kg=scenario.restart_fuel_kg,
        engine_on_power_kw=scenario.engine_on_power_kw,
        terminal_strategy=scenario.terminal_strategy,
        dwell_semantics=scenario.dwell_semantics,
    )
    initial_state = ThermostatState(
        engine_on=scenario.initial_engine_on,
        elapsed_in_state_s=scenario.initial_elapsed_in_state_s,
        restart_count=0,
        terminal_depletion=False,
    )
    mission = replace(
        scenario.mission,
        phases=tuple(replace(phase) for phase in scenario.mission.phases),
    )
    return ConstructedMissionInputs(
        aircraft,
        mission,
        parameters,
        initial_state,
        scenario.timestep_s,
        scenario.initial_soc,
        resolved_design_identity(resolved),
        scenario.identity,
    )


def _duration_summary(steps: Sequence[TimeStep], engine_on: bool) -> DurationSummary:
    durations: list[float] = []
    state: bool | None = None
    elapsed = 0.0
    for step in steps:
        current = not step.engine_shut_down
        if state is None:
            state = current
        if current != state:
            if state is engine_on:
                durations.append(elapsed)
            state = current
            elapsed = 0.0
        elapsed += step.dt_s
    if state is engine_on and elapsed > 0.0:
        durations.append(elapsed)
    return DurationSummary(
        len(durations), min(durations, default=0.0),
        fmean(durations) if durations else 0.0, max(durations, default=0.0),
    )


def _power_range(values: Sequence[float]) -> PowerRange:
    return PowerRange(min(values, default=0.0), max(values, default=0.0))


def _validate_finite_mission_result(result: MissionResult) -> None:
    result_values = {
        "endurance_s": result.endurance_s,
        "fuel_used_kg": result.fuel_used_kg,
        "fuel_remaining_kg": result.fuel_remaining_kg,
        "final_soc": result.final_soc,
        "min_soc": result.min_soc,
        "peak_bus_kw": result.peak_bus_kw,
        "peak_engine_kw": result.peak_engine_kw,
        "mean_system_efficiency": result.mean_system_efficiency,
        **{f"phase_durations_s[{name}]": value for name, value in result.phase_durations_s.items()},
    }
    for name, value in result_values.items():
        _finite(name, value)
    if result.log is None:
        return
    step_fields = (
        "time_s", "altitude_m", "speed_mps", "weight_n", "density_kg_m3",
        "lift_coefficient", "drag_n", "shaft_power_kw", "bus_demand_kw",
        "engine_shaft_kw", "fuel_flow_kg_s", "restart_fuel_kg",
        "battery_bus_kw", "soc", "fuel_remaining_kg", "system_efficiency",
        "dt_s", "bus_from_engine_kw", "battery_internal_kw",
        "battery_ohmic_loss_kw", "battery_stored_energy_change_kwh",
        "thrust_power_kw", "engine_thermal_loss_kw", "source_losses_kw",
        "demand_losses_kw", "propeller_losses_kw",
    )
    for index, step in enumerate(result.log):
        for name in step_fields:
            _finite(f"log[{index}].{name}", getattr(step, name))


@dataclass(frozen=True)
class _Audit:
    feasible: bool
    observed_loiter_s: float
    resources: MissionResources
    behavior: ControllerBehavior
    validity: MissionValidity
    constraints: tuple[DynamicConstraintRecord, ...]
    warnings: tuple[FitnessDiagnostic, ...]


def _audit_mission(
    inputs: ConstructedMissionInputs,
    result: MissionResult,
    scenario: FitnessScenario,
) -> _Audit:
    steps = () if result.log is None else result.log
    phase = result.phase_durations_s
    tolerance_s = scenario.phase_completion_tolerance_s
    seen = tuple(dict.fromkeys(step.phase for step in steps))
    expected = inputs.mission.phase_names
    loiter_s = float(phase.get("loiter", 0.0))
    takeoff_target = float(inputs.mission.phase_by_name("takeoff").duration_s)
    cruise_target = float(inputs.mission.phase_by_name("cruise").duration_s)
    landing_target = float(inputs.mission.phase_by_name("landing").duration_s)
    completed_takeoff = phase.get("takeoff", 0.0) >= takeoff_target - tolerance_s
    completed_climb = "cruise" in seen
    entered_loiter = loiter_s > tolerance_s and "loiter" in seen
    completed_cruise = (
        phase.get("cruise", 0.0) >= cruise_target - tolerance_s and entered_loiter
    )
    completed_loiter = "descent" in seen
    completed_descent = "landing" in seen
    completed_landing = (
        result.mission_complete
        and phase.get("landing", 0.0) >= landing_target - tolerance_s
        and bool(steps)
        and steps[-1].phase == "landing"
    )

    running_fuel = sum(step.fuel_flow_kg_s * step.dt_s for step in steps)
    restart_fuel = sum(step.restart_fuel_kg for step in steps)
    transition_restarts = sum(
        int(step.thermostat_transitioned and bool(step.requested_engine_on))
        for step in steps
    )
    loiter_restarts = sum(
        int(
            step.phase == "loiter"
            and step.thermostat_transitioned
            and bool(step.requested_engine_on)
        )
        for step in steps
    )
    final_state_count = (
        result.thermostat_final_state.restart_count
        if result.thermostat_final_state is not None
        else None
    )
    count_residual = (
        abs(final_state_count - transition_restarts)
        if final_state_count is not None
        else 1.0
    )
    fuel_count_residual = (
        abs(restart_fuel / scenario.restart_fuel_kg - transition_restarts)
        if scenario.restart_fuel_kg > 0.0
        else 0.0
    )
    hidden_restarts = max(float(count_residual), fuel_count_residual)
    duration_s = sum(step.dt_s for step in steps)
    loiter_steps = tuple(step for step in steps if step.phase == "loiter")
    requested = tuple(
        float(step.requested_engine_shaft_kw)
        for step in steps
        if step.requested_engine_on and step.requested_engine_shaft_kw is not None
    )
    delivered = tuple(step.engine_shaft_kw for step in steps if not step.engine_shut_down)
    charge_encounters = sum(
        step.battery_active_limit != "none" and step.battery_bus_kw < -1.0e-10
        for step in steps
    )
    discharge_encounters = sum(
        step.battery_active_limit != "none" and step.battery_bus_kw > 1.0e-10
        for step in steps
    )
    behavior = ControllerBehavior(
        restart_count=transition_restarts,
        restarts_per_loiter_hour=(
            loiter_restarts / (loiter_s / 3600.0) if loiter_s > 0.0 else 0.0
        ),
        overall_engine_off_fraction=(
            sum(step.dt_s for step in steps if step.engine_shut_down) / duration_s
            if duration_s > 0.0 else 0.0
        ),
        loiter_engine_off_fraction=(
            sum(step.dt_s for step in loiter_steps if step.engine_shut_down) / loiter_s
            if loiter_s > 0.0 else 0.0
        ),
        on_run_durations=_duration_summary(steps, True),
        off_run_durations=_duration_summary(steps, False),
        requested_engine_power_range=_power_range(requested),
        delivered_engine_power_range=_power_range(delivered),
        charge_limit_encounter_count=charge_encounters,
        discharge_limit_encounter_count=discharge_encounters,
    )

    initial_fuel = inputs.aircraft.masses.fuel_kg
    floor_energy = float(
        inputs.aircraft.battery.stored_energy_kwh(inputs.aircraft.battery.soc_min)
    )
    final_energy = float(inputs.aircraft.battery.stored_energy_kwh(result.final_soc))
    resources = MissionResources(
        initial_fuel_kg=initial_fuel,
        final_fuel_kg=result.fuel_remaining_kg,
        total_fuel_consumed_kg=result.fuel_used_kg,
        running_fuel_consumed_kg=running_fuel,
        restart_fuel_consumed_kg=restart_fuel,
        final_soc=result.final_soc,
        minimum_soc=result.min_soc,
        final_stored_battery_energy_kwh=final_energy,
        fuel_slack_above_reserve_kg=(
            result.fuel_remaining_kg - inputs.mission.fuel_reserve_kg
        ),
        usable_battery_energy_above_floor_kwh=final_energy - floor_energy,
    )

    bus_residual = (
        max(
            (
                abs(step.bus_from_engine_kw + step.battery_bus_kw - step.bus_demand_kw)
                for step in steps
            ),
            default=0.0,
        )
        if result.log is not None else None
    )
    fuel_residual = (
        result.fuel_used_kg - running_fuel - restart_fuel
        if result.log is not None else None
    )
    balance = mission_energy_balance(result) if result.log is not None else None
    controller_count = sum(int(not step.controller_feasible) for step in steps)
    plant_count = sum(int(not step.plant_feasible) for step in steps)
    controller_count += int("controller_infeasible" in result.failure_flags)
    plant_count += int("power_shortfall" in result.failure_flags)
    dwell_count = sum(int(step.thermostat_dwell_violation) for step in steps)
    dwell_count += int("hard_dwell_infeasible" in result.failure_flags)
    terminal_count = len(_TERMINAL_FAILURE_FLAGS.intersection(result.failure_flags))
    reserve_shortfall = max(inputs.mission.fuel_reserve_kg - result.fuel_remaining_kg, 0.0)
    soc_violation = max(inputs.aircraft.battery.soc_min - result.min_soc, 0.0)
    validity = MissionValidity(
        completed_takeoff, completed_climb, completed_cruise, entered_loiter,
        completed_loiter, completed_descent, completed_landing,
        result.termination_reason, reserve_shortfall, soc_violation,
        controller_count, plant_count, dwell_count, hidden_restarts,
        bus_residual, fuel_residual,
        None if balance is None else balance.residual_kwh,
        None if balance is None else balance.discrete_residual_fraction,
        None if balance is None else balance.battery_integration_residual_kwh,
        result.failure_flags,
    )

    epsilon = scenario.normalization_epsilon
    constraints = (
        DynamicConstraintRecord.at_least(
            "mission_log_present", float(result.log is not None), 1.0, 0.0,
            1.0, "1", "simulation/simulator.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "six_phase_order", float(seen == expected), 1.0, 0.0,
            1.0, "1", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "takeoff_completed", float(completed_takeoff), 1.0, 0.0,
            1.0, "1", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "climb_completed", float(completed_climb), 1.0, 0.0,
            1.0, "1", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "cruise_completed", float(completed_cruise), 1.0, 0.0,
            1.0, "1", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "loiter_entered", float(entered_loiter), 1.0, 0.0,
            1.0, "1", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "loiter_completed", float(completed_loiter), 1.0, 0.0,
            1.0, "1", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "descent_completed", float(completed_descent), 1.0, 0.0,
            1.0, "1", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "landing_completed", float(completed_landing), 1.0, 0.0,
            1.0, "1", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "mission_complete", float(result.mission_complete), 1.0, 0.0,
            1.0, "1", "simulation/simulator.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "fuel_reserve", result.fuel_remaining_kg, inputs.mission.fuel_reserve_kg,
            scenario.reserve_tolerance_kg, inputs.mission.fuel_reserve_kg,
            "kg", "simulation/mission.py", epsilon,
        ),
        DynamicConstraintRecord.at_least(
            "soc_floor", result.min_soc, inputs.aircraft.battery.soc_min,
            scenario.soc_floor_tolerance,
            1.0 - inputs.aircraft.battery.soc_min, "1",
            "models/battery.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "controller_feasible", float(controller_count), 0.0, 0.0,
            1.0, "count", "control/thermostat.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "plant_feasible", float(plant_count), 0.0, 0.0,
            1.0, "count", "simulation/simulator.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "hard_dwell", float(dwell_count), 0.0, 0.0,
            1.0, "count", "control/thermostat.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "restart_accounting", hidden_restarts, 0.0,
            scenario.restart_accounting_tolerance_count,
            1.0, "count", "simulation/simulator.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "terminal_failure_flags", float(terminal_count), 0.0, 0.0,
            1.0, "count", "simulation/simulator.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "candidate_mission_exception", 0.0, 0.0, 0.0,
            1.0, "count", "optimization/fitness.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "bus_power_ledger", 0.0 if bus_residual is None else bus_residual,
            scenario.power_balance_tolerance_kw, 0.0,
            max(result.peak_bus_kw, 1.0), "kW", "simulation/simulator.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "fuel_ledger", 0.0 if fuel_residual is None else abs(fuel_residual),
            scenario.fuel_ledger_tolerance_kg, 0.0,
            max(abs(initial_fuel), 1.0), "kg", "simulation/simulator.py", epsilon,
        ),
        DynamicConstraintRecord.at_most(
            "discrete_energy_ledger",
            1.0 if balance is None else balance.discrete_residual_fraction,
            scenario.discrete_energy_ledger_tolerance_fraction, 0.0,
            1.0, "1", "simulation/simulator.py", epsilon,
        ),
    )
    assert tuple(record.name for record in constraints) == DYNAMIC_CONSTRAINT_NAMES
    warnings: list[FitnessDiagnostic] = []
    if balance is not None and abs(balance.residual_kwh) > 1.0e-12:
        warnings.append(FitnessDiagnostic(
            "endpoint_energy_integration_bias",
            "Endpoint battery energy retains the documented explicit-Euler OCV residual; the discrete ledger is the hard gate.",
            "docs/conventions.md",
        ))
    feasible = all(record.satisfied for record in constraints)
    return _Audit(feasible, loiter_s, resources, behavior, validity, constraints, tuple(warnings))


def _static_diagnostics(
    result: StaticFeasibilityResult,
) -> tuple[FitnessDiagnostic, ...]:
    return tuple(
        FitnessDiagnostic(item.name, item.message, item.source)
        for item in result.warnings
    )


def _screened_result(
    chromosome: NormalizedChromosome,
    bounds: PlantThermostatDesignSpace,
    scenario: FitnessScenario,
    static: StaticFeasibilityResult,
) -> FitnessResult:
    resolved = static.resolved_design
    warnings = (*_static_diagnostics(static), FitnessDiagnostic(
        "static_screen_failed",
        "Mission evaluation was skipped because at least one hard static constraint failed.",
        "optimization/feasibility.py",
    ))
    return FitnessResult(
        schema_version=FITNESS_RESULT_SCHEMA_VERSION,
        chromosome_cache_key=chromosome.cache_key(bounds=bounds),
        design_space_id=bounds.identifier,
        static_scenario_id=scenario.static_scenario.identity,
        fitness_scenario_id=scenario.identity,
        evaluation_key=evaluation_identity(
            chromosome, bounds=bounds, scenario=scenario
        ),
        decoded_design=resolved.decoded,
        resolved_design_id=resolved_design_identity(resolved),
        resolved_design=resolved,
        static_feasible=False,
        static_normalized_total_violation=static.total_normalized_violation,
        static_constraints=(*static.hard_constraints, *static.advisory_constraints),
        run_mission_called=False,
        dynamically_feasible=False,
        objective_loiter_seconds=None,
        objective_hours=None,
        observed_loiter_seconds=None,
        total_mission_seconds=None,
        resources=None,
        controller_behavior=None,
        validity=None,
        dynamic_constraint_names=DYNAMIC_CONSTRAINT_NAMES,
        dynamic_constraints=(),
        total_normalized_dynamic_violation=0.0,
        combined_static_dynamic_violation=static.total_normalized_violation,
        warnings=warnings,
        failure_category="static_infeasible",
        failure_message="hard static constraints failed",
    )


def _candidate_exception_result(
    chromosome: NormalizedChromosome,
    bounds: PlantThermostatDesignSpace,
    scenario: FitnessScenario,
    static: StaticFeasibilityResult,
    error: CandidateMissionInfeasibleError,
) -> FitnessResult:
    resolved = static.resolved_design
    record = DynamicConstraintRecord(
        name="candidate_mission_exception",
        quantity=error.normalized_violation,
        required_or_allowed=0.0,
        tolerance=0.0,
        raw_violation=error.normalized_violation,
        normalization_scale=1.0,
        normalized_violation=error.normalized_violation,
        unit="1",
        satisfied=False,
        relationship="at_most",
        source="mission_runner_adapter",
    )
    warning = FitnessDiagnostic(error.category, str(error), "mission_runner_adapter")
    total = static.total_normalized_violation + error.normalized_violation
    return FitnessResult(
        schema_version=FITNESS_RESULT_SCHEMA_VERSION,
        chromosome_cache_key=chromosome.cache_key(bounds=bounds),
        design_space_id=bounds.identifier,
        static_scenario_id=scenario.static_scenario.identity,
        fitness_scenario_id=scenario.identity,
        evaluation_key=evaluation_identity(
            chromosome, bounds=bounds, scenario=scenario
        ),
        decoded_design=resolved.decoded,
        resolved_design_id=resolved_design_identity(resolved),
        resolved_design=resolved,
        static_feasible=True,
        static_normalized_total_violation=static.total_normalized_violation,
        static_constraints=(*static.hard_constraints, *static.advisory_constraints),
        run_mission_called=True,
        dynamically_feasible=False,
        objective_loiter_seconds=None,
        objective_hours=None,
        observed_loiter_seconds=None,
        total_mission_seconds=None,
        resources=None,
        controller_behavior=None,
        validity=None,
        dynamic_constraint_names=DYNAMIC_CONSTRAINT_NAMES,
        dynamic_constraints=(record,),
        total_normalized_dynamic_violation=error.normalized_violation,
        combined_static_dynamic_violation=total,
        warnings=(*_static_diagnostics(static), warning),
        failure_category=error.category,
        failure_message=str(error),
    )


MissionRunner = Callable[..., MissionResult]


def evaluate_fitness(
    chromosome: NormalizedChromosome,
    *,
    bounds: PlantThermostatDesignSpace,
    scenario: FitnessScenario,
    mission_runner: MissionRunner | None = None,
) -> FitnessResult:
    """Screen, construct, fly once, and audit one normalized chromosome."""
    if not isinstance(chromosome, NormalizedChromosome):
        raise ValueError("chromosome must be a NormalizedChromosome")
    if not isinstance(bounds, PlantThermostatDesignSpace):
        raise ValueError("bounds must be a PlantThermostatDesignSpace")
    if not isinstance(scenario, FitnessScenario):
        raise ValueError("scenario must be a FitnessScenario")
    decoded = decode_chromosome(chromosome, bounds=bounds)
    static = evaluate_static_feasibility(decoded, scenario=scenario.static_scenario)
    if not static.is_feasible:
        return _screened_result(chromosome, bounds, scenario, static)
    inputs = construct_mission_inputs(static.resolved_design, scenario=scenario)
    runner = run_mission if mission_runner is None else mission_runner
    try:
        mission_result = runner(
            inputs.aircraft,
            inputs.mission,
            thermostat_parameters=inputs.thermostat_parameters,
            initial_thermostat_state=inputs.initial_thermostat_state,
            dt_s=inputs.timestep_s,
            initial_soc=inputs.initial_soc,
            record_log=True,
        )
    except CandidateMissionInfeasibleError as error:
        return _candidate_exception_result(chromosome, bounds, scenario, static, error)
    if not isinstance(mission_result, MissionResult):
        raise TypeError("mission_runner must return MissionResult")
    _validate_finite_mission_result(mission_result)
    audit = _audit_mission(inputs, mission_result, scenario)
    dynamic_total = math.fsum(
        record.normalized_violation for record in audit.constraints
    )
    combined = static.total_normalized_violation + dynamic_total
    objective = audit.observed_loiter_s if audit.feasible else None
    resolved = static.resolved_design
    return FitnessResult(
        schema_version=FITNESS_RESULT_SCHEMA_VERSION,
        chromosome_cache_key=chromosome.cache_key(bounds=bounds),
        design_space_id=bounds.identifier,
        static_scenario_id=scenario.static_scenario.identity,
        fitness_scenario_id=scenario.identity,
        evaluation_key=evaluation_identity(chromosome, bounds=bounds, scenario=scenario),
        decoded_design=decoded,
        resolved_design_id=inputs.resolved_design_id,
        resolved_design=resolved,
        static_feasible=True,
        static_normalized_total_violation=static.total_normalized_violation,
        static_constraints=(*static.hard_constraints, *static.advisory_constraints),
        run_mission_called=True,
        dynamically_feasible=audit.feasible,
        objective_loiter_seconds=objective,
        objective_hours=None if objective is None else objective / 3600.0,
        observed_loiter_seconds=audit.observed_loiter_s,
        total_mission_seconds=mission_result.endurance_s,
        resources=audit.resources,
        controller_behavior=audit.behavior,
        validity=audit.validity,
        dynamic_constraint_names=tuple(record.name for record in audit.constraints),
        dynamic_constraints=audit.constraints,
        total_normalized_dynamic_violation=dynamic_total,
        combined_static_dynamic_violation=combined,
        warnings=(*_static_diagnostics(static), *audit.warnings),
        failure_category=None if audit.feasible else "dynamic_infeasible",
        failure_message=None if audit.feasible else mission_result.termination_reason,
    )
