"""Deterministic static resolution and feasibility for GA plant candidates.

This layer performs only algebraic sizing and point-performance checks. It does
not import the mission simulator, evaluate endurance, or assign scalar fitness.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.analysis.constraint_diagram import Airframe, ConstraintCase, power_loading_required
from src.models.aerodynamics import (
    induced_drag_factor,
    parasite_drag_from_wetted_area,
    shaft_power_required,
    stall_speed,
)
from src.models.atmosphere import atmosphere
from src.models.battery import (
    CHARGE_C_RATE,
    DISCHARGE_C_RATE,
    R_REF_CAPACITY_KWH,
    R_REF_OHM,
    SCALE_RESISTANCE,
    SOC_MIN,
    V_MAX_V,
    V_MIN_V,
    BatteryMode,
    BatteryPack,
)
from src.models.engine import LAPSE_EXPONENT, Turboshaft
from src.models.mass import (
    CABLING_COOLING_FRACTION,
    CELL_ENERGY_DENSITY_WH_KG,
    COMPOSITE_FACTOR,
    CRUISE_DYNAMIC_PRESSURE_PA,
    ENGINE_KW_PER_KG,
    FIXED_GROUP_KG,
    FUEL_DENSITY_KG_PER_L,
    FUEL_IN_WING_KG,
    FUEL_SYSTEM_FRACTION,
    GENERATOR_KW_PER_KG,
    INVERTER_KW_PER_KG,
    LIMIT_LOAD_FACTOR,
    MIN_USABLE_FUEL_KG,
    MOTOR_KW_PER_KG,
    MTOW_KG,
    PACK_FACTOR,
    PAYLOAD_KG,
    RECTIFIER_KW_PER_KG,
    SWEEP_RAD,
    TANK_VOLUME_FRACTION,
    TAPER_RATIO,
    THICKNESS_TO_CHORD,
    ULTIMATE_LOAD_SAFETY_FACTOR,
    build_mass_budget,
    fuel_volume_check,
)
from src.models.powertrain import (
    ETA_CABLING,
    ETA_GENERATOR,
    ETA_INVERTER,
    ETA_MOTOR,
    ETA_RECTIFIER,
    NOLOAD_LOSS_FRACTION,
    SeriesPowertrain,
)
from src.optimization.chromosome import DecodedPlantThermostatDesign
from src.simulation.mission import ps1_mission

__all__ = [
    "ConstraintRecord",
    "ResolvedFuel",
    "ResolvedMasses",
    "ResolvedPlantDesign",
    "ResolvedPowerCapability",
    "ResolvedWingGeometry",
    "StaticDiagnostic",
    "StaticFeasibilityResult",
    "StaticFeasibilityScenario",
    "evaluate_static_feasibility",
]

_SCENARIO_SCHEMA_VERSION = 1


def _finite(name: str, value: Any) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


@dataclass(frozen=True)
class StaticFeasibilityScenario:
    """All non-gene assumptions used by one static screen."""

    mtow_kg: float
    payload_kg: float
    fixed_group_kg: float
    cruise_altitude_m: float
    cruise_speed_mps: float
    climb_speed_mps: float
    climb_rate_mps: float
    takeoff_speed_mps: float
    landing_speed_mps: float
    timestep_s: float
    cl_max: float
    propeller_efficiency: float
    stall_speed_margin: float
    sizing_margin: float
    service_ceiling_altitude_m: float
    service_ceiling_climb_rate_mps: float
    service_ceiling_is_hard: bool
    wetted_area_policy: str
    equivalent_skin_friction: float
    thickness_to_chord: float
    reference_wing_area_m2: float
    reference_cd0: float
    fixed_nonwing_wetted_area_m2: float
    oswald_policy: str
    oswald_efficiency: float
    oswald_is_fixed: bool
    battery_chemistry: str
    battery_mode: str
    battery_soc_floor: float
    battery_discharge_c_rate: float
    battery_charge_c_rate: float
    battery_v_min_v: float
    battery_v_max_v: float
    battery_r_ref_ohm: float
    battery_r_ref_capacity_kwh: float
    battery_scale_resistance: bool
    thermostat_soc_low_upper: float
    thermostat_soc_high_upper: float
    thermostat_minimum_gap: float
    engine_lapse_exponent: float
    eta_generator: float
    eta_rectifier: float
    eta_cabling: float
    eta_inverter: float
    eta_motor: float
    powertrain_load_dependent: bool
    powertrain_noload_loss_fraction: float
    engine_allow_shutdown: bool
    engine_kw_per_kg: float
    generator_kw_per_kg: float
    rectifier_kw_per_kg: float
    inverter_kw_per_kg: float
    motor_kw_per_kg: float
    cabling_cooling_fraction: float
    cell_energy_density_wh_kg: float
    pack_factor: float
    fuel_system_fraction: float
    fuel_density_kg_per_l: float
    tank_volume_fraction: float
    minimum_usable_fuel_kg: float
    wing_mass_cruise_dynamic_pressure_pa: float
    wing_mass_nominal_fuel_kg: float
    wing_mass_sweep_rad: float
    wing_mass_taper_ratio: float
    wing_mass_limit_load_factor: float
    wing_mass_ultimate_factor: float
    wing_mass_construction_factor: float
    mass_closure_tolerance_kg: float
    normalization_epsilon: float

    def __post_init__(self) -> None:
        string_fields = (
            "wetted_area_policy", "oswald_policy", "battery_chemistry",
            "battery_mode",
        )
        boolean_fields = (
            "service_ceiling_is_hard", "oswald_is_fixed",
            "battery_scale_resistance", "powertrain_load_dependent",
            "engine_allow_shutdown",
        )
        for name, value in asdict(self).items():
            if name in string_fields:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{name} must be a non-empty string")
            elif name in boolean_fields:
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be boolean")
            else:
                object.__setattr__(self, name, _finite(name, value))
        positive = (
            "mtow_kg", "payload_kg", "fixed_group_kg", "cruise_speed_mps",
            "climb_speed_mps", "takeoff_speed_mps", "landing_speed_mps",
            "timestep_s", "cl_max", "propeller_efficiency", "stall_speed_margin",
            "sizing_margin", "equivalent_skin_friction", "thickness_to_chord",
            "reference_wing_area_m2", "reference_cd0",
            "fixed_nonwing_wetted_area_m2", "oswald_efficiency",
            "battery_discharge_c_rate", "battery_charge_c_rate", "battery_v_min_v",
            "battery_v_max_v", "battery_r_ref_ohm", "battery_r_ref_capacity_kwh",
            "engine_kw_per_kg", "generator_kw_per_kg", "rectifier_kw_per_kg",
            "inverter_kw_per_kg", "motor_kw_per_kg", "cell_energy_density_wh_kg",
            "pack_factor", "fuel_density_kg_per_l", "tank_volume_fraction",
            "minimum_usable_fuel_kg", "mass_closure_tolerance_kg",
            "normalization_epsilon", "climb_rate_mps",
            "service_ceiling_altitude_m", "service_ceiling_climb_rate_mps",
            "wing_mass_cruise_dynamic_pressure_pa",
            "wing_mass_nominal_fuel_kg", "wing_mass_taper_ratio",
            "wing_mass_limit_load_factor", "wing_mass_ultimate_factor",
            "wing_mass_construction_factor",
        )
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.sizing_margin < 1.0:
            raise ValueError("sizing_margin must be at least one")
        if not (
            0.0 <= self.battery_soc_floor
            < self.thermostat_soc_low_upper
            < self.thermostat_soc_high_upper
            <= 1.0
        ):
            raise ValueError("thermostat bounds must be ordered within [0, 1]")
        if not 0.0 < self.thermostat_minimum_gap < 1.0:
            raise ValueError("thermostat_minimum_gap must lie in (0, 1)")
        if self.battery_soc_floor + self.thermostat_minimum_gap > self.thermostat_soc_high_upper:
            raise ValueError("thermostat bounds cannot accommodate the minimum gap")
        for name in (
            "propeller_efficiency", "oswald_efficiency", "eta_generator",
            "eta_rectifier", "eta_cabling", "eta_inverter", "eta_motor",
            "pack_factor", "tank_volume_fraction",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        for name in (
            "cabling_cooling_fraction", "fuel_system_fraction",
            "powertrain_noload_loss_fraction",
        ):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.battery_v_min_v >= self.battery_v_max_v:
            raise ValueError("battery_v_min_v must be below battery_v_max_v")
        if self.cruise_altitude_m < 0.0 or self.engine_lapse_exponent < 0.0:
            raise ValueError("altitude and engine_lapse_exponent must be non-negative")
        if not 0.0 <= self.wing_mass_sweep_rad < math.pi / 2.0:
            raise ValueError("wing_mass_sweep_rad must lie in [0, pi/2)")
        if self.battery_mode != BatteryMode.LEGACY.value:
            raise ValueError("the static scenario currently supports battery_mode='legacy'")
        if self.powertrain_load_dependent:
            raise ValueError("the static constraint diagram requires constant stage efficiencies")

    @classmethod
    def nominal(cls) -> StaticFeasibilityScenario:
        """Build the documented 3 km fixed-e GA screening scenario."""
        mission = ps1_mission()
        reference_wing_area = 7.59175537062125
        reference_cd0 = 0.028
        c_fe = 0.0055
        thickness_to_chord = THICKNESS_TO_CHORD
        reference_total_wet = reference_cd0 * reference_wing_area / c_fe
        reference_wing_wet = 2.0 * reference_wing_area * (
            1.0 + 0.25 * thickness_to_chord
        )
        return cls(
            mtow_kg=MTOW_KG,
            payload_kg=PAYLOAD_KG,
            fixed_group_kg=FIXED_GROUP_KG,
            cruise_altitude_m=mission.phase_by_name("cruise").target_altitude_m,
            cruise_speed_mps=float(mission.phase_by_name("cruise").speed_mps),
            climb_speed_mps=float(mission.phase_by_name("climb").speed_mps),
            climb_rate_mps=float(mission.phase_by_name("climb").climb_rate_mps),
            takeoff_speed_mps=float(mission.phase_by_name("takeoff").speed_mps),
            landing_speed_mps=float(mission.phase_by_name("landing").speed_mps),
            timestep_s=60.0,
            cl_max=1.5,
            propeller_efficiency=0.85,
            stall_speed_margin=1.2,
            sizing_margin=1.10,
            service_ceiling_altitude_m=10_000.0,
            service_ceiling_climb_rate_mps=0.5,
            service_ceiling_is_hard=False,
            wetted_area_policy="reference_calibrated_wetted_area",
            equivalent_skin_friction=c_fe,
            thickness_to_chord=thickness_to_chord,
            reference_wing_area_m2=reference_wing_area,
            reference_cd0=reference_cd0,
            fixed_nonwing_wetted_area_m2=reference_total_wet - reference_wing_wet,
            oswald_policy="fixed_reference",
            oswald_efficiency=0.78,
            oswald_is_fixed=True,
            battery_chemistry="lithium_ion_nmc",
            battery_mode=BatteryMode.LEGACY.value,
            battery_soc_floor=SOC_MIN,
            battery_discharge_c_rate=DISCHARGE_C_RATE,
            battery_charge_c_rate=CHARGE_C_RATE,
            battery_v_min_v=V_MIN_V,
            battery_v_max_v=V_MAX_V,
            battery_r_ref_ohm=R_REF_OHM,
            battery_r_ref_capacity_kwh=R_REF_CAPACITY_KWH,
            battery_scale_resistance=SCALE_RESISTANCE,
            thermostat_soc_low_upper=0.60,
            thermostat_soc_high_upper=0.95,
            thermostat_minimum_gap=0.05,
            engine_lapse_exponent=LAPSE_EXPONENT,
            eta_generator=ETA_GENERATOR,
            eta_rectifier=ETA_RECTIFIER,
            eta_cabling=ETA_CABLING,
            eta_inverter=ETA_INVERTER,
            eta_motor=ETA_MOTOR,
            powertrain_load_dependent=False,
            powertrain_noload_loss_fraction=NOLOAD_LOSS_FRACTION,
            engine_allow_shutdown=True,
            engine_kw_per_kg=ENGINE_KW_PER_KG,
            generator_kw_per_kg=GENERATOR_KW_PER_KG,
            rectifier_kw_per_kg=RECTIFIER_KW_PER_KG,
            inverter_kw_per_kg=INVERTER_KW_PER_KG,
            motor_kw_per_kg=MOTOR_KW_PER_KG,
            cabling_cooling_fraction=CABLING_COOLING_FRACTION,
            cell_energy_density_wh_kg=CELL_ENERGY_DENSITY_WH_KG,
            pack_factor=PACK_FACTOR,
            fuel_system_fraction=FUEL_SYSTEM_FRACTION,
            fuel_density_kg_per_l=FUEL_DENSITY_KG_PER_L,
            tank_volume_fraction=TANK_VOLUME_FRACTION,
            minimum_usable_fuel_kg=MIN_USABLE_FUEL_KG,
            wing_mass_cruise_dynamic_pressure_pa=CRUISE_DYNAMIC_PRESSURE_PA,
            wing_mass_nominal_fuel_kg=FUEL_IN_WING_KG,
            wing_mass_sweep_rad=SWEEP_RAD,
            wing_mass_taper_ratio=TAPER_RATIO,
            wing_mass_limit_load_factor=LIMIT_LOAD_FACTOR,
            wing_mass_ultimate_factor=ULTIMATE_LOAD_SAFETY_FACTOR,
            wing_mass_construction_factor=COMPOSITE_FACTOR,
            mass_closure_tolerance_kg=1.0e-9,
            normalization_epsilon=1.0e-12,
        )

    @property
    def identity(self) -> str:
        payload = json.dumps(
            _canonical(asdict(self)), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return f"static-feasibility-scenario-v{_SCENARIO_SCHEMA_VERSION}:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCENARIO_SCHEMA_VERSION,
            "scenario_id": self.identity,
            **asdict(self),
        }


@dataclass(frozen=True)
class ConstraintRecord:
    """One signed physical margin and its dimensionless violation."""

    name: str
    quantity: float
    required_or_allowed: float
    margin: float
    unit: str
    normalized_violation: float
    satisfied: bool
    relationship: Literal["at_least", "at_most"]
    source: str

    @classmethod
    def at_least(
        cls, name: str, available: float, required: float, unit: str, source: str,
        epsilon: float,
    ) -> ConstraintRecord:
        margin = available - required
        raw_violation = max(0.0, -margin) / max(abs(required), epsilon)
        satisfied = raw_violation <= epsilon
        if satisfied and margin < 0.0:
            margin = 0.0
        violation = 0.0 if satisfied else raw_violation
        return cls(name, available, required, margin, unit, violation, satisfied,
                   "at_least", source)

    @classmethod
    def at_most(
        cls, name: str, actual: float, allowed: float, unit: str, source: str,
        epsilon: float,
    ) -> ConstraintRecord:
        margin = allowed - actual
        raw_violation = max(0.0, -margin) / max(abs(allowed), epsilon)
        satisfied = raw_violation <= epsilon
        if satisfied and margin < 0.0:
            margin = 0.0
        violation = 0.0 if satisfied else raw_violation
        return cls(name, actual, allowed, margin, unit, violation, satisfied,
                   "at_most", source)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StaticDiagnostic:
    name: str
    message: str
    source: str


@dataclass(frozen=True)
class ResolvedWingGeometry:
    wing_area_m2: float
    aspect_ratio: float
    span_m: float
    reference_chord_m: float
    wetted_area_policy: str
    equivalent_skin_friction: float
    thickness_to_chord: float
    reference_wing_area_m2: float
    reference_cd0: float
    fixed_nonwing_wetted_area_m2: float
    wing_wetted_area_m2: float
    total_wetted_area_m2: float
    cd0: float
    oswald_policy: str
    oswald_efficiency: float
    oswald_is_fixed: bool
    induced_drag_factor: float
    wing_loading_pa: float
    stall_speed_mps: float
    maximum_stall_speed_mps: float


@dataclass(frozen=True)
class ResolvedMasses:
    fixed_group_kg: float
    payload_kg: float
    wing_kg: float
    engine_kg: float
    generator_kg: float
    rectifier_kg: float
    inverter_kg: float
    motor_kg: float
    cabling_cooling_kg: float
    battery_kg: float
    fuel_system_kg: float
    electrical_total_kg: float
    propulsion_total_kg: float
    dry_kg: float
    fuel_kg: float
    total_kg: float
    mtow_closure_residual_kg: float


@dataclass(frozen=True)
class ResolvedFuel:
    initial_usable_fuel_kg: float
    required_fuel_volume_l: float
    available_usable_tank_volume_l: float
    fuel_volume_margin_l: float
    minimum_fuel_margin_kg: float


@dataclass(frozen=True)
class ResolvedPowerCapability:
    engine_rating_sea_level_kw: float
    engine_lapsed_3km_shaft_kw: float
    engine_bus_sea_level_kw: float
    engine_bus_3km_kw: float
    battery_discharge_rate_limit_kw: float
    battery_charge_rate_limit_kw: float
    battery_sustainable_discharge_kw: float
    peak_bus_capability_kw: float
    peak_propeller_shaft_capability_kw: float
    cruise_required_engine_rating_kw: float
    cruise_required_with_margin_kw: float
    cruise_rating_margin_kw: float
    climb_required_engine_rating_kw: float
    climb_required_with_margin_kw: float
    climb_rating_margin_kw: float
    takeoff_required_engine_rating_kw: float
    takeoff_required_with_margin_kw: float
    takeoff_rating_margin_kw: float
    battery_peak_required_bus_kw: float
    battery_peak_margin_kw: float
    service_ceiling_required_engine_rating_kw: float
    service_ceiling_required_with_margin_kw: float
    service_ceiling_rating_margin_kw: float


@dataclass(frozen=True)
class ResolvedPlantDesign:
    decoded: DecodedPlantThermostatDesign
    scenario_id: str
    wing: ResolvedWingGeometry
    masses: ResolvedMasses
    fuel: ResolvedFuel
    power: ResolvedPowerCapability
    soc_low: float
    soc_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StaticFeasibilityResult:
    resolved_design: ResolvedPlantDesign
    hard_constraints: tuple[ConstraintRecord, ...]
    advisory_constraints: tuple[ConstraintRecord, ...]
    warnings: tuple[StaticDiagnostic, ...]
    is_feasible: bool
    total_normalized_violation: float
    maximum_normalized_violation: float
    violated_hard_constraint_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _point_requirement(
    *, name: str, altitude_m: float, speed_mps: float | None,
    climb_rate_mps: float, battery_boost_kw: float, weight_n: float,
    wing_loading_pa: float, airframe: Airframe, powertrain: SeriesPowertrain,
    lapse_exponent: float,
) -> float:
    case = ConstraintCase(
        name=name,
        altitude_m=altitude_m,
        speed_mps=speed_mps,
        climb_rate_mps=climb_rate_mps,
        battery_boost_kw=battery_boost_kw,
        weight_n=weight_n,
    )
    loading = power_loading_required(
        wing_loading_pa, case, airframe, powertrain, lapse_exponent
    )
    return float(loading) * weight_n / 1000.0


def _required_battery_boost(
    design: DecodedPlantThermostatDesign,
    scenario: StaticFeasibilityScenario,
    powertrain: SeriesPowertrain,
    engine: Turboshaft,
    cd0: float,
    *, altitude_m: float, speed_mps: float, climb_rate_mps: float,
) -> float:
    state = atmosphere(altitude_m)
    shaft_w, _ = shaft_power_required(
        scenario.mtow_kg * 9.80665,
        float(state.density_kg_m3),
        speed_mps,
        design.wing_area_m2,
        cd0,
        design.aspect_ratio,
        scenario.oswald_efficiency,
        scenario.propeller_efficiency,
        climb_rate_mps,
    )
    bus_demand = float(powertrain.bus_power_required(float(shaft_w) / 1000.0))
    engine_available = engine.max_power_kw(float(state.density_ratio))
    engine_bus = float(powertrain.bus_power_from_engine(engine_available))
    return max(0.0, bus_demand - engine_bus)


def evaluate_static_feasibility(
    design: DecodedPlantThermostatDesign,
    *, scenario: StaticFeasibilityScenario,
) -> StaticFeasibilityResult:
    """Resolve one candidate and return Deb-compatible static violations."""
    if not isinstance(design, DecodedPlantThermostatDesign):
        raise ValueError("design must be a DecodedPlantThermostatDesign")
    if not isinstance(scenario, StaticFeasibilityScenario):
        raise ValueError("scenario must be a StaticFeasibilityScenario")

    weight_n = scenario.mtow_kg * 9.80665
    span_m = math.sqrt(design.aspect_ratio * design.wing_area_m2)
    reference_chord_m = design.wing_area_m2 / span_m
    wing_wet = 2.0 * design.wing_area_m2 * (1.0 + 0.25 * scenario.thickness_to_chord)
    total_wet = scenario.fixed_nonwing_wetted_area_m2 + wing_wet
    cd0 = float(parasite_drag_from_wetted_area(
        total_wet, design.wing_area_m2, scenario.equivalent_skin_friction
    ))
    induced_factor = float(induced_drag_factor(
        design.aspect_ratio, scenario.oswald_efficiency
    ))
    rho_sl = float(atmosphere(0.0).density_kg_m3)
    stall = float(stall_speed(
        weight_n, rho_sl, design.wing_area_m2, scenario.cl_max
    ))
    maximum_stall = scenario.landing_speed_mps / scenario.stall_speed_margin
    wing_loading = weight_n / design.wing_area_m2

    engine = Turboshaft(
        design.engine_rating_kw,
        lapse_exponent=scenario.engine_lapse_exponent,
        allow_shutdown=scenario.engine_allow_shutdown,
    )
    battery = BatteryPack(
        design.battery_capacity_kwh,
        v_min_v=scenario.battery_v_min_v,
        v_max_v=scenario.battery_v_max_v,
        r_ref_ohm=scenario.battery_r_ref_ohm,
        r_ref_capacity_kwh=scenario.battery_r_ref_capacity_kwh,
        scale_resistance=scenario.battery_scale_resistance,
        discharge_c_rate=scenario.battery_discharge_c_rate,
        charge_c_rate=scenario.battery_charge_c_rate,
        soc_min=scenario.battery_soc_floor,
        mode=scenario.battery_mode,
    )
    discharge = float(battery.available_discharge_kw(1.0, dt_s=scenario.timestep_s))
    charge_rate = battery.max_charge_kw
    rated_bus_kw = (
        design.engine_rating_kw * scenario.eta_generator * scenario.eta_rectifier
        + discharge
    )
    powertrain = SeriesPowertrain(
        eta_generator=scenario.eta_generator,
        eta_rectifier=scenario.eta_rectifier,
        eta_cabling=scenario.eta_cabling,
        eta_inverter=scenario.eta_inverter,
        eta_motor=scenario.eta_motor,
        load_dependent=scenario.powertrain_load_dependent,
        noload_loss_fraction=scenario.powertrain_noload_loss_fraction,
        rated_engine_kw=design.engine_rating_kw,
        rated_bus_kw=rated_bus_kw,
    )
    engine_bus_sl = float(powertrain.bus_power_from_engine(design.engine_rating_kw))
    peak_bus = engine_bus_sl + discharge
    peak_shaft = float(powertrain.shaft_power_from_bus(peak_bus))

    masses = build_mass_budget(
        design.engine_rating_kw,
        design.battery_capacity_kwh,
        peak_bus,
        design.wing_area_m2,
        design.aspect_ratio,
        mtow_kg=scenario.mtow_kg,
        payload_kg=scenario.payload_kg,
        fixed_kg=scenario.fixed_group_kg,
        engine_kw_per_kg=scenario.engine_kw_per_kg,
        generator_kw_per_kg=scenario.generator_kw_per_kg,
        rectifier_kw_per_kg=scenario.rectifier_kw_per_kg,
        inverter_kw_per_kg=scenario.inverter_kw_per_kg,
        motor_kw_per_kg=scenario.motor_kw_per_kg,
        cabling_cooling_fraction=scenario.cabling_cooling_fraction,
        cell_wh_kg=scenario.cell_energy_density_wh_kg,
        pack_factor=scenario.pack_factor,
        fuel_system_fraction=scenario.fuel_system_fraction,
        q_pa=scenario.wing_mass_cruise_dynamic_pressure_pa,
        fuel_in_wing_kg=scenario.wing_mass_nominal_fuel_kg,
        sweep_rad=scenario.wing_mass_sweep_rad,
        taper=scenario.wing_mass_taper_ratio,
        tc=scenario.thickness_to_chord,
        n_z=scenario.wing_mass_limit_load_factor,
        ultimate_factor=scenario.wing_mass_ultimate_factor,
        construction_factor=scenario.wing_mass_construction_factor,
    )
    required_fuel_l, available_tank_l, _ = fuel_volume_check(
        masses.fuel_kg,
        design.wing_area_m2,
        design.aspect_ratio,
        tc=scenario.thickness_to_chord,
        tank_volume_fraction=scenario.tank_volume_fraction,
        fuel_density_kg_per_l=scenario.fuel_density_kg_per_l,
    )

    airframe = Airframe(
        design.aspect_ratio,
        scenario.oswald_efficiency,
        cd0,
        scenario.cl_max,
        scenario.propeller_efficiency,
    )
    cruise_raw = _point_requirement(
        name="cruise_3km", altitude_m=scenario.cruise_altitude_m,
        speed_mps=scenario.cruise_speed_mps, climb_rate_mps=0.0,
        battery_boost_kw=0.0, weight_n=weight_n, wing_loading_pa=wing_loading,
        airframe=airframe, powertrain=powertrain,
        lapse_exponent=scenario.engine_lapse_exponent,
    )
    climb_raw = _point_requirement(
        name="climb_3km", altitude_m=scenario.cruise_altitude_m,
        speed_mps=scenario.climb_speed_mps,
        climb_rate_mps=scenario.climb_rate_mps,
        battery_boost_kw=discharge, weight_n=weight_n,
        wing_loading_pa=wing_loading, airframe=airframe, powertrain=powertrain,
        lapse_exponent=scenario.engine_lapse_exponent,
    )
    takeoff_raw = _point_requirement(
        name="takeoff_power", altitude_m=0.0,
        speed_mps=scenario.takeoff_speed_mps, climb_rate_mps=0.0,
        battery_boost_kw=discharge, weight_n=weight_n,
        wing_loading_pa=wing_loading, airframe=airframe, powertrain=powertrain,
        lapse_exponent=scenario.engine_lapse_exponent,
    )
    ceiling_raw = _point_requirement(
        name="service_ceiling_10km", altitude_m=scenario.service_ceiling_altitude_m,
        speed_mps=None, climb_rate_mps=scenario.service_ceiling_climb_rate_mps,
        battery_boost_kw=0.0, weight_n=weight_n,
        wing_loading_pa=wing_loading, airframe=airframe, powertrain=powertrain,
        lapse_exponent=scenario.engine_lapse_exponent,
    )
    climb_boost = _required_battery_boost(
        design, scenario, powertrain, engine, cd0,
        altitude_m=scenario.cruise_altitude_m,
        speed_mps=scenario.climb_speed_mps,
        climb_rate_mps=scenario.climb_rate_mps,
    )
    takeoff_boost = _required_battery_boost(
        design, scenario, powertrain, engine, cd0,
        altitude_m=0.0, speed_mps=scenario.takeoff_speed_mps, climb_rate_mps=0.0,
    )
    required_boost = max(climb_boost, takeoff_boost)
    state_3km = atmosphere(scenario.cruise_altitude_m)
    lapsed_3km = engine.max_power_kw(float(state_3km.density_ratio))

    wing = ResolvedWingGeometry(
        design.wing_area_m2, design.aspect_ratio, span_m, reference_chord_m,
        scenario.wetted_area_policy, scenario.equivalent_skin_friction,
        scenario.thickness_to_chord, scenario.reference_wing_area_m2,
        scenario.reference_cd0, scenario.fixed_nonwing_wetted_area_m2,
        wing_wet, total_wet, cd0, scenario.oswald_policy,
        scenario.oswald_efficiency, scenario.oswald_is_fixed, induced_factor,
        wing_loading, stall, maximum_stall,
    )
    resolved_masses = ResolvedMasses(
        masses.fixed_kg, masses.payload_kg, masses.wing_kg, masses.engine_kg,
        masses.generator_kg, masses.rectifier_kg, masses.inverter_kg,
        masses.motor_kg, masses.cabling_cooling_kg, masses.battery_kg,
        masses.fuel_system_kg, masses.electrical_total_kg,
        masses.propulsion_total_kg, masses.dry_kg, masses.fuel_kg,
        masses.total_kg, masses.total_kg - scenario.mtow_kg,
    )
    fuel = ResolvedFuel(
        masses.fuel_kg, required_fuel_l, available_tank_l,
        available_tank_l - required_fuel_l,
        masses.fuel_kg - scenario.minimum_usable_fuel_kg,
    )
    power = ResolvedPowerCapability(
        design.engine_rating_kw, lapsed_3km, engine_bus_sl,
        float(powertrain.bus_power_from_engine(lapsed_3km)),
        battery.max_discharge_kw, charge_rate, discharge, peak_bus, peak_shaft,
        cruise_raw, scenario.sizing_margin * cruise_raw,
        design.engine_rating_kw - scenario.sizing_margin * cruise_raw,
        climb_raw, scenario.sizing_margin * climb_raw,
        design.engine_rating_kw - scenario.sizing_margin * climb_raw,
        takeoff_raw, scenario.sizing_margin * takeoff_raw,
        design.engine_rating_kw - scenario.sizing_margin * takeoff_raw,
        required_boost, discharge - required_boost,
        ceiling_raw, scenario.sizing_margin * ceiling_raw,
        design.engine_rating_kw - scenario.sizing_margin * ceiling_raw,
    )
    resolved = ResolvedPlantDesign(
        design, scenario.identity, wing, resolved_masses, fuel, power,
        design.soc_low, design.soc_high,
    )

    epsilon = scenario.normalization_epsilon
    component_min = min(
        masses.wing_kg, masses.engine_kg, masses.generator_kg,
        masses.rectifier_kg, masses.inverter_kg, masses.motor_kg,
        masses.cabling_cooling_kg, masses.battery_kg, masses.fuel_system_kg,
    )
    hard = [
        ConstraintRecord.at_least("positive_wing_area", design.wing_area_m2,
                                  0.0, "m^2", "chromosome.py", epsilon),
        ConstraintRecord.at_least("positive_aspect_ratio", design.aspect_ratio,
                                  0.0, "1", "chromosome.py", epsilon),
        ConstraintRecord.at_least("positive_engine_rating",
                                  design.engine_rating_kw, 0.0, "kW",
                                  "chromosome.py", epsilon),
        ConstraintRecord.at_least("positive_battery_capacity",
                                  design.battery_capacity_kwh, 0.0, "kWh",
                                  "chromosome.py", epsilon),
        ConstraintRecord.at_least("nonnegative_component_masses", component_min, 0.0,
                                  "kg", "models/mass.py", epsilon),
        ConstraintRecord.at_least("thermostat_soc_floor", design.soc_low,
                                  scenario.battery_soc_floor, "1", "chromosome.py", epsilon),
        ConstraintRecord.at_most("thermostat_soc_low_upper", design.soc_low,
                                 scenario.thermostat_soc_low_upper,
                                 "1", "chromosome.py", epsilon),
        ConstraintRecord.at_most("thermostat_soc_high_upper", design.soc_high,
                                 scenario.thermostat_soc_high_upper,
                                 "1", "chromosome.py", epsilon),
        ConstraintRecord.at_least("thermostat_minimum_gap",
                                  design.soc_high - design.soc_low,
                                  scenario.thermostat_minimum_gap, "1",
                                  "chromosome.py", epsilon),
        ConstraintRecord.at_most("mtow_mass_closure",
                                 abs(masses.total_kg - scenario.mtow_kg),
                                 scenario.mass_closure_tolerance_kg, "kg",
                                 "models/mass.py", epsilon),
        ConstraintRecord.at_least("minimum_usable_fuel", masses.fuel_kg,
                                  scenario.minimum_usable_fuel_kg, "kg",
                                  "models/mass.py M-09", epsilon),
        ConstraintRecord.at_most("fuel_tank_volume", required_fuel_l,
                                 available_tank_l, "L", "models/mass.py M-06", epsilon),
        ConstraintRecord.at_most("stall_speed", stall, maximum_stall, "m/s",
                                 "constraint_diagram.py CD-01", epsilon),
        ConstraintRecord.at_least("cruise_engine_rating_with_margin",
                                  design.engine_rating_kw,
                                  scenario.sizing_margin * cruise_raw, "kW shaft SL",
                                  "constraint_diagram.py CD-04", epsilon),
        ConstraintRecord.at_least("climb_engine_rating_with_battery_and_margin",
                                  design.engine_rating_kw,
                                  scenario.sizing_margin * climb_raw, "kW shaft SL",
                                  "constraint_diagram.py CD-04", epsilon),
        ConstraintRecord.at_least("takeoff_combined_power_with_margin",
                                  design.engine_rating_kw,
                                  scenario.sizing_margin * takeoff_raw, "kW shaft SL",
                                  "aerodynamics.py/powertrain.py", epsilon),
        ConstraintRecord.at_least("battery_peak_discharge_sustainable_one_step",
                                  discharge, required_boost, "kW bus",
                                  "models/battery.py B-06", epsilon),
    ]
    ceiling_constraint = ConstraintRecord.at_least(
        "service_ceiling_10km", design.engine_rating_kw,
        scenario.sizing_margin * ceiling_raw, "kW shaft SL",
        "constraint_diagram.py CD-02/O-12", epsilon,
    )
    advisory: tuple[ConstraintRecord, ...] = ()
    if scenario.service_ceiling_is_hard:
        hard.append(ceiling_constraint)
    else:
        advisory = (ceiling_constraint,)
    hard_tuple = tuple(hard)
    violations = tuple(item.normalized_violation for item in hard_tuple)
    violated = sum(not item.satisfied for item in hard_tuple)
    warnings = (
        StaticDiagnostic(
            "provisional_wetted_area_model",
            "Fixed non-wing wetted area is reference-calibrated, not measured.",
            "assumptions.md OPT-02",
        ),
        StaticDiagnostic(
            "fixed_oswald_policy",
            "Aspect-ratio results are conditional on fixed Oswald efficiency.",
            "assumptions.md O-01/OPT-02",
        ),
        StaticDiagnostic(
            "mission_feasibility_not_proven",
            "Passing static checks does not prove six-phase mission feasibility.",
            "optimization/feasibility.py",
        ),
    )
    return StaticFeasibilityResult(
        resolved, hard_tuple, advisory, warnings, violated == 0,
        math.fsum(violations), max(violations, default=0.0), violated,
    )
