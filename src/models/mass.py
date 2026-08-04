"""Mass budget for a series hybrid-electric UAV at fixed MTOW.

Fuel is the algebraic residual of the budget, so every kilogram of structure or
powertrain is a kilogram of fuel not carried. The public API is SI (kg, m, m^2,
kW, kWh); the Raymer regression's US customary units are converted inside
:func:`wing_mass`. Rationale for every default is in ``docs/assumptions.md``
(M-01..M-08); the wing regression is Raymer, *Aircraft Design: A Conceptual
Approach*, ch. 15, general-aviation statistical group weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MassBreakdown",
    "battery_mass",
    "build_mass_budget",
    "component_mass",
    "engine_mass",
    "fuel_volume_check",
    "generator_mass",
    "inverter_mass",
    "is_feasible",
    "motor_mass",
    "rectifier_mass",
    "wing_mass",
    "wing_mass_simple",
]

# ---------------------------------------------------------------------------
# Vehicle-level constants
# ---------------------------------------------------------------------------

MTOW_KG = 1000.0
PAYLOAD_KG = 200.0
FIXED_GROUP_KG = 250.0
MIN_USABLE_FUEL_KG = 20.0

# ---------------------------------------------------------------------------
# Wing regression defaults
# ---------------------------------------------------------------------------

COMPOSITE_FACTOR = 0.87
LIMIT_LOAD_FACTOR = 3.8
ULTIMATE_LOAD_SAFETY_FACTOR = 1.5
TAPER_RATIO = 0.5
THICKNESS_TO_CHORD = 0.15
SWEEP_RAD = 0.0
CRUISE_DYNAMIC_PRESSURE_PA = 1590.5
FUEL_IN_WING_KG = 250.0
SIMPLE_WING_COEFFICIENT = 4.0

# ---------------------------------------------------------------------------
# Powertrain specific powers and pack constants
# ---------------------------------------------------------------------------

ENGINE_KW_PER_KG = 3.5
GENERATOR_KW_PER_KG = 3.0
RECTIFIER_KW_PER_KG = 15.0
INVERTER_KW_PER_KG = 15.0
MOTOR_KW_PER_KG = 7.0
CABLING_COOLING_FRACTION = 0.15

CELL_ENERGY_DENSITY_WH_KG = 250.0
PACK_FACTOR = 0.75

# ---------------------------------------------------------------------------
# Fuel constants
# ---------------------------------------------------------------------------

FUEL_SYSTEM_FRACTION = 0.07
FUEL_DENSITY_KG_PER_L = 0.804
TANK_VOLUME_FRACTION = 0.5

# ---------------------------------------------------------------------------
# Unit conversions (SI boundary -> Raymer's US customary units)
# ---------------------------------------------------------------------------

KG_TO_LB = 2.2046226218
M2_TO_FT2 = 10.7639104167
PA_TO_PSF = 0.0208854342


def _require_positive(**values: float) -> None:
    """Raise ``ValueError`` naming the first argument that is not > 0."""
    for name, value in values.items():
        if not value > 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}")


# ---------------------------------------------------------------------------
# Wing
# ---------------------------------------------------------------------------


def wing_mass(
    s_m2: float,
    ar: float,
    design_gross_mass_kg: float = MTOW_KG,
    q_pa: float = CRUISE_DYNAMIC_PRESSURE_PA,
    fuel_in_wing_kg: float = FUEL_IN_WING_KG,
    sweep_rad: float = SWEEP_RAD,
    taper: float = TAPER_RATIO,
    tc: float = THICKNESS_TO_CHORD,
    n_z: float = LIMIT_LOAD_FACTOR,
    ultimate_factor: float = ULTIMATE_LOAD_SAFETY_FACTOR,
    construction_factor: float = COMPOSITE_FACTOR,
) -> float:
    """Wing mass [kg] from the Raymer general-aviation weight regression.

    Args:
        s_m2: Trapezoidal wing reference area [m^2].
        ar: Aspect ratio [-].
        design_gross_mass_kg: Flight design gross mass, i.e. MTOW [kg].
        q_pa: Dynamic pressure at cruise [Pa].
        fuel_in_wing_kg: Fuel carried in the wing [kg]; the regression exponent
            is 0.0035 - see assumptions.md M-03.
        sweep_rad: Quarter-chord sweep [rad].
        n_z: Design LIMIT load factor. Raymer's N_z is the ULTIMATE factor, so
            the regression is fed ``ultimate_factor * n_z``.
        construction_factor: Composite-construction multiplier - see
            assumptions.md M-03.

    Returns:
        Wing structural mass [kg].
    """
    _require_positive(
        s_m2=s_m2,
        ar=ar,
        design_gross_mass_kg=design_gross_mass_kg,
        q_pa=q_pa,
        fuel_in_wing_kg=fuel_in_wing_kg,
        taper=taper,
        tc=tc,
        n_z=n_z,
        ultimate_factor=ultimate_factor,
        construction_factor=construction_factor,
    )

    s_ft2 = s_m2 * M2_TO_FT2
    w_dg_lb = design_gross_mass_kg * KG_TO_LB
    w_fw_lb = fuel_in_wing_kg * KG_TO_LB
    q_psf = q_pa * PA_TO_PSF
    cos_sweep = math.cos(sweep_rad)
    n_ultimate = ultimate_factor * n_z

    # Raymer GA wing weight regression, US customary units in and out.
    w_wing_lb = (
        0.036
        * s_ft2**0.758
        * w_fw_lb**0.0035
        * (ar / cos_sweep**2) ** 0.6
        * q_psf**0.006
        * taper**0.04
        * (100.0 * tc / cos_sweep) ** -0.3
        * (n_ultimate * w_dg_lb) ** 0.49
    )

    return construction_factor * w_wing_lb / KG_TO_LB


def wing_mass_simple(s_m2: float, ar: float, k: float = SIMPLE_WING_COEFFICIENT) -> float:
    """Wing mass [kg] from the simplified m = k * S**0.75 * AR**0.6 scaling law."""
    _require_positive(s_m2=s_m2, ar=ar, k=k)
    return k * s_m2**0.75 * ar**0.6


# ---------------------------------------------------------------------------
# Powertrain components
# ---------------------------------------------------------------------------


def component_mass(power_kw: float, specific_power_kw_kg: float) -> float:
    """Mass [kg] of a powertrain component sized by rated power."""
    _require_positive(power_kw=power_kw, specific_power_kw_kg=specific_power_kw_kg)
    return power_kw / specific_power_kw_kg


def engine_mass(power_kw: float, specific_power_kw_kg: float = ENGINE_KW_PER_KG) -> float:
    """Turboshaft mass [kg] from rated shaft power."""
    return component_mass(power_kw, specific_power_kw_kg)


def generator_mass(power_kw: float, specific_power_kw_kg: float = GENERATOR_KW_PER_KG) -> float:
    """Generator mass [kg] from rated electrical power."""
    return component_mass(power_kw, specific_power_kw_kg)


def rectifier_mass(power_kw: float, specific_power_kw_kg: float = RECTIFIER_KW_PER_KG) -> float:
    """Rectifier mass [kg] from rated throughput power."""
    return component_mass(power_kw, specific_power_kw_kg)


def inverter_mass(power_kw: float, specific_power_kw_kg: float = INVERTER_KW_PER_KG) -> float:
    """Inverter mass [kg] from rated throughput power."""
    return component_mass(power_kw, specific_power_kw_kg)


def motor_mass(power_kw: float, specific_power_kw_kg: float = MOTOR_KW_PER_KG) -> float:
    """Electric motor mass [kg] from rated shaft power."""
    return component_mass(power_kw, specific_power_kw_kg)


def battery_mass(
    capacity_kwh: float,
    cell_wh_kg: float = CELL_ENERGY_DENSITY_WH_KG,
    pack_factor: float = PACK_FACTOR,
) -> float:
    """Battery pack mass [kg] from usable energy capacity.

    Args:
        capacity_kwh: Pack energy capacity [kWh]; zero is allowed (pure thermal).
        cell_wh_kg: Cell-level specific energy [Wh/kg].
        pack_factor: Cell-to-pack mass efficiency [-] - see assumptions.md B-02.
    """
    _require_positive(cell_wh_kg=cell_wh_kg, pack_factor=pack_factor)
    if capacity_kwh < 0.0:
        raise ValueError(f"capacity_kwh must be non-negative, got {capacity_kwh!r}")
    return (capacity_kwh * 1000.0) / (cell_wh_kg * pack_factor)


# ---------------------------------------------------------------------------
# Fuel volume
# ---------------------------------------------------------------------------


def fuel_volume_check(
    fuel_kg: float,
    s_m2: float,
    ar: float,
    tc: float = THICKNESS_TO_CHORD,
    tank_volume_fraction: float = TANK_VOLUME_FRACTION,
    fuel_density_kg_per_l: float = FUEL_DENSITY_KG_PER_L,
) -> tuple[float, float, bool]:
    """Required fuel volume [L], usable tank volume [L], and whether the fuel fits."""
    _require_positive(
        s_m2=s_m2,
        ar=ar,
        tc=tc,
        tank_volume_fraction=tank_volume_fraction,
        fuel_density_kg_per_l=fuel_density_kg_per_l,
    )

    fuel_l = fuel_kg / fuel_density_kg_per_l
    # Wing box volume ~ S * chord * (t/c) with chord = sqrt(S/AR); m^3 -> L.
    tank_l = tank_volume_fraction * s_m2**1.5 / math.sqrt(ar) * tc * 1000.0
    return fuel_l, tank_l, fuel_l <= tank_l


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassBreakdown:
    """Itemized mass budget [kg]. Every field sums into :attr:`total_kg` = MTOW."""

    fixed_kg: float
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
    fuel_kg: float

    @property
    def electrical_total_kg(self) -> float:
        """Generator, rectifier, inverter, motor, cabling and cooling."""
        return (
            self.generator_kg
            + self.rectifier_kg
            + self.inverter_kg
            + self.motor_kg
            + self.cabling_cooling_kg
        )

    @property
    def propulsion_total_kg(self) -> float:
        """Everything installed to produce thrust, excluding the fuel itself."""
        return (
            self.engine_kg + self.electrical_total_kg + self.battery_kg + self.fuel_system_kg
        )

    @property
    def dry_kg(self) -> float:
        """Take-off mass less fuel."""
        return self.fixed_kg + self.payload_kg + self.wing_kg + self.propulsion_total_kg

    @property
    def total_kg(self) -> float:
        """Take-off mass; equals MTOW whenever the budget closed."""
        return self.dry_kg + self.fuel_kg


def build_mass_budget(
    engine_kw: float,
    battery_kwh: float,
    peak_bus_kw: float,
    wing_area_m2: float,
    aspect_ratio: float,
    *,
    mtow_kg: float = MTOW_KG,
    payload_kg: float = PAYLOAD_KG,
    fixed_kg: float = FIXED_GROUP_KG,
    engine_kw_per_kg: float = ENGINE_KW_PER_KG,
    generator_kw_per_kg: float = GENERATOR_KW_PER_KG,
    rectifier_kw_per_kg: float = RECTIFIER_KW_PER_KG,
    inverter_kw_per_kg: float = INVERTER_KW_PER_KG,
    motor_kw_per_kg: float = MOTOR_KW_PER_KG,
    cabling_cooling_fraction: float = CABLING_COOLING_FRACTION,
    cell_wh_kg: float = CELL_ENERGY_DENSITY_WH_KG,
    pack_factor: float = PACK_FACTOR,
    fuel_system_fraction: float = FUEL_SYSTEM_FRACTION,
    q_pa: float = CRUISE_DYNAMIC_PRESSURE_PA,
    fuel_in_wing_kg: float = FUEL_IN_WING_KG,
    sweep_rad: float = SWEEP_RAD,
    taper: float = TAPER_RATIO,
    tc: float = THICKNESS_TO_CHORD,
    n_z: float = LIMIT_LOAD_FACTOR,
    ultimate_factor: float = ULTIMATE_LOAD_SAFETY_FACTOR,
    construction_factor: float = COMPOSITE_FACTOR,
) -> MassBreakdown:
    """Full mass breakdown for one design, with fuel as the residual to MTOW.

    Args:
        engine_kw: Turboshaft rated shaft power [kW].
        battery_kwh: Battery pack capacity [kWh].
        peak_bus_kw: Peak DC bus power [kW], engine output plus battery discharge.
        wing_area_m2: Wing reference area [m^2].
        aspect_ratio: Wing aspect ratio [-].

    Returns:
        :class:`MassBreakdown`. ``fuel_kg`` may be negative for an infeasible
        design; nothing is raised and nothing is clamped.
    """
    # 1 + f is the closure denominator, so it must not vanish or change sign.
    _require_positive(mtow_kg=mtow_kg, one_plus_fuel_system_fraction=1.0 + fuel_system_fraction)

    wing = wing_mass(
        wing_area_m2,
        aspect_ratio,
        design_gross_mass_kg=mtow_kg,
        q_pa=q_pa,
        fuel_in_wing_kg=fuel_in_wing_kg,
        sweep_rad=sweep_rad,
        taper=taper,
        tc=tc,
        n_z=n_z,
        ultimate_factor=ultimate_factor,
        construction_factor=construction_factor,
    )

    # Generator and rectifier sit on the engine side of the DC bus; the inverter
    # and motor carry engine plus battery power - see assumptions.md M-04.
    engine = engine_mass(engine_kw, engine_kw_per_kg)
    generator = generator_mass(engine_kw, generator_kw_per_kg)
    rectifier = rectifier_mass(engine_kw, rectifier_kw_per_kg)
    inverter = inverter_mass(peak_bus_kw, inverter_kw_per_kg)
    motor = motor_mass(peak_bus_kw, motor_kw_per_kg)
    cabling_cooling = cabling_cooling_fraction * (generator + rectifier + inverter + motor)

    battery = battery_mass(battery_kwh, cell_wh_kg=cell_wh_kg, pack_factor=pack_factor)

    non_fuel = (
        fixed_kg
        + payload_kg
        + wing
        + engine
        + generator
        + rectifier
        + inverter
        + motor
        + cabling_cooling
        + battery
    )
    # MTOW closure with m_fuel_system = f * m_fuel substituted and solved for m_fuel.
    fuel = (mtow_kg - non_fuel) / (1.0 + fuel_system_fraction)

    return MassBreakdown(
        fixed_kg=fixed_kg,
        payload_kg=payload_kg,
        wing_kg=wing,
        engine_kg=engine,
        generator_kg=generator,
        rectifier_kg=rectifier,
        inverter_kg=inverter,
        motor_kg=motor,
        cabling_cooling_kg=cabling_cooling,
        battery_kg=battery,
        fuel_system_kg=fuel_system_fraction * fuel,
        fuel_kg=fuel,
    )


def is_feasible(breakdown: MassBreakdown, min_fuel_kg: float = MIN_USABLE_FUEL_KG) -> bool:
    """Whether the design carries at least the minimum usable fuel load."""
    return breakdown.fuel_kg >= min_fuel_kg
