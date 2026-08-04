"""Tests for :mod:`src.models.mass` (mass budget with fuel as the residual).

Verification is in four tiers: budget closure (the arithmetic identity that
every field sums to MTOW), monotonicity (the design trades the module exists to
express), regression sanity (Raymer against an independent scaling law and
against the 8-18 % of MTOW band typical for this class), and guards.
"""

import dataclasses
import math

import pytest

from src.models import mass
from src.models.mass import MassBreakdown, build_mass_budget, is_feasible

# Reference design: 75 kW turboshaft, 10 kWh pack, 90 kW peak bus, S=10, AR=16.
REF = {
    "engine_kw": 75.0,
    "battery_kwh": 10.0,
    "peak_bus_kw": 90.0,
    "wing_area_m2": 10.0,
    "aspect_ratio": 16.0,
}


@pytest.fixture
def ref() -> MassBreakdown:
    return build_mass_budget(**REF)


def _budget(**overrides) -> MassBreakdown:
    return build_mass_budget(**{**REF, **overrides})


# ---------------------------------------------------------------------------
# Closure and consistency
# ---------------------------------------------------------------------------


def test_budget_closes_on_mtow(ref):
    assert ref.total_kg == pytest.approx(mass.MTOW_KG, rel=1e-9)


@pytest.mark.parametrize("engine_kw", [40.0, 75.0, 120.0])
@pytest.mark.parametrize("battery_kwh", [0.0, 10.0, 40.0])
def test_budget_closes_for_any_feasible_design(engine_kw, battery_kwh):
    breakdown = _budget(engine_kw=engine_kw, battery_kwh=battery_kwh)
    assert is_feasible(breakdown)
    assert breakdown.total_kg == pytest.approx(mass.MTOW_KG, rel=1e-9)


def test_fuel_system_is_the_stated_fraction_of_fuel(ref):
    assert ref.fuel_system_kg == pytest.approx(mass.FUEL_SYSTEM_FRACTION * ref.fuel_kg, rel=1e-12)


def test_algebraic_substitution_matches_a_fixed_point_iteration(ref):
    """The closed form must land where an iterative solve would converge."""
    non_fuel = ref.dry_kg - ref.fuel_system_kg
    fuel = 0.0
    for _ in range(200):
        fuel = mass.MTOW_KG - non_fuel - mass.FUEL_SYSTEM_FRACTION * fuel
    assert fuel == pytest.approx(ref.fuel_kg, rel=1e-12)


def test_dry_plus_fuel_is_total(ref):
    assert ref.dry_kg + ref.fuel_kg == pytest.approx(ref.total_kg, rel=1e-12)


def test_electrical_total_is_the_sum_of_its_parts(ref):
    expected = (
        ref.generator_kg
        + ref.rectifier_kg
        + ref.inverter_kg
        + ref.motor_kg
        + ref.cabling_cooling_kg
    )
    assert ref.electrical_total_kg == pytest.approx(expected, rel=1e-12)


def test_propulsion_total_is_the_sum_of_its_parts(ref):
    expected = ref.engine_kg + ref.electrical_total_kg + ref.battery_kg + ref.fuel_system_kg
    assert ref.propulsion_total_kg == pytest.approx(expected, rel=1e-12)


def test_cabling_is_the_stated_fraction_of_the_electrical_chain(ref):
    chain = ref.generator_kg + ref.rectifier_kg + ref.inverter_kg + ref.motor_kg
    assert ref.cabling_cooling_kg == pytest.approx(mass.CABLING_COOLING_FRACTION * chain, rel=1e-12)


def test_every_field_is_accounted_for_in_the_total(ref):
    fields = sum(getattr(ref, name) for name in ref.__dataclass_fields__)
    assert fields == pytest.approx(ref.total_kg, rel=1e-12)


# ---------------------------------------------------------------------------
# Monotonicity - these encode the design trade
# ---------------------------------------------------------------------------


def test_bigger_engine_costs_fuel(ref):
    assert _budget(engine_kw=120.0).fuel_kg < ref.fuel_kg


def test_bigger_battery_costs_fuel(ref):
    assert _budget(battery_kwh=40.0).fuel_kg < ref.fuel_kg


def test_bigger_wing_costs_fuel(ref):
    bigger = _budget(wing_area_m2=14.0)
    assert bigger.wing_kg > ref.wing_kg
    assert bigger.fuel_kg < ref.fuel_kg


def test_higher_aspect_ratio_costs_wing_mass_at_fixed_area(ref):
    assert _budget(aspect_ratio=22.0).wing_kg > ref.wing_kg


@pytest.mark.parametrize("variable", ["engine_kw", "battery_kwh", "wing_area_m2"])
def test_fuel_decreases_strictly_along_each_design_variable(variable, ref):
    values = {
        "engine_kw": [50.0, 75.0, 100.0, 150.0],
        "battery_kwh": [0.0, 10.0, 25.0, 50.0],
        "wing_area_m2": [8.0, 10.0, 13.0, 18.0],
    }[variable]
    fuels = [_budget(**{variable: v}).fuel_kg for v in values]
    assert all(b < a for a, b in zip(fuels, fuels[1:]))


def test_peak_bus_power_only_loads_the_inverter_and_motor(ref):
    """Generator and rectifier are on the engine side of the bus (M-04)."""
    hotter_bus = _budget(peak_bus_kw=150.0)
    assert hotter_bus.generator_kg == pytest.approx(ref.generator_kg)
    assert hotter_bus.rectifier_kg == pytest.approx(ref.rectifier_kg)
    assert hotter_bus.inverter_kg > ref.inverter_kg
    assert hotter_bus.motor_kg > ref.motor_kg


# ---------------------------------------------------------------------------
# Wing regression sanity
# ---------------------------------------------------------------------------


def test_raymer_agrees_with_the_simple_scaling_law():
    full = mass.wing_mass(10.0, 16.0)
    simple = mass.wing_mass_simple(10.0, 16.0)
    assert 1 / 1.5 < full / simple < 1.5


def test_wing_mass_is_a_plausible_fraction_of_mtow():
    """8-18 % of MTOW is the normal band for this class."""
    assert 80.0 <= mass.wing_mass(10.0, 16.0) <= 180.0


def test_doubling_load_factor_scales_wing_mass_by_two_to_the_0_49():
    ratio = mass.wing_mass(10.0, 16.0, n_z=7.6) / mass.wing_mass(10.0, 16.0, n_z=3.8)
    assert ratio == pytest.approx(2.0**0.49, rel=1e-9)


def test_regression_uses_the_ultimate_load_factor():
    """Raymer's N_z is ultimate = 1.5 x limit, so n_z is scaled before use."""
    limit_only = mass.wing_mass(10.0, 16.0, ultimate_factor=1.0)
    assert mass.wing_mass(10.0, 16.0) == pytest.approx(limit_only * 1.5**0.49, rel=1e-9)


def test_construction_factor_scales_the_regression_linearly():
    metal = mass.wing_mass(10.0, 16.0, construction_factor=1.0)
    assert mass.wing_mass(10.0, 16.0) == pytest.approx(mass.COMPOSITE_FACTOR * metal, rel=1e-12)


def test_wing_regression_matches_a_hand_computed_value():
    """Independent recomputation of the regression in US customary units."""
    s_ft2, w_dg_lb, w_fw_lb = 10.0 * 10.7639104167, 2204.6226218, 551.15565545
    q_psf = 1590.5 * 0.0208854342
    w_lb = (
        0.036
        * s_ft2**0.758
        * w_fw_lb**0.0035
        * 16.0**0.6
        * q_psf**0.006
        * 0.5**0.04
        * 15.0**-0.3
        * (5.7 * w_dg_lb) ** 0.49
    )
    expected = 0.87 * w_lb / 2.2046226218
    assert mass.wing_mass(10.0, 16.0) == pytest.approx(expected, rel=1e-9)


def test_sweep_increases_wing_mass():
    assert mass.wing_mass(10.0, 16.0, sweep_rad=math.radians(25.0)) > mass.wing_mass(10.0, 16.0)


def test_fuel_in_wing_barely_matters():
    """Exponent 0.0035 - doubling the wing fuel moves wing mass under 0.3 %."""
    light = mass.wing_mass(10.0, 16.0, fuel_in_wing_kg=150.0)
    heavy = mass.wing_mass(10.0, 16.0, fuel_in_wing_kg=300.0)
    assert heavy / light == pytest.approx(1.0, abs=3e-3)


# ---------------------------------------------------------------------------
# Component and battery masses
# ---------------------------------------------------------------------------


def test_component_mass_is_power_over_specific_power():
    assert mass.component_mass(75.0, 3.5) == pytest.approx(75.0 / 3.5)


def test_named_wrappers_use_their_documented_defaults():
    assert mass.engine_mass(75.0) == pytest.approx(75.0 / mass.ENGINE_KW_PER_KG)
    assert mass.generator_mass(75.0) == pytest.approx(75.0 / mass.GENERATOR_KW_PER_KG)
    assert mass.rectifier_mass(75.0) == pytest.approx(75.0 / mass.RECTIFIER_KW_PER_KG)
    assert mass.inverter_mass(90.0) == pytest.approx(90.0 / mass.INVERTER_KW_PER_KG)
    assert mass.motor_mass(90.0) == pytest.approx(90.0 / mass.MOTOR_KW_PER_KG)


def test_battery_pack_level_specific_energy_is_cell_times_pack_factor():
    pack_wh_kg = 10.0 * 1000.0 / mass.battery_mass(10.0)
    assert pack_wh_kg == pytest.approx(mass.CELL_ENERGY_DENSITY_WH_KG * mass.PACK_FACTOR)
    assert pack_wh_kg == pytest.approx(187.5)


def test_battery_mass_is_linear_in_capacity_and_zero_at_zero():
    assert mass.battery_mass(0.0) == 0.0
    assert mass.battery_mass(40.0) == pytest.approx(4.0 * mass.battery_mass(10.0))


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def test_oversized_powertrain_gives_negative_fuel():
    breakdown = _budget(engine_kw=200.0, battery_kwh=100.0, peak_bus_kw=220.0)
    assert breakdown.fuel_kg < 0.0
    assert not is_feasible(breakdown)


def test_infeasibility_is_graded_not_clipped():
    """The GA needs a gradient, so worse designs must return worse numbers."""
    worse = _budget(engine_kw=250.0, battery_kwh=150.0, peak_bus_kw=280.0)
    bad = _budget(engine_kw=200.0, battery_kwh=100.0, peak_bus_kw=220.0)
    assert worse.fuel_kg < bad.fuel_kg < 0.0


def test_reference_design_is_feasible(ref):
    assert is_feasible(ref)
    assert 200.0 < ref.fuel_kg < 350.0


def test_is_feasible_respects_the_minimum_usable_threshold(ref):
    assert not is_feasible(ref, min_fuel_kg=ref.fuel_kg + 1.0)
    assert is_feasible(ref, min_fuel_kg=ref.fuel_kg)


# ---------------------------------------------------------------------------
# Fuel volume
# ---------------------------------------------------------------------------


def test_reference_fuel_fits_in_the_wing(ref):
    fuel_l, tank_l, fits = mass.fuel_volume_check(ref.fuel_kg, 10.0, 16.0)
    assert fuel_l == pytest.approx(ref.fuel_kg / mass.FUEL_DENSITY_KG_PER_L)
    assert fuel_l < tank_l
    assert fits


def test_tank_volume_shrinks_as_aspect_ratio_rises(ref):
    volumes = [mass.fuel_volume_check(ref.fuel_kg, 10.0, ar)[1] for ar in (8.0, 16.0, 25.0, 40.0)]
    assert all(b < a for a, b in zip(volumes, volumes[1:]))


def test_tank_volume_scales_as_area_to_the_1_5_over_root_ar(ref):
    _, base, _ = mass.fuel_volume_check(ref.fuel_kg, 10.0, 16.0)
    _, quadrupled_ar, _ = mass.fuel_volume_check(ref.fuel_kg, 10.0, 64.0)
    assert quadrupled_ar == pytest.approx(base / 2.0, rel=1e-12)


def test_volume_constraint_binds_only_at_extreme_aspect_ratio(ref):
    """AR=25 still fits at the reference fuel load; the crossover is near AR=46.5.

    The spec expected AR=25 to fail. It does not: the tank holds 474 L there
    against 348 L of fuel. See the summary - the constant was not tuned.
    """
    assert mass.fuel_volume_check(ref.fuel_kg, 10.0, 25.0)[2]
    assert not mass.fuel_volume_check(ref.fuel_kg, 10.0, 47.0)[2]

    crossover = (mass.TANK_VOLUME_FRACTION * 10.0**1.5 * mass.THICKNESS_TO_CHORD * 1000.0) / (
        ref.fuel_kg / mass.FUEL_DENSITY_KG_PER_L
    )
    assert crossover**2 == pytest.approx(46.5, abs=0.5)


def test_shrinking_the_wing_can_break_the_volume_constraint(ref):
    assert not mass.fuel_volume_check(ref.fuel_kg, 5.0, 16.0)[2]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"engine_kw": 0.0},
        {"engine_kw": -75.0},
        {"wing_area_m2": 0.0},
        {"wing_area_m2": -10.0},
        {"aspect_ratio": 0.0},
        {"aspect_ratio": -16.0},
        {"peak_bus_kw": 0.0},
        {"engine_kw_per_kg": 0.0},
        {"motor_kw_per_kg": -7.0},
        {"mtow_kg": 0.0},
    ],
)
def test_build_mass_budget_rejects_non_positive_inputs(overrides):
    with pytest.raises(ValueError):
        _budget(**overrides)


def test_battery_rejects_negative_capacity():
    with pytest.raises(ValueError):
        mass.battery_mass(-1.0)


@pytest.mark.parametrize("args", [(0.0, 3.5), (75.0, 0.0), (-75.0, 3.5), (75.0, -3.5)])
def test_component_mass_rejects_non_positive_arguments(args):
    with pytest.raises(ValueError):
        mass.component_mass(*args)


@pytest.mark.parametrize("args", [(0.0, 16.0), (10.0, 0.0), (-10.0, 16.0)])
def test_wing_functions_reject_non_positive_geometry(args):
    with pytest.raises(ValueError):
        mass.wing_mass(*args)
    with pytest.raises(ValueError):
        mass.wing_mass_simple(*args)


def test_error_message_names_the_offending_argument():
    with pytest.raises(ValueError, match="aspect_ratio|ar"):
        mass.wing_mass(10.0, -1.0)


def test_breakdown_is_immutable(ref):
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.fuel_kg = 0.0
