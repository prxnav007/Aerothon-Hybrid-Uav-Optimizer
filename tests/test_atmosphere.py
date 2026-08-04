"""Tests for :mod:`src.models.atmosphere` (ISA / ICAO Standard Atmosphere).

Three tiers, in increasing order of how much they prove:

1. **Reference values** - agreement with published ISA tables at rel=1e-4.
   These catch a wrong constant but depend on the table being right.
2. **Internal consistency** - the ideal gas law, layer continuity, sea-level
   recovery and monotonicity. These hold no matter which table you trust and
   are the strongest checks here: they validate the pressure and density
   formulas against *each other*.
3. **Behaviour** - vectorization, scalar/array typing, range guarding, and the
   Mach check that justifies incompressible aerodynamics for this vehicle.

Reference-value provenance
--------------------------
All reference altitudes below are GEOPOTENTIAL metres. Values were cross-checked
against the PDAS standard atmosphere table (https://www.pdas.com/atmosTable1SI.html)
and the ISO 2533 constants; note PDAS tabulates *geometric* altitude, so it only
cross-checks these rows after conversion.

Corrections made to the reference values originally supplied for this module:

* ``a(10000 m) = 299.53`` -> **299.463 m/s**. 299.53 m/s is the speed of sound at
  10 km *geometric* (= 9 984.29 m geopotential, T = 223.25 K); the T, p and rho
  in that same row are geopotential values, so the row mixed the two altitude
  conventions. At 10 km geopotential, T = 223.15 K exactly and
  sqrt(gamma*R*T) = 299.4632 m/s. Published tables list 299.46 m/s.
* ``a(5000 m) = 320.55`` -> **320.529 m/s**, and ``a(6000 m) = 316.45`` ->
  **316.428 m/s**. Both originals were within the 1e-4 tolerance but were less
  accurate than the rest of the column.

Every other supplied value was verified correct and is used unchanged.
"""

import dataclasses

import numpy as np
import pytest

from src.models import atmosphere as atm
from src.models.atmosphere import AtmosphericState

# ---------------------------------------------------------------------------
# Published ISA reference table, geopotential altitude.
#   h [m] -> (T [K], p [Pa], rho [kg/m^3], a [m/s])
# ---------------------------------------------------------------------------
ISA_TABLE: dict[float, tuple[float, float, float, float]] = {
    0: (288.15, 101325.0, 1.2250, 340.294),
    1000: (281.65, 89874.6, 1.11164, 336.434),
    3000: (268.65, 70108.5, 0.909122, 328.578),
    5000: (255.65, 54019.9, 0.736116, 320.529),
    6000: (249.15, 47181.0, 0.659697, 316.428),
    8000: (236.15, 35599.8, 0.525168, 308.063),
    10000: (223.15, 26436.3, 0.412707, 299.463),
    11000: (216.65, 22632.1, 0.363918, 295.070),
    15000: (216.65, 12044.6, 0.193674, 295.070),
    20000: (216.65, 5474.89, 0.0880349, 295.070),
}

# PDAS dynamic viscosity, 1e-6 Pa*s, at GEOMETRIC altitude (that table's
# convention) - converted before comparison in the viscosity test.
PDAS_VISCOSITY_GEOMETRIC: dict[float, float] = {
    0: 17.89e-6,
    6000: 15.95e-6,
    8000: 15.27e-6,
    10000: 14.58e-6,
}

# UAV cruise condition: 250 km/h true airspeed at 6 km.
CRUISE_TAS_MPS = 69.44
CRUISE_ALTITUDE_M = 6000.0


# ===========================================================================
# 1. Reference-value tests
# ===========================================================================


@pytest.mark.parametrize("h_m", sorted(ISA_TABLE))
def test_temperature_matches_isa_table(h_m: float) -> None:
    expected = ISA_TABLE[h_m][0]
    assert atm.temperature(float(h_m)) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("h_m", sorted(ISA_TABLE))
def test_pressure_matches_isa_table(h_m: float) -> None:
    expected = ISA_TABLE[h_m][1]
    assert atm.pressure(float(h_m)) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("h_m", sorted(ISA_TABLE))
def test_density_matches_isa_table(h_m: float) -> None:
    expected = ISA_TABLE[h_m][2]
    assert atm.density(float(h_m)) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("h_m", sorted(ISA_TABLE))
def test_speed_of_sound_matches_isa_table(h_m: float) -> None:
    expected = ISA_TABLE[h_m][3]
    assert atm.speed_of_sound(float(h_m)) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("z_m", sorted(PDAS_VISCOSITY_GEOMETRIC))
def test_dynamic_viscosity_matches_pdas_table(z_m: float) -> None:
    """Sutherland's law against an independent table.

    This is what settles the beta_s = 1.458e-6 vs 1.468e-6 disagreement between
    sources: 1.468e-6 would give 18.02e-6 Pa*s at sea level, missing the
    tabulated 17.89e-6 by 0.7 % - far outside this tolerance.
    """
    h_m = atm.geometric_to_geopotential(float(z_m))
    expected = PDAS_VISCOSITY_GEOMETRIC[z_m]
    # rel=1e-3: the PDAS column is quoted to only 4 significant figures.
    assert atm.dynamic_viscosity(h_m) == pytest.approx(expected, rel=1e-3)


def test_atmosphere_bundle_matches_isa_table() -> None:
    """The one-pass entry point reproduces the table on every field at once."""
    for h_m, (t_k, p_pa, rho, a_mps) in ISA_TABLE.items():
        state = atm.atmosphere(float(h_m))
        assert state.altitude_m == pytest.approx(float(h_m))
        assert state.temperature_k == pytest.approx(t_k, rel=1e-4)
        assert state.pressure_pa == pytest.approx(p_pa, rel=1e-4)
        assert state.density_kg_m3 == pytest.approx(rho, rel=1e-4)
        assert state.speed_of_sound_mps == pytest.approx(a_mps, rel=1e-4)
        assert state.density_ratio == pytest.approx(rho / atm.RHO0, rel=1e-4)


# ===========================================================================
# 2. Internal consistency tests
# ===========================================================================


def test_ideal_gas_law_holds_across_full_range() -> None:
    """rho == p / (R*T) at 200 altitudes spanning both layers.

    The strongest single check in this file: it validates the pressure and
    density formulas against each other without reference to any table.
    """
    h = np.linspace(atm.H_MIN, atm.H_MAX, 200)
    state = atm.atmosphere(h)
    rho_from_gas_law = state.pressure_pa / (atm.R_AIR * state.temperature_k)
    np.testing.assert_allclose(
        state.density_kg_m3, rho_from_gas_law, rtol=1e-9, atol=0.0
    )


def test_density_matches_direct_troposphere_form() -> None:
    """rho == RHO0 * (T/T0)**(n-1) below the tropopause.

    The module computes density as p/(R*T); this checks that against the
    independent closed form, so a mistake in either would show up.
    """
    h = np.linspace(0.0, atm.H_TROPOPAUSE, 120, endpoint=False)
    t = atm.temperature(h)
    direct = atm.RHO0 * np.power(t / atm.T0, atm.BARO_EXPONENT - 1.0)
    np.testing.assert_allclose(atm.density(h), direct, rtol=1e-12, atol=0.0)


def test_density_matches_direct_stratosphere_form() -> None:
    """rho == RHO_TROPOPAUSE * exp(-(h - 11000)/H_s) above the tropopause."""
    h = np.linspace(atm.H_TROPOPAUSE, atm.H_MAX, 120)
    direct = atm.RHO_TROPOPAUSE * np.exp(
        -(h - atm.H_TROPOPAUSE) / atm.SCALE_HEIGHT_STRATO
    )
    np.testing.assert_allclose(atm.density(h), direct, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    ("h_m", "min_divergence"),
    [
        # Measured divergence between the two forms: 4.0 % at 15 km, 21 % at
        # 20 km. Thresholds sit safely below the measured values.
        (15000.0, 0.03),
        (20000.0, 0.15),
    ],
)
def test_stratosphere_does_not_use_troposphere_power_law(
    h_m: float, min_divergence: float
) -> None:
    """Guard against the classic ISA bug: power law applied above 11 km.

    Not redundant with the table tests. The two forms agree exactly at 11 km and
    separate slowly above it, so the wrong branch is easy to miss just under the
    tropopause - it only reaches 0.24 % at 12 km. This pins that we are on the
    exponential, and quantifies how badly the power law would fail: it
    under-predicts stratospheric pressure, which would flatter the UAV's
    high-altitude cruise performance.
    """
    wrong = atm.P0 * np.power(
        (atm.T0 - atm.L_TROPO * h_m) / atm.T0, atm.BARO_EXPONENT
    )
    correct = atm.P_TROPOPAUSE * np.exp(
        -(h_m - atm.H_TROPOPAUSE) / atm.SCALE_HEIGHT_STRATO
    )

    assert atm.pressure(h_m) == pytest.approx(correct, rel=1e-12)
    assert wrong < correct, "the power law under-predicts above the tropopause"
    assert abs(wrong - correct) / correct > min_divergence
    assert atm.pressure(h_m) != pytest.approx(wrong, rel=1e-3)


def test_layer_continuity_at_tropopause() -> None:
    """T, p and rho are continuous across the 11 km branch point.

    A jump here means a wrong tropopause constant or a mis-ordered branch.
    """
    below = atm.atmosphere(atm.H_TROPOPAUSE - 1e-6)
    above = atm.atmosphere(atm.H_TROPOPAUSE + 1e-6)

    assert below.temperature_k == pytest.approx(above.temperature_k, rel=1e-6)
    assert below.pressure_pa == pytest.approx(above.pressure_pa, rel=1e-6)
    assert below.density_kg_m3 == pytest.approx(above.density_kg_m3, rel=1e-6)


def test_tropopause_boundary_belongs_to_isothermal_layer() -> None:
    """h == 11000 exactly is evaluated on the stratosphere branch (h < 11000)."""
    state = atm.atmosphere(atm.H_TROPOPAUSE)
    assert state.temperature_k == pytest.approx(atm.T_TROPOPAUSE, rel=1e-15)
    assert state.pressure_pa == pytest.approx(atm.P_TROPOPAUSE, rel=1e-15)
    assert state.density_kg_m3 == pytest.approx(atm.RHO_TROPOPAUSE, rel=1e-15)


def test_sea_level_recovers_isa_definitions() -> None:
    """atmosphere(0) returns the sea-level standard values.

    Density is compared against the module's derived ``RHO0`` (exact) and,
    separately and slightly more loosely, against the table value 1.225. The
    1.4e-9 gap between them is the ISA constant set's own closure error:
    P0/(R*T0) = 1.2250000017, not 1.225 exactly. Asserting 1.225 at 1e-9 would
    be asserting that the standard is more self-consistent than it is.
    """
    sl = atm.atmosphere(0.0)
    assert sl.temperature_k == pytest.approx(atm.T0, rel=1e-9)
    assert sl.pressure_pa == pytest.approx(atm.P0, rel=1e-9)
    assert sl.density_kg_m3 == pytest.approx(atm.RHO0, rel=1e-15)
    assert sl.density_kg_m3 == pytest.approx(1.225, rel=1e-8)
    assert sl.density_ratio == pytest.approx(1.0, rel=1e-15)


def test_pressure_and_density_strictly_decrease_with_altitude() -> None:
    h = np.linspace(atm.H_MIN, atm.H_MAX, 2001)
    assert np.all(np.diff(atm.pressure(h)) < 0.0)
    assert np.all(np.diff(atm.density(h)) < 0.0)


def test_temperature_decreases_in_troposphere_and_is_constant_above() -> None:
    h_tropo = np.linspace(0.0, atm.H_TROPOPAUSE, 1000, endpoint=False)
    assert np.all(np.diff(atm.temperature(h_tropo)) < 0.0)

    h_strato = np.linspace(atm.H_TROPOPAUSE, atm.H_MAX, 1000)
    t_strato = atm.temperature(h_strato)
    # "Exactly constant": bitwise identical, not merely close.
    assert np.all(t_strato == atm.T_TROPOPAUSE)


def test_speed_of_sound_and_viscosity_constant_above_tropopause() -> None:
    """Both depend on temperature alone, so both are flat in the isothermal layer."""
    h = np.linspace(atm.H_TROPOPAUSE, atm.H_MAX, 500)
    np.testing.assert_allclose(atm.speed_of_sound(h), 295.0695, rtol=1e-5)
    np.testing.assert_allclose(atm.dynamic_viscosity(h), 1.421613e-05, rtol=1e-5)


# --- Derived-constant assertions -------------------------------------------


def test_specific_gas_constant_is_derived_not_hardcoded() -> None:
    assert atm.R_STAR / atm.M_AIR == atm.R_AIR
    assert atm.R_AIR == pytest.approx(287.05287, rel=1e-7)


def test_sea_level_density_is_derived_from_ideal_gas_law() -> None:
    assert atm.P0 / (atm.R_AIR * atm.T0) == atm.RHO0
    assert atm.RHO0 == pytest.approx(1.225, rel=1e-8)


def test_tropopause_temperature_is_derived_from_lapse_rate() -> None:
    assert atm.T0 - atm.L_TROPO * 11000.0 == pytest.approx(216.65, rel=1e-12)
    assert atm.T_TROPOPAUSE == pytest.approx(216.65, rel=1e-12)


def test_barometric_exponent_and_scale_height() -> None:
    assert atm.BARO_EXPONENT == pytest.approx(5.2559, rel=1e-4)
    assert atm.SCALE_HEIGHT_STRATO == pytest.approx(6341.6, rel=1e-4)
    assert atm.SCALE_HEIGHT_STRATO == atm.R_AIR * atm.T_TROPOPAUSE / atm.g0


def test_tropopause_pressure_and_density_are_derived() -> None:
    assert atm.P_TROPOPAUSE == pytest.approx(22632.1, rel=1e-4)
    assert atm.RHO_TROPOPAUSE == pytest.approx(0.363918, rel=1e-4)
    assert atm.RHO_TROPOPAUSE == pytest.approx(
        atm.P_TROPOPAUSE / (atm.R_AIR * atm.T_TROPOPAUSE), rel=1e-15
    )


# ===========================================================================
# 3. Behavioural tests
# ===========================================================================

_SCALAR_FUNCS = [
    atm.temperature,
    atm.pressure,
    atm.density,
    atm.speed_of_sound,
    atm.density_ratio,
    atm.dynamic_viscosity,
]


@pytest.mark.parametrize("func", _SCALAR_FUNCS, ids=lambda f: f.__name__)
def test_scalar_input_returns_python_float(func) -> None:
    """Documented convention: anything with ndim == 0 gives back a float."""
    for scalar in (6000.0, 6000, np.float64(6000.0), np.array(6000.0)):
        result = func(scalar)
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)


@pytest.mark.parametrize("func", _SCALAR_FUNCS, ids=lambda f: f.__name__)
def test_array_input_returns_array_of_matching_shape(func) -> None:
    h = np.array([[0.0, 5000.0], [11000.0, 20000.0]])
    result = func(h)
    assert isinstance(result, np.ndarray)
    assert result.shape == h.shape
    assert result.dtype == np.float64


@pytest.mark.parametrize("func", _SCALAR_FUNCS, ids=lambda f: f.__name__)
def test_vectorized_matches_elementwise_scalar_calls(func) -> None:
    """Vectorization must not change the numbers. Spans both layers."""
    h = np.linspace(atm.H_MIN, atm.H_MAX, 137)
    vectorized = func(h)
    elementwise = np.array([func(float(x)) for x in h])
    np.testing.assert_allclose(vectorized, elementwise, rtol=0.0, atol=0.0)


def test_atmosphere_scalar_fields_are_floats() -> None:
    state = atm.atmosphere(6000.0)
    assert isinstance(state, AtmosphericState)
    for value in vars(state).values():
        assert isinstance(value, float)


def test_atmosphere_array_fields_are_arrays_of_matching_shape() -> None:
    h = np.linspace(0.0, 20000.0, 11)
    state = atm.atmosphere(h)
    for value in vars(state).values():
        assert isinstance(value, np.ndarray)
        assert value.shape == h.shape


def test_atmosphere_agrees_with_individual_functions() -> None:
    """The one-pass path must be numerically identical to the readable path."""
    h = np.linspace(atm.H_MIN, atm.H_MAX, 101)
    state = atm.atmosphere(h)
    for computed, expected in (
        (state.temperature_k, atm.temperature(h)),
        (state.pressure_pa, atm.pressure(h)),
        (state.density_kg_m3, atm.density(h)),
        (state.speed_of_sound_mps, atm.speed_of_sound(h)),
        (state.density_ratio, atm.density_ratio(h)),
        (state.dynamic_viscosity_pa_s, atm.dynamic_viscosity(h)),
    ):
        np.testing.assert_allclose(computed, expected, rtol=0.0, atol=0.0)


def test_atmospheric_state_is_frozen() -> None:
    state = atm.atmosphere(0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.temperature_k = 300.0  # type: ignore[misc]


# --- Range guarding ---------------------------------------------------------


@pytest.mark.parametrize("bad_h", [-1.0, 20001.0])
def test_out_of_range_altitude_raises_value_error(bad_h: float) -> None:
    with pytest.raises(ValueError) as excinfo:
        atm.atmosphere(bad_h)

    message = str(excinfo.value)
    assert "20000" in message, "error should name the valid range"
    assert repr(bad_h) in message, "error should name the offending value"


@pytest.mark.parametrize("func", _SCALAR_FUNCS, ids=lambda f: f.__name__)
def test_every_public_function_guards_its_range(func) -> None:
    with pytest.raises(ValueError):
        func(-1.0)
    with pytest.raises(ValueError):
        func(20001.0)


def test_out_of_range_element_in_array_raises() -> None:
    """One bad element poisons the batch - it is not silently dropped."""
    with pytest.raises(ValueError, match="20500"):
        atm.atmosphere(np.array([0.0, 5000.0, 20500.0]))


def test_non_finite_altitude_raises() -> None:
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            atm.atmosphere(bad)


def test_range_boundaries_are_inclusive() -> None:
    """0 m and 20 000 m are valid; the guard must not be off by one."""
    assert atm.atmosphere(atm.H_MIN).pressure_pa == pytest.approx(atm.P0)
    assert atm.atmosphere(atm.H_MAX).pressure_pa == pytest.approx(5474.89, rel=1e-4)


def test_altitudes_are_not_silently_clamped() -> None:
    """Explicitly pin the no-clamping contract that the GA depends on."""
    with pytest.raises(ValueError):
        atm.density(25000.0)


# --- Altitude conversions ---------------------------------------------------


def test_geopotential_geometric_round_trip() -> None:
    z = np.linspace(0.0, 20000.0, 200)
    round_tripped = atm.geopotential_to_geometric(atm.geometric_to_geopotential(z))
    np.testing.assert_allclose(round_tripped, z, rtol=1e-9, atol=1e-9)

    h = np.linspace(0.0, 20000.0, 200)
    round_tripped_h = atm.geometric_to_geopotential(atm.geopotential_to_geometric(h))
    np.testing.assert_allclose(round_tripped_h, h, rtol=1e-9, atol=1e-9)


def test_geopotential_is_below_geometric_by_about_0p16_percent_at_10km() -> None:
    """The conversion matters: ~16 m at 10 km, which is not noise."""
    h = atm.geometric_to_geopotential(10000.0)
    assert h == pytest.approx(9984.293, abs=1e-3)
    assert (10000.0 - h) / 10000.0 == pytest.approx(0.0016, abs=1e-4)


def test_conversions_are_scalar_and_array_polymorphic() -> None:
    assert isinstance(atm.geometric_to_geopotential(1000.0), float)
    assert isinstance(atm.geopotential_to_geometric(1000.0), float)
    arr = np.array([0.0, 1000.0])
    assert atm.geometric_to_geopotential(arr).shape == arr.shape
    assert atm.geopotential_to_geometric(arr).shape == arr.shape


def test_conversions_are_identity_at_sea_level() -> None:
    assert atm.geometric_to_geopotential(0.0) == 0.0
    assert atm.geopotential_to_geometric(0.0) == 0.0


# --- Mach number ------------------------------------------------------------


def test_cruise_mach_number_justifies_incompressible_aerodynamics() -> None:
    """UAV cruise: 250 km/h (69.44 m/s) at 6 km -> M ~ 0.22.

    M < 0.3 is the standard threshold below which compressibility effects on
    lift and drag are under ~5 % and the incompressible aerodynamic model used
    downstream is valid. Documented here as an executable assumption: if a
    future design change pushes cruise speed or altitude far enough that this
    fails, the aerodynamics module needs a compressibility correction.
    """
    mach = atm.mach_number(CRUISE_TAS_MPS, CRUISE_ALTITUDE_M)
    assert mach == pytest.approx(0.22, abs=0.005)
    assert mach < 0.3


def test_mach_number_equals_velocity_over_speed_of_sound() -> None:
    h = np.linspace(atm.H_MIN, atm.H_MAX, 50)
    v = 69.44
    np.testing.assert_allclose(
        atm.mach_number(v, h), v / atm.speed_of_sound(h), rtol=0.0, atol=0.0
    )


def test_mach_number_scalar_and_broadcasting() -> None:
    assert isinstance(atm.mach_number(69.44, 6000.0), float)

    # Array velocity against scalar altitude broadcasts to an array.
    v = np.array([50.0, 69.44, 90.0])
    result = atm.mach_number(v, 6000.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == v.shape

    # Array against array, elementwise.
    h = np.array([0.0, 6000.0, 11000.0])
    paired = atm.mach_number(v, h)
    assert paired.shape == v.shape
    np.testing.assert_allclose(paired, v / atm.speed_of_sound(h), rtol=0.0, atol=0.0)


def test_mach_number_guards_altitude_range() -> None:
    with pytest.raises(ValueError):
        atm.mach_number(69.44, 25000.0)


# ===========================================================================
# Regression guard for the GA hot path
# ===========================================================================


def test_large_batch_evaluates_without_warnings() -> None:
    """1e5 altitudes at once, with no numpy warnings.

    The module evaluates both layer formulas everywhere and selects afterwards;
    this pins the claim that neither is singular anywhere in [0, H_MAX], so no
    invalid-value or overflow warning can leak into a GA run.
    """
    h = np.linspace(atm.H_MIN, atm.H_MAX, 100_000)
    with np.errstate(all="raise"):
        state = atm.atmosphere(h)
    assert np.all(np.isfinite(state.pressure_pa))
    assert np.all(np.isfinite(state.density_kg_m3))
    assert np.all(state.density_kg_m3 > 0.0)
    assert np.all(state.temperature_k > 0.0)
