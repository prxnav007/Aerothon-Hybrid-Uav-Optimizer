"""Tests for :mod:`src.models.engine` (Willans-line turboshaft).

Verification is in five tiers: the Willans identities that define the model,
calibration scaling with engine size, physical plausibility of the resulting
efficiencies, the altitude lapse, and the operating logic around the low-power
floor. A final test quantifies the part-load penalty against the constant-SFC
model this one replaces.
"""

import dataclasses

import numpy as np
import pytest

from src.models import engine
from src.models.engine import Turboshaft

RATED_KW = 75.0
LOITER_KW = 35.0
PREVIOUS_MODEL_SFC = 0.30

# Density ratios at the cruise band, from models/atmosphere.py.
SIGMA_3KM = 0.742
SIGMA_6KM = 0.538


@pytest.fixture
def eng() -> Turboshaft:
    return Turboshaft(rated_power_kw=RATED_KW)


def _sweep(lo: float = 1.0, hi: float = RATED_KW, n: int = 40) -> np.ndarray:
    return np.linspace(lo, hi, n)


# ---------------------------------------------------------------------------
# Willans identities
# ---------------------------------------------------------------------------


def test_sfc_at_rated_power_recovers_the_calibration(eng):
    assert eng.sfc_kg_kwh(RATED_KW) == pytest.approx(eng.sfc_rated_kg_kwh, rel=1e-12)


def test_sfc_is_the_willans_line_divided_by_power(eng):
    power = _sweep()
    assert eng.sfc_kg_kwh(power) == pytest.approx(eng.willans_a + eng.willans_b / power, rel=1e-12)


def test_fuel_flow_is_affine_in_power(eng):
    """Equal power increments cost equal fuel, wherever you sit on the line."""
    power = _sweep(5.0, 70.0)
    step = 5.0
    increments = eng.fuel_flow_kg_s(power + step) - eng.fuel_flow_kg_s(power)
    assert increments == pytest.approx(eng.willans_a * step / 3600.0, rel=1e-12)


def test_doubling_power_costs_the_marginal_rate_times_that_power(eng):
    power = _sweep(5.0, 37.0)
    delta_kg_h = (eng.fuel_flow_kg_s(2.0 * power) - eng.fuel_flow_kg_s(power)) * 3600.0
    assert delta_kg_h == pytest.approx(eng.willans_a * power, rel=1e-12)


def test_sfc_decreases_strictly_with_power(eng):
    sfc = eng.sfc_kg_kwh(_sweep(0.5, RATED_KW))
    assert np.all(np.diff(sfc) < 0.0)


def test_marginal_efficiency_beats_average_efficiency_everywhere(eng):
    assert np.all(eng.marginal_efficiency > eng.thermal_efficiency(_sweep()))


def test_marginal_efficiency_is_the_willans_slope_in_energy_terms(eng):
    assert eng.marginal_efficiency == pytest.approx(
        3600.0 / (eng.willans_a * engine.LHV_KJ_KG), rel=1e-12
    )


def test_thermal_efficiency_is_shaft_power_over_chemical_power(eng):
    power = _sweep()
    chemical_kw = eng.fuel_flow_kg_s(power) * engine.LHV_KJ_KG
    assert eng.thermal_efficiency(power) == pytest.approx(power / chemical_kw, rel=1e-12)


def test_thermal_efficiency_approaches_marginal_efficiency_asymptotically(eng):
    """b/P vanishes as P grows, so average efficiency climbs toward marginal."""
    huge = Turboshaft(rated_power_kw=RATED_KW).thermal_efficiency(1.0e6)
    assert huge == pytest.approx(eng.marginal_efficiency, rel=1e-3)


# ---------------------------------------------------------------------------
# Calibration and scaling with engine size
# ---------------------------------------------------------------------------


def test_marginal_rate_is_size_invariant_and_parasitic_flow_scales_with_size():
    small, large = Turboshaft(50.0), Turboshaft(100.0)
    assert small.willans_a == pytest.approx(large.willans_a, rel=1e-12)
    assert large.willans_b == pytest.approx(2.0 * small.willans_b, rel=1e-12)


def test_bigger_engine_carries_a_bigger_idle_penalty_at_the_same_power():
    """The mechanism that makes load-levelling worth anything."""
    small, large = Turboshaft(50.0), Turboshaft(100.0)
    assert large.sfc_kg_kwh(30.0) > small.sfc_kg_kwh(30.0)


def test_zero_idle_fraction_collapses_to_constant_sfc():
    """Regression guard against the previous constant-SFC model."""
    flat = Turboshaft(RATED_KW, idle_fuel_fraction=0.0)
    assert flat.willans_b == 0.0
    assert flat.sfc_kg_kwh(_sweep()) == pytest.approx(flat.sfc_rated_kg_kwh, rel=1e-12)
    assert flat.thermal_efficiency(_sweep()) == pytest.approx(flat.marginal_efficiency, rel=1e-12)


def test_calibration_reproduces_rated_fuel_flow(eng):
    assert eng.fuel_flow_kg_s(RATED_KW) * 3600.0 == pytest.approx(
        eng.sfc_rated_kg_kwh * RATED_KW, rel=1e-12
    )


def test_idle_fuel_flow_is_the_stated_fraction_of_rated_flow(eng):
    assert eng.willans_b == pytest.approx(
        eng.idle_fuel_fraction * eng.sfc_rated_kg_kwh * RATED_KW, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------


def test_thermal_efficiency_at_rated_is_plausible_for_a_small_turboshaft(eng):
    """0.1856 for the default calibration - see the summary."""
    assert 0.18 <= eng.thermal_efficiency(RATED_KW) <= 0.25


def test_marginal_efficiency_is_plausible(eng):
    assert 0.20 <= eng.marginal_efficiency <= 0.35


def test_fuel_flow_is_non_negative_over_the_whole_power_range(eng):
    assert np.all(eng.fuel_flow_kg_s(_sweep(0.0, RATED_KW)) >= 0.0)


def test_sfc_is_infinite_at_zero_power_and_efficiency_is_zero(eng):
    assert eng.sfc_kg_kwh(0.0) == float("inf")
    assert eng.thermal_efficiency(0.0) == 0.0


def test_zero_power_is_handled_inside_an_array_too(eng):
    sfc = eng.sfc_kg_kwh(np.array([0.0, 10.0, RATED_KW]))
    assert np.isinf(sfc[0]) and np.all(np.isfinite(sfc[1:]))


# ---------------------------------------------------------------------------
# Altitude lapse
# ---------------------------------------------------------------------------


def test_no_lapse_at_sea_level(eng):
    assert eng.max_power_kw(1.0) == pytest.approx(RATED_KW, rel=1e-12)


def test_available_power_follows_sigma_to_the_lapse_exponent(eng):
    for sigma in (0.2, 0.3376, SIGMA_6KM, SIGMA_3KM, 1.0):
        assert eng.max_power_kw(sigma) == pytest.approx(RATED_KW * sigma**0.8, rel=1e-12)


def test_available_power_increases_strictly_with_sigma(eng):
    powers = [eng.max_power_kw(s) for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert all(a < b for a, b in zip(powers, powers[1:]))


def test_available_power_in_the_cruise_band(eng):
    """59.1 kW at 3 km and 45.7 kW at 6 km for a 75 kW engine - see the summary."""
    assert eng.max_power_kw(SIGMA_3KM) == pytest.approx(59.07, abs=0.02)
    assert eng.max_power_kw(SIGMA_6KM) == pytest.approx(45.68, abs=0.02)


def test_sfc_at_a_given_absolute_power_is_altitude_independent(eng):
    for sigma in (1.0, SIGMA_3KM, SIGMA_6KM):
        assert eng.fuel_flow_kg_s(30.0, sigma) == pytest.approx(
            eng.fuel_flow_kg_s(30.0, 1.0), rel=1e-12
        )


# ---------------------------------------------------------------------------
# Operating logic
# ---------------------------------------------------------------------------


def test_command_above_the_lapsed_rating_is_clamped_and_flagged(eng):
    state = eng.operate(90.0, sigma=SIGMA_6KM)
    assert state.power_limited
    assert state.delivered_kw == pytest.approx(eng.max_power_kw(SIGMA_6KM))
    assert state.load_fraction == pytest.approx(1.0)
    assert state.commanded_kw == 90.0


def test_command_within_the_envelope_is_delivered_untouched(eng):
    state = eng.operate(40.0, sigma=SIGMA_6KM)
    assert state.delivered_kw == pytest.approx(40.0)
    assert not (state.power_limited or state.at_idle or state.shut_down)
    assert state.load_fraction == pytest.approx(40.0 / eng.max_power_kw(SIGMA_6KM))


def test_idle_mode_holds_the_floor_and_keeps_burning(eng):
    state = eng.operate(2.0)
    assert state.at_idle and not state.shut_down
    assert state.delivered_kw == pytest.approx(eng.min_power_fraction * RATED_KW)
    assert state.fuel_flow_kg_s > 0.0
    assert state.load_fraction == pytest.approx(eng.min_power_fraction)


def test_shutdown_mode_delivers_nothing_and_burns_nothing():
    state = Turboshaft(RATED_KW, allow_shutdown=True).operate(2.0)
    assert state.shut_down and not state.at_idle
    assert state.delivered_kw == 0.0
    assert state.fuel_flow_kg_s == 0.0
    assert state.thermal_efficiency == 0.0
    assert state.load_fraction == 0.0


def test_commanding_exactly_at_the_floor_is_not_idle(eng):
    state = eng.operate(eng.idle_power_kw)
    assert not state.at_idle
    assert state.delivered_kw == pytest.approx(eng.idle_power_kw)


def test_the_two_low_power_modes_disagree_on_fuel(eng):
    """O-04 is open precisely because this difference is not small."""
    idling = eng.operate(2.0)
    stopped = Turboshaft(RATED_KW, allow_shutdown=True).operate(2.0)
    assert idling.fuel_flow_kg_s > stopped.fuel_flow_kg_s == 0.0


def test_the_idle_floor_never_exceeds_the_lapsed_rating():
    """At extreme altitude the floor is unreachable; delivered power must not lie."""
    thin = 0.05
    eng = Turboshaft(RATED_KW)
    state = eng.operate(1.0, sigma=thin)
    assert state.at_idle
    assert state.delivered_kw == pytest.approx(eng.max_power_kw(thin))
    assert state.delivered_kw < eng.idle_power_kw


def test_restart_is_detectable_from_consecutive_states():
    """The module is stateless, so the caller spots the off-to-on transition."""
    eng = Turboshaft(RATED_KW, allow_shutdown=True, restart_fuel_kg=0.05)
    off = eng.operate(2.0)
    on = eng.operate(40.0)
    assert off.shut_down and not on.shut_down
    assert eng.restart_fuel_kg == 0.05


def test_operate_is_stateless(eng):
    assert eng.operate(40.0) == eng.operate(40.0)
    eng.operate(2.0)
    assert eng.operate(40.0) == eng.operate(40.0, 1.0)


def test_engine_state_is_immutable(eng):
    with pytest.raises(dataclasses.FrozenInstanceError):
        eng.operate(40.0).delivered_kw = 0.0


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["fuel_flow_kg_s", "sfc_kg_kwh", "thermal_efficiency"])
def test_vectorized_results_match_element_wise_scalar_calls(method, eng):
    power = _sweep()
    call = getattr(eng, method)
    vector = call(power)
    assert isinstance(vector, np.ndarray) and vector.shape == power.shape
    assert vector == pytest.approx(np.array([call(float(p)) for p in power]), rel=1e-12)


@pytest.mark.parametrize("method", ["fuel_flow_kg_s", "sfc_kg_kwh", "thermal_efficiency"])
def test_scalar_input_returns_a_python_float(method, eng):
    assert type(getattr(eng, method)(40.0)) is float


def test_multidimensional_input_keeps_its_shape(eng):
    power = np.linspace(1.0, RATED_KW, 12).reshape(3, 4)
    assert eng.fuel_flow_kg_s(power).shape == (3, 4)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rated", [0.0, -75.0])
def test_non_positive_rated_power_is_rejected(rated):
    with pytest.raises(ValueError, match="rated_power_kw"):
        Turboshaft(rated_power_kw=rated)


@pytest.mark.parametrize("fraction", [1.0, 1.5, -0.1])
def test_idle_fuel_fraction_outside_the_unit_interval_is_rejected(fraction):
    with pytest.raises(ValueError, match="idle_fuel_fraction"):
        Turboshaft(RATED_KW, idle_fuel_fraction=fraction)


@pytest.mark.parametrize("sigma", [0.0, -0.5, 1.0001, 2.0])
def test_sigma_outside_zero_to_one_is_rejected(sigma, eng):
    with pytest.raises(ValueError, match="sigma"):
        eng.max_power_kw(sigma)
    with pytest.raises(ValueError, match="sigma"):
        eng.operate(40.0, sigma)
    with pytest.raises(ValueError, match="sigma"):
        eng.fuel_flow_kg_s(40.0, sigma)


def test_non_positive_rated_sfc_is_rejected():
    with pytest.raises(ValueError, match="sfc_rated_kg_kwh"):
        Turboshaft(RATED_KW, sfc_rated_kg_kwh=0.0)


# ---------------------------------------------------------------------------
# Comparison against the constant-SFC model this one replaces
# ---------------------------------------------------------------------------


def test_part_load_penalty_at_the_loiter_point(eng):
    """35 kW on a 75 kW engine: 0.553 vs 0.300 kg/kWh, a factor of 1.84."""
    willans_sfc = float(eng.sfc_kg_kwh(LOITER_KW))
    assert willans_sfc == pytest.approx(0.5529, abs=1e-4)
    assert willans_sfc / PREVIOUS_MODEL_SFC == pytest.approx(1.843, abs=1e-3)

    willans_kg_h = float(eng.fuel_flow_kg_s(LOITER_KW)) * 3600.0
    assert willans_kg_h == pytest.approx(19.35, abs=1e-2)
    assert PREVIOUS_MODEL_SFC * LOITER_KW == pytest.approx(10.50, abs=1e-2)


def test_the_penalty_grows_as_the_engine_is_throttled_back(eng):
    """Load-levelling pays because this ratio is not flat."""
    ratios = [float(eng.sfc_kg_kwh(p)) / PREVIOUS_MODEL_SFC for p in (75.0, 50.0, 35.0, 20.0)]
    assert all(a < b for a, b in zip(ratios, ratios[1:]))
