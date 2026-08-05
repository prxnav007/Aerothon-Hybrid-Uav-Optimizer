"""Tests for :mod:`src.models.powertrain` (series conversion chain).

Six tiers: the chain identities and their inverses, the DC bus balance and its
loss accounting, the load-dependent loss model, the comparisons quoted in the
report, vectorization, and guards. The regression at the end pins the engine
power required against the previous implementation, which divided by motor
efficiency alone and took generator bus output for engine shaft power.
"""

import dataclasses

import numpy as np
import pytest

from src.models.powertrain import SeriesPowertrain

SHAFT_KW = 30.0
RATED_ENGINE_KW = 75.0
RATED_BUS_KW = 90.0

SOURCE_CHAIN = 0.95 * 0.95
DEMAND_CHAIN = 0.95 * 0.95 * 0.99


@pytest.fixture
def chain() -> SeriesPowertrain:
    return SeriesPowertrain()


@pytest.fixture
def loaded() -> SeriesPowertrain:
    return SeriesPowertrain(
        load_dependent=True, rated_engine_kw=RATED_ENGINE_KW, rated_bus_kw=RATED_BUS_KW
    )


def _power_sweep(n: int = 25) -> np.ndarray:
    return np.linspace(5.0, RATED_ENGINE_KW, n)


# ---------------------------------------------------------------------------
# Chain identities
# ---------------------------------------------------------------------------


def test_demand_chain_spends_inverter_motor_and_cabling(chain):
    assert float(chain.bus_power_required(SHAFT_KW)) == pytest.approx(
        SHAFT_KW / DEMAND_CHAIN, rel=1e-12
    )


def test_source_chain_spends_generator_and_rectifier(chain):
    assert float(chain.bus_power_from_engine(SHAFT_KW)) == pytest.approx(
        SHAFT_KW * SOURCE_CHAIN, rel=1e-12
    )


def test_chain_efficiency_is_the_product_of_all_five_stages(chain):
    assert chain.source_chain_efficiency == pytest.approx(SOURCE_CHAIN, rel=1e-12)
    assert chain.demand_chain_efficiency == pytest.approx(DEMAND_CHAIN, rel=1e-12)
    assert chain.chain_efficiency == pytest.approx(
        chain.eta_generator
        * chain.eta_rectifier
        * chain.eta_inverter
        * chain.eta_motor
        * chain.eta_cabling,
        rel=1e-12,
    )


def test_cabling_is_the_only_new_stage_against_the_bare_four(chain):
    bare = SeriesPowertrain(eta_cabling=1.0)
    assert chain.chain_efficiency == pytest.approx(bare.chain_efficiency * 0.99, rel=1e-12)


@pytest.mark.parametrize("mode", ["constant", "load_dependent"])
def test_source_conversion_round_trips(chain, loaded, mode):
    subject = chain if mode == "constant" else loaded
    power = _power_sweep()
    assert subject.engine_power_for_bus(subject.bus_power_from_engine(power)) == pytest.approx(
        power, rel=1e-9
    )


@pytest.mark.parametrize("mode", ["constant", "load_dependent"])
def test_demand_conversion_round_trips(chain, loaded, mode):
    subject = chain if mode == "constant" else loaded
    power = _power_sweep()
    assert subject.shaft_power_from_bus(subject.bus_power_required(power)) == pytest.approx(
        power, rel=1e-9
    )


def test_constant_mode_round_trips_to_machine_precision(chain):
    power = _power_sweep()
    assert chain.engine_power_for_bus(chain.bus_power_from_engine(power)) == pytest.approx(
        power, rel=1e-12
    )
    assert chain.shaft_power_from_bus(chain.bus_power_required(power)) == pytest.approx(
        power, rel=1e-12
    )


def test_every_conversion_costs_something(chain):
    assert float(chain.bus_power_required(SHAFT_KW)) > SHAFT_KW
    assert float(chain.bus_power_from_engine(SHAFT_KW)) < SHAFT_KW
    assert float(chain.engine_power_for_bus(SHAFT_KW)) > SHAFT_KW
    assert float(chain.shaft_power_from_bus(SHAFT_KW)) < SHAFT_KW


def test_peak_bus_power_is_the_demand_chain_at_the_worst_case(chain):
    assert chain.peak_bus_power(120.0) == pytest.approx(120.0 / DEMAND_CHAIN, rel=1e-12)


# ---------------------------------------------------------------------------
# Bus balance
# ---------------------------------------------------------------------------


def _balanced_split(chain: SeriesPowertrain, shaft_kw: float, battery_kw: float):
    """Engine shaft power that closes the bus for a given battery contribution."""
    bus_demand_kw = float(chain.bus_power_required(shaft_kw))
    return float(chain.engine_power_for_bus(bus_demand_kw - battery_kw))


@pytest.mark.parametrize("battery_kw", [-8.0, 0.0, 5.0, 20.0])
def test_a_balanced_split_reports_no_residual(chain, battery_kw):
    engine_kw = _balanced_split(chain, SHAFT_KW, battery_kw)
    state = chain.solve(SHAFT_KW, engine_kw, battery_kw)

    assert state.balanced
    assert state.bus_residual_kw == pytest.approx(0.0, abs=1e-9)


def test_an_under_supplied_split_reports_a_negative_residual(chain):
    engine_kw = _balanced_split(chain, SHAFT_KW, 0.0)
    state = chain.solve(SHAFT_KW, 0.5 * engine_kw, 0.0)

    assert not state.balanced
    assert state.bus_residual_kw < 0.0


def test_an_over_supplied_split_reports_a_positive_residual(chain):
    engine_kw = _balanced_split(chain, SHAFT_KW, 0.0)
    state = chain.solve(SHAFT_KW, engine_kw, 5.0)

    assert not state.balanced
    assert state.bus_residual_kw == pytest.approx(5.0, rel=1e-9)


def test_the_residual_is_computed_not_assumed(chain):
    """The previous shortfall check could only fire on the all-zero fallback."""
    residuals = [chain.solve(SHAFT_KW, engine_kw, 0.0).bus_residual_kw for engine_kw in (0.0, 20.0, 60.0)]
    assert residuals[0] < residuals[1] < residuals[2]
    assert len(set(residuals)) == 3


def test_an_imbalance_is_reported_never_raised(chain):
    assert not chain.solve(SHAFT_KW, 0.0, 0.0).balanced
    assert not chain.solve(SHAFT_KW, 1e6, 0.0).balanced


def test_demand_side_losses_close_against_the_bus_draw(chain):
    state = chain.solve(SHAFT_KW, _balanced_split(chain, SHAFT_KW, 0.0), 0.0)
    assert state.bus_demand_kw == pytest.approx(
        state.shaft_demand_kw + state.demand_losses_kw, rel=1e-12
    )


def test_source_side_losses_close_against_the_engine_shaft(chain):
    state = chain.solve(SHAFT_KW, _balanced_split(chain, SHAFT_KW, 0.0), 0.0)
    assert state.engine_shaft_kw == pytest.approx(
        state.bus_from_engine_kw + state.source_losses_kw, rel=1e-12
    )


@pytest.mark.parametrize("battery_kw", [-8.0, 0.0, 5.0, 20.0])
def test_a_balanced_split_conserves_power_end_to_end(chain, battery_kw):
    """Everything in equals shaft power out plus every loss along the way."""
    engine_kw = _balanced_split(chain, SHAFT_KW, battery_kw)
    state = chain.solve(SHAFT_KW, engine_kw, battery_kw)

    assert state.engine_shaft_kw + state.battery_bus_kw == pytest.approx(
        state.shaft_demand_kw + state.total_losses_kw, rel=1e-9
    )
    assert state.total_losses_kw == pytest.approx(
        state.source_losses_kw + state.demand_losses_kw, rel=1e-12
    )


def test_reported_efficiencies_match_the_chain(chain):
    state = chain.solve(SHAFT_KW, _balanced_split(chain, SHAFT_KW, 0.0), 0.0)
    assert state.source_efficiency == pytest.approx(SOURCE_CHAIN, rel=1e-12)
    assert state.demand_efficiency == pytest.approx(DEMAND_CHAIN, rel=1e-12)


def test_efficiencies_are_zero_rather_than_undefined_at_zero_flow(chain):
    state = chain.solve(0.0, 0.0, 0.0)
    assert state.source_efficiency == 0.0
    assert state.demand_efficiency == 0.0
    assert state.balanced


def test_state_is_immutable(chain):
    with pytest.raises(dataclasses.FrozenInstanceError):
        chain.solve(SHAFT_KW, 40.0, 0.0).balanced = True


def test_solve_is_stateless(chain):
    assert chain.solve(SHAFT_KW, 40.0, 2.0) == chain.solve(SHAFT_KW, 40.0, 2.0)


# ---------------------------------------------------------------------------
# Load-dependent mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("eta_rated", "rated_kw"),
    [(0.95, RATED_ENGINE_KW), (0.95, RATED_BUS_KW), (0.92, 40.0), (0.98, 200.0)],
)
def test_efficiency_recovers_the_rated_value_at_the_rating(eta_rated, rated_kw):
    chain = SeriesPowertrain(
        load_dependent=True, rated_engine_kw=RATED_ENGINE_KW, rated_bus_kw=RATED_BUS_KW
    )
    assert float(chain.stage_efficiency(rated_kw, eta_rated, rated_kw)) == pytest.approx(
        eta_rated, rel=1e-12
    )


def test_efficiency_peaks_below_rated_where_the_two_loss_terms_balance(loaded):
    """The proposal expected a monotonic rise to rated; this model does not.

    Efficiency is maximised where the no-load loss equals the load loss, at
    sqrt(f / (1 - f)) of the rating - 65.5% of rated at the default f = 0.30.
    Above that point efficiency falls back to the rated value.
    """
    peak_fraction = np.sqrt(
        loaded.noload_loss_fraction / (1.0 - loaded.noload_loss_fraction)
    )
    assert peak_fraction == pytest.approx(0.65465, abs=1e-5)

    fractions = np.linspace(0.02, 1.0, 4000)
    eta = np.asarray(loaded.stage_efficiency(fractions * RATED_BUS_KW, 0.95, RATED_BUS_KW))

    assert fractions[np.argmax(eta)] == pytest.approx(peak_fraction, abs=2e-3)
    assert eta.max() > 0.95
    rising, falling = eta[fractions < peak_fraction], eta[fractions > peak_fraction]
    assert np.all(np.diff(rising) > 0.0)
    assert np.all(np.diff(falling) < 0.0)


def test_efficiency_rises_with_load_across_the_operating_band(loaded):
    """Monotonic below the peak, which is where the vehicle actually operates."""
    eta = [
        float(loaded.stage_efficiency(fraction * RATED_BUS_KW, 0.95, RATED_BUS_KW))
        for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    ]
    assert all(low < high for low, high in zip(eta, eta[1:]))


def test_part_load_is_less_efficient_than_the_rated_value(loaded):
    assert float(loaded.stage_efficiency(0.33 * RATED_BUS_KW, 0.95, RATED_BUS_KW)) < 0.95


def test_a_zero_no_load_fraction_leaves_pure_load_loss():
    chain = SeriesPowertrain(
        load_dependent=True,
        noload_loss_fraction=0.0,
        rated_engine_kw=RATED_ENGINE_KW,
        rated_bus_kw=RATED_BUS_KW,
    )
    eta = [
        float(chain.stage_efficiency(fraction * RATED_BUS_KW, 0.95, RATED_BUS_KW))
        for fraction in (0.2, 0.5, 1.0)
    ]
    assert all(low > high for low, high in zip(eta, eta[1:]))
    assert eta[-1] == pytest.approx(0.95, rel=1e-12)


def test_a_stage_below_its_no_load_loss_delivers_nothing(loaded):
    no_load_kw = loaded.noload_loss_fraction * RATED_ENGINE_KW * (1.0 / 0.95 - 1.0)
    assert float(loaded.bus_power_from_engine(0.5 * no_load_kw)) == 0.0


def test_load_dependent_mode_needs_both_ratings():
    with pytest.raises(ValueError, match="rated_engine_kw"):
        SeriesPowertrain(load_dependent=True, rated_bus_kw=RATED_BUS_KW)
    with pytest.raises(ValueError, match="rated_bus_kw"):
        SeriesPowertrain(load_dependent=True, rated_engine_kw=RATED_ENGINE_KW)


@pytest.mark.parametrize("rating", [0.0, -75.0])
def test_load_dependent_mode_rejects_a_non_positive_rating(rating):
    with pytest.raises(ValueError, match="rated_engine_kw"):
        SeriesPowertrain(
            load_dependent=True, rated_engine_kw=rating, rated_bus_kw=RATED_BUS_KW
        )


def test_ratings_are_ignored_when_the_mode_is_off():
    assert SeriesPowertrain(rated_engine_kw=None).chain_efficiency == pytest.approx(
        SeriesPowertrain().chain_efficiency
    )


# ---------------------------------------------------------------------------
# Reported comparisons
# ---------------------------------------------------------------------------


def test_loiter_demand_chain_constant_against_load_dependent(chain, loaded):
    """30 kW shaft on a 90 kW bus rating - a third of rated, where it hurts."""
    constant_bus_kw = float(chain.bus_power_required(SHAFT_KW))
    loaded_bus_kw = float(loaded.bus_power_required(SHAFT_KW))

    assert constant_bus_kw == pytest.approx(33.5768, abs=1e-4)
    assert loaded_bus_kw == pytest.approx(33.9638, abs=1e-4)
    assert loaded_bus_kw / constant_bus_kw - 1.0 == pytest.approx(0.01153, abs=1e-5)

    assert SHAFT_KW / constant_bus_kw == pytest.approx(0.89347, abs=1e-5)
    assert SHAFT_KW / loaded_bus_kw == pytest.approx(0.88329, abs=1e-5)


def test_loiter_compounded_chain_constant_against_load_dependent(chain, loaded):
    """The source stages sit nearer their peak, so the total moves less."""
    constant_engine_kw = float(chain.engine_power_for_bus(chain.bus_power_required(SHAFT_KW)))
    loaded_engine_kw = float(loaded.engine_power_for_bus(loaded.bus_power_required(SHAFT_KW)))

    assert SHAFT_KW / constant_engine_kw == pytest.approx(0.80636, abs=1e-5)
    assert SHAFT_KW / loaded_engine_kw == pytest.approx(0.79946, abs=1e-5)
    assert constant_engine_kw == pytest.approx(37.2042, abs=1e-4)
    assert loaded_engine_kw == pytest.approx(37.5255, abs=1e-4)


def test_engine_power_against_the_previous_implementation(chain):
    """It divided by motor efficiency alone and took bus output for shaft power."""
    previous_engine_kw = SHAFT_KW / 0.95
    correct_engine_kw = float(chain.engine_power_for_bus(chain.bus_power_required(SHAFT_KW)))

    assert previous_engine_kw == pytest.approx(31.5789, abs=1e-4)
    assert correct_engine_kw == pytest.approx(37.2042, abs=1e-4)
    assert correct_engine_kw / previous_engine_kw == pytest.approx(1.17813, abs=1e-5)
    assert 1.0 - previous_engine_kw / correct_engine_kw == pytest.approx(0.1512, abs=1e-4)


# ---------------------------------------------------------------------------
# System efficiency
# ---------------------------------------------------------------------------


def test_system_efficiency_is_output_over_fuel_plus_discharge(chain):
    assert chain.system_efficiency(25.0, 100.0, 10.0) == pytest.approx(25.0 / 110.0, rel=1e-12)


def test_charging_is_not_an_input_to_the_system(chain):
    """A negative battery power must not be subtracted from the denominator."""
    assert chain.system_efficiency(25.0, 100.0, -10.0) == pytest.approx(0.25, rel=1e-12)
    assert chain.system_efficiency(25.0, 100.0, -10.0) == chain.system_efficiency(25.0, 100.0, 0.0)


@pytest.mark.parametrize("fuel_kw", [0.0, -5.0])
def test_system_efficiency_guards_the_denominator(chain, fuel_kw):
    assert chain.system_efficiency(25.0, fuel_kw, 0.0) == 0.0


def test_system_efficiency_on_battery_alone(chain):
    assert chain.system_efficiency(25.0, 0.0, 40.0) == pytest.approx(0.625, rel=1e-12)


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------


CONVERSIONS = [
    "bus_power_required",
    "bus_power_from_engine",
    "engine_power_for_bus",
    "shaft_power_from_bus",
]


@pytest.mark.parametrize("method", CONVERSIONS)
@pytest.mark.parametrize("mode", ["constant", "load_dependent"])
def test_vectorized_results_match_element_wise_scalar_calls(chain, loaded, method, mode):
    subject = chain if mode == "constant" else loaded
    power = _power_sweep()
    call = getattr(subject, method)

    vector = call(power)
    assert isinstance(vector, np.ndarray) and vector.shape == power.shape
    assert vector == pytest.approx(np.array([call(float(p)) for p in power]), rel=1e-12)


@pytest.mark.parametrize("method", CONVERSIONS)
def test_scalar_input_returns_a_python_float(chain, method):
    assert type(getattr(chain, method)(SHAFT_KW)) is float


@pytest.mark.parametrize("method", CONVERSIONS)
def test_multidimensional_input_keeps_its_shape(loaded, method):
    power = np.linspace(5.0, RATED_ENGINE_KW, 12).reshape(3, 4)
    assert np.asarray(getattr(loaded, method)(power)).shape == (3, 4)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0.0, -0.5, 1.01, 2.0])
@pytest.mark.parametrize(
    "name", ["eta_generator", "eta_rectifier", "eta_inverter", "eta_motor", "eta_cabling"]
)
def test_efficiency_outside_the_unit_interval_is_rejected(name, eta):
    with pytest.raises(ValueError, match=name):
        SeriesPowertrain(**{name: eta})


@pytest.mark.parametrize("fraction", [1.0, 1.5, -0.1])
def test_no_load_fraction_outside_its_range_is_rejected(fraction):
    with pytest.raises(ValueError, match="noload_loss_fraction"):
        SeriesPowertrain(noload_loss_fraction=fraction)


def test_unit_efficiency_is_allowed():
    assert SeriesPowertrain(eta_cabling=1.0).eta_cabling == 1.0
