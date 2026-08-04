"""Tests for :mod:`src.models.battery` (Rint pack on the DC bus).

Verification is in six tiers: energy closure between the rated capacity and the
coulomb count, the quadratic current solution that replaces P/V_oc, resistance
scaling with pack size, the C-rate and state-of-charge limits, the sign
convention, and vectorization. A final test pins the power limit to capacity
rather than to engine rating, which is what the previous model got wrong.
"""

import dataclasses

import numpy as np
import pytest

from src.models.battery import BatteryPack

CAPACITY_KWH = 10.0
SOC_MID = 0.5
BUS_KW = 30.0
DT_S = 60.0

PREVIOUS_MODEL_ENGINE_KW = 75.0
SMALL_PACK_KWH = 5.0


@pytest.fixture
def pack() -> BatteryPack:
    return BatteryPack(capacity_kwh=CAPACITY_KWH)


def _naive_current_a(pack: BatteryPack, power_kw: float, soc: float) -> float:
    """The P/V_oc approximation this model replaces."""
    return power_kw * 1000.0 / float(pack.open_circuit_voltage(soc))


def _discharge_sweep(pack: BatteryPack, n: int = 25) -> np.ndarray:
    return np.linspace(0.1, pack.max_discharge_kw, n)


def _charge_sweep(pack: BatteryPack, n: int = 25) -> np.ndarray:
    return -np.linspace(0.1, pack.max_charge_kw, n)


# ---------------------------------------------------------------------------
# Energy closure
# ---------------------------------------------------------------------------


def test_charge_capacity_recovers_rated_energy_at_nominal_voltage(pack):
    assert pack.charge_capacity_ah * pack.nominal_voltage_v / 1000.0 == pytest.approx(
        CAPACITY_KWH, rel=1e-12
    )


def test_nominal_voltage_is_the_midpoint_of_the_linear_ocv(pack):
    assert pack.nominal_voltage_v == pytest.approx(float(pack.open_circuit_voltage(0.5)))


def test_full_discharge_at_negligible_power_delivers_the_rated_energy():
    """Validates the Ah derivation against the linear OCV model."""
    pack = BatteryPack(CAPACITY_KWH, soc_min=0.0)
    trickle_kw = 0.5

    soc, delivered_kwh = 1.0, 0.0
    while soc > 0.0:
        state = pack.step(soc, trickle_kw, DT_S)
        delivered_kwh += state.power_kw * DT_S / 3600.0
        soc = state.soc

    assert delivered_kwh == pytest.approx(CAPACITY_KWH, rel=5e-3)


def test_open_circuit_voltage_spans_the_stated_range(pack):
    assert float(pack.open_circuit_voltage(0.0)) == pytest.approx(pack.v_min_v)
    assert float(pack.open_circuit_voltage(1.0)) == pytest.approx(pack.v_max_v)


# ---------------------------------------------------------------------------
# Quadratic current solution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("soc", [0.1, 0.5, 0.9, 1.0])
def test_current_satisfies_the_rint_power_balance(pack, soc):
    """P_bus = V_oc*I - I**2*R, signed current, on both sides of zero power."""
    power = np.concatenate([_charge_sweep(pack), _discharge_sweep(pack)])
    current = pack.current_from_power(power, soc)
    v_oc = float(pack.open_circuit_voltage(soc))
    r = pack.internal_resistance_ohm
    assert (v_oc * current - current**2 * r) / 1000.0 == pytest.approx(power, rel=1e-9)


@pytest.mark.parametrize("soc", [0.1, 0.5, 0.9])
def test_charging_current_flows_against_the_rising_terminal_voltage(pack, soc):
    """|P_bus| = (V_oc + |I|*R) * |I| is the charging analogue."""
    power = _charge_sweep(pack)
    magnitude = -np.asarray(pack.current_from_power(power, soc))
    v_oc = float(pack.open_circuit_voltage(soc))
    r = pack.internal_resistance_ohm
    assert np.all(magnitude > 0.0)
    assert (v_oc + magnitude * r) * magnitude / 1000.0 == pytest.approx(-power, rel=1e-9)


def test_current_exceeds_the_naive_power_over_ocv_estimate(pack):
    """86.790 A against 85.714 A at 30 kW on a 10 kWh pack, 1.26% higher."""
    power = _discharge_sweep(pack)
    naive = power * 1000.0 / float(pack.open_circuit_voltage(SOC_MID))
    assert np.all(pack.current_from_power(power, SOC_MID) > naive)

    solved = float(pack.current_from_power(BUS_KW, SOC_MID))
    assert solved == pytest.approx(86.7904, abs=1e-4)
    assert _naive_current_a(pack, BUS_KW, SOC_MID) == pytest.approx(85.7143, abs=1e-4)
    assert solved / _naive_current_a(pack, BUS_KW, SOC_MID) == pytest.approx(1.01255, abs=1e-5)


def test_ohmic_loss_at_the_reference_condition(pack):
    """0.377 kW at 30 kW on a 10 kWh pack at soc = 0.5 - the loss the old model dropped."""
    assert float(pack.ohmic_loss_kw(BUS_KW, SOC_MID)) == pytest.approx(0.37663, abs=1e-5)
    assert float(pack.terminal_voltage(BUS_KW, SOC_MID)) == pytest.approx(345.660, abs=1e-3)


def test_ohmic_loss_is_zero_at_zero_power_and_positive_otherwise(pack):
    assert float(pack.ohmic_loss_kw(0.0, SOC_MID)) == 0.0
    both_ways = np.concatenate([_charge_sweep(pack), _discharge_sweep(pack)])
    assert np.all(pack.ohmic_loss_kw(both_ways, SOC_MID) > 0.0)


def test_ohmic_loss_is_quadratic_in_power_at_fixed_soc(pack):
    low = float(pack.ohmic_loss_kw(1.0, SOC_MID))
    assert float(pack.ohmic_loss_kw(2.0, SOC_MID)) / low == pytest.approx(4.0, rel=1e-2)


def test_ohmic_loss_rises_with_falling_state_of_charge(pack):
    """Same bus power costs more current as the pack sags toward v_min."""
    loss = pack.ohmic_loss_kw(BUS_KW, np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    assert np.all(np.diff(loss) < 0.0)


def test_power_above_the_ohmic_ceiling_is_clamped_and_never_nan():
    """The C-rate limit normally binds first, so the guard needs an absurd C-rate."""
    pack = BatteryPack(CAPACITY_KWH, discharge_c_rate=200.0)
    ceiling_kw = float(pack.ohmic_power_ceiling_kw(1.0))
    assert ceiling_kw == pytest.approx(800.0)

    state = pack.step(1.0, 2.0 * ceiling_kw, DT_S)
    assert state.power_limited
    assert state.power_kw == pytest.approx(ceiling_kw)
    assert np.isfinite(state.current_a)
    # Maximum power transfer sits at half the open-circuit voltage.
    assert state.current_a == pytest.approx(
        pack.v_max_v / (2.0 * pack.internal_resistance_ohm)
    )


def test_raw_current_stays_finite_beyond_the_ceiling(pack):
    beyond = np.array([1.0, 1.0e3, 1.0e6]) * float(pack.ohmic_power_ceiling_kw(SOC_MID))
    assert np.all(np.isfinite(pack.current_from_power(beyond, SOC_MID)))


# ---------------------------------------------------------------------------
# Resistance scaling
# ---------------------------------------------------------------------------


def test_halving_capacity_doubles_internal_resistance():
    small, large = BatteryPack(5.0), BatteryPack(10.0)
    assert small.internal_resistance_ohm == pytest.approx(2.0 * large.internal_resistance_ohm)


def test_resistance_at_the_reference_capacity_is_the_reference_value(pack):
    assert pack.internal_resistance_ohm == pytest.approx(pack.r_ref_ohm)


@pytest.mark.parametrize("capacity", [1.0, 5.0, 10.0, 50.0])
def test_unscaled_resistance_ignores_capacity(capacity):
    """The previous behaviour, retained for comparison - see O-05."""
    fixed = BatteryPack(capacity, scale_resistance=False)
    assert fixed.internal_resistance_ohm == fixed.r_ref_ohm


def test_a_bigger_pack_loses_less_at_the_same_bus_power():
    small, large = BatteryPack(5.0), BatteryPack(50.0)
    assert float(large.ohmic_loss_kw(15.0, SOC_MID)) < float(small.ohmic_loss_kw(15.0, SOC_MID))


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_power_limits_are_the_c_rate_times_capacity(pack):
    assert pack.max_discharge_kw == pytest.approx(pack.discharge_c_rate * CAPACITY_KWH)
    assert pack.max_charge_kw == pytest.approx(pack.charge_c_rate * CAPACITY_KWH)


@pytest.mark.parametrize("soc", [0.0, 0.01, 0.05])
def test_no_discharge_at_or_below_the_hard_cutoff(pack, soc):
    assert pack.available_discharge_kw(soc) == 0.0


def test_discharge_is_available_above_the_cutoff(pack):
    assert pack.available_discharge_kw(0.06) == pytest.approx(pack.max_discharge_kw)


def test_no_charge_at_full_state_of_charge(pack):
    assert pack.available_charge_kw(1.0) == 0.0
    assert pack.available_charge_kw(0.99) == pytest.approx(pack.max_charge_kw)


def test_command_above_the_discharge_limit_is_clamped_and_flagged(pack):
    state = pack.step(SOC_MID, 45.0, DT_S)
    assert state.power_limited
    assert state.power_kw == pytest.approx(pack.max_discharge_kw)
    assert state.commanded_kw == 45.0


def test_command_above_the_charge_limit_is_clamped_and_flagged(pack):
    state = pack.step(SOC_MID, -25.0, DT_S)
    assert state.power_limited
    assert state.power_kw == pytest.approx(-pack.max_charge_kw)


def test_command_within_the_envelope_is_delivered_untouched(pack):
    state = pack.step(SOC_MID, 20.0, DT_S)
    assert not state.power_limited
    assert state.power_kw == 20.0


def test_discharge_below_the_cutoff_is_locked_out(pack):
    state = pack.step(pack.soc_min, BUS_KW, DT_S)
    assert state.at_cutoff and state.power_limited
    assert state.power_kw == 0.0
    assert state.soc == pytest.approx(pack.soc_min)


def test_charging_is_still_allowed_at_the_cutoff(pack):
    state = pack.step(pack.soc_min, -5.0, DT_S)
    assert state.power_kw == -5.0
    assert state.soc > pack.soc_min


def test_below_safe_floor_marks_the_recommended_operating_band(pack):
    """True between the hard cutoff and the floor, false above; read off the new soc."""
    assert pack.step(0.10, 0.1, DT_S).below_safe_floor
    assert not pack.step(0.21, 0.1, DT_S).below_safe_floor
    assert not pack.step(SOC_MID, 0.1, DT_S).below_safe_floor


def test_a_state_of_charge_clamp_is_flagged_not_absorbed():
    """A step long enough to run the pack flat must not report an honest result."""
    pack = BatteryPack(CAPACITY_KWH, soc_min=0.0)
    state = pack.step(0.02, pack.max_discharge_kw, 3600.0)
    assert state.soc == 0.0
    assert state.power_limited and state.at_cutoff


def test_state_of_charge_never_leaves_the_unit_interval(pack):
    assert pack.step(0.999, -pack.max_charge_kw, 3600.0).soc == 1.0


# ---------------------------------------------------------------------------
# Round-trip efficiency
# ---------------------------------------------------------------------------


def test_round_trip_efficiency_is_plausible_at_moderate_power(pack):
    """0.9755 at 30 kW on a 10 kWh pack at soc = 0.5 - see the summary."""
    eta = pack.round_trip_efficiency(BUS_KW, SOC_MID)
    assert 0.95 <= eta <= 0.99
    assert eta == pytest.approx(0.97551, abs=1e-5)


def test_round_trip_efficiency_falls_as_power_rises(pack):
    etas = [pack.round_trip_efficiency(p, SOC_MID) for p in (5.0, 15.0, 30.0)]
    assert all(a > b for a, b in zip(etas, etas[1:]))


def test_round_trip_efficiency_is_the_terminal_voltage_ratio(pack):
    current = float(pack.current_from_power(BUS_KW, SOC_MID))
    sag = current * pack.internal_resistance_ohm
    v_oc = float(pack.open_circuit_voltage(SOC_MID))
    assert pack.round_trip_efficiency(BUS_KW, SOC_MID) == pytest.approx(
        (v_oc - sag) / (v_oc + sag), rel=1e-12
    )


def test_round_trip_efficiency_ignores_the_sign_of_the_argument(pack):
    assert pack.round_trip_efficiency(-BUS_KW, SOC_MID) == pytest.approx(
        pack.round_trip_efficiency(BUS_KW, SOC_MID), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------


def test_positive_power_discharges_and_negative_power_charges(pack):
    assert pack.step(SOC_MID, 20.0, DT_S).soc < SOC_MID
    assert pack.step(SOC_MID, -8.0, DT_S).soc > SOC_MID


def test_current_is_positive_on_discharge_and_negative_on_charge(pack):
    assert pack.step(SOC_MID, 20.0, DT_S).current_a > 0.0
    assert pack.step(SOC_MID, -8.0, DT_S).current_a < 0.0


def test_terminal_voltage_sags_on_discharge_and_rises_on_charge(pack):
    discharging = pack.step(SOC_MID, 20.0, DT_S)
    charging = pack.step(SOC_MID, -8.0, DT_S)
    assert discharging.terminal_voltage_v < discharging.open_circuit_voltage_v
    assert charging.terminal_voltage_v > charging.open_circuit_voltage_v


def test_zero_power_moves_nothing(pack):
    state = pack.step(SOC_MID, 0.0, DT_S)
    assert state.soc == pytest.approx(SOC_MID)
    assert state.current_a == 0.0
    assert state.ohmic_loss_kw == 0.0
    assert state.terminal_voltage_v == pytest.approx(state.open_circuit_voltage_v)


def test_the_bus_sees_less_than_the_pack_gives_and_more_than_the_pack_takes(pack):
    """Losses land on the right side of the bus in both directions."""
    discharging = pack.step(SOC_MID, 20.0, DT_S)
    internal_kw = discharging.open_circuit_voltage_v * discharging.current_a / 1000.0
    assert discharging.power_kw == pytest.approx(internal_kw - discharging.ohmic_loss_kw)

    charging = pack.step(SOC_MID, -8.0, DT_S)
    stored_kw = -charging.open_circuit_voltage_v * charging.current_a / 1000.0
    assert -charging.power_kw == pytest.approx(stored_kw + charging.ohmic_loss_kw)


def test_step_is_stateless(pack):
    assert pack.step(SOC_MID, 20.0, DT_S) == pack.step(SOC_MID, 20.0, DT_S)
    pack.step(0.1, pack.max_discharge_kw, 3600.0)
    assert pack.step(SOC_MID, 20.0, DT_S) == pack.step(SOC_MID, 20.0, DT_S)


def test_battery_state_is_immutable(pack):
    with pytest.raises(dataclasses.FrozenInstanceError):
        pack.step(SOC_MID, 20.0, DT_S).soc = 0.0


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------


def test_open_circuit_voltage_is_vectorized_over_soc(pack):
    soc = np.linspace(0.0, 1.0, 11)
    vector = pack.open_circuit_voltage(soc)
    assert isinstance(vector, np.ndarray) and vector.shape == soc.shape
    assert vector == pytest.approx(
        np.array([float(pack.open_circuit_voltage(float(s))) for s in soc]), rel=1e-12
    )


@pytest.mark.parametrize("method", ["current_from_power", "ohmic_loss_kw", "terminal_voltage"])
def test_vectorized_results_match_element_wise_scalar_calls(method, pack):
    power = np.concatenate([_charge_sweep(pack), _discharge_sweep(pack)])
    call = getattr(pack, method)
    vector = call(power, SOC_MID)
    assert isinstance(vector, np.ndarray) and vector.shape == power.shape
    assert vector == pytest.approx(
        np.array([call(float(p), SOC_MID) for p in power]), rel=1e-12
    )


@pytest.mark.parametrize("method", ["current_from_power", "ohmic_loss_kw", "terminal_voltage"])
def test_scalar_input_returns_a_python_float(method, pack):
    assert type(getattr(pack, method)(BUS_KW, SOC_MID)) is float


def test_power_and_soc_broadcast_against_each_other(pack):
    power = np.linspace(1.0, BUS_KW, 4).reshape(4, 1)
    soc = np.linspace(0.2, 0.9, 3).reshape(1, 3)
    assert np.asarray(pack.current_from_power(power, soc)).shape == (4, 3)


def test_multidimensional_input_keeps_its_shape(pack):
    power = np.linspace(1.0, BUS_KW, 12).reshape(3, 4)
    assert np.asarray(pack.ohmic_loss_kw(power, SOC_MID)).shape == (3, 4)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [0.0, -5.0])
def test_non_positive_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="capacity_kwh"):
        BatteryPack(capacity_kwh=capacity)


@pytest.mark.parametrize("v_min", [400.0, 450.0])
def test_v_min_not_below_v_max_is_rejected(v_min):
    with pytest.raises(ValueError, match="v_min_v"):
        BatteryPack(CAPACITY_KWH, v_min_v=v_min, v_max_v=400.0)


@pytest.mark.parametrize("resistance", [0.0, -0.05])
def test_non_positive_reference_resistance_is_rejected(resistance):
    with pytest.raises(ValueError, match="r_ref_ohm"):
        BatteryPack(CAPACITY_KWH, r_ref_ohm=resistance)


@pytest.mark.parametrize("rate", [0.0, -3.0])
def test_non_positive_c_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="discharge_c_rate"):
        BatteryPack(CAPACITY_KWH, discharge_c_rate=rate)
    with pytest.raises(ValueError, match="charge_c_rate"):
        BatteryPack(CAPACITY_KWH, charge_c_rate=rate)


@pytest.mark.parametrize("soc_min", [-0.01, 1.0, 1.5])
def test_soc_min_outside_the_valid_range_is_rejected(soc_min):
    with pytest.raises(ValueError, match="soc_min"):
        BatteryPack(CAPACITY_KWH, soc_min=soc_min)


@pytest.mark.parametrize("dt", [0.0, -60.0])
def test_non_positive_timestep_is_rejected(pack, dt):
    with pytest.raises(ValueError, match="dt_s"):
        pack.step(SOC_MID, BUS_KW, dt)


# ---------------------------------------------------------------------------
# Comparison against the engine-rated power limit this model replaces
# ---------------------------------------------------------------------------


def test_pack_power_limit_derives_from_capacity_not_engine_rating():
    """5 kWh caps at 15 kW; the previous 75 kW limit implied 15C - see B-04."""
    small = BatteryPack(SMALL_PACK_KWH)
    assert small.max_discharge_kw == pytest.approx(15.0)

    previous_c_rate = PREVIOUS_MODEL_ENGINE_KW / SMALL_PACK_KWH
    assert previous_c_rate == pytest.approx(15.0)
    assert previous_c_rate / small.discharge_c_rate == pytest.approx(5.0)

    state = small.step(SOC_MID, PREVIOUS_MODEL_ENGINE_KW, DT_S)
    assert state.power_limited
    assert state.power_kw == pytest.approx(15.0)
