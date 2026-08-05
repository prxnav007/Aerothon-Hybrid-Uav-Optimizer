"""Tests for proportional state-of-charge feedback in adaptive ECMS."""

import dataclasses

import pytest

from src.control.base import ControlContext, assert_controller_contract
from src.control.pi_ecms import PIECMS


def _context(soc: float = 0.6, neutral_s: float = 6.0) -> ControlContext:
    return ControlContext(
        soc=soc,
        bus_demand_kw=40.0,
        max_bus_kw=100.0,
        neutral_s=neutral_s,
        time_s=60.0,
        phase="loiter",
    )


def test_absolute_anchor_is_recovered_exactly_at_the_reference_state_of_charge() -> None:
    controller = PIECMS(s0=7.0, s0_ratio=None)
    assert controller.equivalence_factor(_context(soc=controller.soc_ref)) == 7.0


def test_ratio_anchor_is_recovered_exactly_at_the_reference_state_of_charge() -> None:
    controller = PIECMS(s0_ratio=1.2)
    ctx = _context(soc=controller.soc_ref, neutral_s=6.0)
    assert controller.equivalence_factor(ctx) == pytest.approx(7.2, rel=1.0e-12)


def test_output_is_linear_in_state_of_charge_with_slope_minus_kp() -> None:
    controller = PIECMS(kp=7.5)
    low_soc, high_soc = 0.2, 0.8
    low_output = controller.equivalence_factor(_context(soc=low_soc))
    high_output = controller.equivalence_factor(_context(soc=high_soc))
    slope = (high_output - low_output) / (high_soc - low_soc)
    assert slope == pytest.approx(-controller.kp, rel=1.0e-12)


def test_ratio_anchor_tracks_the_operating_point_without_scaling_feedback() -> None:
    controller = PIECMS(kp=5.0, s0_ratio=1.2)
    low = controller.equivalence_factor(_context(soc=0.3, neutral_s=4.0))
    high = controller.equivalence_factor(_context(soc=0.3, neutral_s=7.0))
    assert high - low == pytest.approx(1.2 * (7.0 - 4.0), rel=1.0e-12)


def test_absolute_anchor_ignores_the_operating_point_neutral_factor() -> None:
    controller = PIECMS(s0=6.5, s0_ratio=None)
    low = controller.equivalence_factor(_context(soc=0.3, neutral_s=4.0))
    high = controller.equivalence_factor(_context(soc=0.3, neutral_s=9.0))
    assert high == low


def test_default_controller_passes_every_shared_contract_assertion() -> None:
    assert_controller_contract(PIECMS())


def test_zero_gain_degenerates_to_a_flat_controller_rejected_at_assertion_four() -> None:
    with pytest.raises(AssertionError, match="assertion 4"):
        assert_controller_contract(PIECMS(kp=0.0))


def test_reachable_range_matches_the_two_state_of_charge_endpoints() -> None:
    controller = PIECMS(kp=5.0, soc_ref=0.6, s0_ratio=1.2)
    assert controller.reachable_range(6.0) == pytest.approx((5.2, 10.2), rel=1.0e-12)


def test_default_reachable_range_straddles_the_neutral_factor() -> None:
    controller = PIECMS()
    assert controller.reachable_range(6.0) == pytest.approx((4.0, 9.0), rel=1.0e-12)
    assert controller.straddles_neutral(6.0)


def test_absolute_anchor_diagnostics_use_the_same_raw_feedback_law() -> None:
    controller = PIECMS(kp=5.0, soc_ref=0.6, s0=7.0, s0_ratio=None)
    assert controller.reachable_range(7.0) == pytest.approx((5.0, 10.0), rel=1.0e-12)
    assert controller.straddles_neutral(7.0)


def test_high_anchor_and_low_gain_can_fail_to_straddle_without_raising() -> None:
    controller = PIECMS(kp=1.0, s0_ratio=1.5)
    assert controller.reachable_range(6.0) == pytest.approx((8.6, 9.6), rel=1.0e-12)
    assert not controller.straddles_neutral(6.0)


def test_name_encodes_gain_and_ratio_anchor() -> None:
    assert PIECMS(kp=5.0, s0_ratio=1.0).name == "pi_kp=5.00_r=1.00"


def test_name_encodes_gain_and_absolute_anchor() -> None:
    assert PIECMS(kp=4.25, s0=6.5, s0_ratio=None).name == "pi_kp=4.25_s0=6.50"


def test_controller_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PIECMS().kp = 10.0


@pytest.mark.parametrize("kp", [-1.0, float("nan"), float("inf")])
def test_negative_or_non_finite_feedback_gain_is_rejected(kp: float) -> None:
    with pytest.raises(ValueError, match="kp"):
        PIECMS(kp=kp)


@pytest.mark.parametrize("soc_ref", [-0.01, 1.01, float("nan"), float("inf")])
def test_out_of_range_or_non_finite_reference_state_of_charge_is_rejected(
    soc_ref: float,
) -> None:
    with pytest.raises(ValueError, match="soc_ref"):
        PIECMS(soc_ref=soc_ref)


def test_both_anchors_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PIECMS(s0=6.0)


def test_missing_both_anchors_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PIECMS(s0=None, s0_ratio=None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"s0": float("nan"), "s0_ratio": None},
        {"s0": float("inf"), "s0_ratio": None},
        {"s0": None, "s0_ratio": float("nan")},
        {"s0": None, "s0_ratio": float("inf")},
    ],
)
def test_non_finite_anchor_is_rejected(kwargs: dict[str, float | None]) -> None:
    with pytest.raises(ValueError):
        PIECMS(**kwargs)


@pytest.mark.parametrize("neutral_s", [0.0, -1.0, float("nan"), float("inf")])
def test_diagnostics_reject_a_non_physical_neutral_factor(neutral_s: float) -> None:
    with pytest.raises(ValueError, match="neutral_s"):
        PIECMS().reachable_range(neutral_s)
