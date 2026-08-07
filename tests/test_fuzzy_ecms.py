"""Tests for the opt-in fuzzy ECMS controller and its integration boundary."""

import dataclasses
import math

import pytest

from src.control.base import ControlContext, assert_controller_contract
from src.control.fuzzy_ecms import FuzzyECMS
from src.control.power_split import solve_split
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.powertrain import SeriesPowertrain


def _context(
    *,
    soc: float = 0.5,
    demand_normalised: float = 0.5,
    switching_s: float = 6.0,
    neutral_s: float = 8.0,
    time_s: float = 60.0,
    phase: str = "loiter",
) -> ControlContext:
    max_bus_kw = 100.0
    return ControlContext(
        soc=soc,
        bus_demand_kw=demand_normalised * max_bus_kw,
        max_bus_kw=max_bus_kw,
        neutral_s=neutral_s,
        switching_s=switching_s,
        time_s=time_s,
        phase=phase,
    )


@pytest.mark.parametrize("demand_normalised", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_low_and_high_state_of_charge_recover_the_configured_ratio_bounds(
    demand_normalised: float,
) -> None:
    controller = FuzzyECMS()
    switching = 6.0
    low_soc = controller.equivalence_factor(
        _context(
            soc=controller.soc_low,
            demand_normalised=demand_normalised,
            switching_s=switching,
        )
    )
    high_soc = controller.equivalence_factor(
        _context(
            soc=controller.soc_high,
            demand_normalised=demand_normalised,
            switching_s=switching,
        )
    )
    assert low_soc == pytest.approx(controller.s_max_ratio * switching)
    assert high_soc == pytest.approx(controller.s_min_ratio * switching)


def test_medium_state_of_charge_uses_demand_to_price_peak_assistance() -> None:
    controller = FuzzyECMS()
    medium_soc = 0.5 * (controller.soc_low + controller.soc_high)
    outputs = tuple(
        controller.equivalence_factor(
            _context(soc=medium_soc, demand_normalised=demand)
        )
        for demand in (0.0, 0.5, 1.0)
    )
    assert outputs == pytest.approx((6.75, 6.0, 5.25))
    assert outputs[0] > outputs[1] > outputs[2]


def test_output_tracks_switching_reference_and_ignores_average_cost_reference() -> None:
    controller = FuzzyECMS()
    first = controller.equivalence_factor(
        _context(soc=0.4, switching_s=4.0, neutral_s=3.0)
    )
    second = controller.equivalence_factor(
        _context(soc=0.4, switching_s=8.0, neutral_s=12.0)
    )
    changed_neutral = controller.equivalence_factor(
        _context(soc=0.4, switching_s=4.0, neutral_s=18.0)
    )
    assert second == pytest.approx(2.0 * first)
    assert changed_neutral == first


def test_time_and_phase_do_not_overfit_the_fuzzy_policy_to_this_mission() -> None:
    controller = FuzzyECMS()
    loiter = controller.equivalence_factor(
        _context(soc=0.4, time_s=60.0, phase="loiter")
    )
    takeoff = controller.equivalence_factor(
        _context(soc=0.4, time_s=3600.0, phase="takeoff")
    )
    assert takeoff == loiter


def test_every_membership_boundary_has_a_finite_active_rule() -> None:
    controller = FuzzyECMS()
    soc_values = (
        0.0,
        controller.soc_low,
        0.5 * (controller.soc_low + controller.soc_high),
        controller.soc_high,
        1.0,
    )
    for soc in soc_values:
        for demand_index in range(101):
            result = controller.equivalence_factor(
                _context(soc=soc, demand_normalised=demand_index / 100.0)
            )
            assert math.isfinite(result)


def test_default_controller_passes_every_shared_contract_assertion() -> None:
    assert_controller_contract(FuzzyECMS())


def test_output_range_genes_move_the_saturated_outputs_instead_of_staying_flat() -> None:
    narrow = FuzzyECMS(s_min_ratio=0.9, s_max_ratio=1.1)
    wide = FuzzyECMS(s_min_ratio=0.6, s_max_ratio=1.6)
    switching = 5.0
    assert narrow.reachable_range(switching) == pytest.approx((4.5, 5.5))
    assert wide.reachable_range(switching) == pytest.approx((3.0, 8.0))
    assert wide.equivalence_factor(_context(soc=0.0, switching_s=switching)) > (
        narrow.equivalence_factor(_context(soc=0.0, switching_s=switching))
    )
    assert wide.equivalence_factor(_context(soc=1.0, switching_s=switching)) < (
        narrow.equivalence_factor(_context(soc=1.0, switching_s=switching))
    )


def test_reachable_range_always_straddles_the_marginal_switching_reference() -> None:
    controller = FuzzyECMS(s_min_ratio=0.7, s_max_ratio=1.4)
    assert controller.reachable_range(6.0) == pytest.approx((4.2, 8.4))
    assert controller.straddles_switching(6.0)


def test_controller_factor_is_accepted_by_the_existing_power_split_solver() -> None:
    engine = Turboshaft(75.0)
    battery = BatteryPack(20.0)
    powertrain = SeriesPowertrain()
    factor = FuzzyECMS().clamped_equivalence_factor(
        _context(soc=0.5, demand_normalised=0.4)
    )
    decision = solve_split(
        bus_demand_kw=40.0,
        engine=engine,
        battery=battery,
        powertrain=powertrain,
        s=factor,
        soc=0.5,
        sigma=1.0,
        dt_s=60.0,
    )
    assert decision.feasible
    assert decision.bus_from_engine_kw + decision.battery_bus_kw == pytest.approx(40.0)


def test_name_is_stable_and_encodes_every_tunable_parameter() -> None:
    controller = FuzzyECMS(
        soc_low=0.25,
        soc_high=0.75,
        s_min_ratio=0.7,
        s_max_ratio=1.4,
    )
    assert controller.name == "fuzzy_soc=0.25-0.75_r=0.70-1.40"


def test_controller_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        FuzzyECMS().soc_low = 0.2


@pytest.mark.parametrize(
    ("soc_low", "soc_high"),
    [
        (0.0, 0.7),
        (0.3, 1.0),
        (0.7, 0.3),
        (0.5, 0.5),
        (float("nan"), 0.7),
        (0.3, float("inf")),
    ],
)
def test_invalid_state_of_charge_thresholds_are_rejected(
    soc_low: float,
    soc_high: float,
) -> None:
    with pytest.raises(ValueError, match="soc"):
        FuzzyECMS(soc_low=soc_low, soc_high=soc_high)


@pytest.mark.parametrize(
    ("s_min_ratio", "s_max_ratio"),
    [
        (0.0, 1.25),
        (1.0, 1.25),
        (0.75, 1.0),
        (0.75, 0.9),
        (float("nan"), 1.25),
        (0.75, float("inf")),
    ],
)
def test_ratio_bounds_that_cannot_straddle_switching_are_rejected(
    s_min_ratio: float,
    s_max_ratio: float,
) -> None:
    with pytest.raises(ValueError, match="ratio"):
        FuzzyECMS(s_min_ratio=s_min_ratio, s_max_ratio=s_max_ratio)


@pytest.mark.parametrize("switching_s", [0.0, -1.0, float("nan"), float("inf")])
def test_diagnostics_reject_a_non_physical_switching_factor(
    switching_s: float,
) -> None:
    with pytest.raises(ValueError, match="switching_s"):
        FuzzyECMS().reachable_range(switching_s)
