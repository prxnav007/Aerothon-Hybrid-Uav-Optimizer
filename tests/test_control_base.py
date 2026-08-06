"""Tests for the shared energy-management controller interface and contract."""

import dataclasses
import math

import pytest

from src.control.base import (
    ControlContext,
    EMSController,
    S_ABSOLUTE_MAX,
    S_ABSOLUTE_MIN,
    assert_controller_contract,
    neutral_equivalence_factor,
)

SOURCE_CHAIN = 0.95 * 0.95
LHV_KJ_KG = 43100.0


def _context(
    soc: float = 0.5,
    bus_demand_kw: float = 40.0,
    max_bus_kw: float = 100.0,
    neutral_s: float = 5.0,
    switching_s: float = 4.8,
    time_s: float = 60.0,
    phase: str = "loiter",
) -> ControlContext:
    return ControlContext(
        soc=soc,
        bus_demand_kw=bus_demand_kw,
        max_bus_kw=max_bus_kw,
        neutral_s=neutral_s,
        switching_s=switching_s,
        time_s=time_s,
        phase=phase,
    )


@dataclasses.dataclass(frozen=True)
class ConstantController(EMSController):
    value: float

    def equivalence_factor(self, ctx: ControlContext) -> float:
        return self.value

    @property
    def name(self) -> str:
        return "constant-test-controller"


class MonotoneLinearController(EMSController):
    def equivalence_factor(self, ctx: ControlContext) -> float:
        return ctx.switching_s + 4.0 * (0.5 - ctx.soc)

    @property
    def name(self) -> str:
        return "monotone-linear-test-controller"


class BelowSwitchingController(EMSController):
    def equivalence_factor(self, ctx: ControlContext) -> float:
        return ctx.switching_s - 1.0 - ctx.soc

    @property
    def name(self) -> str:
        return "below-switching-test-controller"


def test_neutral_factor_matches_both_specific_fuel_consumption_anchors() -> None:
    assert neutral_equivalence_factor(0.30, SOURCE_CHAIN, LHV_KJ_KG) == pytest.approx(
        3.98, abs=0.01
    )
    assert neutral_equivalence_factor(0.45, SOURCE_CHAIN, LHV_KJ_KG) == pytest.approx(
        5.97, abs=0.01
    )


def test_neutral_factor_is_linear_in_specific_fuel_consumption() -> None:
    base = neutral_equivalence_factor(0.30, SOURCE_CHAIN, LHV_KJ_KG)
    doubled = neutral_equivalence_factor(0.60, SOURCE_CHAIN, LHV_KJ_KG)
    assert doubled == pytest.approx(2.0 * base, rel=1.0e-12)


def test_neutral_factor_is_inversely_proportional_to_source_efficiency() -> None:
    efficient = neutral_equivalence_factor(0.30, 0.90, LHV_KJ_KG)
    inefficient = neutral_equivalence_factor(0.30, 0.45, LHV_KJ_KG)
    assert inefficient == pytest.approx(2.0 * efficient, rel=1.0e-12)


def test_round_trip_correction_raises_the_neutral_factor() -> None:
    uncorrected = neutral_equivalence_factor(0.45, SOURCE_CHAIN, LHV_KJ_KG)
    corrected = neutral_equivalence_factor(
        0.45, SOURCE_CHAIN, LHV_KJ_KG, round_trip_efficiency=0.90
    )
    assert corrected == pytest.approx(uncorrected / 0.90, rel=1.0e-12)
    assert corrected > uncorrected


def test_unit_round_trip_efficiency_leaves_the_neutral_factor_unchanged() -> None:
    default = neutral_equivalence_factor(0.45, SOURCE_CHAIN, LHV_KJ_KG)
    explicit = neutral_equivalence_factor(
        0.45, SOURCE_CHAIN, LHV_KJ_KG, round_trip_efficiency=1.0
    )
    assert explicit == pytest.approx(default, rel=1.0e-12)


def test_controller_interface_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        EMSController()


def test_minimal_concrete_controller_can_be_instantiated_and_called() -> None:
    controller = ConstantController(5.0)
    assert controller.name == "constant-test-controller"
    assert controller.equivalence_factor(_context()) == 5.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1.0e6, S_ABSOLUTE_MAX), (-5.0, S_ABSOLUTE_MIN), (6.0, 6.0)],
)
def test_concrete_clamping_method_enforces_the_absolute_bounds(
    raw: float, expected: float
) -> None:
    assert ConstantController(raw).clamped_equivalence_factor(_context()) == expected


@pytest.mark.parametrize(
    ("demand_kw", "max_bus_kw", "expected"),
    [(-10.0, 100.0, 0.0), (40.0, 100.0, 0.4), (150.0, 100.0, 1.0), (40.0, 0.0, 0.0)],
)
def test_normalised_demand_is_clamped_and_zero_safe(
    demand_kw: float, max_bus_kw: float, expected: float
) -> None:
    ctx = _context(bus_demand_kw=demand_kw, max_bus_kw=max_bus_kw)
    assert ctx.demand_normalised == pytest.approx(expected, rel=1.0e-12)


def test_control_context_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _context().soc = 0.2


def test_contract_accepts_a_responsive_monotone_controller() -> None:
    assert_controller_contract(MonotoneLinearController())


def test_contract_rejects_a_flat_controller_at_assertion_four() -> None:
    with pytest.raises(AssertionError, match="assertion 4"):
        assert_controller_contract(ConstantController(5.0))


def test_contract_rejects_a_non_finite_controller_at_assertion_one() -> None:
    with pytest.raises(AssertionError, match="assertion 1"):
        assert_controller_contract(ConstantController(float("nan")))


def test_contract_rejects_a_controller_confined_below_switching_at_assertion_five(
) -> None:
    with pytest.raises(AssertionError, match="assertion 5"):
        assert_controller_contract(BelowSwitchingController())


def test_contract_requires_fixed_controller_exceptions_to_be_named() -> None:
    assert_controller_contract(ConstantController(5.0), skip_assertions=(3, 4, 5))


def test_contract_does_not_allow_safety_assertions_to_be_skipped() -> None:
    with pytest.raises(ValueError, match="only 3, 4, and 5"):
        assert_controller_contract(ConstantController(5.0), skip_assertions=(1,))


@pytest.mark.parametrize("soc", [-0.01, 1.01])
def test_control_context_rejects_state_of_charge_outside_the_unit_interval(soc: float) -> None:
    with pytest.raises(ValueError, match="soc"):
        _context(soc=soc)


def test_control_context_rejects_a_negative_bus_power_normaliser() -> None:
    with pytest.raises(ValueError, match="max_bus_kw"):
        _context(max_bus_kw=-1.0)


@pytest.mark.parametrize("switching_s", [0.0, -1.0, math.inf, math.nan])
def test_control_context_rejects_a_non_physical_switching_factor(
    switching_s: float,
) -> None:
    with pytest.raises(ValueError, match="switching_s"):
        _context(switching_s=switching_s)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sfc_kg_kwh": 0.0},
        {"source_chain_efficiency": 0.0},
        {"source_chain_efficiency": 1.1},
        {"lhv_kj_kg": 0.0},
        {"round_trip_efficiency": 0.0},
        {"round_trip_efficiency": 1.1},
    ],
)
def test_neutral_factor_rejects_non_physical_arguments(kwargs: dict[str, float]) -> None:
    values = {
        "sfc_kg_kwh": 0.45,
        "source_chain_efficiency": SOURCE_CHAIN,
        "lhv_kj_kg": LHV_KJ_KG,
        "round_trip_efficiency": 1.0,
    }
    with pytest.raises(ValueError):
        neutral_equivalence_factor(**{**values, **kwargs})
