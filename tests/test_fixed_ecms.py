"""Tests for the fixed-factor ECMS experimental control group."""

import dataclasses
import math

import pytest

from src.control.base import (
    ControlContext,
    S_ABSOLUTE_MAX,
    S_ABSOLUTE_MIN,
    assert_controller_contract,
)
from src.control.fixed_ecms import FixedECMS
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.mass import build_mass_budget
from src.models.powertrain import SeriesPowertrain
from src.simulation.mission import ps1_mission
from src.simulation.simulator import Aircraft, run_mission


def _context(
    *,
    soc: float = 0.5,
    demand_kw: float = 40.0,
    neutral_s: float = 5.0,
    switching_s: float = 4.8,
) -> ControlContext:
    return ControlContext(
        soc=soc,
        bus_demand_kw=demand_kw,
        max_bus_kw=100.0,
        neutral_s=neutral_s,
        switching_s=switching_s,
        time_s=60.0,
        phase="loiter",
    )


def test_absolute_mode_returns_exactly_s_over_every_context() -> None:
    controller = FixedECMS(s=5.25)
    for soc in (0.0, 0.2, 0.8, 1.0):
        for demand_kw in (0.0, 30.0, 100.0):
            for neutral_s in (2.0, 5.0, 12.0):
                assert controller.equivalence_factor(
                    _context(soc=soc, demand_kw=demand_kw, neutral_s=neutral_s)
                ) == 5.25


def test_ratio_mode_tracks_switching_but_not_state_of_charge() -> None:
    controller = FixedECMS(s_ratio=1.1)
    low_switching = controller.equivalence_factor(
        _context(soc=0.5, switching_s=4.0)
    )
    high_switching = controller.equivalence_factor(
        _context(soc=0.5, switching_s=7.0)
    )
    changed_soc = controller.equivalence_factor(
        _context(soc=0.9, switching_s=7.0)
    )
    assert low_switching == 1.1 * 4.0
    assert high_switching == 1.1 * 7.0
    assert changed_soc == high_switching


def test_switching_ratio_ignores_the_average_cost_neutral_diagnostic() -> None:
    controller = FixedECMS(s_ratio=1.1)
    low = controller.equivalence_factor(_context(neutral_s=4.0, switching_s=5.0))
    high = controller.equivalence_factor(_context(neutral_s=9.0, switching_s=5.0))
    assert high == low


def test_clamped_factor_passes_in_range_and_clamps_out_of_range() -> None:
    context = _context()
    assert FixedECMS(s=6.0).clamped_equivalence_factor(context) == 6.0
    assert (
        FixedECMS(s=100.0).clamped_equivalence_factor(context)
        == S_ABSOLUTE_MAX
    )
    assert FixedECMS(s=0.1).clamped_equivalence_factor(context) == S_ABSOLUTE_MIN


@pytest.mark.parametrize(
    "controller",
    [FixedECMS(s=5.0), FixedECMS(s_ratio=1.1)],
)
def test_contract_passes_when_only_assertions_four_and_five_are_skipped(
    controller: FixedECMS,
) -> None:
    # Assertion 3 remains active here and therefore passes explicitly.
    assert_controller_contract(controller, skip_assertions=(4, 5))


@pytest.mark.parametrize(
    "controller",
    [FixedECMS(s=5.0), FixedECMS(s_ratio=1.1)],
)
def test_full_contract_fails_at_assertion_four(controller: FixedECMS) -> None:
    with pytest.raises(AssertionError, match="assertion 4"):
        assert_controller_contract(controller)


@pytest.mark.parametrize(
    "controller",
    [FixedECMS(s=5.0), FixedECMS(s_ratio=1.1)],
)
def test_assertion_five_also_fails_when_four_alone_is_skipped(
    controller: FixedECMS,
) -> None:
    with pytest.raises(AssertionError, match="assertion 5"):
        assert_controller_contract(controller, skip_assertions=(4,))


def test_reference_constructors_select_the_contract_rails_and_both_references() -> None:
    context = _context(neutral_s=7.25, switching_s=4.75)
    assert FixedECMS.pure_thermal().equivalence_factor(context) >= S_ABSOLUTE_MAX
    assert FixedECMS.battery_first().equivalence_factor(context) <= S_ABSOLUTE_MIN
    assert FixedECMS.at_neutral().equivalence_factor(context) == context.neutral_s
    assert FixedECMS.at_switching().equivalence_factor(context) == context.switching_s


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"s": 5.0, "s_ratio": 1.0},
    ],
)
def test_exactly_one_parameter_must_be_set(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FixedECMS(**kwargs)


@pytest.mark.parametrize("field_name", ["s", "s_ratio"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_parameter_must_be_finite_and_positive(field_name: str, value: float) -> None:
    with pytest.raises(ValueError, match=field_name):
        FixedECMS(**{field_name: value})


def test_ratio_anchor_must_name_a_supported_reference() -> None:
    with pytest.raises(ValueError, match="ratio_anchor"):
        FixedECMS(s_ratio=1.0, ratio_anchor="average")  # type: ignore[arg-type]


def test_name_is_stable_and_encodes_the_parameter() -> None:
    first = FixedECMS(s=4.5)
    identical = FixedECMS(s=4.5)
    different = FixedECMS(s=5.5)
    ratio = FixedECMS(s_ratio=1.1)
    assert first.name == identical.name == "fixed_s=4.50"
    assert first.name != different.name
    assert ratio.name == "fixed_switching_ratio=1.10"
    assert ratio.name != first.name


def test_controller_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        FixedECMS(s=5.0).s = 6.0


def test_actual_reference_aircraft_cannot_fly_the_mission_pure_thermal() -> None:
    masses = build_mass_budget(
        engine_kw=75.0,
        battery_kwh=10.0,
        peak_bus_kw=90.0,
        wing_area_m2=10.0,
        aspect_ratio=16.0,
    )
    aircraft = Aircraft(
        wing_area_m2=10.0,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        engine=Turboshaft(75.0),
        battery=BatteryPack(10.0),
        powertrain=SeriesPowertrain(),
        masses=masses,
    )
    result = run_mission(
        aircraft,
        ps1_mission(),
        FixedECMS.pure_thermal(),
        record_log=True,
    )

    assert masses.fuel_kg == pytest.approx(279.88050779926476, rel=1.0e-12)
    assert not result.mission_complete
    assert result.termination_reason == "altitude_unreachable"
    assert result.final_soc == pytest.approx(aircraft.battery.soc_min, abs=1.0e-12)
    assert result.log is not None
    assert any(step.battery_bus_kw > 0.0 for step in result.log)
    assert not any(step.phase == "loiter" for step in result.log)

    print(f"reference_fuel_load_kg={masses.fuel_kg:.12f}")
    print(f"reference_time_to_failure_s={result.endurance_s:.12f}")
    print("reference_pure_thermal_endurance_h=unavailable")
    print("reference_mean_loiter_fuel_flow_kg_s=unavailable")
