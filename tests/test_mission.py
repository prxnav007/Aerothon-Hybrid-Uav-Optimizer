"""Tests for immutable mission-profile data and validation."""

import dataclasses
import math

import pytest

from src.simulation.mission import (
    AltitudeMode,
    MissionProfile,
    Phase,
    SpeedMode,
    Termination,
    ps1_mission,
)


MANDATED_PHASE_NAMES = (
    "takeoff",
    "climb",
    "cruise",
    "loiter",
    "descent",
    "landing",
)


def _phase(**changes: object) -> Phase:
    values: dict[str, object] = {
        "name": "test_phase",
        "speed_mode": SpeedMode.FIXED,
        "altitude_mode": AltitudeMode.HOLD,
        "termination": Termination.DURATION,
        "target_altitude_m": 0.0,
        "speed_mps": 50.0,
        "climb_rate_mps": None,
        "duration_s": 60.0,
    }
    return Phase(**{**values, **changes})  # type: ignore[arg-type]


def _profile(phases: tuple[Phase, ...]) -> MissionProfile:
    return MissionProfile(
        phases=phases,
        initial_altitude_m=0.0,
        fuel_reserve_kg=5.0,
        min_usable_fuel_kg=20.0,
        max_mission_time_s=86400.0,
    )


def test_factory_returns_the_six_mandated_phases_in_order() -> None:
    mission = ps1_mission()
    assert len(mission.phases) == 6
    assert mission.phase_names == MANDATED_PHASE_NAMES


def test_factory_keeps_the_mandated_cruise_speed_fixed() -> None:
    cruise = ps1_mission().phase_by_name("cruise")
    assert cruise.speed_mode is SpeedMode.FIXED
    assert cruise.speed_mps == pytest.approx(69.44, abs=0.005)


def test_factory_defaults_loiter_to_open_ended_minimum_power_flight() -> None:
    loiter = ps1_mission().phase_by_name("loiter")
    assert loiter.termination is Termination.RESOURCE
    assert loiter.speed_mode is SpeedMode.MIN_POWER
    assert loiter.speed_mps is None


def test_factory_altitude_chain_is_continuous_from_initial_altitude_to_ground() -> None:
    mission = ps1_mission()
    altitude_m = mission.initial_altitude_m
    for phase in mission.phases:
        if phase.altitude_mode is AltitudeMode.HOLD:
            assert phase.target_altitude_m == altitude_m
        else:
            altitude_m = phase.target_altitude_m
    assert altitude_m == 0.0


def test_cruise_altitude_argument_propagates_to_every_altitude_phase() -> None:
    mission = ps1_mission(cruise_altitude_m=8000.0)
    assert mission.phase_by_name("climb").target_altitude_m == 8000.0
    assert mission.phase_by_name("cruise").target_altitude_m == 8000.0
    assert mission.phase_by_name("loiter").target_altitude_m == 8000.0
    assert all(phase.target_altitude_m != 3000.0 for phase in mission.phases)


def test_endurance_phase_index_is_the_loiter_index() -> None:
    mission = ps1_mission()
    assert mission.endurance_phase_index == 3
    assert mission.phases[mission.endurance_phase_index].name == "loiter"


def test_phase_lookup_raises_for_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown phase name"):
        ps1_mission().phase_by_name("missing")


def test_factory_keeps_fixed_loiter_speed_reachable() -> None:
    mission = ps1_mission(
        loiter_speed_mode=SpeedMode.FIXED,
        loiter_speed_mps=70.0,
    )
    assert mission.phase_by_name("loiter").speed_mps == 70.0


@pytest.mark.parametrize("speed_mps", [None, 0.0, -1.0, math.inf, math.nan])
def test_fixed_speed_requires_a_finite_positive_speed_mps(speed_mps: float | None) -> None:
    with pytest.raises(ValueError, match="speed_mps"):
        _phase(speed_mps=speed_mps)


@pytest.mark.parametrize("speed_mode", [SpeedMode.MIN_POWER, SpeedMode.BEST_LD])
def test_solved_speed_rejects_an_ambiguous_speed_mps(speed_mode: SpeedMode) -> None:
    with pytest.raises(ValueError, match="speed_mps"):
        _phase(speed_mode=speed_mode, speed_mps=50.0)


@pytest.mark.parametrize("climb_rate_mps", [None, 0.0, math.inf, math.nan])
def test_altitude_change_requires_a_finite_nonzero_climb_rate_mps(
    climb_rate_mps: float | None,
) -> None:
    with pytest.raises(ValueError, match="climb_rate_mps"):
        _phase(
            altitude_mode=AltitudeMode.CLIMB_TO,
            termination=Termination.ALTITUDE,
            target_altitude_m=1000.0,
            climb_rate_mps=climb_rate_mps,
            duration_s=None,
        )


def test_climb_to_rejects_a_negative_climb_rate_mps() -> None:
    with pytest.raises(ValueError, match="climb_rate_mps"):
        _phase(
            altitude_mode=AltitudeMode.CLIMB_TO,
            termination=Termination.ALTITUDE,
            target_altitude_m=1000.0,
            climb_rate_mps=-2.0,
            duration_s=None,
        )


def test_descend_to_rejects_a_positive_climb_rate_mps() -> None:
    with pytest.raises(ValueError, match="climb_rate_mps"):
        _phase(
            altitude_mode=AltitudeMode.DESCEND_TO,
            termination=Termination.ALTITUDE,
            climb_rate_mps=2.0,
            duration_s=None,
        )


def test_hold_rejects_a_climb_rate_mps() -> None:
    with pytest.raises(ValueError, match="climb_rate_mps"):
        _phase(climb_rate_mps=2.0)


@pytest.mark.parametrize("duration_s", [None, 0.0, -1.0, math.inf, math.nan])
def test_duration_termination_requires_a_finite_positive_duration_s(
    duration_s: float | None,
) -> None:
    with pytest.raises(ValueError, match="duration_s"):
        _phase(duration_s=duration_s)


@pytest.mark.parametrize("termination", [Termination.ALTITUDE, Termination.RESOURCE])
def test_non_duration_termination_rejects_duration_s(termination: Termination) -> None:
    with pytest.raises(ValueError, match="duration_s"):
        _phase(termination=termination, duration_s=60.0)


@pytest.mark.parametrize("target_altitude_m", [-1.0, math.inf, math.nan])
def test_target_altitude_m_must_be_finite_and_non_negative(target_altitude_m: float) -> None:
    with pytest.raises(ValueError, match="target_altitude_m"):
        _phase(target_altitude_m=target_altitude_m)


@pytest.mark.parametrize("name", ["", "take-off", "two words", "1phase"])
def test_name_must_be_a_non_empty_identifier(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        _phase(name=name)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("speed_mode", "fixed"),
        ("altitude_mode", "hold"),
        ("termination", "duration"),
    ],
)
def test_phase_mode_fields_require_their_enum_types(field_name: str, value: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _phase(**{field_name: value})


def test_profile_rejects_fewer_than_one_phase() -> None:
    with pytest.raises(ValueError, match="phases"):
        _profile(())


def test_profile_rejects_two_resource_phases() -> None:
    mission = ps1_mission()
    cruise = dataclasses.replace(
        mission.phase_by_name("cruise"),
        termination=Termination.RESOURCE,
        duration_s=None,
    )
    phases = mission.phases[:2] + (cruise,) + mission.phases[3:]
    with pytest.raises(ValueError, match="RESOURCE"):
        _profile(phases)


def test_profile_rejects_zero_resource_phases() -> None:
    mission = ps1_mission()
    loiter = dataclasses.replace(
        mission.phase_by_name("loiter"),
        termination=Termination.DURATION,
        duration_s=60.0,
    )
    phases = mission.phases[:3] + (loiter,) + mission.phases[4:]
    with pytest.raises(ValueError, match="RESOURCE"):
        _profile(phases)


def test_profile_rejects_a_resource_phase_at_the_end() -> None:
    with pytest.raises(ValueError, match="RESOURCE"):
        _profile(ps1_mission().phases[:4])


def test_profile_requires_a_descent_to_ground_after_the_resource_phase() -> None:
    mission = ps1_mission()
    phases = mission.phases[:4] + (mission.phase_by_name("landing"),)
    with pytest.raises(ValueError, match="DESCEND_TO"):
        _profile(phases)


def test_profile_reports_both_phases_at_an_altitude_discontinuity() -> None:
    mission = ps1_mission()
    cruise = dataclasses.replace(
        mission.phase_by_name("cruise"),
        target_altitude_m=3100.0,
    )
    phases = mission.phases[:2] + (cruise,) + mission.phases[3:]
    with pytest.raises(ValueError, match=r"climb.*cruise"):
        _profile(phases)


def test_profile_requires_unique_phase_names() -> None:
    mission = ps1_mission()
    duplicate = dataclasses.replace(mission.phase_by_name("landing"), name="descent")
    with pytest.raises(ValueError, match="name"):
        _profile(mission.phases[:-1] + (duplicate,))


def test_profile_rejects_a_climb_target_below_its_starting_altitude() -> None:
    mission = ps1_mission()
    climb = dataclasses.replace(
        mission.phase_by_name("climb"),
        target_altitude_m=0.0,
    )
    with pytest.raises(ValueError, match="target_altitude_m"):
        _profile(mission.phases[:1] + (climb,) + mission.phases[2:])


def test_profile_rejects_a_descent_target_above_its_starting_altitude() -> None:
    mission = ps1_mission()
    descent = dataclasses.replace(
        mission.phase_by_name("descent"),
        name="bad_descent",
        target_altitude_m=4000.0,
    )
    with pytest.raises(ValueError, match="target_altitude_m"):
        _profile(mission.phases[:4] + (descent,) + mission.phases[4:])


def test_profile_rejects_a_final_phase_that_does_not_end_at_zero_altitude() -> None:
    mission = ps1_mission()
    final_climb = _phase(
        name="departure",
        altitude_mode=AltitudeMode.CLIMB_TO,
        termination=Termination.ALTITUDE,
        target_altitude_m=100.0,
        climb_rate_mps=1.0,
        duration_s=None,
    )
    with pytest.raises(ValueError, match="final phase.*target_altitude_m"):
        _profile(mission.phases + (final_climb,))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("fuel_reserve_kg", 0.0),
        ("fuel_reserve_kg", math.inf),
        ("min_usable_fuel_kg", 0.0),
        ("max_mission_time_s", 0.0),
    ],
)
def test_profile_limits_must_be_finite_and_positive(field_name: str, value: float) -> None:
    values = {
        "phases": ps1_mission().phases,
        "initial_altitude_m": 0.0,
        "fuel_reserve_kg": 5.0,
        "min_usable_fuel_kg": 20.0,
        "max_mission_time_s": 86400.0,
        field_name: value,
    }
    with pytest.raises(ValueError, match=field_name):
        MissionProfile(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("initial_altitude_m", [-1.0, math.inf, math.nan])
def test_profile_initial_altitude_m_must_be_finite_and_non_negative(
    initial_altitude_m: float,
) -> None:
    with pytest.raises(ValueError, match="initial_altitude_m"):
        MissionProfile(
            phases=ps1_mission().phases,
            initial_altitude_m=initial_altitude_m,
            fuel_reserve_kg=5.0,
            min_usable_fuel_kg=20.0,
            max_mission_time_s=86400.0,
        )


def test_profile_requires_reserve_to_be_below_minimum_usable_fuel() -> None:
    with pytest.raises(ValueError, match=r"fuel_reserve_kg.*min_usable_fuel_kg"):
        MissionProfile(
            phases=ps1_mission().phases,
            initial_altitude_m=0.0,
            fuel_reserve_kg=20.0,
            min_usable_fuel_kg=20.0,
            max_mission_time_s=86400.0,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", "changed"),
        ("speed_mode", SpeedMode.BEST_LD),
        ("altitude_mode", AltitudeMode.CLIMB_TO),
        ("termination", Termination.RESOURCE),
        ("target_altitude_m", 1.0),
        ("speed_mps", 60.0),
        ("climb_rate_mps", 1.0),
        ("duration_s", 120.0),
    ],
)
def test_every_phase_field_is_immutable(field_name: str, value: object) -> None:
    phase = _phase()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(phase, field_name, value)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("phases", ()),
        ("initial_altitude_m", 1.0),
        ("fuel_reserve_kg", 6.0),
        ("min_usable_fuel_kg", 21.0),
        ("max_mission_time_s", 1.0),
    ],
)
def test_every_profile_field_is_immutable(field_name: str, value: object) -> None:
    mission = ps1_mission()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(mission, field_name, value)


def test_profile_stores_phases_as_a_tuple_even_when_given_a_list() -> None:
    mission = ps1_mission()
    rebuilt = MissionProfile(
        phases=list(mission.phases),  # type: ignore[arg-type]
        initial_altitude_m=mission.initial_altitude_m,
        fuel_reserve_kg=mission.fuel_reserve_kg,
        min_usable_fuel_kg=mission.min_usable_fuel_kg,
        max_mission_time_s=mission.max_mission_time_s,
    )
    assert isinstance(rebuilt.phases, tuple)


@pytest.mark.parametrize(
    ("cruise_altitude_m", "expected_duration_s"),
    [(3000.0, 1500.0), (8000.0, 4000.0)],
)
def test_implied_climb_duration_changes_plausibly_with_cruise_altitude(
    cruise_altitude_m: float,
    expected_duration_s: float,
) -> None:
    mission = ps1_mission(cruise_altitude_m=cruise_altitude_m)
    climb = mission.phase_by_name("climb")
    takeoff = mission.phase_by_name("takeoff")
    altitude_delta_m = climb.target_altitude_m - takeoff.target_altitude_m
    implied_duration_s = altitude_delta_m / float(climb.climb_rate_mps)
    assert implied_duration_s == pytest.approx(expected_duration_s)
    assert 1200.0 <= implied_duration_s <= 4500.0
