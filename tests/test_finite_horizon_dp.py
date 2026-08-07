"""Finite-horizon DP state, transition, and endpoint-bracket tests."""

from __future__ import annotations

import pytest

from src.analysis.finite_horizon_dp import (
    FiniteHorizonDPProblem,
    solve_finite_horizon_dp,
)
from src.analysis.mode_decomposition import select_post_crossing_window
from src.analysis.thermostat_reduction import reduce_dp_to_thermostat
from tests.test_baseline_regression import _reproduce_baseline


@pytest.fixture(scope="module")
def short_dp_result():
    _, _, mission, _, aircraft, _ = _reproduce_baseline()
    assert mission.log is not None
    window = select_post_crossing_window(
        mission.log, aircraft.battery.max_discharge_kw
    )
    problem = FiniteHorizonDPProblem(
        steps=tuple(window.steps[:60]),
        aircraft=aircraft,
        initial_soc=window.initial_soc,
        target_energy_change_kwh=-0.2,
        restart_fuel_kg=0.1,
        minimum_on_time_s=180.0,
        minimum_off_time_s=120.0,
        soc_grid_points=21,
        action_grid_points=5,
    )
    return solve_finite_horizon_dp(problem), problem.steps, aircraft


def test_dp_terminal_energy_is_reported_as_a_bracket_not_an_invalid_point(
    short_dp_result,
) -> None:
    short_dp_result, _, _ = short_dp_result
    low, high = short_dp_result.endpoint_energy_interval_kwh
    assert short_dp_result.endpoint_bracketed
    assert low <= short_dp_result.target_energy_change_kwh <= high
    assert low != high


def test_dp_enforces_dwell_and_charges_restart_fuel_exactly_once(
    short_dp_result,
) -> None:
    short_dp_result, _, _ = short_dp_result
    for result in (
        short_dp_result.lower_energy_result,
        short_dp_result.upper_energy_result,
    ):
        assert result.restart_fuel_kg == pytest.approx(0.1 * result.restart_count)
        assert all(duration >= 180.0 for duration in result.on_durations_s[:-1])
        assert all(duration >= 120.0 for duration in result.off_durations_s[:-1])


def test_mission_depleting_dp_places_discharge_before_the_terminal_tail(
    short_dp_result,
) -> None:
    short_dp_result, _, _ = short_dp_result
    fractions = (
        short_dp_result.lower_energy_result.depletion_before_final_tenth_fraction,
        short_dp_result.upper_energy_result.depletion_before_final_tenth_fraction,
    )
    assert min(fractions) > 0.5
    for result in (
        short_dp_result.lower_energy_result,
        short_dp_result.upper_energy_result,
    ):
        assert result.minimum_soc >= result.problem.aircraft.battery.soc_min
        assert result.maximum_soc <= 1.0


def test_reduced_thermostat_retains_independent_thresholds_and_energy_bracketing(
    short_dp_result,
) -> None:
    result, steps, aircraft = short_dp_result
    reduced = reduce_dp_to_thermostat(result, steps, aircraft)
    assert reduced.extracted_soc_low < reduced.extracted_soc_high
    assert reduced.preview_dependent
    assert reduced.endpoint_energy_matched or reduced.replay_status.startswith(
        "unresolved:"
    )
