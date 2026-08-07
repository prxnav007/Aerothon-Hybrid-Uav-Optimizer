"""Finite-horizon DP state, transition, and endpoint-bracket tests."""

from __future__ import annotations

import pytest

from src.analysis.finite_horizon_dp import (
    FiniteHorizonDPProblem,
    clear_transition_kernel_cache,
    run_finite_horizon_scenarios,
    solve_finite_horizon_dp,
)
from src.analysis.mode_decomposition import select_post_crossing_window
from src.analysis.thermostat_reduction import reduce_dp_to_thermostat
from tests.test_baseline_regression import _reproduce_baseline


@pytest.fixture(scope="module")
def short_dp_result():
    clear_transition_kernel_cache()
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
        max_backward_inductions=16,
        scenario_name="targeted_short",
    )
    return solve_finite_horizon_dp(problem, progress=None), problem.steps, aircraft


def test_dp_reports_supported_endpoint_policies_without_a_fuel_bracket(
    short_dp_result,
) -> None:
    short_dp_result, _, _ = short_dp_result
    low, high = short_dp_result.endpoint_energy_interval_kwh
    assert short_dp_result.endpoint_target_bracketed
    assert low <= short_dp_result.target_energy_change_kwh <= high
    assert len(short_dp_result.policy_fuel_values_kg) == 2
    assert not hasattr(short_dp_result, "fuel_interval_kg")


def test_dp_dual_lower_bound_and_exact_discrete_upper_bound_are_ordered(
    short_dp_result,
) -> None:
    result, _, _ = short_dp_result
    assert result.feasible_upper_bound_policy is not None
    assert result.feasible_upper_bound_kg is not None
    assert result.dual_lower_bound_kg <= result.feasible_upper_bound_kg + 1.0e-12
    assert result.optimality_gap_kg == pytest.approx(
        result.feasible_upper_bound_kg - result.dual_lower_bound_kg
    )
    assert result.feasible_upper_bound_policy.discrete_terminal_target_residual_kwh == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_dp_enforces_dwell_and_charges_restart_fuel_exactly_once(
    short_dp_result,
) -> None:
    short_dp_result, _, _ = short_dp_result
    for result in (
        short_dp_result.lower_energy_policy,
        short_dp_result.upper_energy_policy,
    ):
        assert result.restart_fuel_kg == pytest.approx(0.1 * result.restart_count)
        assert result.dwell_violation_count == 0
        assert result.continuous_constraints_satisfied == (
            result.continuous_constraint_violations == ()
        )
        assert all(duration >= 180.0 for duration in result.on_durations_s[:-1])
        assert all(duration >= 120.0 for duration in result.off_durations_s[:-1])


def test_mission_depleting_dp_places_discharge_before_the_terminal_tail(
    short_dp_result,
) -> None:
    short_dp_result, _, _ = short_dp_result
    fractions = (
        short_dp_result.lower_energy_policy.depletion_before_final_tenth_fraction,
        short_dp_result.upper_energy_policy.depletion_before_final_tenth_fraction,
    )
    assert min(fractions) > 0.5
    for result in (
        short_dp_result.lower_energy_policy,
        short_dp_result.upper_energy_policy,
    ):
        assert result.minimum_soc >= result.problem.aircraft.battery.soc_min
        assert result.maximum_soc <= 1.0


def test_reduced_thermostat_retains_independent_thresholds_without_a_fuel_gap(
    short_dp_result,
) -> None:
    result, steps, aircraft = short_dp_result
    reduced = reduce_dp_to_thermostat(result, steps, aircraft)
    assert reduced.extracted_soc_low < reduced.extracted_soc_high
    assert reduced.preview_dependent
    assert reduced.endpoint_target_surrounded or reduced.replay_status.startswith(
        "unresolved:"
    )
    assert reduced.fuel_gap_status.startswith("invalid across unequal")


def test_transition_kernel_is_cached_for_the_same_scenario(short_dp_result) -> None:
    _, _, _ = short_dp_result
    result, _, _ = short_dp_result
    problem = result.lower_energy_policy.problem
    cached = solve_finite_horizon_dp(problem, progress=None)
    assert cached.kernel_cache_hit
    assert cached.kernel_build_runtime_s == 0.0


def test_repeated_shadow_policies_stop_before_the_induction_cap(
    short_dp_result,
) -> None:
    result, _, _ = short_dp_result
    assert result.backward_inductions < result.lower_energy_policy.problem.max_backward_inductions
    assert result.termination_reason in {
        "repeated_adjacent_policy_pair",
        "exact_shadow_supported_target",
        "identical_supported_policy",
    }


def test_scenario_csv_is_checkpointed_and_resumable(short_dp_result, tmp_path) -> None:
    result, _, _ = short_dp_result
    problem = result.lower_energy_policy.problem
    checkpoint = tmp_path / "dp_checkpoint.csv"
    first = run_finite_horizon_scenarios(
        (("short", problem),), checkpoint, progress=None
    )
    before = checkpoint.read_text(encoding="utf-8")
    resumed = run_finite_horizon_scenarios(
        (("short", problem),), checkpoint, progress=None
    )
    assert len(first) == 1
    assert resumed == ()
    assert checkpoint.read_text(encoding="utf-8") == before
