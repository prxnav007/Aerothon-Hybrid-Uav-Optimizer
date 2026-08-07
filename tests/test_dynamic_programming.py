"""Matched periodic dynamic-programming verification tests."""

from __future__ import annotations

import pytest

from src.analysis.dynamic_programming import PeriodicDPProblem, solve_periodic_dp


def _problem(**changes) -> PeriodicDPProblem:
    values = dict(
        demand_bus_kw=25.0,
        engine_max_kw=70.0,
        charge_limit_bus_kw=10.0,
        willans_a_kg_kwh=0.36,
        willans_b_kg_h=7.81,
        source_efficiency=0.9025,
        eta_charge=0.985,
        eta_discharge=0.985,
        energy_capacity_kwh=20.0,
        initial_energy_kwh=10.0,
        horizon_s=8.0 * 3600.0,
        timestep_s=60.0,
        energy_grid_points=401,
        action_grid_points=17,
    )
    values.update(changes)
    return PeriodicDPProblem(**values)


def test_periodic_dp_recovers_the_matched_analytical_fuel_and_duty() -> None:
    result = solve_periodic_dp(_problem())
    assert abs(result.fuel_error_fraction) < 3.0e-4
    assert abs(result.duty_error) < 5.0e-4
    assert abs(result.terminal_target_residual_kwh) <= 0.025
    assert abs(
        result.engine_on_power_mean_kw - result.analytical.engine_on_kw
    ) / result.analytical.engine_on_kw < 1.0e-3


def test_exact_periodic_construction_gives_valid_matched_problem_bounds() -> None:
    result = solve_periodic_dp(_problem(timestep_s=120.0))
    exact_residual = sum(
        (
            -result.problem.demand_bus_kw / result.problem.eta_discharge
            if action == 0.0
            else result.problem.eta_charge
            * (
                result.problem.source_efficiency * action
                - result.problem.demand_bus_kw
            )
        )
        * result.problem.timestep_s
        / 3600.0
        for action in result.exact_periodic_actions_kw
    )
    assert exact_residual == pytest.approx(0.0, abs=1.0e-12)
    assert result.analytical_lower_bound_kg <= result.feasible_upper_bound_kg
    assert result.optimality_gap_kg == pytest.approx(
        result.feasible_upper_bound_kg - result.analytical_lower_bound_kg
    )


def test_periodic_dp_uses_the_same_battery_sign_convention_as_the_cycle_model() -> None:
    result = solve_periodic_dp(_problem(horizon_s=4.0 * 3600.0))
    path = result.energy_trajectory_kwh
    for action, before, after in zip(result.actions_kw, path, path[1:]):
        if action == 0.0:
            assert after < before
        else:
            assert after > before


def test_action_refinement_does_not_manufacture_an_interior_optimum() -> None:
    coarse = solve_periodic_dp(_problem(action_grid_points=9))
    fine = solve_periodic_dp(_problem(action_grid_points=33))
    assert coarse.analytical.active_bound == "charge_ceiling"
    assert fine.analytical.active_bound == "charge_ceiling"
    assert abs(coarse.engine_on_power_mean_kw - fine.engine_on_power_mean_kw) < 0.002
    assert coarse.average_fuel_kg_h == pytest.approx(
        fine.average_fuel_kg_h, abs=3.0e-4
    )
