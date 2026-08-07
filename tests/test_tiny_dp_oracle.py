"""Exhaustive terminal-energy and Lagrangian-bound regression tests."""

from __future__ import annotations

import pytest

from src.analysis.tiny_dp_oracle import (
    enumerate_attainable_policies,
    minimum_fuel_by_terminal_energy,
    shadow_supported_policies,
    solve_tiny_bounds,
    tiny_regression_problem,
)


def test_exhaustive_oracle_replays_every_policy_through_one_transition() -> None:
    problem = tiny_regression_problem()
    policies = enumerate_attainable_policies(problem)
    minima = minimum_fuel_by_terminal_energy(policies)
    assert len(policies) == 208
    assert tuple(minima) == pytest.approx(
        (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    )
    assert minima[0.4].fuel_kg == pytest.approx(0.47)
    assert minima[0.4].restarts == 1


def test_true_pareto_frontier_excludes_the_dominated_exact_target_policy() -> None:
    minima = minimum_fuel_by_terminal_energy(
        enumerate_attainable_policies(tiny_regression_problem())
    )
    frontier = tuple(
        energy
        for energy, policy in minima.items()
        if not any(
            other_energy >= energy
            and other.fuel_kg <= policy.fuel_kg
            and (other_energy > energy or other.fuel_kg < policy.fuel_kg)
            for other_energy, other in minima.items()
        )
    )
    assert 0.4 not in frontier
    assert frontier == pytest.approx(
        (0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    )


def test_shadow_supported_policies_can_skip_an_attainable_target() -> None:
    problem = tiny_regression_problem()
    supported = shadow_supported_policies(
        enumerate_attainable_policies(problem), problem.terminal_target_kwh
    )
    energies = {policy.terminal_energy_kwh for policy in supported}
    assert 0.4 not in energies
    assert 0.1 in energies
    assert 0.5 in energies


def test_opposite_energy_policy_fuels_do_not_bound_exact_target_fuel() -> None:
    bounds = solve_tiny_bounds(tiny_regression_problem())
    assert bounds.lower_energy_policy.terminal_energy_kwh == pytest.approx(0.1)
    assert bounds.upper_energy_policy.terminal_energy_kwh == pytest.approx(0.5)
    assert bounds.lower_energy_policy.fuel_kg == pytest.approx(0.38)
    assert bounds.upper_energy_policy.fuel_kg == pytest.approx(0.44)
    assert bounds.exact_target_policy.fuel_kg == pytest.approx(0.47)
    assert bounds.exact_target_policy.fuel_kg > max(
        bounds.lower_energy_policy.fuel_kg,
        bounds.upper_energy_policy.fuel_kg,
    )


def test_lagrangian_dual_is_a_valid_lower_bound_with_a_known_gap() -> None:
    bounds = solve_tiny_bounds(tiny_regression_problem())
    assert bounds.dual_lower_bound_kg == pytest.approx(0.425)
    assert bounds.feasible_upper_bound_kg == pytest.approx(0.47)
    assert bounds.optimality_gap_kg == pytest.approx(0.045)
    assert bounds.dual_lower_bound_kg <= bounds.exact_target_policy.fuel_kg
