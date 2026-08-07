"""Exhaustive terminal-energy oracle for a tiny hard-dwell dispatch problem."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass

__all__ = [
    "TinyAction",
    "TinyAttainablePolicy",
    "TinyBounds",
    "TinyDPProblem",
    "enumerate_attainable_policies",
    "minimum_fuel_by_terminal_energy",
    "shadow_supported_policies",
    "solve_tiny_bounds",
    "tiny_regression_problem",
]

_ENERGY_DIGITS = 12


@dataclass(frozen=True)
class TinyAction:
    """One OFF or ON action at one time index."""

    name: str
    engine_on: bool
    energy_change_kwh: float
    fuel_kg: float


@dataclass(frozen=True)
class TinyDPProblem:
    """Small deterministic problem whose complete policy set is enumerable."""

    actions_by_step: tuple[tuple[TinyAction, ...], ...]
    initial_energy_kwh: float
    capacity_kwh: float
    terminal_target_kwh: float
    initial_engine_on: bool
    initial_remaining_dwell_steps: int
    minimum_on_dwell_steps: int
    minimum_off_dwell_steps: int
    restart_fuel_kg: float


@dataclass(frozen=True)
class TinyAttainablePolicy:
    """Exact terminal state and cost of one admissible policy."""

    terminal_energy_kwh: float
    fuel_kg: float
    restarts: int
    policy: tuple[str, ...]
    action_indices: tuple[int, ...]
    policy_hash: str


@dataclass(frozen=True)
class TinyBounds:
    """Exact primal value and Lagrangian dual bound at one attainable target."""

    target_energy_kwh: float
    exact_target_policy: TinyAttainablePolicy
    lower_energy_policy: TinyAttainablePolicy
    upper_energy_policy: TinyAttainablePolicy
    dual_lower_bound_kg: float
    feasible_upper_bound_kg: float
    optimality_gap_kg: float
    best_multiplier_kg_kwh: float


def _energy_key(value: float) -> float:
    return round(float(value), _ENERGY_DIGITS)


def _policy_hash(indices: tuple[int, ...]) -> str:
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def _transition(
    problem: TinyDPProblem,
    *,
    time_index: int,
    energy_kwh: float,
    engine_on: bool,
    remaining_dwell_steps: int,
    action_index: int,
) -> tuple[float, bool, int, float, int] | None:
    action = problem.actions_by_step[time_index][action_index]
    if action.engine_on != engine_on and remaining_dwell_steps > 0:
        return None
    next_energy = _energy_key(energy_kwh + action.energy_change_kwh)
    if not -1.0e-12 <= next_energy <= problem.capacity_kwh + 1.0e-12:
        return None
    restarted = int(not engine_on and action.engine_on)
    if action.engine_on == engine_on:
        next_remaining = max(remaining_dwell_steps - 1, 0)
    else:
        dwell = (
            problem.minimum_on_dwell_steps
            if action.engine_on
            else problem.minimum_off_dwell_steps
        )
        next_remaining = max(dwell - 1, 0)
    fuel = action.fuel_kg + restarted * problem.restart_fuel_kg
    return next_energy, action.engine_on, next_remaining, fuel, restarted


def enumerate_attainable_policies(
    problem: TinyDPProblem,
) -> tuple[TinyAttainablePolicy, ...]:
    """Replay every action sequence through the oracle's shared transition."""
    action_ranges = [range(len(actions)) for actions in problem.actions_by_step]
    records: list[TinyAttainablePolicy] = []
    for indices_raw in itertools.product(*action_ranges):
        indices = tuple(int(index) for index in indices_raw)
        energy = problem.initial_energy_kwh
        engine_on = problem.initial_engine_on
        remaining = problem.initial_remaining_dwell_steps
        fuel = 0.0
        restarts = 0
        names: list[str] = []
        for time_index, action_index in enumerate(indices):
            transition = _transition(
                problem,
                time_index=time_index,
                energy_kwh=energy,
                engine_on=engine_on,
                remaining_dwell_steps=remaining,
                action_index=action_index,
            )
            if transition is None:
                break
            energy, engine_on, remaining, step_fuel, restarted = transition
            fuel += step_fuel
            restarts += restarted
            names.append(problem.actions_by_step[time_index][action_index].name)
        else:
            records.append(
                TinyAttainablePolicy(
                    terminal_energy_kwh=energy,
                    fuel_kg=fuel,
                    restarts=restarts,
                    policy=tuple(names),
                    action_indices=indices,
                    policy_hash=_policy_hash(indices),
                )
            )
    return tuple(records)


def minimum_fuel_by_terminal_energy(
    policies: tuple[TinyAttainablePolicy, ...],
) -> dict[float, TinyAttainablePolicy]:
    """Return the exact constrained optimum at each attainable endpoint."""
    minima: dict[float, TinyAttainablePolicy] = {}
    for policy in policies:
        key = _energy_key(policy.terminal_energy_kwh)
        incumbent = minima.get(key)
        if incumbent is None or (policy.fuel_kg, policy.policy) < (
            incumbent.fuel_kg,
            incumbent.policy,
        ):
            minima[key] = policy
    return dict(sorted(minima.items()))


def _multiplier_candidates(
    policies: tuple[TinyAttainablePolicy, ...],
) -> tuple[float, ...]:
    crossings = {0.0}
    for first, second in itertools.combinations(policies, 2):
        difference = first.terminal_energy_kwh - second.terminal_energy_kwh
        if abs(difference) > 1.0e-15:
            crossings.add((second.fuel_kg - first.fuel_kg) / difference)
    ordered = sorted(crossings)
    probes = set(ordered)
    probes.update(0.5 * (left + right) for left, right in zip(ordered, ordered[1:]))
    span = max((abs(value) for value in ordered), default=1.0) + 1.0
    probes.update((-span, span))
    return tuple(sorted(probes))


def _shadow_minimisers(
    policies: tuple[TinyAttainablePolicy, ...],
    target_kwh: float,
    multiplier: float,
) -> tuple[TinyAttainablePolicy, ...]:
    values = [
        policy.fuel_kg
        + multiplier * (policy.terminal_energy_kwh - target_kwh)
        for policy in policies
    ]
    minimum = min(values)
    return tuple(
        policy
        for policy, value in zip(policies, values)
        if math.isclose(value, minimum, rel_tol=0.0, abs_tol=1.0e-12)
    )


def shadow_supported_policies(
    policies: tuple[TinyAttainablePolicy, ...], target_kwh: float
) -> tuple[TinyAttainablePolicy, ...]:
    """Return every deterministic policy supported by a scalar multiplier."""
    supported: dict[str, TinyAttainablePolicy] = {}
    for multiplier in _multiplier_candidates(policies):
        for policy in _shadow_minimisers(policies, target_kwh, multiplier):
            supported[policy.policy_hash] = policy
    return tuple(
        sorted(
            supported.values(),
            key=lambda policy: (policy.terminal_energy_kwh, policy.fuel_kg),
        )
    )


def solve_tiny_bounds(problem: TinyDPProblem) -> TinyBounds:
    """Compute exhaustive primal and Lagrangian dual values."""
    policies = enumerate_attainable_policies(problem)
    minima = minimum_fuel_by_terminal_energy(policies)
    target = _energy_key(problem.terminal_target_kwh)
    if target not in minima:
        raise RuntimeError("tiny terminal target is not attainable")
    exact = minima[target]
    supported = shadow_supported_policies(policies, target)
    lower = max(
        (policy for policy in supported if policy.terminal_energy_kwh < target),
        key=lambda policy: policy.terminal_energy_kwh,
    )
    upper = min(
        (policy for policy in supported if policy.terminal_energy_kwh > target),
        key=lambda policy: policy.terminal_energy_kwh,
    )
    candidates = _multiplier_candidates(policies)
    dual_values = []
    for multiplier in candidates:
        dual = min(
            policy.fuel_kg
            + multiplier * (policy.terminal_energy_kwh - target)
            for policy in policies
        )
        dual_values.append((dual, multiplier))
    dual, multiplier = max(dual_values)
    return TinyBounds(
        target_energy_kwh=target,
        exact_target_policy=exact,
        lower_energy_policy=lower,
        upper_energy_policy=upper,
        dual_lower_bound_kg=dual,
        feasible_upper_bound_kg=exact.fuel_kg,
        optimality_gap_kg=exact.fuel_kg - dual,
        best_multiplier_kg_kwh=multiplier,
    )


def tiny_regression_problem() -> TinyDPProblem:
    """Return the non-convex six-step terminal-energy regression problem."""
    rows = (
        ((-0.3, 0.00), (0.0, 0.02), (0.1, 0.06)),
        ((-0.2, 0.00), (-0.1, 0.02), (0.2, 0.03)),
        ((-0.3, 0.00), (-0.2, 0.07), (0.0, 0.13)),
        ((-0.2, 0.00), (0.1, 0.06), (0.3, 0.11)),
        ((-0.1, 0.00), (0.0, 0.04), (0.3, 0.06)),
        ((-0.3, 0.00), (-0.1, 0.08), (0.2, 0.14)),
    )
    actions = tuple(
        tuple(
            TinyAction(name, index > 0, energy, fuel)
            for index, (name, (energy, fuel)) in enumerate(
                zip(("off", "on_low", "on_high"), row)
            )
        )
        for row in rows
    )
    return TinyDPProblem(
        actions_by_step=actions,
        initial_energy_kwh=0.4,
        capacity_kwh=1.0,
        terminal_target_kwh=0.4,
        initial_engine_on=False,
        initial_remaining_dwell_steps=0,
        minimum_on_dwell_steps=3,
        minimum_off_dwell_steps=1,
        restart_fuel_kg=0.2,
    )
