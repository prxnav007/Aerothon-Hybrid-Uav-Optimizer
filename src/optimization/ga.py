"""Deterministic constrained real-coded genetic algorithm.

The engine operates only on normalized chromosomes and an injected fitness
contract. It contains no aircraft, controller, or simulator knowledge.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import fmean, median
from typing import Any, Literal, Protocol, TypeVar

import numpy as np

from src.optimization.chromosome import (
    CHROMOSOME_SCHEMA_VERSION,
    GENE_NAMES,
    NormalizedChromosome,
    PlantThermostatDesignSpace,
    ideal_restart_fuel_seed,
    practical_thermostat_seed,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "EVALUATION_LEDGER_SCHEMA_VERSION",
    "GA_CONFIG_SCHEMA_VERSION",
    "GAConfig",
    "GABudget",
    "GAProgress",
    "GAResult",
    "GenerationRecord",
    "InitialPopulation",
    "EvaluatedIndividual",
    "EvaluationStatistics",
    "FitnessCodec",
    "FitnessLike",
    "ReproducibilityMetadata",
    "StagnationState",
    "bounded_polynomial_mutation",
    "bounded_sbx",
    "compare_fitness",
    "fitness_rank_key",
    "fitness_result_codec",
    "initialize_population",
    "maximum_candidate_placements",
    "theoretical_evaluation_budget",
    "rank_population",
    "run_ga",
    "tournament_select",
    "update_stagnation",
]

GA_CONFIG_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
EVALUATION_LEDGER_SCHEMA_VERSION = 1
_GA_RESULT_SCHEMA_VERSION = 1
# Feasible results inherit the normalization roundoff convention in OPT-03.
_FEASIBLE_VIOLATION_TOLERANCE = 1.0e-12
_UNIT_ROUNDOFF_TOLERANCE = 8.0 * math.ulp(1.0)

TerminationReason = Literal[
    "max_generations",
    "stagnation",
    "externally_interrupted",
    "completed_requested_generation",
]
InitializationOrigin = Literal[
    "latin_hypercube",
    "practical_seed",
    "ideal_seed",
    "practical_perturbation",
    "ideal_perturbation",
]


class FitnessLike(Protocol):
    """Runtime fields consumed from the existing fitness result contract."""

    schema_version: int
    chromosome_cache_key: str
    fitness_scenario_id: str
    evaluation_key: str
    static_feasible: bool
    dynamically_feasible: bool
    objective_loiter_seconds: float | None
    combined_static_dynamic_violation: float
    run_mission_called: bool

    def to_dict(self) -> Mapping[str, Any]: ...


FitnessT = TypeVar("FitnessT", bound=FitnessLike)


def _finite(name: str, value: Any) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(name: str, value: Any, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return {"binary64": float(value).hex()}
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _digest(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class GAConfig:
    """Immutable algorithm settings; max generations includes generation zero."""

    population_size: int = 64
    max_generations: int = 40
    elite_count: int = 2
    tournament_size: int = 3
    crossover_probability: float = 0.90
    sbx_distribution_index: float = 15.0
    mutation_probability_per_gene: float = 1.0 / len(GENE_NAMES)
    mutation_distribution_index: float = 20.0
    early_stop_patience: int = 10
    material_relative_improvement: float = 1.0e-4
    random_seed: int = 20260808
    initial_perturbation_scale: float = 0.05
    duplicate_retry_limit: int = 32

    def __post_init__(self) -> None:
        population = _integer("population_size", self.population_size, minimum=4)
        generations = _integer("max_generations", self.max_generations, minimum=1)
        elites = _integer("elite_count", self.elite_count, minimum=1)
        tournament = _integer("tournament_size", self.tournament_size, minimum=2)
        patience = _integer("early_stop_patience", self.early_stop_patience, minimum=1)
        retries = _integer("duplicate_retry_limit", self.duplicate_retry_limit, minimum=1)
        seed = _integer("random_seed", self.random_seed, minimum=0)
        if seed > np.iinfo(np.uint64).max:
            raise ValueError("random_seed must fit an unsigned 64-bit integer")
        if elites > population - 2:
            raise ValueError("elite_count must lie in [1, population_size - 2]")
        if tournament > population:
            raise ValueError("tournament_size must not exceed population_size")
        probability_fields = (
            "crossover_probability",
            "mutation_probability_per_gene",
        )
        for name in probability_fields:
            value = _finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        positive_fields = (
            "sbx_distribution_index",
            "mutation_distribution_index",
            "material_relative_improvement",
            "initial_perturbation_scale",
        )
        for name in positive_fields:
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.initial_perturbation_scale > 1.0:
            raise ValueError("initial_perturbation_scale must not exceed 1")
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "max_generations", generations)
        object.__setattr__(self, "elite_count", elites)
        object.__setattr__(self, "tournament_size", tournament)
        object.__setattr__(self, "early_stop_patience", patience)
        object.__setattr__(self, "duplicate_retry_limit", retries)
        object.__setattr__(self, "random_seed", seed)

    @property
    def identity(self) -> str:
        return _digest(
            f"ga-config-v{GA_CONFIG_SCHEMA_VERSION}",
            {"schema_version": GA_CONFIG_SCHEMA_VERSION, **asdict(self)},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GA_CONFIG_SCHEMA_VERSION,
            "ga_config_id": self.identity,
            "generation_convention": "max_generations_total_including_generation_zero",
            **asdict(self),
        }


def maximum_candidate_placements(config: GAConfig, *, independent_seeds: int = 1) -> int:
    """Return the pre-cache placement upper bound for the configured run count."""
    if not isinstance(config, GAConfig):
        raise ValueError("config must be a GAConfig")
    seeds = _integer("independent_seeds", independent_seeds, minimum=1)
    per_seed = config.population_size + (config.max_generations - 1) * (
        config.population_size - config.elite_count
    )
    return seeds * per_seed


@dataclass(frozen=True)
class GABudget:
    initial_population_size: int
    later_population_count: int
    offspring_per_later_population: int
    independent_seeds: int
    candidate_placements_per_seed: int
    unique_evaluations_per_seed_upper_bound: int
    total_candidate_placements: int
    total_unique_evaluations_upper_bound: int


def theoretical_evaluation_budget(
    config: GAConfig, *, independent_seeds: int = 1
) -> GABudget:
    """Report placement and unique-evaluation bounds before cache effects."""
    seeds = _integer("independent_seeds", independent_seeds, minimum=1)
    per_seed = maximum_candidate_placements(config)
    return GABudget(
        initial_population_size=config.population_size,
        later_population_count=config.max_generations - 1,
        offspring_per_later_population=config.population_size - config.elite_count,
        independent_seeds=seeds,
        candidate_placements_per_seed=per_seed,
        unique_evaluations_per_seed_upper_bound=per_seed,
        total_candidate_placements=seeds * per_seed,
        total_unique_evaluations_upper_bound=seeds * per_seed,
    )


@dataclass(frozen=True)
class FitnessCodec:
    """Injected durable serialization for the evaluator's existing result type."""

    identity: str
    serialize: Callable[[FitnessLike], Mapping[str, Any]]
    deserialize: Callable[[Mapping[str, Any]], FitnessLike]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("codec identity must be a non-empty string")
        if not callable(self.serialize) or not callable(self.deserialize):
            raise ValueError("codec serialize and deserialize must be callable")


@dataclass(frozen=True)
class InitialPopulation:
    chromosomes: tuple[NormalizedChromosome, ...]
    origins: tuple[InitializationOrigin, ...]

    def __post_init__(self) -> None:
        if len(self.chromosomes) != len(self.origins):
            raise ValueError("initial chromosomes and origins must have equal lengths")

    @property
    def latin_hypercube_count(self) -> int:
        return self.origins.count("latin_hypercube")

    @property
    def seeded_count(self) -> int:
        return len(self.origins) - self.latin_hypercube_count


def _reflect_unit(value: float) -> float:
    reflected = abs(_finite("normalized gene", value)) % 2.0
    return 2.0 - reflected if reflected > 1.0 else reflected


def _validated_unit(value: float) -> float:
    result = _finite("normalized gene", value)
    if -_UNIT_ROUNDOFF_TOLERANCE <= result <= 0.0:
        return 0.0
    if 1.0 <= result <= 1.0 + _UNIT_ROUNDOFF_TOLERANCE:
        return 1.0
    if not 0.0 <= result <= 1.0:
        raise ArithmeticError(f"bounded operator produced {result!r}")
    return result


def _fresh_uniform(rng: np.random.Generator) -> NormalizedChromosome:
    return NormalizedChromosome(tuple(float(item) for item in rng.random(len(GENE_NAMES))))


def _perturbed(
    anchor: NormalizedChromosome,
    rng: np.random.Generator,
    scale: float,
) -> NormalizedChromosome:
    offsets = rng.normal(0.0, scale, len(GENE_NAMES))
    genes = tuple(
        _reflect_unit(gene + float(offset))
        for gene, offset in zip(anchor.genes, offsets)
    )
    return NormalizedChromosome(genes)


def _latin_hypercube(
    count: int, rng: np.random.Generator
) -> tuple[NormalizedChromosome, ...]:
    if count == 0:
        return ()
    columns = []
    for _ in GENE_NAMES:
        points = (np.arange(count, dtype=float) + rng.random(count)) / count
        columns.append(points[rng.permutation(count)])
    return tuple(
        NormalizedChromosome(
            tuple(float(columns[column][row]) for column in range(len(GENE_NAMES)))
        )
        for row in range(count)
    )


def initialize_population(
    config: GAConfig,
    *,
    bounds: PlantThermostatDesignSpace,
    rng: np.random.Generator,
) -> InitialPopulation:
    """Build deterministic LHS coverage plus exact and locally perturbed anchors."""
    if not isinstance(config, GAConfig):
        raise ValueError("config must be a GAConfig")
    if not isinstance(bounds, PlantThermostatDesignSpace):
        raise ValueError("bounds must be a PlantThermostatDesignSpace")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")

    seeded_target = min(16, max(2, config.population_size // 4))
    global_target = config.population_size - seeded_target
    practical = practical_thermostat_seed(bounds=bounds)
    ideal = ideal_restart_fuel_seed(bounds=bounds)
    chromosomes: list[NormalizedChromosome] = [practical, ideal]
    origins: list[InitializationOrigin] = ["practical_seed", "ideal_seed"]
    used = {item.cache_key(bounds=bounds) for item in chromosomes}

    perturbation_count = seeded_target - 2
    practical_count = (perturbation_count + 1) // 2
    ideal_count = perturbation_count - practical_count
    for anchor, count, origin in (
        (practical, practical_count, "practical_perturbation"),
        (ideal, ideal_count, "ideal_perturbation"),
    ):
        for _ in range(count):
            candidate = anchor
            for _attempt in range(config.duplicate_retry_limit):
                candidate = _perturbed(anchor, rng, config.initial_perturbation_scale)
                if candidate.cache_key(bounds=bounds) not in used:
                    break
            else:
                for _attempt in range(config.duplicate_retry_limit):
                    candidate = _fresh_uniform(rng)
                    if candidate.cache_key(bounds=bounds) not in used:
                        break
            key = candidate.cache_key(bounds=bounds)
            if key in used:
                raise RuntimeError("failed to create a unique seeded chromosome")
            chromosomes.append(candidate)
            origins.append(origin)
            used.add(key)

    for candidate in _latin_hypercube(global_target, rng):
        key = candidate.cache_key(bounds=bounds)
        if key in used:
            for _attempt in range(config.duplicate_retry_limit):
                candidate = _fresh_uniform(rng)
                key = candidate.cache_key(bounds=bounds)
                if key not in used:
                    break
        if key in used:
            raise RuntimeError("failed to create a unique Latin-hypercube chromosome")
        chromosomes.append(candidate)
        origins.append("latin_hypercube")
        used.add(key)
    return InitialPopulation(tuple(chromosomes), tuple(origins))


def _fitness_fields(result: FitnessLike) -> tuple[bool, float | None, float, str]:
    try:
        static_feasible = result.static_feasible
        dynamic_feasible = result.dynamically_feasible
        objective = result.objective_loiter_seconds
        violation = _finite(
            "combined_static_dynamic_violation",
            result.combined_static_dynamic_violation,
        )
        tie_key = result.evaluation_key or result.chromosome_cache_key
    except AttributeError as error:
        raise TypeError("evaluator result does not implement the fitness contract") from error
    if type(static_feasible) is not bool or type(dynamic_feasible) is not bool:
        raise ValueError("fitness feasibility fields must be boolean")
    if dynamic_feasible and not static_feasible:
        raise ValueError("dynamic feasibility requires static feasibility")
    if violation < 0.0:
        raise ValueError("combined normalized violation must be non-negative")
    if not isinstance(tie_key, str) or not tie_key:
        raise ValueError("fitness result requires a deterministic evaluation key")
    feasible = static_feasible and dynamic_feasible
    if feasible:
        if objective is None:
            raise ValueError("feasible fitness result is missing its loiter objective")
        objective = _finite("objective_loiter_seconds", objective)
        if violation > _FEASIBLE_VIOLATION_TOLERANCE:
            raise ValueError("feasible fitness result has material normalized violation")
    elif violation <= _FEASIBLE_VIOLATION_TOLERANCE:
        raise ValueError("infeasible fitness result requires a nonzero violation")
    return feasible, objective, violation, tie_key


def fitness_rank_key(result: FitnessLike) -> tuple[int, float, str]:
    """Return the sole Deb-style deterministic ordering key."""
    feasible, objective, violation, tie_key = _fitness_fields(result)
    if feasible:
        assert objective is not None
        return (0, -objective, tie_key)
    return (1, violation, tie_key)


def compare_fitness(left: FitnessLike, right: FitnessLike) -> int:
    """Return -1 when left ranks better, +1 when right ranks better."""
    left_key = fitness_rank_key(left)
    right_key = fitness_rank_key(right)
    return -1 if left_key < right_key else 1 if left_key > right_key else 0


@dataclass(frozen=True)
class EvaluatedIndividual:
    chromosome: NormalizedChromosome
    fitness: FitnessLike

    @property
    def chromosome_cache_key(self) -> str:
        return self.fitness.chromosome_cache_key

    @property
    def evaluation_key(self) -> str:
        return self.fitness.evaluation_key


def rank_population(
    population: Sequence[EvaluatedIndividual],
) -> tuple[EvaluatedIndividual, ...]:
    if not population:
        raise ValueError("population must not be empty")
    return tuple(sorted(population, key=lambda item: fitness_rank_key(item.fitness)))


def tournament_select(
    population: Sequence[EvaluatedIndividual],
    *,
    tournament_size: int,
    rng: np.random.Generator,
) -> EvaluatedIndividual:
    """Sample indices with replacement and return the Deb-ranked winner."""
    if not population:
        raise ValueError("population must not be empty")
    size = _integer("tournament_size", tournament_size, minimum=2)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    indices = rng.integers(0, len(population), size=size)
    competitors = tuple(population[int(index)] for index in indices)
    return min(competitors, key=lambda item: fitness_rank_key(item.fitness))


def bounded_sbx(
    parent_a: NormalizedChromosome,
    parent_b: NormalizedChromosome,
    *,
    distribution_index: float,
    crossover_probability: float,
    rng: np.random.Generator,
) -> tuple[NormalizedChromosome, NormalizedChromosome]:
    """Apply Deb's bounded SBX equations once to a normalized parent pair."""
    eta = _finite("distribution_index", distribution_index)
    probability = _finite("crossover_probability", crossover_probability)
    if eta <= 0.0 or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid bounded-SBX parameters")
    if rng.random() > probability:
        return (
            NormalizedChromosome(tuple(parent_a.genes)),
            NormalizedChromosome(tuple(parent_b.genes)),
        )
    child_a = list(parent_a.genes)
    child_b = list(parent_b.genes)
    for index, (raw_a, raw_b) in enumerate(zip(parent_a.genes, parent_b.genes)):
        if rng.random() > 0.5 or abs(raw_a - raw_b) <= 1.0e-14:
            continue
        lower_parent, upper_parent = sorted((raw_a, raw_b))
        random_value = float(rng.random())
        beta = 1.0 + 2.0 * lower_parent / (upper_parent - lower_parent)
        alpha = 2.0 - beta ** -(eta + 1.0)
        if random_value <= 1.0 / alpha:
            beta_q = (random_value * alpha) ** (1.0 / (eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - random_value * alpha)) ** (1.0 / (eta + 1.0))
        first = 0.5 * (
            lower_parent + upper_parent - beta_q * (upper_parent - lower_parent)
        )
        beta = 1.0 + 2.0 * (1.0 - upper_parent) / (upper_parent - lower_parent)
        alpha = 2.0 - beta ** -(eta + 1.0)
        if random_value <= 1.0 / alpha:
            beta_q = (random_value * alpha) ** (1.0 / (eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - random_value * alpha)) ** (1.0 / (eta + 1.0))
        second = 0.5 * (
            lower_parent + upper_parent + beta_q * (upper_parent - lower_parent)
        )
        first = _validated_unit(first)
        second = _validated_unit(second)
        if rng.random() <= 0.5:
            child_a[index], child_b[index] = second, first
        else:
            child_a[index], child_b[index] = first, second
    return NormalizedChromosome(tuple(child_a)), NormalizedChromosome(tuple(child_b))


def bounded_polynomial_mutation(
    chromosome: NormalizedChromosome,
    *,
    probability_per_gene: float,
    distribution_index: float,
    rng: np.random.Generator,
) -> NormalizedChromosome:
    """Apply Deb's bounded polynomial mutation independently to every gene."""
    probability = _finite("probability_per_gene", probability_per_gene)
    eta = _finite("distribution_index", distribution_index)
    if not 0.0 <= probability <= 1.0 or eta <= 0.0:
        raise ValueError("invalid bounded polynomial-mutation parameters")
    mutated = list(chromosome.genes)
    power = 1.0 / (eta + 1.0)
    for index, value in enumerate(chromosome.genes):
        if rng.random() > probability:
            continue
        random_value = float(rng.random())
        if random_value <= 0.5:
            base = 1.0 - value
            term = 2.0 * random_value + (1.0 - 2.0 * random_value) * base ** (
                eta + 1.0
            )
            delta = term**power - 1.0
        else:
            base = value
            term = 2.0 * (1.0 - random_value) + 2.0 * (
                random_value - 0.5
            ) * base ** (eta + 1.0)
            delta = 1.0 - term**power
        mutated[index] = _validated_unit(value + delta)
    return NormalizedChromosome(tuple(mutated))


@dataclass(frozen=True)
class EvaluationStatistics:
    candidate_placements: int = 0
    unique_fitness_evaluations: int = 0
    mission_calls: int = 0
    static_infeasible_results: int = 0
    dynamic_infeasible_results: int = 0
    feasible_results: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class GAProgress:
    generation: int
    candidate_index: int
    candidate_context: str
    fitness: FitnessLike
    cache_hit: bool
    evaluation_runtime_s: float
    statistics: EvaluationStatistics


@dataclass
class _MutableStatistics:
    candidate_placements: int = 0
    unique_fitness_evaluations: int = 0
    mission_calls: int = 0
    static_infeasible_results: int = 0
    dynamic_infeasible_results: int = 0
    feasible_results: int = 0
    cache_hits: int = 0

    def snapshot(self) -> EvaluationStatistics:
        return EvaluationStatistics(**asdict(self))

    @classmethod
    def from_snapshot(cls, snapshot: EvaluationStatistics) -> _MutableStatistics:
        return cls(**asdict(snapshot))


@dataclass(frozen=True)
class StagnationState:
    material_best_objective: float | None = None
    exact_best_objective: float | None = None
    stagnant_generation_count: int = 0

    def __post_init__(self) -> None:
        if self.material_best_objective is not None:
            _finite("material_best_objective", self.material_best_objective)
        if self.exact_best_objective is not None:
            _finite("exact_best_objective", self.exact_best_objective)
        _integer(
            "stagnant_generation_count", self.stagnant_generation_count, minimum=0
        )


def update_stagnation(
    state: StagnationState,
    best_feasible_objective: float | None,
    *,
    material_relative_improvement: float,
) -> tuple[StagnationState, bool]:
    """Update exact and material-best states without claiming convergence."""
    if not isinstance(state, StagnationState):
        raise ValueError("state must be a StagnationState")
    relative = _finite(
        "material_relative_improvement", material_relative_improvement
    )
    if relative <= 0.0:
        raise ValueError("material_relative_improvement must be positive")
    if best_feasible_objective is None:
        return state, False
    current = _finite("best_feasible_objective", best_feasible_objective)
    exact = (
        current
        if state.exact_best_objective is None
        else max(state.exact_best_objective, current)
    )
    if state.material_best_objective is None:
        return StagnationState(current, exact, 0), True
    threshold = relative * max(abs(state.material_best_objective), 1.0)
    if exact > state.material_best_objective + threshold:
        return StagnationState(exact, exact, 0), True
    return StagnationState(
        state.material_best_objective,
        exact,
        state.stagnant_generation_count + 1,
    ), False


@dataclass(frozen=True)
class GenerationRecord:
    run_seed: int
    generation: int
    population_size: int
    feasible_count: int
    feasible_fraction: float
    static_infeasible_count: int
    dynamic_infeasible_count: int
    best_feasible_objective: float | None
    median_feasible_objective: float | None
    worst_feasible_objective: float | None
    best_normalized_infeasible_violation: float | None
    mean_normalized_gene_diversity: float
    best_chromosome_key: str
    cumulative_candidate_placements: int
    cumulative_unique_evaluations: int
    cumulative_mission_calls: int
    cumulative_cache_hits: int
    elapsed_wall_time_s: float
    material_improvement: bool
    stagnation_counter: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _generation_record_from_dict(record: Mapping[str, Any]) -> GenerationRecord:
    try:
        return GenerationRecord(**dict(record))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid generation record in checkpoint") from error


def _population_counts(
    population: Sequence[EvaluatedIndividual],
) -> tuple[int, int, int]:
    feasible = static = dynamic = 0
    for individual in population:
        is_feasible, _, _, _ = _fitness_fields(individual.fitness)
        if is_feasible:
            feasible += 1
        elif individual.fitness.static_feasible:
            dynamic += 1
        else:
            static += 1
    return feasible, static, dynamic


def _gene_diversity(population: Sequence[EvaluatedIndividual]) -> float:
    matrix = np.asarray([item.chromosome.genes for item in population], dtype=float)
    return float(np.mean(np.std(matrix, axis=0, ddof=0)))


def _make_generation_record(
    population: Sequence[EvaluatedIndividual],
    *,
    generation: int,
    config: GAConfig,
    statistics: EvaluationStatistics,
    elapsed_wall_time_s: float,
    material_improvement: bool,
    stagnation_state: StagnationState,
) -> GenerationRecord:
    ranked = rank_population(population)
    feasible_count, static_count, dynamic_count = _population_counts(ranked)
    feasible_objectives = sorted(
        float(item.fitness.objective_loiter_seconds)
        for item in ranked
        if item.fitness.static_feasible and item.fitness.dynamically_feasible
    )
    infeasible_violations = [
        float(item.fitness.combined_static_dynamic_violation)
        for item in ranked
        if not (item.fitness.static_feasible and item.fitness.dynamically_feasible)
    ]
    return GenerationRecord(
        run_seed=config.random_seed,
        generation=generation,
        population_size=len(ranked),
        feasible_count=feasible_count,
        feasible_fraction=feasible_count / len(ranked),
        static_infeasible_count=static_count,
        dynamic_infeasible_count=dynamic_count,
        best_feasible_objective=(
            max(feasible_objectives) if feasible_objectives else None
        ),
        median_feasible_objective=(
            float(median(feasible_objectives)) if feasible_objectives else None
        ),
        worst_feasible_objective=(
            min(feasible_objectives) if feasible_objectives else None
        ),
        best_normalized_infeasible_violation=(
            min(infeasible_violations) if infeasible_violations else None
        ),
        mean_normalized_gene_diversity=_gene_diversity(ranked),
        best_chromosome_key=ranked[0].chromosome_cache_key,
        cumulative_candidate_placements=statistics.candidate_placements,
        cumulative_unique_evaluations=statistics.unique_fitness_evaluations,
        cumulative_mission_calls=statistics.mission_calls,
        cumulative_cache_hits=statistics.cache_hits,
        elapsed_wall_time_s=_finite("elapsed_wall_time_s", elapsed_wall_time_s),
        material_improvement=material_improvement,
        stagnation_counter=stagnation_state.stagnant_generation_count,
    )


def _validate_result_for_candidate(
    result: FitnessLike,
    chromosome: NormalizedChromosome,
    *,
    bounds: PlantThermostatDesignSpace,
    fitness_scenario_id: str,
) -> tuple[bool, float, str, str]:
    feasible, _, violation, evaluation_key = _fitness_fields(result)
    expected_chromosome_key = chromosome.cache_key(bounds=bounds)
    try:
        chromosome_key = result.chromosome_cache_key
        scenario_id = result.fitness_scenario_id
        mission_called = result.run_mission_called
        schema_version = result.schema_version
    except AttributeError as error:
        raise TypeError("evaluator result does not implement the fitness contract") from error
    if chromosome_key != expected_chromosome_key:
        raise ValueError("fitness chromosome cache key does not match its candidate")
    if scenario_id != fitness_scenario_id:
        raise ValueError("fitness result belongs to a different scenario")
    if type(mission_called) is not bool:
        raise ValueError("run_mission_called must be boolean")
    _integer("fitness schema_version", schema_version, minimum=1)
    return feasible, violation, chromosome_key, evaluation_key


def _classify_new_result(result: FitnessLike, statistics: _MutableStatistics) -> None:
    if result.static_feasible and result.dynamically_feasible:
        statistics.feasible_results += 1
    elif result.static_feasible:
        statistics.dynamic_infeasible_results += 1
    else:
        statistics.static_infeasible_results += 1
    statistics.mission_calls += int(result.run_mission_called)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class _LedgerEntry:
    chromosome: NormalizedChromosome
    fitness: FitnessLike


class _EvaluationLedger:
    def __init__(
        self,
        path: Path,
        *,
        config: GAConfig,
        bounds: PlantThermostatDesignSpace,
        fitness_scenario_id: str,
        codec: FitnessCodec,
    ) -> None:
        self.path = path
        self.config = config
        self.bounds = bounds
        self.fitness_scenario_id = fitness_scenario_id
        self.codec = codec

    def append(
        self,
        chromosome: NormalizedChromosome,
        result: FitnessLike,
        *,
        generation: int,
        candidate_index: int,
        candidate_context: str,
        runtime_s: float,
    ) -> None:
        serialized = _json_ready(self.codec.serialize(result))
        payload = {
            "schema_version": EVALUATION_LEDGER_SCHEMA_VERSION,
            "ga_config_id": self.config.identity,
            "design_space_id": self.bounds.identifier,
            "fitness_scenario_id": self.fitness_scenario_id,
            "fitness_codec_id": self.codec.identity,
            "evaluation_key": result.evaluation_key,
            "chromosome_cache_key": result.chromosome_cache_key,
            "chromosome_genes": list(chromosome.genes),
            "static_feasible": result.static_feasible,
            "dynamically_feasible": result.dynamically_feasible,
            "objective_loiter_seconds": result.objective_loiter_seconds,
            "combined_normalized_violation": (
                result.combined_static_dynamic_violation
            ),
            "run_mission_called": result.run_mission_called,
            "generation": generation,
            "candidate_index": candidate_index,
            "candidate_context": candidate_context,
            "evaluated_at_utc": _utc_timestamp(),
            "evaluation_runtime_s": runtime_s,
            "fitness_result": serialized,
        }
        payload["record_digest"] = _digest("ga-evaluation-record-v1", payload)
        encoded = json.dumps(
            payload, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def load(self) -> tuple[tuple[_LedgerEntry, ...], tuple[str, ...]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return (), ()
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        entries: list[_LedgerEntry] = []
        warnings: list[str] = []
        evaluation_keys: set[str] = set()
        chromosome_keys: set[str] = set()
        for index, raw_line in enumerate(lines):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if index == len(lines) - 1:
                    warnings.append("ignored_interrupted_final_evaluation_ledger_line")
                    self._discard_interrupted_tail(lines[:index])
                    break
                raise ValueError(
                    f"corrupt evaluation ledger line {index + 1}"
                ) from error
            self._validate_record(record, index + 1)
            chromosome = NormalizedChromosome(tuple(record["chromosome_genes"]))
            result = self.codec.deserialize(record["fitness_result"])
            _, _, chromosome_key, evaluation_key = _validate_result_for_candidate(
                result,
                chromosome,
                bounds=self.bounds,
                fitness_scenario_id=self.fitness_scenario_id,
            )
            if evaluation_key != record["evaluation_key"]:
                raise ValueError("ledger evaluation key does not match serialized fitness")
            if chromosome_key != record["chromosome_cache_key"]:
                raise ValueError("ledger chromosome key does not match serialized fitness")
            if evaluation_key in evaluation_keys or chromosome_key in chromosome_keys:
                raise ValueError("evaluation ledger contains a duplicate committed result")
            evaluation_keys.add(evaluation_key)
            chromosome_keys.add(chromosome_key)
            entries.append(_LedgerEntry(chromosome, result))
        return tuple(entries), tuple(warnings)

    def _discard_interrupted_tail(self, valid_lines: Sequence[str]) -> None:
        temporary = self.path.with_name(self.path.name + ".repair.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            for line in valid_lines:
                stream.write(line if line.endswith(("\n", "\r")) else line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def _validate_record(self, record: Mapping[str, Any], line_number: int) -> None:
        if record.get("schema_version") != EVALUATION_LEDGER_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation ledger schema at line {line_number}")
        checks = (
            ("ga_config_id", self.config.identity),
            ("design_space_id", self.bounds.identifier),
            ("fitness_scenario_id", self.fitness_scenario_id),
            ("fitness_codec_id", self.codec.identity),
        )
        for field, expected in checks:
            if record.get(field) != expected:
                raise ValueError(f"evaluation ledger {field} mismatch")
        payload = dict(record)
        recorded_digest = payload.pop("record_digest", None)
        if recorded_digest != _digest("ga-evaluation-record-v1", payload):
            raise ValueError(f"evaluation ledger digest mismatch at line {line_number}")


class _EvaluationCache:
    def __init__(
        self,
        *,
        bounds: PlantThermostatDesignSpace,
        fitness_scenario_id: str,
        statistics: _MutableStatistics | None = None,
        ledger: _EvaluationLedger | None = None,
        progress: Callable[[GAProgress], None] | None = None,
    ) -> None:
        self.bounds = bounds
        self.fitness_scenario_id = fitness_scenario_id
        self.statistics = statistics or _MutableStatistics()
        self.ledger = ledger
        self.progress = progress
        self._by_evaluation_key: dict[str, FitnessLike] = {}
        self._evaluation_key_by_chromosome_key: dict[str, str] = {}

    def restore(self, entries: Sequence[_LedgerEntry]) -> None:
        for entry in entries:
            _, _, chromosome_key, evaluation_key = _validate_result_for_candidate(
                entry.fitness,
                entry.chromosome,
                bounds=self.bounds,
                fitness_scenario_id=self.fitness_scenario_id,
            )
            if evaluation_key in self._by_evaluation_key:
                raise ValueError("duplicate evaluation key while restoring cache")
            self._by_evaluation_key[evaluation_key] = entry.fitness
            self._evaluation_key_by_chromosome_key[chromosome_key] = evaluation_key

    def result_for_evaluation_key(self, evaluation_key: str) -> FitnessLike:
        try:
            return self._by_evaluation_key[evaluation_key]
        except KeyError as error:
            raise ValueError("checkpoint references a missing ledger evaluation") from error

    def evaluate(
        self,
        chromosome: NormalizedChromosome,
        evaluator: Callable[[NormalizedChromosome], FitnessLike],
        *,
        generation: int,
        candidate_index: int,
        candidate_context: str,
    ) -> FitnessLike:
        self.statistics.candidate_placements += 1
        chromosome_key = chromosome.cache_key(bounds=self.bounds)
        evaluation_key = self._evaluation_key_by_chromosome_key.get(chromosome_key)
        if evaluation_key is not None:
            self.statistics.cache_hits += 1
            result = self._by_evaluation_key[evaluation_key]
            if self.progress is not None:
                self.progress(
                    GAProgress(
                        generation, candidate_index, candidate_context, result,
                        True, 0.0, self.statistics.snapshot(),
                    )
                )
            return result
        started = time.perf_counter()
        result = evaluator(chromosome)
        runtime = time.perf_counter() - started
        _, _, returned_chromosome_key, evaluation_key = _validate_result_for_candidate(
            result,
            chromosome,
            bounds=self.bounds,
            fitness_scenario_id=self.fitness_scenario_id,
        )
        if evaluation_key in self._by_evaluation_key:
            raise ValueError("different chromosomes returned the same evaluation key")
        self._by_evaluation_key[evaluation_key] = result
        self._evaluation_key_by_chromosome_key[returned_chromosome_key] = evaluation_key
        self.statistics.unique_fitness_evaluations += 1
        _classify_new_result(result, self.statistics)
        if self.ledger is not None:
            self.ledger.append(
                chromosome,
                result,
                generation=generation,
                candidate_index=candidate_index,
                candidate_context=candidate_context,
                runtime_s=runtime,
            )
        if self.progress is not None:
            self.progress(
                GAProgress(
                    generation, candidate_index, candidate_context, result,
                    False, runtime, self.statistics.snapshot(),
                )
            )
        return result


@dataclass(frozen=True)
class ReproducibilityMetadata:
    ga_result_schema_version: int
    ga_config_schema_version: int
    checkpoint_schema_version: int
    evaluation_ledger_schema_version: int
    chromosome_schema_version: int
    design_space_id: str
    fitness_scenario_id: str
    fitness_codec_id: str | None
    python_version: str
    numpy_version: str
    bit_generator: str


@dataclass(frozen=True)
class GAResult:
    """Immutable best-found search result for one random seed."""

    schema_version: int
    config: GAConfig
    ga_config_id: str
    run_seed: int
    termination_reason: TerminationReason
    completed_generation_count: int
    current_generation: int
    best_found: EvaluatedIndividual
    best_feasible_chromosome: NormalizedChromosome | None
    best_feasible_fitness: FitnessLike | None
    final_ranked_population: tuple[EvaluatedIndividual, ...]
    generation_history: tuple[GenerationRecord, ...]
    evaluation_statistics: EvaluationStatistics
    checkpoint_path: Path | None
    evaluation_ledger_path: Path | None
    is_complete: bool
    warnings: tuple[str, ...]
    reproducibility: ReproducibilityMetadata

    @property
    def best_feasible_found(self) -> bool:
        return self.best_feasible_fitness is not None


@dataclass(frozen=True)
class _CheckpointPaths:
    checkpoint: Path
    ledger: Path


@dataclass(frozen=True)
class _RestoredState:
    generation: int
    population: tuple[EvaluatedIndividual, ...]
    best_feasible: EvaluatedIndividual | None
    stagnation: StagnationState
    history: tuple[GenerationRecord, ...]
    statistics: _MutableStatistics
    rng: np.random.Generator
    warnings: tuple[str, ...]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_checkpoint(
    path: Path,
    *,
    config: GAConfig,
    bounds: PlantThermostatDesignSpace,
    fitness_scenario_id: str,
    codec: FitnessCodec,
    generation: int,
    population: Sequence[EvaluatedIndividual],
    best_feasible: EvaluatedIndividual | None,
    rng: np.random.Generator,
    stagnation: StagnationState,
    statistics: EvaluationStatistics,
    history: Sequence[GenerationRecord],
) -> None:
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "ga_config": config.to_dict(),
        "ga_config_id": config.identity,
        "chromosome_schema_version": CHROMOSOME_SCHEMA_VERSION,
        "gene_names": list(GENE_NAMES),
        "design_space_id": bounds.identifier,
        "fitness_scenario_id": fitness_scenario_id,
        "fitness_codec_id": codec.identity,
        "evaluation_ledger_schema_version": EVALUATION_LEDGER_SCHEMA_VERSION,
        "current_generation": generation,
        "population": [
            {
                "genes": list(item.chromosome.genes),
                "chromosome_cache_key": item.chromosome_cache_key,
                "evaluation_key": item.evaluation_key,
            }
            for item in population
        ],
        "best_feasible_evaluation_key": (
            None if best_feasible is None else best_feasible.evaluation_key
        ),
        "rng_state": _json_ready(rng.bit_generator.state),
        "stagnation_state": asdict(stagnation),
        "evaluation_statistics": asdict(statistics),
        "generation_history": [record.to_dict() for record in history],
    }
    payload["checkpoint_digest"] = _digest("ga-checkpoint-v1", payload)
    _atomic_write_json(path, payload)


def _read_checkpoint(
    path: Path,
    *,
    config: GAConfig,
    bounds: PlantThermostatDesignSpace,
    fitness_scenario_id: str,
    codec: FitnessCodec,
) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("generation checkpoint is corrupt or incomplete") from error
    if not isinstance(payload, Mapping):
        raise ValueError("generation checkpoint must contain a JSON object")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported generation checkpoint schema version")
    checks = (
        ("ga_config_id", config.identity),
        ("chromosome_schema_version", CHROMOSOME_SCHEMA_VERSION),
        ("gene_names", list(GENE_NAMES)),
        ("design_space_id", bounds.identifier),
        ("fitness_scenario_id", fitness_scenario_id),
        ("fitness_codec_id", codec.identity),
        ("evaluation_ledger_schema_version", EVALUATION_LEDGER_SCHEMA_VERSION),
    )
    for field, expected in checks:
        if payload.get(field) != expected:
            raise ValueError(f"generation checkpoint {field} mismatch")
    candidate = dict(payload)
    recorded_digest = candidate.pop("checkpoint_digest", None)
    if recorded_digest != _digest("ga-checkpoint-v1", candidate):
        raise ValueError("generation checkpoint digest mismatch")
    return payload


def _statistics_from_dict(record: Mapping[str, Any]) -> EvaluationStatistics:
    try:
        statistics = EvaluationStatistics(**dict(record))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid evaluation statistics in checkpoint") from error
    for name, value in asdict(statistics).items():
        _integer(name, value, minimum=0)
    return statistics


def _restore_state(
    paths: _CheckpointPaths,
    *,
    config: GAConfig,
    bounds: PlantThermostatDesignSpace,
    fitness_scenario_id: str,
    codec: FitnessCodec,
    cache: _EvaluationCache,
) -> _RestoredState:
    payload = _read_checkpoint(
        paths.checkpoint,
        config=config,
        bounds=bounds,
        fitness_scenario_id=fitness_scenario_id,
        codec=codec,
    )
    assert cache.ledger is not None
    entries, ledger_warnings = cache.ledger.load()
    saved = _statistics_from_dict(payload["evaluation_statistics"])
    if len(entries) < saved.unique_fitness_evaluations:
        raise ValueError("evaluation ledger is missing checkpointed results")
    cache.restore(entries)
    statistics = _MutableStatistics.from_snapshot(saved)
    if len(entries) > saved.unique_fitness_evaluations:
        for entry in entries[saved.unique_fitness_evaluations :]:
            statistics.unique_fitness_evaluations += 1
            _classify_new_result(entry.fitness, statistics)
        ledger_warnings = (
            *ledger_warnings,
            "recovered_committed_post_checkpoint_evaluations",
        )
    cache.statistics = statistics

    try:
        generation = _integer(
            "current_generation", payload["current_generation"], minimum=0
        )
        population_records = payload["population"]
        population = []
        for record in population_records:
            chromosome = NormalizedChromosome(tuple(record["genes"]))
            chromosome_key = chromosome.cache_key(bounds=bounds)
            if chromosome_key != record["chromosome_cache_key"]:
                raise ValueError("checkpoint chromosome cache key mismatch")
            fitness = cache.result_for_evaluation_key(record["evaluation_key"])
            if fitness.chromosome_cache_key != chromosome_key:
                raise ValueError("checkpoint population result mismatch")
            population.append(EvaluatedIndividual(chromosome, fitness))
        history = tuple(
            _generation_record_from_dict(item)
            for item in payload["generation_history"]
        )
        stagnation = StagnationState(**payload["stagnation_state"])
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("generation checkpoint has incomplete run state") from error
    if len(population) != config.population_size:
        raise ValueError("checkpoint population size mismatch")
    if len(history) != generation + 1 or history[-1].generation != generation:
        raise ValueError("checkpoint history does not end at its current generation")
    best_key = payload.get("best_feasible_evaluation_key")
    best_feasible = None
    if best_key is not None:
        best_fitness = cache.result_for_evaluation_key(best_key)
        best_chromosome = next(
            (
                entry.chromosome
                for entry in entries
                if entry.fitness.evaluation_key == best_key
            ),
            None,
        )
        if best_chromosome is None:
            raise ValueError("checkpoint best result has no ledger chromosome")
        best_feasible = EvaluatedIndividual(best_chromosome, best_fitness)
        if not (best_fitness.static_feasible and best_fitness.dynamically_feasible):
            raise ValueError("checkpoint best-feasible result is infeasible")
    rng = np.random.default_rng(config.random_seed)
    try:
        rng.bit_generator.state = payload["rng_state"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint RNG state is invalid") from error
    return _RestoredState(
        generation,
        tuple(population),
        best_feasible,
        stagnation,
        history,
        statistics,
        rng,
        ledger_warnings,
    )


def _best_feasible(
    population: Sequence[EvaluatedIndividual],
) -> EvaluatedIndividual | None:
    return next(
        (
            item
            for item in rank_population(population)
            if item.fitness.static_feasible and item.fitness.dynamically_feasible
        ),
        None,
    )


def _better_feasible(
    candidate: EvaluatedIndividual | None,
    incumbent: EvaluatedIndividual | None,
) -> EvaluatedIndividual | None:
    if candidate is None:
        return incumbent
    if incumbent is None or compare_fitness(candidate.fitness, incumbent.fitness) < 0:
        return candidate
    return incumbent


def _generate_offspring(
    ranked_population: Sequence[EvaluatedIndividual],
    *,
    config: GAConfig,
    bounds: PlantThermostatDesignSpace,
    rng: np.random.Generator,
) -> tuple[tuple[NormalizedChromosome, ...], int]:
    required = config.population_size - config.elite_count
    elite_keys = {
        item.chromosome.cache_key(bounds=bounds)
        for item in ranked_population[: config.elite_count]
    }
    offspring: list[NormalizedChromosome] = []
    used = set(elite_keys)
    fallback_duplicates = 0
    while len(offspring) < required:
        accepted: list[NormalizedChromosome] = []
        for _attempt in range(config.duplicate_retry_limit):
            first_parent = tournament_select(
                ranked_population,
                tournament_size=config.tournament_size,
                rng=rng,
            )
            second_parent = tournament_select(
                ranked_population,
                tournament_size=config.tournament_size,
                rng=rng,
            )
            children = bounded_sbx(
                first_parent.chromosome,
                second_parent.chromosome,
                distribution_index=config.sbx_distribution_index,
                crossover_probability=config.crossover_probability,
                rng=rng,
            )
            for child in children:
                mutated = bounded_polynomial_mutation(
                    child,
                    probability_per_gene=config.mutation_probability_per_gene,
                    distribution_index=config.mutation_distribution_index,
                    rng=rng,
                )
                key = mutated.cache_key(bounds=bounds)
                if key not in used and all(
                    key != item.cache_key(bounds=bounds) for item in accepted
                ):
                    accepted.append(mutated)
                if len(offspring) + len(accepted) >= required:
                    break
            if accepted:
                break
        if not accepted:
            candidate = _fresh_uniform(rng)
            for _attempt in range(config.duplicate_retry_limit):
                key = candidate.cache_key(bounds=bounds)
                if key not in used:
                    break
                candidate = _fresh_uniform(rng)
            if candidate.cache_key(bounds=bounds) in used:
                fallback_duplicates += 1
            accepted.append(candidate)
        for candidate in accepted:
            if len(offspring) == required:
                break
            offspring.append(candidate)
            used.add(candidate.cache_key(bounds=bounds))
    return tuple(offspring), fallback_duplicates


def _evaluate_initial_population(
    initial: InitialPopulation,
    *,
    evaluator: Callable[[NormalizedChromosome], FitnessLike],
    cache: _EvaluationCache,
) -> tuple[EvaluatedIndividual, ...]:
    evaluated = []
    for index, (chromosome, origin) in enumerate(
        zip(initial.chromosomes, initial.origins)
    ):
        fitness = cache.evaluate(
            chromosome,
            evaluator,
            generation=0,
            candidate_index=index,
            candidate_context=origin,
        )
        evaluated.append(EvaluatedIndividual(chromosome, fitness))
    return tuple(evaluated)


def _evaluate_offspring(
    offspring: Sequence[NormalizedChromosome],
    *,
    evaluator: Callable[[NormalizedChromosome], FitnessLike],
    cache: _EvaluationCache,
    generation: int,
    externally_interrupted: Callable[[], bool] | None,
) -> tuple[EvaluatedIndividual, ...]:
    evaluated = []
    for index, chromosome in enumerate(offspring):
        fitness = cache.evaluate(
            chromosome,
            evaluator,
            generation=generation,
            candidate_index=index,
            candidate_context="offspring",
        )
        evaluated.append(EvaluatedIndividual(chromosome, fitness))
        if (
            externally_interrupted is not None
            and index < len(offspring) - 1
            and externally_interrupted()
        ):
            raise _ExternalInterruptionRequested(tuple(evaluated))
    return tuple(evaluated)


class _ExternalInterruptionRequested(Exception):
    def __init__(self, evaluated: tuple[EvaluatedIndividual, ...]) -> None:
        super().__init__("external interruption requested between evaluations")
        self.evaluated = evaluated


def fitness_result_codec() -> FitnessCodec:
    """Return the lazy durable codec for the repository's FitnessResult."""
    from src.optimization.fitness import FITNESS_RESULT_SCHEMA_VERSION

    return FitnessCodec(
        identity=f"fitness-result-v{FITNESS_RESULT_SCHEMA_VERSION}",
        serialize=lambda result: result.to_dict(),
        deserialize=_deserialize_fitness_result,
    )


def _deserialize_fitness_result(record: Mapping[str, Any]) -> FitnessLike:
    from src.optimization.chromosome import DecodedPlantThermostatDesign
    from src.optimization.feasibility import (
        ConstraintRecord,
        ResolvedFuel,
        ResolvedMasses,
        ResolvedPlantDesign,
        ResolvedPowerCapability,
        ResolvedWingGeometry,
    )
    from src.optimization.fitness import (
        ControllerBehavior,
        DurationSummary,
        DynamicConstraintRecord,
        FitnessDiagnostic,
        FitnessResult,
        MissionResources,
        MissionValidity,
        PowerRange,
    )

    try:
        values = dict(record)
        values["decoded_design"] = DecodedPlantThermostatDesign(
            **values["decoded_design"]
        )
        resolved = values["resolved_design"]
        values["resolved_design"] = ResolvedPlantDesign(
            decoded=DecodedPlantThermostatDesign(**resolved["decoded"]),
            scenario_id=resolved["scenario_id"],
            wing=ResolvedWingGeometry(**resolved["wing"]),
            masses=ResolvedMasses(**resolved["masses"]),
            fuel=ResolvedFuel(**resolved["fuel"]),
            power=ResolvedPowerCapability(**resolved["power"]),
            soc_low=resolved["soc_low"],
            soc_high=resolved["soc_high"],
        )
        values["static_constraints"] = tuple(
            ConstraintRecord(**item) for item in values["static_constraints"]
        )
        if values["resources"] is not None:
            values["resources"] = MissionResources(**values["resources"])
        if values["controller_behavior"] is not None:
            behavior = values["controller_behavior"]
            values["controller_behavior"] = ControllerBehavior(
                restart_count=behavior["restart_count"],
                restarts_per_loiter_hour=behavior["restarts_per_loiter_hour"],
                overall_engine_off_fraction=behavior["overall_engine_off_fraction"],
                loiter_engine_off_fraction=behavior["loiter_engine_off_fraction"],
                on_run_durations=DurationSummary(**behavior["on_run_durations"]),
                off_run_durations=DurationSummary(**behavior["off_run_durations"]),
                requested_engine_power_range=PowerRange(
                    **behavior["requested_engine_power_range"]
                ),
                delivered_engine_power_range=PowerRange(
                    **behavior["delivered_engine_power_range"]
                ),
                charge_limit_encounter_count=behavior[
                    "charge_limit_encounter_count"
                ],
                discharge_limit_encounter_count=behavior[
                    "discharge_limit_encounter_count"
                ],
            )
        if values["validity"] is not None:
            validity = dict(values["validity"])
            validity["failure_flags"] = tuple(validity["failure_flags"])
            values["validity"] = MissionValidity(**validity)
        values["dynamic_constraint_names"] = tuple(
            values["dynamic_constraint_names"]
        )
        values["dynamic_constraints"] = tuple(
            DynamicConstraintRecord(**item)
            for item in values["dynamic_constraints"]
        )
        values["warnings"] = tuple(
            FitnessDiagnostic(**item) for item in values["warnings"]
        )
        return FitnessResult(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("serialized FitnessResult is corrupt or incomplete") from error


def _checkpoint_paths(directory: str | Path) -> _CheckpointPaths:
    path = Path(directory)
    return _CheckpointPaths(
        checkpoint=path / "ga_checkpoint.json",
        ledger=path / "evaluation_ledger.jsonl",
    )


def _result(
    *,
    config: GAConfig,
    bounds: PlantThermostatDesignSpace,
    fitness_scenario_id: str,
    codec: FitnessCodec | None,
    termination_reason: TerminationReason,
    population: Sequence[EvaluatedIndividual],
    best_feasible: EvaluatedIndividual | None,
    history: Sequence[GenerationRecord],
    statistics: EvaluationStatistics,
    paths: _CheckpointPaths | None,
    warnings: Sequence[str],
) -> GAResult:
    ranked = rank_population(population)
    complete = termination_reason in {"max_generations", "stagnation"}
    return GAResult(
        schema_version=_GA_RESULT_SCHEMA_VERSION,
        config=config,
        ga_config_id=config.identity,
        run_seed=config.random_seed,
        termination_reason=termination_reason,
        completed_generation_count=len(history),
        current_generation=history[-1].generation,
        best_found=ranked[0],
        best_feasible_chromosome=(
            None if best_feasible is None else best_feasible.chromosome
        ),
        best_feasible_fitness=(
            None if best_feasible is None else best_feasible.fitness
        ),
        final_ranked_population=ranked,
        generation_history=tuple(history),
        evaluation_statistics=statistics,
        checkpoint_path=None if paths is None else paths.checkpoint,
        evaluation_ledger_path=None if paths is None else paths.ledger,
        is_complete=complete,
        warnings=tuple(dict.fromkeys(warnings)),
        reproducibility=ReproducibilityMetadata(
            ga_result_schema_version=_GA_RESULT_SCHEMA_VERSION,
            ga_config_schema_version=GA_CONFIG_SCHEMA_VERSION,
            checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
            evaluation_ledger_schema_version=EVALUATION_LEDGER_SCHEMA_VERSION,
            chromosome_schema_version=CHROMOSOME_SCHEMA_VERSION,
            design_space_id=bounds.identifier,
            fitness_scenario_id=fitness_scenario_id,
            fitness_codec_id=None if codec is None else codec.identity,
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            bit_generator="PCG64",
        ),
    )


def run_ga(
    evaluator: Callable[[NormalizedChromosome], FitnessLike],
    *,
    bounds: PlantThermostatDesignSpace,
    fitness_scenario_id: str,
    config: GAConfig | None = None,
    checkpoint_directory: str | Path | None = None,
    resume: bool = False,
    stop_after_generation: int | None = None,
    externally_interrupted: Callable[[], bool] | None = None,
    fitness_codec: FitnessCodec | None = None,
    progress: Callable[[GAProgress], None] | None = None,
) -> GAResult:
    """Run one seeded GA without knowing or importing the evaluator's physics."""
    settings = config or GAConfig()
    if not isinstance(settings, GAConfig):
        raise ValueError("config must be a GAConfig")
    if not isinstance(bounds, PlantThermostatDesignSpace):
        raise ValueError("bounds must be a PlantThermostatDesignSpace")
    if not isinstance(fitness_scenario_id, str) or not fitness_scenario_id:
        raise ValueError("fitness_scenario_id must be a non-empty string")
    if not callable(evaluator):
        raise ValueError("evaluator must be callable")
    if externally_interrupted is not None and not callable(externally_interrupted):
        raise ValueError("externally_interrupted must be callable")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    if stop_after_generation is not None:
        stop_after_generation = _integer(
            "stop_after_generation", stop_after_generation, minimum=0
        )
        if stop_after_generation >= settings.max_generations:
            raise ValueError("stop_after_generation lies outside the configured run")
        if checkpoint_directory is None:
            raise ValueError("controlled stopping requires checkpoint_directory")
    if resume and checkpoint_directory is None:
        raise ValueError("resume requires checkpoint_directory")

    paths = (
        None
        if checkpoint_directory is None
        else _checkpoint_paths(checkpoint_directory)
    )
    if paths is not None and fitness_codec is None:
        raise ValueError("checkpointing requires an explicit fitness codec")
    if fitness_codec is not None and not isinstance(fitness_codec, FitnessCodec):
        raise ValueError("fitness_codec must be a FitnessCodec")
    if paths is not None and not resume:
        existing = [path for path in (paths.checkpoint, paths.ledger) if path.exists()]
        if existing:
            raise FileExistsError("checkpoint files already exist; use resume=True")
    if paths is not None and resume:
        if not paths.checkpoint.exists() or not paths.ledger.exists():
            raise FileNotFoundError("resume requires both checkpoint and evaluation ledger")

    ledger = None
    if paths is not None:
        assert fitness_codec is not None
        ledger = _EvaluationLedger(
            paths.ledger,
            config=settings,
            bounds=bounds,
            fitness_scenario_id=fitness_scenario_id,
            codec=fitness_codec,
        )
    cache = _EvaluationCache(
        bounds=bounds,
        fitness_scenario_id=fitness_scenario_id,
        ledger=ledger,
        progress=progress,
    )
    warnings: list[str] = []
    run_started = time.perf_counter()

    if resume:
        assert paths is not None and fitness_codec is not None
        restored = _restore_state(
            paths,
            config=settings,
            bounds=bounds,
            fitness_scenario_id=fitness_scenario_id,
            codec=fitness_codec,
            cache=cache,
        )
        generation = restored.generation
        population = restored.population
        best_feasible = restored.best_feasible
        stagnation = restored.stagnation
        history = list(restored.history)
        rng = restored.rng
        warnings.extend(restored.warnings)
        elapsed_offset = history[-1].elapsed_wall_time_s
    else:
        rng = np.random.default_rng(settings.random_seed)
        initial = initialize_population(settings, bounds=bounds, rng=rng)
        population = _evaluate_initial_population(
            initial,
            evaluator=evaluator,
            cache=cache,
        )
        generation = 0
        best_feasible = _best_feasible(population)
        best_objective = (
            None
            if best_feasible is None
            else float(best_feasible.fitness.objective_loiter_seconds)
        )
        stagnation, improved = update_stagnation(
            StagnationState(),
            best_objective,
            material_relative_improvement=settings.material_relative_improvement,
        )
        history = [
            _make_generation_record(
                population,
                generation=0,
                config=settings,
                statistics=cache.statistics.snapshot(),
                elapsed_wall_time_s=time.perf_counter() - run_started,
                material_improvement=improved,
                stagnation_state=stagnation,
            )
        ]
        elapsed_offset = 0.0
        if paths is not None:
            assert fitness_codec is not None
            _write_checkpoint(
                paths.checkpoint,
                config=settings,
                bounds=bounds,
                fitness_scenario_id=fitness_scenario_id,
                codec=fitness_codec,
                generation=0,
                population=population,
                best_feasible=best_feasible,
                rng=rng,
                stagnation=stagnation,
                statistics=cache.statistics.snapshot(),
                history=history,
            )

    if stop_after_generation == generation:
        termination: TerminationReason = "completed_requested_generation"
    elif externally_interrupted is not None and externally_interrupted():
        termination = "externally_interrupted"
    elif stagnation.stagnant_generation_count >= settings.early_stop_patience:
        termination = "stagnation"
    elif generation >= settings.max_generations - 1:
        termination = "max_generations"
    else:
        termination = "max_generations"
        for next_generation in range(generation + 1, settings.max_generations):
            ranked = rank_population(population)
            elites = ranked[: settings.elite_count]
            offspring, duplicate_fallbacks = _generate_offspring(
                ranked,
                config=settings,
                bounds=bounds,
                rng=rng,
            )
            if duplicate_fallbacks:
                warnings.append("offspring_uniqueness_fallback_allowed_duplicates")
            try:
                evaluated_offspring = _evaluate_offspring(
                    offspring,
                    evaluator=evaluator,
                    cache=cache,
                    generation=next_generation,
                    externally_interrupted=externally_interrupted,
                )
            except _ExternalInterruptionRequested as interruption:
                best_feasible = _better_feasible(
                    _best_feasible(interruption.evaluated), best_feasible
                )
                warnings.append(
                    "interrupted_mid_generation; latest completed-generation "
                    "checkpoint retained and committed evaluations will be reused"
                )
                termination = "externally_interrupted"
                break
            population = (*elites, *evaluated_offspring)
            generation = next_generation
            best_feasible = _better_feasible(
                _best_feasible(population), best_feasible
            )
            best_objective = (
                None
                if best_feasible is None
                else float(best_feasible.fitness.objective_loiter_seconds)
            )
            stagnation, improved = update_stagnation(
                stagnation,
                best_objective,
                material_relative_improvement=(
                    settings.material_relative_improvement
                ),
            )
            history.append(
                _make_generation_record(
                    population,
                    generation=generation,
                    config=settings,
                    statistics=cache.statistics.snapshot(),
                    elapsed_wall_time_s=(
                        elapsed_offset + time.perf_counter() - run_started
                    ),
                    material_improvement=improved,
                    stagnation_state=stagnation,
                )
            )
            if paths is not None:
                assert fitness_codec is not None
                _write_checkpoint(
                    paths.checkpoint,
                    config=settings,
                    bounds=bounds,
                    fitness_scenario_id=fitness_scenario_id,
                    codec=fitness_codec,
                    generation=generation,
                    population=population,
                    best_feasible=best_feasible,
                    rng=rng,
                    stagnation=stagnation,
                    statistics=cache.statistics.snapshot(),
                    history=history,
                )
            if stop_after_generation == generation:
                termination = "completed_requested_generation"
                break
            if externally_interrupted is not None and externally_interrupted():
                termination = "externally_interrupted"
                break
            if stagnation.stagnant_generation_count >= settings.early_stop_patience:
                termination = "stagnation"
                break
            if generation == settings.max_generations - 1:
                termination = "max_generations"
                break

    return _result(
        config=settings,
        bounds=bounds,
        fitness_scenario_id=fitness_scenario_id,
        codec=fitness_codec,
        termination_reason=termination,
        population=population,
        best_feasible=best_feasible,
        history=history,
        statistics=cache.statistics.snapshot(),
        paths=paths,
        warnings=warnings,
    )
