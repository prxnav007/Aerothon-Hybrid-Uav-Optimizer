"""Deterministic GA configuration, operators, caching, and resume tests."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, replace

import numpy as np
import pytest

from src.models.battery import BatteryPack
from src.optimization.chromosome import (
    NormalizedChromosome,
    PlantThermostatDesignSpace,
    ideal_restart_fuel_seed,
    practical_thermostat_seed,
)
from src.optimization.ga import (
    CHECKPOINT_SCHEMA_VERSION,
    EVALUATION_LEDGER_SCHEMA_VERSION,
    GAConfig,
    EvaluatedIndividual,
    FitnessCodec,
    StagnationState,
    bounded_polynomial_mutation,
    bounded_sbx,
    compare_fitness,
    fitness_rank_key,
    fitness_result_codec,
    initialize_population,
    maximum_candidate_placements,
    rank_population,
    run_ga,
    tournament_select,
    theoretical_evaluation_budget,
    update_stagnation,
)

SCENARIO_ID = "analytical-scenario-v1"


@pytest.fixture(scope="module")
def bounds() -> PlantThermostatDesignSpace:
    return PlantThermostatDesignSpace.from_battery(BatteryPack(10.0))


@dataclass(frozen=True)
class AnalyticalFitness:
    schema_version: int
    chromosome_cache_key: str
    fitness_scenario_id: str
    evaluation_key: str
    static_feasible: bool
    dynamically_feasible: bool
    objective_loiter_seconds: float | None
    combined_static_dynamic_violation: float
    run_mission_called: bool = False
    restart_count: int = 0
    fuel_slack_kg: float = 0.0
    battery_slack_kwh: float = 0.0

    def to_dict(self):
        return asdict(self)


ANALYTICAL_CODEC = FitnessCodec(
    "analytical-fitness-v1",
    lambda result: result.to_dict(),
    lambda record: AnalyticalFitness(**record),
)


def _fitness(
    chromosome: NormalizedChromosome,
    bounds: PlantThermostatDesignSpace,
    *,
    objective: float | None,
    violation: float,
    static: bool = True,
    dynamic: bool = True,
    scenario_id: str = SCENARIO_ID,
    evaluation_suffix: str = "",
    restart_count: int = 0,
    fuel_slack_kg: float = 0.0,
) -> AnalyticalFitness:
    chromosome_key = chromosome.cache_key(bounds=bounds)
    return AnalyticalFitness(
        schema_version=1,
        chromosome_cache_key=chromosome_key,
        fitness_scenario_id=scenario_id,
        evaluation_key=f"analytical:{scenario_id}:{chromosome_key}:{evaluation_suffix}",
        static_feasible=static,
        dynamically_feasible=dynamic,
        objective_loiter_seconds=objective,
        combined_static_dynamic_violation=violation,
        restart_count=restart_count,
        fuel_slack_kg=fuel_slack_kg,
    )


class AnalyticalEvaluator:
    def __init__(
        self,
        bounds: PlantThermostatDesignSpace,
        *,
        scenario_id: str = SCENARIO_ID,
        target: tuple[float, ...] = (0.91, 0.83, 0.76, 0.69, 0.62, 0.55),
        minimum_first_pair_sum: float | None = None,
        all_infeasible: bool = False,
        constant_objective: float | None = None,
    ) -> None:
        self.bounds = bounds
        self.scenario_id = scenario_id
        self.target = target
        self.minimum_first_pair_sum = minimum_first_pair_sum
        self.all_infeasible = all_infeasible
        self.constant_objective = constant_objective
        self.calls = 0
        self.evaluation_keys: list[str] = []

    def __call__(self, chromosome: NormalizedChromosome) -> AnalyticalFitness:
        self.calls += 1
        distance = sum(
            (gene - target) ** 2
            for gene, target in zip(chromosome.genes, self.target)
        )
        objective = (
            self.constant_objective
            if self.constant_objective is not None
            else 1000.0 * (1.0 - distance)
        )
        violation = 1.0 if self.all_infeasible else 0.0
        if self.minimum_first_pair_sum is not None:
            violation = max(
                self.minimum_first_pair_sum
                - chromosome.genes[0]
                - chromosome.genes[1],
                0.0,
            )
        feasible = violation == 0.0 and not self.all_infeasible
        result = _fitness(
            chromosome,
            self.bounds,
            objective=objective if feasible else None,
            violation=violation,
            static=feasible,
            dynamic=feasible,
            scenario_id=self.scenario_id,
        )
        self.evaluation_keys.append(result.evaluation_key)
        return result


def _history_without_time(result):
    return tuple(
        {key: value for key, value in record.to_dict().items() if key != "elapsed_wall_time_s"}
        for record in result.generation_history
    )


def test_default_configuration_is_the_declared_untuned_production_contract() -> None:
    config = GAConfig()
    assert config.population_size == 64
    assert config.max_generations == 40
    assert config.elite_count == 2
    assert config.tournament_size == 3
    assert config.crossover_probability == 0.90
    assert config.sbx_distribution_index == 15.0
    assert config.mutation_probability_per_gene == pytest.approx(1.0 / 6.0)
    assert config.mutation_distribution_index == 20.0
    assert config.early_stop_patience == 10
    assert config.material_relative_improvement == 1.0e-4
    assert config.initial_perturbation_scale == 0.05


@pytest.mark.parametrize(
    "changes",
    (
        {"population_size": 3},
        {"elite_count": 0},
        {"elite_count": 63},
        {"tournament_size": 1},
        {"tournament_size": 65},
        {"crossover_probability": -0.1},
        {"crossover_probability": 1.1},
        {"mutation_probability_per_gene": math.nan},
        {"sbx_distribution_index": 0.0},
        {"mutation_distribution_index": -1.0},
        {"max_generations": 0},
        {"early_stop_patience": 0},
        {"material_relative_improvement": 0.0},
        {"random_seed": -1},
    ),
)
def test_invalid_configuration_is_rejected_without_repair(changes) -> None:
    with pytest.raises(ValueError):
        GAConfig(**changes)


def test_generation_zero_plus_thirty_nine_offspring_populations_costs_2482() -> None:
    assert maximum_candidate_placements(GAConfig()) == 2482
    assert maximum_candidate_placements(GAConfig(), independent_seeds=3) == 7446
    budget = theoretical_evaluation_budget(GAConfig(), independent_seeds=3)
    assert budget.later_population_count == 39
    assert budget.offspring_per_later_population == 62
    assert budget.candidate_placements_per_seed == 2482
    assert budget.unique_evaluations_per_seed_upper_bound == 2482
    assert budget.total_candidate_placements == 7446
    assert budget.total_unique_evaluations_upper_bound == 7446


def test_default_initial_population_has_exact_lhs_and_seeded_composition(bounds) -> None:
    initial = initialize_population(
        GAConfig(), bounds=bounds, rng=np.random.default_rng(GAConfig().random_seed)
    )
    assert len(initial.chromosomes) == 64
    assert initial.latin_hypercube_count == 48
    assert initial.seeded_count == 16
    assert initial.origins.count("practical_perturbation") == 7
    assert initial.origins.count("ideal_perturbation") == 7
    assert practical_thermostat_seed(bounds=bounds) in initial.chromosomes
    assert ideal_restart_fuel_seed(bounds=bounds) in initial.chromosomes
    assert len({item.cache_key(bounds=bounds) for item in initial.chromosomes}) == 64
    assert all(0.0 <= gene <= 1.0 for item in initial.chromosomes for gene in item.genes)

    lhs = [
        item
        for item, origin in zip(initial.chromosomes, initial.origins)
        if origin == "latin_hypercube"
    ]
    for gene_index in range(6):
        strata = sorted(int(item.genes[gene_index] * len(lhs)) for item in lhs)
        assert strata == list(range(len(lhs)))


def test_initialization_is_seed_deterministic_but_nonanchors_change(bounds) -> None:
    config = GAConfig(random_seed=77)
    first = initialize_population(config, bounds=bounds, rng=np.random.default_rng(77))
    second = initialize_population(config, bounds=bounds, rng=np.random.default_rng(77))
    other = initialize_population(
        replace(config, random_seed=78), bounds=bounds, rng=np.random.default_rng(78)
    )
    assert first == second
    assert first.chromosomes[:2] == other.chromosomes[:2]
    assert first.chromosomes[2:] != other.chromosomes[2:]


def test_deb_ordering_uses_only_feasibility_objective_violation_and_key(bounds) -> None:
    a = NormalizedChromosome((0.1,) * 6)
    b = NormalizedChromosome((0.2,) * 6)
    feasible_low = _fitness(a, bounds, objective=10.0, violation=0.0)
    feasible_high = _fitness(b, bounds, objective=20.0, violation=0.0)
    infeasible_low = _fitness(
        a, bounds, objective=10000.0, violation=0.1, static=False, dynamic=False,
        evaluation_suffix="infeasible-low",
    )
    infeasible_high = _fitness(
        b, bounds, objective=20000.0, violation=0.2, static=False, dynamic=False,
        evaluation_suffix="infeasible-high",
    )
    assert compare_fitness(feasible_low, infeasible_low) < 0
    assert compare_fitness(feasible_high, feasible_low) < 0
    assert compare_fitness(infeasible_low, infeasible_high) < 0

    first = replace(feasible_low, evaluation_key="a", restart_count=999, fuel_slack_kg=-99.0)
    second = replace(feasible_low, evaluation_key="b", restart_count=0, fuel_slack_kg=999.0)
    assert fitness_rank_key(first) < fitness_rank_key(second)


def test_invalid_feasible_objective_or_violation_and_nonfinite_violation_are_rejected(bounds) -> None:
    chromosome = NormalizedChromosome((0.5,) * 6)
    missing = _fitness(chromosome, bounds, objective=None, violation=0.0)
    nonfinite = _fitness(chromosome, bounds, objective=1.0, violation=math.inf)
    inconsistent = _fitness(chromosome, bounds, objective=1.0, violation=1.0e-6)
    for result in (missing, nonfinite, inconsistent):
        with pytest.raises(ValueError):
            fitness_rank_key(result)


def test_tournament_selection_uses_deb_ordering_and_fixed_rng_state(bounds) -> None:
    chromosomes = tuple(NormalizedChromosome((value,) * 6) for value in (0.1, 0.2, 0.3))
    population = tuple(
        EvaluatedIndividual(
            chromosome,
            _fitness(chromosome, bounds, objective=float(index), violation=0.0),
        )
        for index, chromosome in enumerate(chromosomes)
    )
    first = tournament_select(
        population, tournament_size=100, rng=np.random.default_rng(12)
    )
    second = tournament_select(
        population, tournament_size=100, rng=np.random.default_rng(12)
    )
    assert first == second
    assert first == rank_population(population)[0]


def test_bounded_sbx_is_symmetric_finite_and_deterministic_under_stress() -> None:
    generator = np.random.default_rng(405)
    pairs = []
    for _ in range(500):
        parent_a = NormalizedChromosome(tuple(generator.random(6)))
        parent_b = NormalizedChromosome(tuple(generator.random(6)))
        children = bounded_sbx(
            parent_a, parent_b, distribution_index=15.0,
            crossover_probability=1.0, rng=generator,
        )
        assert all(0.0 <= gene <= 1.0 for child in children for gene in child.genes)
        pairs.append(children)
    replay = np.random.default_rng(405)
    replay_pairs = []
    for _ in range(500):
        parent_a = NormalizedChromosome(tuple(replay.random(6)))
        parent_b = NormalizedChromosome(tuple(replay.random(6)))
        replay_pairs.append(
            bounded_sbx(
                parent_a, parent_b, distribution_index=15.0,
                crossover_probability=1.0, rng=replay,
            )
        )
    assert tuple(pairs) == tuple(replay_pairs)


def test_identical_parent_genes_and_skipped_crossover_are_copied_unchanged() -> None:
    parent = NormalizedChromosome((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    identical = bounded_sbx(
        parent, parent, distribution_index=15.0,
        crossover_probability=1.0, rng=np.random.default_rng(9),
    )
    skipped = bounded_sbx(
        parent, NormalizedChromosome((1.0,) * 6), distribution_index=15.0,
        crossover_probability=0.0, rng=np.random.default_rng(9),
    )
    assert identical == (parent, parent)
    assert skipped[0] == parent


def test_polynomial_mutation_is_bounded_moves_boundaries_and_is_deterministic() -> None:
    lower = NormalizedChromosome((0.0,) * 6)
    upper = NormalizedChromosome((1.0,) * 6)
    mutated_lower = bounded_polynomial_mutation(
        lower, probability_per_gene=1.0, distribution_index=20.0,
        rng=np.random.default_rng(41),
    )
    mutated_upper = bounded_polynomial_mutation(
        upper, probability_per_gene=1.0, distribution_index=20.0,
        rng=np.random.default_rng(42),
    )
    replay = bounded_polynomial_mutation(
        lower, probability_per_gene=1.0, distribution_index=20.0,
        rng=np.random.default_rng(41),
    )
    assert mutated_lower == replay
    assert any(0.0 < gene < 1.0 for gene in mutated_lower.genes)
    assert any(0.0 < gene < 1.0 for gene in mutated_upper.genes)
    assert all(0.0 <= gene <= 1.0 for gene in (*mutated_lower.genes, *mutated_upper.genes))


def test_two_elites_survive_and_exactly_population_minus_two_offspring_are_placed(bounds) -> None:
    config = GAConfig(
        population_size=16, max_generations=2, elite_count=2,
        early_stop_patience=10, random_seed=81,
    )
    initial = initialize_population(config, bounds=bounds, rng=np.random.default_rng(81))
    evaluator_for_ranking = AnalyticalEvaluator(bounds)
    initial_ranked = rank_population(
        tuple(
            EvaluatedIndividual(item, evaluator_for_ranking(item))
            for item in initial.chromosomes
        )
    )
    evaluator = AnalyticalEvaluator(bounds)
    result = run_ga(evaluator, bounds=bounds, fitness_scenario_id=SCENARIO_ID, config=config)
    final_keys = {item.chromosome_cache_key for item in result.final_ranked_population}
    assert all(item.chromosome_cache_key in final_keys for item in initial_ranked[:2])
    assert len(result.final_ranked_population) == 16
    assert result.evaluation_statistics.candidate_placements == 16 + 14


def test_duplicate_candidates_use_the_exact_cache_without_reinvoking_evaluator(bounds) -> None:
    config = GAConfig(
        population_size=12, max_generations=3, elite_count=2,
        crossover_probability=0.0, mutation_probability_per_gene=0.0,
        duplicate_retry_limit=1, early_stop_patience=10, random_seed=33,
    )
    evaluator = AnalyticalEvaluator(bounds)
    result = run_ga(evaluator, bounds=bounds, fitness_scenario_id=SCENARIO_ID, config=config)
    statistics = result.evaluation_statistics
    assert statistics.candidate_placements == 32
    assert statistics.cache_hits > 0
    assert evaluator.calls == statistics.unique_fitness_evaluations
    assert evaluator.calls + statistics.cache_hits == statistics.candidate_placements


@pytest.mark.parametrize("category", ("static", "dynamic"))
def test_infeasible_duplicate_results_are_cached_and_static_failures_call_no_mission(
    bounds, category
) -> None:
    calls = 0

    def evaluator(chromosome):
        nonlocal calls
        calls += 1
        if category == "static":
            return _fitness(
                chromosome, bounds, objective=None, violation=0.5,
                static=False, dynamic=False,
            )
        return replace(
            _fitness(
                chromosome, bounds, objective=None, violation=0.25,
                static=True, dynamic=False,
            ),
            run_mission_called=True,
        )

    config = GAConfig(
        population_size=12, max_generations=3, elite_count=2,
        crossover_probability=0.0, mutation_probability_per_gene=0.0,
        duplicate_retry_limit=1, early_stop_patience=10, random_seed=34,
    )
    result = run_ga(
        evaluator, bounds=bounds, fitness_scenario_id=SCENARIO_ID, config=config
    )
    statistics = result.evaluation_statistics
    assert statistics.cache_hits > 0
    assert calls == statistics.unique_fitness_evaluations
    if category == "static":
        assert statistics.static_infeasible_results == calls
        assert statistics.mission_calls == 0
    else:
        assert statistics.dynamic_infeasible_results == calls
        assert statistics.mission_calls == calls


def test_feasible_static_dynamic_and_mission_accounting_are_separate(bounds) -> None:
    calls = 0

    def evaluator(chromosome):
        nonlocal calls
        calls += 1
        index = calls % 3
        if index == 0:
            return replace(
                _fitness(chromosome, bounds, objective=1.0, violation=0.0),
                run_mission_called=True,
            )
        if index == 1:
            return _fitness(
                chromosome, bounds, objective=None, violation=0.5,
                static=False, dynamic=False,
            )
        return replace(
            _fitness(
                chromosome, bounds, objective=None, violation=0.25,
                static=True, dynamic=False,
            ),
            run_mission_called=True,
        )

    config = GAConfig(population_size=6, max_generations=1, elite_count=1, random_seed=9)
    result = run_ga(evaluator, bounds=bounds, fitness_scenario_id=SCENARIO_ID, config=config)
    stats = result.evaluation_statistics
    assert stats.candidate_placements == stats.unique_fitness_evaluations == 6
    assert stats.static_infeasible_results == 2
    assert stats.dynamic_infeasible_results == 2
    assert stats.feasible_results == 2
    assert stats.mission_calls == 4


def test_scenario_identity_changes_evaluation_keys_and_run_metadata(bounds) -> None:
    config = GAConfig(population_size=6, max_generations=1, elite_count=1, random_seed=10)
    first_evaluator = AnalyticalEvaluator(bounds, scenario_id="scenario-a")
    second_evaluator = AnalyticalEvaluator(bounds, scenario_id="scenario-b")
    first = run_ga(
        first_evaluator, bounds=bounds, fitness_scenario_id="scenario-a", config=config
    )
    second = run_ga(
        second_evaluator, bounds=bounds, fitness_scenario_id="scenario-b", config=config
    )
    assert first.best_found.evaluation_key != second.best_found.evaluation_key
    assert first.reproducibility.fitness_scenario_id != second.reproducibility.fitness_scenario_id


def test_material_and_subthreshold_improvements_have_distinct_patience_semantics() -> None:
    initial, improved = update_stagnation(
        StagnationState(), 100.0, material_relative_improvement=1.0e-4
    )
    small, small_flag = update_stagnation(
        initial, 100.005, material_relative_improvement=1.0e-4
    )
    material, material_flag = update_stagnation(
        small, 100.02, material_relative_improvement=1.0e-4
    )
    assert improved
    assert small.exact_best_objective == 100.005
    assert small.material_best_objective == 100.0
    assert small.stagnant_generation_count == 1
    assert not small_flag
    assert material.material_best_objective == 100.02
    assert material.stagnant_generation_count == 0
    assert material_flag


def test_no_feasible_candidate_never_triggers_stagnation(bounds) -> None:
    config = GAConfig(
        population_size=8, max_generations=4, elite_count=1,
        early_stop_patience=1, random_seed=17,
    )
    result = run_ga(
        AnalyticalEvaluator(bounds, all_infeasible=True),
        bounds=bounds,
        fitness_scenario_id=SCENARIO_ID,
        config=config,
    )
    assert result.termination_reason == "max_generations"
    assert result.completed_generation_count == 4
    assert not result.best_feasible_found


def test_ten_completed_stagnant_generations_stop_with_stagnation_reason(bounds) -> None:
    config = GAConfig(
        population_size=8, max_generations=20, elite_count=1,
        early_stop_patience=10, random_seed=18,
    )
    result = run_ga(
        AnalyticalEvaluator(bounds, constant_objective=100.0),
        bounds=bounds,
        fitness_scenario_id=SCENARIO_ID,
        config=config,
    )
    assert result.termination_reason == "stagnation"
    assert result.current_generation == 10
    assert result.completed_generation_count == 11
    assert result.generation_history[-1].stagnation_counter == 10


def test_generation_zero_record_keeps_exact_counts_objectives_and_diversity(bounds) -> None:
    config = GAConfig(population_size=10, max_generations=1, elite_count=2, random_seed=19)
    result = run_ga(
        AnalyticalEvaluator(bounds, minimum_first_pair_sum=1.1),
        bounds=bounds,
        fitness_scenario_id=SCENARIO_ID,
        config=config,
    )
    record = result.generation_history[0]
    assert record.feasible_count + record.static_infeasible_count == 10
    assert record.dynamic_infeasible_count == 0
    assert record.feasible_fraction == record.feasible_count / 10
    assert record.best_feasible_objective >= record.median_feasible_objective
    assert record.median_feasible_objective >= record.worst_feasible_objective
    assert record.best_normalized_infeasible_violation is not None
    assert record.mean_normalized_gene_diversity > 0.0
    assert record.cumulative_candidate_placements == 10


def test_checkpoint_is_versioned_atomic_and_controlled_stop_is_structured(bounds, tmp_path) -> None:
    config = GAConfig(
        population_size=10, max_generations=5, elite_count=2,
        early_stop_patience=20, random_seed=91,
    )
    result = run_ga(
        AnalyticalEvaluator(bounds),
        bounds=bounds,
        fitness_scenario_id=SCENARIO_ID,
        config=config,
        checkpoint_directory=tmp_path,
        stop_after_generation=2,
        fitness_codec=ANALYTICAL_CODEC,
    )
    checkpoint = json.loads(result.checkpoint_path.read_text(encoding="utf-8"))
    first_ledger = json.loads(
        result.evaluation_ledger_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert result.termination_reason == "completed_requested_generation"
    assert not result.is_complete
    assert result.current_generation == 2
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert first_ledger["schema_version"] == EVALUATION_LEDGER_SCHEMA_VERSION
    assert not list(tmp_path.glob("*.tmp"))


def test_stopped_and_resumed_run_matches_uninterrupted_run_exactly(bounds, tmp_path) -> None:
    config = GAConfig(
        population_size=14, max_generations=7, elite_count=2,
        early_stop_patience=20, random_seed=1234,
    )
    uninterrupted_evaluator = AnalyticalEvaluator(bounds)
    uninterrupted = run_ga(
        uninterrupted_evaluator,
        bounds=bounds,
        fitness_scenario_id=SCENARIO_ID,
        config=config,
    )
    resumed_directory = tmp_path / "resumed"
    partial_evaluator = AnalyticalEvaluator(bounds)
    partial = run_ga(
        partial_evaluator,
        bounds=bounds,
        fitness_scenario_id=SCENARIO_ID,
        config=config,
        checkpoint_directory=resumed_directory,
        stop_after_generation=3,
        fitness_codec=ANALYTICAL_CODEC,
    )
    continuation_evaluator = AnalyticalEvaluator(bounds)
    resumed = run_ga(
        continuation_evaluator,
        bounds=bounds,
        fitness_scenario_id=SCENARIO_ID,
        config=config,
        checkpoint_directory=resumed_directory,
        resume=True,
        fitness_codec=ANALYTICAL_CODEC,
    )
    assert partial.termination_reason == "completed_requested_generation"
    assert resumed.termination_reason == uninterrupted.termination_reason
    assert tuple(item.chromosome for item in resumed.final_ranked_population) == tuple(
        item.chromosome for item in uninterrupted.final_ranked_population
    )
    assert tuple(item.evaluation_key for item in resumed.final_ranked_population) == tuple(
        item.evaluation_key for item in uninterrupted.final_ranked_population
    )
    assert resumed.best_feasible_chromosome == uninterrupted.best_feasible_chromosome
    assert (
        resumed.best_feasible_fitness.objective_loiter_seconds
        == uninterrupted.best_feasible_fitness.objective_loiter_seconds
    )
    assert _history_without_time(resumed) == _history_without_time(uninterrupted)
    assert resumed.evaluation_statistics == uninterrupted.evaluation_statistics
    assert partial_evaluator.calls + continuation_evaluator.calls == uninterrupted_evaluator.calls


def test_resume_safely_ignores_an_interrupted_final_ledger_tail(bounds, tmp_path) -> None:
    config = GAConfig(
        population_size=8, max_generations=4, elite_count=1,
        early_stop_patience=20, random_seed=42,
    )
    partial = run_ga(
        AnalyticalEvaluator(bounds), bounds=bounds,
        fitness_scenario_id=SCENARIO_ID, config=config,
        checkpoint_directory=tmp_path, stop_after_generation=1,
        fitness_codec=ANALYTICAL_CODEC,
    )
    with partial.evaluation_ledger_path.open("a", encoding="utf-8") as stream:
        stream.write('{"interrupted"')
    resumed = run_ga(
        AnalyticalEvaluator(bounds), bounds=bounds,
        fitness_scenario_id=SCENARIO_ID, config=config,
        checkpoint_directory=tmp_path, resume=True,
        fitness_codec=ANALYTICAL_CODEC,
    )
    assert "ignored_interrupted_final_evaluation_ledger_line" in resumed.warnings
    for line in resumed.evaluation_ledger_path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_external_generation_boundary_stop_returns_a_partial_search_result(bounds) -> None:
    config = GAConfig(population_size=8, max_generations=4, elite_count=1, random_seed=43)
    result = run_ga(
        AnalyticalEvaluator(bounds), bounds=bounds,
        fitness_scenario_id=SCENARIO_ID, config=config,
        externally_interrupted=lambda: True,
    )
    assert result.termination_reason == "externally_interrupted"
    assert result.current_generation == 0
    assert not result.is_complete


def test_midgeneration_stop_retains_checkpoint_and_reuses_committed_evaluations(
    bounds, tmp_path
) -> None:
    config = GAConfig(
        population_size=10, max_generations=4, elite_count=2,
        early_stop_patience=20, random_seed=46,
    )
    progress_records = []
    evaluator = AnalyticalEvaluator(bounds)
    interrupted = run_ga(
        evaluator, bounds=bounds, fitness_scenario_id=SCENARIO_ID, config=config,
        checkpoint_directory=tmp_path, fitness_codec=ANALYTICAL_CODEC,
        progress=progress_records.append,
        externally_interrupted=lambda: len(progress_records) >= 13,
    )
    checkpoint = json.loads(interrupted.checkpoint_path.read_text(encoding="utf-8"))
    assert interrupted.termination_reason == "externally_interrupted"
    assert interrupted.current_generation == 0
    assert interrupted.evaluation_statistics.candidate_placements == 13
    assert checkpoint["current_generation"] == 0
    assert len(interrupted.evaluation_ledger_path.read_text(encoding="utf-8").splitlines()) == 13

    continuation = AnalyticalEvaluator(bounds)
    resumed = run_ga(
        continuation, bounds=bounds, fitness_scenario_id=SCENARIO_ID, config=config,
        checkpoint_directory=tmp_path, fitness_codec=ANALYTICAL_CODEC, resume=True,
    )
    uninterrupted = run_ga(
        AnalyticalEvaluator(bounds), bounds=bounds,
        fitness_scenario_id=SCENARIO_ID, config=config,
    )
    assert tuple(item.chromosome for item in resumed.final_ranked_population) == tuple(
        item.chromosome for item in uninterrupted.final_ranked_population
    )
    assert evaluator.calls + continuation.calls == uninterrupted.evaluation_statistics.unique_fitness_evaluations


@pytest.mark.parametrize("mismatch", ("config", "scenario"))
def test_resume_rejects_configuration_or_scenario_mismatch(bounds, tmp_path, mismatch) -> None:
    config = GAConfig(
        population_size=8, max_generations=4, elite_count=1,
        early_stop_patience=20, random_seed=44,
    )
    run_ga(
        AnalyticalEvaluator(bounds), bounds=bounds,
        fitness_scenario_id=SCENARIO_ID, config=config,
        checkpoint_directory=tmp_path, stop_after_generation=1,
        fitness_codec=ANALYTICAL_CODEC,
    )
    resumed_config = replace(config, random_seed=45) if mismatch == "config" else config
    resumed_scenario = "different-scenario" if mismatch == "scenario" else SCENARIO_ID
    with pytest.raises(ValueError, match="mismatch"):
        run_ga(
            AnalyticalEvaluator(bounds, scenario_id=resumed_scenario), bounds=bounds,
            fitness_scenario_id=resumed_scenario, config=resumed_config,
            checkpoint_directory=tmp_path, resume=True,
            fitness_codec=ANALYTICAL_CODEC,
        )


def test_synthetic_objective_improves_and_reported_best_was_evaluated(bounds) -> None:
    config = GAConfig(
        population_size=24, max_generations=14, elite_count=2,
        early_stop_patience=20, random_seed=2026,
    )
    evaluator = AnalyticalEvaluator(bounds)
    result = run_ga(
        evaluator, bounds=bounds, fitness_scenario_id=SCENARIO_ID, config=config
    )
    initial_best = result.generation_history[0].best_feasible_objective
    final_best = result.best_feasible_fitness.objective_loiter_seconds
    assert final_best > initial_best + 1.0
    assert result.best_found.evaluation_key in evaluator.evaluation_keys
    assert result.best_found == rank_population(result.final_ranked_population)[0]


def test_constrained_synthetic_objective_returns_a_feasible_best(bounds) -> None:
    config = GAConfig(
        population_size=24, max_generations=10, elite_count=2,
        early_stop_patience=20, random_seed=2027,
    )
    evaluator = AnalyticalEvaluator(bounds, minimum_first_pair_sum=1.5)
    result = run_ga(
        evaluator, bounds=bounds, fitness_scenario_id=SCENARIO_ID, config=config
    )
    assert result.best_feasible_found
    assert result.best_feasible_fitness.static_feasible
    assert result.best_feasible_fitness.dynamically_feasible
    assert sum(result.best_feasible_chromosome.genes[:2]) >= 1.5


def test_ga_import_does_not_import_the_mission_simulator() -> None:
    code = (
        "import sys; import src.optimization.ga; "
        "print(int('src.simulation.simulator' in sys.modules))"
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert output == "0"


def test_repository_fitness_codec_round_trips_a_static_screened_result(bounds) -> None:
    from src.optimization.fitness import (
        ControllerBehavior,
        DurationSummary,
        DynamicConstraintRecord,
        FitnessDiagnostic,
        FitnessScenario,
        MissionResources,
        MissionValidity,
        PowerRange,
        evaluate_fitness,
    )

    scenario = FitnessScenario.nominal()
    result = evaluate_fitness(
        NormalizedChromosome((0.0,) * 6),
        bounds=bounds,
        scenario=scenario,
        mission_runner=lambda *args, **kwargs: pytest.fail("mission must be skipped"),
    )
    result = replace(
        result,
        resources=MissionResources(10.0, 5.0, 5.0, 4.9, 0.1, 0.4, 0.2, 3.0, 0.0, 2.0),
        controller_behavior=ControllerBehavior(
            1, 0.5, 0.2, 0.3,
            DurationSummary(2, 60.0, 90.0, 120.0),
            DurationSummary(1, 60.0, 60.0, 60.0),
            PowerRange(20.0, 80.0), PowerRange(18.0, 75.0), 3, 4,
        ),
        validity=MissionValidity(
            True, True, True, True, True, True, True, "fuel_reserve",
            0.0, 0.0, 0, 0, 0, 0.0, 1.0e-13, -1.0e-14,
            -0.7, 0.0, 0.7, ("engine_power_limited",),
        ),
        dynamic_constraint_names=("mock_constraint",),
        dynamic_constraints=(
            DynamicConstraintRecord(
                "mock_constraint", 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                "1", True, "at_most", "test",
            ),
        ),
        warnings=(FitnessDiagnostic("mock_warning", "diagnostic", "test"),),
    )
    codec = fitness_result_codec()
    recovered = codec.deserialize(codec.serialize(result))
    assert recovered == result
