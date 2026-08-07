"""Focused production-runner configuration, gate, and recovery tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.optimization.chromosome import (
    ideal_restart_fuel_seed,
    practical_thermostat_seed,
)
from src.optimization.ga import GAProgress, initialize_population
from src.optimization.ga_runner import (
    DEFAULT_OUTPUT_DIRECTORY,
    ProgressReporter,
    _checkpoint_mode,
    _reference_comparison,
    production_context,
)


def test_production_context_uses_the_authorized_single_seed_and_nominal_scenario() -> None:
    config, bounds, scenario = production_context()
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
    assert config.random_seed == 20260808
    assert scenario.restart_fuel_kg == 0.1
    assert scenario.minimum_on_time_s == scenario.minimum_off_time_s == 60.0
    assert scenario.timestep_s == 60.0
    assert scenario.static_scenario.cruise_altitude_m == 3000.0
    assert bounds.soc_floor == scenario.static_scenario.battery_soc_floor
    assert DEFAULT_OUTPUT_DIRECTORY.name == "ga_production_seed_20260808"


def test_generation_zero_initializer_has_48_lhs_two_anchors_and_14_perturbations() -> None:
    config, bounds, _ = production_context()
    initial = initialize_population(
        config, bounds=bounds, rng=np.random.default_rng(config.random_seed)
    )
    counts = Counter(initial.origins)
    assert counts == {
        "latin_hypercube": 48,
        "practical_seed": 1,
        "ideal_seed": 1,
        "practical_perturbation": 7,
        "ideal_perturbation": 7,
    }
    assert practical_thermostat_seed(bounds=bounds) in initial.chromosomes
    assert ideal_restart_fuel_seed(bounds=bounds) in initial.chromosomes


def test_reference_comparison_uses_the_established_deterministic_tolerances() -> None:
    result = SimpleNamespace(
        total_mission_seconds=54876.880130153375,
        objective_loiter_seconds=48536.880130153375,
        resources=SimpleNamespace(
            final_fuel_kg=5.081139209846588,
            final_soc=0.3636680823973519,
            minimum_soc=0.17802150457596896,
        ),
        controller_behavior=SimpleNamespace(restart_count=50),
        validity=SimpleNamespace(termination_reason="fuel_reserve"),
    )
    assert all(item["passed"] for item in _reference_comparison(result).values())
    result.resources.final_fuel_kg += 2.0e-10
    assert not _reference_comparison(result)["final_fuel_kg"]["passed"]


def test_recovery_refuses_an_orphan_checkpoint_or_ledger(tmp_path: Path) -> None:
    (tmp_path / "ga_checkpoint.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires both"):
        _checkpoint_mode(tmp_path)


def test_progress_reports_unique_evaluations_and_tracks_best_objective() -> None:
    messages = []
    statistics = SimpleNamespace(
        candidate_placements=1, mission_calls=1, cache_hits=0
    )
    fitness = SimpleNamespace(
        static_feasible=True,
        dynamically_feasible=True,
        objective_loiter_seconds=123.0,
        run_mission_called=True,
    )
    update = GAProgress(0, 4, "latin_hypercube", fitness, False, 0.25, statistics)
    ProgressReporter(started_at=10.0, clock=lambda: 12.0, emit=messages.append)(update)
    assert len(messages) == 1
    assert "generation=0" in messages[0]
    assert "candidate=4" in messages[0]
    assert "best_s=123.000000" in messages[0]
    assert "mission_runtime_s=0.250" in messages[0]
