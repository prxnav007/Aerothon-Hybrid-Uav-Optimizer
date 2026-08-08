"""Mission-connected orchestration for the first production GA run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.models.battery import BatteryPack
from src.optimization.chromosome import (
    GENE_NAMES,
    NormalizedChromosome,
    PlantThermostatDesignSpace,
    ideal_restart_fuel_seed,
    practical_thermostat_seed,
)
from src.optimization.fitness import FitnessResult, FitnessScenario, evaluate_fitness
from src.optimization.ga import (
    GAConfig,
    GAProgress,
    GAResult,
    fitness_result_codec,
    initialize_population,
    run_ga,
)

__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "PRODUCTION_WALL_LIMIT_S",
    "GenerationZeroGate",
    "ProductionRunOutcome",
    "ProgressReporter",
    "production_context",
    "run_production_ga",
]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = (
    _REPOSITORY_ROOT
    / "deliverables"
    / "optimization"
    / "ga_production_seed_20260808"
)
PRODUCTION_WALL_LIMIT_S = 60.0 * 60.0
_RUNTIME_DIRECTORY_NAME = "runtime"
_REFERENCE_TARGETS: dict[str, Any] = {
    "total_mission_seconds": 54876.880130153375,
    "objective_loiter_seconds": 48536.880130153375,
    "final_fuel_kg": 5.081139209846588,
    "final_soc": 0.3636680823973519,
    "minimum_soc": 0.17802150457596896,
    "restart_count": 50,
    "termination_reason": "fuel_reserve",
}
_REFERENCE_ABSOLUTE_TOLERANCES = {
    "total_mission_seconds": 1.0e-9,
    "objective_loiter_seconds": 1.0e-9,
    "final_fuel_kg": 1.0e-10,
    "final_soc": 1.0e-12,
    "minimum_soc": 1.0e-12,
}


@dataclass(frozen=True)
class GenerationZeroGate:
    passed: bool
    feasible_candidate_count: int
    practical_reference_evaluation_key: str
    reference_comparison: Mapping[str, Mapping[str, Any]]
    initialization_counts: Mapping[str, int]


@dataclass(frozen=True)
class ProductionRunOutcome:
    result: GAResult
    generation_zero_gate: GenerationZeroGate
    elapsed_runtime_s: float
    output_directory: Path
    generation_history_path: Path
    evaluated_candidates_path: Path
    best_found_path: Path
    summary_path: Path


def production_context(
) -> tuple[GAConfig, PlantThermostatDesignSpace, FitnessScenario]:
    """Return the explicit, immutable production search context."""
    config = GAConfig()
    bounds = PlantThermostatDesignSpace.from_battery(BatteryPack(10.0))
    scenario = FitnessScenario.nominal()
    return config, bounds, scenario


def _print_flush(message: str) -> None:
    print(message, flush=True)


class ProgressReporter:
    """Print concise candidate-level status while tracking the incumbent objective."""

    def __init__(
        self,
        *,
        started_at: float,
        clock: Callable[[], float] = time.perf_counter,
        emit: Callable[[str], None] = _print_flush,
    ) -> None:
        self.started_at = started_at
        self.clock = clock
        self.emit = emit
        self.best_objective_s: float | None = None

    def __call__(self, update: GAProgress) -> None:
        fitness = update.fitness
        if fitness.static_feasible and fitness.dynamically_feasible:
            objective = float(fitness.objective_loiter_seconds)
            self.best_objective_s = (
                objective
                if self.best_objective_s is None
                else max(self.best_objective_s, objective)
            )
        statistics = update.statistics
        if update.cache_hit and statistics.candidate_placements % 5:
            return
        objective_text = _optional_number(fitness.objective_loiter_seconds)
        best_text = _optional_number(self.best_objective_s)
        runtime_text = (
            f"{update.evaluation_runtime_s:.3f}"
            if fitness.run_mission_called and not update.cache_hit
            else "n/a"
        )
        self.emit(
            " ".join(
                (
                    f"generation={update.generation}",
                    f"candidate={update.candidate_index}",
                    f"static={fitness.static_feasible}",
                    f"dynamic={fitness.dynamically_feasible}",
                    f"mission={'called' if fitness.run_mission_called else 'skipped'}",
                    f"loiter_s={objective_text}",
                    f"best_s={best_text}",
                    f"mission_runtime_s={runtime_text}",
                    f"placements={statistics.candidate_placements}",
                    f"missions={statistics.mission_calls}",
                    f"cache_hits={statistics.cache_hits}",
                    f"elapsed_s={self.clock() - self.started_at:.3f}",
                )
            )
        )


def _optional_number(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _checkpoint_mode(output_directory: Path) -> bool:
    checkpoint = output_directory / "ga_checkpoint.json"
    ledger = output_directory / "evaluation_ledger.jsonl"
    if checkpoint.exists() != ledger.exists():
        raise RuntimeError(
            "production recovery requires both ga_checkpoint.json and "
            "evaluation_ledger.jsonl; existing files were not overwritten"
        )
    return checkpoint.exists()


def _checkpoint_generation(output_directory: Path) -> int:
    try:
        record = json.loads(
            (output_directory / "ga_checkpoint.json").read_text(encoding="utf-8")
        )
        generation = record["current_generation"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("the production checkpoint is unreadable") from error
    if type(generation) is not int or generation < 0:
        raise RuntimeError("the production checkpoint generation is invalid")
    return generation


def _read_ledger_records(path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"evaluation ledger line {line_number} is not valid JSON"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(f"evaluation ledger line {line_number} is not an object")
        records.append(record)
    return tuple(records)


def _reference_comparison(result: FitnessResult) -> dict[str, dict[str, Any]]:
    if result.resources is None or result.controller_behavior is None or result.validity is None:
        raise RuntimeError("the practical reference fitness result lacks mission audit data")
    actual = {
        "total_mission_seconds": result.total_mission_seconds,
        "objective_loiter_seconds": result.objective_loiter_seconds,
        "final_fuel_kg": result.resources.final_fuel_kg,
        "final_soc": result.resources.final_soc,
        "minimum_soc": result.resources.minimum_soc,
        "restart_count": result.controller_behavior.restart_count,
        "termination_reason": result.validity.termination_reason,
    }
    comparison: dict[str, dict[str, Any]] = {}
    for name, target in _REFERENCE_TARGETS.items():
        observed = actual[name]
        if name in _REFERENCE_ABSOLUTE_TOLERANCES:
            tolerance = _REFERENCE_ABSOLUTE_TOLERANCES[name]
            difference = float(observed) - float(target)
            passed = math.isclose(
                float(observed), float(target), rel_tol=0.0, abs_tol=tolerance
            )
        else:
            tolerance = 0
            difference = None if observed == target else "mismatch"
            passed = observed == target
        comparison[name] = {
            "actual": observed,
            "target": target,
            "difference": difference,
            "absolute_tolerance": tolerance,
            "passed": passed,
        }
    return comparison


def _validate_generation_zero(
    *,
    output_directory: Path,
    config: GAConfig,
    bounds: PlantThermostatDesignSpace,
) -> GenerationZeroGate:
    records = _read_ledger_records(output_directory / "evaluation_ledger.jsonl")
    generation_zero = tuple(record for record in records if record.get("generation") == 0)
    if len(generation_zero) != config.population_size:
        raise RuntimeError(
            "generation zero does not contain exactly the production population"
        )

    expected = initialize_population(
        config, bounds=bounds, rng=np.random.default_rng(config.random_seed)
    )
    expected_keys = {
        chromosome.cache_key(bounds=bounds) for chromosome in expected.chromosomes
    }
    actual_keys = {record.get("chromosome_cache_key") for record in generation_zero}
    if actual_keys != expected_keys:
        raise RuntimeError("generation-zero chromosomes do not match deterministic initialization")
    counts = Counter(str(record.get("candidate_context")) for record in generation_zero)
    expected_counts = Counter(expected.origins)
    if counts != expected_counts:
        raise RuntimeError("generation-zero initialization origins do not match production policy")

    practical_key = practical_thermostat_seed(bounds=bounds).cache_key(bounds=bounds)
    ideal_key = ideal_restart_fuel_seed(bounds=bounds).cache_key(bounds=bounds)
    if practical_key not in actual_keys or ideal_key not in actual_keys:
        raise RuntimeError("generation zero is missing an exact controller anchor")
    feasible_count = sum(
        bool(record.get("static_feasible") and record.get("dynamically_feasible"))
        for record in generation_zero
    )
    if feasible_count == 0:
        raise RuntimeError("generation zero produced no dynamically feasible candidate")

    reference_record = next(
        record
        for record in generation_zero
        if record.get("chromosome_cache_key") == practical_key
    )
    reference = fitness_result_codec().deserialize(reference_record["fitness_result"])
    if not isinstance(reference, FitnessResult):
        raise RuntimeError("the practical reference did not deserialize as FitnessResult")
    comparison = _reference_comparison(reference)
    accounting_valid = (
        reference.static_feasible
        and reference.dynamically_feasible
        and all(item.satisfied for item in reference.dynamic_constraints)
    )
    if not accounting_valid:
        raise RuntimeError("the practical reference mission accounting is invalid")
    failed = tuple(name for name, item in comparison.items() if not item["passed"])
    if failed:
        raise RuntimeError(
            "the practical reference regression failed for: " + ", ".join(failed)
        )
    return GenerationZeroGate(
        passed=True,
        feasible_candidate_count=feasible_count,
        practical_reference_evaluation_key=reference.evaluation_key,
        reference_comparison=comparison,
        initialization_counts=dict(sorted(counts.items())),
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _candidate_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        fitness = record["fitness_result"]
        decoded = fitness["decoded_design"]
        row = {
            "evaluation_key": record["evaluation_key"],
            "chromosome_cache_key": record["chromosome_cache_key"],
            **{
                f"normalized_{name}": record["chromosome_genes"][index]
                for index, name in enumerate(GENE_NAMES)
            },
            **decoded,
            "generation": record["generation"],
            "candidate_index": record["candidate_index"],
            "candidate_context": record["candidate_context"],
            "static_feasible": record["static_feasible"],
            "dynamically_feasible": record["dynamically_feasible"],
            "objective_loiter_seconds": record["objective_loiter_seconds"],
            "total_mission_seconds": fitness["total_mission_seconds"],
            "combined_normalized_violation": record[
                "combined_normalized_violation"
            ],
            "run_mission_called": record["run_mission_called"],
            "evaluation_runtime_s": record["evaluation_runtime_s"],
            "evaluated_at_utc": record["evaluated_at_utc"],
            "failure_category": fitness["failure_category"],
            "failure_message": fitness["failure_message"],
        }
        rows.append(row)
    return rows


def _dynamic_margin(record: Any) -> float:
    if record.relationship == "at_least":
        return record.quantity - record.required_or_allowed
    return record.required_or_allowed - record.quantity


def _bound_pinning(chromosome: NormalizedChromosome) -> list[dict[str, Any]]:
    pinned = []
    for name, value in zip(GENE_NAMES, chromosome.genes):
        if value <= 0.01:
            pinned.append({"gene": name, "normalized_value": value, "bound": "lower"})
        elif value >= 0.99:
            pinned.append({"gene": name, "normalized_value": value, "bound": "upper"})
    return pinned


def _constraint_pinning(fitness: FitnessResult) -> dict[str, list[dict[str, Any]]]:
    static = []
    for record in fitness.static_constraints:
        relative = abs(record.margin) / max(abs(record.required_or_allowed), 1.0e-12)
        if record.satisfied and relative <= 0.01:
            static.append({**asdict(record), "relative_margin": relative})
    dynamic = []
    for record in fitness.dynamic_constraints:
        margin = _dynamic_margin(record)
        relative = abs(margin) / max(abs(record.normalization_scale), 1.0e-12)
        if record.satisfied and relative <= 0.01:
            dynamic.append(
                {**asdict(record), "signed_margin": margin, "relative_margin": relative}
            )
    return {"static": static, "dynamic": dynamic}


def _best_payload(
    result: GAResult,
    gate: GenerationZeroGate,
    *,
    elapsed_runtime_s: float,
    bounds: PlantThermostatDesignSpace,
    scenario: FitnessScenario,
) -> dict[str, Any]:
    if result.best_feasible_chromosome is None or result.best_feasible_fitness is None:
        raise RuntimeError("the GA result has no feasible design to report")
    chromosome = result.best_feasible_chromosome
    fitness = result.best_feasible_fitness
    if not isinstance(fitness, FitnessResult):
        raise TypeError("production GA best result must be FitnessResult")
    if fitness.resources is None or fitness.controller_behavior is None or fitness.validity is None:
        raise RuntimeError("best feasible result lacks mission audit data")
    reference = float(_REFERENCE_TARGETS["objective_loiter_seconds"])
    objective = float(fitness.objective_loiter_seconds)
    improvement = objective - reference
    resolved = fitness.resolved_design
    return {
        "claim": "best feasible design found by this one-seed GA run",
        "run": {
            "ga_config": result.config.to_dict(),
            "fitness_scenario": scenario.to_dict(),
            "design_space": bounds.to_dict(),
            "termination_reason": result.termination_reason,
            "completed_generation_count": result.completed_generation_count,
            "current_generation": result.current_generation,
            "is_complete": result.is_complete,
            "elapsed_runtime_s": elapsed_runtime_s,
            "statistics": asdict(result.evaluation_statistics),
            "warnings": list(result.warnings),
        },
        "generation_zero_gate": asdict(gate),
        "normalized_chromosome": chromosome.to_dict(bounds=bounds),
        "decoded_design": asdict(fitness.decoded_design),
        "resolved_design_id": fitness.resolved_design_id,
        "aircraft": {
            "wing_area_m2": resolved.wing.wing_area_m2,
            "aspect_ratio": resolved.wing.aspect_ratio,
            "span_m": resolved.wing.span_m,
            "cd0": resolved.wing.cd0,
            "engine_rating_kw": resolved.power.engine_rating_sea_level_kw,
            "battery_capacity_kwh": fitness.decoded_design.battery_capacity_kwh,
            "soc_low": resolved.soc_low,
            "soc_high": resolved.soc_high,
            "soc_gap": resolved.soc_high - resolved.soc_low,
            "dry_mass_kg": resolved.masses.dry_kg,
            "initial_fuel_kg": resolved.fuel.initial_usable_fuel_kg,
        },
        "mission": {
            "objective_loiter_seconds": objective,
            "total_mission_seconds": fitness.total_mission_seconds,
            "objective_hours": fitness.objective_hours,
            "reference_improvement_seconds": improvement,
            "reference_improvement_minutes": improvement / 60.0,
            "reference_improvement_percent": improvement / reference * 100.0,
            "resources": asdict(fitness.resources),
            "controller_behavior": asdict(fitness.controller_behavior),
            "validity": asdict(fitness.validity),
        },
        "constraints": {
            "static": [asdict(item) for item in fitness.static_constraints],
            "dynamic": [
                {**asdict(item), "signed_margin": _dynamic_margin(item)}
                for item in fitness.dynamic_constraints
            ],
            "combined_normalized_violation": (
                fitness.combined_static_dynamic_violation
            ),
        },
        "bound_pinning_within_one_percent": _bound_pinning(chromosome),
        "constraint_pinning_within_one_percent": _constraint_pinning(fitness),
        "best_objective_by_generation": [
            {
                "generation": item.generation,
                "best_feasible_objective_seconds": item.best_feasible_objective,
            }
            for item in result.generation_history
        ],
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    run = payload["run"]
    aircraft = payload["aircraft"]
    mission = payload["mission"]
    resources = mission["resources"]
    behavior = mission["controller_behavior"]
    statistics = run["statistics"]
    pinned = payload["bound_pinning_within_one_percent"]
    return "\n".join(
        (
            "# Production plant-thermostat GA — seed 20260808",
            "",
            "This records the **best feasible design found by this one-seed GA run**; "
            "it is not a claim of global optimality.",
            "",
            "## Search outcome",
            "",
            f"- Generation-zero gate: **passed**.",
            f"- Termination: `{run['termination_reason']}` after "
            f"{run['completed_generation_count']} completed evaluated populations.",
            f"- Runtime: {run['elapsed_runtime_s']:.3f} s.",
            f"- Placements / unique evaluations / missions: "
            f"{statistics['candidate_placements']} / "
            f"{statistics['unique_fitness_evaluations']} / {statistics['mission_calls']}.",
            f"- Static-infeasible / dynamic-infeasible / feasible: "
            f"{statistics['static_infeasible_results']} / "
            f"{statistics['dynamic_infeasible_results']} / "
            f"{statistics['feasible_results']}.",
            f"- Cache hits: {statistics['cache_hits']}.",
            f"- Stagnation stop: {str(run['termination_reason'] == 'stagnation').lower()}.",
            "",
            "## Best design",
            "",
            f"- Wing: {aircraft['wing_area_m2']:.12g} m², AR "
            f"{aircraft['aspect_ratio']:.12g}, span {aircraft['span_m']:.12g} m, "
            f"C_D0 {aircraft['cd0']:.12g}.",
            f"- Engine / battery: {aircraft['engine_rating_kw']:.12g} kW / "
            f"{aircraft['battery_capacity_kwh']:.12g} kWh.",
            f"- Thermostat: {aircraft['soc_low']:.12g}–{aircraft['soc_high']:.12g} "
            f"(gap {aircraft['soc_gap']:.12g}).",
            f"- Dry mass / initial fuel: {aircraft['dry_mass_kg']:.12g} kg / "
            f"{aircraft['initial_fuel_kg']:.12g} kg.",
            "",
            "## Mission",
            "",
            f"- Loiter / total: {mission['objective_loiter_seconds']:.12g} s / "
            f"{mission['total_mission_seconds']:.12g} s.",
            f"- Gain over practical reference: "
            f"{mission['reference_improvement_seconds']:.12g} s "
            f"({mission['reference_improvement_minutes']:.12g} min, "
            f"{mission['reference_improvement_percent']:.9g}%).",
            f"- Final fuel / reserve slack: {resources['final_fuel_kg']:.12g} kg / "
            f"{resources['fuel_slack_above_reserve_kg']:.12g} kg.",
            f"- Final / minimum SoC: {resources['final_soc']:.12g} / "
            f"{resources['minimum_soc']:.12g}.",
            f"- Terminal usable battery energy: "
            f"{resources['usable_battery_energy_above_floor_kwh']:.12g} kWh.",
            f"- Restarts / restart fuel: {behavior['restart_count']} / "
            f"{resources['restart_fuel_consumed_kg']:.12g} kg.",
            f"- Overall / loiter engine-OFF fraction: "
            f"{behavior['overall_engine_off_fraction']:.12g} / "
            f"{behavior['loiter_engine_off_fraction']:.12g}.",
            "",
            "## Bounds and recovery",
            "",
            f"- Genes within 1% of a normalized bound: "
            f"`{json.dumps(pinned, sort_keys=True)}`.",
            "- Checkpoint and evaluation ledger are the authoritative recovery state.",
            "- Resume command: `python -u -m src.optimization.ga_runner`.",
            "",
        )
    )


def _write_result_artifacts(
    result: GAResult,
    gate: GenerationZeroGate,
    *,
    output_directory: Path,
    runtime_directory: Path,
    elapsed_runtime_s: float,
    bounds: PlantThermostatDesignSpace,
    scenario: FitnessScenario,
) -> tuple[Path, Path, Path, Path]:
    history_path = output_directory / "generation_history.csv"
    candidates_path = output_directory / "evaluated_candidates.csv"
    best_path = output_directory / "best_found.json"
    summary_path = output_directory / "ga_run_summary.md"
    history_rows = [item.to_dict() for item in result.generation_history]
    _atomic_csv(history_path, history_rows, tuple(history_rows[0]))
    ledger_records = _read_ledger_records(runtime_directory / "evaluation_ledger.jsonl")
    candidate_rows = _candidate_rows(ledger_records)
    _atomic_csv(candidates_path, candidate_rows, tuple(candidate_rows[0]))
    payload = _best_payload(
        result,
        gate,
        elapsed_runtime_s=elapsed_runtime_s,
        bounds=bounds,
        scenario=scenario,
    )
    _atomic_json(best_path, payload)
    _atomic_text(summary_path, _summary_markdown(payload))
    return history_path, candidates_path, best_path, summary_path


def run_production_ga(
    *,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    wall_limit_s: float = PRODUCTION_WALL_LIMIT_S,
    evaluator: Callable[[NormalizedChromosome], FitnessResult] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    emit: Callable[[str], None] = _print_flush,
) -> ProductionRunOutcome:
    """Run or resume the single authorized production seed and write reports."""
    limit = float(wall_limit_s)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("wall_limit_s must be a positive finite number")
    directory = Path(output_directory).resolve()
    runtime_directory = directory / _RUNTIME_DIRECTORY_NAME
    config, bounds, scenario = production_context()
    mission_evaluator = evaluator or (
        lambda chromosome: evaluate_fitness(
            chromosome, bounds=bounds, scenario=scenario
        )
    )
    started = clock()
    reporter = ProgressReporter(started_at=started, clock=clock, emit=emit)
    codec = fitness_result_codec()
    resume = _checkpoint_mode(runtime_directory)

    if resume:
        current_generation = _checkpoint_generation(runtime_directory)
        if current_generation < config.max_generations - 1:
            gate_result = run_ga(
                mission_evaluator,
                bounds=bounds,
                fitness_scenario_id=scenario.identity,
                config=config,
                checkpoint_directory=runtime_directory,
                resume=True,
                stop_after_generation=current_generation,
                fitness_codec=codec,
                progress=reporter,
            )
        else:
            gate_result = run_ga(
                mission_evaluator,
                bounds=bounds,
                fitness_scenario_id=scenario.identity,
                config=config,
                checkpoint_directory=runtime_directory,
                resume=True,
                fitness_codec=codec,
                progress=reporter,
            )
    else:
        gate_result = run_ga(
            mission_evaluator,
            bounds=bounds,
            fitness_scenario_id=scenario.identity,
            config=config,
            checkpoint_directory=runtime_directory,
            stop_after_generation=0,
            fitness_codec=codec,
            progress=reporter,
        )

    gate = _validate_generation_zero(
        output_directory=runtime_directory, config=config, bounds=bounds
    )
    emit(
        f"generation_zero_gate=passed feasible={gate.feasible_candidate_count} "
        "reference=passed continuing_same_checkpoint=true"
    )
    if gate_result.current_generation >= config.max_generations - 1:
        result = gate_result
    else:
        result = run_ga(
            mission_evaluator,
            bounds=bounds,
            fitness_scenario_id=scenario.identity,
            config=config,
            checkpoint_directory=runtime_directory,
            resume=True,
            externally_interrupted=lambda: clock() - started >= limit,
            fitness_codec=codec,
            progress=reporter,
        )
    elapsed = clock() - started
    paths = _write_result_artifacts(
        result,
        gate,
        output_directory=directory,
        runtime_directory=runtime_directory,
        elapsed_runtime_s=elapsed,
        bounds=bounds,
        scenario=scenario,
    )
    emit(
        f"termination={result.termination_reason} generation={result.current_generation} "
        f"elapsed_s={elapsed:.3f} reports_written=true"
    )
    return ProductionRunOutcome(result, gate, elapsed, directory, *paths)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--wall-limit-seconds", type=float, default=PRODUCTION_WALL_LIMIT_S
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    outcome = run_production_ga(
        output_directory=arguments.output_directory,
        wall_limit_s=arguments.wall_limit_seconds,
    )
    return 0 if outcome.result.best_feasible_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
