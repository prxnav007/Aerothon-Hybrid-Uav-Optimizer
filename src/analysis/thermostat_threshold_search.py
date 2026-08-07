"""Deterministic full-mission tuning of the two global thermostat thresholds."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Sequence

from src.analysis.thermostat_mission import (
    REFERENCE_INITIAL_THERMOSTAT_STATE,
    REFERENCE_THERMOSTAT_PARAMETERS,
    REFERENCE_TIMESTEP_S,
    ThermostatReferenceRun,
    build_reference_aircraft,
    summarise_thermostat_mission,
)
from src.control.pi_ecms import PIECMS
from src.control.thermostat import ThermostatParameters, ThermostatState
from src.models.atmosphere import atmosphere
from src.simulation.mission import MissionProfile, Phase, Termination, ps1_mission
from src.simulation.simulator import (
    Aircraft,
    MissionResult,
    TimeStep,
    mission_energy_balance,
    run_mission,
)

__all__ = [
    "PhaseDependentGate",
    "PhaseLedger",
    "ThresholdEvaluation",
    "ThresholdSearchConfig",
    "ThresholdSearchResult",
    "assess_phase_dependent_gate",
    "build_phase_ledger",
    "check_one_step_loiter_extension",
    "coarse_threshold_candidates",
    "default_threshold_search_config",
    "evaluate_threshold_candidate",
    "run_threshold_search",
    "select_best_evaluation",
    "write_phase_ledger_csv",
]

_REFERENCE_PAIR = (0.4, 0.6)
_PAIR_PRECISION = 9
_POWER_BALANCE_TOLERANCE_KW = 1.0e-6
_FUEL_LEDGER_TOLERANCE_KG = 1.0e-9
_DISCRETE_ENERGY_TOLERANCE = 1.0e-12
_ENDURANCE_TIE_TOLERANCE_S = 1.0e-6


def _pair_key(soc_low: float, soc_high: float) -> str:
    return f"{soc_low:.{_PAIR_PRECISION}f}:{soc_high:.{_PAIR_PRECISION}f}"


@dataclass(frozen=True)
class ThresholdSearchConfig:
    """Bounds, coarse mesh, refinement, and work cap for one search."""

    soc_min: float
    soc_max: float
    # Bounded deterministic search policy; see assumptions.md C-09.
    minimum_separation: float = 0.05
    maximum_evaluations: int = 72
    retained_regions: int = 4
    refinement_step: float = 0.025
    coarse_low_values: tuple[float, ...] = (
        0.05,
        0.15,
        0.25,
        0.35,
        0.4,
        0.45,
        0.55,
        0.7,
        0.85,
    )
    coarse_high_values: tuple[float, ...] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )

    def __post_init__(self) -> None:
        values = (
            self.soc_min,
            self.soc_max,
            self.minimum_separation,
            self.refinement_step,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("threshold search bounds and steps must be finite")
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError("search bounds must satisfy 0 <= soc_min < soc_max <= 1")
        if not 0.0 < self.minimum_separation < self.soc_max - self.soc_min:
            raise ValueError("minimum_separation must be positive and fit inside the bounds")
        if self.maximum_evaluations < 1:
            raise ValueError("maximum_evaluations must be positive")
        if self.retained_regions < 1 or self.refinement_step <= 0.0:
            raise ValueError("retained_regions and refinement_step must be positive")
        if not self.coarse_low_values or not self.coarse_high_values:
            raise ValueError("coarse threshold meshes must not be empty")

    def validate_pair(self, soc_low: float, soc_high: float) -> tuple[float, float]:
        low = float(soc_low)
        high = float(soc_high)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("thresholds must be finite")
        if low < self.soc_min - 1.0e-12 or high > self.soc_max + 1.0e-12:
            raise ValueError("threshold pair lies outside the battery SoC bounds")
        if high - low < self.minimum_separation - 1.0e-12:
            raise ValueError("threshold pair violates the minimum separation")
        return low, high


def default_threshold_search_config() -> ThresholdSearchConfig:
    """Return the bounded search tied to the reference battery limits."""
    battery = build_reference_aircraft().battery
    return ThresholdSearchConfig(soc_min=battery.soc_min, soc_max=1.0)


def coarse_threshold_candidates(
    config: ThresholdSearchConfig,
) -> tuple[tuple[float, float], ...]:
    """Return the deterministic triangular mesh with the reference first."""
    candidates: list[tuple[float, float]] = []
    for low in config.coarse_low_values:
        for high in config.coarse_high_values:
            try:
                pair = config.validate_pair(low, high)
            except ValueError:
                continue
            candidates.append(pair)
    config.validate_pair(*_REFERENCE_PAIR)
    unique = {_pair_key(*pair): pair for pair in candidates}
    unique[_pair_key(*_REFERENCE_PAIR)] = _REFERENCE_PAIR
    ordered = sorted(unique.values())
    ordered.remove(_REFERENCE_PAIR)
    return (_REFERENCE_PAIR, *ordered)


@dataclass(frozen=True)
class PhaseLedger:
    """Fuel, battery, engine-state, and regime ledger for one mission phase."""

    controller: str
    phase: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    start_fuel_kg: float
    end_fuel_kg: float
    fuel_consumed_kg: float
    mean_fuel_rate_kg_h: float
    start_soc: float
    end_soc: float
    start_stored_energy_kwh: float
    end_stored_energy_kwh: float
    stored_energy_change_kwh: float
    engine_off_fraction: float
    restart_count: int
    regime_time_s_json: str


def build_phase_ledger(
    controller_name: str,
    aircraft: Aircraft,
    mission: MissionProfile,
    result: MissionResult,
    *,
    initial_soc: float = 1.0,
) -> tuple[PhaseLedger, ...]:
    """Build phase endpoints from the immutable mission log."""
    if result.log is None:
        raise ValueError("phase ledger requires a recorded mission log")
    entries = result.log
    previous_time = 0.0
    previous_fuel = aircraft.masses.fuel_kg
    previous_soc = float(initial_soc)
    previous_engine_on = True
    ledgers: list[PhaseLedger] = []
    for phase in mission.phases:
        steps = tuple(step for step in entries if step.phase == phase.name)
        if not steps:
            continue
        restarts = 0
        regimes: dict[str, float] = {}
        for step in steps:
            engine_on = not step.engine_shut_down
            restarts += int(not previous_engine_on and engine_on)
            previous_engine_on = engine_on
            if step.controller_regime is not None:
                regimes[step.controller_regime] = (
                    regimes.get(step.controller_regime, 0.0) + step.dt_s
                )
        end = steps[-1]
        duration_s = sum(step.dt_s for step in steps)
        fuel_consumed = previous_fuel - end.fuel_remaining_kg
        start_energy = float(aircraft.battery.stored_energy_kwh(previous_soc))
        end_energy = float(aircraft.battery.stored_energy_kwh(end.soc))
        ledgers.append(
            PhaseLedger(
                controller=controller_name,
                phase=phase.name,
                start_time_s=previous_time,
                end_time_s=end.time_s,
                duration_s=duration_s,
                start_fuel_kg=previous_fuel,
                end_fuel_kg=end.fuel_remaining_kg,
                fuel_consumed_kg=fuel_consumed,
                mean_fuel_rate_kg_h=(
                    fuel_consumed / (duration_s / 3600.0) if duration_s > 0.0 else 0.0
                ),
                start_soc=previous_soc,
                end_soc=end.soc,
                start_stored_energy_kwh=start_energy,
                end_stored_energy_kwh=end_energy,
                stored_energy_change_kwh=end_energy - start_energy,
                engine_off_fraction=sum(
                    step.dt_s for step in steps if step.engine_shut_down
                )
                / duration_s,
                restart_count=restarts,
                regime_time_s_json=json.dumps(regimes, sort_keys=True),
            )
        )
        previous_time = end.time_s
        previous_fuel = end.fuel_remaining_kg
        previous_soc = end.soc
    return tuple(ledgers)


def write_phase_ledger_csv(
    ledgers: Sequence[PhaseLedger], output_path: str | Path
) -> Path:
    """Write PI and thermostat phase ledgers in one machine-readable table."""
    if not ledgers:
        raise ValueError("ledgers must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(asdict(ledgers[0])))
        writer.writeheader()
        writer.writerows(asdict(ledger) for ledger in ledgers)
    return path


@dataclass(frozen=True)
class ThresholdEvaluation:
    """One checkpoint row containing objective, feasibility, and diagnostics."""

    candidate_id: str
    stage: str
    soc_low: float
    soc_high: float
    total_time_s: float
    loiter_time_s: float
    mission_complete: bool
    feasible: bool
    feasibility_reasons_json: str
    termination_reason: str
    final_soc: float
    minimum_soc: float
    final_stored_energy_kwh: float
    fuel_consumed_kg: float
    fuel_remaining_kg: float
    restart_count: int
    loiter_restarts_per_hour: float
    overall_engine_off_fraction: float
    loiter_engine_off_fraction: float
    mean_engine_on_power_kw: float
    minimum_engine_on_power_kw: float
    maximum_engine_on_power_kw: float
    constraint_encounters_json: str
    descent_landing_completed: bool
    reserve_shortfall_kg: float
    maximum_power_balance_residual_kw: float
    fuel_ledger_residual_kg: float
    energy_ledger_residual_kwh: float
    discrete_energy_residual_fraction: float
    minimum_transition_margin_s: float
    minimum_bus_supply_margin_kw: float
    failure_flags_json: str
    runtime_s: float

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "ThresholdEvaluation":
        boolean_fields = {"mission_complete", "feasible", "descent_landing_completed"}
        integer_fields = {"restart_count"}
        string_fields = {
            "candidate_id",
            "stage",
            "feasibility_reasons_json",
            "termination_reason",
            "constraint_encounters_json",
            "failure_flags_json",
        }
        values: dict[str, object] = {}
        for field in fields(cls):
            value = row[field.name]
            if field.name in string_fields:
                values[field.name] = value
            elif field.name in boolean_fields:
                values[field.name] = value.lower() == "true"
            elif field.name in integer_fields:
                values[field.name] = int(value)
            else:
                values[field.name] = float(value)
        return cls(**values)  # type: ignore[arg-type]


def _minimum_transition_margin(steps: Sequence[TimeStep]) -> float:
    if not steps:
        return 0.0
    runs: list[tuple[bool, float]] = []
    current = not steps[0].engine_shut_down
    duration = 0.0
    for step in steps:
        engine_on = not step.engine_shut_down
        if engine_on != current:
            runs.append((current, duration))
            current = engine_on
            duration = 0.0
        duration += step.dt_s
    runs.append((current, duration))
    margins = [duration_s - 60.0 for _, duration_s in runs]
    return min(margins, default=0.0)


def _minimum_bus_supply_margin(
    aircraft: Aircraft, steps: Sequence[TimeStep], initial_soc: float
) -> float:
    sea_level_density = float(atmosphere(0.0).density_kg_m3)
    soc = initial_soc
    margins = []
    for step in steps:
        sigma = step.density_kg_m3 / sea_level_density
        engine_bus = float(
            aircraft.powertrain.bus_power_from_engine(
                aircraft.engine.max_power_kw(sigma)
            )
        )
        battery_bus = aircraft.battery.available_discharge_kw(soc, step.dt_s)
        margins.append(engine_bus + battery_bus - step.bus_demand_kw)
        soc = step.soc
    return min(margins, default=0.0)


def _feasibility_reasons(
    run: ThermostatReferenceRun, summary: dict[str, object]
) -> tuple[str, ...]:
    result = run.result
    if result.log is None:
        return ("missing_log",)
    steps = result.log
    reasons: list[str] = []
    phases_seen = tuple(dict.fromkeys(step.phase for step in steps))
    if phases_seen != run.mission.phase_names:
        reasons.append("six_phases_not_completed")
    mission_summary = summary["mission"]
    failure_summary = summary["failures"]
    accounting = summary["accounting"]
    if not result.mission_complete:
        reasons.append("mission_incomplete")
    if not mission_summary["descent_and_landing_completed"]:
        reasons.append("descent_landing_incomplete")
    if result.fuel_remaining_kg < run.mission.fuel_reserve_kg - 1.0e-10:
        reasons.append("reserve_shortfall")
    if result.min_soc < run.aircraft.battery.soc_min - 1.0e-10:
        reasons.append("soc_below_floor")
    if failure_summary["hard_dwell_violation_count"]:
        reasons.append("hard_dwell_violation")
    if failure_summary["hard_dwell_infeasible"]:
        reasons.append("hard_dwell_infeasible")
    if failure_summary["controller_infeasible"]:
        reasons.append("controller_infeasible")
    if failure_summary["plant_infeasible"]:
        reasons.append("plant_infeasible")
    terminal_flags = {
        "power_shortfall",
        "fuel_exhausted",
        "fuel_reserve_shortfall",
        "altitude_unreachable",
    }
    if terminal_flags.intersection(result.failure_flags):
        reasons.append("terminal_constraint_failure")
    if accounting["maximum_bus_balance_residual_kw"] > _POWER_BALANCE_TOLERANCE_KW:
        reasons.append("power_ledger_open")
    if abs(accounting["fuel_reconstruction_residual_kg"]) > _FUEL_LEDGER_TOLERANCE_KG:
        reasons.append("fuel_ledger_open")
    if accounting["discrete_energy_balance_residual_fraction"] > (
        _DISCRETE_ENERGY_TOLERANCE
    ):
        reasons.append("energy_ledger_open")
    return tuple(reasons)


def _evaluation_from_run(
    run: ThermostatReferenceRun,
    *,
    stage: str,
    runtime_s: float,
) -> ThresholdEvaluation:
    summary = summarise_thermostat_mission(run)
    result = run.result
    assert result.log is not None
    parameters = run.parameters
    reasons = _feasibility_reasons(run, summary)
    on_steps = tuple(step for step in result.log if not step.engine_shut_down)
    on_time_s = sum(step.dt_s for step in on_steps)
    mean_on_power = (
        sum(step.engine_shaft_kw * step.dt_s for step in on_steps) / on_time_s
        if on_time_s > 0.0
        else 0.0
    )
    mission_summary = summary["mission"]
    engine_summary = summary["engine_schedule"]
    accounting = summary["accounting"]
    return ThresholdEvaluation(
        candidate_id=_pair_key(parameters.soc_low, parameters.soc_high),
        stage=stage,
        soc_low=parameters.soc_low,
        soc_high=parameters.soc_high,
        total_time_s=result.endurance_s,
        loiter_time_s=mission_summary["loiter_time_s"],
        mission_complete=result.mission_complete,
        feasible=not reasons,
        feasibility_reasons_json=json.dumps(reasons),
        termination_reason=result.termination_reason,
        final_soc=result.final_soc,
        minimum_soc=result.min_soc,
        final_stored_energy_kwh=float(
            run.aircraft.battery.stored_energy_kwh(result.final_soc)
        ),
        fuel_consumed_kg=result.fuel_used_kg,
        fuel_remaining_kg=result.fuel_remaining_kg,
        restart_count=engine_summary["restart_count"],
        loiter_restarts_per_hour=engine_summary["loiter_restarts_per_hour"],
        overall_engine_off_fraction=engine_summary["total_engine_off_fraction"],
        loiter_engine_off_fraction=engine_summary["loiter_engine_off_fraction"],
        mean_engine_on_power_kw=mean_on_power,
        minimum_engine_on_power_kw=engine_summary["delivered_on_power_range"][
            "minimum_kw"
        ],
        maximum_engine_on_power_kw=engine_summary["delivered_on_power_range"][
            "maximum_kw"
        ],
        constraint_encounters_json=json.dumps(
            summary["constraints"], sort_keys=True
        ),
        descent_landing_completed=mission_summary["descent_and_landing_completed"],
        reserve_shortfall_kg=mission_summary["reserve_shortfall_kg"],
        maximum_power_balance_residual_kw=accounting[
            "maximum_bus_balance_residual_kw"
        ],
        fuel_ledger_residual_kg=accounting["fuel_reconstruction_residual_kg"],
        energy_ledger_residual_kwh=accounting["energy_balance_residual_kwh"],
        discrete_energy_residual_fraction=accounting[
            "discrete_energy_balance_residual_fraction"
        ],
        minimum_transition_margin_s=_minimum_transition_margin(result.log),
        minimum_bus_supply_margin_kw=_minimum_bus_supply_margin(
            run.aircraft, result.log, run.initial_soc
        ),
        failure_flags_json=json.dumps(result.failure_flags),
        runtime_s=runtime_s,
    )


def simulate_threshold_candidate(
    soc_low: float,
    soc_high: float,
    *,
    stage: str,
    config: ThresholdSearchConfig | None = None,
) -> tuple[ThresholdEvaluation, ThermostatReferenceRun]:
    """Run one candidate with fresh immutable models and explicit state."""
    search_config = config or default_threshold_search_config()
    low, high = search_config.validate_pair(soc_low, soc_high)
    aircraft = build_reference_aircraft()
    mission = ps1_mission()
    parameters = replace(
        REFERENCE_THERMOSTAT_PARAMETERS,
        soc_low=low,
        soc_high=high,
    )
    initial_state = ThermostatState(
        engine_on=REFERENCE_INITIAL_THERMOSTAT_STATE.engine_on,
        elapsed_in_state_s=REFERENCE_INITIAL_THERMOSTAT_STATE.elapsed_in_state_s,
        restart_count=REFERENCE_INITIAL_THERMOSTAT_STATE.restart_count,
        terminal_depletion=REFERENCE_INITIAL_THERMOSTAT_STATE.terminal_depletion,
    )
    start = time.perf_counter()
    result = run_mission(
        aircraft,
        mission,
        thermostat_parameters=parameters,
        initial_thermostat_state=initial_state,
        dt_s=REFERENCE_TIMESTEP_S,
        initial_soc=1.0,
        record_log=True,
    )
    runtime_s = time.perf_counter() - start
    run = ThermostatReferenceRun(
        aircraft=aircraft,
        mission=mission,
        parameters=parameters,
        initial_state=initial_state,
        initial_soc=1.0,
        result=result,
    )
    return _evaluation_from_run(run, stage=stage, runtime_s=runtime_s), run


def evaluate_threshold_candidate(
    soc_low: float,
    soc_high: float,
    stage: str,
    config: ThresholdSearchConfig,
) -> ThresholdEvaluation:
    """Return one checkpoint-ready full-mission threshold evaluation."""
    evaluation, _ = simulate_threshold_candidate(
        soc_low, soc_high, stage=stage, config=config
    )
    return evaluation


def _append_checkpoint(path: Path, evaluation: ThresholdEvaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(asdict(evaluation)))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(evaluation))
        stream.flush()
        os.fsync(stream.fileno())


def _load_checkpoint(path: Path) -> list[ThresholdEvaluation]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [ThresholdEvaluation.from_csv_row(row) for row in csv.DictReader(stream)]


def select_best_evaluation(
    evaluations: Sequence[ThresholdEvaluation],
) -> tuple[ThresholdEvaluation, tuple[ThresholdEvaluation, ...]]:
    """Select the maximum-endurance feasible row and expose exact event ties."""
    feasible = tuple(evaluation for evaluation in evaluations if evaluation.feasible)
    if not feasible:
        raise RuntimeError("threshold search produced no feasible candidate")
    maximum = max(evaluation.total_time_s for evaluation in feasible)
    ties = tuple(
        evaluation
        for evaluation in feasible
        if maximum - evaluation.total_time_s <= _ENDURANCE_TIE_TOLERANCE_S
    )
    best = min(
        ties,
        key=lambda evaluation: (
            evaluation.restart_count,
            -evaluation.minimum_transition_margin_s,
            -evaluation.fuel_remaining_kg,
            -evaluation.final_stored_energy_kwh,
            evaluation.soc_low,
            evaluation.soc_high,
        ),
    )
    return best, ties


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Completed bounded search and its exact-resolution tie set."""

    evaluations: tuple[ThresholdEvaluation, ...]
    best: ThresholdEvaluation
    tied_best: tuple[ThresholdEvaluation, ...]
    checkpoint_path: Path
    skipped_candidates: int


def _refinement_candidates(
    regions: Sequence[ThresholdEvaluation], config: ThresholdSearchConfig
) -> tuple[tuple[float, float], ...]:
    offsets = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-2, -1),
        (-2, 1),
        (2, -1),
        (2, 1),
    )
    candidates: list[tuple[float, float]] = []
    seen: set[str] = set()
    for low_offset, high_offset in offsets:
        for region in regions:
            pair = (
                round(region.soc_low + low_offset * config.refinement_step, 12),
                round(region.soc_high + high_offset * config.refinement_step, 12),
            )
            try:
                pair = config.validate_pair(*pair)
            except ValueError:
                continue
            key = _pair_key(*pair)
            if key not in seen:
                candidates.append(pair)
                seen.add(key)
    return tuple(candidates)


def run_threshold_search(
    checkpoint_path: str | Path,
    *,
    config: ThresholdSearchConfig | None = None,
    seed_evaluation: ThresholdEvaluation | None = None,
    evaluator: Callable[
        [float, float, str, ThresholdSearchConfig], ThresholdEvaluation
    ] = evaluate_threshold_candidate,
    progress: Callable[[str], None] | None = None,
) -> ThresholdSearchResult:
    """Run or resume the bounded coarse-to-fine search."""
    search_config = config or default_threshold_search_config()
    path = Path(checkpoint_path)
    evaluations = _load_checkpoint(path)
    by_key = {evaluation.candidate_id: evaluation for evaluation in evaluations}
    skipped = 0
    if seed_evaluation is not None and seed_evaluation.candidate_id not in by_key:
        search_config.validate_pair(seed_evaluation.soc_low, seed_evaluation.soc_high)
        _append_checkpoint(path, seed_evaluation)
        evaluations.append(seed_evaluation)
        by_key[seed_evaluation.candidate_id] = seed_evaluation

    def evaluate(pair: tuple[float, float], stage: str) -> None:
        nonlocal skipped
        key = _pair_key(*pair)
        if key in by_key:
            skipped += 1
            return
        if len(evaluations) >= search_config.maximum_evaluations:
            return
        row = evaluator(pair[0], pair[1], stage, search_config)
        if row.candidate_id != key:
            raise RuntimeError("evaluator returned a mismatched threshold pair")
        _append_checkpoint(path, row)
        evaluations.append(row)
        by_key[key] = row
        if progress is not None:
            progress(
                f"evaluation={len(evaluations)}/{search_config.maximum_evaluations} "
                f"stage={stage} low={pair[0]:.3f} high={pair[1]:.3f} "
                f"endurance_s={row.total_time_s:.6f} feasible={row.feasible}"
            )

    coarse = coarse_threshold_candidates(search_config)
    for pair in coarse:
        evaluate(pair, "coarse")
    coarse_rows = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.candidate_id in {_pair_key(*pair) for pair in coarse}
        and evaluation.feasible
    )
    retained = sorted(
        coarse_rows,
        key=lambda evaluation: (
            -evaluation.total_time_s,
            evaluation.restart_count,
            evaluation.soc_low,
            evaluation.soc_high,
        ),
    )[: search_config.retained_regions]
    for pair in _refinement_candidates(retained, search_config):
        evaluate(pair, "refine")
        if len(evaluations) >= search_config.maximum_evaluations:
            break
    best, ties = select_best_evaluation(evaluations)
    return ThresholdSearchResult(tuple(evaluations), best, ties, path, skipped)


@dataclass(frozen=True)
class OneStepExtensionResult:
    """Direct extra-loiter-step and reserve-completion result."""

    feasible: bool
    reason: str
    requested_extension_s: float
    reproduced_base_loiter_s: float
    extra_step_fuel_kg: float
    fuel_after_extra_step_kg: float
    soc_after_extra_step: float
    final_fuel_kg: float
    final_soc: float
    reserve_margin_kg: float
    termination_reason: str
    descent_landing_completed: bool


def _extension_mission(
    mission: MissionProfile,
    base_loiter_s: float,
    descent_landing_fuel_kg: float,
) -> MissionProfile:
    index = mission.endurance_phase_index
    loiter = mission.phases[index]
    base = replace(
        loiter,
        name="loiter_base",
        termination=Termination.DURATION,
        duration_s=base_loiter_s,
    )
    extension = replace(
        loiter,
        name="loiter_extension",
        termination=Termination.DURATION,
        duration_s=REFERENCE_TIMESTEP_S,
    )
    guard = replace(loiter, name="loiter_guard")
    phases = (
        *mission.phases[:index],
        base,
        extension,
        guard,
        *mission.phases[index + 1 :],
    )
    return replace(
        mission,
        phases=phases,
        descent_landing_fuel_kg=descent_landing_fuel_kg,
    )


def _run_extension_variant(
    parameters: ThermostatParameters, mission: MissionProfile
) -> MissionResult:
    aircraft = build_reference_aircraft()
    return run_mission(
        aircraft,
        mission,
        thermostat_parameters=parameters,
        initial_thermostat_state=REFERENCE_INITIAL_THERMOSTAT_STATE,
        dt_s=REFERENCE_TIMESTEP_S,
        initial_soc=1.0,
        record_log=True,
    )


def check_one_step_loiter_extension(
    best_run: ThermostatReferenceRun,
) -> OneStepExtensionResult:
    """Force exactly one extra 60 s loiter step, then validate reserves."""
    if best_run.result.log is None:
        raise ValueError("one-step extension requires a recorded best mission")
    base_loiter_s = best_run.result.phase_durations_s["loiter"]
    probe_mission = _extension_mission(best_run.mission, base_loiter_s, 0.0)
    probe = _run_extension_variant(best_run.parameters, probe_mission)
    if probe.log is None:
        raise RuntimeError("extension probe did not record a log")
    base_steps = tuple(step for step in probe.log if step.phase == "loiter_base")
    extension_steps = tuple(
        step for step in probe.log if step.phase == "loiter_extension"
    )
    original_loiter = tuple(
        step for step in best_run.result.log if step.phase == "loiter"
    )
    if not base_steps or not original_loiter:
        raise RuntimeError("extension probe did not reproduce the base loiter")
    reproduced = base_steps[-1]
    original = original_loiter[-1]
    if (
        abs(reproduced.fuel_remaining_kg - original.fuel_remaining_kg) > 1.0e-9
        or abs(reproduced.soc - original.soc) > 1.0e-10
    ):
        return OneStepExtensionResult(
            False,
            "base_loiter_reproduction_mismatch",
            REFERENCE_TIMESTEP_S,
            sum(step.dt_s for step in base_steps),
            0.0,
            reproduced.fuel_remaining_kg,
            reproduced.soc,
            probe.fuel_remaining_kg,
            probe.final_soc,
            probe.fuel_remaining_kg - probe_mission.fuel_reserve_kg,
            probe.termination_reason,
            False,
        )
    if len(extension_steps) != 1 or abs(extension_steps[0].dt_s - 60.0) > 1.0e-9:
        return OneStepExtensionResult(
            False,
            "extension_step_not_completed",
            REFERENCE_TIMESTEP_S,
            sum(step.dt_s for step in base_steps),
            0.0,
            reproduced.fuel_remaining_kg,
            reproduced.soc,
            probe.fuel_remaining_kg,
            probe.final_soc,
            probe.fuel_remaining_kg - probe_mission.fuel_reserve_kg,
            probe.termination_reason,
            False,
        )
    extended = extension_steps[0]
    allocation = extended.fuel_remaining_kg - best_run.mission.fuel_reserve_kg
    if allocation < 0.0:
        return OneStepExtensionResult(
            False,
            "extra_step_consumes_post_landing_reserve",
            REFERENCE_TIMESTEP_S,
            sum(step.dt_s for step in base_steps),
            reproduced.fuel_remaining_kg - extended.fuel_remaining_kg,
            extended.fuel_remaining_kg,
            extended.soc,
            probe.fuel_remaining_kg,
            probe.final_soc,
            allocation,
            probe.termination_reason,
            False,
        )
    validation_mission = _extension_mission(
        best_run.mission, base_loiter_s, allocation
    )
    validation = _run_extension_variant(best_run.parameters, validation_mission)
    validation_phases = (
        tuple(dict.fromkeys(step.phase for step in validation.log))
        if validation.log is not None
        else ()
    )
    descent_landing = (
        "descent" in validation_phases
        and "landing" in validation_phases
        and validation_phases[-1] == "landing"
    )
    guard_steps = (
        tuple(step for step in validation.log if step.phase == "loiter_guard")
        if validation.log is not None
        else ()
    )
    feasible = (
        validation.mission_complete
        and descent_landing
        and not guard_steps
        and validation.fuel_remaining_kg
        >= validation_mission.fuel_reserve_kg - 1.0e-10
        and "hard_dwell_infeasible" not in validation.failure_flags
        and "controller_infeasible" not in validation.failure_flags
        and "power_shortfall" not in validation.failure_flags
    )
    if feasible:
        reason = "extra_step_and_mandatory_phases_feasible"
    elif guard_steps:
        reason = "resource_guard_required_additional_loiter"
    else:
        reason = validation.termination_reason
    return OneStepExtensionResult(
        feasible=feasible,
        reason=reason,
        requested_extension_s=REFERENCE_TIMESTEP_S,
        reproduced_base_loiter_s=sum(step.dt_s for step in base_steps),
        extra_step_fuel_kg=reproduced.fuel_remaining_kg
        - extended.fuel_remaining_kg,
        fuel_after_extra_step_kg=extended.fuel_remaining_kg,
        soc_after_extra_step=extended.soc,
        final_fuel_kg=validation.fuel_remaining_kg,
        final_soc=validation.final_soc,
        reserve_margin_kg=(
            validation.fuel_remaining_kg - validation_mission.fuel_reserve_kg
        ),
        termination_reason=validation.termination_reason,
        descent_landing_completed=descent_landing,
    )


@dataclass(frozen=True)
class PhaseDependentGate:
    """Evidence gate before any loiter/non-loiter threshold extension."""

    justified: bool
    phase_specific_conflict: bool
    global_candidate_feasible: bool
    one_step_extension_feasible: bool
    rationale: str


def assess_phase_dependent_gate(
    *,
    global_candidate_feasible: bool,
    phase_specific_conflict: bool,
    one_step_extension_feasible: bool,
) -> PhaseDependentGate:
    """Require demonstrated phase conflict, not merely unused terminal energy."""
    justified = global_candidate_feasible and phase_specific_conflict
    if not global_candidate_feasible:
        rationale = "the global candidate is infeasible and must be fixed before adding phases"
    elif phase_specific_conflict:
        rationale = "a mandatory non-loiter phase demonstrably conflicts with the loiter band"
    elif one_step_extension_feasible:
        rationale = (
            "the extra step exposes reserve-allocation conservatism, not a threshold "
            "conflict between loiter and non-loiter phases"
        )
    else:
        rationale = "the feasible global band leaves no demonstrated phase-specific conflict"
    return PhaseDependentGate(
        justified,
        phase_specific_conflict,
        global_candidate_feasible,
        one_step_extension_feasible,
        rationale,
    )


def run_frozen_pi_mission() -> tuple[Aircraft, MissionProfile, MissionResult]:
    """Run the frozen reference aircraft with the existing PI controller."""
    aircraft = build_reference_aircraft()
    mission = ps1_mission()
    result = run_mission(aircraft, mission, PIECMS(), record_log=True)
    return aircraft, mission, result


def _ledger_by_phase(
    ledgers: Sequence[PhaseLedger], controller: str, phase: str
) -> PhaseLedger:
    return next(
        ledger
        for ledger in ledgers
        if ledger.controller == controller and ledger.phase == phase
    )


def _deterministic_signature(evaluation: ThresholdEvaluation) -> tuple[object, ...]:
    values = asdict(evaluation)
    values.pop("runtime_s")
    values.pop("stage")
    return tuple(values.items())


def run_threshold_study(output_dir: str | Path) -> dict[str, object]:
    """Run ledgers, bounded search, winner repeat, extension, and gate."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config = default_threshold_search_config()
    pi_aircraft, mission, pi_result = run_frozen_pi_mission()
    untuned_eval, untuned_run = simulate_threshold_candidate(
        *_REFERENCE_PAIR, stage="reference", config=config
    )
    pi_ledger = build_phase_ledger("frozen_pi_ecms", pi_aircraft, mission, pi_result)
    thermostat_ledger = build_phase_ledger(
        "untuned_thermostat", untuned_run.aircraft, mission, untuned_run.result
    )
    ledgers = (*pi_ledger, *thermostat_ledger)
    ledger_path = write_phase_ledger_csv(
        ledgers, directory / "thermostat_threshold_phase_ledger.csv"
    )
    pi_loiter = _ledger_by_phase(ledgers, "frozen_pi_ecms", "loiter")
    thermostat_loiter = _ledger_by_phase(
        ledgers, "untuned_thermostat", "loiter"
    )
    threshold = mission.loiter_fuel_floor_kg
    thresholds_agree = (
        abs(pi_loiter.end_fuel_kg - threshold) <= 1.0e-9
        and abs(thermostat_loiter.end_fuel_kg - threshold) <= 1.0e-9
    )
    if not thresholds_agree:
        raise RuntimeError("PI and thermostat loiter fuel thresholds are inconsistent")

    checkpoint = directory / "thermostat_threshold_search.csv"
    search = run_threshold_search(
        checkpoint,
        config=config,
        seed_evaluation=untuned_eval,
        progress=lambda message: print(message, flush=True),
    )
    repeated, best_run = simulate_threshold_candidate(
        search.best.soc_low,
        search.best.soc_high,
        stage="determinism_repeat",
        config=config,
    )
    deterministic = _deterministic_signature(search.best) == _deterministic_signature(
        repeated
    )
    if not deterministic:
        raise RuntimeError("winning thermostat mission was not deterministic")
    extension = check_one_step_loiter_extension(best_run)
    phase_conflict_candidates = tuple(
        evaluation
        for evaluation in search.evaluations
        if not evaluation.feasible
        and evaluation.loiter_time_s
        > search.best.loiter_time_s + _ENDURANCE_TIE_TOLERANCE_S
        and "descent_landing_incomplete" in evaluation.feasibility_reasons_json
    )
    phase_specific_conflict = bool(phase_conflict_candidates)
    gate = assess_phase_dependent_gate(
        global_candidate_feasible=search.best.feasible,
        phase_specific_conflict=phase_specific_conflict,
        one_step_extension_feasible=extension.feasible,
    )
    initial_energy = float(best_run.aircraft.battery.stored_energy_kwh(1.0))
    floor_energy = float(
        best_run.aircraft.battery.stored_energy_kwh(best_run.aircraft.battery.soc_min)
    )
    comparison = {
        "frozen_pi_ecms": {
            "total_time_s": pi_result.endurance_s,
            "loiter_time_s": pi_result.phase_durations_s["loiter"],
            "fuel_remaining_kg": pi_result.fuel_remaining_kg,
            "final_soc": pi_result.final_soc,
            "minimum_soc": pi_result.min_soc,
            "restart_count": sum(ledger.restart_count for ledger in pi_ledger),
            "termination_reason": pi_result.termination_reason,
        },
        "untuned_thermostat": asdict(untuned_eval),
        "best_global_thermostat": asdict(search.best),
    }
    report: dict[str, object] = {
        "classification": "best feasible pair within stated bounds and search resolution",
        "search": {
            "bounds": {
                "soc_low_minimum": config.soc_min,
                "soc_high_maximum": config.soc_max,
                "minimum_separation": config.minimum_separation,
            },
            "coarse_candidate_count": len(coarse_threshold_candidates(config)),
            "maximum_evaluations": config.maximum_evaluations,
            "completed_evaluations": len(search.evaluations),
            "refinement_step": config.refinement_step,
            "tied_best_candidate_ids": [
                evaluation.candidate_id for evaluation in search.tied_best
            ],
            "checkpoint_path": str(checkpoint),
        },
        "loiter_termination": {
            "active_threshold_kg": threshold,
            "pi_exit_fuel_kg": pi_loiter.end_fuel_kg,
            "untuned_exit_fuel_kg": thermostat_loiter.end_fuel_kg,
            "thresholds_agree": thresholds_agree,
            "pi_loiter_fuel_kg": pi_loiter.fuel_consumed_kg,
            "untuned_loiter_fuel_kg": thermostat_loiter.fuel_consumed_kg,
            "pi_loiter_mean_fuel_rate_kg_h": pi_loiter.mean_fuel_rate_kg_h,
            "untuned_loiter_mean_fuel_rate_kg_h": (
                thermostat_loiter.mean_fuel_rate_kg_h
            ),
            "pi_entry_soc": pi_loiter.start_soc,
            "pi_exit_soc": pi_loiter.end_soc,
            "untuned_entry_soc": thermostat_loiter.start_soc,
            "untuned_exit_soc": thermostat_loiter.end_soc,
            "pi_loiter_energy_change_kwh": pi_loiter.stored_energy_change_kwh,
            "untuned_loiter_energy_change_kwh": (
                thermostat_loiter.stored_energy_change_kwh
            ),
        },
        "comparison": comparison,
        "endurance_recovery": {
            "original_deficit_s": pi_result.endurance_s - untuned_eval.total_time_s,
            "recovered_s": search.best.total_time_s - untuned_eval.total_time_s,
            "remaining_deficit_s": pi_result.endurance_s - search.best.total_time_s,
            "recovered_fraction": (
                (search.best.total_time_s - untuned_eval.total_time_s)
                / (pi_result.endurance_s - untuned_eval.total_time_s)
            ),
        },
        "bound_status": {
            "soc_low_at_bound": abs(search.best.soc_low - config.soc_min) <= 1.0e-12,
            "soc_high_at_bound": abs(search.best.soc_high - config.soc_max)
            <= 1.0e-12,
            "minimum_separation_active": abs(
                search.best.soc_high
                - search.best.soc_low
                - config.minimum_separation
            )
            <= 1.0e-12,
        },
        "winner_repeat_deterministic": deterministic,
        "one_step_extension": asdict(extension),
        "phase_dependent_gate": {
            **asdict(gate),
            "conflict_candidate_ids": [
                evaluation.candidate_id
                for evaluation in phase_conflict_candidates
            ],
        },
        "terminal_energy": {
            "initial_stored_energy_kwh": initial_energy,
            "floor_stored_energy_kwh": floor_energy,
            "best_final_stored_energy_kwh": search.best.final_stored_energy_kwh,
            "best_usable_energy_slack_kwh": (
                search.best.final_stored_energy_kwh - floor_energy
            ),
        },
        "artifacts": {
            "phase_ledger_csv": str(ledger_path),
            "search_csv": str(checkpoint),
        },
    }
    report_path = directory / "thermostat_threshold_best.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> int:
    """Execute the bounded controller-only threshold study."""
    output_dir = Path(__file__).resolve().parents[2] / "deliverables" / "figures"
    report = run_threshold_study(output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
