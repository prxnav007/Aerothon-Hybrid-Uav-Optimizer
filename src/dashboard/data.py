"""Authoritative scenario, telemetry, validation, and GA artifact adapters."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.atmosphere import g0
from src.models.engine import LHV_KJ_KG
from src.optimization.chromosome import (
    NormalizedChromosome,
    PlantThermostatDesignSpace,
    decode_chromosome,
    practical_thermostat_seed,
)
from src.optimization.fitness import (
    FitnessResult,
    FitnessScenario,
    evaluate_fitness,
    evaluation_identity,
)
from src.optimization.ga_runner import production_context
from src.simulation.simulator import Aircraft, MissionResult, TimeStep, run_mission

__all__ = [
    "ARCHITECTURE_LABEL",
    "DashboardDataError",
    "FrozenDashboardScenario",
    "GAArtifacts",
    "PHASE_ORDER",
    "PRODUCTION_GA_DIRECTORY",
    "ValidationBundle",
    "build_telemetry_records",
    "ga_candidates_dataframe",
    "ga_history_dataframe",
    "ga_tradeoff_csv_bytes",
    "load_dashboard_scenarios",
    "load_ga_artifacts",
    "load_validation_summary",
    "run_validation_scenario",
    "telemetry_csv_bytes",
    "telemetry_dataframe",
    "validation_summary",
    "validation_summary_json_bytes",
    "write_validation_summary",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_GA_DIRECTORY = (
    REPOSITORY_ROOT / "deliverables" / "optimization" / "ga_production_seed_20260808"
)
VALIDATION_SUMMARY_PATH = (
    REPOSITORY_ROOT / "deliverables" / "validation" / "mission_15s_validation.json"
)
PHASE_ORDER = ("takeoff", "climb", "cruise", "loiter", "descent", "landing")
ARCHITECTURE_LABEL = (
    "Fuel -> Engine -> Generator -> Electrical bus -> Motor -> Propeller; "
    "the battery connects bidirectionally at the electrical bus."
)


class DashboardDataError(RuntimeError):
    """Actionable dashboard data or artifact failure."""


@dataclass(frozen=True)
class FrozenDashboardScenario:
    """One exact aircraft/controller configuration for 15-second validation."""

    key: str
    label: str
    claim: str
    chromosome: NormalizedChromosome
    bounds: PlantThermostatDesignSpace
    fitness_scenario: FitnessScenario
    source_artifact: str

    @property
    def evaluation_key(self) -> str:
        return evaluation_identity(
            self.chromosome,
            bounds=self.bounds,
            scenario=self.fitness_scenario,
        )

    @property
    def decoded_design(self) -> dict[str, float]:
        return decode_chromosome(self.chromosome, bounds=self.bounds).to_dict()


@dataclass(frozen=True)
class ValidationBundle:
    """One completed simulation and the exact records used by the dashboard."""

    scenario: FrozenDashboardScenario
    fitness: FitnessResult
    mission_result: MissionResult
    telemetry_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GAArtifacts:
    """Parsed completed-GA products; loading never executes the optimizer."""

    directory: Path
    best_found: Mapping[str, Any]
    history_rows: tuple[dict[str, Any], ...]
    candidate_rows: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DashboardDataError(f"Required artifact is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardDataError(f"Artifact is unreadable or malformed: {path}") from error
    if not isinstance(value, dict):
        raise DashboardDataError(f"Artifact must contain a JSON object: {path}")
    return value


def _frozen_15s_fitness_scenario() -> FitnessScenario:
    nominal = FitnessScenario.nominal()
    return replace(
        nominal,
        name="practical_3km_thermostat_restart_0_1_validation_15s",
        classification="15-second validation of the uncalibrated practical restart-fuel sensitivity",
        timestep_s=15.0,
    )


def load_dashboard_scenarios(
    best_found_path: str | Path | None = None,
) -> dict[str, FrozenDashboardScenario]:
    """Load the exact practical and production-GA designs without simulation."""
    artifact = (
        PRODUCTION_GA_DIRECTORY / "best_found.json"
        if best_found_path is None
        else Path(best_found_path)
    )
    best = _read_json(artifact)
    _, bounds, _ = production_context()
    scenario = _frozen_15s_fitness_scenario()
    practical = practical_thermostat_seed(bounds=bounds)
    try:
        ga_chromosome = NormalizedChromosome.from_dict(
            best["normalized_chromosome"], bounds=bounds
        )
        aircraft = best["aircraft"]
    except (KeyError, TypeError, ValueError) as error:
        raise DashboardDataError(
            f"best_found.json does not match the production chromosome schema: {artifact}"
        ) from error
    decoded = decode_chromosome(ga_chromosome, bounds=bounds)
    checks = {
        "wing_area_m2": decoded.wing_area_m2,
        "aspect_ratio": decoded.aspect_ratio,
        "engine_rating_kw": decoded.engine_rating_kw,
        "battery_capacity_kwh": decoded.battery_capacity_kwh,
        "soc_low": decoded.soc_low,
        "soc_high": decoded.soc_high,
    }
    for name, expected in checks.items():
        if name not in aircraft or not math.isclose(
            float(aircraft[name]), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise DashboardDataError(
                f"best_found.json aircraft field {name!r} does not match its chromosome"
            )
    return {
        "practical_reference": FrozenDashboardScenario(
            "practical_reference",
            "Practical reference",
            "Authoritative practical reference configuration.",
            practical,
            bounds,
            scenario,
            "src.optimization.chromosome.practical_thermostat_seed",
        ),
        "ga_selected": FrozenDashboardScenario(
            "ga_selected",
            "GA-selected design",
            "Best feasible design found by one completed GA seed.",
            ga_chromosome,
            bounds,
            scenario,
            str(artifact.resolve()),
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1.0e-15:
        return None
    return numerator / denominator


def build_telemetry_records(
    result: MissionResult,
    aircraft: Aircraft,
    scenario: FrozenDashboardScenario,
) -> tuple[dict[str, Any], ...]:
    if result.log is None:
        raise DashboardDataError("Recorded mission returned no telemetry log")
    if aircraft.powertrain.load_dependent:
        raise DashboardDataError(
            "Dashboard component split telemetry supports the active constant-efficiency branch only"
        )
    running_fuel = 0.0
    restart_fuel = 0.0
    useful_energy = 0.0
    battery_throughput = 0.0
    phase_elapsed = {name: 0.0 for name in PHASE_ORDER}
    records: list[dict[str, Any]] = []
    parameters = scenario.decoded_design
    for step in result.log:
        if step.phase not in phase_elapsed:
            raise DashboardDataError(f"Unexpected mission phase in telemetry: {step.phase}")
        phase_elapsed[step.phase] += step.dt_s
        running_fuel += step.fuel_flow_kg_s * step.dt_s
        restart_fuel += step.restart_fuel_kg
        useful_energy += step.thrust_power_kw * step.dt_s / 3600.0
        battery_throughput += abs(step.battery_bus_kw) * step.dt_s / 3600.0
        generator_output = step.engine_shaft_kw * aircraft.powertrain.eta_generator
        motor_input = step.shaft_power_kw / aircraft.powertrain.eta_motor
        inverter_input = motor_input / aircraft.powertrain.eta_inverter
        fuel_chemical_kw = (
            step.fuel_flow_kg_s + step.restart_fuel_kg / step.dt_s
        ) * LHV_KJ_KG
        battery_efficiency = None
        if step.battery_bus_kw > 0.0 and step.battery_internal_kw > 0.0:
            battery_efficiency = step.battery_bus_kw / step.battery_internal_kw
        elif step.battery_bus_kw < 0.0 and step.battery_internal_kw < 0.0:
            battery_efficiency = abs(step.battery_internal_kw / step.battery_bus_kw)
        record = asdict(step)
        record.update(
            {
                "time_start_s": step.time_s - step.dt_s,
                "mission_time_h": step.time_s / 3600.0,
                "phase_elapsed_s": phase_elapsed[step.phase],
                "phase_index": PHASE_ORDER.index(step.phase),
                "mass_kg": step.weight_n / g0,
                "engine_on": not step.engine_shut_down,
                "restart_event": bool(
                    step.thermostat_transitioned and step.requested_engine_on
                ),
                "generator_electrical_output_kw": generator_output,
                "rectifier_bus_output_kw": step.bus_from_engine_kw,
                "inverter_electrical_input_kw": inverter_input,
                "motor_electrical_input_kw": motor_input,
                "motor_shaft_output_kw": step.shaft_power_kw,
                "engine_source_efficiency": _safe_ratio(
                    step.bus_from_engine_kw, step.engine_shaft_kw
                ),
                "demand_path_efficiency": _safe_ratio(
                    step.shaft_power_kw, step.bus_demand_kw
                ),
                "battery_terminal_efficiency": battery_efficiency,
                "fuel_chemical_power_kw": fuel_chemical_kw,
                "fuel_flow_kg_h": step.fuel_flow_kg_s * 3600.0,
                "cumulative_running_fuel_kg": running_fuel,
                "cumulative_restart_fuel_kg": restart_fuel,
                "cumulative_useful_propulsion_energy_kwh": useful_energy,
                "cumulative_battery_bus_throughput_kwh": battery_throughput,
                "bus_power_residual_kw": (
                    step.bus_from_engine_kw
                    + step.battery_bus_kw
                    - step.bus_demand_kw
                ),
                "thermostat_soc_low": parameters["soc_low"],
                "thermostat_soc_high": parameters["soc_high"],
                "battery_soc_floor": aircraft.battery.soc_min,
            }
        )
        records.append(record)
    return tuple(records)


def run_validation_scenario(scenario: FrozenDashboardScenario) -> ValidationBundle:
    """Run exactly one authoritative recorded mission for a frozen scenario."""
    captured: list[tuple[Aircraft, MissionResult]] = []

    def capture(aircraft: Aircraft, mission: Any, **kwargs: Any) -> MissionResult:
        result = run_mission(aircraft, mission, **kwargs)
        captured.append((aircraft, result))
        return result

    fitness = evaluate_fitness(
        scenario.chromosome,
        bounds=scenario.bounds,
        scenario=scenario.fitness_scenario,
        mission_runner=capture,
    )
    if len(captured) != 1:
        raise DashboardDataError(
            f"Expected one mission call for {scenario.key}, observed {len(captured)}"
        )
    aircraft, result = captured[0]
    records = build_telemetry_records(result, aircraft, scenario)
    return ValidationBundle(scenario, fitness, result, records)


def telemetry_dataframe(bundle: ValidationBundle) -> pd.DataFrame:
    return pd.DataFrame.from_records(bundle.telemetry_records)


def telemetry_csv_bytes(bundle: ValidationBundle) -> bytes:
    return telemetry_dataframe(bundle).to_csv(index=False).encode("utf-8")


def _constraint_payload(record: Any) -> dict[str, Any]:
    value = asdict(record)
    if "margin" not in value:
        if record.relationship == "at_least":
            value["signed_margin"] = record.quantity - record.required_or_allowed
        else:
            value["signed_margin"] = record.required_or_allowed - record.quantity
    return value


def validation_summary(bundle: ValidationBundle) -> dict[str, Any]:
    """Compact serializable summary; telemetry remains in memory only."""
    fitness = bundle.fitness
    result = bundle.mission_result
    resources = fitness.resources
    behavior = fitness.controller_behavior
    validity = fitness.validity
    if resources is None or behavior is None or validity is None:
        raise DashboardDataError("Validation mission lacks fitness audit data")
    return {
        "scenario_key": bundle.scenario.key,
        "scenario_label": bundle.scenario.label,
        "claim": bundle.scenario.claim,
        "evaluation_key": bundle.scenario.evaluation_key,
        "source_artifact": bundle.scenario.source_artifact,
        "architecture": ARCHITECTURE_LABEL,
        "input_configuration": {
            **bundle.scenario.decoded_design,
            "timestep_s": bundle.scenario.fitness_scenario.timestep_s,
            "restart_fuel_kg_per_start": (
                bundle.scenario.fitness_scenario.restart_fuel_kg
            ),
            "minimum_on_time_s": (
                bundle.scenario.fitness_scenario.minimum_on_time_s
            ),
            "minimum_off_time_s": (
                bundle.scenario.fitness_scenario.minimum_off_time_s
            ),
            "cruise_altitude_m": (
                bundle.scenario.fitness_scenario.static_scenario.cruise_altitude_m
            ),
            "mtow_kg": bundle.scenario.fitness_scenario.static_scenario.mtow_kg,
            "payload_kg": bundle.scenario.fitness_scenario.static_scenario.payload_kg,
        },
        "phase_durations_s": dict(result.phase_durations_s),
        "total_mission_seconds": result.endurance_s,
        "loiter_seconds": result.phase_durations_s.get("loiter", 0.0),
        "initial_fuel_kg": resources.initial_fuel_kg,
        "final_fuel_kg": resources.final_fuel_kg,
        "running_fuel_consumed_kg": resources.running_fuel_consumed_kg,
        "restart_fuel_consumed_kg": resources.restart_fuel_consumed_kg,
        "final_soc": resources.final_soc,
        "minimum_soc": resources.minimum_soc,
        "restart_count": behavior.restart_count,
        "termination_reason": validity.termination_reason,
        "mission_complete": result.mission_complete,
        "dynamically_feasible": fitness.dynamically_feasible,
        "failure_flags": list(result.failure_flags),
        "static_constraints": [
            _constraint_payload(record) for record in fitness.static_constraints
        ],
        "dynamic_constraints": [
            _constraint_payload(record) for record in fitness.dynamic_constraints
        ],
        "maximum_bus_power_residual_kw": validity.maximum_bus_power_residual_kw,
        "fuel_ledger_residual_kg": validity.fuel_ledger_residual_kg,
        "battery_energy_ledger_residual_kwh": (
            validity.battery_energy_ledger_residual_kwh
        ),
        "discrete_energy_ledger_residual_fraction": (
            validity.discrete_energy_ledger_residual_fraction
        ),
        "telemetry_sample_count": len(bundle.telemetry_records),
    }


def validation_summary_json_bytes(bundle: ValidationBundle) -> bytes:
    return (
        json.dumps(validation_summary(bundle), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_validation_summary(
    bundles: Sequence[ValidationBundle],
    path: str | Path = VALIDATION_SUMMARY_PATH,
) -> Path:
    by_key = {bundle.scenario.key: validation_summary(bundle) for bundle in bundles}
    if tuple(by_key) != ("practical_reference", "ga_selected"):
        raise DashboardDataError(
            "Validation output requires practical_reference followed by ga_selected"
        )
    reference = by_key["practical_reference"]["loiter_seconds"]
    selected = by_key["ga_selected"]["loiter_seconds"]
    payload = {
        "schema_version": 1,
        "description": "Exactly two deterministic 15-second mission validations.",
        "optimization_result_label": "Optimization evaluation - 60-second timestep.",
        "validation_result_label": "Validation evaluation - 15-second timestep.",
        "scenarios": by_key,
        "validated_loiter_improvement_percent": (
            (selected - reference) / reference * 100.0
        ),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_validation_summary(
    path: str | Path = VALIDATION_SUMMARY_PATH,
) -> dict[str, Any]:
    return _read_json(Path(path))


def _csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))
    except FileNotFoundError as error:
        raise DashboardDataError(f"Required artifact is missing: {path}") from error
    except (OSError, csv.Error) as error:
        raise DashboardDataError(f"CSV artifact is unreadable: {path}") from error


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DashboardDataError(f"Expected numeric GA artifact value, got {value!r}") from error
    if not math.isfinite(number):
        raise DashboardDataError(f"GA artifact contains nonfinite value {value!r}")
    return number


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise DashboardDataError(f"Expected boolean GA artifact value, got {value!r}")


def _ledger_details(path: Path) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return details
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                fitness = record["fitness_result"]
                resolved = fitness["resolved_design"]
                resources = fitness.get("resources")
                behavior = fitness.get("controller_behavior")
                details[str(record["evaluation_key"])] = {
                    "dry_mass_kg": resolved["masses"]["dry_kg"],
                    "initial_fuel_kg": resolved["fuel"]["initial_usable_fuel_kg"],
                    "restart_count": (
                        None if behavior is None else behavior["restart_count"]
                    ),
                    "final_fuel_kg": (
                        None if resources is None else resources["final_fuel_kg"]
                    ),
                    "minimum_soc": (
                        None if resources is None else resources["minimum_soc"]
                    ),
                    "static_normalized_violation": fitness[
                        "static_normalized_total_violation"
                    ],
                    "dynamic_normalized_violation": fitness[
                        "total_normalized_dynamic_violation"
                    ],
                }
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DashboardDataError(
            f"GA evaluation ledger is malformed near line {line_number}: {path}"
        ) from error
    return details


def load_ga_artifacts(
    directory: str | Path = PRODUCTION_GA_DIRECTORY,
) -> GAArtifacts:
    """Load the completed GA reports and optional detailed evaluation ledger."""
    root = Path(directory)
    best = _read_json(root / "best_found.json")
    history_raw = _csv_rows(root / "generation_history.csv")
    candidates_raw = _csv_rows(root / "evaluated_candidates.csv")
    if not history_raw or not candidates_raw:
        raise DashboardDataError("GA history and evaluated-candidate artifacts must not be empty")
    details = _ledger_details(root / "evaluation_ledger.jsonl")
    warnings: list[str] = []
    if not details:
        warnings.append(
            "Detailed GA ledger is unavailable; dry mass, fuel, restart and resource hover fields are omitted."
        )
    history: list[dict[str, Any]] = []
    for row in history_raw:
        history.append(
            {
                **row,
                "generation": int(row["generation"]),
                "feasible_count": int(row["feasible_count"]),
                "static_infeasible_count": int(row["static_infeasible_count"]),
                "dynamic_infeasible_count": int(row["dynamic_infeasible_count"]),
                "best_feasible_objective": _number(row["best_feasible_objective"]),
            }
        )
    numeric_candidate_fields = (
        "wing_area_m2", "aspect_ratio", "engine_rating_kw",
        "battery_capacity_kwh", "soc_low", "soc_high",
        "objective_loiter_seconds", "total_mission_seconds",
        "combined_normalized_violation",
    )
    candidates: list[dict[str, Any]] = []
    for row in candidates_raw:
        converted: dict[str, Any] = dict(row)
        for name in numeric_candidate_fields:
            converted[name] = _number(row[name])
        converted["generation"] = int(row["generation"])
        converted["static_feasible"] = _boolean(row["static_feasible"])
        converted["dynamically_feasible"] = _boolean(row["dynamically_feasible"])
        converted["run_mission_called"] = _boolean(row["run_mission_called"])
        converted.update(details.get(row["evaluation_key"], {}))
        converted["feasibility_status"] = (
            "Feasible"
            if converted["dynamically_feasible"]
            else "Dynamic infeasible"
            if converted["static_feasible"]
            else "Static infeasible"
        )
        objective = converted["objective_loiter_seconds"]
        converted["objective_loiter_hours"] = (
            None if objective is None else objective / 3600.0
        )
        candidates.append(converted)
    return GAArtifacts(root, best, tuple(history), tuple(candidates), tuple(warnings))


def ga_history_dataframe(artifacts: GAArtifacts) -> pd.DataFrame:
    return pd.DataFrame.from_records(artifacts.history_rows)


def ga_candidates_dataframe(artifacts: GAArtifacts) -> pd.DataFrame:
    return pd.DataFrame.from_records(artifacts.candidate_rows)


def ga_tradeoff_csv_bytes(artifacts: GAArtifacts) -> bytes:
    return ga_candidates_dataframe(artifacts).to_csv(index=False).encode("utf-8")
