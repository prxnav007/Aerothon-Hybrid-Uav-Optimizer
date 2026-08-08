"""Frozen-aircraft full-mission controller comparison and figure generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Sequence

from src.analysis.thermostat_mission import (
    REFERENCE_INITIAL_THERMOSTAT_STATE,
    REFERENCE_THERMOSTAT_PARAMETERS,
    REFERENCE_TIMESTEP_S,
    build_reference_aircraft,
)
from src.control.base import S_ABSOLUTE_MAX, S_ABSOLUTE_MIN
from src.control.fixed_ecms import FixedECMS
from src.control.pi_ecms import PIECMS
from src.control.thermostat import ThermostatParameters, ThermostatState
from src.simulation.mission import MissionProfile, ps1_mission
from src.simulation.simulator import (
    Aircraft,
    MissionResult,
    TimeStep,
    mission_energy_balance,
    run_mission,
)

__all__ = [
    "ControllerSpec",
    "MissionRecord",
    "OPTIMISED_THERMOSTAT",
    "UNTUNED_THERMOSTAT",
    "controller_local_candidates",
    "generate_controller_figures",
    "pareto_controller_keys",
    "run_controller_mission",
    "select_verified_specs",
]

ControllerFamily = Literal["fixed", "pi", "thermostat"]
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[2] / "deliverables" / "figures"
REPORT_PATH = Path(__file__).resolve().parents[2] / "docs" / "controller_comparison.md"
RESTART_COSTS_KG = (0.0, 0.1, 0.5)
FIXED_LOCAL_RATIOS = (1.0, 1.1, 1.2)
PI_LOCAL_RATIOS = (1.2, 1.3, 1.4)
PI_LOCAL_GAINS = (0.0, 2.5, 5.0)
RESTART_RETUNE_THERMOSTAT_BANDS = (
    (0.200, 0.275),
    (0.200, 0.300),
    (0.225, 0.300),
    (0.225, 0.325),
    (0.225, 0.350),
    (0.250, 0.350),
)
_LEDGER_TOLERANCE = 1.0e-10
_POWER_TOLERANCE_KW = 1.0e-8
# see assumptions.md C-10
_PRACTICAL_ENDURANCE_RETENTION_MIN = 0.99

CONTROLLER_COLOURS = {
    "fixed_s_ecms": "#0072B2",
    "adaptive_pi_ecms": "#E69F00",
    "optimised_thermostat": "#009E73",
    "untuned_thermostat": "#86BFA8",
}
CONTROLLER_MARKERS = {
    "fixed_s_ecms": "s",
    "adaptive_pi_ecms": "o",
    "optimised_thermostat": "D",
    "untuned_thermostat": "^",
}


@dataclass(frozen=True)
class ControllerSpec:
    """Exact executable configuration for one compared controller."""

    key: str
    display_name: str
    family: ControllerFamily
    tuned: bool
    s_ratio: float | None = None
    kp: float | None = None
    soc_ref: float | None = None
    soc_low: float | None = None
    soc_high: float | None = None

    def __post_init__(self) -> None:
        if self.family == "fixed":
            if self.s_ratio is None or any(
                value is not None
                for value in (self.kp, self.soc_ref, self.soc_low, self.soc_high)
            ):
                raise ValueError("fixed controller requires only s_ratio")
        elif self.family == "pi":
            if any(value is None for value in (self.s_ratio, self.kp, self.soc_ref)):
                raise ValueError("PI controller requires s_ratio, kp, and soc_ref")
            if self.soc_low is not None or self.soc_high is not None:
                raise ValueError("PI controller does not accept thermostat thresholds")
        elif self.family == "thermostat":
            if self.soc_low is None or self.soc_high is None:
                raise ValueError("thermostat requires both SoC thresholds")
            if any(value is not None for value in (self.s_ratio, self.kp, self.soc_ref)):
                raise ValueError("thermostat does not accept ECMS parameters")
        else:
            raise ValueError(f"unsupported controller family {self.family!r}")

    @property
    def candidate_id(self) -> str:
        if self.family == "fixed":
            return f"fixed:r={self.s_ratio:.3f}"
        if self.family == "pi":
            return f"pi:r={self.s_ratio:.3f}:kp={self.kp:.3f}"
        return f"thermostat:low={self.soc_low:.3f}:high={self.soc_high:.3f}"

    def configuration(self, restart_cost_kg: float) -> dict[str, Any]:
        """Return a complete, presentation-auditable configuration record."""
        shared = {
            "controller_family": self.family,
            "engine_allow_shutdown": True,
            "engine_min_power_fraction": 0.15,
            "restart_fuel_kg_per_start": restart_cost_kg,
            "mission_timestep_s": REFERENCE_TIMESTEP_S,
        }
        if self.family == "fixed":
            return {
                **shared,
                "class": "FixedECMS",
                "s": None,
                "s_ratio": self.s_ratio,
                "ratio_anchor": "switching_s",
                "adaptation_enabled": False,
                "equivalence_factor_clamp": [S_ABSOLUTE_MIN, S_ABSOLUTE_MAX],
                "minimum_on_time_s": None,
                "minimum_off_time_s": None,
            }
        if self.family == "pi":
            return {
                **shared,
                "class": "PIECMS",
                "base_equivalence_factor": None,
                "s0_ratio": self.s_ratio,
                "base_anchor": "switching_s",
                "neutral_s_role": "diagnostic_only",
                "kp": self.kp,
                "soc_ref": self.soc_ref,
                "integral_action": False,
                "equivalence_factor_clamp": [S_ABSOLUTE_MIN, S_ABSOLUTE_MAX],
                "minimum_on_time_s": None,
                "minimum_off_time_s": None,
            }
        return {
            **shared,
            "class": "ThermostatParameters",
            "soc_low": self.soc_low,
            "soc_high": self.soc_high,
            "terminal_strategy": "causal",
            "minimum_on_time_s": 60.0,
            "minimum_off_time_s": 60.0,
            "dwell_semantics": "hard",
            "engine_on_power_kw": None,
            "engine_on_power_rule": "existing maximum-feasible selection",
            "initial_engine_on": True,
            "initial_elapsed_in_state_s": 60.0,
        }


OPTIMISED_THERMOSTAT = ControllerSpec(
    "optimised_thermostat",
    "Optimised thermostat",
    "thermostat",
    True,
    soc_low=0.225,
    soc_high=0.300,
)
UNTUNED_THERMOSTAT = ControllerSpec(
    "untuned_thermostat",
    "Untuned thermostat (0.4, 0.6)",
    "thermostat",
    False,
    soc_low=0.4,
    soc_high=0.6,
)


def controller_local_candidates() -> tuple[ControllerSpec, ...]:
    """Return the bounded reconstruction of the missing historical sweeps."""
    fixed = tuple(
        ControllerSpec(
            "fixed_s_ecms",
            "Tuned fixed-(s) ECMS",
            "fixed",
            True,
            s_ratio=ratio,
        )
        for ratio in FIXED_LOCAL_RATIOS
    )
    adaptive = tuple(
        ControllerSpec(
            "adaptive_pi_ecms",
            "Tuned adaptive PI-ECMS",
            "pi",
            True,
            s_ratio=ratio,
            kp=kp,
            soc_ref=0.6,
        )
        for ratio in PI_LOCAL_RATIOS
        for kp in PI_LOCAL_GAINS
    )
    return (*fixed, *adaptive)


@dataclass(frozen=True)
class MissionRecord:
    """One checkpointed full-mission result with flat CSV-safe fields."""

    run_id: str
    study_stage: str
    controller_key: str
    display_name: str
    controller_family: str
    tuned: bool
    candidate_id: str
    configuration_json: str
    restart_cost_per_start_kg: float
    total_time_s: float
    loiter_time_s: float
    takeoff_time_s: float
    climb_time_s: float
    cruise_time_s: float
    descent_time_s: float
    landing_time_s: float
    phase_durations_json: str
    initial_fuel_kg: float
    running_fuel_kg: float
    restart_fuel_kg: float
    total_fuel_kg: float
    final_fuel_kg: float
    final_soc: float
    minimum_soc: float
    final_stored_energy_kwh: float
    restart_count: int
    loiter_restarts_per_hour: float
    overall_engine_off_fraction: float
    loiter_engine_off_fraction: float
    on_duration_distribution_json: str
    off_duration_distribution_json: str
    mean_engine_on_power_kw: float
    minimum_engine_on_power_kw: float
    maximum_engine_on_power_kw: float
    battery_limit_encounters_json: str
    mission_complete: bool
    feasible: bool
    feasibility_reasons_json: str
    termination_reason: str
    reserve_shortfall_kg: float
    descent_landing_completed: bool
    violations_json: str
    maximum_power_residual_kw: float
    fuel_ledger_residual_kg: float
    energy_ledger_residual_kwh: float
    discrete_energy_residual_fraction: float
    battery_integration_residual_kwh: float
    simulation_runtime_s: float

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "MissionRecord":
        strings = {
            "run_id",
            "study_stage",
            "controller_key",
            "display_name",
            "controller_family",
            "candidate_id",
            "configuration_json",
            "phase_durations_json",
            "on_duration_distribution_json",
            "off_duration_distribution_json",
            "battery_limit_encounters_json",
            "feasibility_reasons_json",
            "termination_reason",
            "violations_json",
        }
        booleans = {"tuned", "mission_complete", "feasible", "descent_landing_completed"}
        integers = {"restart_count"}
        values: dict[str, object] = {}
        for field in fields(cls):
            value = row[field.name]
            if field.name in strings:
                values[field.name] = value
            elif field.name in booleans:
                values[field.name] = value.lower() == "true"
            elif field.name in integers:
                values[field.name] = int(value)
            else:
                values[field.name] = float(value)
        return cls(**values)  # type: ignore[arg-type]


def _duration_distribution(steps: Sequence[TimeStep], engine_on: bool) -> dict[str, Any]:
    durations: list[float] = []
    state: bool | None = None
    elapsed_s = 0.0
    for step in steps:
        current = not step.engine_shut_down
        if state is None:
            state = current
        if current != state:
            if state is engine_on:
                durations.append(elapsed_s)
            state = current
            elapsed_s = 0.0
        elapsed_s += step.dt_s
    if state is engine_on and elapsed_s > 0.0:
        durations.append(elapsed_s)
    return {
        "count": len(durations),
        "minimum_s": min(durations, default=0.0),
        "mean_s": fmean(durations) if durations else 0.0,
        "maximum_s": max(durations, default=0.0),
        "samples_s": durations,
    }


def _restart_counts(steps: Sequence[TimeStep]) -> tuple[int, int]:
    previous_off = False
    total = 0
    loiter = 0
    for step in steps:
        restarted = previous_off and not step.engine_shut_down
        total += int(restarted)
        loiter += int(restarted and step.phase == "loiter")
        previous_off = step.engine_shut_down
    return total, loiter


def _battery_limit_encounters(steps: Sequence[TimeStep]) -> dict[str, Any]:
    encounters: dict[str, dict[str, float]] = {}
    for step in steps:
        if step.battery_active_limit == "none" or abs(step.battery_bus_kw) <= 1.0e-10:
            continue
        direction = "discharge" if step.battery_bus_kw > 0.0 else "charge"
        key = f"{direction}:{step.battery_active_limit}"
        entry = encounters.setdefault(key, {"steps": 0, "duration_s": 0.0})
        entry["steps"] += 1
        entry["duration_s"] += step.dt_s
    return encounters


def _build_inputs(
    spec: ControllerSpec, restart_cost_kg: float
) -> tuple[Aircraft, MissionProfile, dict[str, Any]]:
    if restart_cost_kg < 0.0 or not math.isfinite(restart_cost_kg):
        raise ValueError("restart cost must be finite and non-negative")
    reference = build_reference_aircraft()
    aircraft = replace(
        reference,
        engine=replace(reference.engine, restart_fuel_kg=restart_cost_kg),
    )
    mission = ps1_mission()
    if spec.family == "fixed":
        return aircraft, mission, {"controller": FixedECMS(s_ratio=spec.s_ratio)}
    if spec.family == "pi":
        return aircraft, mission, {
            "controller": PIECMS(
                s0_ratio=spec.s_ratio,
                kp=spec.kp,
                soc_ref=spec.soc_ref,
            )
        }
    parameters = replace(
        REFERENCE_THERMOSTAT_PARAMETERS,
        soc_low=spec.soc_low,
        soc_high=spec.soc_high,
        restart_fuel_kg=restart_cost_kg,
    )
    initial_state = ThermostatState(
        engine_on=REFERENCE_INITIAL_THERMOSTAT_STATE.engine_on,
        elapsed_in_state_s=REFERENCE_INITIAL_THERMOSTAT_STATE.elapsed_in_state_s,
        restart_count=0,
        terminal_depletion=False,
    )
    return aircraft, mission, {
        "thermostat_parameters": parameters,
        "initial_thermostat_state": initial_state,
    }


def _record_from_result(
    spec: ControllerSpec,
    restart_cost_kg: float,
    stage: str,
    aircraft: Aircraft,
    mission: MissionProfile,
    result: MissionResult,
    runtime_s: float,
) -> MissionRecord:
    if result.log is None:
        raise ValueError("controller comparison requires a recorded mission log")
    steps = result.log
    loiter = tuple(step for step in steps if step.phase == "loiter")
    duration_s = sum(step.dt_s for step in steps)
    loiter_s = sum(step.dt_s for step in loiter)
    running_fuel = sum(step.fuel_flow_kg_s * step.dt_s for step in steps)
    restart_fuel = sum(step.restart_fuel_kg for step in steps)
    restart_count, loiter_restart_count = _restart_counts(steps)
    on_steps = tuple(step for step in steps if not step.engine_shut_down)
    on_time_s = sum(step.dt_s for step in on_steps)
    mean_on_power = (
        sum(step.engine_shaft_kw * step.dt_s for step in on_steps) / on_time_s
        if on_time_s
        else 0.0
    )
    phases_seen = tuple(dict.fromkeys(step.phase for step in steps))
    descent_landing = (
        result.mission_complete
        and "descent" in phases_seen
        and "landing" in phases_seen
        and bool(steps)
        and steps[-1].phase == "landing"
    )
    balance = mission_energy_balance(result)
    bus_residual = max(
        (
            abs(step.bus_from_engine_kw + step.battery_bus_kw - step.bus_demand_kw)
            for step in steps
        ),
        default=0.0,
    )
    clamp_steps = sum(
        step.equivalence_factor in (S_ABSOLUTE_MIN, S_ABSOLUTE_MAX)
        for step in steps
        if math.isfinite(step.equivalence_factor)
    )
    violations = {
        "controller_infeasible": "controller_infeasible" in result.failure_flags,
        "hard_dwell_infeasible": "hard_dwell_infeasible" in result.failure_flags,
        "hard_dwell_violation_count": sum(
            int(step.thermostat_dwell_violation) for step in steps
        ),
        "plant_infeasible_step_count": sum(int(not step.plant_feasible) for step in steps),
        "equivalence_factor_clamp_step_count": clamp_steps,
        "failure_flags": result.failure_flags,
    }
    terminal_flags = {
        "power_shortfall",
        "fuel_exhausted",
        "fuel_reserve_shortfall",
        "altitude_unreachable",
        "controller_infeasible",
        "hard_dwell_infeasible",
    }
    reasons: list[str] = []
    if not result.mission_complete:
        reasons.append("mission_incomplete")
    if phases_seen != mission.phase_names:
        reasons.append("six_phases_not_completed")
    if not descent_landing:
        reasons.append("descent_landing_incomplete")
    if result.fuel_remaining_kg < mission.fuel_reserve_kg - _LEDGER_TOLERANCE:
        reasons.append("reserve_shortfall")
    if result.min_soc < aircraft.battery.soc_min - _LEDGER_TOLERANCE:
        reasons.append("soc_below_floor")
    if terminal_flags.intersection(result.failure_flags):
        reasons.append("terminal_constraint_failure")
    fuel_residual = result.fuel_used_kg - running_fuel - restart_fuel
    if bus_residual > _POWER_TOLERANCE_KW:
        reasons.append("power_ledger_open")
    if abs(fuel_residual) > _LEDGER_TOLERANCE:
        reasons.append("fuel_ledger_open")
    if balance.discrete_residual_fraction > 1.0e-12:
        reasons.append("energy_ledger_open")
    phase = result.phase_durations_s
    run_id = f"{stage}:{spec.candidate_id}:restart={restart_cost_kg:.3f}"
    return MissionRecord(
        run_id=run_id,
        study_stage=stage,
        controller_key=spec.key,
        display_name=spec.display_name,
        controller_family=spec.family,
        tuned=spec.tuned,
        candidate_id=spec.candidate_id,
        configuration_json=json.dumps(
            spec.configuration(restart_cost_kg), sort_keys=True, separators=(",", ":")
        ),
        restart_cost_per_start_kg=restart_cost_kg,
        total_time_s=result.endurance_s,
        loiter_time_s=loiter_s,
        takeoff_time_s=phase["takeoff"],
        climb_time_s=phase["climb"],
        cruise_time_s=phase["cruise"],
        descent_time_s=phase["descent"],
        landing_time_s=phase["landing"],
        phase_durations_json=json.dumps(phase, sort_keys=True, separators=(",", ":")),
        initial_fuel_kg=aircraft.masses.fuel_kg,
        running_fuel_kg=running_fuel,
        restart_fuel_kg=restart_fuel,
        total_fuel_kg=result.fuel_used_kg,
        final_fuel_kg=result.fuel_remaining_kg,
        final_soc=result.final_soc,
        minimum_soc=result.min_soc,
        final_stored_energy_kwh=float(
            aircraft.battery.stored_energy_kwh(result.final_soc)
        ),
        restart_count=restart_count,
        loiter_restarts_per_hour=(
            loiter_restart_count / (loiter_s / 3600.0) if loiter_s else 0.0
        ),
        overall_engine_off_fraction=(
            sum(step.dt_s for step in steps if step.engine_shut_down) / duration_s
            if duration_s
            else 0.0
        ),
        loiter_engine_off_fraction=(
            sum(step.dt_s for step in loiter if step.engine_shut_down) / loiter_s
            if loiter_s
            else 0.0
        ),
        on_duration_distribution_json=json.dumps(
            _duration_distribution(steps, True), separators=(",", ":")
        ),
        off_duration_distribution_json=json.dumps(
            _duration_distribution(steps, False), separators=(",", ":")
        ),
        mean_engine_on_power_kw=mean_on_power,
        minimum_engine_on_power_kw=min(
            (step.engine_shaft_kw for step in on_steps), default=0.0
        ),
        maximum_engine_on_power_kw=max(
            (step.engine_shaft_kw for step in on_steps), default=0.0
        ),
        battery_limit_encounters_json=json.dumps(
            _battery_limit_encounters(steps), sort_keys=True, separators=(",", ":")
        ),
        mission_complete=result.mission_complete,
        feasible=not reasons,
        feasibility_reasons_json=json.dumps(reasons, separators=(",", ":")),
        termination_reason=result.termination_reason,
        reserve_shortfall_kg=max(mission.fuel_reserve_kg - result.fuel_remaining_kg, 0.0),
        descent_landing_completed=descent_landing,
        violations_json=json.dumps(violations, sort_keys=True, separators=(",", ":")),
        maximum_power_residual_kw=bus_residual,
        fuel_ledger_residual_kg=fuel_residual,
        energy_ledger_residual_kwh=balance.residual_kwh,
        discrete_energy_residual_fraction=balance.discrete_residual_fraction,
        battery_integration_residual_kwh=balance.battery_integration_residual_kwh,
        simulation_runtime_s=runtime_s,
    )


def run_controller_mission(
    spec: ControllerSpec,
    restart_cost_kg: float = 0.0,
    *,
    stage: str = "comparison",
) -> MissionRecord:
    """Run one fresh full mission through the authoritative simulator path."""
    aircraft, mission, controller_arguments = _build_inputs(spec, restart_cost_kg)
    started = time.perf_counter()
    result = run_mission(
        aircraft,
        mission,
        dt_s=REFERENCE_TIMESTEP_S,
        initial_soc=1.0,
        record_log=True,
        **controller_arguments,
    )
    runtime_s = time.perf_counter() - started
    return _record_from_result(
        spec,
        restart_cost_kg,
        stage,
        aircraft,
        mission,
        result,
        runtime_s,
    )


class _Checkpoint:
    def __init__(self, csv_path: Path, json_path: Path, classification: str) -> None:
        self.csv_path = csv_path
        self.json_path = json_path
        self.classification = classification
        self.records = self._load()

    def _load(self) -> list[MissionRecord]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return []
        with self.csv_path.open(newline="", encoding="utf-8") as stream:
            return [MissionRecord.from_csv_row(row) for row in csv.DictReader(stream)]

    def get(self, run_id: str) -> MissionRecord | None:
        return next((record for record in self.records if record.run_id == run_id), None)

    def append(self, record: MissionRecord) -> None:
        if self.get(record.run_id) is not None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(asdict(record)))
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(record))
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)
        payload = {
            "classification": self.classification,
            "record_count": len(self.records),
            "records": [asdict(item) for item in self.records],
        }
        temporary = self.json_path.with_suffix(self.json_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.json_path)


def _run_checkpointed(
    checkpoint: _Checkpoint,
    spec: ControllerSpec,
    restart_cost_kg: float,
    stage: str,
) -> MissionRecord:
    run_id = f"{stage}:{spec.candidate_id}:restart={restart_cost_kg:.3f}"
    existing = checkpoint.get(run_id)
    if existing is not None:
        print(f"checkpoint {run_id}", flush=True)
        return existing
    record = run_controller_mission(spec, restart_cost_kg, stage=stage)
    checkpoint.append(record)
    print(
        f"completed {run_id}: {record.total_time_s:.6f} s, "
        f"restarts={record.restart_count}, feasible={record.feasible}",
        flush=True,
    )
    return record


def _best(records: Sequence[MissionRecord], family: str) -> MissionRecord:
    eligible = [
        record for record in records if record.controller_family == family and record.feasible
    ]
    if not eligible:
        raise RuntimeError(f"no feasible {family} controller in local verification")
    return max(eligible, key=lambda record: (record.total_time_s, -record.restart_count))


def _spec_from_record(record: MissionRecord) -> ControllerSpec:
    configuration = json.loads(record.configuration_json)
    if record.controller_family == "fixed":
        return ControllerSpec(
            "fixed_s_ecms",
            "Tuned fixed-(s) ECMS",
            "fixed",
            True,
            s_ratio=float(configuration["s_ratio"]),
        )
    if record.controller_family == "pi":
        return ControllerSpec(
            "adaptive_pi_ecms",
            "Tuned adaptive PI-ECMS",
            "pi",
            True,
            s_ratio=float(configuration["s0_ratio"]),
            kp=float(configuration["kp"]),
            soc_ref=float(configuration["soc_ref"]),
        )
    return ControllerSpec(
        "optimised_thermostat",
        "Optimised thermostat",
        "thermostat",
        True,
        soc_low=float(configuration["soc_low"]),
        soc_high=float(configuration["soc_high"]),
    )


def select_verified_specs(
    records: Sequence[MissionRecord],
) -> tuple[ControllerSpec, ControllerSpec]:
    """Select the feasible endurance winner in each local ECMS neighbourhood."""
    return _spec_from_record(_best(records, "fixed")), _spec_from_record(_best(records, "pi"))


def _configuration_record_path(directory: Path) -> Path:
    return directory / "controller_configuration_record.json"


def run_local_verification(directory: Path) -> tuple[ControllerSpec, ControllerSpec]:
    checkpoint = _Checkpoint(
        directory / "controller_local_verification.csv",
        directory / "controller_local_verification.json",
        "bounded reconstruction because original full-mission sweep artifacts are absent",
    )
    for spec in controller_local_candidates():
        _run_checkpointed(checkpoint, spec, 0.0, "local_verification")
    fixed, pi = select_verified_specs(checkpoint.records)
    fixed_best = _best(checkpoint.records, "fixed")
    pi_best = _best(checkpoint.records, "pi")
    tied = {
        "fixed": [
            record.candidate_id
            for record in checkpoint.records
            if record.controller_family == "fixed"
            and record.feasible
            and abs(record.total_time_s - fixed_best.total_time_s) <= 1.0e-9
        ],
        "pi": [
            record.candidate_id
            for record in checkpoint.records
            if record.controller_family == "pi"
            and record.feasible
            and abs(record.total_time_s - pi_best.total_time_s) <= 1.0e-9
        ],
    }
    record = {
        "aircraft": {
            "mtow_kg": 1000.0,
            "wing_area_m2": build_reference_aircraft().wing_area_m2,
            "engine_rated_power_kw": build_reference_aircraft().engine.rated_power_kw,
            "battery_capacity_kwh": build_reference_aircraft().battery.capacity_kwh,
            "dry_mass_kg": build_reference_aircraft().masses.dry_kg,
            "initial_fuel_kg": build_reference_aircraft().masses.fuel_kg,
            "cruise_altitude_m": 3000.0,
        },
        "local_verification": {
            "original_artifacts_found": False,
            "fixed_s_ratio_resolution": 0.1,
            "pi_s0_ratio_resolution": 0.1,
            "pi_kp_resolution": 2.5,
            "candidate_count": len(controller_local_candidates()),
            "event_time_ties": tied,
            "selection_rule": "feasibility, endurance, then fewer restarts; stable candidate order for exact ties",
        },
        "selected": {
            fixed.key: fixed.configuration(0.0),
            pi.key: pi.configuration(0.0),
            OPTIMISED_THERMOSTAT.key: OPTIMISED_THERMOSTAT.configuration(0.0),
            UNTUNED_THERMOSTAT.key: UNTUNED_THERMOSTAT.configuration(0.0),
        },
    }
    _configuration_record_path(directory).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return fixed, pi


def _load_verified_specs(directory: Path) -> tuple[ControllerSpec, ControllerSpec]:
    checkpoint = _Checkpoint(
        directory / "controller_local_verification.csv",
        directory / "controller_local_verification.json",
        "bounded local verification",
    )
    if not checkpoint.records:
        raise RuntimeError("run the local verification stage first")
    return select_verified_specs(checkpoint.records)


def run_zero_cost_comparison(directory: Path) -> tuple[MissionRecord, ...]:
    fixed, pi = _load_verified_specs(directory)
    checkpoint = _Checkpoint(
        directory / "controller_zero_restart_comparison.csv",
        directory / "controller_zero_restart_comparison.json",
        "idealised zero restart-fuel comparison; transition counts remain physical diagnostics",
    )
    records = tuple(
        _run_checkpointed(checkpoint, spec, 0.0, "zero_cost_comparison")
        for spec in (fixed, pi, OPTIMISED_THERMOSTAT, UNTUNED_THERMOSTAT)
    )
    thermostat = next(
        record for record in records if record.controller_key == OPTIMISED_THERMOSTAT.key
    )
    expected = {
        "total_time_s": 56094.32539909772,
        "loiter_time_s": 49754.32539909772,
        "final_soc": 0.22035789264692873,
        "minimum_soc": 0.17966161710819917,
        "final_fuel_kg": 5.5078150370516985,
        "restart_count": 81,
        "overall_engine_off_fraction": 0.19966369004912132,
    }
    for name, value in expected.items():
        actual = getattr(thermostat, name)
        tolerance = 1.0e-9 if name != "restart_count" else 0.0
        if not math.isclose(float(actual), float(value), rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(
                f"optimised thermostat regression in {name}: {actual!r} != {value!r}"
            )
    return records


def run_restart_sensitivity(directory: Path) -> tuple[MissionRecord, ...]:
    fixed, pi = _load_verified_specs(directory)
    checkpoint = _Checkpoint(
        directory / "controller_restart_sensitivity.csv",
        directory / "controller_restart_sensitivity.json",
        "parameter-frozen sensitivity; controller parameters were selected at zero restart cost",
    )
    records: list[MissionRecord] = []
    for restart_cost in RESTART_COSTS_KG:
        for spec in (fixed, pi, OPTIMISED_THERMOSTAT):
            records.append(
                _run_checkpointed(
                    checkpoint,
                    spec,
                    restart_cost,
                    "restart_sensitivity",
                )
            )
    return tuple(records)


def _ranking(records: Sequence[MissionRecord], restart_cost: float) -> tuple[str, ...]:
    matching = [
        record
        for record in records
        if math.isclose(record.restart_cost_per_start_kg, restart_cost, abs_tol=1.0e-12)
        and record.tuned
        and record.feasible
    ]
    return tuple(
        record.controller_key
        for record in sorted(matching, key=lambda item: item.total_time_s, reverse=True)
    )


def run_optional_restart_retuning(directory: Path) -> tuple[bool, tuple[MissionRecord, ...]]:
    sensitivity = _Checkpoint(
        directory / "controller_restart_sensitivity.csv",
        directory / "controller_restart_sensitivity.json",
        "parameter-frozen restart sensitivity",
    ).records
    zero_ranking = _ranking(sensitivity, 0.0)
    positive_ranking = _ranking(sensitivity, 0.1)
    required = bool(zero_ranking and positive_ranking and zero_ranking != positive_ranking)
    gate_path = directory / "controller_restart_retuning_gate.json"
    if not required:
        gate_path.write_text(
            json.dumps(
                {
                    "required": False,
                    "zero_cost_ranking": zero_ranking,
                    "restart_0_1_ranking": positive_ranking,
                    "reason": "the parameter-frozen ranking did not change",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return False, ()
    fixed, pi = _load_verified_specs(directory)
    candidates = (
        *(
            replace(fixed, s_ratio=ratio)
            for ratio in FIXED_LOCAL_RATIOS
        ),
        *(
            replace(pi, s_ratio=ratio, kp=kp)
            for ratio in PI_LOCAL_RATIOS
            for kp in PI_LOCAL_GAINS
        ),
        *(
            replace(OPTIMISED_THERMOSTAT, soc_low=low, soc_high=high)
            for low, high in RESTART_RETUNE_THERMOSTAT_BANDS
        ),
    )
    checkpoint = _Checkpoint(
        directory / "controller_restart_retuning_0_1.csv",
        directory / "controller_restart_retuning_0_1.json",
        "bounded 18-run local retuning at 0.1 kg/start; no 0.5 kg/start retuning",
    )
    records = tuple(
        _run_checkpointed(checkpoint, spec, 0.1, "restart_retuning_0_1")
        for spec in candidates
    )
    best_by_family = {
        family: asdict(_best(records, family)) for family in ("fixed", "pi", "thermostat")
    }
    gate_path.write_text(
        json.dumps(
            {
                "required": True,
                "zero_cost_ranking": zero_ranking,
                "restart_0_1_ranking": positive_ranking,
                "evaluation_budget": len(candidates),
                "best_by_family": best_by_family,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return True, records


def pareto_controller_keys(records: Sequence[MissionRecord]) -> tuple[str, ...]:
    """Return nondominated keys for higher endurance and fewer restarts."""
    tuned = tuple(record for record in records if record.tuned and record.feasible)
    frontier = []
    for candidate in tuned:
        dominated = any(
            other.total_time_s >= candidate.total_time_s
            and other.restart_count <= candidate.restart_count
            and (
                other.total_time_s > candidate.total_time_s
                or other.restart_count < candidate.restart_count
            )
            for other in tuned
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate.controller_key)
    return tuple(frontier)


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty figure data source")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "svg.hashsalt": "aerothon-controller-comparison",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save_figure(figure: Any, base_path: Path) -> tuple[Path, Path, Path]:
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    pdf = base_path.with_suffix(".pdf")
    figure.savefig(png, dpi=150, facecolor="white")
    figure.savefig(
        svg,
        facecolor="white",
        metadata={"Date": None, "Creator": "Aerothon controller comparison"},
    )
    figure.savefig(
        pdf,
        facecolor="white",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "Aerothon controller comparison",
        },
    )
    return png, svg, pdf


def _short_label(key: str) -> str:
    return {
        "fixed_s_ecms": "Fixed-(s)",
        "adaptive_pi_ecms": "Adaptive PI",
        "optimised_thermostat": "Thermostat",
        "untuned_thermostat": "Untuned thermostat",
    }[key]


def _endurance_figure(zero: Sequence[MissionRecord], directory: Path) -> tuple[Path, ...]:
    tuned = [record for record in zero if record.tuned]
    best = max(record.total_time_s for record in tuned)
    untuned = next(record for record in zero if not record.tuned)
    rows = [
        {
            "controller_key": record.controller_key,
            "controller": record.display_name,
            "role": "tuned" if record.tuned else "development_reference",
            "endurance_s": record.total_time_s,
            "endurance_h": record.total_time_s / 3600.0,
            "delta_from_best_s": record.total_time_s - best,
            "delta_from_best_min": (record.total_time_s - best) / 60.0,
        }
        for record in (*tuned, untuned)
    ]
    csv_path = _write_rows(directory / "controller_endurance_comparison.csv", rows)
    plotted = _read_rows(csv_path)
    tuned_rows = [row for row in plotted if row["role"] == "tuned"]
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(16, 9))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.78, wspace=0.28)
    labels = [_short_label(row["controller_key"]) for row in tuned_rows]
    colours = [CONTROLLER_COLOURS[row["controller_key"]] for row in tuned_rows]
    values = [float(row["endurance_h"]) for row in tuned_rows]
    bars = axes[0].bar(labels, values, color=colours, width=0.62)
    axes[0].set_title("A. Full-mission endurance")
    axes[0].set_ylabel("Endurance (h)")
    axes[0].set_ylim(0.0, max(values) * 1.13)
    axes[0].bar_label(bars, labels=[f"{value:.3f} h" for value in values], padding=6)
    untuned_h = float(next(row for row in plotted if row["role"] != "tuned")["endurance_h"])
    axes[0].axhline(untuned_h, color=CONTROLLER_COLOURS["untuned_thermostat"], linestyle="--")
    axes[0].text(
        0.02,
        untuned_h / axes[0].get_ylim()[1] + 0.015,
        f"Untuned thermostat reference: {untuned_h:.3f} h",
        transform=axes[0].transAxes,
        color="#4D6F61",
        fontsize=14,
    )
    deltas = [float(row["delta_from_best_min"]) for row in tuned_rows]
    bars = axes[1].bar(labels, deltas, color=colours, width=0.62)
    axes[1].set_title("B. Difference from zero-cost winner")
    axes[1].set_ylabel("Endurance difference (min)")
    axes[1].axhline(0.0, color="#555555", linewidth=1.2)
    for bar, value in zip(bars, deltas, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value - 0.10,
            f"{value:+.2f} min",
            ha="center",
            va="top",
            fontsize=15,
        )
    axes[1].margins(y=0.25)
    figure.suptitle("Controller endurance comparison", fontsize=26, y=0.965)
    figure.text(
        0.5,
        0.865,
        "Frozen 1000 kg aircraft - 3 km mission - zero restart fuel",
        ha="center",
        fontsize=17,
    )
    paths = _save_figure(figure, directory / "controller_endurance_comparison")
    plt.close(figure)
    return (csv_path, *paths)


def _tradeoff_figure(zero: Sequence[MissionRecord], directory: Path) -> tuple[Path, ...]:
    tuned = [record for record in zero if record.tuned]
    frontier = set(pareto_controller_keys(tuned))
    rows = [
        {
            "controller_key": record.controller_key,
            "controller": record.display_name,
            "restart_count": record.restart_count,
            "endurance_s": record.total_time_s,
            "endurance_h": record.total_time_s / 3600.0,
            "pareto_optimal": record.controller_key in frontier,
        }
        for record in tuned
    ]
    csv_path = _write_rows(directory / "controller_endurance_restart_tradeoff.csv", rows)
    plotted = _read_rows(csv_path)
    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(16, 9))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.79)
    for index, row in enumerate(plotted):
        key = row["controller_key"]
        x = int(row["restart_count"])
        y = float(row["endurance_h"])
        axis.scatter(
            x,
            y,
            s=180,
            color=CONTROLLER_COLOURS[key],
            marker=CONTROLLER_MARKERS[key],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        offsets = {
            "fixed_s_ecms": (-150, -30),
            "adaptive_pi_ecms": (18, 12),
            "optimised_thermostat": (12, 18),
        }
        axis.annotate(
            f"{_short_label(key)}\n{x} starts, {y:.3f} h",
            (x, y),
            xytext=offsets[key],
            textcoords="offset points",
            fontsize=15,
        )
    frontier_rows = sorted(
        (row for row in plotted if row["pareto_optimal"].lower() == "true"),
        key=lambda row: int(row["restart_count"]),
    )
    if len(frontier_rows) > 1:
        axis.plot(
            [int(row["restart_count"]) for row in frontier_rows],
            [float(row["endurance_h"]) for row in frontier_rows],
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label="Zero-cost Pareto frontier",
        )
        axis.legend(frameon=False, loc="lower right")
    axis.set_xlabel("Engine restart count")
    axis.set_ylabel("Endurance (h)")
    axis.set_title(
        "Frozen 1000 kg aircraft - 3 km mission - zero restart fuel",
        fontsize=17,
        pad=26,
    )
    figure.suptitle("Endurance versus engine restarts", fontsize=26, y=0.965)
    axis.text(
        0.02,
        0.95,
        "Preferable region: higher endurance, fewer restarts",
        transform=axis.transAxes,
        va="top",
        fontsize=17,
    )
    axis.margins(x=0.17, y=0.22)
    paths = _save_figure(figure, directory / "controller_endurance_restart_tradeoff")
    plt.close(figure)
    return (csv_path, *paths)


def _sensitivity_figure(
    sensitivity: Sequence[MissionRecord], directory: Path
) -> tuple[Path, ...]:
    csv_path = directory / "controller_restart_sensitivity.csv"
    if not csv_path.exists():
        _write_rows(csv_path, [asdict(record) for record in sensitivity])
    plotted = _read_rows(csv_path)
    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=(16, 9))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.80)
    annotation_offsets = {
        ("fixed_s_ecms", 0.1): (-28, 18),
        ("adaptive_pi_ecms", 0.1): (46, -36),
        ("optimised_thermostat", 0.1): (0, 18),
        ("fixed_s_ecms", 0.5): (-78, 18),
        ("adaptive_pi_ecms", 0.5): (-78, -42),
        ("optimised_thermostat", 0.5): (-92, -35),
    }
    for key in ("fixed_s_ecms", "adaptive_pi_ecms", "optimised_thermostat"):
        series = sorted(
            (row for row in plotted if row["controller_key"] == key),
            key=lambda row: float(row["restart_cost_per_start_kg"]),
        )
        x = [float(row["restart_cost_per_start_kg"]) for row in series]
        y = [float(row["total_time_s"]) / 3600.0 for row in series]
        axis.plot(
            x,
            y,
            color=CONTROLLER_COLOURS[key],
            marker=CONTROLLER_MARKERS[key],
            markersize=10,
            linewidth=3,
            label=_short_label(key),
        )
        for row, x_value, y_value in zip(series, x, y, strict=True):
            feasible = row["feasible"].lower() == "true"
            offset = annotation_offsets.get((key, x_value))
            if offset is not None:
                status = "" if feasible else " - INFEASIBLE"
                axis.annotate(
                    f"{_short_label(key)} {y_value:.2f} h{status}",
                    (x_value, y_value),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                    fontweight="bold" if not feasible else "normal",
                )
            if not feasible:
                axis.scatter(
                    [x_value],
                    [y_value],
                    s=260,
                    marker="x",
                    color=CONTROLLER_COLOURS[key],
                    linewidth=3,
                    zorder=4,
                )
    axis.set_xlabel("Restart fuel assumption (kg/start)")
    axis.set_ylabel("Full-mission endurance (h)")
    axis.set_xticks(RESTART_COSTS_KG)
    figure.suptitle("Restart-fuel sensitivity", fontsize=26, y=0.965)
    figure.text(
        0.5,
        0.895,
        "Restart fuel values are sensitivity assumptions, not calibrated measurements.",
        ha="center",
        fontsize=17,
    )
    axis.legend(frameon=False, ncol=3, loc="lower left")
    axis.grid(axis="y", color="#D5D5D5", linewidth=0.8)
    axis.margins(x=0.08, y=0.18)
    paths = _save_figure(figure, directory / "controller_restart_sensitivity")
    plt.close(figure)
    return (csv_path, *paths)


def _resources_figure(zero: Sequence[MissionRecord], directory: Path) -> tuple[Path, ...]:
    tuned = [record for record in zero if record.tuned]
    rows = [
        {
            "controller_key": record.controller_key,
            "controller": record.display_name,
            "final_soc": record.final_soc,
            "fuel_remaining_kg": record.final_fuel_kg,
            "mandatory_reserve_kg": 5.0,
            "fuel_above_reserve_kg": record.final_fuel_kg - 5.0,
        }
        for record in tuned
    ]
    csv_path = _write_rows(directory / "controller_terminal_resources.csv", rows)
    plotted = _read_rows(csv_path)
    labels = [_short_label(row["controller_key"]) for row in plotted]
    colours = [CONTROLLER_COLOURS[row["controller_key"]] for row in plotted]
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(16, 9))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.78, wspace=0.30)
    soc = [float(row["final_soc"]) for row in plotted]
    bars = axes[0].barh(labels, soc, color=colours, height=0.55)
    axes[0].set_title("A. Final battery state")
    axes[0].set_xlabel("")
    axes[0].set_xlim(0.0, max(soc) * 1.35)
    axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in soc], padding=6)
    fuel = [float(row["fuel_above_reserve_kg"]) for row in plotted]
    bars = axes[1].barh(labels, fuel, color=colours, height=0.55)
    axes[1].set_title("B. Fuel slack after landing")
    axes[1].set_xlabel("")
    axes[1].set_xlim(0.0, max(fuel) * 1.35)
    axes[1].bar_label(bars, labels=[f"{value:.3f} kg" for value in fuel], padding=6)
    figure.suptitle("Terminal resource utilisation", fontsize=26, y=0.965)
    figure.text(
        0.5,
        0.895,
        "Frozen 1000 kg aircraft - 3 km mission - zero restart fuel",
        ha="center",
        fontsize=17,
    )
    figure.text(0.29, 0.075, "Final state of charge", ha="center", fontsize=18)
    figure.text(
        0.735,
        0.065,
        "Fuel above mandatory 5 kg reserve\n(kg)",
        ha="center",
        fontsize=18,
    )
    paths = _save_figure(figure, directory / "controller_terminal_resources")
    plt.close(figure)
    return (csv_path, *paths)


def generate_controller_figures(
    zero_records: Sequence[MissionRecord],
    sensitivity_records: Sequence[MissionRecord],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Generate four deterministic 16:9 figure sets from exact source CSVs."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return (
        *_endurance_figure(zero_records, directory),
        *_tradeoff_figure(zero_records, directory),
        *_sensitivity_figure(sensitivity_records, directory),
        *_resources_figure(zero_records, directory),
    )


def _load_records(path: Path) -> tuple[MissionRecord, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        return tuple(MissionRecord.from_csv_row(row) for row in csv.DictReader(stream))


def _comparison_rows(
    zero: Sequence[MissionRecord], sensitivity: Sequence[MissionRecord]
) -> list[dict[str, Any]]:
    tuned = [record for record in zero if record.tuned]
    best = max(record.total_time_s for record in tuned)
    sensitivity_map = {
        (record.controller_key, record.restart_cost_per_start_kg): record
        for record in sensitivity
    }
    ranked = sorted(tuned, key=lambda record: record.total_time_s, reverse=True)
    return [
        {
            "controller_key": record.controller_key,
            "controller": record.display_name,
            "configuration_json": record.configuration_json,
            "zero_cost_rank": index + 1,
            "zero_cost_endurance_s": record.total_time_s,
            "zero_cost_endurance_h": record.total_time_s / 3600.0,
            "delta_from_best_s": record.total_time_s - best,
            "delta_from_best_min": (record.total_time_s - best) / 60.0,
            "final_soc": record.final_soc,
            "minimum_soc": record.minimum_soc,
            "final_fuel_kg": record.final_fuel_kg,
            "fuel_above_reserve_kg": record.final_fuel_kg - 5.0,
            "restart_count": record.restart_count,
            "loiter_restarts_per_hour": record.loiter_restarts_per_hour,
            "overall_engine_off_fraction": record.overall_engine_off_fraction,
            "pareto_optimal": record.controller_key in pareto_controller_keys(tuned),
            "restart_0_1_endurance_s": sensitivity_map[
                (record.controller_key, 0.1)
            ].total_time_s,
            "restart_0_1_feasible": sensitivity_map[
                (record.controller_key, 0.1)
            ].feasible,
            "restart_0_5_endurance_s": sensitivity_map[
                (record.controller_key, 0.5)
            ].total_time_s,
            "restart_0_5_feasible": sensitivity_map[
                (record.controller_key, 0.5)
            ].feasible,
        }
        for index, record in enumerate(ranked)
    ]


def _write_report(
    directory: Path,
    zero: Sequence[MissionRecord],
    sensitivity: Sequence[MissionRecord],
) -> Path:
    rows = _comparison_rows(zero, sensitivity)
    comparison_path = _write_rows(directory / "controller_comparison.csv", rows)
    best = rows[0]
    thermostat = next(row for row in rows if row["controller_key"] == "optimised_thermostat")
    fixed = next(row for row in rows if row["controller_key"] == "fixed_s_ecms")
    pi = next(row for row in rows if row["controller_key"] == "adaptive_pi_ecms")
    sensitivity_rankings = {
        cost: _ranking(sensitivity, cost) for cost in RESTART_COSTS_KG
    }
    gate = json.loads(
        (directory / "controller_restart_retuning_gate.json").read_text(encoding="utf-8")
    )
    positive_best = sensitivity_rankings[0.1][0] if sensitivity_rankings[0.1] else None
    retained_fraction = float(thermostat["zero_cost_endurance_s"]) / float(
        best["zero_cost_endurance_s"]
    )
    thermostat_close = retained_fraction >= _PRACTICAL_ENDURANCE_RETENTION_MIN
    thermostat_robust = positive_best == "optimised_thermostat" and bool(
        thermostat["restart_0_1_feasible"]
    )
    recommended_key = (
        "optimised_thermostat" if thermostat_close and thermostat_robust else best["controller_key"]
    )
    recommendation_kind = "practical" if recommended_key != best["controller_key"] else "both"
    numerical_statement = (
        f"{best['controller']} achieved the greatest ideal zero-restart-fuel endurance at "
        f"{float(best['zero_cost_endurance_h']):.4f} h."
    )
    restart_reduction = int(pi["restart_count"]) - int(thermostat["restart_count"])
    retained = retained_fraction * 100.0
    engineering_statement = (
        f"The optimised thermostat retained {retained:.3f}% of the best ideal endurance "
        f"while reducing starts by {restart_reduction} relative to adaptive PI-ECMS. "
        f"Its positive-restart-cost sensitivity and two-threshold chromosome support its "
        f"selection for plant-controller co-optimisation."
        if recommended_key == "optimised_thermostat"
        else f"{best['controller']} is recommended because it leads the measured endurance "
        "comparison without a demonstrated positive-cost robustness disadvantage."
    )
    lines = [
        "# Frozen-aircraft full-mission controller comparison",
        "",
        "This study compares controller parameters frozen at their zero-restart-fuel selections. "
        "Restart costs of 0.1 and 0.5 kg/start are sensitivity assumptions, not calibrated measurements. "
        "No plant variable, controller default, battery model, DP, fuzzy controller or GA was changed.",
        "",
        "## Specification audit",
        "",
        "- The supplied phrase `zero-restart thermostat result` is interpreted as zero restart *fuel*; "
        "the mission contains 81 measured OFF-to-ON transitions.",
        "- No executable historical complete-mission fixed/PI sweep or checkpoint is present. The "
        "verification therefore reconstructs a 3-point fixed neighbourhood and a 3 x 3 PI neighbourhood "
        "at the historical 0.1 ratio and 2.5 gain resolution.",
        "- Fixed ratios 1.1 and 1.2 produce the same event-time endurance to 1e-9 s. Ratio 1.1 is "
        "retained as the historical lower-ratio representative, not called a unique optimum.",
        "- The former project-local interpreter was absent; execution used the configured Python "
        "with the repository dependency directory.",
        "- The existing pure-thermal control group is not a feasible full-mission continuous comparator "
        "for this aircraft, so it is not included.",
        "",
        "## Exact selected configurations",
        "",
        "The full records are in `controller_configuration_record.json`. Fixed-(s) uses the switching-ratio "
        "parameter with adaptation disabled. PI uses a switching-relative base, proportional SoC feedback, "
        "no integral state and the shared [0.5, 20] clamp. Thermostat uses causal 60/60 s hard dwell and the "
        "existing maximum-feasible ON-power rule.",
        "",
        "## Zero restart-fuel result",
        "",
        "| Controller | Endurance (h) | Delta (min) | Final SoC | Starts | Fuel above reserve (kg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['controller']} | {float(row['zero_cost_endurance_h']):.4f} | "
            f"{float(row['delta_from_best_min']):+.3f} | {float(row['final_soc']):.4f} | "
            f"{int(row['restart_count'])} | {float(row['fuel_above_reserve_kg']):.4f} |"
        )
    lines.extend(
        [
            "",
            f"Zero-cost Pareto frontier: {', '.join(pareto_controller_keys(zero))}.",
            "",
            "## Parameter-frozen restart sensitivity",
            "",
            "| Cost (kg/start) | Controller | Endurance (h) | Starts | Running fuel (kg) | Restart fuel (kg) | Final fuel (kg) | Final SoC | Feasible | Termination |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for record in sorted(
        sensitivity,
        key=lambda item: (item.restart_cost_per_start_kg, item.controller_key),
    ):
        lines.append(
            f"| {record.restart_cost_per_start_kg:.1f} | {record.display_name} | "
            f"{record.total_time_s / 3600.0:.4f} | {record.restart_count} | "
            f"{record.running_fuel_kg:.4f} | {record.restart_fuel_kg:.4f} | "
            f"{record.final_fuel_kg:.4f} | {record.final_soc:.4f} | "
            f"{record.feasible} | `{record.termination_reason}` |"
        )
    ranking_text = {
        cost: ", ".join(ranking) if ranking else "no feasible controller"
        for cost, ranking in sensitivity_rankings.items()
    }
    lines.extend(
        [
            "",
            f"- 0 kg/start ranking: {ranking_text[0.0]}.",
            f"- 0.1 kg/start ranking: {ranking_text[0.1]}.",
            f"- 0.5 kg/start ranking: {ranking_text[0.5]}.",
            f"- Optional 0.1 kg/start retuning required: {gate['required']}.",
        ]
    )
    if gate["required"]:
        retuned = gate["best_by_family"]
        lines.extend(
            [
                "- Bounded 0.1 kg/start retuning used 18 missions and no 0.5 kg/start retuning.",
                "- Best retuned thermostat: "
                f"`{retuned['thermostat']['candidate_id']}`, "
                f"{retuned['thermostat']['total_time_s']:.3f} s with "
                f"{retuned['thermostat']['restart_count']} starts.",
                "- Best feasible fixed/PI neighbourhood result: "
                f"{max(retuned['fixed']['total_time_s'], retuned['pi']['total_time_s']):.3f} s.",
            ]
        )
    lines.extend(
        [
            "",
            "## PPT-ready statements",
            "",
            f"**Numerical statement.** {numerical_statement}",
            "",
            f"**Engineering recommendation.** {engineering_statement}",
            "",
            "## Verdict",
            "",
            f"Recommended controller: `{recommended_key}`. The recommendation is {recommendation_kind}, "
            "and is bounded by this frozen aircraft, timestep, local controller searches and uncalibrated "
            "restart-cost sensitivity. It is not a global-optimality claim.",
            "",
            "## Reproducibility",
            "",
            f"Machine-readable tuned comparison: `{comparison_path.as_posix()}`. Every plotted value is read "
            "back from its figure-specific CSV before rendering. Full-suite status: pending final execution.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH


def run_figure_and_report_stage(directory: Path) -> tuple[Path, ...]:
    zero = _load_records(directory / "controller_zero_restart_comparison.csv")
    sensitivity = _load_records(directory / "controller_restart_sensitivity.csv")
    paths = generate_controller_figures(zero, sensitivity, directory)
    report = _write_report(directory, zero, sensitivity)
    manifest = {
        "files": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in (*paths, report)
        ]
    }
    manifest_path = directory / "controller_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return (*paths, report, manifest_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("local", "zero", "sensitivity", "retune", "figures"),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    directory = arguments.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "local":
        result: Any = run_local_verification(directory)
    elif arguments.stage == "zero":
        result = run_zero_cost_comparison(directory)
    elif arguments.stage == "sensitivity":
        result = run_restart_sensitivity(directory)
    elif arguments.stage == "retune":
        result = run_optional_restart_retuning(directory)
    else:
        result = run_figure_and_report_stage(directory)
    if arguments.stage == "retune":
        required, records = result
        summary = {"required": required, "record_count": len(records)}
    elif arguments.stage == "figures":
        summary = {"files": [str(path) for path in result]}
    elif arguments.stage == "local":
        summary = {"selected": [asdict(spec) for spec in result]}
    else:
        summary = {
            "record_count": len(result),
            "run_ids": [record.run_id for record in result],
        }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
