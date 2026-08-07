"""Named uncalibrated thermostat experiment on the reference six-phase mission."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from src.control.thermostat import (
    TerminalStrategy,
    ThermostatParameters,
    ThermostatState,
)
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.mass import build_mass_budget
from src.models.powertrain import SeriesPowertrain
from src.simulation.mission import MissionProfile, ps1_mission
from src.simulation.simulator import (
    Aircraft,
    MissionResult,
    TimeStep,
    mission_energy_balance,
    run_mission,
)

__all__ = [
    "REFERENCE_INITIAL_THERMOSTAT_STATE",
    "REFERENCE_THERMOSTAT_PARAMETERS",
    "ThermostatReferenceRun",
    "build_reference_aircraft",
    "run_reference_thermostat_mission",
    "summarise_thermostat_mission",
    "write_thermostat_mission_artifacts",
]

REFERENCE_WING_AREA_M2 = 7.59175537062125
REFERENCE_ENGINE_POWER_KW = 86.7791369750147
REFERENCE_BATTERY_CAPACITY_KWH = 10.0
REFERENCE_TIMESTEP_S = 60.0

# Uncalibrated integration reference; see assumptions.md C-08.
REFERENCE_THERMOSTAT_PARAMETERS = ThermostatParameters(
    soc_low=0.4,
    soc_high=0.6,
    minimum_on_time_s=60.0,
    minimum_off_time_s=60.0,
    restart_fuel_kg=0.0,
    engine_on_power_kw=None,
    terminal_strategy=TerminalStrategy.CAUSAL,
)
REFERENCE_INITIAL_THERMOSTAT_STATE = ThermostatState(
    engine_on=True,
    elapsed_in_state_s=60.0,
)


@dataclass(frozen=True)
class ThermostatReferenceRun:
    """Models, configuration, and result for the named reference experiment."""

    aircraft: Aircraft
    mission: MissionProfile
    parameters: ThermostatParameters
    initial_state: ThermostatState
    initial_soc: float
    result: MissionResult


def build_reference_aircraft() -> Aircraft:
    """Build the explicit 1000 kg reference experiment, not a GA optimum."""
    powertrain = SeriesPowertrain()
    peak_bus_kw = float(
        powertrain.bus_power_from_engine(REFERENCE_ENGINE_POWER_KW)
    ) + 30.0
    masses = build_mass_budget(
        REFERENCE_ENGINE_POWER_KW,
        REFERENCE_BATTERY_CAPACITY_KWH,
        peak_bus_kw,
        REFERENCE_WING_AREA_M2,
        16.0,
    )
    return Aircraft(
        wing_area_m2=REFERENCE_WING_AREA_M2,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        engine=Turboshaft(REFERENCE_ENGINE_POWER_KW),
        battery=BatteryPack(REFERENCE_BATTERY_CAPACITY_KWH),
        powertrain=powertrain,
        masses=masses,
    )


def run_reference_thermostat_mission() -> ThermostatReferenceRun:
    """Execute exactly one default-timestep reference thermostat mission."""
    aircraft = build_reference_aircraft()
    mission = ps1_mission()
    result = run_mission(
        aircraft,
        mission,
        thermostat_parameters=REFERENCE_THERMOSTAT_PARAMETERS,
        initial_thermostat_state=REFERENCE_INITIAL_THERMOSTAT_STATE,
        dt_s=REFERENCE_TIMESTEP_S,
        initial_soc=1.0,
        record_log=True,
    )
    return ThermostatReferenceRun(
        aircraft=aircraft,
        mission=mission,
        parameters=REFERENCE_THERMOSTAT_PARAMETERS,
        initial_state=REFERENCE_INITIAL_THERMOSTAT_STATE,
        initial_soc=1.0,
        result=result,
    )


def _duration_distribution(
    steps: Sequence[TimeStep], engine_on: bool
) -> dict[str, Any]:
    durations: list[float] = []
    current_on: bool | None = None
    elapsed_s = 0.0
    for step in steps:
        step_on = not step.engine_shut_down
        if current_on is None:
            current_on = step_on
        if step_on != current_on:
            if current_on is engine_on:
                durations.append(elapsed_s)
            current_on = step_on
            elapsed_s = 0.0
        elapsed_s += step.dt_s
    if current_on is engine_on and elapsed_s > 0.0:
        durations.append(elapsed_s)
    return {
        "count": len(durations),
        "minimum_s": min(durations, default=0.0),
        "mean_s": fmean(durations) if durations else 0.0,
        "maximum_s": max(durations, default=0.0),
        "samples_s": durations,
    }


def _range(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum_kw": min(values, default=0.0),
        "maximum_kw": max(values, default=0.0),
    }


def _constraint_encounters(steps: Sequence[TimeStep]) -> dict[str, Any]:
    controller: dict[str, dict[str, float]] = {}
    battery: dict[str, dict[str, float]] = {}
    for step in steps:
        active = step.controller_active_constraint or "none"
        entry = controller.setdefault(active, {"steps": 0, "duration_s": 0.0})
        entry["steps"] += 1
        entry["duration_s"] += step.dt_s
        if step.battery_active_limit == "none" or abs(step.battery_bus_kw) <= 1.0e-10:
            continue
        direction = "discharge" if step.battery_bus_kw > 0.0 else "charge"
        key = f"{direction}:{step.battery_active_limit}"
        entry = battery.setdefault(key, {"steps": 0, "duration_s": 0.0})
        entry["steps"] += 1
        entry["duration_s"] += step.dt_s
    return {"controller": controller, "battery": battery}


def summarise_thermostat_mission(run: ThermostatReferenceRun) -> dict[str, Any]:
    """Return the complete auditable thermostat mission report."""
    result = run.result
    if result.log is None:
        raise ValueError("thermostat mission summary requires a recorded log")
    steps = result.log
    duration_s = sum(step.dt_s for step in steps)
    loiter = tuple(step for step in steps if step.phase == "loiter")
    loiter_s = sum(step.dt_s for step in loiter)
    running_fuel_kg = sum(step.fuel_flow_kg_s * step.dt_s for step in steps)
    restart_fuel_kg = sum(step.restart_fuel_kg for step in steps)
    restarts = sum(
        int(step.thermostat_transitioned and bool(step.requested_engine_on))
        for step in steps
    )
    loiter_restarts = sum(
        int(step.thermostat_transitioned and bool(step.requested_engine_on))
        for step in loiter
    )
    regime_times: dict[str, float] = {}
    for step in steps:
        regime = step.controller_regime or "none"
        regime_times[regime] = regime_times.get(regime, 0.0) + step.dt_s
    requested_on = tuple(
        float(step.requested_engine_shaft_kw)
        for step in steps
        if step.requested_engine_on and step.requested_engine_shaft_kw is not None
    )
    delivered_on = tuple(
        step.engine_shaft_kw for step in steps if not step.engine_shut_down
    )
    balance = mission_energy_balance(result)
    bus_residuals = tuple(
        step.bus_from_engine_kw + step.battery_bus_kw - step.bus_demand_kw
        for step in steps
    )
    phases_seen = {step.phase for step in steps}
    descent_landing_completed = (
        result.mission_complete
        and "descent" in phases_seen
        and "landing" in phases_seen
        and steps[-1].phase == "landing"
    )
    total_off_s = sum(step.dt_s for step in steps if step.engine_shut_down)
    loiter_off_s = sum(step.dt_s for step in loiter if step.engine_shut_down)
    return {
        "experiment": {
            "name": "uncalibrated_causal_thermostat_reference",
            "classification": "integration reference, not calibrated or optimised",
            "reference_design_is_ga_optimised": False,
        },
        "configuration": {
            "parameters": {
                **asdict(run.parameters),
                "terminal_strategy": run.parameters.terminal_strategy.value,
                "dwell_semantics": run.parameters.dwell_semantics.value,
            },
            "initial_state": asdict(run.initial_state),
            "initial_state_rationale": (
                "engine available at mission start with minimum ON dwell already "
                "satisfied; initial availability is not counted as a restart"
            ),
            "timestep_s": REFERENCE_TIMESTEP_S,
            "restart_fuel_owner": (
                "run_mission charges the engine-model restart value once per actual "
                "OFF-to-ON transition"
            ),
            "causal_future_information_used": False,
        },
        "aircraft": {
            "mtow_kg": run.aircraft.masses.total_kg,
            "dry_mass_kg": run.aircraft.masses.dry_kg,
            "wing_area_m2": run.aircraft.wing_area_m2,
            "engine_rated_power_kw": run.aircraft.engine.rated_power_kw,
            "battery_capacity_kwh": run.aircraft.battery.capacity_kwh,
        },
        "mission": {
            "mission_complete": result.mission_complete,
            "total_time_s": result.endurance_s,
            "loiter_time_s": loiter_s,
            "phase_durations_s": result.phase_durations_s,
            "termination_reason": result.termination_reason,
            "descent_and_landing_completed": descent_landing_completed,
            "reserve_shortfall_kg": max(
                run.mission.fuel_reserve_kg - result.fuel_remaining_kg, 0.0
            ),
        },
        "fuel": {
            "initial_kg": run.aircraft.masses.fuel_kg,
            "running_kg": running_fuel_kg,
            "restart_kg": restart_fuel_kg,
            "total_consumed_kg": result.fuel_used_kg,
            "remaining_kg": result.fuel_remaining_kg,
        },
        "battery": {
            "initial_soc": run.initial_soc,
            "final_soc": result.final_soc,
            "minimum_soc": result.min_soc,
            "bus_charge_kwh": sum(
                max(-step.battery_bus_kw, 0.0) * step.dt_s / 3600.0
                for step in steps
            ),
            "bus_discharge_kwh": sum(
                max(step.battery_bus_kw, 0.0) * step.dt_s / 3600.0
                for step in steps
            ),
        },
        "engine_schedule": {
            "restart_count": restarts,
            "loiter_restarts_per_hour": (
                loiter_restarts / (loiter_s / 3600.0) if loiter_s > 0.0 else 0.0
            ),
            "total_engine_off_fraction": total_off_s / duration_s,
            "loiter_engine_off_fraction": loiter_off_s / loiter_s if loiter_s else 0.0,
            "on_duration_distribution": _duration_distribution(steps, True),
            "off_duration_distribution": _duration_distribution(steps, False),
            "regime_time_s": regime_times,
            "requested_on_power_range": _range(requested_on),
            "delivered_on_power_range": _range(delivered_on),
        },
        "constraints": _constraint_encounters(steps),
        "accounting": {
            "fuel_reconstruction_residual_kg": result.fuel_used_kg
            - running_fuel_kg
            - restart_fuel_kg,
            "maximum_bus_balance_residual_kw": max(
                (abs(value) for value in bus_residuals), default=0.0
            ),
            "energy_balance_residual_kwh": balance.residual_kwh,
            "energy_balance_residual_fraction": balance.residual_fraction,
            "discrete_energy_balance_residual_fraction": (
                balance.discrete_residual_fraction
            ),
            "battery_integration_residual_kwh": (
                balance.battery_integration_residual_kwh
            ),
        },
        "failures": {
            "failure_flags": result.failure_flags,
            "hard_dwell_violation_count": sum(
                int(step.thermostat_dwell_violation) for step in steps
            ),
            "hard_dwell_infeasible": "hard_dwell_infeasible" in result.failure_flags,
            "controller_infeasible": "controller_infeasible" in result.failure_flags,
            "plant_infeasible": any(not step.plant_feasible for step in steps),
        },
    }


def _phase_regime_rows(steps: Sequence[TimeStep]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[TimeStep]] = {}
    for step in steps:
        groups.setdefault((step.phase, step.controller_regime or "none"), []).append(step)
    rows = []
    for (phase, regime), entries in groups.items():
        duration_s = sum(step.dt_s for step in entries)
        constraints = sorted(
            {step.controller_active_constraint or "none" for step in entries}
        )
        rows.append(
            {
                "phase": phase,
                "regime": regime,
                "duration_s": duration_s,
                "engine_off_fraction": sum(
                    step.dt_s for step in entries if step.engine_shut_down
                )
                / duration_s,
                "running_fuel_kg": sum(
                    step.fuel_flow_kg_s * step.dt_s for step in entries
                ),
                "restart_fuel_kg": sum(step.restart_fuel_kg for step in entries),
                "battery_charge_kwh": sum(
                    max(-step.battery_bus_kw, 0.0) * step.dt_s / 3600.0
                    for step in entries
                ),
                "battery_discharge_kwh": sum(
                    max(step.battery_bus_kw, 0.0) * step.dt_s / 3600.0
                    for step in entries
                ),
                "minimum_soc": min(step.soc for step in entries),
                "active_constraints": json.dumps(constraints),
            }
        )
    return rows


def write_thermostat_mission_artifacts(
    run: ThermostatReferenceRun, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write the complete JSON report and compact phase/regime CSV."""
    if run.result.log is None:
        raise ValueError("artifact writing requires a recorded log")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "thermostat_mission_reference.json"
    summary_path = directory / "thermostat_mission_phase_regime.csv"
    report_path.write_text(
        json.dumps(summarise_thermostat_mission(run), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rows = _phase_regime_rows(run.result.log)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report_path, summary_path


def main() -> int:
    """Run and persist the single named reference experiment."""
    run = run_reference_thermostat_mission()
    output_dir = Path(__file__).resolve().parents[2] / "deliverables" / "figures"
    report_path, summary_path = write_thermostat_mission_artifacts(run, output_dir)
    report = summarise_thermostat_mission(run)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_path={report_path}")
    print(f"summary_path={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
