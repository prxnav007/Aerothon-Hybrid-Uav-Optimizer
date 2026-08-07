"""Single-candidate fitness construction, auditing, and reference gate."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.analysis.thermostat_mission import (
    REFERENCE_INITIAL_THERMOSTAT_STATE,
    REFERENCE_THERMOSTAT_PARAMETERS,
    build_reference_aircraft,
)
from src.models.battery import BatteryPack
from src.optimization.chromosome import (
    NormalizedChromosome,
    PhysicalGeneBound,
    PlantThermostatDesignSpace,
    decode_chromosome,
    practical_thermostat_seed,
)
from src.optimization.feasibility import evaluate_static_feasibility
from src.optimization.fitness import (
    CandidateMissionInfeasibleError,
    DYNAMIC_CONSTRAINT_NAMES,
    FitnessScenario,
    construct_mission_inputs,
    evaluate_fitness,
    evaluation_identity,
)
from src.simulation.simulator import MissionResult, TimeStep
from src.control.thermostat import ThermostatState

RUN_REFERENCE_GATE = os.environ.get("AEROTHON_RUN_REFERENCE_FITNESS_GATE") == "1"


@pytest.fixture(scope="module")
def bounds() -> PlantThermostatDesignSpace:
    return PlantThermostatDesignSpace.from_battery(BatteryPack(10.0))


@pytest.fixture(scope="module")
def scenario() -> FitnessScenario:
    return FitnessScenario.nominal()


@pytest.fixture(scope="module")
def reference_chromosome(bounds) -> NormalizedChromosome:
    return practical_thermostat_seed(bounds=bounds)


def _step(
    phase: str,
    time_s: float,
    dt_s: float,
    fuel_kg: float,
    *,
    soc: float = 0.5,
    engine_off: bool = False,
    transitioned: bool = False,
    requested_on: bool = True,
    restart_fuel_kg: float = 0.0,
    bus_residual_kw: float = 0.0,
    controller_feasible: bool = True,
    plant_feasible: bool = True,
    dwell_violation: bool = False,
    battery_bus_kw: float = 0.0,
    battery_active_limit: str = "none",
) -> TimeStep:
    return TimeStep(
        time_s=time_s,
        phase=phase,
        altitude_m=0.0,
        speed_mps=50.0,
        weight_n=9810.0,
        density_kg_m3=1.225,
        lift_coefficient=0.5,
        drag_n=500.0,
        shaft_power_kw=0.0,
        bus_demand_kw=0.0,
        neutral_s=math.nan,
        switching_s=math.nan,
        equivalence_factor=math.nan,
        engine_shaft_kw=0.0 if engine_off else 40.0,
        engine_load_fraction=0.0 if engine_off else 0.5,
        sfc_kg_kwh=math.inf if engine_off else 0.5,
        fuel_flow_kg_s=0.0,
        restart_fuel_kg=restart_fuel_kg,
        engine_shut_down=engine_off,
        battery_bus_kw=battery_bus_kw,
        soc=soc,
        fuel_remaining_kg=fuel_kg,
        system_efficiency=0.0,
        power_off=False,
        power_limited=False,
        dt_s=dt_s,
        bus_from_engine_kw=bus_residual_kw,
        battery_internal_kw=0.0,
        battery_ohmic_loss_kw=0.0,
        battery_stored_energy_change_kwh=0.0,
        thrust_power_kw=0.0,
        engine_thermal_loss_kw=0.0,
        source_losses_kw=0.0,
        demand_losses_kw=0.0,
        propeller_losses_kw=0.0,
        requested_engine_on=requested_on,
        requested_engine_shaft_kw=0.0 if not requested_on else 40.0,
        controller_regime="cycling",
        controller_regime_reason="mock",
        controller_active_constraint="mock",
        thermostat_elapsed_in_state_s=dt_s,
        thermostat_restart_count=int(transitioned and requested_on),
        thermostat_transitioned=transitioned,
        thermostat_dwell_violation=dwell_violation,
        battery_active_limit=battery_active_limit,
        controller_feasible=controller_feasible,
        plant_feasible=plant_feasible,
    )


def _mock_result(
    aircraft,
    mission,
    *,
    mission_complete: bool = True,
    include_descent: bool = True,
    include_landing: bool = True,
    final_fuel_kg: float | None = None,
    final_soc: float = 0.5,
    minimum_soc: float = 0.5,
    failure_flags: tuple[str, ...] = (),
    dwell_violation: bool = False,
    controller_feasible: bool = True,
    plant_feasible: bool = True,
    bus_residual_kw: float = 0.0,
) -> MissionResult:
    durations = {
        "takeoff": 120.0,
        "climb": 1500.0,
        "cruise": 3600.0,
        "loiter": 1000.0,
        "descent": 1000.0 if include_descent else 0.0,
        "landing": 120.0 if include_landing else 0.0,
    }
    fuel = aircraft.masses.fuel_kg if final_fuel_kg is None else final_fuel_kg
    names = ["takeoff", "climb", "cruise", "loiter"]
    if include_descent:
        names.append("descent")
    if include_landing:
        names.append("landing")
    time_s = 0.0
    steps = []
    for name in names:
        time_s += durations[name]
        steps.append(
            _step(
                name,
                time_s,
                durations[name],
                fuel,
                soc=final_soc,
                dwell_violation=dwell_violation and name == names[-1],
                controller_feasible=controller_feasible,
                plant_feasible=plant_feasible,
                bus_residual_kw=bus_residual_kw,
            )
        )
    return MissionResult(
        endurance_s=time_s,
        mission_complete=mission_complete,
        termination_reason="fuel_reserve" if mission_complete else "mock_incomplete",
        phase_durations_s=durations,
        fuel_used_kg=aircraft.masses.fuel_kg - fuel,
        fuel_remaining_kg=fuel,
        final_soc=final_soc,
        min_soc=minimum_soc,
        peak_bus_kw=100.0,
        peak_engine_kw=80.0,
        mean_system_efficiency=0.2,
        failure_flags=failure_flags,
        log=tuple(steps),
        thermostat_final_state=ThermostatState(True, 60.0, 0, False),
    )


def _runner(**changes):
    def run(aircraft, mission, **kwargs):
        return _mock_result(aircraft, mission, **changes)

    return run


def _constraint(result, name: str):
    return next(record for record in result.dynamic_constraints if record.name == name)


def _constructed(chromosome, bounds, scenario):
    decoded = decode_chromosome(chromosome, bounds=bounds)
    static = evaluate_static_feasibility(decoded, scenario=scenario.static_scenario)
    return static, construct_mission_inputs(static.resolved_design, scenario=scenario)


def test_reference_chromosome_is_statically_feasible_before_fitness(
    bounds, scenario, reference_chromosome
) -> None:
    static, _ = _constructed(reference_chromosome, bounds, scenario)
    assert static.is_feasible


def test_static_infeasibility_skips_the_mission_runner(bounds, scenario) -> None:
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("mission runner must not be called")

    result = evaluate_fitness(
        NormalizedChromosome((0.0,) * 6),
        bounds=bounds,
        scenario=scenario,
        mission_runner=forbidden,
    )
    assert not result.static_feasible
    assert not result.run_mission_called
    assert result.failure_category == "static_infeasible"
    assert result.objective_loiter_seconds is None
    assert result.dynamic_constraint_names == DYNAMIC_CONSTRAINT_NAMES
    assert {record.name for record in result.static_constraints} >= {
        "minimum_usable_fuel",
        "service_ceiling_10km",
    }
    assert calls == 0


def test_statically_feasible_evaluation_calls_the_runner_exactly_once(
    bounds, scenario, reference_chromosome
) -> None:
    calls = 0

    def counting(aircraft, mission, **kwargs):
        nonlocal calls
        calls += 1
        return _mock_result(aircraft, mission)

    result = evaluate_fitness(
        reference_chromosome,
        bounds=bounds,
        scenario=scenario,
        mission_runner=counting,
    )
    assert calls == 1
    assert result.run_mission_called


def test_constructed_reference_plant_preserves_resolved_cd0_and_mtow_closure(
    bounds, scenario, reference_chromosome
) -> None:
    static, inputs = _constructed(reference_chromosome, bounds, scenario)
    assert inputs.aircraft.cd0 == pytest.approx(0.028, abs=1.0e-15)
    assert inputs.aircraft.cd0 == static.resolved_design.wing.cd0
    assert inputs.aircraft.masses.fuel_kg == static.resolved_design.masses.fuel_kg
    assert inputs.aircraft.masses.dry_kg + inputs.aircraft.masses.fuel_kg == pytest.approx(
        1000.0, abs=1.0e-12
    )


def test_reference_construction_matches_the_authoritative_comparison_inputs(
    bounds, scenario, reference_chromosome
) -> None:
    _, inputs = _constructed(reference_chromosome, bounds, scenario)
    reference = build_reference_aircraft()
    expected_aircraft = replace(
        reference,
        engine=replace(reference.engine, restart_fuel_kg=0.1),
    )
    assert inputs.aircraft == expected_aircraft
    assert inputs.mission == scenario.mission
    assert inputs.thermostat_parameters == replace(
        REFERENCE_THERMOSTAT_PARAMETERS,
        soc_low=inputs.thermostat_parameters.soc_low,
        soc_high=inputs.thermostat_parameters.soc_high,
        restart_fuel_kg=0.1,
    )
    assert inputs.thermostat_parameters.soc_low == pytest.approx(0.225, abs=1.0e-15)
    assert inputs.thermostat_parameters.soc_high == pytest.approx(0.350, abs=1.0e-15)
    assert inputs.initial_thermostat_state == REFERENCE_INITIAL_THERMOSTAT_STATE


def test_nonreference_plant_uses_area_dependent_resolved_cd0(
    bounds, scenario, reference_chromosome
) -> None:
    genes = list(reference_chromosome.genes)
    genes[0] += 0.05
    static, inputs = _constructed(NormalizedChromosome(tuple(genes)), bounds, scenario)
    assert inputs.aircraft.cd0 == static.resolved_design.wing.cd0
    assert inputs.aircraft.cd0 != pytest.approx(0.028)


def test_all_six_genes_reach_their_constructed_mission_inputs_without_missions(
    bounds, scenario, reference_chromosome
) -> None:
    base_static, base = _constructed(reference_chromosome, bounds, scenario)
    variants = []
    for index in range(6):
        genes = list(reference_chromosome.genes)
        genes[index] += 0.01
        variants.append(_constructed(NormalizedChromosome(tuple(genes)), bounds, scenario))
    wing_static, wing = variants[0]
    aspect_static, aspect = variants[1]
    engine_static, engine = variants[2]
    battery_static, battery = variants[3]
    _, low = variants[4]
    _, gap = variants[5]
    assert wing.aircraft.wing_area_m2 != base.aircraft.wing_area_m2
    assert wing.aircraft.cd0 != base.aircraft.cd0
    assert wing.aircraft.masses.wing_kg != base.aircraft.masses.wing_kg
    assert wing_static.resolved_design.wing.wing_loading_pa != base_static.resolved_design.wing.wing_loading_pa
    assert aspect.aircraft.aspect_ratio != base.aircraft.aspect_ratio
    assert aspect_static.resolved_design.wing.induced_drag_factor != base_static.resolved_design.wing.induced_drag_factor
    assert aspect.aircraft.masses.wing_kg != base.aircraft.masses.wing_kg
    assert engine.aircraft.engine.rated_power_kw != base.aircraft.engine.rated_power_kw
    assert engine.aircraft.masses.engine_kg != base.aircraft.masses.engine_kg
    assert engine.aircraft.masses.propulsion_total_kg != base.aircraft.masses.propulsion_total_kg
    assert battery.aircraft.battery.capacity_kwh != base.aircraft.battery.capacity_kwh
    assert battery.aircraft.masses.battery_kg != base.aircraft.masses.battery_kg
    assert battery.aircraft.battery.max_discharge_kw != base.aircraft.battery.max_discharge_kw
    assert low.thermostat_parameters.soc_low != base.thermostat_parameters.soc_low
    assert gap.thermostat_parameters.soc_low == base.thermostat_parameters.soc_low
    assert gap.thermostat_parameters.soc_high != base.thermostat_parameters.soc_high
    for _, inputs in variants:
        assert inputs.aircraft.masses.total_kg == pytest.approx(1000.0)
        assert inputs.aircraft.masses.payload_kg == 200.0
        assert inputs.mission.phase_by_name("cruise").target_altitude_m == 3000.0
        assert inputs.aircraft.cl_max == 1.5
        assert inputs.aircraft.oswald_efficiency == 0.78
        assert inputs.thermostat_parameters.minimum_on_time_s == 60.0
        assert inputs.thermostat_parameters.minimum_off_time_s == 60.0
        assert inputs.thermostat_parameters.restart_fuel_kg == 0.1


def test_fitness_scenario_identity_distinguishes_restart_costs(scenario) -> None:
    zero_cost = replace(scenario, restart_fuel_kg=0.0)
    assert zero_cost.identity != scenario.identity


def test_evaluation_key_is_deterministic_across_processes(
    bounds, scenario, reference_chromosome
) -> None:
    expected = evaluation_identity(reference_chromosome, bounds=bounds, scenario=scenario)
    code = (
        "from src.models.battery import BatteryPack; "
        "from src.optimization.chromosome import PlantThermostatDesignSpace, practical_thermostat_seed; "
        "from src.optimization.fitness import FitnessScenario, evaluation_identity; "
        "b=PlantThermostatDesignSpace.from_battery(BatteryPack(10.0)); "
        "c=practical_thermostat_seed(bounds=b); s=FitnessScenario.nominal(); "
        "print(evaluation_identity(c,bounds=b,scenario=s))"
    )
    actual = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert actual == expected


def test_evaluation_key_changes_with_chromosome_bounds_and_scenario(
    bounds, scenario, reference_chromosome
) -> None:
    base = evaluation_identity(reference_chromosome, bounds=bounds, scenario=scenario)
    genes = list(reference_chromosome.genes)
    genes[0] += 0.01
    changed_chromosome = NormalizedChromosome(tuple(genes))
    changed_bounds = replace(
        bounds,
        wing_area=PhysicalGeneBound("wing_area", 5.5, 16.0, "m^2"),
    )
    changed_scenario = replace(scenario, restart_fuel_kg=0.0)
    assert evaluation_identity(changed_chromosome, bounds=bounds, scenario=scenario) != base
    assert evaluation_identity(reference_chromosome, bounds=changed_bounds, scenario=scenario) != base
    assert evaluation_identity(reference_chromosome, bounds=bounds, scenario=changed_scenario) != base


def test_valid_mocked_mission_uses_loiter_alone_as_the_objective(
    bounds, scenario, reference_chromosome
) -> None:
    result = evaluate_fitness(
        reference_chromosome,
        bounds=bounds,
        scenario=scenario,
        mission_runner=_runner(),
    )
    assert result.dynamically_feasible
    assert result.objective_loiter_seconds == 1000.0
    assert result.total_mission_seconds == 7340.0
    assert result.dynamic_constraint_names == DYNAMIC_CONSTRAINT_NAMES
    assert result.total_normalized_dynamic_violation == 0.0


@pytest.mark.parametrize(
    "changes",
    (
        {"mission_complete": False, "include_descent": False, "include_landing": False},
        {"mission_complete": False, "include_landing": False},
    ),
)
def test_incomplete_descent_or_landing_is_infeasible_regardless_of_elapsed_time(
    bounds, scenario, reference_chromosome, changes
) -> None:
    result = evaluate_fitness(
        reference_chromosome,
        bounds=bounds,
        scenario=scenario,
        mission_runner=_runner(**changes),
    )
    assert not result.dynamically_feasible
    assert result.objective_loiter_seconds is None
    assert not result.validity.completed_landing


def test_reserve_and_soc_floor_shortfalls_are_explicit_dynamic_violations(
    bounds, scenario, reference_chromosome
) -> None:
    reserve = evaluate_fitness(
        reference_chromosome, bounds=bounds, scenario=scenario,
        mission_runner=_runner(final_fuel_kg=4.0),
    )
    soc = evaluate_fitness(
        reference_chromosome, bounds=bounds, scenario=scenario,
        mission_runner=_runner(final_soc=0.04, minimum_soc=0.04),
    )
    assert not _constraint(reserve, "fuel_reserve").satisfied
    assert reserve.validity.reserve_shortfall_kg == 1.0
    assert not _constraint(soc, "soc_floor").satisfied
    assert soc.validity.soc_floor_violation == pytest.approx(0.01)


def test_power_controller_and_hard_dwell_failures_are_infeasible(
    bounds, scenario, reference_chromosome
) -> None:
    power = evaluate_fitness(
        reference_chromosome, bounds=bounds, scenario=scenario,
        mission_runner=_runner(failure_flags=("power_shortfall",), plant_feasible=False),
    )
    controller = evaluate_fitness(
        reference_chromosome, bounds=bounds, scenario=scenario,
        mission_runner=_runner(
            failure_flags=("controller_infeasible",), controller_feasible=False
        ),
    )
    dwell = evaluate_fitness(
        reference_chromosome, bounds=bounds, scenario=scenario,
        mission_runner=_runner(dwell_violation=True),
    )
    assert not _constraint(power, "plant_feasible").satisfied
    assert not _constraint(controller, "controller_feasible").satisfied
    assert not _constraint(dwell, "hard_dwell").satisfied


def test_active_power_limits_remain_diagnostics_when_the_bus_is_balanced(
    bounds, scenario, reference_chromosome
) -> None:
    result = evaluate_fitness(
        reference_chromosome, bounds=bounds, scenario=scenario,
        mission_runner=_runner(failure_flags=("engine_power_limited", "battery_rate_limited")),
    )
    assert result.dynamically_feasible


def test_ledger_tolerances_are_not_reused_as_normalization_scales(
    bounds, scenario, reference_chromosome
) -> None:
    residual = 2.0 * scenario.power_balance_tolerance_kw
    result = evaluate_fitness(
        reference_chromosome, bounds=bounds, scenario=scenario,
        mission_runner=_runner(bus_residual_kw=residual),
    )
    record = _constraint(result, "bus_power_ledger")
    assert record.required_or_allowed == scenario.power_balance_tolerance_kw
    assert record.normalization_scale == 100.0
    assert record.normalization_scale != record.required_or_allowed
    assert record.raw_violation == pytest.approx(scenario.power_balance_tolerance_kw)


def test_known_candidate_physical_exception_becomes_structured_data(
    bounds, scenario, reference_chromosome
) -> None:
    def physically_infeasible(*args, **kwargs):
        raise CandidateMissionInfeasibleError(
            "candidate_numerical_boundary", "documented candidate failure",
            normalized_violation=0.25,
        )

    result = evaluate_fitness(
        reference_chromosome, bounds=bounds, scenario=scenario,
        mission_runner=physically_infeasible,
    )
    assert result.run_mission_called
    assert not result.dynamically_feasible
    assert result.failure_category == "candidate_numerical_boundary"
    assert result.dynamic_constraint_names == DYNAMIC_CONSTRAINT_NAMES
    assert result.dynamic_constraints[0].name == "candidate_mission_exception"
    assert result.total_normalized_dynamic_violation == 0.25


def test_unexpected_mission_runner_exception_propagates(
    bounds, scenario, reference_chromosome
) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("programming defect")

    with pytest.raises(RuntimeError, match="programming defect"):
        evaluate_fitness(
            reference_chromosome, bounds=bounds, scenario=scenario,
            mission_runner=broken,
        )


def test_nonfinite_mission_output_is_rejected(
    bounds, scenario, reference_chromosome
) -> None:
    def nonfinite(aircraft, mission, **kwargs):
        return replace(_mock_result(aircraft, mission), endurance_s=math.nan)

    with pytest.raises(ValueError, match="endurance_s must be finite"):
        evaluate_fitness(
            reference_chromosome, bounds=bounds, scenario=scenario,
            mission_runner=nonfinite,
        )


def test_repeated_mocked_evaluations_construct_fresh_plant_and_controller_state(
    bounds, scenario, reference_chromosome
) -> None:
    captured = []

    def capture(aircraft, mission, **kwargs):
        captured.append((aircraft, aircraft.engine, aircraft.battery, kwargs["initial_thermostat_state"]))
        return _mock_result(aircraft, mission)

    evaluate_fitness(reference_chromosome, bounds=bounds, scenario=scenario, mission_runner=capture)
    evaluate_fitness(reference_chromosome, bounds=bounds, scenario=scenario, mission_runner=capture)
    assert all(left is not right for left, right in zip(captured[0], captured[1]))


@pytest.mark.skipif(
    not RUN_REFERENCE_GATE,
    reason="set AEROTHON_RUN_REFERENCE_FITNESS_GATE=1 for the one intentional real mission",
)
def test_authoritative_practical_reference_mission_reproduces_the_stored_artifact(
    bounds, scenario, reference_chromosome
) -> None:
    result = evaluate_fitness(reference_chromosome, bounds=bounds, scenario=scenario)
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "deliverables"
        / "figures"
        / "controller_restart_retuning_gate.json"
    )
    target = json.loads(artifact_path.read_text(encoding="utf-8"))["best_by_family"]["thermostat"]
    assert result.static_feasible and result.dynamically_feasible
    assert result.total_mission_seconds == pytest.approx(target["total_time_s"], abs=1.0e-9)
    assert result.objective_loiter_seconds == pytest.approx(target["loiter_time_s"], abs=1.0e-9)
    assert result.resources.initial_fuel_kg == pytest.approx(target["initial_fuel_kg"], abs=1.0e-12)
    assert result.resources.final_fuel_kg == pytest.approx(target["final_fuel_kg"], abs=1.0e-10)
    assert result.resources.final_soc == pytest.approx(target["final_soc"], abs=1.0e-12)
    assert result.resources.minimum_soc == pytest.approx(target["minimum_soc"], abs=1.0e-12)
    assert result.controller_behavior.restart_count == target["restart_count"]
    assert result.controller_behavior.overall_engine_off_fraction == pytest.approx(
        target["overall_engine_off_fraction"], abs=1.0e-12
    )
    assert result.controller_behavior.loiter_engine_off_fraction == pytest.approx(
        target["loiter_engine_off_fraction"], abs=1.0e-12
    )
    assert result.validity.termination_reason == target["termination_reason"]
    assert result.validity.completed_descent and result.validity.completed_landing
    assert result.validity.maximum_bus_power_residual_kw == pytest.approx(
        target["maximum_power_residual_kw"], abs=1.0e-12
    )
    assert result.validity.fuel_ledger_residual_kg == pytest.approx(
        target["fuel_ledger_residual_kg"], abs=1.0e-12
    )
    assert result.validity.battery_energy_ledger_residual_kwh == pytest.approx(
        target["energy_ledger_residual_kwh"], abs=1.0e-10
    )
    assert result.validity.battery_integration_residual_kwh == pytest.approx(
        target["battery_integration_residual_kwh"], abs=1.0e-10
    )
    assert result.validity.discrete_energy_ledger_residual_fraction == pytest.approx(
        target["discrete_energy_residual_fraction"], abs=1.0e-14
    )
