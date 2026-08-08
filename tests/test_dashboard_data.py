"""Dashboard scenario, telemetry, export, and completed-GA artifact tests."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pandas as pd
import pytest

from src.control.thermostat import ThermostatParameters, ThermostatState, thermostat_step
from src.dashboard.data import (
    ARCHITECTURE_LABEL,
    DashboardDataError,
    PHASE_ORDER,
    build_telemetry_records,
    ga_candidates_dataframe,
    load_dashboard_scenarios,
    load_ga_artifacts,
    telemetry_csv_bytes,
)
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.powertrain import SeriesPowertrain
from src.optimization.chromosome import decode_chromosome
from src.optimization.feasibility import evaluate_static_feasibility
from src.optimization.fitness import construct_mission_inputs
from src.simulation.simulator import MissionResult, TimeStep


@pytest.fixture(scope="module")
def scenarios():
    return load_dashboard_scenarios()


@pytest.fixture(scope="module")
def practical_aircraft(scenarios):
    scenario = scenarios["practical_reference"]
    design = decode_chromosome(scenario.chromosome, bounds=scenario.bounds)
    static = evaluate_static_feasibility(
        design, scenario=scenario.fitness_scenario.static_scenario
    )
    return construct_mission_inputs(
        static.resolved_design, scenario=scenario.fitness_scenario
    ).aircraft


def _step(
    aircraft,
    phase: str,
    time_s: float,
    dt_s: float,
    fuel_kg: float,
    *,
    battery_kw: float = 4.0,
    soc: float = 0.8,
    restart: bool = False,
    optional_telemetry: bool = True,
) -> TimeStep:
    engine_kw = 40.0
    bus_engine = float(aircraft.powertrain.bus_power_from_engine(engine_kw))
    bus_demand = bus_engine + battery_kw
    shaft_kw = float(aircraft.powertrain.shaft_power_from_bus(bus_demand))
    current = 12.0 if battery_kw > 0.0 else -8.0
    internal = battery_kw + abs(battery_kw) * 0.01 if battery_kw > 0.0 else battery_kw * 0.99
    return TimeStep(
        time_s=time_s,
        phase=phase,
        altitude_m=3000.0 if phase not in ("takeoff", "landing") else 0.0,
        speed_mps=65.0,
        weight_n=9806.65,
        density_kg_m3=0.909,
        lift_coefficient=0.6,
        drag_n=500.0,
        shaft_power_kw=shaft_kw,
        bus_demand_kw=bus_demand,
        neutral_s=math.nan,
        switching_s=math.nan,
        equivalence_factor=math.nan,
        engine_shaft_kw=engine_kw,
        engine_load_fraction=0.5,
        sfc_kg_kwh=0.5,
        fuel_flow_kg_s=0.01,
        restart_fuel_kg=0.1 if restart else 0.0,
        engine_shut_down=False,
        battery_bus_kw=battery_kw,
        soc=soc,
        fuel_remaining_kg=fuel_kg,
        system_efficiency=0.2,
        power_off=False,
        power_limited=False,
        dt_s=dt_s,
        bus_from_engine_kw=bus_engine,
        battery_internal_kw=internal,
        battery_ohmic_loss_kw=abs(internal - battery_kw),
        battery_stored_energy_change_kwh=-internal * dt_s / 3600.0,
        thrust_power_kw=20.0,
        engine_thermal_loss_kw=100.0,
        source_losses_kw=engine_kw - bus_engine,
        demand_losses_kw=bus_demand - shaft_kw,
        propeller_losses_kw=shaft_kw - 20.0,
        requested_engine_on=True,
        requested_engine_shaft_kw=engine_kw,
        controller_regime="cycling",
        controller_regime_reason="test",
        controller_active_constraint="test",
        thermostat_elapsed_in_state_s=60.0,
        thermostat_restart_count=int(restart),
        thermostat_transitioned=restart,
        battery_active_limit="none",
        engine_available_kw=80.0 if optional_telemetry else None,
        engine_thermal_efficiency=0.25 if optional_telemetry else None,
        battery_current_a=current if optional_telemetry else None,
        battery_open_circuit_voltage_v=380.0 if optional_telemetry else None,
        battery_terminal_voltage_v=(379.0 if battery_kw > 0 else 381.0)
        if optional_telemetry
        else None,
        battery_constraint_terminal_voltage_v=379.0 if optional_telemetry else None,
    )


def _recorded_result(aircraft, *, optional_telemetry: bool = True) -> MissionResult:
    steps = []
    time_s = 0.0
    fuel = 20.0
    durations = {}
    for index, phase in enumerate(PHASE_ORDER):
        dt_s = 5.0 if phase == "landing" else 15.0
        time_s += dt_s
        fuel -= 0.01 * dt_s + (0.1 if phase == "cruise" else 0.0)
        durations[phase] = dt_s
        steps.append(
            _step(
                aircraft,
                phase,
                time_s,
                dt_s,
                fuel,
                battery_kw=-3.0 if phase == "loiter" else 4.0,
                soc=0.9 - 0.05 * index,
                restart=phase == "cruise",
                optional_telemetry=optional_telemetry,
            )
        )
    return MissionResult(
        endurance_s=time_s,
        mission_complete=True,
        termination_reason="fuel_reserve",
        phase_durations_s=durations,
        fuel_used_kg=20.0 - fuel,
        fuel_remaining_kg=fuel,
        final_soc=steps[-1].soc,
        min_soc=steps[-1].soc,
        peak_bus_kw=max(step.bus_demand_kw for step in steps),
        peak_engine_kw=40.0,
        mean_system_efficiency=0.2,
        failure_flags=(),
        log=tuple(steps),
        thermostat_final_state=ThermostatState(True, 60.0, 1, False),
    )


def test_practical_reference_loader_preserves_every_exact_input(scenarios) -> None:
    design = scenarios["practical_reference"].decoded_design
    assert design == {
        "wing_area_m2": 7.59175537062125,
        "aspect_ratio": 16.0,
        "engine_rating_kw": 86.7791369750147,
        "battery_capacity_kwh": 10.0,
        "soc_low": pytest.approx(0.225),
        "soc_high": pytest.approx(0.350),
    }
    assert scenarios["practical_reference"].fitness_scenario.timestep_s == 15.0


def test_ga_design_is_loaded_at_full_precision_from_best_found(scenarios) -> None:
    design = scenarios["ga_selected"].decoded_design
    assert design["wing_area_m2"] == 9.022737420331685
    assert design["aspect_ratio"] == 22.644481390865387
    assert design["engine_rating_kw"] == 83.40205998293928
    assert design["battery_capacity_kwh"] == 8.343141144862464
    assert design["soc_low"] == 0.20841628367847814
    assert design["soc_high"] == 0.6274222947688504


def test_scenario_cache_identity_is_deterministic_and_design_specific(scenarios) -> None:
    repeated = load_dashboard_scenarios()
    assert repeated["practical_reference"].evaluation_key == scenarios["practical_reference"].evaluation_key
    assert repeated["ga_selected"].evaluation_key == scenarios["ga_selected"].evaluation_key
    assert scenarios["practical_reference"].evaluation_key != scenarios["ga_selected"].evaluation_key
    assert repeated["practical_reference"] is not scenarios["practical_reference"]


def test_telemetry_keeps_15_second_steps_and_a_short_terminal_step(
    scenarios, practical_aircraft
) -> None:
    result = _recorded_result(practical_aircraft)
    records = build_telemetry_records(
        result, practical_aircraft, scenarios["practical_reference"]
    )
    assert [record["dt_s"] for record in records[:-1]] == [15.0] * 5
    assert records[-1]["dt_s"] == 5.0
    assert [record["phase"] for record in records] == list(PHASE_ORDER)
    assert all(left < right for left, right in zip(
        [record["time_s"] for record in records],
        [record["time_s"] for record in records][1:],
    ))


def test_fuel_restart_and_power_domain_ledgers_remain_authoritative(
    scenarios, practical_aircraft
) -> None:
    result = _recorded_result(practical_aircraft)
    records = build_telemetry_records(
        result, practical_aircraft, scenarios["practical_reference"]
    )
    assert records[-1]["cumulative_running_fuel_kg"] == pytest.approx(0.8)
    assert records[-1]["cumulative_restart_fuel_kg"] == pytest.approx(0.1)
    assert sum(record["restart_event"] for record in records) == 1
    for record in records:
        assert record["rectifier_bus_output_kw"] == pytest.approx(record["bus_from_engine_kw"])
        assert record["motor_shaft_output_kw"] == pytest.approx(record["shaft_power_kw"])
        assert record["bus_power_residual_kw"] == pytest.approx(0.0, abs=1.0e-12)


def test_battery_sign_and_efficiency_are_direction_aware(
    scenarios, practical_aircraft
) -> None:
    records = build_telemetry_records(
        _recorded_result(practical_aircraft),
        practical_aircraft,
        scenarios["practical_reference"],
    )
    discharge = records[0]
    charge = next(record for record in records if record["battery_bus_kw"] < 0.0)
    assert discharge["battery_current_a"] > 0.0
    assert charge["battery_current_a"] < 0.0
    assert 0.0 < discharge["battery_terminal_efficiency"] < 1.0
    assert 0.0 < charge["battery_terminal_efficiency"] < 1.0


def test_unavailable_optional_telemetry_remains_missing_instead_of_zero(
    scenarios, practical_aircraft
) -> None:
    records = build_telemetry_records(
        _recorded_result(practical_aircraft, optional_telemetry=False),
        practical_aircraft,
        scenarios["practical_reference"],
    )
    assert records[0]["battery_current_a"] is None
    assert records[0]["battery_terminal_voltage_v"] is None
    assert records[0]["engine_available_kw"] is None


def test_telemetry_csv_is_exactly_the_chart_record_table(
    scenarios, practical_aircraft
) -> None:
    from src.dashboard.data import ValidationBundle

    result = _recorded_result(practical_aircraft)
    records = build_telemetry_records(
        result, practical_aircraft, scenarios["practical_reference"]
    )
    bundle = ValidationBundle(scenarios["practical_reference"], None, result, records)  # type: ignore[arg-type]
    exported = pd.read_csv(pd.io.common.BytesIO(telemetry_csv_bytes(bundle)))
    expected = pd.DataFrame.from_records(records)
    pd.testing.assert_frame_equal(exported, expected, check_dtype=False)


def test_completed_ga_artifacts_load_without_invoking_the_ga(monkeypatch) -> None:
    import src.optimization.ga_runner as runner

    monkeypatch.setattr(
        runner, "run_production_ga",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GA ran")),
    )
    artifacts = load_ga_artifacts()
    assert len(artifacts.history_rows) == 40
    assert len(artifacts.candidate_rows) == 2384
    assert artifacts.best_found["run"]["statistics"]["mission_calls"] == 1809


def test_ga_candidates_distinguish_all_three_feasibility_states() -> None:
    candidates = ga_candidates_dataframe(load_ga_artifacts())
    assert set(candidates["feasibility_status"]) == {
        "Feasible", "Static infeasible", "Dynamic infeasible"
    }
    assert candidates.loc[candidates["feasibility_status"] == "Feasible", "objective_loiter_seconds"].notna().all()


def test_malformed_best_found_is_reported_without_fallback(tmp_path) -> None:
    path = tmp_path / "best_found.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(DashboardDataError, match="chromosome schema"):
        load_dashboard_scenarios(path)


def test_series_hybrid_identity_never_implies_parallel_propulsion() -> None:
    assert "Engine -> Generator -> Electrical bus -> Motor -> Propeller" in ARCHITECTURE_LABEL
    assert "battery connects" in ARCHITECTURE_LABEL


def test_sixty_second_dwell_survives_four_completed_fifteen_second_updates() -> None:
    parameters = ThermostatParameters(
        soc_low=0.4,
        soc_high=0.6,
        minimum_on_time_s=60.0,
        minimum_off_time_s=60.0,
        restart_fuel_kg=0.1,
        engine_on_power_kw=None,
        terminal_strategy="causal",
    )
    engine, battery, powertrain = Turboshaft(100.0, restart_fuel_kg=0.1), BatteryPack(40.0), SeriesPowertrain()
    state = ThermostatState(False, 0.0)
    decisions = []
    for _ in range(4):
        decision = thermostat_step(
            parameters, state, demand_bus_kw=20.0, soc=0.3, sigma=1.0,
            dt_s=15.0, engine=engine, battery=battery, powertrain=powertrain,
        )
        decisions.append(decision)
        state = decision.next_state
    assert all(decision.engine_off for decision in decisions)
    assert state.elapsed_in_state_s == 60.0
    restart = thermostat_step(
        parameters, state, demand_bus_kw=20.0, soc=0.3, sigma=1.0,
        dt_s=15.0, engine=engine, battery=battery, powertrain=powertrain,
    )
    assert not restart.engine_off
    assert restart.next_state.restart_count == 1

