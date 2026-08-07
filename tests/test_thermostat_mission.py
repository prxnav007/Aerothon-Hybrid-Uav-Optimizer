"""Full-mission thermostat scheduling, state, and accounting integration."""

from __future__ import annotations

from dataclasses import replace

import pytest

import src.simulation.simulator as simulator_module
from src.analysis.thermostat_mission import (
    REFERENCE_INITIAL_THERMOSTAT_STATE,
    REFERENCE_THERMOSTAT_PARAMETERS,
    build_reference_aircraft,
)
from src.control.fixed_ecms import FixedECMS
from src.control.thermostat import (
    TerminalStrategy,
    ThermostatParameters,
    ThermostatRegime,
    ThermostatState,
)
from src.models.atmosphere import atmosphere
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.mass import MassBreakdown
from src.models.powertrain import SeriesPowertrain
from src.simulation.mission import ps1_mission
from src.simulation.simulator import Aircraft, run_mission


@pytest.fixture()
def aircraft() -> Aircraft:
    masses = MassBreakdown(
        fixed_kg=250.0,
        payload_kg=200.0,
        wing_kg=100.0,
        engine_kg=35.0,
        generator_kg=25.0,
        rectifier_kg=5.0,
        inverter_kg=5.0,
        motor_kg=15.0,
        cabling_cooling_kg=10.0,
        battery_kg=80.0,
        fuel_system_kg=10.0,
        fuel_kg=80.0,
    )
    return Aircraft(
        wing_area_m2=10.0,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        engine=Turboshaft(120.0),
        battery=BatteryPack(40.0),
        powertrain=SeriesPowertrain(),
        masses=masses,
    )


@pytest.fixture()
def mission():
    return ps1_mission(
        cruise_altitude_m=60.0,
        takeoff_duration_s=180.0,
        cruise_duration_s=60.0,
        landing_duration_s=60.0,
        fuel_reserve_kg=1.0,
        descent_landing_fuel_kg=0.5,
        min_usable_fuel_kg=2.0,
        max_mission_time_s=600.0,
    )


def _parameters(**changes) -> ThermostatParameters:
    values = {
        "soc_low": 0.4,
        "soc_high": 0.6,
        "minimum_on_time_s": 60.0,
        "minimum_off_time_s": 60.0,
        "restart_fuel_kg": 0.0,
        "engine_on_power_kw": None,
        "terminal_strategy": TerminalStrategy.CAUSAL,
    }
    values.update(changes)
    return ThermostatParameters(**values)


def _run(aircraft, mission, parameters, state, *, initial_soc=0.7):
    return run_mission(
        aircraft,
        mission,
        thermostat_parameters=parameters,
        initial_thermostat_state=state,
        initial_soc=initial_soc,
        record_log=True,
    )


def test_thermostat_state_persists_between_steps_and_across_a_phase_boundary(
    aircraft, mission
) -> None:
    result = _run(
        aircraft,
        mission,
        _parameters(soc_high=0.99),
        ThermostatState(True, 10.0),
    )
    assert result.log is not None
    takeoff = tuple(step for step in result.log if step.phase == "takeoff")
    first_climb = next(step for step in result.log if step.phase == "climb")
    assert [step.thermostat_elapsed_in_state_s for step in takeoff] == [
        70.0,
        130.0,
        190.0,
    ]
    assert first_climb.thermostat_elapsed_in_state_s == pytest.approx(220.0)
    assert first_climb.thermostat_restart_count == takeoff[-1].thermostat_restart_count


def test_state_is_not_reset_on_the_way_through_loiter_descent_and_landing(
    aircraft,
) -> None:
    complete_mission = ps1_mission(
        cruise_altitude_m=60.0,
        takeoff_duration_s=180.0,
        cruise_duration_s=60.0,
        landing_duration_s=60.0,
        fuel_reserve_kg=70.0,
        descent_landing_fuel_kg=5.0,
        min_usable_fuel_kg=71.0,
        max_mission_time_s=1200.0,
    )
    result = _run(
        aircraft,
        complete_mission,
        _parameters(soc_high=0.99),
        ThermostatState(True, 10.0),
    )
    assert result.mission_complete
    assert result.log is not None
    assert tuple(dict.fromkeys(step.phase for step in result.log)) == (
        "takeoff",
        "climb",
        "cruise",
        "loiter",
        "descent",
        "landing",
    )
    boundaries = tuple(
        (previous, current)
        for previous, current in zip(result.log, result.log[1:])
        if previous.phase != current.phase
    )
    for previous, current in boundaries:
        assert current.thermostat_restart_count >= previous.thermostat_restart_count
        if previous.engine_shut_down == current.engine_shut_down:
            assert current.thermostat_elapsed_in_state_s == pytest.approx(
                previous.thermostat_elapsed_in_state_s + current.dt_s
            )
        else:
            assert current.thermostat_elapsed_in_state_s == pytest.approx(current.dt_s)


def test_ecms_path_is_identical_when_thermostat_is_not_selected(
    aircraft, mission
) -> None:
    first = run_mission(aircraft, mission, FixedECMS(s=5.0), record_log=True)
    second = run_mission(aircraft, mission, FixedECMS(s=5.0), record_log=True)
    assert first == second
    assert first.thermostat_final_state is None
    assert first.log is not None
    assert all(step.controller_regime is None for step in first.log)


def test_thermostat_command_reaches_each_real_plant_model_once(
    monkeypatch, aircraft, mission
) -> None:
    engine_calls = 0
    battery_calls = 0
    original_operate = Turboshaft.operate
    original_step = BatteryPack.step

    def counted_operate(self, commanded_kw, sigma=1.0):
        nonlocal engine_calls
        engine_calls += 1
        return original_operate(self, commanded_kw, sigma)

    def counted_step(self, soc, power_kw, dt_s):
        nonlocal battery_calls
        battery_calls += 1
        return original_step(self, soc, power_kw, dt_s)

    monkeypatch.setattr(Turboshaft, "operate", counted_operate)
    monkeypatch.setattr(BatteryPack, "step", counted_step)
    result = _run(
        aircraft,
        mission,
        _parameters(soc_high=0.99),
        ThermostatState(True, 60.0),
    )
    assert result.log is not None
    assert engine_calls == len(result.log)
    assert battery_calls == len(result.log)
    for step in result.log:
        if step.phase != "takeoff":
            continue
        sigma = float(atmosphere(step.altitude_m).density_ratio)
        expected = original_operate(
            aircraft.engine, step.requested_engine_shaft_kw, sigma
        )
        assert step.engine_shaft_kw == pytest.approx(expected.delivered_kw)
        assert step.bus_from_engine_kw == pytest.approx(
            aircraft.powertrain.bus_power_from_engine(step.engine_shaft_kw)
        )
        assert step.bus_from_engine_kw + step.battery_bus_kw == pytest.approx(
            step.bus_demand_kw, abs=1.0e-9
        )


def test_fuel_and_battery_state_are_integrated_once(aircraft, mission) -> None:
    initial_soc = 0.7
    result = _run(
        aircraft,
        mission,
        _parameters(),
        ThermostatState(True, 60.0),
        initial_soc=initial_soc,
    )
    assert result.log is not None
    integrated_fuel = sum(
        step.fuel_flow_kg_s * step.dt_s + step.restart_fuel_kg
        for step in result.log
    )
    assert result.fuel_used_kg == pytest.approx(integrated_fuel, abs=1.0e-10)
    reproduced_soc = initial_soc
    for step in result.log:
        reproduced_soc = aircraft.battery.step(
            reproduced_soc, step.battery_bus_kw, step.dt_s
        ).soc
        assert step.soc == pytest.approx(reproduced_soc, abs=1.0e-12)
    assert result.final_soc == pytest.approx(reproduced_soc, abs=1.0e-12)


def test_restart_fuel_is_charged_once_per_requested_off_to_on_transition(
    aircraft, mission
) -> None:
    engine = replace(aircraft.engine, restart_fuel_kg=0.1)
    configured = replace(aircraft, engine=engine)
    result = _run(
        configured,
        mission,
        _parameters(restart_fuel_kg=0.1),
        ThermostatState(False, 60.0),
        initial_soc=0.4,
    )
    assert result.log is not None
    restarts = tuple(
        step
        for step in result.log
        if step.thermostat_transitioned and step.requested_engine_on
    )
    charged = tuple(step for step in result.log if step.restart_fuel_kg > 0.0)
    assert restarts
    assert charged == restarts
    assert sum(step.restart_fuel_kg for step in charged) == pytest.approx(
        0.1 * len(restarts)
    )


def test_genuine_engine_off_has_zero_running_fuel(aircraft, mission) -> None:
    result = _run(
        aircraft,
        mission,
        _parameters(),
        ThermostatState(False, 60.0),
        initial_soc=0.8,
    )
    assert result.log is not None
    engine_off = tuple(step for step in result.log if step.engine_shut_down)
    assert engine_off
    assert all(step.engine_shaft_kw == 0.0 for step in engine_off)
    assert all(step.fuel_flow_kg_s == 0.0 for step in engine_off)


def test_hard_off_dwell_blocks_restart_until_the_timer_expires(
    aircraft, mission
) -> None:
    result = _run(
        aircraft,
        mission,
        _parameters(minimum_off_time_s=120.0),
        ThermostatState(False, 0.0),
        initial_soc=0.4,
    )
    assert result.log is not None
    first_on_index = next(
        index for index, step in enumerate(result.log) if not step.engine_shut_down
    )
    assert first_on_index == 2
    assert result.log[0].controller_active_constraint == "hard_off_dwell"
    assert result.log[1].controller_active_constraint == "hard_off_dwell"
    assert all(not step.thermostat_dwell_violation for step in result.log)


def test_high_demand_rejects_shutdown_and_uses_a_real_on_regime(
    aircraft, mission
) -> None:
    constrained = replace(aircraft, battery=BatteryPack(5.0))
    result = _run(
        constrained,
        mission,
        _parameters(),
        ThermostatState(True, 60.0),
        initial_soc=0.8,
    )
    assert result.log is not None
    first = result.log[0]
    assert not first.engine_shut_down
    assert first.controller_regime in {
        ThermostatRegime.CONTINUOUS.value,
        ThermostatRegime.BATTERY_ASSISTED.value,
    }
    assert first.controller_regime_reason in {
        "battery_cannot_carry_full_demand",
        "demand_above_engine_bus_ceiling",
    }


def test_causal_mission_path_passes_no_future_information(
    monkeypatch, aircraft, mission
) -> None:
    calls = 0
    original = simulator_module.schedule_thermostat

    def inspected(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs.get("time_to_go_s") is None
        assert kwargs.get("terminal_energy_target_kwh") is None
        assert kwargs.get("off_dwell_feasible") is None
        return original(*args, **kwargs)

    monkeypatch.setattr(simulator_module, "schedule_thermostat", inspected)
    _run(
        aircraft,
        mission,
        _parameters(),
        ThermostatState(True, 60.0),
    )
    assert calls > 0


def test_hard_dwell_infeasibility_is_reported_instead_of_safety_restart(
    aircraft, mission
) -> None:
    constrained = replace(aircraft, battery=BatteryPack(5.0))
    result = _run(
        constrained,
        mission,
        _parameters(minimum_off_time_s=120.0),
        ThermostatState(False, 0.0),
        initial_soc=0.4,
    )
    assert not result.mission_complete
    assert result.termination_reason == "power_shortfall"
    assert "hard_dwell_infeasible" in result.failure_flags
    assert "controller_infeasible" in result.failure_flags
    assert result.log == ()


def test_threshold_parameter_changes_the_integrated_schedule(aircraft, mission) -> None:
    low_high = _run(
        aircraft,
        mission,
        _parameters(soc_high=0.6),
        ThermostatState(True, 60.0),
        initial_soc=0.7,
    )
    high_high = _run(
        aircraft,
        mission,
        _parameters(soc_high=0.8),
        ThermostatState(True, 60.0),
        initial_soc=0.7,
    )
    assert low_high.log is not None and high_high.log is not None
    assert low_high.log[0].requested_engine_on is False
    assert high_high.log[0].requested_engine_on is True


def test_identical_thermostat_missions_are_deterministic(aircraft, mission) -> None:
    arguments = (
        aircraft,
        mission,
        _parameters(),
        ThermostatState(True, 60.0),
    )
    assert _run(*arguments) == _run(*arguments)


def test_named_reference_configuration_matches_the_declared_experiment() -> None:
    reference = build_reference_aircraft()
    assert reference.masses.total_kg == pytest.approx(1000.0, abs=1.0e-10)
    assert reference.masses.dry_kg == pytest.approx(711.3898016890586)
    assert reference.masses.fuel_kg == pytest.approx(288.6101983109414)
    assert reference.wing_area_m2 == pytest.approx(7.59175537062125)
    assert reference.engine.rated_power_kw == pytest.approx(86.7791369750147)
    assert reference.battery.capacity_kwh == 10.0
    assert REFERENCE_THERMOSTAT_PARAMETERS.soc_low == 0.4
    assert REFERENCE_THERMOSTAT_PARAMETERS.soc_high == 0.6
    assert REFERENCE_INITIAL_THERMOSTAT_STATE == ThermostatState(True, 60.0)


def test_ambiguous_or_noncausal_configuration_is_rejected(aircraft, mission) -> None:
    state = ThermostatState(True, 60.0)
    with pytest.raises(ValueError, match="exactly one"):
        run_mission(
            aircraft,
            mission,
            FixedECMS(s=5.0),
            thermostat_parameters=_parameters(),
            initial_thermostat_state=state,
        )
    with pytest.raises(ValueError, match="causal"):
        run_mission(
            aircraft,
            mission,
            thermostat_parameters=_parameters(
                terminal_strategy=TerminalStrategy.HORIZON_AWARE
            ),
            initial_thermostat_state=state,
        )
