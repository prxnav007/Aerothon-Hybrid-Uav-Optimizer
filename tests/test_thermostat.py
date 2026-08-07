"""Pure thermostat transition, dwell, regime, and preview tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.control.thermostat import (
    TerminalStrategy,
    ThermostatParameters,
    ThermostatRegime,
    ThermostatState,
    select_engine_on_power,
    thermostat_step,
)
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.powertrain import SeriesPowertrain


@pytest.fixture()
def components():
    return Turboshaft(100.0), BatteryPack(40.0), SeriesPowertrain()


@pytest.fixture()
def parameters() -> ThermostatParameters:
    return ThermostatParameters(
        soc_low=0.4,
        soc_high=0.6,
        minimum_on_time_s=120.0,
        minimum_off_time_s=120.0,
        restart_fuel_kg=0.1,
        engine_on_power_kw=None,
        terminal_strategy=TerminalStrategy.CAUSAL,
    )


def _step(parameters, state, components, *, soc=0.5, demand=20.0, **kwargs):
    engine, battery, powertrain = components
    return thermostat_step(
        parameters,
        state,
        demand_bus_kw=demand,
        soc=soc,
        sigma=1.0,
        dt_s=60.0,
        engine=engine,
        battery=battery,
        powertrain=powertrain,
        **kwargs,
    )


def test_off_engine_restarts_at_the_independent_lower_threshold(
    parameters, components
) -> None:
    state = ThermostatState(False, 120.0)
    above = _step(parameters, state, components, soc=0.5)
    at_low = _step(parameters, state, components, soc=0.4)
    assert above.engine_off
    assert not at_low.engine_off
    assert at_low.next_state.restart_count == 1


def test_on_engine_stops_only_at_the_independent_upper_threshold(
    parameters, components
) -> None:
    state = ThermostatState(True, 120.0)
    below = _step(parameters, state, components, soc=0.5)
    at_high = _step(parameters, state, components, soc=0.6)
    assert not below.engine_off
    assert at_high.engine_off


def test_minimum_on_dwell_blocks_an_early_stop(parameters, components) -> None:
    decision = _step(
        parameters, ThermostatState(True, 60.0), components, soc=0.7
    )
    assert not decision.engine_off
    assert decision.next_state.elapsed_in_state_s == 120.0


def test_minimum_off_dwell_blocks_an_early_restart(parameters, components) -> None:
    decision = _step(
        parameters, ThermostatState(False, 60.0), components, soc=0.3
    )
    assert decision.engine_off
    assert decision.restart_fuel_kg == 0.0


def test_restart_fuel_is_charged_once_and_not_while_remaining_on(
    parameters, components
) -> None:
    first = _step(
        parameters, ThermostatState(False, 120.0), components, soc=0.4
    )
    second = _step(parameters, first.next_state, components, soc=0.45)
    assert first.restart_fuel_kg == pytest.approx(0.1)
    assert second.restart_fuel_kg == 0.0
    assert second.next_state.restart_count == 1


def test_initial_engine_state_and_timer_are_explicit(parameters, components) -> None:
    off = _step(parameters, ThermostatState(False, 0.0), components, soc=0.3)
    on = _step(parameters, ThermostatState(True, 0.0), components, soc=0.7)
    assert off.engine_off
    assert not on.engine_off


def test_infeasible_off_state_forces_continuous_engine_operation(parameters) -> None:
    components = Turboshaft(100.0), BatteryPack(5.0), SeriesPowertrain()
    decision = _step(
        parameters, ThermostatState(False, 0.0), components, demand=20.0
    )
    assert not decision.engine_off
    assert decision.regime is ThermostatRegime.CONTINUOUS
    assert decision.regime_reason == "battery_cannot_carry_full_demand"


def test_demand_above_engine_ceiling_is_reported_as_battery_assisted(
    parameters,
) -> None:
    components = Turboshaft(30.0), BatteryPack(40.0), SeriesPowertrain()
    decision = _step(
        parameters, ThermostatState(True, 300.0), components, demand=35.0
    )
    assert decision.regime is ThermostatRegime.BATTERY_ASSISTED
    assert decision.active_constraint == "engine_max"
    assert decision.battery_bus_kw > 0.0


def test_engine_without_shutdown_cannot_be_labelled_off(parameters) -> None:
    components = (
        Turboshaft(100.0, allow_shutdown=False),
        BatteryPack(40.0),
        SeriesPowertrain(),
    )
    decision = _step(
        parameters, ThermostatState(False, 300.0), components, soc=0.7
    )
    assert not decision.engine_off
    assert decision.regime is ThermostatRegime.CONTINUOUS


def test_cutoff_and_discharge_limit_override_nominal_off_dwell(
    parameters, components
) -> None:
    decision = _step(
        parameters, ThermostatState(False, 0.0), components, soc=0.05
    )
    assert not decision.engine_off
    assert decision.restart_fuel_kg == pytest.approx(0.1)


def test_computed_on_power_selects_a_real_feasible_boundary(components) -> None:
    engine, battery, powertrain = components
    selection = select_engine_on_power(
        demand_bus_kw=20.0,
        soc=0.5,
        sigma=1.0,
        dt_s=60.0,
        engine=engine,
        battery=battery,
        powertrain=powertrain,
    )
    assert selection.cycling_beneficial
    assert selection.active_constraint in {"engine_max", "battery_charge_limit"}


def test_terminal_depletion_requires_and_exposes_preview(parameters, components) -> None:
    engine, battery, _ = components
    target = float(battery.stored_energy_kwh(0.3))
    causal = _step(
        parameters,
        ThermostatState(True, 300.0),
        components,
        soc=0.5,
        time_to_go_s=60.0,
        terminal_energy_target_kwh=target,
    )
    preview_parameters = replace(
        parameters, terminal_strategy=TerminalStrategy.HORIZON_AWARE
    )
    preview = _step(
        preview_parameters,
        ThermostatState(True, 300.0),
        components,
        soc=0.5,
        time_to_go_s=60.0,
        terminal_energy_target_kwh=target,
    )
    assert not causal.next_state.terminal_depletion
    assert preview.next_state.terminal_depletion
    assert preview.regime is ThermostatRegime.TERMINAL_DEPLETION
