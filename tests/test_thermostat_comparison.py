"""Thermostat replay conservation, preview, and convergence tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.analysis.mode_decomposition import select_post_crossing_window
from src.analysis.replay_comparison import resample_replay_steps
from src.analysis.thermostat_comparison import replay_thermostat
from src.control.thermostat import (
    TerminalStrategy,
    ThermostatParameters,
    ThermostatState,
)
from src.models.battery import BatteryMode
from tests.test_baseline_regression import _reproduce_baseline


@pytest.fixture(scope="module")
def thermostat_case():
    _, _, result, _, aircraft, _ = _reproduce_baseline()
    assert result.log is not None
    window = select_post_crossing_window(
        result.log, aircraft.battery.max_discharge_kw
    )
    return window, aircraft


def _parameters(strategy: TerminalStrategy) -> ThermostatParameters:
    return ThermostatParameters(
        soc_low=0.4,
        soc_high=0.6,
        minimum_on_time_s=60.0,
        minimum_off_time_s=60.0,
        restart_fuel_kg=0.0,
        engine_on_power_kw=None,
        terminal_strategy=strategy,
    )


def test_thermostat_restart_rate_remains_finite_under_timestep_refinement(
    thermostat_case,
) -> None:
    window, aircraft = thermostat_case
    rates = []
    counts = []
    for dt_s in (60.0, 30.0, 15.0):
        result = replay_thermostat(
            resample_replay_steps(window.steps, dt_s),
            aircraft,
            _parameters(TerminalStrategy.CAUSAL),
            initial_soc=window.initial_soc,
            initial_state=ThermostatState(True, 60.0),
            target_energy_change_kwh=0.0,
        )
        counts.append(result.restart_count)
        rates.append(result.restarts_per_flight_hour)
    assert counts == [32, 35, 36]
    assert rates[-1] < 1.15 * rates[0]
    assert rates[-1] < 0.06 * (3600.0 / 15.0)


def test_preview_terminal_mode_reaches_the_depletion_target_more_closely(
    thermostat_case,
) -> None:
    window, aircraft = thermostat_case
    steps = resample_replay_steps(window.steps, 30.0)
    target = -4.689742503060938
    initial = ThermostatState(True, 60.0)
    causal = replay_thermostat(
        steps,
        aircraft,
        _parameters(TerminalStrategy.CAUSAL),
        initial_soc=window.initial_soc,
        initial_state=initial,
        target_energy_change_kwh=target,
    )
    preview = replay_thermostat(
        steps,
        aircraft,
        _parameters(TerminalStrategy.HORIZON_AWARE),
        initial_soc=window.initial_soc,
        initial_state=initial,
        target_energy_change_kwh=target,
    )
    assert causal.terminal_depletion_fraction == 0.0
    assert preview.terminal_depletion_fraction > 0.0
    assert abs(preview.target_residual_kwh) < abs(causal.target_residual_kwh)


def test_physical_thermostat_endpoint_and_integrated_ledgers_close(
    thermostat_case,
) -> None:
    window, aircraft = thermostat_case
    q_nominal_ah = 10_000.0 / 350.0
    physical = replace(
        aircraft,
        battery=replace(
            aircraft.battery,
            mode=BatteryMode.PHYSICAL,
            i_charge_max_a=q_nominal_ah,
            i_discharge_max_a=3.0 * q_nominal_ah,
            terminal_voltage_min_v=242.5,
            terminal_voltage_max_v=407.4,
            q_nominal_ah=q_nominal_ah,
        ),
    )
    result = replay_thermostat(
        resample_replay_steps(window.steps, 60.0),
        physical,
        _parameters(TerminalStrategy.CAUSAL),
        initial_soc=window.initial_soc,
        initial_state=ThermostatState(True, 60.0),
        target_energy_change_kwh=0.0,
    )
    assert result.energy_ledger_residual_kwh == pytest.approx(0.0, abs=2.0e-13)
    assert result.minimum_soc >= physical.battery.soc_min
