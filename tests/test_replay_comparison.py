"""Energy-normalised replay tests for the baseline post-crossing window."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.analysis.mode_decomposition import select_post_crossing_window
from src.analysis.replay_comparison import (
    ENERGY_MATCH_TOLERANCE_KWH,
    EnergyMismatchError,
    compare_equal_energy_replays,
    compare_replays,
    compare_replays_at_initial_soc,
    replay_pi_ecms,
    replay_pi_ecms_trace,
    resample_replay_steps,
    validated_fuel_gap,
)
from src.models.battery import BatteryMode
from tests.test_baseline_regression import _reproduce_baseline


@pytest.fixture(scope="module")
def baseline_replays():
    _, _, result, _, aircraft, controller = _reproduce_baseline()
    assert result.log is not None
    window = select_post_crossing_window(
        result.log, aircraft.battery.max_discharge_kw
    )
    raw = replay_pi_ecms(
        window.steps,
        aircraft,
        controller,
        initial_soc=window.initial_soc,
        initial_engine_shut_down=window.crossing_step.engine_shut_down,
    )
    comparison = compare_replays(
        window.steps,
        aircraft,
        controller,
        initial_engine_shut_down=window.crossing_step.engine_shut_down,
    )
    return raw, comparison


@pytest.fixture(scope="module")
def representative_case():
    _, _, result, _, aircraft, controller = _reproduce_baseline()
    assert result.log is not None
    window = select_post_crossing_window(
        result.log, aircraft.battery.max_discharge_kw
    )
    return window, aircraft, controller


@pytest.fixture(scope="module")
def equal_energy_legacy(representative_case):
    window, aircraft, controller = representative_case
    common = dict(
        initial_soc=window.initial_soc,
        initial_engine_shut_down=window.crossing_step.engine_shut_down,
    )
    sustaining = compare_equal_energy_replays(
        window.steps,
        aircraft,
        controller,
        target_battery_energy_change_kwh=0.0,
        **common,
    )
    depleting = compare_equal_energy_replays(
        window.steps,
        aircraft,
        controller,
        target_battery_energy_change_kwh=-4.689742503060938,
        **common,
    )
    return sustaining, depleting


def test_unchanged_pi_replay_reproduces_the_measured_window(baseline_replays) -> None:
    raw, _ = baseline_replays
    assert raw.fuel_consumed_kg == pytest.approx(163.60701514132478)
    assert raw.terminal_soc == pytest.approx(0.06367471452871735)
    assert raw.battery_energy_change_kwh == pytest.approx(-4.689742503060938)
    assert raw.engine_off_fraction == pytest.approx(0.2897324666859644)
    assert raw.restart_count == 174


def test_common_initial_soc_targets_zero_terminal_energy_without_tuning_pi(
    baseline_replays,
) -> None:
    _, comparison = baseline_replays
    results = {result.strategy: result for result in comparison.strategies}
    assert 0.05 <= comparison.common_initial_soc <= 1.0
    assert results["continuous"].battery_energy_change_kwh == 0.0
    assert results["ideal_analytical_cycle"].battery_energy_change_kwh == 0.0
    # The discrete PI policy has a one-switch discontinuity at the shooting root.
    assert abs(results["pi_ecms"].terminal_energy_shortfall_kwh) < 0.01
    assert comparison.normalisation_method.startswith("common initial SoC shooting")


def test_replay_gap_fields_follow_the_reported_fuel_totals(baseline_replays) -> None:
    _, comparison = baseline_replays
    results = {result.strategy: result for result in comparison.strategies}
    continuous = results["continuous"].fuel_consumed_kg
    ideal = results["ideal_analytical_cycle"].fuel_consumed_kg
    pi = results["pi_ecms"].fuel_consumed_kg
    assert comparison.pi_to_continuous_gap_kg == pytest.approx(continuous - pi)
    assert comparison.pi_to_continuous_gap_fraction == pytest.approx(
        (continuous - pi) / continuous
    )
    assert comparison.pi_to_ideal_cycle_gap_kg == pytest.approx(pi - ideal)
    assert comparison.pi_to_ideal_cycle_gap_fraction == pytest.approx(
        (pi - ideal) / ideal
    )
    assert comparison.pi_nearer_strategy == "ideal_analytical_cycle"


def test_analytical_duty_fraction_does_not_invent_a_restart_count(
    baseline_replays,
) -> None:
    _, comparison = baseline_replays
    ideal = next(
        result
        for result in comparison.strategies
        if result.strategy == "ideal_analytical_cycle"
    )
    assert ideal.restart_count is None
    assert "no cycle period" in ideal.restart_count_status


def test_representative_soc_replay_holds_initial_soc_and_reports_shortfalls(
    representative_case,
) -> None:
    window, aircraft, controller = representative_case
    target = -4.689742503060938
    comparison = compare_replays_at_initial_soc(
        window.steps,
        aircraft,
        controller,
        initial_soc=window.initial_soc,
        initial_engine_shut_down=window.crossing_step.engine_shut_down,
        target_battery_energy_change_kwh=target,
    )
    rows = {row.strategy: row for row in comparison.strategies}
    assert comparison.common_initial_soc == pytest.approx(0.559342988331593)
    assert rows["pi_ecms"].terminal_energy_shortfall_kwh == pytest.approx(0.0)
    assert rows["continuous"].terminal_energy_shortfall_kwh == pytest.approx(
        -target
    )
    assert rows["ideal_analytical_cycle"].terminal_energy_shortfall_kwh == pytest.approx(
        -target
    )
    assert rows["pi_ecms"].minimum_soc == pytest.approx(0.05000527247581649)
    assert rows["pi_ecms"].maximum_soc == pytest.approx(window.initial_soc)
    assert not comparison.energy_normalisation_achieved
    assert comparison.fuel_gap_status.startswith("not energy-normalised")


def test_physical_replay_trace_has_consistent_energy_and_current_encounters(
    representative_case,
) -> None:
    window, aircraft, controller = representative_case
    q_nominal_ah = 10_000.0 / 350.0
    physical_battery = replace(
        aircraft.battery,
        mode=BatteryMode.PHYSICAL,
        i_charge_max_a=q_nominal_ah,
        i_discharge_max_a=3.0 * q_nominal_ah,
        terminal_voltage_min_v=242.5,
        terminal_voltage_max_v=407.4,
        q_nominal_ah=q_nominal_ah,
    )
    trace = replay_pi_ecms_trace(
        resample_replay_steps(window.steps, 60.0),
        replace(aircraft, battery=physical_battery),
        controller,
        initial_soc=window.initial_soc,
        initial_engine_shut_down=window.crossing_step.engine_shut_down,
        target_battery_energy_change_kwh=-4.689742503060938,
    )
    assert trace.result.battery_mode == "physical"
    assert trace.result.euler_energy_residual_kwh == pytest.approx(0.0, abs=2.0e-13)
    assert any(
        encounter.direction == "charge" and encounter.limit == "current"
        for encounter in trace.result.constraint_encounters
    )
    assert sum(step.dt_s for step in trace.steps) == pytest.approx(
        sum(step.dt_s for step in window.steps)
    )


def test_unmatched_endpoint_energy_rejects_a_normalised_fuel_gap(
    representative_case,
) -> None:
    window, aircraft, controller = representative_case
    target = -4.689742503060938
    comparison = compare_replays_at_initial_soc(
        window.steps,
        aircraft,
        controller,
        initial_soc=window.initial_soc,
        initial_engine_shut_down=window.crossing_step.engine_shut_down,
        target_battery_energy_change_kwh=target,
    )
    continuous, _, pi = comparison.strategies
    with pytest.raises(EnergyMismatchError, match="continuous endpoint energy"):
        validated_fuel_gap(continuous, pi, target_kwh=target)


def test_equal_energy_replay_uses_the_measured_soc_and_endpoint_gate(
    equal_energy_legacy,
) -> None:
    sustaining, depleting = equal_energy_legacy
    assert sustaining.initial_soc == pytest.approx(0.559342988331593)
    assert sustaining.energy_match_tolerance_kwh == pytest.approx(1.0e-12)
    for comparison in (sustaining, depleting):
        target = comparison.target_battery_energy_change_kwh
        for result in (comparison.continuous, comparison.ideal_relaxed):
            assert abs(result.battery_energy_change_kwh - target) <= (
                ENERGY_MATCH_TOLERANCE_KWH
            )
    assert depleting.continuous.calibration_parameter_value == pytest.approx(
        0.4503825786896414
    )
    assert depleting.ideal_relaxed.restart_count is None


def test_discrete_pi_is_reported_as_a_bracket_not_an_interpolated_point(
    equal_energy_legacy,
) -> None:
    sustaining, depleting = equal_energy_legacy
    bracket = sustaining.pi_bracket
    assert not bracket.exact_match
    assert (
        bracket.lower_result.terminal_energy_shortfall_kwh
        * bracket.upper_result.terminal_energy_shortfall_kwh
        < 0.0
    )
    assert bracket.parameter_width > 0.0
    assert "bounded gap" in sustaining.fuel_gap_status
    assert depleting.pi_bracket.exact_match
