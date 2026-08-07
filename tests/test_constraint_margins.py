"""Constraint-observability tests for logged battery operation."""

from __future__ import annotations

import csv

import pytest

from src.analysis.constraint_margins import (
    audit_constraint_margins,
    build_battery_mode_comparison,
    write_battery_mode_comparison_csv,
    write_constraint_margins_csv,
)
from src.models.battery import BatteryMode, BatteryPack
from tests.test_baseline_regression import _reproduce_baseline


@pytest.fixture(scope="module")
def baseline_audit():
    _, _, result, _, aircraft, _ = _reproduce_baseline()
    assert result.log is not None
    return audit_constraint_margins(
        result.log,
        aircraft.battery,
        initial_soc=1.0,
    )


def test_physical_current_and_terminal_voltage_margins_remain_undefined(
    baseline_audit,
) -> None:
    for summary in (baseline_audit.charge, baseline_audit.discharge):
        assert summary.physical_current_margin_min_a is None
        assert summary.physical_voltage_margin_min_v is None
        assert summary.physical_limit_proximity_fraction is None
        assert "undefined" in summary.physical_current_limit_source
        assert "open-circuit" in summary.physical_voltage_limit_source
    assert all(
        point.physical_current_limit_a is None
        and point.physical_terminal_voltage_limit_v is None
        for point in baseline_audit.points
    )


def test_observed_charge_distribution_and_proxy_saturation_are_measured(
    baseline_audit,
) -> None:
    charge = baseline_audit.charge
    assert charge.point_count == 604
    assert charge.observed_current_max_a == pytest.approx(32.61247257955745)
    assert charge.observed_terminal_voltage_max_v == pytest.approx(
        360.64905494349705
    )
    assert charge.minimum_ocv_endpoint_difference_v == pytest.approx(
        39.35094505650295
    )
    assert charge.endpoint_difference_soc == pytest.approx(0.5926311671639084)
    assert charge.fraction_at_bus_power_proxy == pytest.approx(0.8658940397350994)


def test_observed_discharge_distribution_is_reported_separately(
    baseline_audit,
) -> None:
    discharge = baseline_audit.discharge
    assert discharge.point_count == 333
    assert discharge.observed_current_max_a == pytest.approx(96.40937361280464)
    assert discharge.observed_terminal_voltage_min_v == pytest.approx(
        305.5817114702844
    )
    assert discharge.minimum_ocv_endpoint_difference_v == pytest.approx(
        5.581711470284404
    )
    assert discharge.endpoint_difference_soc == pytest.approx(0.09559422762788808)


def test_constraint_csv_uses_explicit_physical_margin_headers(
    baseline_audit, tmp_path
) -> None:
    path = write_constraint_margins_csv(
        baseline_audit, tmp_path / "constraint_margins.csv"
    )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 937
    assert rows[0]["physical_current_margin_a"] == ""
    assert rows[0]["physical_voltage_margin_v"] == ""
    assert "ocv_endpoint_difference_v" in rows[0]


def test_battery_mode_comparison_reports_direction_and_binding_limit() -> None:
    legacy = BatteryPack(10.0)
    q_nominal_ah = 10_000.0 / 350.0
    physical = BatteryPack(
        10.0,
        mode=BatteryMode.PHYSICAL,
        i_charge_max_a=q_nominal_ah,
        i_discharge_max_a=3.0 * q_nominal_ah,
        terminal_voltage_min_v=242.5,
        terminal_voltage_max_v=407.4,
        q_nominal_ah=q_nominal_ah,
    )
    points = build_battery_mode_comparison(
        (0.25, 0.75), legacy, physical, scenario="reference"
    )
    assert points[0].charge_binding_limit == "current"
    assert points[0].charge_permissiveness == "physical_less_permissive"
    assert points[1].charge_permissiveness == "physical_more_permissive"
    assert points[0].discharge_binding_limit == "current"
    assert points[0].discharge_permissiveness == "physical_less_permissive"
    assert points[1].discharge_permissiveness == "physical_more_permissive"


def test_battery_mode_comparison_csv_carries_unverified_scenario_inputs(
    tmp_path,
) -> None:
    legacy = BatteryPack(10.0)
    q_nominal_ah = 10_000.0 / 350.0
    physical = BatteryPack(
        10.0,
        mode="physical",
        i_charge_max_a=q_nominal_ah,
        i_discharge_max_a=3.0 * q_nominal_ah,
        terminal_voltage_min_v=242.5,
        terminal_voltage_max_v=407.4,
        q_nominal_ah=q_nominal_ah,
    )
    points = build_battery_mode_comparison(
        (0.5,), legacy, physical, scenario="unverified_97s"
    )
    path = write_battery_mode_comparison_csv(points, tmp_path / "limits.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["scenario"] == "unverified_97s"
    assert float(row["terminal_voltage_max_v"]) == 407.4
