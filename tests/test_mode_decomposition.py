"""Energy-ledger decomposition tests for the Milestone-1 baseline."""

from __future__ import annotations

import csv

import pytest

from src.analysis.mode_decomposition import (
    decompose_modes,
    select_post_crossing_window,
    write_mode_decomposition_csv,
)
from src.models.battery import BatteryPack
from tests.test_baseline_regression import _reproduce_baseline


@pytest.fixture(scope="module")
def baseline_decomposition():
    _, _, result, _, aircraft, _ = _reproduce_baseline()
    assert result.log is not None
    window = select_post_crossing_window(
        result.log, aircraft.battery.max_discharge_kw
    )
    decomposition = decompose_modes(
        window.steps,
        aircraft.battery,
        initial_soc=window.initial_soc,
    )
    return window, decomposition


def test_post_crossing_window_is_partitioned_into_exactly_four_modes(
    baseline_decomposition,
) -> None:
    window, decomposition = baseline_decomposition
    assert len(decomposition.modes) == 4
    assert {row.mode for row in decomposition.modes} == {
        "on_charging",
        "on_battery_neutral",
        "on_battery_assisting",
        "off_discharging",
    }
    assert sum(row.point_count for row in decomposition.modes) == len(window.steps)
    assert sum(row.time_fraction for row in decomposition.modes) == pytest.approx(1.0)
    assert decomposition.neutral_band_kw == 1.0e-6


def test_baseline_mode_energies_reproduce_the_logged_power_integrals(
    baseline_decomposition,
) -> None:
    _, decomposition = baseline_decomposition
    modes = {row.mode: row for row in decomposition.modes}
    assert modes["on_charging"].point_count == 444
    assert modes["on_charging"].battery_bus_energy_kwh == pytest.approx(
        -73.9523746396835
    )
    assert modes["off_discharging"].point_count == 181
    assert modes["off_discharging"].battery_internal_energy_kwh == pytest.approx(
        79.1037716058185
    )
    assert modes["on_battery_neutral"].elapsed_h == 0.0
    assert modes["on_battery_assisting"].elapsed_h == 0.0


def test_cyclic_and_depletion_off_time_are_allocated_from_internal_energy(
    baseline_decomposition,
) -> None:
    _, decomposition = baseline_decomposition
    assert decomposition.endpoint_battery_energy_change_kwh == pytest.approx(
        -4.689742503060938
    )
    assert decomposition.delta_soc == pytest.approx(-0.49566827380287565)
    assert decomposition.recirculated_internal_energy_kwh == pytest.approx(
        decomposition.stored_charge_energy_kwh
    )
    assert decomposition.engine_off_cyclic_h == pytest.approx(2.8055367108500624)
    assert decomposition.engine_off_depletion_h == pytest.approx(0.21112995581659844)
    assert decomposition.engine_off_total_h == pytest.approx(
        decomposition.engine_off_cyclic_h + decomposition.engine_off_depletion_h
    )
    assert decomposition.engine_off_cyclic_fraction == pytest.approx(
        0.26945471987158537
    )
    assert decomposition.engine_off_depletion_fraction == pytest.approx(
        0.020277746814378477
    )


def test_legacy_off_time_reports_both_energy_ledger_allocations(
    baseline_decomposition,
) -> None:
    _, decomposition = baseline_decomposition
    assert decomposition.endpoint_stored_charge_energy_kwh == pytest.approx(
        73.79189228810013
    )
    assert decomposition.endpoint_stored_discharge_energy_kwh == pytest.approx(
        78.48163479116104
    )
    assert decomposition.engine_off_cyclic_endpoint_fraction == pytest.approx(
        0.2724192357988968
    )
    assert decomposition.engine_off_depletion_endpoint_fraction == pytest.approx(
        0.017313230887067023
    )
    assert decomposition.cyclic_off_fraction_uncertainty_low == pytest.approx(
        0.26945471987158537
    )
    assert decomposition.cyclic_off_fraction_uncertainty_high == pytest.approx(
        0.2724192357988968
    )
    assert decomposition.cyclic_off_fraction_uncertainty_width == pytest.approx(
        0.0029645159273114507
    )


def test_legacy_ledger_residual_is_exactly_the_omitted_midpoint_ocv_term(
    baseline_decomposition,
) -> None:
    window, decomposition = baseline_decomposition
    battery = BatteryPack(10.0)
    q_nominal_ah = battery.charge_capacity_ah
    slope_v = battery.v_max_v - battery.v_min_v
    soc = window.initial_soc
    predicted_kwh = 0.0
    for step in window.steps:
        v_oc = float(battery.open_circuit_voltage(soc))
        current_a = step.battery_internal_kw * 1000.0 / v_oc
        predicted_kwh += (
            slope_v
            * current_a**2
            * step.dt_s**2
            / (2.0 * 1000.0 * q_nominal_ah * 3600.0**2)
        )
        soc = step.soc
    assert predicted_kwh == pytest.approx(0.8465588684515882)
    assert predicted_kwh == pytest.approx(decomposition.euler_ledger_residual_kwh)


def test_mode_csv_states_the_neutral_band_and_energy_allocation(
    baseline_decomposition, tmp_path
) -> None:
    _, decomposition = baseline_decomposition
    path = write_mode_decomposition_csv(decomposition, tmp_path / "modes.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["neutral_band_kw"] == "1e-06"
    assert "pro rata" in rows[0]["off_time_allocation"]
