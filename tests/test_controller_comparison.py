"""Tests for the frozen-aircraft controller comparison study."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from PIL import Image

from src.analysis.controller_comparison import (
    ControllerSpec,
    OPTIMISED_THERMOSTAT,
    UNTUNED_THERMOSTAT,
    _comparison_rows,
    _write_rows,
    controller_local_candidates,
    generate_controller_figures,
    pareto_controller_keys,
    run_controller_mission,
    select_verified_specs,
)
from src.control.pi_ecms import PIECMS


@pytest.fixture(scope="module")
def optimised_thermostat_record():
    return run_controller_mission(
        OPTIMISED_THERMOSTAT, stage="targeted_regression"
    )


def _synthetic_zero_records(base):
    fixed = replace(
        base,
        run_id="fixed",
        controller_key="fixed_s_ecms",
        display_name="Tuned fixed-(s) ECMS",
        controller_family="fixed",
        candidate_id="fixed:r=1.100",
        configuration_json=json.dumps({"s_ratio": 1.1}),
        total_time_s=56413.8,
        loiter_time_s=50073.8,
        final_soc=0.08,
        minimum_soc=0.05,
        final_fuel_kg=5.1,
        restart_count=200,
    )
    pi = replace(
        base,
        run_id="pi",
        controller_key="adaptive_pi_ecms",
        display_name="Tuned adaptive PI-ECMS",
        controller_family="pi",
        candidate_id="pi:r=1.300:kp=2.500",
        configuration_json=json.dumps({"s0_ratio": 1.3, "kp": 2.5}),
        total_time_s=56436.5,
        loiter_time_s=50096.5,
        final_soc=0.10,
        minimum_soc=0.05,
        final_fuel_kg=5.2,
        restart_count=180,
    )
    thermostat = replace(
        base,
        run_id="thermostat",
        total_time_s=56094.32539909772,
        loiter_time_s=49754.32539909772,
        final_soc=0.22035789264692873,
        minimum_soc=0.17966161710819917,
        final_fuel_kg=5.5078150370516985,
        restart_count=81,
    )
    untuned = replace(
        thermostat,
        run_id="untuned",
        controller_key="untuned_thermostat",
        display_name="Untuned thermostat (0.4, 0.6)",
        tuned=False,
        candidate_id="thermostat:low=0.400:high=0.600",
        total_time_s=55614.32539909772,
        loiter_time_s=49274.32539909772,
        final_soc=0.5187794330353487,
        minimum_soc=0.35323007103028303,
        final_fuel_kg=6.112466123543756,
        restart_count=33,
    )
    return fixed, pi, thermostat, untuned


def _synthetic_sensitivity(zero):
    rows = []
    for cost in (0.0, 0.1, 0.5):
        for record in zero[:3]:
            penalty = cost * record.restart_count * 28.0
            feasible = not (cost == 0.5 and record.controller_key == "adaptive_pi_ecms")
            rows.append(
                replace(
                    record,
                    run_id=f"{record.controller_key}:{cost}",
                    study_stage="restart_sensitivity",
                    restart_cost_per_start_kg=cost,
                    total_time_s=record.total_time_s - penalty,
                    loiter_time_s=record.loiter_time_s - penalty,
                    restart_fuel_kg=cost * record.restart_count,
                    total_fuel_kg=record.running_fuel_kg + cost * record.restart_count,
                    feasible=feasible,
                    mission_complete=feasible,
                    termination_reason="fuel_reserve" if feasible else "fuel_reserve_shortfall",
                )
            )
    return tuple(rows)


def test_exact_controller_specs_keep_fixed_adaptation_off_and_thermostat_causal() -> None:
    candidates = controller_local_candidates()
    fixed = next(candidate for candidate in candidates if candidate.family == "fixed")
    pi = next(candidate for candidate in candidates if candidate.family == "pi")
    assert fixed.configuration(0.0)["adaptation_enabled"] is False
    assert fixed.configuration(0.0)["ratio_anchor"] == "switching_s"
    assert pi.configuration(0.0)["soc_ref"] == 0.6
    assert pi.configuration(0.0)["integral_action"] is False
    assert OPTIMISED_THERMOSTAT.configuration(0.0)["terminal_strategy"] == "causal"
    assert OPTIMISED_THERMOSTAT.configuration(0.0)["minimum_on_time_s"] == 60.0
    assert UNTUNED_THERMOSTAT.tuned is False


def test_local_selection_uses_feasibility_then_endurance(optimised_thermostat_record) -> None:
    base = optimised_thermostat_record
    fixed_low = replace(
        base,
        controller_family="fixed",
        candidate_id="fixed:r=1.000",
        configuration_json=json.dumps({"s_ratio": 1.0}),
        total_time_s=10.0,
    )
    fixed_best = replace(
        fixed_low,
        candidate_id="fixed:r=1.100",
        configuration_json=json.dumps({"s_ratio": 1.1}),
        total_time_s=11.0,
    )
    fixed_infeasible = replace(
        fixed_low,
        candidate_id="fixed:r=1.200",
        configuration_json=json.dumps({"s_ratio": 1.2}),
        total_time_s=12.0,
        feasible=False,
    )
    pi_best = replace(
        base,
        controller_family="pi",
        candidate_id="pi:r=1.300:kp=2.500",
        configuration_json=json.dumps({"s0_ratio": 1.3, "kp": 2.5, "soc_ref": 0.6}),
        total_time_s=13.0,
    )
    fixed, pi = select_verified_specs(
        (fixed_low, fixed_best, fixed_infeasible, pi_best)
    )
    assert fixed.s_ratio == 1.1
    assert pi.s_ratio == 1.3
    assert pi.kp == 2.5


def test_current_simulator_rerun_is_reproducible(optimised_thermostat_record) -> None:
    repeated = run_controller_mission(
        OPTIMISED_THERMOSTAT, stage="targeted_regression"
    )
    first = asdict(optimised_thermostat_record)
    second = asdict(repeated)
    first.pop("simulation_runtime_s")
    second.pop("simulation_runtime_s")
    assert first == second


def test_optimised_thermostat_regression_is_unchanged(
    optimised_thermostat_record,
) -> None:
    record = optimised_thermostat_record
    assert record.total_time_s == pytest.approx(56094.32539909772, abs=1.0e-9)
    assert record.loiter_time_s == pytest.approx(49754.32539909772, abs=1.0e-9)
    assert record.final_soc == pytest.approx(0.22035789264692873, abs=1.0e-12)
    assert record.minimum_soc == pytest.approx(0.17966161710819917, abs=1.0e-12)
    assert record.final_fuel_kg == pytest.approx(5.5078150370516985, abs=1.0e-10)
    assert record.restart_count == 81
    assert record.termination_reason == "fuel_reserve"


def test_restart_fuel_uses_the_authoritative_simulator_accounting_path() -> None:
    spec = ControllerSpec(
        "adaptive_pi_ecms",
        "Tuned adaptive PI-ECMS",
        "pi",
        True,
        s_ratio=1.3,
        kp=2.5,
        soc_ref=0.6,
    )
    record = run_controller_mission(spec, 0.1, stage="targeted_restart")
    assert record.restart_fuel_kg == pytest.approx(0.1 * record.restart_count)
    assert record.total_fuel_kg == pytest.approx(
        record.running_fuel_kg + record.restart_fuel_kg, abs=1.0e-10
    )
    assert abs(record.fuel_ledger_residual_kg) <= 1.0e-10


def test_default_pi_ecms_configuration_remains_unchanged() -> None:
    controller = PIECMS()
    assert controller.kp == 5.0
    assert controller.soc_ref == 0.6
    assert controller.s0 is None
    assert controller.s0_ratio == 1.0


def test_comparison_csv_contains_only_tuned_rows_and_exact_deltas(
    tmp_path: Path, optimised_thermostat_record
) -> None:
    zero = _synthetic_zero_records(optimised_thermostat_record)
    sensitivity = _synthetic_sensitivity(zero)
    rows = _comparison_rows(zero, sensitivity)
    path = _write_rows(tmp_path / "comparison.csv", rows)
    with path.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == 3
    assert {row["controller_key"] for row in written} == {
        "fixed_s_ecms",
        "adaptive_pi_ecms",
        "optimised_thermostat",
    }
    assert float(written[0]["delta_from_best_s"]) == 0.0
    thermostat = next(row for row in written if row["controller_key"] == "optimised_thermostat")
    assert float(thermostat["delta_from_best_s"]) == pytest.approx(
        56094.32539909772 - 56436.5
    )


def test_figures_read_exact_csv_values_and_show_infeasible_sensitivity(
    tmp_path: Path, optimised_thermostat_record
) -> None:
    zero = _synthetic_zero_records(optimised_thermostat_record)
    sensitivity = _synthetic_sensitivity(zero)
    paths = generate_controller_figures(zero, sensitivity, tmp_path)
    endurance_csv = tmp_path / "controller_endurance_comparison.csv"
    with endurance_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pi = next(row for row in rows if row["controller_key"] == "adaptive_pi_ecms")
    assert float(pi["endurance_s"]) == 56436.5
    sensitivity_csv = tmp_path / "controller_restart_sensitivity.csv"
    with sensitivity_csv.open(newline="", encoding="utf-8") as stream:
        sensitivity_rows = list(csv.DictReader(stream))
    infeasible = [row for row in sensitivity_rows if row["feasible"].lower() == "false"]
    assert len(infeasible) == 1
    assert infeasible[0]["controller_key"] == "adaptive_pi_ecms"
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    with Image.open(tmp_path / "controller_endurance_comparison.png") as image:
        assert image.width >= 1920
        assert image.height >= 1080


def test_figure_generation_is_deterministic(
    tmp_path: Path, optimised_thermostat_record
) -> None:
    zero = _synthetic_zero_records(optimised_thermostat_record)
    sensitivity = _synthetic_sensitivity(zero)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_controller_figures(zero, sensitivity, first)
    generate_controller_figures(zero, sensitivity, second)
    for name in (
        "controller_endurance_comparison.png",
        "controller_endurance_comparison.svg",
        "controller_restart_sensitivity.png",
        "controller_restart_sensitivity.svg",
    ):
        first_hash = hashlib.sha256((first / name).read_bytes()).hexdigest()
        second_hash = hashlib.sha256((second / name).read_bytes()).hexdigest()
        assert first_hash == second_hash


def test_pareto_frontier_requires_both_endurance_and_restart_dominance(
    optimised_thermostat_record,
) -> None:
    zero = _synthetic_zero_records(optimised_thermostat_record)[:3]
    assert pareto_controller_keys(zero) == (
        "adaptive_pi_ecms",
        "optimised_thermostat",
    )
