"""Bounds, ledgers, checkpoints, ranking, isolation, and phase gate tests."""

from __future__ import annotations

import csv
import pytest

import src.analysis.thermostat_threshold_search as search_module
import src.simulation.simulator as simulator_module
from src.analysis.thermostat_threshold_search import (
    ThresholdEvaluation,
    ThresholdSearchConfig,
    assess_phase_dependent_gate,
    build_phase_ledger,
    coarse_threshold_candidates,
    default_threshold_search_config,
    run_threshold_search,
    select_best_evaluation,
    simulate_threshold_candidate,
)
from src.control.pi_ecms import PIECMS
from src.simulation.mission import ps1_mission
from src.simulation.simulator import run_mission


def _evaluation(
    low: float,
    high: float,
    endurance_s: float,
    *,
    feasible: bool = True,
    restarts: int = 10,
    stage: str = "test",
) -> ThresholdEvaluation:
    return ThresholdEvaluation(
        candidate_id=f"{low:.9f}:{high:.9f}",
        stage=stage,
        soc_low=low,
        soc_high=high,
        total_time_s=endurance_s,
        loiter_time_s=endurance_s - 100.0,
        mission_complete=feasible,
        feasible=feasible,
        feasibility_reasons_json="[]" if feasible else '["infeasible"]',
        termination_reason="fuel_reserve" if feasible else "power_shortfall",
        final_soc=0.1,
        minimum_soc=0.05,
        final_stored_energy_kwh=1.0,
        fuel_consumed_kg=10.0,
        fuel_remaining_kg=5.0,
        restart_count=restarts,
        loiter_restarts_per_hour=1.0,
        overall_engine_off_fraction=0.2,
        loiter_engine_off_fraction=0.2,
        mean_engine_on_power_kw=50.0,
        minimum_engine_on_power_kw=30.0,
        maximum_engine_on_power_kw=70.0,
        constraint_encounters_json="{}",
        descent_landing_completed=feasible,
        reserve_shortfall_kg=0.0,
        maximum_power_balance_residual_kw=0.0,
        fuel_ledger_residual_kg=0.0,
        energy_ledger_residual_kwh=0.0,
        discrete_energy_residual_fraction=0.0,
        minimum_transition_margin_s=60.0,
        minimum_bus_supply_margin_kw=0.0,
        failure_flags_json="[]",
        runtime_s=0.01,
    )


@pytest.fixture(scope="module")
def untuned_simulation():
    return simulate_threshold_candidate(
        0.4,
        0.6,
        stage="test_fixture",
        config=default_threshold_search_config(),
    )


def test_search_bounds_use_the_actual_battery_floor_and_reject_equal_thresholds() -> None:
    config = default_threshold_search_config()
    assert config.soc_min == pytest.approx(0.05)
    assert config.soc_max == 1.0
    assert config.validate_pair(0.05, 0.1) == (0.05, 0.1)
    with pytest.raises(ValueError, match="outside"):
        config.validate_pair(0.049, 0.2)
    with pytest.raises(ValueError, match="separation"):
        config.validate_pair(0.4, 0.4)
    with pytest.raises(ValueError, match="separation"):
        config.validate_pair(0.4, 0.449)


def test_coarse_mesh_includes_reference_lower_narrow_wide_and_high_bands() -> None:
    candidates = coarse_threshold_candidates(default_threshold_search_config())
    assert candidates[0] == (0.4, 0.6)
    assert (0.05, 0.1) in candidates
    assert (0.05, 1.0) in candidates
    assert (0.85, 1.0) in candidates
    assert len(candidates) == len(set(candidates))


def test_checkpoint_resume_skips_every_completed_candidate(tmp_path) -> None:
    config = ThresholdSearchConfig(
        soc_min=0.05,
        soc_max=1.0,
        maximum_evaluations=3,
        retained_regions=1,
        coarse_low_values=(0.05, 0.4),
        coarse_high_values=(0.1, 0.6),
    )
    calls: list[tuple[float, float]] = []

    def fake(low, high, stage, supplied_config):
        supplied_config.validate_pair(low, high)
        calls.append((low, high))
        return _evaluation(low, high, 1000.0 - low - high, stage=stage)

    checkpoint = tmp_path / "thresholds.csv"
    first = run_threshold_search(checkpoint, config=config, evaluator=fake)
    assert len(first.evaluations) == 3
    assert len(calls) == 3
    calls.clear()
    resumed = run_threshold_search(checkpoint, config=config, evaluator=fake)
    assert len(resumed.evaluations) == 3
    assert calls == []
    with checkpoint.open(newline="", encoding="utf-8") as stream:
        assert len(tuple(csv.DictReader(stream))) == 3


def test_feasibility_precedes_endurance_and_exact_ties_use_restart_count() -> None:
    infeasible = _evaluation(0.1, 0.2, 2000.0, feasible=False)
    many_starts = _evaluation(0.2, 0.3, 1000.0, restarts=20)
    few_starts = _evaluation(0.3, 0.4, 1000.0, restarts=5)
    best, ties = select_best_evaluation((infeasible, many_starts, few_starts))
    assert best == few_starts
    assert set(ties) == {many_starts, few_starts}


def test_phase_ledger_closes_fuel_energy_and_phase_boundaries(
    untuned_simulation,
) -> None:
    evaluation, run = untuned_simulation
    ledgers = build_phase_ledger(
        "thermostat", run.aircraft, run.mission, run.result
    )
    assert tuple(ledger.phase for ledger in ledgers) == run.mission.phase_names
    assert ledgers[0].start_time_s == 0.0
    assert ledgers[-1].end_time_s == pytest.approx(run.result.endurance_s)
    assert sum(ledger.fuel_consumed_kg for ledger in ledgers) == pytest.approx(
        run.result.fuel_used_kg, abs=1.0e-10
    )
    for earlier, later in zip(ledgers, ledgers[1:]):
        assert later.start_time_s == pytest.approx(earlier.end_time_s)
        assert later.start_fuel_kg == pytest.approx(earlier.end_fuel_kg)
        assert later.start_soc == pytest.approx(earlier.end_soc)
        assert later.start_stored_energy_kwh == pytest.approx(
            earlier.end_stored_energy_kwh
        )
    assert sum(ledger.restart_count for ledger in ledgers) == evaluation.restart_count


def test_repeated_evaluation_is_deterministic_and_state_is_isolated(
    untuned_simulation,
) -> None:
    first, first_run = untuned_simulation
    other, other_run = simulate_threshold_candidate(
        0.05,
        0.1,
        stage="intervening_candidate",
        config=default_threshold_search_config(),
    )
    repeated, repeated_run = simulate_threshold_candidate(
        0.4,
        0.6,
        stage="repeat",
        config=default_threshold_search_config(),
    )
    assert search_module._deterministic_signature(first) == (
        search_module._deterministic_signature(repeated)
    )
    assert first_run.initial_state.restart_count == 0
    assert other_run.initial_state.restart_count == 0
    assert repeated_run.initial_state.restart_count == 0
    assert other.candidate_id != first.candidate_id


def test_search_mission_path_passes_no_future_information(
    monkeypatch,
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
    simulate_threshold_candidate(
        0.4,
        0.6,
        stage="causality_test",
        config=default_threshold_search_config(),
    )
    assert calls > 0


def test_phase_dependent_gate_requires_a_demonstrated_phase_conflict() -> None:
    no_conflict = assess_phase_dependent_gate(
        global_candidate_feasible=True,
        phase_specific_conflict=False,
        one_step_extension_feasible=True,
    )
    conflict = assess_phase_dependent_gate(
        global_candidate_feasible=True,
        phase_specific_conflict=True,
        one_step_extension_feasible=False,
    )
    infeasible = assess_phase_dependent_gate(
        global_candidate_feasible=False,
        phase_specific_conflict=True,
        one_step_extension_feasible=False,
    )
    assert not no_conflict.justified
    assert conflict.justified
    assert not infeasible.justified


def test_ecms_default_path_remains_deterministic_without_thermostat_keywords() -> None:
    aircraft = search_module.build_reference_aircraft()
    mission = ps1_mission(
        cruise_altitude_m=60.0,
        takeoff_duration_s=60.0,
        cruise_duration_s=60.0,
        landing_duration_s=60.0,
        fuel_reserve_kg=280.0,
        descent_landing_fuel_kg=3.0,
        min_usable_fuel_kg=281.0,
        max_mission_time_s=600.0,
    )
    first = run_mission(aircraft, mission, PIECMS(), record_log=True)
    second = run_mission(aircraft, mission, PIECMS(), record_log=True)
    assert first == second
    assert first.thermostat_final_state is None
