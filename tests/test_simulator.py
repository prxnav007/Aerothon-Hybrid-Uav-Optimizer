"""Mission-loop composition, conservation, numerics, and failure handling."""

import dataclasses
import math
import statistics
import time
from dataclasses import replace

import pytest

import src.simulation.simulator as simulator_module
from src.control.fixed_ecms import FixedECMS
from src.control.pi_ecms import PIECMS
from src.control.power_split import SplitDecision
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.mass import MassBreakdown
from src.models.powertrain import SeriesPowertrain
from src.simulation.mission import ps1_mission
from src.simulation.simulator import (
    Aircraft,
    MissionResult,
    TimeStep,
    log_to_dataframe,
    mission_energy_balance,
    run_mission,
)

@pytest.fixture(scope="module")
def reference_masses() -> MassBreakdown:
    return MassBreakdown(
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


@pytest.fixture(scope="module")
def reference_aircraft(reference_masses: MassBreakdown) -> Aircraft:
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
        masses=reference_masses,
    )


@pytest.fixture(scope="module")
def reference_mission():
    # Short fixed phases keep the suite quick while retaining the mandated
    # 3000 m climb and a resource-dominant loiter.
    return ps1_mission(
        takeoff_duration_s=60.0,
        cruise_duration_s=600.0,
        landing_duration_s=60.0,
        fuel_reserve_kg=15.0,
        descent_landing_fuel_kg=7.0,
        min_usable_fuel_kg=20.0,
        max_mission_time_s=12.0 * 3600.0,
    )


@pytest.fixture(scope="module")
def thermal_result(reference_aircraft: Aircraft, reference_mission) -> MissionResult:
    return run_mission(
        reference_aircraft,
        reference_mission,
        FixedECMS.pure_thermal(),
        record_log=True,
    )


@pytest.fixture(scope="module")
def battery_preferring_result(
    reference_aircraft: Aircraft, reference_mission
) -> MissionResult:
    return run_mission(
        reference_aircraft,
        reference_mission,
        FixedECMS.battery_first(),
        record_log=True,
    )


def test_fuel_and_weight_mass_balances(
    thermal_result: MissionResult,
    reference_aircraft: Aircraft,
) -> None:
    assert thermal_result.log is not None
    assert (
        reference_aircraft.masses.fuel_kg - thermal_result.fuel_used_kg
        == pytest.approx(thermal_result.fuel_remaining_kg, abs=1.0e-12)
    )
    weights = [step.weight_n for step in thermal_result.log]
    assert all(later <= earlier for earlier, later in zip(weights, weights[1:]))
    for step in thermal_result.log:
        expected_weight_n = (
            reference_aircraft.masses.dry_kg + step.fuel_remaining_kg
        ) * 9.80665
        assert step.weight_n == pytest.approx(expected_weight_n, rel=1.0e-12)


def test_energy_balance_closes_with_all_accounted_losses(
    battery_preferring_result: MissionResult,
) -> None:
    assert battery_preferring_result.log is not None
    balance = mission_energy_balance(battery_preferring_result)
    print(f"endpoint_energy_balance_residual_fraction={balance.residual_fraction:.12e}")
    print(
        "discrete_energy_balance_residual_fraction="
        f"{balance.discrete_residual_fraction:.12e}"
    )
    assert balance.fuel_chemical_in_kwh > 0.0
    assert balance.battery_ohmic_loss_kwh > 0.0
    assert balance.battery_stored_energy_change_kwh < 0.0
    assert balance.residual_fraction == pytest.approx(0.001404986785693786)
    assert balance.residual_kwh == pytest.approx(
        -balance.battery_integration_residual_kwh, abs=1.0e-12
    )
    assert balance.discrete_residual_fraction < 1.0e-12


def test_endpoint_energy_residual_halves_with_the_explicit_euler_timestep(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    residuals = []
    for step_s in (60.0, 30.0, 15.0):
        result = run_mission(
            reference_aircraft,
            reference_mission,
            FixedECMS.battery_first(),
            dt_s=step_s,
            record_log=True,
        )
        residuals.append(mission_energy_balance(result).residual_fraction)
    assert residuals == pytest.approx(
        (
            0.001404986785693786,
            0.0007022433338113694,
            0.0003533144534039296,
        )
    )
    assert residuals[1] / residuals[0] == pytest.approx(0.5, rel=0.01)
    assert residuals[2] / residuals[1] == pytest.approx(0.5, rel=0.01)


def test_energy_accounting_requires_an_instrumented_mission(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    result = run_mission(
        reference_aircraft,
        reference_mission,
        FixedECMS.pure_thermal(),
        record_log=False,
    )
    with pytest.raises(ValueError, match="record_log=True"):
        mission_energy_balance(result)


def test_every_phase_reaches_its_target_and_order_is_preserved(
    thermal_result: MissionResult,
    reference_mission,
) -> None:
    assert thermal_result.mission_complete
    assert thermal_result.log is not None
    logged_order = tuple(dict.fromkeys(step.phase for step in thermal_result.log))
    assert logged_order == reference_mission.phase_names
    assert tuple(thermal_result.phase_durations_s) == reference_mission.phase_names
    for phase in reference_mission.phases:
        final_step = next(
            step for step in reversed(thermal_result.log) if step.phase == phase.name
        )
        assert final_step.altitude_m == pytest.approx(
            phase.target_altitude_m, abs=1.0e-12
        )


def test_resource_phase_retains_descent_fuel_and_post_landing_reserve(
    thermal_result: MissionResult,
    reference_mission,
) -> None:
    assert thermal_result.termination_reason == "fuel_reserve"
    assert thermal_result.log is not None
    last_loiter = next(
        step for step in reversed(thermal_result.log) if step.phase == "loiter"
    )
    assert last_loiter.fuel_remaining_kg == pytest.approx(
        reference_mission.loiter_fuel_floor_kg, abs=1.0e-9
    )
    assert thermal_result.fuel_remaining_kg >= reference_mission.fuel_reserve_kg
    descent_landing_burn_kg = (
        last_loiter.fuel_remaining_kg - thermal_result.fuel_remaining_kg
    )
    assert descent_landing_burn_kg <= reference_mission.descent_landing_fuel_kg


def test_log_records_both_average_and_marginal_equivalence_references(
    thermal_result: MissionResult,
) -> None:
    assert thermal_result.log is not None
    loiter = next(step for step in thermal_result.log if step.phase == "loiter")
    assert loiter.neutral_s > loiter.switching_s > 0.0


def test_underallocated_descent_fuel_reports_a_post_landing_reserve_shortfall(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    mission = replace(reference_mission, descent_landing_fuel_kg=0.0)
    result = run_mission(
        reference_aircraft, mission, FixedECMS.pure_thermal(), record_log=True
    )
    assert not result.mission_complete
    assert result.termination_reason == "fuel_reserve_shortfall"
    assert "fuel_reserve_shortfall" in result.failure_flags
    assert result.fuel_remaining_kg < mission.fuel_reserve_kg


def test_restart_fuel_is_charged_once_per_logged_off_to_on_transition(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    aircraft = replace(
        reference_aircraft,
        engine=replace(reference_aircraft.engine, restart_fuel_kg=0.1),
    )
    result = run_mission(aircraft, reference_mission, PIECMS(), record_log=True)
    assert result.log is not None
    transitions = sum(
        previous.engine_shut_down and not current.engine_shut_down
        for previous, current in zip(result.log, result.log[1:])
    )
    charged = tuple(step for step in result.log if step.restart_fuel_kg > 0.0)
    assert transitions > 0
    assert len(charged) == transitions
    assert all(step.restart_fuel_kg == pytest.approx(0.1) for step in charged)
    integrated_burn_kg = sum(
        step.fuel_flow_kg_s * step.dt_s + step.restart_fuel_kg
        for step in result.log
    )
    assert result.fuel_used_kg == pytest.approx(integrated_burn_kg, abs=1.0e-10)


def test_high_s_is_the_pure_thermal_reference(
    thermal_result: MissionResult,
) -> None:
    assert thermal_result.mission_complete
    assert thermal_result.final_soc == pytest.approx(1.0, abs=1.0e-12)
    assert thermal_result.min_soc == pytest.approx(1.0, abs=1.0e-12)
    assert thermal_result.log is not None
    assert all(step.battery_bus_kw <= 1.0e-9 for step in thermal_result.log)
    print(f"pure_thermal_endurance_s={thermal_result.endurance_s:.12f}")


def test_low_s_reaches_the_battery_cutoff(
    battery_preferring_result: MissionResult,
    reference_aircraft: Aircraft,
) -> None:
    assert battery_preferring_result.mission_complete
    assert battery_preferring_result.termination_reason == "soc_cutoff"
    assert battery_preferring_result.final_soc == pytest.approx(
        reference_aircraft.battery.soc_min, abs=1.0e-12
    )


def test_endurance_increases_with_initial_fuel(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    results = []
    for fuel_kg in (40.0, 60.0, 80.0):
        masses = replace(reference_aircraft.masses, fuel_kg=fuel_kg)
        aircraft = replace(reference_aircraft, masses=masses)
        results.append(
            run_mission(aircraft, reference_mission, FixedECMS.pure_thermal())
        )
    assert all(result.mission_complete for result in results)
    assert results[0].endurance_s < results[1].endurance_s < results[2].endurance_s


def test_endurance_improves_with_aspect_ratio(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    lower = run_mission(
        replace(reference_aircraft, aspect_ratio=12.0),
        reference_mission,
        FixedECMS.pure_thermal(),
    )
    higher = run_mission(
        replace(reference_aircraft, aspect_ratio=20.0),
        reference_mission,
        FixedECMS.pure_thermal(),
    )
    assert lower.mission_complete and higher.mission_complete
    assert higher.endurance_s > lower.endurance_s


def test_timestep_convergence_is_below_one_percent(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    series = {
        step_s: run_mission(
            reference_aircraft,
            reference_mission,
            FixedECMS.pure_thermal(),
            dt_s=step_s,
        ).endurance_s
        for step_s in (120.0, 60.0, 30.0, 15.0)
    }
    relative_change = abs(series[15.0] - series[30.0]) / series[15.0]
    error_at_60 = abs(series[60.0] - series[15.0]) / series[15.0]
    print(f"timestep_convergence_s={series!r}")
    print(f"timestep_60s_relative_error={error_at_60:.12e}")
    assert relative_change < 0.01


def test_partial_climb_step_lands_exactly_without_full_step_time_error(
    reference_aircraft: Aircraft,
) -> None:
    mission = ps1_mission(
        cruise_altitude_m=2700.0,
        takeoff_duration_s=60.0,
        cruise_duration_s=60.0,
        landing_duration_s=60.0,
        fuel_reserve_kg=15.0,
        descent_landing_fuel_kg=7.0,
        min_usable_fuel_kg=20.0,
        max_mission_time_s=12.0 * 3600.0,
    )
    result = run_mission(
        reference_aircraft,
        mission,
        FixedECMS.pure_thermal(),
        dt_s=60.0,
        record_log=True,
    )
    assert result.mission_complete
    assert result.phase_durations_s["climb"] == pytest.approx(1350.0, abs=1.0e-12)
    assert result.log is not None
    climb_steps = tuple(step for step in result.log if step.phase == "climb")
    assert climb_steps[-1].dt_s == pytest.approx(30.0, abs=1.0e-12)
    assert climb_steps[-1].altitude_m == pytest.approx(2700.0, abs=1.0e-12)


def test_per_phase_loiter_timestep_preserves_endurance_and_is_faster(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    uniform_times = []
    coarse_times = []
    uniform = coarse = None
    for _ in range(3):
        start = time.perf_counter()
        uniform = run_mission(
            reference_aircraft,
            reference_mission,
            FixedECMS.pure_thermal(),
            dt_s=60.0,
        )
        uniform_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        coarse = run_mission(
            reference_aircraft,
            reference_mission,
            FixedECMS.pure_thermal(),
            dt_s=60.0,
            phase_dt_s={"loiter": 300.0},
        )
        coarse_times.append(time.perf_counter() - start)

    assert uniform is not None and coarse is not None
    difference = abs(coarse.endurance_s - uniform.endurance_s) / uniform.endurance_s
    speedup = statistics.median(uniform_times) / statistics.median(coarse_times)
    print(f"loiter_300s_relative_endurance_difference={difference:.12e}")
    print(f"loiter_300s_wall_clock_speedup={speedup:.6f}")
    assert difference < 0.01
    assert speedup > 1.2


def test_identical_missions_are_bit_deterministic(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    first = run_mission(
        reference_aircraft,
        reference_mission,
        FixedECMS(s=5.0),
        record_log=True,
    )
    second = run_mission(
        reference_aircraft,
        reference_mission,
        FixedECMS(s=5.0),
        record_log=True,
    )
    assert first == second


def test_logging_disabled_allocates_no_timestep_objects(
    monkeypatch: pytest.MonkeyPatch,
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    def forbidden_timestep(*args, **kwargs):
        raise AssertionError("TimeStep allocated while record_log=False")

    monkeypatch.setattr(simulator_module, "TimeStep", forbidden_timestep)
    result = run_mission(
        reference_aircraft,
        reference_mission,
        FixedECMS.pure_thermal(),
        record_log=False,
    )
    assert result.log is None


def test_logging_produces_one_object_per_step_and_reports_timing(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    timings: dict[bool, list[float]] = {False: [], True: []}
    results: dict[bool, MissionResult] = {}
    for enabled in (False, True):
        for _ in range(5):
            start = time.perf_counter()
            results[enabled] = run_mission(
                reference_aircraft,
                reference_mission,
                FixedECMS.pure_thermal(),
                record_log=enabled,
            )
            timings[enabled].append(time.perf_counter() - start)
    assert results[False].log is None
    assert results[True].log is not None
    assert len(results[True].log) > 0
    off_s = statistics.median(timings[False])
    on_s = statistics.median(timings[True])
    print(f"logging_off_median_s={off_s:.9f}")
    print(f"logging_on_median_s={on_s:.9f}")
    print(f"logging_time_ratio_on_over_off={on_s / off_s:.6f}")


def test_log_to_dataframe_has_one_row_and_column_per_logged_value(
    thermal_result: MissionResult,
) -> None:
    dataframe = log_to_dataframe(thermal_result)
    assert thermal_result.log is not None
    assert len(dataframe) == len(thermal_result.log)
    assert tuple(dataframe.columns) == tuple(
        field.name for field in dataclasses.fields(TimeStep)
    )


def test_insufficient_fuel_returns_a_failure_instead_of_raising(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    aircraft = replace(
        reference_aircraft,
        masses=replace(reference_aircraft.masses, fuel_kg=0.01),
    )
    result = run_mission(aircraft, reference_mission, FixedECMS.pure_thermal())
    assert not result.mission_complete
    assert result.termination_reason == "fuel_exhausted"
    assert "fuel_exhausted" in result.failure_flags


def test_injected_infeasible_solver_ends_with_power_shortfall(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    def infeasible_solver(**kwargs) -> SplitDecision:
        return SplitDecision(
            engine_shaft_kw=0.0,
            bus_from_engine_kw=0.0,
            battery_bus_kw=0.0,
            battery_internal_kw=0.0,
            fuel_flow_kg_s=0.0,
            hamiltonian_kg_s=math.inf,
            feasible=False,
            engine_off=False,
            engine_at_idle=False,
            active_bound="infeasible",
        )

    result = run_mission(
        reference_aircraft,
        reference_mission,
        FixedECMS(s=5.0),
        split_solver=infeasible_solver,
    )
    assert not result.mission_complete
    assert result.termination_reason == "power_shortfall"
    assert "power_shortfall" in result.failure_flags


def test_nonpositive_available_climb_rate_ends_without_looping(
    reference_aircraft: Aircraft,
    reference_mission,
) -> None:
    underpowered = replace(
        reference_aircraft,
        engine=Turboshaft(60.0),
        battery=BatteryPack(0.1),
    )
    result = run_mission(
        underpowered,
        reference_mission,
        FixedECMS.pure_thermal(),
    )
    assert not result.mission_complete
    assert result.termination_reason == "altitude_unreachable"
    assert "climb_rate_unachievable" in result.failure_flags
    assert "altitude_unreachable" in result.failure_flags
    assert result.phase_durations_s["climb"] == 0.0
