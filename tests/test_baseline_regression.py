"""Frozen pre-milestone baseline and its directly executable reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.constraint_diagram import (
    Airframe,
    ConstraintCase,
    constraint_curves,
    feasible_design_point,
    stall_wing_loading_limit,
)
from src.control.pi_ecms import PIECMS
from src.models.atmosphere import atmosphere, g0
from src.models.battery import BatteryPack
from src.models.engine import LAPSE_EXPONENT, Turboshaft
from src.models.mass import build_mass_budget
from src.models.powertrain import SeriesPowertrain
from src.simulation.mission import ps1_mission
from src.simulation.simulator import Aircraft, run_mission

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "milestone1_baseline.json"


def _reproduce_baseline():
    weight_n = 1000.0 * g0
    cruise_mps = 250.0 * 1000.0 / 3600.0
    airframe = Airframe(16.0, 0.78, 0.028, 1.5, 0.85)
    powertrain = SeriesPowertrain()
    cases = (
        ConstraintCase(
            "cruise_3km", 3000.0, cruise_mps, 0.0, 0.0, weight_n
        ),
        ConstraintCase(
            "climb_3km_transient", 3000.0, None, 2.0, 30.0, weight_n
        ),
    )
    grid = np.linspace(150.0, 1800.0, 6601)
    curves = constraint_curves(
        grid, cases, airframe, powertrain, LAPSE_EXPONENT
    )
    stall_limit = stall_wing_loading_limit(
        airframe, 45.0 / 1.2, float(atmosphere(0.0).density_kg_m3)
    )
    design = feasible_design_point(curves, stall_limit)
    peak_bus_kw = float(
        powertrain.bus_power_from_engine(design.engine_power_sl_kw)
    ) + 30.0
    masses = build_mass_budget(
        design.engine_power_sl_kw,
        10.0,
        peak_bus_kw,
        design.wing_area_m2,
        airframe.aspect_ratio,
    )
    engine = Turboshaft(design.engine_power_sl_kw)
    aircraft = Aircraft(
        design.wing_area_m2,
        airframe.aspect_ratio,
        airframe.oswald_efficiency,
        airframe.cd0,
        airframe.cl_max,
        airframe.propeller_efficiency,
        engine,
        BatteryPack(10.0),
        powertrain,
        masses,
    )
    controller = PIECMS()
    result = run_mission(aircraft, ps1_mission(), controller, record_log=True)
    assert result.log is not None
    restarts = sum(
        previous.engine_shut_down and not current.engine_shut_down
        for previous, current in zip(result.log, result.log[1:])
    )
    return design, masses, result, restarts, aircraft, controller


def test_band_aircraft_baseline_headlines_are_unchanged() -> None:
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    design, masses, result, restarts, aircraft, controller = _reproduce_baseline()
    headline = expected["headline"]
    expected_design = expected["design"]
    configuration = expected["configuration"]

    assert design.wing_area_m2 == pytest.approx(
        expected_design["wing_area_m2"], abs=1.0e-12
    )
    assert design.engine_power_sl_kw == pytest.approx(
        expected_design["engine_power_sl_kw"], abs=1.0e-12
    )
    assert masses.fuel_kg == pytest.approx(
        expected_design["initial_fuel_kg"], abs=1.0e-12
    )
    assert result.endurance_s == pytest.approx(headline["endurance_s"], abs=1.0e-9)
    assert result.endurance_s / 3600.0 == pytest.approx(
        headline["endurance_h"], abs=1.0e-12
    )
    assert result.fuel_used_kg == pytest.approx(
        headline["fuel_used_kg"], abs=1.0e-10
    )
    assert result.final_soc == pytest.approx(headline["final_soc"], abs=1.0e-12)
    assert result.min_soc == pytest.approx(headline["min_soc"], abs=1.0e-12)
    assert restarts == headline["restart_count"]
    assert result.termination_reason == headline["termination_reason"]

    controller_config = configuration["controller"]
    assert controller.__class__.__name__ == controller_config["class"]
    assert controller.kp == controller_config["kp"]
    assert controller.soc_ref == controller_config["soc_ref"]
    assert controller.s0 == controller_config["s0"]
    assert controller.s0_ratio == controller_config["s0_ratio"]
    assert not controller_config["has_integral_action"]
    engine_config = configuration["engine"]
    assert aircraft.engine.allow_shutdown is engine_config["allow_shutdown"]
    assert aircraft.engine.idle_fuel_fraction == engine_config["idle_fuel_fraction"]
    assert aircraft.engine.min_power_fraction == engine_config["min_power_fraction"]
    assert aircraft.engine.restart_fuel_kg == engine_config["restart_fuel_kg"]
    transition = configuration["transition_model"]
    assert transition["minimum_on_time_s"] is None
    assert transition["minimum_off_time_s"] is None
    assert transition["start_transient_modelled"] is False

    crossing = expected["loiter_discharge_crossing"]
    loiter = [step for step in result.log if step.phase == "loiter"]
    first_loiter_index = next(
        index for index, step in enumerate(result.log) if step.phase == "loiter"
    )
    start_mass_kg = result.log[first_loiter_index - 1].weight_n / g0
    first_feasible = next(
        step for step in loiter if step.bus_demand_kw <= 30.0 + 1.0e-10
    )
    phase_time_before_loiter_s = sum(
        result.phase_durations_s[name] for name in ("takeoff", "climb", "cruise")
    )
    loiter_duration_s = sum(step.dt_s for step in loiter)
    feasible_duration_s = sum(
        step.dt_s for step in loiter if step.bus_demand_kw <= 30.0 + 1.0e-10
    )
    engine_off_s = sum(step.dt_s for step in loiter if step.engine_shut_down)
    after_crossing = [step for step in loiter if step.time_s > first_feasible.time_s]
    engine_off_after_s = sum(
        step.dt_s for step in after_crossing if step.engine_shut_down
    )
    assert start_mass_kg == pytest.approx(crossing["start_mass_kg"])
    assert first_feasible.weight_n / g0 == pytest.approx(
        crossing["first_feasible_logged_mass_kg"]
    )
    assert start_mass_kg - first_feasible.weight_n / g0 == pytest.approx(
        crossing["fuel_burned_to_first_feasible_step_kg"]
    )
    assert (
        first_feasible.time_s - phase_time_before_loiter_s
    ) / 3600.0 == pytest.approx(crossing["elapsed_to_first_feasible_step_h"])
    assert loiter_duration_s / 3600.0 == pytest.approx(
        crossing["loiter_duration_h"]
    )
    assert feasible_duration_s / 3600.0 == pytest.approx(
        crossing["feasible_duration_h"]
    )
    assert feasible_duration_s / loiter_duration_s == pytest.approx(
        crossing["logged_loiter_time_power_feasible_fraction"]
    )
    assert engine_off_s / loiter_duration_s == pytest.approx(
        crossing["engine_off_fraction"]
    )
    assert engine_off_after_s / sum(
        step.dt_s for step in after_crossing
    ) == pytest.approx(crossing["engine_off_fraction_after_crossing"])

    print(f"endurance_s={result.endurance_s:.15f}")
    print(f"endurance_h={result.endurance_s / 3600.0:.15f}")
    print(f"fuel_used_kg={result.fuel_used_kg:.15f}")
    print(f"final_soc={result.final_soc:.15f}")
    print(f"min_soc={result.min_soc:.15f}")
    print(f"restart_count={restarts}")
    print(f"termination_reason={result.termination_reason}")
