"""Static GA plant resolution without mission or fitness evaluation."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest

from src.models import aerodynamics
from src.optimization.chromosome import DecodedPlantThermostatDesign, GENE_NAMES
from src.optimization.feasibility import (
    StaticFeasibilityScenario,
    evaluate_static_feasibility,
)

REFERENCE_AREA_M2 = 7.59175537062125
REFERENCE_ENGINE_KW = 86.7791369750147


def reference_design(**changes: float) -> DecodedPlantThermostatDesign:
    values = {
        "wing_area_m2": REFERENCE_AREA_M2,
        "aspect_ratio": 16.0,
        "engine_rating_kw": REFERENCE_ENGINE_KW,
        "battery_capacity_kwh": 10.0,
        "soc_low": 0.225,
        "soc_high": 0.350,
    }
    values.update(changes)
    return DecodedPlantThermostatDesign(**values)


def resolve(
    design: DecodedPlantThermostatDesign | None = None,
    scenario: StaticFeasibilityScenario | None = None,
):
    return evaluate_static_feasibility(
        design or reference_design(),
        scenario=scenario or StaticFeasibilityScenario.nominal(),
    )


def constraint(result, name: str):
    return next(item for item in result.hard_constraints if item.name == name)


def test_reference_design_resolves_deterministically() -> None:
    scenario = StaticFeasibilityScenario.nominal()
    first = resolve(scenario=scenario)
    second = resolve(scenario=scenario)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_reference_design_passes_every_justified_hard_static_constraint() -> None:
    result = resolve()
    assert result.is_feasible
    assert result.violated_hard_constraint_count == 0
    assert result.total_normalized_violation == 0.0
    assert all(item.satisfied for item in result.hard_constraints)


def test_reference_wetted_area_calibration_exactly_preserves_cd0() -> None:
    wing = resolve().resolved_design.wing
    assert wing.cd0 == pytest.approx(0.028, abs=1.0e-15)
    assert wing.fixed_nonwing_wetted_area_m2 == pytest.approx(
        22.896044038214548, abs=1.0e-14
    )


def test_wing_and_total_wetted_areas_retain_both_physical_contributions() -> None:
    wing = resolve().resolved_design.wing
    expected_wing = 2.0 * REFERENCE_AREA_M2 * (1.0 + 0.25 * 0.15)
    assert wing.wing_wetted_area_m2 == pytest.approx(expected_wing, rel=1.0e-15)
    assert wing.total_wetted_area_m2 == pytest.approx(
        wing.fixed_nonwing_wetted_area_m2 + wing.wing_wetted_area_m2,
        rel=1.0e-15,
    )


@pytest.mark.parametrize(
    ("area_m2", "total_wetted_area_m2", "cd0"),
    (
        (6.0, 35.34604403821455, 0.03240054036836334),
        (REFERENCE_AREA_M2, 38.64893643225364, 0.028),
        (10.0, 43.64604403821455, 0.024005324221018),
        (16.0, 56.09604403821455, 0.01928301513813625),
    ),
)
def test_ga_cd0_varies_with_wing_area_as_authorised(
    area_m2: float, total_wetted_area_m2: float, cd0: float
) -> None:
    wing = resolve(reference_design(wing_area_m2=area_m2)).resolved_design.wing
    assert wing.total_wetted_area_m2 == pytest.approx(total_wetted_area_m2, rel=1.0e-14)
    assert wing.cd0 == pytest.approx(cd0, rel=1.0e-14)


def test_parasite_drag_area_increases_when_the_wing_is_enlarged() -> None:
    areas = (6.0, REFERENCE_AREA_M2, 10.0, 16.0)
    drag_areas = [
        resolve(reference_design(wing_area_m2=area)).resolved_design.wing.cd0 * area
        for area in areas
    ]
    assert all(right > left for left, right in zip(drag_areas, drag_areas[1:]))


def test_cd0_remains_excluded_from_the_chromosome() -> None:
    assert "cd0" not in GENE_NAMES
    assert "wing_area" in GENE_NAMES


def test_aspect_ratio_changes_induced_drag_and_authoritative_wing_mass() -> None:
    low_ar = resolve(reference_design(aspect_ratio=12.0)).resolved_design
    high_ar = resolve(reference_design(aspect_ratio=20.0)).resolved_design
    assert high_ar.wing.induced_drag_factor < low_ar.wing.induced_drag_factor
    assert high_ar.masses.wing_kg != pytest.approx(low_ar.masses.wing_kg)


def test_nominal_oswald_policy_is_explicitly_fixed_at_point_78() -> None:
    wing = resolve().resolved_design.wing
    assert wing.oswald_policy == "fixed_reference"
    assert wing.oswald_efficiency == 0.78
    assert wing.oswald_is_fixed is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("equivalent_skin_friction", 0.0056),
        ("thickness_to_chord", 0.16),
        ("reference_wing_area_m2", 8.0),
        ("reference_cd0", 0.029),
        ("fixed_nonwing_wetted_area_m2", 23.0),
        ("oswald_efficiency", 0.79),
    ),
)
def test_scenario_identity_changes_for_material_aero_assumptions(
    field: str, replacement: float
) -> None:
    scenario = StaticFeasibilityScenario.nominal()
    changed = replace(scenario, **{field: replacement})
    assert changed.identity != scenario.identity


def test_scenario_serialization_contains_the_wetted_area_policy_and_identity() -> None:
    scenario = StaticFeasibilityScenario.nominal()
    serialized = scenario.to_dict()
    assert serialized["scenario_id"] == scenario.identity
    assert serialized["wetted_area_policy"] == "reference_calibrated_wetted_area"
    assert serialized["fixed_nonwing_wetted_area_m2"] == pytest.approx(
        22.896044038214548
    )


def test_increasing_battery_capacity_adds_mass_and_reduces_residual_fuel() -> None:
    small = resolve(reference_design(battery_capacity_kwh=5.0)).resolved_design.masses
    large = resolve(reference_design(battery_capacity_kwh=20.0)).resolved_design.masses
    assert large.battery_kg > small.battery_kg
    assert large.dry_kg > small.dry_kg
    assert large.fuel_kg < small.fuel_kg


def test_increasing_engine_rating_adds_propulsion_mass_and_reduces_fuel() -> None:
    small = resolve(reference_design(engine_rating_kw=70.0)).resolved_design.masses
    large = resolve(reference_design(engine_rating_kw=120.0)).resolved_design.masses
    assert large.engine_kg > small.engine_kg
    assert large.propulsion_total_kg > small.propulsion_total_kg
    assert large.fuel_kg < small.fuel_kg


def test_fuel_below_minimum_is_returned_and_not_clipped() -> None:
    result = resolve(reference_design(
        wing_area_m2=16.0,
        aspect_ratio=24.0,
        engine_rating_kw=140.0,
        battery_capacity_kwh=30.0,
    ))
    fuel = result.resolved_design.masses.fuel_kg
    record = constraint(result, "minimum_usable_fuel")
    assert 0.0 < fuel < 20.0
    assert record.quantity == fuel
    assert not record.satisfied
    assert record.normalized_violation > 0.0


def test_tank_volume_shortfall_is_a_normalized_hard_violation() -> None:
    result = resolve(reference_design(
        wing_area_m2=6.0,
        aspect_ratio=10.0,
        engine_rating_kw=60.0,
        battery_capacity_kwh=5.0,
    ))
    record = constraint(result, "fuel_tank_volume")
    assert result.resolved_design.fuel.fuel_volume_margin_l < 0.0
    assert not record.satisfied
    assert record.margin < 0.0
    assert record.normalized_violation > 0.0


def test_underpowered_engine_produces_cruise_and_climb_power_violations() -> None:
    result = resolve(reference_design(engine_rating_kw=60.0, battery_capacity_kwh=5.0))
    assert not constraint(result, "cruise_engine_rating_with_margin").satisfied
    assert not constraint(
        result, "climb_engine_rating_with_battery_and_margin"
    ).satisfied


def test_battery_shortfall_uses_discharge_sustainable_for_the_screen_timestep() -> None:
    result = resolve(reference_design(engine_rating_kw=60.0, battery_capacity_kwh=5.0))
    record = constraint(result, "battery_peak_discharge_sustainable_one_step")
    power = result.resolved_design.power
    assert power.battery_sustainable_discharge_kw < power.battery_peak_required_bus_kw
    assert record.quantity == power.battery_sustainable_discharge_kw
    assert not record.satisfied


def test_total_and_maximum_violation_are_exact_rollups_of_hard_records() -> None:
    result = resolve(reference_design(
        wing_area_m2=6.0,
        aspect_ratio=10.0,
        engine_rating_kw=60.0,
        battery_capacity_kwh=5.0,
    ))
    violations = tuple(item.normalized_violation for item in result.hard_constraints)
    assert result.total_normalized_violation == math.fsum(violations)
    assert result.maximum_normalized_violation == max(violations)
    assert result.violated_hard_constraint_count == sum(
        not item.satisfied for item in result.hard_constraints
    )


def test_advisory_ceiling_and_warnings_do_not_fail_the_reference_design() -> None:
    result = resolve()
    assert result.warnings
    assert len(result.advisory_constraints) == 1
    assert not result.advisory_constraints[0].satisfied
    assert result.is_feasible


def test_service_ceiling_becomes_hard_only_when_the_scenario_requests_it() -> None:
    scenario = replace(
        StaticFeasibilityScenario.nominal(), service_ceiling_is_hard=True
    )
    result = resolve(scenario=scenario)
    assert not result.is_feasible
    assert not constraint(result, "service_ceiling_10km").satisfied
    assert result.advisory_constraints == ()


def test_controller_threshold_invariants_are_checked_without_controller_execution() -> None:
    result = resolve(reference_design(soc_low=0.01, soc_high=0.03))
    assert not constraint(result, "thermostat_soc_floor").satisfied
    assert not constraint(result, "thermostat_minimum_gap").satisfied


def test_constraint_and_result_serialization_are_deterministic_json() -> None:
    result = resolve()
    first = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(resolve().to_dict(), sort_keys=True, separators=(",", ":"))
    assert first == second
    assert constraint(result, "stall_speed").to_dict()["unit"] == "m/s"


def test_evaluation_preserves_immutable_inputs_and_model_defaults() -> None:
    scenario = StaticFeasibilityScenario.nominal()
    design = reference_design()
    before_scenario = scenario.to_dict()
    before_design = design.to_dict()
    resolve(design, scenario)
    assert scenario.to_dict() == before_scenario
    assert design.to_dict() == before_design


def test_reference_aircraft_builder_keeps_the_frozen_cd0() -> None:
    from src.analysis.thermostat_mission import build_reference_aircraft

    assert build_reference_aircraft().cd0 == 0.028


def test_existing_51_square_metre_formula_example_remains_unchanged() -> None:
    assert aerodynamics.parasite_drag_from_wetted_area(51.0, 10.0) == pytest.approx(
        0.02805, rel=1.0e-12
    )


def test_isolated_feasibility_import_does_not_import_the_simulator() -> None:
    code = (
        "import sys; import src.optimization.feasibility; "
        "assert 'src.simulation.simulator' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
