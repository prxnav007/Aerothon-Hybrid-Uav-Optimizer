"""Independent identities and behavioural checks for the Breguet analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from src.analysis.breguet import (
    GRAVITY_MPS2,
    BreguetResult,
    breguet_endurance,
    breguet_loiter_estimate,
    breguet_range,
    effective_sfc,
    endurance_optimal_cl,
)
from src.models.atmosphere import atmosphere
from src.models.engine import Turboshaft
from src.models.powertrain import SeriesPowertrain

BASE = {
    "weight_initial_n": 9810.0,
    "weight_final_n": 8000.0,
    "rho": 0.9091,
    "wing_area_m2": 10.0,
    "cl": 1.0,
    "cd": 0.055,
    "sfc_kg_kwh": 0.45,
    "eta_prop": 0.8,
}


@dataclass(frozen=True)
class ReferenceAircraft:
    dry_mass_kg: float = 750.0
    wing_area_m2: float = 10.0
    cd0: float = 0.028
    aspect_ratio: float = 16.0
    oswald_e: float = 0.78
    cl_max: float = 1.5
    eta_prop: float = 0.8
    engine: Turboshaft = field(default_factory=lambda: Turboshaft(75.0))
    powertrain: SeriesPowertrain = field(default_factory=SeriesPowertrain)
    stall_margin: float = 1.2


def _endurance(**changes: float) -> float:
    arguments = BASE | changes
    return breguet_endurance(**arguments)


def _pure_thermal_loiter_time_march(
    aircraft: ReferenceAircraft,
    altitude_m: float,
    fuel_available_kg: float,
    dt_s: float = 60.0,
) -> float:
    """Small matched integrator used until the project simulator is available."""
    analytical = breguet_loiter_estimate(
        aircraft,
        altitude_m,
        fuel_available_kg,
        sfc_mode="fixed" if aircraft.engine.idle_fuel_fraction == 0.0 else "mean_operating_point",
    )
    atmospheric_state = atmosphere(altitude_m)
    rho = float(atmospheric_state.density_kg_m3)
    sigma = float(atmospheric_state.density_ratio)
    fuel_remaining_kg = fuel_available_kg
    elapsed_s = 0.0

    while fuel_remaining_kg > 0.0:
        weight_n = (aircraft.dry_mass_kg + fuel_remaining_kg) * GRAVITY_MPS2
        speed_mps = math.sqrt(
            2.0 * weight_n / (rho * aircraft.wing_area_m2 * analytical.cl)
        )
        thrust_power_kw = weight_n / analytical.lift_to_drag * speed_mps / 1000.0
        propeller_shaft_kw = thrust_power_kw / aircraft.eta_prop
        bus_demand_kw = float(aircraft.powertrain.bus_power_required(propeller_shaft_kw))
        engine_command_kw = float(aircraft.powertrain.engine_power_for_bus(bus_demand_kw))
        engine_state = aircraft.engine.operate(engine_command_kw, sigma)
        powertrain_state = aircraft.powertrain.solve(
            propeller_shaft_kw,
            engine_state.delivered_kw,
            battery_bus_kw=0.0,
        )

        assert not engine_state.power_limited
        assert not engine_state.at_idle
        assert powertrain_state.balanced
        step_s = min(dt_s, fuel_remaining_kg / engine_state.fuel_flow_kg_s)
        fuel_remaining_kg -= engine_state.fuel_flow_kg_s * step_s
        elapsed_s += step_s

    return elapsed_s


def test_endurance_scales_linearly_with_propeller_efficiency_and_inversely_with_sfc() -> None:
    baseline = _endurance()

    assert _endurance(eta_prop=0.4) == pytest.approx(0.5 * baseline)
    assert _endurance(sfc_kg_kwh=0.9) == pytest.approx(0.5 * baseline)


def test_endurance_scales_with_lift_coefficient_to_three_halves_over_drag_coefficient() -> None:
    baseline = _endurance()

    assert _endurance(cl=2.0 * BASE["cl"]) == pytest.approx(2.0**1.5 * baseline)
    assert _endurance(cd=2.0 * BASE["cd"]) == pytest.approx(0.5 * baseline)


def test_doubling_wing_area_scales_endurance_by_the_square_root_of_two() -> None:
    assert _endurance(wing_area_m2=20.0) == pytest.approx(math.sqrt(2.0) * _endurance())


def test_range_and_endurance_differentials_recover_the_segment_speed() -> None:
    final_weight = BASE["weight_final_n"]
    perturbation_n = 0.01

    def endurance_to(weight_n: float) -> float:
        return _endurance(weight_final_n=weight_n)

    def range_to(weight_n: float) -> float:
        return breguet_range(
            BASE["weight_initial_n"],
            weight_n,
            BASE["cl"],
            BASE["cd"],
            BASE["sfc_kg_kwh"],
            BASE["eta_prop"],
        )

    d_endurance = (
        endurance_to(final_weight + perturbation_n)
        - endurance_to(final_weight - perturbation_n)
    ) / (2.0 * perturbation_n)
    d_range = (
        range_to(final_weight + perturbation_n)
        - range_to(final_weight - perturbation_n)
    ) / (2.0 * perturbation_n)
    expected_speed = math.sqrt(
        2.0
        * final_weight
        / (BASE["rho"] * BASE["wing_area_m2"] * BASE["cl"])
    )

    assert d_range / d_endurance == pytest.approx(expected_speed, rel=1.0e-8)


def test_endurance_optimal_lift_coefficient_returns_the_unconstrained_solution() -> None:
    cd0, aspect_ratio, oswald_e = 0.02, 10.0, 0.8
    expected = math.sqrt(3.0 * cd0 * math.pi * aspect_ratio * oswald_e)

    assert endurance_optimal_cl(cd0, aspect_ratio, oswald_e, cl_max=5.0) == pytest.approx(
        expected
    )


def test_endurance_optimal_lift_coefficient_returns_the_stall_margin_cap() -> None:
    stall_margin = 1.2
    cl_max = 1.2

    assert endurance_optimal_cl(
        0.04, 18.0, 0.85, cl_max, stall_margin
    ) == pytest.approx(cl_max / stall_margin**2)


def test_effective_sfc_is_engine_sfc_divided_by_every_electrical_efficiency() -> None:
    powertrain = SeriesPowertrain()
    expected_chain = (
        powertrain.eta_generator
        * powertrain.eta_rectifier
        * powertrain.eta_inverter
        * powertrain.eta_motor
        * powertrain.eta_cabling
    )

    assert effective_sfc(0.45, powertrain) == pytest.approx(0.45 / expected_chain)


def test_zero_fuel_has_zero_endurance_and_preserves_the_operating_point() -> None:
    result = breguet_loiter_estimate(ReferenceAircraft(), 6000.0, 0.0, sfc_mode="fixed")

    assert isinstance(result, BreguetResult)
    assert result.endurance_s == 0.0
    assert result.speed_initial_mps == pytest.approx(result.speed_final_mps)


def test_loiter_endurance_increases_monotonically_with_available_fuel() -> None:
    aircraft = ReferenceAircraft()
    results = [
        breguet_loiter_estimate(aircraft, 6000.0, fuel, sfc_mode="fixed").endurance_s
        for fuel in (0.0, 50.0, 100.0, 200.0)
    ]

    assert results == sorted(results)
    assert all(later > earlier for earlier, later in zip(results, results[1:]))


def test_reference_airframe_loiter_is_stall_margin_limited() -> None:
    aircraft = ReferenceAircraft()
    result = breguet_loiter_estimate(aircraft, 6000.0, 250.0)

    assert result.stall_limited
    assert result.cl == pytest.approx(aircraft.cl_max / aircraft.stall_margin**2)


def test_zero_idle_fuel_fraction_makes_both_sfc_modes_identical() -> None:
    constant_sfc_aircraft = ReferenceAircraft(
        engine=Turboshaft(75.0, idle_fuel_fraction=0.0)
    )

    fixed = breguet_loiter_estimate(
        constant_sfc_aircraft, 6000.0, 250.0, sfc_mode="fixed"
    )
    mean = breguet_loiter_estimate(
        constant_sfc_aircraft, 6000.0, 250.0, sfc_mode="mean_operating_point"
    )

    assert mean.sfc_used_kg_kwh == pytest.approx(fixed.sfc_used_kg_kwh)
    assert mean.endurance_s == pytest.approx(fixed.endurance_s)


def test_closed_form_agrees_with_a_matched_pure_thermal_time_march_within_one_percent() -> None:
    aircraft = ReferenceAircraft(engine=Turboshaft(75.0, idle_fuel_fraction=0.0))
    analytical = breguet_loiter_estimate(
        aircraft, 6000.0, 250.0, sfc_mode="fixed"
    ).endurance_s
    time_marched = _pure_thermal_loiter_time_march(aircraft, 6000.0, 250.0)
    discrepancy = abs(time_marched - analytical) / analytical

    assert discrepancy < 0.01, f"matched-case discrepancy was {discrepancy:.6%}"


def test_willans_mean_operating_point_reports_a_part_load_sfc_penalty() -> None:
    aircraft = ReferenceAircraft()
    fixed = breguet_loiter_estimate(aircraft, 6000.0, 250.0, sfc_mode="fixed")
    mean = breguet_loiter_estimate(
        aircraft, 6000.0, 250.0, sfc_mode="mean_operating_point"
    )

    assert mean.sfc_used_kg_kwh > fixed.sfc_used_kg_kwh
    assert mean.endurance_s < fixed.endurance_s


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"weight_final_n": 10_000.0}, "weight_final_n"),
        ({"rho": 0.0}, "rho"),
        ({"wing_area_m2": 0.0}, "wing_area_m2"),
        ({"sfc_kg_kwh": 0.0}, "sfc_kg_kwh"),
    ],
)
def test_breguet_endurance_rejects_invalid_physical_inputs(
    changes: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _endurance(**changes)


def test_loiter_estimate_rejects_unknown_sfc_mode() -> None:
    with pytest.raises(ValueError, match="sfc_mode"):
        breguet_loiter_estimate(
            ReferenceAircraft(),
            6000.0,
            100.0,
            sfc_mode="unknown",  # type: ignore[arg-type]
        )


def test_loiter_estimate_rejects_a_load_dependent_chain() -> None:
    powertrain = SeriesPowertrain(
        load_dependent=True,
        rated_engine_kw=75.0,
        rated_bus_kw=75.0,
    )
    aircraft = ReferenceAircraft(powertrain=powertrain)

    with pytest.raises(ValueError, match="constant conversion efficiencies"):
        breguet_loiter_estimate(aircraft, 6000.0, 100.0)


def test_engine_power_conversion_uses_newtons_and_the_complete_chain() -> None:
    aircraft = ReferenceAircraft(engine=Turboshaft(75.0, idle_fuel_fraction=0.0))
    result = breguet_loiter_estimate(aircraft, 6000.0, 250.0)
    weight_initial_n = (aircraft.dry_mass_kg + 250.0) * GRAVITY_MPS2
    thrust_power_kw = (
        weight_initial_n
        / result.lift_to_drag
        * result.speed_initial_mps
        / 1000.0
    )
    engine_power_kw = thrust_power_kw / aircraft.eta_prop / aircraft.powertrain.chain_efficiency

    assert engine_power_kw > thrust_power_kw
    assert result.effective_sfc_kg_kwh == pytest.approx(
        aircraft.engine.sfc_rated_kg_kwh / aircraft.powertrain.chain_efficiency
    )
