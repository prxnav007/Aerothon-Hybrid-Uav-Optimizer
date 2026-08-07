"""Analytical cycle equations, optima, regimes, and map integration."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from src.analysis.cycle_feasibility import (
    battery_efficiencies,
    build_pack_size_trade,
    build_feasibility_map,
    discharge_condition_removal,
    discharge_crossing_mass_kg,
    find_power_boundary_altitude_m,
    loiter_bus_demand_kw,
    loiter_feasibility_summary,
    minimum_discharge_c_rate,
    minimum_pack_capacity_kwh,
)
from src.analysis.cycle_model import (
    DiagnosticParameters,
    OperatingRegime,
    analytical_cycle_energy_balance,
    battery_energy_for_equal_duration_kwh,
    classify_regime,
    continuous_fuel_rate_kg_h,
    cycle_average_fuel_rate_kg_h,
    cycle_period_s,
    duty_cycle,
    economic_cycling_threshold_kw,
    optimal_battery_assisted_power,
    optimal_engine_on_power,
    two_level_cycle_fuel_rate,
    two_level_penalty_vs_continuous,
    willans_coefficients,
)
from src.models.atmosphere import atmosphere
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.powertrain import SeriesPowertrain

A = 0.36
B = 7.810
G = 0.9025
R = 0.9755
ETA_CHARGE = math.sqrt(R)
ETA_DISCHARGE = math.sqrt(R)
DEMAND_KW = 33.54
CHARGE_LIMIT_KW = 10.0
HAND_ENGINE_MAX_KW = 68.352


def test_two_level_penalty_is_positive_over_random_valid_cycles() -> None:
    rng = np.random.default_rng(1407)
    for _ in range(1000):
        demand = rng.uniform(5.0, 100.0)
        source = rng.uniform(0.75, 1.0)
        high = demand / source + rng.uniform(1.0e-4, 80.0)
        low = rng.uniform(0.0, demand / source - 1.0e-4)
        round_trip = rng.uniform(0.5, 1.0 - 1.0e-10)
        penalty = two_level_penalty_vs_continuous(
            demand, high, low, A, B, source, round_trip
        )
        cycled = two_level_cycle_fuel_rate(
            demand, high, low, A, B, source, round_trip
        )
        assert penalty > 0.0
        assert cycled - continuous_fuel_rate_kg_h(
            demand, A, B, source
        ) == pytest.approx(penalty, rel=2.0e-10, abs=1.0e-14)


def test_two_level_penalty_is_exactly_zero_at_unit_round_trip_efficiency() -> None:
    assert two_level_penalty_vs_continuous(
        DEMAND_KW, 60.0, 20.0, A, B, G, 1.0
    ) == 0.0


def test_two_level_penalty_increases_monotonically_with_round_trip_loss() -> None:
    losses = np.linspace(0.0, 0.45, 101)
    penalties = [
        two_level_penalty_vs_continuous(
            DEMAND_KW, 60.0, 20.0, A, B, G, 1.0 - loss
        )
        for loss in losses
    ]
    assert all(later > earlier for earlier, later in zip(penalties, penalties[1:]))


def test_zero_output_idle_phase_is_worse_than_continuous_operation() -> None:
    cycled = two_level_cycle_fuel_rate(
        DEMAND_KW, 60.0, 0.0, A, B, G, R
    )
    continuous = continuous_fuel_rate_kg_h(DEMAND_KW, A, B, G)
    assert cycled > continuous
    assert cycled - continuous == pytest.approx(
        two_level_penalty_vs_continuous(
            DEMAND_KW, 60.0, 0.0, A, B, G, R
        )
    )


def test_two_level_penalty_vanishes_at_both_no_cycle_limits() -> None:
    continuous_shaft_kw = DEMAND_KW / G
    charge_limit_penalties = [
        two_level_penalty_vs_continuous(
            DEMAND_KW,
            continuous_shaft_kw + epsilon,
            20.0,
            A,
            B,
            G,
            R,
        )
        for epsilon in (1.0e-2, 1.0e-4, 1.0e-6)
    ]
    discharge_limit_penalties = [
        two_level_penalty_vs_continuous(
            DEMAND_KW,
            60.0,
            continuous_shaft_kw - epsilon,
            A,
            B,
            G,
            R,
        )
        for epsilon in (1.0e-2, 1.0e-4, 1.0e-6)
    ]
    assert charge_limit_penalties[-1] < 1.0e-7
    assert discharge_limit_penalties[-1] < 1.0e-7
    assert all(
        later < earlier
        for earlier, later in zip(charge_limit_penalties, charge_limit_penalties[1:])
    )
    assert all(
        later < earlier
        for earlier, later in zip(discharge_limit_penalties, discharge_limit_penalties[1:])
    )


def test_charge_limited_hand_calculation_recomputes_without_adjustment() -> None:
    optimum = optimal_engine_on_power(
        DEMAND_KW,
        100.0,
        CHARGE_LIMIT_KW,
        A,
        B,
        G,
        ETA_CHARGE,
        ETA_DISCHARGE,
    )
    assert optimum.engine_on_kw == pytest.approx(48.24376731301939)
    assert optimum.duty_cycle == pytest.approx(0.7746852985333179)
    assert optimum.cycle_fuel_kg_h == pytest.approx(19.50483760351822)
    assert optimum.continuous_fuel_kg_h == pytest.approx(21.188836565096953)
    assert optimum.benefit_fraction == pytest.approx(0.07947576340046347)
    assert optimum.active_bound == "charge_ceiling"


def test_economic_threshold_matches_the_hand_calculation() -> None:
    threshold = economic_cycling_threshold_kw(
        A, B, G, ETA_CHARGE, ETA_DISCHARGE
    )
    assert threshold == pytest.approx(779.5732582199557)


def test_engine_ceiling_hand_calculation_recomputes_without_adjustment() -> None:
    optimum = optimal_engine_on_power(
        DEMAND_KW,
        HAND_ENGINE_MAX_KW,
        100.0,
        A,
        B,
        G,
        ETA_CHARGE,
        ETA_DISCHARGE,
    )
    assert optimum.engine_on_kw == pytest.approx(HAND_ENGINE_MAX_KW)
    assert optimum.duty_cycle == pytest.approx(0.5498535361332719)
    assert optimum.cycle_fuel_kg_h == pytest.approx(17.824448121842156)
    assert optimum.benefit_fraction == pytest.approx(0.15878117861349417)
    assert optimum.active_bound == "engine_ceiling"


def test_cycle_period_matches_the_hand_calculation_when_efficiencies_are_split() -> None:
    engine_kw = (DEMAND_KW + CHARGE_LIMIT_KW) / G
    fraction = duty_cycle(
        DEMAND_KW, engine_kw, G, ETA_CHARGE, ETA_DISCHARGE
    )
    assert cycle_period_s(
        0.1, 10.0, ETA_DISCHARGE, fraction, DEMAND_KW
    ) == pytest.approx(470.5042488171657)


def test_unit_round_trip_limit_recovers_the_lossless_duty_cycle() -> None:
    engine_kw = 60.0
    actual = duty_cycle(DEMAND_KW, engine_kw, G, 1.0, 1.0)
    assert actual == pytest.approx(DEMAND_KW / (G * engine_kw))


def test_zero_intercept_makes_the_threshold_zero_and_cycling_never_wins() -> None:
    assert economic_cycling_threshold_kw(
        A, 0.0, G, ETA_CHARGE, ETA_DISCHARGE
    ) == 0.0
    optimum = optimal_engine_on_power(
        DEMAND_KW,
        80.0,
        CHARGE_LIMIT_KW,
        A,
        0.0,
        G,
        ETA_CHARGE,
        ETA_DISCHARGE,
    )
    assert optimum.engine_on_kw == pytest.approx(optimum.lower_bound_kw)
    assert optimum.benefit_fraction == pytest.approx(0.0, abs=1.0e-15)


def test_cycle_fuel_rate_decreases_with_engine_power_below_the_threshold() -> None:
    threshold = economic_cycling_threshold_kw(
        A, B, G, ETA_CHARGE, ETA_DISCHARGE
    )
    assert DEMAND_KW < threshold
    rates = [
        cycle_average_fuel_rate_kg_h(
            DEMAND_KW, power, A, B, G, ETA_CHARGE, ETA_DISCHARGE
        )
        for power in (40.0, 50.0, 60.0, 70.0)
    ]
    assert all(later < earlier for earlier, later in zip(rates, rates[1:]))


def test_rates_are_equal_at_the_threshold_for_every_valid_engine_power() -> None:
    threshold = economic_cycling_threshold_kw(
        A, B, G, ETA_CHARGE, ETA_DISCHARGE
    )
    continuous = continuous_fuel_rate_kg_h(threshold, A, B, G)
    for engine_kw in (threshold / G, 1.2 * threshold / G, 2.0 * threshold / G):
        cycled = cycle_average_fuel_rate_kg_h(
            threshold,
            engine_kw,
            A,
            B,
            G,
            ETA_CHARGE,
            ETA_DISCHARGE,
        )
        assert cycled == pytest.approx(continuous, rel=1.0e-13)


def test_analytical_cycle_energy_conservation_includes_every_conversion_loss() -> None:
    engine_kw = (DEMAND_KW + CHARGE_LIMIT_KW) / G
    balance = analytical_cycle_energy_balance(
        DEMAND_KW,
        engine_kw,
        500.0,
        A,
        B,
        G,
        0.95 * 0.95 * 0.99,
        0.85,
        ETA_CHARGE,
        ETA_DISCHARGE,
        43_100.0,
    )
    assert balance.battery_stored_energy_change_kwh == pytest.approx(0.0)
    assert balance.battery_ohmic_loss_kwh > 0.0
    assert balance.propeller_loss_kwh > 0.0
    assert balance.residual_fraction < 1.0e-14


def test_regime_classification_requires_the_pack_to_hold_full_off_demand() -> None:
    result = classify_regime(
        DEMAND_KW,
        70.0,
        CHARGE_LIMIT_KW,
        30.0,
        A,
        B,
        G,
        ETA_CHARGE,
        ETA_DISCHARGE,
    )
    assert result.regime is OperatingRegime.ENGINE_LIMITED_CONTINUOUS
    assert result.cycling_blocker == "battery_discharge_limit"


def test_regime_classification_identifies_cycling_and_assistance() -> None:
    cycling = classify_regime(
        25.0, 70.0, 10.0, 30.0, A, B, G, ETA_CHARGE, ETA_DISCHARGE
    )
    assisted = classify_regime(
        70.0, 70.0, 10.0, 100.0, A, B, G, ETA_CHARGE, ETA_DISCHARGE
    )
    assert cycling.regime is OperatingRegime.CYCLING_FEASIBLE
    assert assisted.regime is OperatingRegime.BATTERY_ASSISTED


def test_engine_only_boundary_is_continuous_without_a_false_cycle_interval() -> None:
    result = classify_regime(
        G * 70.0,
        70.0,
        10.0,
        100.0,
        A,
        B,
        G,
        ETA_CHARGE,
        ETA_DISCHARGE,
    )
    assert result.regime is OperatingRegime.ENGINE_LIMITED_CONTINUOUS
    assert result.cycle_optimum is None
    assert result.cycling_blocker == "no_engine_surplus"


def test_battery_assisted_optimum_can_be_an_interior_equalisation() -> None:
    optimum = optimal_battery_assisted_power(
        demand_bus_kw=70.0,
        engine_min_kw=0.0,
        engine_max_kw=70.0,
        source_efficiency=G,
        fuel_available_kg=20.0,
        battery_usable_bus_kwh=20.0,
        willans_a_kg_kwh=A,
        willans_b_kg_h=B,
    )
    assert optimum.active_bound == "interior_equalisation"
    assert optimum.fuel_duration_h == pytest.approx(
        optimum.battery_duration_h, rel=1.0e-7
    )


def test_consistent_mtow_ten_kilometre_case_is_battery_bound_at_engine_max() -> None:
    powertrain = SeriesPowertrain()
    engine = Turboshaft(86.7791369750147)
    battery = BatteryPack(10.0)
    demand_kw = loiter_bus_demand_kw(
        10_000.0,
        1000.0,
        wing_area_m2=7.59175537062125,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        powertrain=powertrain,
    )
    engine_max_kw = engine.max_power_kw(float(atmosphere(10_000.0).density_ratio))
    assist_kw = demand_kw - G * engine_max_kw
    eta_discharge = battery_efficiencies(
        battery,
        charge_bus_kw=0.0,
        discharge_bus_kw=assist_kw,
        soc=0.55,
    ).discharge
    a, b = willans_coefficients(engine.rated_power_kw, 0.45, 0.20)
    optimum = optimal_battery_assisted_power(
        demand_kw,
        0.0,
        engine_max_kw,
        G,
        288.6101983109414,
        0.95 * 10.0 * eta_discharge,
        a,
        b,
    )
    required_bus_kwh = battery_energy_for_equal_duration_kwh(
        engine_max_kw, demand_kw, G, 288.6101983109414, a, b
    )
    assert optimum.active_bound == "engine_ceiling"
    assert optimum.limiting_source == "battery"
    assert optimum.fuel_duration_h == pytest.approx(13.813359889303054)
    assert optimum.battery_duration_h == pytest.approx(0.4550720286626884)
    assert required_bus_kwh == pytest.approx(285.9768026059839)


def test_separate_battery_efficiencies_use_the_actual_charge_and_discharge_powers() -> None:
    efficiencies = battery_efficiencies(
        BatteryPack(10.0),
        charge_bus_kw=10.0,
        discharge_bus_kw=33.631429393239564,
        soc=0.55,
    )
    assert efficiencies.charge == pytest.approx(0.9960637059853846)
    assert efficiencies.discharge == pytest.approx(0.986473886387726)
    assert efficiencies.round_trip == pytest.approx(0.9825908351331637)
    assert efficiencies.round_trip != pytest.approx(0.9755, rel=1.0e-3)


def test_idle_fraction_sensitivity_is_nonlinear_over_the_required_sweep() -> None:
    demand_kw = 33.631429393239564
    efficiencies = battery_efficiencies(
        BatteryPack(10.0),
        charge_bus_kw=10.0,
        discharge_bus_kw=demand_kw,
        soc=0.55,
    )
    fractions = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
    benefits = []
    for idle_fraction in fractions:
        a, b = willans_coefficients(86.7791369750147, 0.45, idle_fraction)
        optimum = optimal_engine_on_power(
            demand_kw,
            68.36,
            10.0,
            a,
            b,
            G,
            efficiencies.charge,
            efficiencies.discharge,
        )
        benefits.append(optimum.benefit_fraction)
    increments = [later - earlier for earlier, later in zip(benefits, benefits[1:])]
    assert benefits[0] == pytest.approx(0.021117994509395333)
    assert benefits[-1] == pytest.approx(0.12401140272315248)
    assert max(increments) - min(increments) > 1.0e-3


def test_direct_aerodynamics_exposes_both_demand_reference_weights() -> None:
    powertrain = SeriesPowertrain()
    common = dict(
        wing_area_m2=7.59175537062125,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        powertrain=powertrain,
    )
    mtow = loiter_bus_demand_kw(3000.0, 1000.0, **common)
    loiter_start = loiter_bus_demand_kw(3000.0, 954.7917908626673, **common)
    assert mtow == pytest.approx(36.0480943788494)
    assert loiter_start == pytest.approx(33.631429393239564)
    assert mtow > loiter_start


def test_discharge_sizing_rules_recover_the_loiter_start_hand_estimates() -> None:
    demand_kw = 33.631429393239564
    assert minimum_pack_capacity_kwh(demand_kw, 3.0) == pytest.approx(
        11.210476464413188
    )
    assert minimum_discharge_c_rate(demand_kw, 10.0) == pytest.approx(
        3.3631429393239564
    )


def test_mass_crossing_matches_the_constant_lift_power_scaling() -> None:
    start_mass_kg = 954.7917908626673
    start_demand_kw = 33.631429393239564
    crossing = discharge_crossing_mass_kg(
        start_mass_kg, start_demand_kw, 30.0
    )
    powertrain = SeriesPowertrain()
    demand_at_crossing = loiter_bus_demand_kw(
        3000.0,
        crossing,
        wing_area_m2=7.59175537062125,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        powertrain=powertrain,
    )
    assert crossing == pytest.approx(884.7609824502671)
    assert demand_at_crossing == pytest.approx(30.0, rel=1.0e-12)


def test_mass_crossing_time_and_feasible_share_match_the_baseline_log() -> None:
    engine = Turboshaft(86.7791369750147)
    a, b = willans_coefficients(
        engine.rated_power_kw, engine.sfc_rated_kg_kwh, 0.20
    )
    summary = loiter_feasibility_summary(
        954.7917908626673,
        33.631429393239564,
        30.0,
        49842.85487028608 / 3600.0,
        a,
        b,
        G,
    )
    assert summary.fuel_burned_to_crossing_kg == pytest.approx(70.03080841240023)
    assert summary.elapsed_to_crossing_h == pytest.approx(3.4181333743297113)
    assert summary.loiter_time_power_feasible_fraction == pytest.approx(
        0.7531184724548597
    )
    assert summary.elapsed_to_crossing_h == pytest.approx(3.433333333333333, abs=0.02)


def test_direct_mtow_boundaries_match_the_service_and_absolute_ceilings() -> None:
    powertrain = SeriesPowertrain()
    engine = Turboshaft(86.7791369750147)
    common = dict(
        wing_area_m2=7.59175537062125,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        engine=engine,
        powertrain=powertrain,
    )
    absolute = find_power_boundary_altitude_m(1000.0, 0.0, **common)
    service = find_power_boundary_altitude_m(
        1000.0, 0.0, climb_rate_mps=0.5, **common
    )
    charge_boundary = find_power_boundary_altitude_m(1000.0, 10.0, **common)
    assert absolute == pytest.approx(6825.429425120803)
    assert service == pytest.approx(5842.332266356088)
    assert charge_boundary == pytest.approx(5307.355476454084)
    assert service < absolute


def test_loiter_start_mass_recovers_the_quoted_crossover_estimates() -> None:
    powertrain = SeriesPowertrain()
    engine = Turboshaft(86.7791369750147)
    common = dict(
        wing_area_m2=7.59175537062125,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        engine=engine,
        powertrain=powertrain,
    )
    mass_kg = 954.7917908626673
    assert find_power_boundary_altitude_m(
        mass_kg, 0.0, **common
    ) == pytest.approx(7292.900977664871)
    assert find_power_boundary_altitude_m(
        mass_kg, 10.0, **common
    ) == pytest.approx(5728.969071835909)


def test_map_carries_every_uncalibrated_parameter_and_all_three_regimes() -> None:
    parameters = DiagnosticParameters(0.20, 1.0, 3.0, 0.0, 300.0, 300.0)
    engine = Turboshaft(86.7791369750147)
    battery = BatteryPack(10.0)
    points = build_feasibility_map(
        (0.0, 6000.0, 10_000.0),
        (720.0, 1000.0),
        wing_area_m2=7.59175537062125,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        dry_mass_kg=711.3898016890586,
        engine=engine,
        battery=battery,
        powertrain=SeriesPowertrain(),
        parameters=parameters,
        efficiency_evaluation_soc=0.55,
        usable_soc_low=0.05,
        usable_soc_high=1.0,
    )
    regimes = {point.regime for point in points}
    assert OperatingRegime.CYCLING_FEASIBLE.value in regimes
    assert OperatingRegime.ENGINE_LIMITED_CONTINUOUS.value in regimes
    assert OperatingRegime.BATTERY_ASSISTED.value in regimes
    assert all(point.idle_fuel_fraction == 0.20 for point in points)
    assert all(point.minimum_on_time_s == 300.0 for point in points)


def test_discharge_blocked_map_points_do_not_report_a_cycling_benefit() -> None:
    parameters = DiagnosticParameters(0.20, 1.0, 3.0, 0.0, 300.0, 300.0)
    points = build_feasibility_map(
        (3000.0,),
        (884.0, 954.7917908626673),
        wing_area_m2=7.59175537062125,
        aspect_ratio=16.0,
        oswald_efficiency=0.78,
        cd0=0.028,
        cl_max=1.5,
        propeller_efficiency=0.85,
        dry_mass_kg=711.3898016890586,
        engine=Turboshaft(86.7791369750147),
        battery=BatteryPack(10.0),
        powertrain=SeriesPowertrain(),
        parameters=parameters,
        efficiency_evaluation_soc=0.55,
        usable_soc_low=0.05,
        usable_soc_high=1.0,
    )
    feasible, blocked = points
    assert feasible.cycling_feasible
    assert math.isfinite(feasible.cycle_benefit_fraction)
    assert blocked.charge_side_cycling_candidate
    assert not blocked.discharge_side_feasible
    assert not blocked.cycling_feasible
    assert math.isnan(blocked.cycle_benefit_fraction)
    removal = discharge_condition_removal(points)
    assert removal.charge_side_candidate_points == 2
    assert removal.removed_points == 1
    assert removal.sampled_grid_point_removed_fraction == 0.5


def test_pack_trade_computes_both_benefits_and_an_interior_default_optimum() -> None:
    engine = Turboshaft(86.7791369750147)
    powertrain = SeriesPowertrain()
    points = build_pack_size_trade(
        (10.0, 11.25, 12.0),
        (0.20,),
        reference_capacity_kwh=10.0,
        loiter_start_mass_kg=954.7917908626673,
        loiter_fuel_floor_kg=9.7,
        loiter_start_demand_bus_kw=33.631429393239564,
        altitude_m=3000.0,
        wing_area_m2=7.59175537062125,
        aspect_ratio=16.0,
        peak_bus_kw=float(
            powertrain.bus_power_from_engine(engine.rated_power_kw)
        )
        + 30.0,
        engine=engine,
        powertrain=powertrain,
        charge_c_rate=1.0,
        discharge_c_rate=3.0,
        efficiency_evaluation_soc=0.55,
        integration_points=401,
    )
    middle = points[1]
    assert middle.pack_mass_delta_kg == pytest.approx(6.666666666666664)
    assert middle.fuel_capacity_cost_kg == pytest.approx(6.2305295950155255)
    assert middle.discharge_enabling_benefit_kg > 0.0
    assert middle.charge_ceiling_benefit_kg > 0.0
    assert middle.gross_operational_benefit_kg == pytest.approx(
        middle.discharge_enabling_benefit_kg
        + middle.charge_ceiling_benefit_kg
    )
    assert middle.net_fuel_benefit_kg > 0.0
    assert middle.loiter_endurance_h == max(
        point.loiter_endurance_h for point in points
    )


def test_pack_trade_stops_at_the_peak_power_capacity_boundary() -> None:
    engine = Turboshaft(86.7791369750147)
    powertrain = SeriesPowertrain()
    with pytest.raises(ValueError, match="peak-power feasibility boundary at 10 kWh"):
        build_pack_size_trade(
            (9.75, 10.0),
            (0.10,),
            reference_capacity_kwh=10.0,
            loiter_start_mass_kg=954.7917908626673,
            loiter_fuel_floor_kg=9.7,
            loiter_start_demand_bus_kw=33.631429393239564,
            altitude_m=3000.0,
            wing_area_m2=7.59175537062125,
            aspect_ratio=16.0,
            peak_bus_kw=float(
                powertrain.bus_power_from_engine(engine.rated_power_kw)
            )
            + 30.0,
            engine=engine,
            powertrain=powertrain,
            charge_c_rate=1.0,
            discharge_c_rate=3.0,
            efficiency_evaluation_soc=0.55,
        )


@pytest.mark.parametrize(
    "function",
    [
        DiagnosticParameters,
        willans_coefficients,
    ],
)
def test_uncalibrated_inputs_have_no_function_defaults(function) -> None:
    signature = inspect.signature(function)
    for name in (
        "idle_fuel_fraction",
        "charge_c_rate",
        "discharge_c_rate",
        "restart_fuel_kg",
        "minimum_on_time_s",
        "minimum_off_time_s",
    ):
        if name in signature.parameters:
            assert signature.parameters[name].default is inspect.Parameter.empty
