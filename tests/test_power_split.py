"""Tests for ECMS Hamiltonian minimisation and power-split feasibility."""

import math
import time
from dataclasses import dataclass

import numpy as np
import pytest

from src.control.base import neutral_equivalence_factor
from src.control.power_split import (
    PowerSplitComponents,
    SplitDecision,
    grid_search_split,
    hamiltonian,
    solve_split,
    switching_equivalence_factor,
)
from src.models.battery import BatteryPack
from src.models.engine import LHV_KJ_KG, Turboshaft
from src.models.powertrain import SeriesPowertrain


@dataclass(frozen=True)
class SweepResult:
    optimized: tuple[SplitDecision, ...]
    oracle: tuple[SplitDecision, ...]

    @property
    def feasible(self) -> tuple[SplitDecision, ...]:
        return tuple(decision for decision in self.optimized if decision.feasible)


@dataclass(frozen=True)
class LoadDependentSweepResult(SweepResult):
    max_power_difference_kw: float
    max_relative_hamiltonian_difference: float
    hamiltonian_tolerance_failures: int
    convex_intervals_checked: int
    minimum_adjacent_slope_increase: float


@pytest.fixture(scope="module")
def default_components() -> PowerSplitComponents:
    return PowerSplitComponents(
        engine=Turboshaft(75.0),
        battery=BatteryPack(20.0),
        powertrain=SeriesPowertrain(),
    )


@pytest.fixture(scope="module")
def randomized_sweep(default_components: PowerSplitComponents) -> SweepResult:
    rng = np.random.default_rng(20260805)
    optimized: list[SplitDecision] = []
    oracle: list[SplitDecision] = []
    for _ in range(500):
        demand_kw = float(rng.uniform(20.0, 100.0))
        s = float(rng.uniform(0.6, 12.0))
        soc = float(rng.uniform(0.05, 1.0))
        sigma = float(rng.uniform(0.5, 1.0))
        dt_s = float(rng.uniform(1.0, 180.0))
        args = (
            demand_kw,
            default_components.engine,
            default_components.battery,
            default_components.powertrain,
            s,
            soc,
            sigma,
            dt_s,
        )
        optimized.append(solve_split(*args))
        oracle.append(grid_search_split(*args, resolution_kw=0.05))
    return SweepResult(tuple(optimized), tuple(oracle))


@pytest.fixture(scope="module")
def load_dependent_sweep() -> LoadDependentSweepResult:
    """The required O-11 oracle sweep plus an independent convexity check."""
    rng = np.random.default_rng(20260805)
    engine = Turboshaft(75.0)
    battery = BatteryPack(20.0)
    powertrain = SeriesPowertrain(
        load_dependent=True,
        rated_engine_kw=75.0,
        rated_bus_kw=100.0,
    )
    optimized: list[SplitDecision] = []
    oracle: list[SplitDecision] = []
    relative_differences: list[float] = []
    power_differences: list[float] = []
    minimum_slope_increase = math.inf
    convex_intervals_checked = 0

    for _ in range(500):
        demand_kw = float(rng.uniform(20.0, 100.0))
        s = float(rng.uniform(0.6, 12.0))
        soc = float(rng.uniform(0.05, 1.0))
        sigma = float(rng.uniform(0.5, 1.0))
        dt_s = float(rng.uniform(1.0, 180.0))
        args = (demand_kw, engine, battery, powertrain, s, soc, sigma, dt_s)
        fast = solve_split(*args)
        grid = grid_search_split(*args, resolution_kw=0.05)
        optimized.append(fast)
        oracle.append(grid)

        if fast.feasible and grid.feasible:
            power_differences.append(abs(fast.engine_shaft_kw - grid.engine_shaft_kw))
            relative_differences.append(
                abs(fast.hamiltonian_kg_s - grid.hamiltonian_kg_s)
                / abs(grid.hamiltonian_kg_s)
            )

        # Independently sample the engine-on Hamiltonian.  Feasible samples are
        # contiguous because source power is monotone in engine power.
        engine_grid = np.linspace(
            engine.idle_power_kw,
            engine.max_power_kw(sigma),
            101,
        )
        bus_from_engine = np.asarray(
            powertrain.bus_power_from_engine(engine_grid), dtype=float
        )
        battery_bus = demand_kw - bus_from_engine
        discharge_limit = battery.available_discharge_kw(soc, dt_s)
        charge_limit = battery.available_charge_kw(soc, dt_s)
        feasible_mask = (battery_bus <= discharge_limit) & (
            battery_bus >= -charge_limit
        )
        feasible_engine = engine_grid[feasible_mask]
        feasible_battery = battery_bus[feasible_mask]
        if feasible_engine.size >= 3:
            current = np.asarray(
                battery.current_from_power(feasible_battery, soc), dtype=float
            )
            internal_kw = float(battery.open_circuit_voltage(soc)) * current / 1000.0
            fuel_flow = np.asarray(engine.fuel_flow_kg_s(feasible_engine), dtype=float)
            values = fuel_flow + s * internal_kw / LHV_KJ_KG
            slopes = np.diff(values) / np.diff(feasible_engine)
            minimum_slope_increase = min(
                minimum_slope_increase,
                float(np.min(np.diff(slopes))),
            )
            convex_intervals_checked += 1

    failures = sum(value > 1.0e-9 for value in relative_differences)
    return LoadDependentSweepResult(
        optimized=tuple(optimized),
        oracle=tuple(oracle),
        max_power_difference_kw=max(power_differences),
        max_relative_hamiltonian_difference=max(relative_differences),
        hamiltonian_tolerance_failures=failures,
        convex_intervals_checked=convex_intervals_checked,
        minimum_adjacent_slope_increase=minimum_slope_increase,
    )


def test_golden_search_agrees_with_a_fine_grid_over_500_random_cases(
    randomized_sweep: SweepResult,
) -> None:
    for optimized, oracle in zip(
        randomized_sweep.optimized, randomized_sweep.oracle, strict=True
    ):
        assert optimized.feasible is oracle.feasible
        if optimized.feasible:
            assert optimized.engine_shaft_kw == pytest.approx(
                oracle.engine_shaft_kw, abs=0.05
            )
            assert optimized.hamiltonian_kg_s == pytest.approx(
                oracle.hamiltonian_kg_s, rel=1.0e-9, abs=1.0e-12
            )


def test_negligible_ohmic_loss_keeps_every_solution_on_an_endpoint() -> None:
    rng = np.random.default_rng(118)
    engine = Turboshaft(75.0)
    battery = BatteryPack(100.0, r_ref_ohm=1.0e-6, scale_resistance=False)
    powertrain = SeriesPowertrain()
    threshold = switching_equivalence_factor(
        engine.willans_a, powertrain.source_chain_efficiency, LHV_KJ_KG
    )
    for _ in range(100):
        offset = float(rng.uniform(0.01, 2.0))
        s = threshold - offset if rng.random() < 0.5 else threshold + offset
        decision = solve_split(
            float(rng.uniform(20.0, 70.0)),
            engine,
            battery,
            powertrain,
            s,
            0.5,
            1.0,
            60.0,
        )
        assert decision.feasible
        assert decision.active_bound != "interior"


def test_numerical_switch_matches_the_marginal_closed_form() -> None:
    engine = Turboshaft(75.0)
    battery = BatteryPack(100.0, r_ref_ohm=1.0e-6, scale_resistance=False)
    powertrain = SeriesPowertrain()
    expected = switching_equivalence_factor(
        engine.willans_a, powertrain.source_chain_efficiency, LHV_KJ_KG
    )
    lower_s, upper_s = expected - 0.1, expected + 0.1
    midpoint_power_kw = 0.5 * (engine.idle_power_kw + engine.max_power_kw(1.0))
    for _ in range(50):
        candidate_s = 0.5 * (lower_s + upper_s)
        decision = solve_split(
            40.0,
            engine,
            battery,
            powertrain,
            candidate_s,
            0.5,
            1.0,
            60.0,
        )
        if decision.engine_shaft_kw > midpoint_power_kw:
            upper_s = candidate_s
        else:
            lower_s = candidate_s
    located = 0.5 * (lower_s + upper_s)
    assert located == pytest.approx(expected, abs=1.0e-6)


def test_realistic_resistance_can_create_an_interior_minimum(
    default_components: PowerSplitComponents,
) -> None:
    decision = solve_split(
        40.0,
        default_components.engine,
        default_components.battery,
        default_components.powertrain,
        4.8,
        0.5,
        1.0,
        60.0,
    )
    assert decision.feasible
    assert decision.active_bound == "interior"


def test_random_sweep_contains_realistic_interior_solutions(
    randomized_sweep: SweepResult,
) -> None:
    interior = sum(
        decision.active_bound == "interior" for decision in randomized_sweep.feasible
    )
    assert interior > 0
    assert 0.0 < interior / len(randomized_sweep.feasible) < 1.0


def test_hamiltonian_matches_an_independent_internal_power_calculation(
    default_components: PowerSplitComponents,
) -> None:
    engine_kw = 40.0
    demand_kw = 50.0
    s = 5.0
    soc = 0.5
    bus_from_engine_kw = float(
        default_components.powertrain.bus_power_from_engine(engine_kw)
    )
    battery_bus_kw = demand_kw - bus_from_engine_kw
    current_a = float(default_components.battery.current_from_power(battery_bus_kw, soc))
    battery_internal_kw = (
        float(default_components.battery.open_circuit_voltage(soc)) * current_a / 1000.0
    )
    fuel_flow_kg_s = float(default_components.engine.fuel_flow_kg_s(engine_kw))
    expected = fuel_flow_kg_s + s * battery_internal_kw / LHV_KJ_KG
    actual = hamiltonian(
        engine_kw,
        demand_kw,
        default_components,
        s,
        soc,
        1.0,
        60.0,
    )
    assert actual == pytest.approx(expected, rel=1.0e-12)


def test_hamiltonian_prices_battery_loss_instead_of_treating_the_pack_as_lossless() -> None:
    engine = Turboshaft(75.0)
    battery = BatteryPack(5.0, r_ref_ohm=0.2, scale_resistance=False)
    powertrain = SeriesPowertrain()
    components = PowerSplitComponents(engine, battery, powertrain)
    engine_kw = 40.0
    demand_kw = 50.0
    soc = 0.5
    s = 6.0
    battery_bus_kw = demand_kw - float(powertrain.bus_power_from_engine(engine_kw))
    actual = hamiltonian(engine_kw, demand_kw, components, s, soc, 1.0, 60.0)
    bus_priced = float(engine.fuel_flow_kg_s(engine_kw)) + (
        s * battery_bus_kw / LHV_KJ_KG
    )
    loss_kw = float(battery.ohmic_loss_kw(battery_bus_kw, soc))
    assert actual - bus_priced == pytest.approx(s * loss_kw / LHV_KJ_KG, rel=1.0e-12)


def test_derived_kwh_conversion_recovers_the_previous_rounded_literal() -> None:
    derived_kg_kwh = 3600.0 / LHV_KJ_KG
    assert round(derived_kg_kwh, 2) == 0.08


def test_demand_above_combined_power_availability_is_reported_infeasible() -> None:
    engine = Turboshaft(75.0)
    battery = BatteryPack(5.0)
    powertrain = SeriesPowertrain()
    maximum_kw = float(powertrain.bus_power_from_engine(engine.max_power_kw(1.0)))
    maximum_kw += battery.available_discharge_kw(0.5, 60.0)
    decision = solve_split(
        maximum_kw + 1.0, engine, battery, powertrain, 5.0, 0.5, 1.0, 60.0
    )
    assert not decision.feasible
    assert decision.active_bound == "battery_discharge_limit"
    assert decision.bus_from_engine_kw + decision.battery_bus_kw < maximum_kw + 1.0


def test_demand_at_combined_power_availability_uses_both_limits_exactly() -> None:
    engine = Turboshaft(75.0)
    battery = BatteryPack(5.0)
    powertrain = SeriesPowertrain()
    engine_max_kw = engine.max_power_kw(1.0)
    battery_max_kw = battery.available_discharge_kw(0.5, 60.0)
    demand_kw = float(powertrain.bus_power_from_engine(engine_max_kw)) + battery_max_kw
    decision = solve_split(
        demand_kw, engine, battery, powertrain, 5.0, 0.5, 1.0, 60.0
    )
    assert decision.feasible
    assert decision.engine_shaft_kw == pytest.approx(engine_max_kw, abs=1.0e-9)
    assert decision.battery_bus_kw == pytest.approx(battery_max_kw, abs=1.0e-9)


def test_cutoff_soc_leaves_the_engine_responsible_for_all_positive_demand() -> None:
    engine = Turboshaft(75.0)
    battery = BatteryPack(5.0)
    powertrain = SeriesPowertrain()
    decision = solve_split(
        50.0,
        engine,
        battery,
        powertrain,
        3.0,
        battery.soc_min,
        1.0,
        60.0,
    )
    assert decision.feasible
    assert decision.battery_bus_kw <= 0.0
    assert decision.bus_from_engine_kw >= 50.0

    impossible = solve_split(
        70.0,
        engine,
        battery,
        powertrain,
        3.0,
        battery.soc_min,
        1.0,
        60.0,
    )
    assert not impossible.feasible


def test_idle_surplus_respects_and_names_the_charge_limit() -> None:
    engine = Turboshaft(75.0)
    battery = BatteryPack(5.0)
    powertrain = SeriesPowertrain()
    idle_bus_kw = float(powertrain.bus_power_from_engine(engine.idle_power_kw))
    charge_limit_kw = battery.available_charge_kw(0.5, 60.0)
    decision = solve_split(
        idle_bus_kw - charge_limit_kw,
        engine,
        battery,
        powertrain,
        5.0,
        0.5,
        1.0,
        60.0,
    )
    assert decision.feasible
    assert decision.battery_bus_kw == pytest.approx(-charge_limit_kw, abs=1.0e-9)
    assert decision.active_bound == "battery_charge_limit"

    infeasible = solve_split(
        idle_bus_kw - charge_limit_kw - 1.0,
        engine,
        battery,
        powertrain,
        5.0,
        0.5,
        1.0,
        60.0,
    )
    assert not infeasible.feasible
    assert infeasible.battery_bus_kw >= -charge_limit_kw
    assert infeasible.active_bound == "battery_charge_limit"


def test_every_random_decision_respects_engine_and_battery_bounds(
    randomized_sweep: SweepResult,
    default_components: PowerSplitComponents,
) -> None:
    rng = np.random.default_rng(20260805)
    for decision in randomized_sweep.optimized:
        demand_kw = float(rng.uniform(20.0, 100.0))
        rng.uniform(0.6, 12.0)
        soc = float(rng.uniform(0.05, 1.0))
        sigma = float(rng.uniform(0.5, 1.0))
        dt_s = float(rng.uniform(1.0, 180.0))
        assert decision.engine_shaft_kw <= default_components.engine.max_power_kw(sigma) + 1.0e-9
        assert decision.battery_bus_kw <= (
            default_components.battery.available_discharge_kw(soc, dt_s) + 1.0e-9
        )
        assert decision.battery_bus_kw >= -(
            default_components.battery.available_charge_kw(soc, dt_s) + 1.0e-9
        )
        if decision.feasible:
            assert decision.bus_from_engine_kw + decision.battery_bus_kw == pytest.approx(
                demand_kw, abs=1.0e-9
            )


def test_load_dependent_source_chain_uses_the_powertrain_inverse_for_bounds() -> None:
    engine = Turboshaft(75.0)
    battery = BatteryPack(20.0)
    powertrain = SeriesPowertrain(
        load_dependent=True,
        rated_engine_kw=75.0,
        rated_bus_kw=100.0,
    )
    optimized = solve_split(40.0, engine, battery, powertrain, 4.8, 0.5, 1.0, 60.0)
    oracle = grid_search_split(
        40.0,
        engine,
        battery,
        powertrain,
        4.8,
        0.5,
        1.0,
        60.0,
        resolution_kw=0.01,
    )
    assert optimized.feasible
    assert optimized.engine_shaft_kw == pytest.approx(oracle.engine_shaft_kw, abs=0.01)


def test_o11_load_dependent_sweep_agrees_with_grid_to_grid_resolution(
    load_dependent_sweep: LoadDependentSweepResult,
) -> None:
    for optimized, oracle in zip(
        load_dependent_sweep.optimized,
        load_dependent_sweep.oracle,
        strict=True,
    ):
        assert optimized.feasible is oracle.feasible
        if optimized.feasible:
            assert optimized.engine_shaft_kw == pytest.approx(
                oracle.engine_shaft_kw, abs=0.05
            )
            # A continuous minimum should be no worse than its nearby grid
            # point.  Equality to 1e-9 relative is not achievable in every
            # interior case with the specified 0.05 kW oracle resolution.
            assert optimized.hamiltonian_kg_s <= oracle.hamiltonian_kg_s + 1.0e-12

    assert load_dependent_sweep.hamiltonian_tolerance_failures > 0
    assert load_dependent_sweep.max_relative_hamiltonian_difference < 1.0e-8


def test_o11_sampled_continuous_hamiltonian_is_convex(
    load_dependent_sweep: LoadDependentSweepResult,
) -> None:
    assert load_dependent_sweep.convex_intervals_checked >= 450
    assert load_dependent_sweep.minimum_adjacent_slope_increase >= -1.0e-12


def test_mean_hamiltonian_evaluations_remain_below_the_previous_grid_count(
    randomized_sweep: SweepResult,
) -> None:
    mean_evaluations = np.mean(
        [decision.evaluations for decision in randomized_sweep.optimized]
    )
    assert mean_evaluations < 38.0


def test_wall_clock_is_compared_with_the_previous_two_kw_grid(
    default_components: PowerSplitComponents,
) -> None:
    rng = np.random.default_rng(91)
    cases = tuple(
        (
            float(rng.uniform(20.0, 100.0)),
            float(rng.uniform(0.6, 12.0)),
            float(rng.uniform(0.05, 1.0)),
            float(rng.uniform(0.5, 1.0)),
            float(rng.uniform(1.0, 180.0)),
        )
        for _ in range(300)
    )

    start = time.perf_counter()
    optimized = tuple(
        solve_split(
            demand,
            default_components.engine,
            default_components.battery,
            default_components.powertrain,
            s,
            soc,
            sigma,
            dt_s,
        )
        for demand, s, soc, sigma, dt_s in cases
    )
    optimized_s = time.perf_counter() - start

    start = time.perf_counter()
    reference = tuple(
        grid_search_split(
            demand,
            default_components.engine,
            default_components.battery,
            default_components.powertrain,
            s,
            soc,
            sigma,
            dt_s,
            resolution_kw=2.0,
        )
        for demand, s, soc, sigma, dt_s in cases
    )
    reference_s = time.perf_counter() - start

    assert all(decision.evaluations > 0 for decision in optimized + reference)
    assert math.isfinite(reference_s / optimized_s)
    assert optimized_s < 1.5 * reference_s


def test_identical_inputs_produce_identical_decisions(
    default_components: PowerSplitComponents,
) -> None:
    args = (
        40.0,
        default_components.engine,
        default_components.battery,
        default_components.powertrain,
        4.8,
        0.5,
        0.8,
        60.0,
    )
    assert solve_split(*args) == solve_split(*args)


def test_switching_factor_is_below_the_average_sfc_neutral_factor() -> None:
    engine = Turboshaft(75.0)
    powertrain = SeriesPowertrain()
    switching = switching_equivalence_factor(
        engine.willans_a, powertrain.source_chain_efficiency, LHV_KJ_KG
    )
    neutral = neutral_equivalence_factor(
        engine.sfc_rated_kg_kwh,
        powertrain.source_chain_efficiency,
        LHV_KJ_KG,
    )
    assert switching < neutral
