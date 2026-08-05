"""Equivalent-consumption minimisation for the series-hybrid power split.

The equivalence factor is supplied by a controller.  This module intersects
the engine and battery limits, prices feasible splits with the ECMS
Hamiltonian, and returns the minimum without carrying state between calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.models.battery import BatteryPack
from src.models.engine import LHV_KJ_KG, Turboshaft
from src.models.powertrain import SeriesPowertrain

__all__ = [
    "PowerSplitComponents",
    "SplitDecision",
    "grid_search_split",
    "hamiltonian",
    "solve_split",
    "switching_equivalence_factor",
]

_GOLDEN_CONJUGATE = (math.sqrt(5.0) - 1.0) / 2.0
_WATTS_PER_KW = 1000.0
_SECONDS_PER_HOUR = 3600.0
_NUMERICAL_EPS_KW = 1.0e-10
_DERIVATIVE_PROBE_FRACTION = 0.1


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


@dataclass(frozen=True)
class PowerSplitComponents:
    """Models required to evaluate one ECMS Hamiltonian."""

    engine: Turboshaft
    battery: BatteryPack
    powertrain: SeriesPowertrain


@dataclass(frozen=True)
class SplitDecision:
    """Selected engine and battery powers for one simulation step."""

    engine_shaft_kw: float
    bus_from_engine_kw: float
    battery_bus_kw: float
    battery_internal_kw: float
    fuel_flow_kg_s: float
    hamiltonian_kg_s: float
    feasible: bool
    engine_off: bool
    engine_at_idle: bool
    active_bound: str
    evaluations: int = 0


@dataclass(frozen=True, slots=True)
class _Problem:
    demand_kw: float
    s: float
    soc: float
    sigma: float
    dt_s: float
    engine: Turboshaft
    battery: BatteryPack
    powertrain: SeriesPowertrain
    engine_max_kw: float
    engine_on_min_kw: float | None
    discharge_limit_kw: float
    charge_limit_kw: float
    battery_ocv_v: float
    fuel_slope_kg_s_kw: float
    fuel_intercept_kg_s: float
    continuous_lower_kw: float | None
    continuous_upper_kw: float | None


def switching_equivalence_factor(
    willans_a: float,
    source_chain_efficiency: float,
    lhv_kj_kg: float,
) -> float:
    """Marginal ECMS switching factor with constant source efficiency."""
    slope = _finite("willans_a", willans_a)
    efficiency = _finite("source_chain_efficiency", source_chain_efficiency)
    lhv = _finite("lhv_kj_kg", lhv_kj_kg)
    if slope <= 0.0:
        raise ValueError(f"willans_a must be positive, got {willans_a!r}")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(
            "source_chain_efficiency must lie in (0, 1], "
            f"got {source_chain_efficiency!r}"
        )
    if lhv <= 0.0:
        raise ValueError(f"lhv_kj_kg must be positive, got {lhv_kj_kg!r}")
    return slope * lhv / (_SECONDS_PER_HOUR * efficiency)


def _unpack_components(
    components: PowerSplitComponents
    | tuple[Turboshaft, BatteryPack, SeriesPowertrain],
) -> tuple[Turboshaft, BatteryPack, SeriesPowertrain]:
    if isinstance(components, PowerSplitComponents):
        return components.engine, components.battery, components.powertrain
    try:
        engine, battery, powertrain = components
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "aircraft_components must contain engine, battery, and powertrain"
        ) from exc
    return engine, battery, powertrain


def _validate_common(
    bus_demand_kw: float,
    s: float,
    soc: float,
    sigma: float,
    dt_s: float,
) -> tuple[float, float, float, float, float]:
    demand = _finite("bus_demand_kw", bus_demand_kw)
    factor = _finite("s", s)
    state_of_charge = _finite("soc", soc)
    density_ratio = _finite("sigma", sigma)
    duration = _finite("dt_s", dt_s)
    if demand < 0.0:
        raise ValueError(f"bus_demand_kw must be non-negative, got {bus_demand_kw!r}")
    if factor <= 0.0:
        raise ValueError(f"s must be positive, got {s!r}")
    if not 0.0 <= state_of_charge <= 1.0:
        raise ValueError(f"soc must lie in [0, 1], got {soc!r}")
    if not 0.0 < density_ratio <= 1.0:
        raise ValueError(f"sigma must lie in (0, 1], got {sigma!r}")
    if duration <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s!r}")
    return demand, factor, state_of_charge, density_ratio, duration


def _fuel_flow_kg_s(problem: _Problem, engine_shaft_kw: float) -> float:
    if engine_shaft_kw <= _NUMERICAL_EPS_KW and problem.engine.allow_shutdown:
        return 0.0
    return (
        problem.fuel_slope_kg_s_kw * engine_shaft_kw
        + problem.fuel_intercept_kg_s
    )


def _battery_internal_kw(problem: _Problem, battery_bus_kw: float) -> float:
    current_a = float(problem.battery.current_from_power(battery_bus_kw, problem.soc))
    return problem.battery_ocv_v * current_a / _WATTS_PER_KW


def _objective(problem: _Problem, engine_shaft_kw: float) -> float:
    bus_from_engine_kw = float(
        problem.powertrain.bus_power_from_engine(engine_shaft_kw)
    )
    battery_bus_kw = problem.demand_kw - bus_from_engine_kw
    battery_internal_kw = _battery_internal_kw(problem, battery_bus_kw)
    return _fuel_flow_kg_s(problem, engine_shaft_kw) + (
        problem.s * battery_internal_kw / LHV_KJ_KG
    )


def hamiltonian(
    engine_shaft_kw: float,
    bus_demand_kw: float,
    aircraft_components: PowerSplitComponents
    | tuple[Turboshaft, BatteryPack, SeriesPowertrain],
    s: float,
    soc: float,
    sigma: float,
    dt_s: float,
) -> float:
    """ECMS equivalent fuel rate for one candidate engine power [kg/s]."""
    engine_kw = _finite("engine_shaft_kw", engine_shaft_kw)
    if engine_kw < 0.0:
        raise ValueError(
            f"engine_shaft_kw must be non-negative, got {engine_shaft_kw!r}"
        )
    demand, factor, state_of_charge, density_ratio, duration = _validate_common(
        bus_demand_kw, s, soc, sigma, dt_s
    )
    engine, battery, powertrain = _unpack_components(aircraft_components)
    engine.max_power_kw(density_ratio)
    problem = _problem_without_bounds(
        demand,
        engine,
        battery,
        powertrain,
        factor,
        state_of_charge,
        density_ratio,
        duration,
    )
    return _objective(problem, engine_kw)


def _problem_without_bounds(
    demand_kw: float,
    engine: Turboshaft,
    battery: BatteryPack,
    powertrain: SeriesPowertrain,
    s: float,
    soc: float,
    sigma: float,
    dt_s: float,
) -> _Problem:
    engine_max_kw = engine.max_power_kw(sigma)
    return _Problem(
        demand_kw=demand_kw,
        s=s,
        soc=soc,
        sigma=sigma,
        dt_s=dt_s,
        engine=engine,
        battery=battery,
        powertrain=powertrain,
        engine_max_kw=engine_max_kw,
        engine_on_min_kw=None,
        discharge_limit_kw=0.0,
        charge_limit_kw=0.0,
        battery_ocv_v=float(battery.open_circuit_voltage(soc)),
        fuel_slope_kg_s_kw=engine.willans_a / _SECONDS_PER_HOUR,
        fuel_intercept_kg_s=engine.willans_b / _SECONDS_PER_HOUR,
        continuous_lower_kw=None,
        continuous_upper_kw=None,
    )


def _prepare_problem(
    bus_demand_kw: float,
    engine: Turboshaft,
    battery: BatteryPack,
    powertrain: SeriesPowertrain,
    s: float,
    soc: float,
    sigma: float,
    dt_s: float,
) -> _Problem:
    demand, factor, state_of_charge, density_ratio, duration = _validate_common(
        bus_demand_kw, s, soc, sigma, dt_s
    )
    engine_max_kw = engine.max_power_kw(density_ratio)
    discharge_limit_kw = battery.available_discharge_kw(state_of_charge, duration)
    charge_limit_kw = battery.available_charge_kw(state_of_charge, duration)

    required_engine_bus_kw = max(demand - discharge_limit_kw, 0.0)
    lower_from_battery_kw = (
        float(powertrain.engine_power_for_bus(required_engine_bus_kw))
        if required_engine_bus_kw > 0.0
        else 0.0
    )
    upper_from_battery_kw = float(
        powertrain.engine_power_for_bus(demand + charge_limit_kw)
    )

    if engine_max_kw >= engine.idle_power_kw:
        engine_on_min_kw: float | None = engine.idle_power_kw
    elif engine.allow_shutdown:
        engine_on_min_kw = None
    else:
        engine_on_min_kw = engine_max_kw

    lower_kw: float | None = None
    upper_kw: float | None = None
    if engine_on_min_kw is not None:
        candidate_lower_kw = max(engine_on_min_kw, lower_from_battery_kw)
        candidate_upper_kw = min(engine_max_kw, upper_from_battery_kw)
        if candidate_lower_kw <= candidate_upper_kw + _NUMERICAL_EPS_KW:
            lower_kw = min(candidate_lower_kw, candidate_upper_kw)
            upper_kw = candidate_upper_kw

    return _Problem(
        demand_kw=demand,
        s=factor,
        soc=state_of_charge,
        sigma=density_ratio,
        dt_s=duration,
        engine=engine,
        battery=battery,
        powertrain=powertrain,
        engine_max_kw=engine_max_kw,
        engine_on_min_kw=engine_on_min_kw,
        discharge_limit_kw=discharge_limit_kw,
        charge_limit_kw=charge_limit_kw,
        battery_ocv_v=float(battery.open_circuit_voltage(state_of_charge)),
        fuel_slope_kg_s_kw=engine.willans_a / _SECONDS_PER_HOUR,
        fuel_intercept_kg_s=engine.willans_b / _SECONDS_PER_HOUR,
        continuous_lower_kw=lower_kw,
        continuous_upper_kw=upper_kw,
    )


def _engine_candidate_is_feasible(problem: _Problem, engine_kw: float) -> bool:
    if engine_kw < -_NUMERICAL_EPS_KW or engine_kw > problem.engine_max_kw + _NUMERICAL_EPS_KW:
        return False
    battery_kw = problem.demand_kw - float(
        problem.powertrain.bus_power_from_engine(engine_kw)
    )
    return (
        battery_kw <= problem.discharge_limit_kw + _NUMERICAL_EPS_KW
        and battery_kw >= -problem.charge_limit_kw - _NUMERICAL_EPS_KW
    )


def _active_bound(
    problem: _Problem,
    engine_kw: float,
    battery_kw: float,
    tolerance_kw: float,
) -> str:
    tolerance = max(tolerance_kw, _NUMERICAL_EPS_KW)
    if engine_kw <= tolerance and problem.engine.allow_shutdown:
        return "engine_off"
    if abs(battery_kw - problem.discharge_limit_kw) <= tolerance:
        return "battery_discharge_limit"
    if abs(battery_kw + problem.charge_limit_kw) <= tolerance:
        return "battery_charge_limit"
    if abs(engine_kw - problem.engine_max_kw) <= tolerance:
        return "engine_max"
    if (
        problem.engine_on_min_kw is not None
        and abs(engine_kw - problem.engine_on_min_kw) <= tolerance
    ):
        return "engine_min"
    return "interior"


def _decision(
    problem: _Problem,
    engine_kw: float,
    evaluations: int,
    tolerance_kw: float,
    *,
    feasible: bool,
    battery_kw_override: float | None = None,
    active_bound_override: str | None = None,
) -> SplitDecision:
    bus_from_engine_kw = float(problem.powertrain.bus_power_from_engine(engine_kw))
    battery_bus_kw = (
        problem.demand_kw - bus_from_engine_kw
        if battery_kw_override is None
        else battery_kw_override
    )
    battery_internal_kw = _battery_internal_kw(problem, battery_bus_kw)
    fuel_flow_kg_s = _fuel_flow_kg_s(problem, engine_kw)
    hamiltonian_kg_s = fuel_flow_kg_s + (
        problem.s * battery_internal_kw / LHV_KJ_KG
    )
    engine_off = engine_kw <= _NUMERICAL_EPS_KW and problem.engine.allow_shutdown
    engine_at_idle = (
        not engine_off
        and engine_kw <= problem.engine.idle_power_kw + max(tolerance_kw, _NUMERICAL_EPS_KW)
    )
    active_bound = active_bound_override or _active_bound(
        problem, engine_kw, battery_bus_kw, tolerance_kw
    )
    return SplitDecision(
        engine_shaft_kw=engine_kw,
        bus_from_engine_kw=bus_from_engine_kw,
        battery_bus_kw=battery_bus_kw,
        battery_internal_kw=battery_internal_kw,
        fuel_flow_kg_s=fuel_flow_kg_s,
        hamiltonian_kg_s=hamiltonian_kg_s,
        feasible=feasible,
        engine_off=engine_off,
        engine_at_idle=engine_at_idle,
        active_bound=active_bound,
        evaluations=evaluations,
    )


def _infeasible_decision(problem: _Problem, tolerance_kw: float) -> SplitDecision:
    if problem.engine.allow_shutdown:
        engine_points = [0.0]
        if problem.engine_on_min_kw is not None:
            engine_points.extend((problem.engine_on_min_kw, problem.engine_max_kw))
    else:
        engine_points = [float(problem.engine_on_min_kw), problem.engine_max_kw]

    best_engine_kw = engine_points[0]
    best_battery_kw = 0.0
    best_residual_kw = math.inf
    best_hamiltonian = math.inf
    best_bound = "infeasible"
    evaluations = 0
    for engine_kw in dict.fromkeys(engine_points):
        bus_from_engine_kw = float(problem.powertrain.bus_power_from_engine(engine_kw))
        requested_battery_kw = problem.demand_kw - bus_from_engine_kw
        battery_kw = min(
            max(requested_battery_kw, -problem.charge_limit_kw),
            problem.discharge_limit_kw,
        )
        residual_kw = abs(bus_from_engine_kw + battery_kw - problem.demand_kw)
        battery_internal_kw = _battery_internal_kw(problem, battery_kw)
        candidate_hamiltonian = _fuel_flow_kg_s(problem, engine_kw) + (
            problem.s * battery_internal_kw / LHV_KJ_KG
        )
        evaluations += 1
        if requested_battery_kw > problem.discharge_limit_kw:
            bound = "battery_discharge_limit"
        elif requested_battery_kw < -problem.charge_limit_kw:
            bound = "battery_charge_limit"
        else:
            bound = "infeasible"
        if (residual_kw, candidate_hamiltonian) < (best_residual_kw, best_hamiltonian):
            best_engine_kw = engine_kw
            best_battery_kw = battery_kw
            best_residual_kw = residual_kw
            best_hamiltonian = candidate_hamiltonian
            best_bound = bound

    return _decision(
        problem,
        best_engine_kw,
        evaluations,
        tolerance_kw,
        feasible=False,
        battery_kw_override=best_battery_kw,
        active_bound_override=best_bound,
    )


def _initial_candidates(problem: _Problem) -> tuple[float, ...]:
    candidates: list[float] = []
    if problem.engine.allow_shutdown and _engine_candidate_is_feasible(problem, 0.0):
        candidates.append(0.0)
    if problem.continuous_lower_kw is not None:
        lower_kw = problem.continuous_lower_kw
        upper_kw = float(problem.continuous_upper_kw)
        if (
            problem.engine_on_min_kw is not None
            and lower_kw - _NUMERICAL_EPS_KW
            <= problem.engine_on_min_kw
            <= upper_kw + _NUMERICAL_EPS_KW
        ):
            candidates.append(problem.engine_on_min_kw)
        candidates.extend((lower_kw, upper_kw))
    return tuple(dict.fromkeys(candidates))


def solve_split(
    bus_demand_kw: float,
    engine: Turboshaft,
    battery: BatteryPack,
    powertrain: SeriesPowertrain,
    s: float,
    soc: float,
    sigma: float,
    dt_s: float,
    tolerance_kw: float = 0.01,
) -> SplitDecision:
    """Minimise the ECMS Hamiltonian over all feasible power splits.

    The continuous engine-on Hamiltonian is convex for both powertrain modes
    implemented here.  One-sided finite differences therefore resolve the
    overwhelmingly common boundary optima before the golden-section search is
    entered.  The derivative probe is deliberately smaller than the requested
    power tolerance, so a sign obscured by the finite difference can move the
    returned power by no more than an already-accepted numerical tolerance.
    """
    tolerance = _finite("tolerance_kw", tolerance_kw)
    if tolerance <= 0.0:
        raise ValueError(f"tolerance_kw must be positive, got {tolerance_kw!r}")
    problem = _prepare_problem(
        bus_demand_kw, engine, battery, powertrain, s, soc, sigma, dt_s
    )
    candidates = _initial_candidates(problem)
    if not candidates:
        return _infeasible_decision(problem, tolerance)

    evaluations = 0
    lower_kw = problem.continuous_lower_kw
    upper_kw = problem.continuous_upper_kw
    lower_value = math.inf
    upper_value = math.inf
    best_engine_kw = candidates[0]
    best_hamiltonian = math.inf
    for engine_kw in candidates:
        value = _objective(problem, engine_kw)
        evaluations += 1
        if lower_kw is not None and engine_kw == lower_kw:
            lower_value = value
        if upper_kw is not None and engine_kw == upper_kw:
            upper_value = value
        if value < best_hamiltonian:
            best_engine_kw = engine_kw
            best_hamiltonian = value

    search_interior = False
    if lower_kw is not None and upper_kw - lower_kw > tolerance:
        interval_kw = upper_kw - lower_kw
        probe_kw = min(
            0.5 * interval_kw,
            max(_DERIVATIVE_PROBE_FRACTION * tolerance, 1.0e-6),
        )

        lower_probe_kw = lower_kw + probe_kw
        lower_probe_value = _objective(problem, lower_probe_kw)
        evaluations += 1
        lower_derivative = (lower_probe_value - lower_value) / probe_kw

        if lower_derivative < 0.0:
            upper_probe_kw = upper_kw - probe_kw
            upper_probe_value = _objective(problem, upper_probe_kw)
            evaluations += 1
            upper_derivative = (upper_value - upper_probe_value) / probe_kw
            search_interior = upper_derivative > 0.0

    if search_interior:
        left_kw = lower_kw
        right_kw = upper_kw
        first_kw = right_kw - _GOLDEN_CONJUGATE * (right_kw - left_kw)
        second_kw = left_kw + _GOLDEN_CONJUGATE * (right_kw - left_kw)
        first_value = _objective(problem, first_kw)
        second_value = _objective(problem, second_kw)
        evaluations += 2
        if first_value < best_hamiltonian:
            best_engine_kw, best_hamiltonian = first_kw, first_value
        if second_value < best_hamiltonian:
            best_engine_kw, best_hamiltonian = second_kw, second_value

        while right_kw - left_kw > tolerance:
            if first_value <= second_value:
                right_kw = second_kw
                second_kw, second_value = first_kw, first_value
                first_kw = right_kw - _GOLDEN_CONJUGATE * (right_kw - left_kw)
                first_value = _objective(problem, first_kw)
                value_kw, value = first_kw, first_value
            else:
                left_kw = first_kw
                first_kw, first_value = second_kw, second_value
                second_kw = left_kw + _GOLDEN_CONJUGATE * (right_kw - left_kw)
                second_value = _objective(problem, second_kw)
                value_kw, value = second_kw, second_value
            evaluations += 1
            if value < best_hamiltonian:
                best_engine_kw, best_hamiltonian = value_kw, value

    return _decision(
        problem,
        best_engine_kw,
        evaluations,
        tolerance,
        feasible=True,
    )


def grid_search_split(
    bus_demand_kw: float,
    engine: Turboshaft,
    battery: BatteryPack,
    powertrain: SeriesPowertrain,
    s: float,
    soc: float,
    sigma: float,
    dt_s: float,
    resolution_kw: float = 0.05,
) -> SplitDecision:
    """Fine-grid test oracle; intentionally unsuitable for production use."""
    resolution = _finite("resolution_kw", resolution_kw)
    if resolution <= 0.0:
        raise ValueError(f"resolution_kw must be positive, got {resolution_kw!r}")
    problem = _prepare_problem(
        bus_demand_kw, engine, battery, powertrain, s, soc, sigma, dt_s
    )
    candidates: list[float] = []
    if problem.engine.allow_shutdown:
        candidates.append(0.0)
    if problem.engine_on_min_kw is not None:
        engine_kw = problem.engine_on_min_kw
        while engine_kw < problem.engine_max_kw:
            candidates.append(engine_kw)
            engine_kw += resolution
        candidates.append(problem.engine_max_kw)
    candidates.extend(_initial_candidates(problem))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return _infeasible_decision(problem, resolution)

    best_engine_kw = 0.0
    best_hamiltonian = math.inf
    found_feasible = False
    evaluations = 0
    for engine_kw in candidates:
        value = _objective(problem, engine_kw)
        evaluations += 1
        if _engine_candidate_is_feasible(problem, engine_kw) and value < best_hamiltonian:
            best_engine_kw = engine_kw
            best_hamiltonian = value
            found_feasible = True
    if not found_feasible:
        return _infeasible_decision(problem, resolution)
    return _decision(
        problem,
        best_engine_kw,
        evaluations,
        max(resolution * 1.0e-6, _NUMERICAL_EPS_KW),
        feasible=True,
    )
