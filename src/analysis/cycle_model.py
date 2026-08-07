"""Analytical engine-cycling diagnostics for a series-hybrid powertrain.

The model assumes an affine fuel law, constant charge and discharge
efficiencies evaluated at stated operating points, constant DC-bus demand over
one charge-sustaining cycle, no restart cost, and no minimum ON or OFF dwell.
Its results are exact only while all of those assumptions hold.  Power is in
kW, energy in kWh, fuel rate in kg/h, and duration in seconds at the public
boundary.

Under these assumptions, a cycle that keeps the engine running in both phases
cannot save fuel.  The Willans intercept is then paid throughout, while the
linear power term also pays the battery round-trip loss.  Cycling benefit
therefore requires a genuine zero-fuel engine-OFF phase; an idle fallback is a
loss, not a reduced version of the shutdown benefit.

The functions are stateless and independent of ``src.control`` and
``src.simulation``.  Uncalibrated study inputs have no computational defaults;
callers must state them explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

__all__ = [
    "BatteryAssistedOptimum",
    "CycleEnergyBalance",
    "CycleOptimum",
    "DiagnosticParameters",
    "OperatingRegime",
    "RegimeClassification",
    "analytical_cycle_energy_balance",
    "battery_assisted_duration_h",
    "battery_energy_for_equal_duration_kwh",
    "classify_regime",
    "continuous_fuel_rate_kg_h",
    "cycle_average_fuel_rate_kg_h",
    "cycle_period_s",
    "duty_cycle",
    "economic_cycling_threshold_kw",
    "optimal_battery_assisted_power",
    "optimal_engine_on_power",
    "two_level_cycle_fuel_rate",
    "two_level_penalty_vs_continuous",
    "willans_coefficients",
]

SECONDS_PER_HOUR = 3600.0


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result


def _efficiency(name: str, value: float) -> float:
    result = _finite(name, value)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1], got {value!r}")
    return result


@dataclass(frozen=True)
class DiagnosticParameters:
    """Explicit uncalibrated inputs carried on every diagnostic artifact."""

    idle_fuel_fraction: float
    charge_c_rate: float
    discharge_c_rate: float
    restart_fuel_kg: float
    minimum_on_time_s: float
    minimum_off_time_s: float

    def __post_init__(self) -> None:
        idle = _finite("idle_fuel_fraction", self.idle_fuel_fraction)
        if not 0.0 <= idle < 1.0:
            raise ValueError(
                "idle_fuel_fraction must lie in [0, 1), "
                f"got {self.idle_fuel_fraction!r}"
            )
        _positive("charge_c_rate", self.charge_c_rate)
        _positive("discharge_c_rate", self.discharge_c_rate)
        _nonnegative("restart_fuel_kg", self.restart_fuel_kg)
        _positive("minimum_on_time_s", self.minimum_on_time_s)
        _positive("minimum_off_time_s", self.minimum_off_time_s)

    def label(self) -> str:
        """Compact label suitable for a figure or table caption."""
        return (
            f"idle={self.idle_fuel_fraction:.2f}; charge={self.charge_c_rate:g}C; "
            f"discharge={self.discharge_c_rate:g}C; restart={self.restart_fuel_kg:g} kg; "
            f"min ON/OFF={self.minimum_on_time_s:g}/{self.minimum_off_time_s:g} s"
        )


def willans_coefficients(
    rated_power_kw: float,
    rated_sfc_kg_kwh: float,
    idle_fuel_fraction: float,
) -> tuple[float, float]:
    """Return affine fuel coefficients ``(a, b)`` from a rated calibration."""
    rated_power = _positive("rated_power_kw", rated_power_kw)
    rated_sfc = _positive("rated_sfc_kg_kwh", rated_sfc_kg_kwh)
    idle_fraction = _finite("idle_fuel_fraction", idle_fuel_fraction)
    if not 0.0 <= idle_fraction < 1.0:
        raise ValueError(
            "idle_fuel_fraction must lie in [0, 1), "
            f"got {idle_fuel_fraction!r}"
        )
    rated_fuel_kg_h = rated_sfc * rated_power
    b = idle_fraction * rated_fuel_kg_h
    a = (rated_fuel_kg_h - b) / rated_power
    return a, b


def duty_cycle(
    demand_bus_kw: float,
    engine_on_kw: float,
    source_efficiency: float,
    eta_charge: float,
    eta_discharge: float,
) -> float:
    """Charge-sustaining engine-ON fraction for one constant-demand cycle."""
    demand = _positive("demand_bus_kw", demand_bus_kw)
    engine_on = _positive("engine_on_kw", engine_on_kw)
    source = _efficiency("source_efficiency", source_efficiency)
    charge = _efficiency("eta_charge", eta_charge)
    discharge = _efficiency("eta_discharge", eta_discharge)
    continuous_kw = demand / source
    if engine_on < continuous_kw:
        raise ValueError(
            "engine_on_kw must be at least demand_bus_kw/source_efficiency, "
            f"got {engine_on!r} below {continuous_kw!r}"
        )

    # Charge-sustaining balance with r = eta_charge * eta_discharge.
    round_trip = charge * discharge
    return demand / (
        round_trip * source * engine_on + demand * (1.0 - round_trip)
    )


def cycle_average_fuel_rate_kg_h(
    demand_bus_kw: float,
    engine_on_kw: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    source_efficiency: float,
    eta_charge: float,
    eta_discharge: float,
) -> float:
    """Time-average fuel rate [kg/h] for charge-sustaining cycling."""
    slope = _positive("willans_a_kg_kwh", willans_a_kg_kwh)
    intercept = _nonnegative("willans_b_kg_h", willans_b_kg_h)
    fraction = duty_cycle(
        demand_bus_kw,
        engine_on_kw,
        source_efficiency,
        eta_charge,
        eta_discharge,
    )
    return fraction * (slope * float(engine_on_kw) + intercept)


def continuous_fuel_rate_kg_h(
    demand_bus_kw: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    source_efficiency: float,
) -> float:
    """Fuel rate [kg/h] when the engine continuously follows DC-bus demand."""
    demand = _positive("demand_bus_kw", demand_bus_kw)
    slope = _positive("willans_a_kg_kwh", willans_a_kg_kwh)
    intercept = _nonnegative("willans_b_kg_h", willans_b_kg_h)
    source = _efficiency("source_efficiency", source_efficiency)
    return slope * demand / source + intercept


def _two_level_inputs(
    D: float,
    x_high: float,
    x_low: float,
    a: float,
    b: float,
    g: float,
    eta_rt: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    demand = _positive("D", D)
    high = _positive("x_high", x_high)
    low = _nonnegative("x_low", x_low)
    slope = _positive("a", a)
    intercept = _nonnegative("b", b)
    source = _efficiency("g", g)
    round_trip = _efficiency("eta_rt", eta_rt)
    charge_surplus = source * high - demand
    discharge_deficit = demand - source * low
    if charge_surplus <= 0.0:
        raise ValueError("x_high must satisfy g*x_high > D")
    if discharge_deficit <= 0.0:
        raise ValueError("x_low must satisfy g*x_low < D")
    return (
        demand,
        high,
        low,
        slope,
        intercept,
        source,
        round_trip,
        charge_surplus,
        discharge_deficit,
    )


def two_level_cycle_fuel_rate(
    D: float,
    x_high: float,
    x_low: float,
    a: float,
    b: float,
    g: float,
    eta_rt: float,
) -> float:
    """Average fuel rate [kg/h] when the engine runs at both cycle levels."""
    (
        _,
        high,
        low,
        slope,
        intercept,
        _,
        round_trip,
        charge_surplus,
        discharge_deficit,
    ) = _two_level_inputs(D, x_high, x_low, a, b, g, eta_rt)

    # Charge-sustaining two-level duty fraction.
    fraction = discharge_deficit / (
        round_trip * charge_surplus + discharge_deficit
    )
    average_shaft_kw = fraction * high + (1.0 - fraction) * low
    return slope * average_shaft_kw + intercept


def two_level_penalty_vs_continuous(
    D: float,
    x_high: float,
    x_low: float,
    a: float,
    b: float,
    g: float,
    eta_rt: float,
) -> float:
    """Fuel-rate penalty [kg/h] versus continuous demand following."""
    (
        _,
        _,
        _,
        slope,
        _,
        source,
        round_trip,
        charge_surplus,
        discharge_deficit,
    ) = _two_level_inputs(D, x_high, x_low, a, b, g, eta_rt)

    # Marginal fuel needed to replace round-trip loss; b is paid throughout.
    return (
        slope
        * discharge_deficit
        * charge_surplus
        * (1.0 - round_trip)
        / (source * (round_trip * charge_surplus + discharge_deficit))
    )


def economic_cycling_threshold_kw(
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    source_efficiency: float,
    eta_charge: float,
    eta_discharge: float,
) -> float:
    """Demand below which an above-demand engine point reduces fuel use."""
    slope = _positive("willans_a_kg_kwh", willans_a_kg_kwh)
    intercept = _nonnegative("willans_b_kg_h", willans_b_kg_h)
    source = _efficiency("source_efficiency", source_efficiency)
    round_trip = _efficiency("eta_charge", eta_charge) * _efficiency(
        "eta_discharge", eta_discharge
    )
    if intercept == 0.0:
        return 0.0
    if round_trip == 1.0:
        return math.inf
    return intercept * round_trip * source / (slope * (1.0 - round_trip))


def cycle_period_s(
    delta_soc: float,
    pack_energy_kwh: float,
    eta_discharge: float,
    engine_on_fraction: float,
    demand_bus_kw: float,
) -> float:
    """Full ON-plus-OFF cycle period [s] for a specified SoC swing."""
    band = _positive("delta_soc", delta_soc)
    if band > 1.0:
        raise ValueError(f"delta_soc must not exceed 1, got {delta_soc!r}")
    energy = _positive("pack_energy_kwh", pack_energy_kwh)
    discharge = _efficiency("eta_discharge", eta_discharge)
    fraction = _finite("engine_on_fraction", engine_on_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError(
            "engine_on_fraction must lie in [0, 1) for a finite cycle, "
            f"got {engine_on_fraction!r}"
        )
    demand = _positive("demand_bus_kw", demand_bus_kw)
    return (
        band * energy * discharge / ((1.0 - fraction) * demand)
    ) * SECONDS_PER_HOUR


def _bounded_minimum(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float,
) -> tuple[float, float]:
    """Golden-section minimum including both endpoints."""
    if upper - lower <= tolerance:
        candidates = ((lower, objective(lower)), (upper, objective(upper)))
        return min(candidates, key=lambda item: item[1])

    conjugate = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = lower, upper
    first = right - conjugate * (right - left)
    second = left + conjugate * (right - left)
    first_value, second_value = objective(first), objective(second)
    while right - left > tolerance:
        if first_value <= second_value:
            right = second
            second, second_value = first, first_value
            first = right - conjugate * (right - left)
            first_value = objective(first)
        else:
            left = first
            first, first_value = second, second_value
            second = left + conjugate * (right - left)
            second_value = objective(second)

    candidates = (
        (lower, objective(lower)),
        (upper, objective(upper)),
        (first, first_value),
        (second, second_value),
    )
    return min(candidates, key=lambda item: item[1])


@dataclass(frozen=True)
class CycleOptimum:
    """Bounded cycling optimum and its active power constraint."""

    engine_on_kw: float
    duty_cycle: float
    cycle_fuel_kg_h: float
    continuous_fuel_kg_h: float
    benefit_fraction: float
    active_bound: str
    lower_bound_kw: float
    upper_bound_kw: float
    charge_ceiling_kw: float


def optimal_engine_on_power(
    demand_bus_kw: float,
    engine_max_kw: float,
    charge_limit_bus_kw: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    source_efficiency: float,
    eta_charge: float,
    eta_discharge: float,
    *,
    tolerance_kw: float = 1.0e-7,
) -> CycleOptimum:
    """Numerically minimize cycling fuel rate over the feasible ON-power band."""
    demand = _positive("demand_bus_kw", demand_bus_kw)
    maximum = _positive("engine_max_kw", engine_max_kw)
    charge_limit = _nonnegative("charge_limit_bus_kw", charge_limit_bus_kw)
    source = _efficiency("source_efficiency", source_efficiency)
    tolerance = _positive("tolerance_kw", tolerance_kw)
    lower = demand / source
    charge_ceiling = (demand + charge_limit) / source
    upper = min(maximum, charge_ceiling)
    if upper < lower:
        raise ValueError(
            "no engine-only operating point exists: engine_max_kw is below "
            "demand_bus_kw/source_efficiency"
        )

    def objective(engine_kw: float) -> float:
        return cycle_average_fuel_rate_kg_h(
            demand,
            engine_kw,
            willans_a_kg_kwh,
            willans_b_kg_h,
            source,
            eta_charge,
            eta_discharge,
        )

    engine_kw, cycled = _bounded_minimum(objective, lower, upper, tolerance)
    continuous = continuous_fuel_rate_kg_h(
        demand, willans_a_kg_kwh, willans_b_kg_h, source
    )
    bound_tolerance = max(tolerance, 1.0e-9)
    if abs(engine_kw - lower) <= bound_tolerance:
        active = "lower_bound"
        engine_kw = lower
    elif abs(engine_kw - upper) <= bound_tolerance:
        engine_kw = upper
        engine_bound = abs(upper - maximum) <= bound_tolerance
        charge_bound = abs(upper - charge_ceiling) <= bound_tolerance
        if engine_bound and charge_bound:
            active = "engine_and_charge_ceiling"
        elif engine_bound:
            active = "engine_ceiling"
        else:
            active = "charge_ceiling"
    else:
        active = "interior"

    cycled = objective(engine_kw)
    return CycleOptimum(
        engine_on_kw=engine_kw,
        duty_cycle=duty_cycle(
            demand, engine_kw, source, eta_charge, eta_discharge
        ),
        cycle_fuel_kg_h=cycled,
        continuous_fuel_kg_h=continuous,
        benefit_fraction=1.0 - cycled / continuous,
        active_bound=active,
        lower_bound_kw=lower,
        upper_bound_kw=upper,
        charge_ceiling_kw=charge_ceiling,
    )


class OperatingRegime(str, Enum):
    """Sustainable level-flight power regime."""

    CYCLING_FEASIBLE = "cycling_feasible"
    ENGINE_LIMITED_CONTINUOUS = "engine_limited_continuous"
    BATTERY_ASSISTED = "battery_assisted"


@dataclass(frozen=True)
class RegimeClassification:
    """Regime result with the available surplus and optional cycle optimum."""

    regime: OperatingRegime
    engine_bus_available_kw: float
    surplus_bus_kw: float
    cycle_optimum: CycleOptimum | None
    cycling_blocker: str | None


def classify_regime(
    demand_bus_kw: float,
    engine_max_kw: float,
    charge_limit_bus_kw: float,
    discharge_limit_bus_kw: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    source_efficiency: float,
    eta_charge: float,
    eta_discharge: float,
    *,
    tolerance_kw: float = 1.0e-7,
    benefit_tolerance: float = 1.0e-12,
) -> RegimeClassification:
    """Classify demand using physical power bounds and economic cycle benefit."""
    demand = _positive("demand_bus_kw", demand_bus_kw)
    maximum = _positive("engine_max_kw", engine_max_kw)
    source = _efficiency("source_efficiency", source_efficiency)
    tolerance = _positive("tolerance_kw", tolerance_kw)
    benefit_eps = _nonnegative("benefit_tolerance", benefit_tolerance)
    discharge_limit = _nonnegative(
        "discharge_limit_bus_kw", discharge_limit_bus_kw
    )
    available = source * maximum
    surplus = available - demand
    if surplus < -tolerance:
        return RegimeClassification(
            OperatingRegime.BATTERY_ASSISTED, available, surplus, None, None
        )
    if surplus <= tolerance:
        return RegimeClassification(
            OperatingRegime.ENGINE_LIMITED_CONTINUOUS,
            available,
            surplus,
            None,
            "no_engine_surplus",
        )

    optimum = optimal_engine_on_power(
        demand,
        maximum,
        charge_limit_bus_kw,
        willans_a_kg_kwh,
        willans_b_kg_h,
        source,
        eta_charge,
        eta_discharge,
        tolerance_kw=tolerance,
    )
    has_off_time = optimum.duty_cycle < 1.0 - tolerance
    beneficial = optimum.benefit_fraction > benefit_eps
    discharge_can_hold_demand = demand <= discharge_limit + tolerance
    regime = (
        OperatingRegime.CYCLING_FEASIBLE
        if has_off_time and beneficial and discharge_can_hold_demand
        else OperatingRegime.ENGINE_LIMITED_CONTINUOUS
    )
    blocker = None
    if not discharge_can_hold_demand:
        blocker = "battery_discharge_limit"
    elif not has_off_time:
        blocker = "no_engine_surplus"
    elif not beneficial:
        blocker = "cycling_not_economic"
    return RegimeClassification(regime, available, surplus, optimum, blocker)


def battery_assisted_duration_h(
    engine_power_kw: float,
    demand_bus_kw: float,
    source_efficiency: float,
    fuel_available_kg: float,
    battery_usable_bus_kwh: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
) -> tuple[float, float, float]:
    """Return endurance, fuel duration, and battery duration [h]."""
    engine_kw = _nonnegative("engine_power_kw", engine_power_kw)
    demand = _positive("demand_bus_kw", demand_bus_kw)
    source = _efficiency("source_efficiency", source_efficiency)
    fuel = _nonnegative("fuel_available_kg", fuel_available_kg)
    battery = _nonnegative("battery_usable_bus_kwh", battery_usable_bus_kwh)
    slope = _positive("willans_a_kg_kwh", willans_a_kg_kwh)
    intercept = _nonnegative("willans_b_kg_h", willans_b_kg_h)
    assist_kw = demand - source * engine_kw
    if assist_kw <= 0.0:
        raise ValueError(
            "battery-assisted duration requires demand_bus_kw greater than "
            "source_efficiency*engine_power_kw"
        )
    fuel_rate = slope * engine_kw + intercept
    fuel_h = fuel / fuel_rate if fuel_rate > 0.0 else math.inf
    battery_h = battery / assist_kw
    return min(fuel_h, battery_h), fuel_h, battery_h


@dataclass(frozen=True)
class BatteryAssistedOptimum:
    """Maximum of the lesser fuel and battery duration."""

    engine_power_kw: float
    endurance_h: float
    fuel_duration_h: float
    battery_duration_h: float
    assist_bus_kw: float
    active_bound: str
    limiting_source: str


def optimal_battery_assisted_power(
    demand_bus_kw: float,
    engine_min_kw: float,
    engine_max_kw: float,
    source_efficiency: float,
    fuel_available_kg: float,
    battery_usable_bus_kwh: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    *,
    tolerance_kw: float = 1.0e-7,
) -> BatteryAssistedOptimum:
    """Solve the bounded max-min endurance problem in the assisted regime."""
    demand = _positive("demand_bus_kw", demand_bus_kw)
    lower = _nonnegative("engine_min_kw", engine_min_kw)
    upper = _positive("engine_max_kw", engine_max_kw)
    source = _efficiency("source_efficiency", source_efficiency)
    tolerance = _positive("tolerance_kw", tolerance_kw)
    if lower > upper:
        raise ValueError("engine_min_kw must not exceed engine_max_kw")
    if demand <= source * upper:
        raise ValueError(
            "optimal_battery_assisted_power requires demand above maximum "
            "engine bus power"
        )

    def durations(engine_kw: float) -> tuple[float, float, float]:
        return battery_assisted_duration_h(
            engine_kw,
            demand,
            source,
            fuel_available_kg,
            battery_usable_bus_kwh,
            willans_a_kg_kwh,
            willans_b_kg_h,
        )

    _, lower_fuel, lower_battery = durations(lower)
    _, upper_fuel, upper_battery = durations(upper)
    if lower_fuel <= lower_battery:
        engine_kw, active = lower, "lower_bound"
    elif upper_fuel >= upper_battery:
        engine_kw, active = upper, "engine_ceiling"
    else:
        left, right = lower, upper
        while right - left > tolerance:
            trial = 0.5 * (left + right)
            _, fuel_h, battery_h = durations(trial)
            if fuel_h > battery_h:
                left = trial
            else:
                right = trial
        engine_kw, active = 0.5 * (left + right), "interior_equalisation"

    endurance, fuel_h, battery_h = durations(engine_kw)
    difference = fuel_h - battery_h
    if abs(difference) <= max(1.0e-10, tolerance * max(endurance, 1.0)):
        limiting = "equal"
    elif fuel_h < battery_h:
        limiting = "fuel"
    else:
        limiting = "battery"
    return BatteryAssistedOptimum(
        engine_power_kw=engine_kw,
        endurance_h=endurance,
        fuel_duration_h=fuel_h,
        battery_duration_h=battery_h,
        assist_bus_kw=demand - source * engine_kw,
        active_bound=active,
        limiting_source=limiting,
    )


def battery_energy_for_equal_duration_kwh(
    engine_power_kw: float,
    demand_bus_kw: float,
    source_efficiency: float,
    fuel_available_kg: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
) -> float:
    """Bus-usable battery energy [kWh] that matches fuel duration."""
    _, fuel_h, _ = battery_assisted_duration_h(
        engine_power_kw,
        demand_bus_kw,
        source_efficiency,
        fuel_available_kg,
        0.0,
        willans_a_kg_kwh,
        willans_b_kg_h,
    )
    assist_kw = float(demand_bus_kw) - float(source_efficiency) * float(engine_power_kw)
    return assist_kw * fuel_h


@dataclass(frozen=True)
class CycleEnergyBalance:
    """One analytical cycle's energy ledger [kWh]."""

    fuel_chemical_in_kwh: float
    propulsive_work_out_kwh: float
    engine_thermal_loss_kwh: float
    source_chain_loss_kwh: float
    demand_chain_loss_kwh: float
    propeller_loss_kwh: float
    battery_ohmic_loss_kwh: float
    battery_stored_energy_change_kwh: float
    residual_kwh: float
    residual_fraction: float


def analytical_cycle_energy_balance(
    demand_bus_kw: float,
    engine_on_kw: float,
    cycle_duration_s: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    source_efficiency: float,
    demand_chain_efficiency: float,
    propeller_efficiency: float,
    eta_charge: float,
    eta_discharge: float,
    fuel_lhv_kj_kg: float,
) -> CycleEnergyBalance:
    """Resolve the full fuel-to-thrust ledger over one sustaining cycle."""
    demand = _positive("demand_bus_kw", demand_bus_kw)
    engine_kw = _positive("engine_on_kw", engine_on_kw)
    period_h = _positive("cycle_duration_s", cycle_duration_s) / SECONDS_PER_HOUR
    slope = _positive("willans_a_kg_kwh", willans_a_kg_kwh)
    intercept = _nonnegative("willans_b_kg_h", willans_b_kg_h)
    source = _efficiency("source_efficiency", source_efficiency)
    demand_eff = _efficiency("demand_chain_efficiency", demand_chain_efficiency)
    propeller_eff = _efficiency("propeller_efficiency", propeller_efficiency)
    charge = _efficiency("eta_charge", eta_charge)
    discharge = _efficiency("eta_discharge", eta_discharge)
    lhv = _positive("fuel_lhv_kj_kg", fuel_lhv_kj_kg)
    fraction = duty_cycle(demand, engine_kw, source, charge, discharge)
    on_h = fraction * period_h
    off_h = (1.0 - fraction) * period_h

    fuel_kg = (slope * engine_kw + intercept) * on_h
    fuel_chemical = fuel_kg * lhv / SECONDS_PER_HOUR
    engine_shaft = engine_kw * on_h
    source_bus = source * engine_kw * on_h
    engine_thermal = fuel_chemical - engine_shaft
    source_loss = engine_shaft - source_bus

    bus_work = demand * period_h
    shaft_work = demand_eff * bus_work
    propulsive_work = propeller_eff * shaft_work
    demand_loss = bus_work - shaft_work
    propeller_loss = shaft_work - propulsive_work

    charge_bus = max(source * engine_kw - demand, 0.0) * on_h
    stored_charge = charge * charge_bus
    stored_discharge = demand * off_h / discharge
    battery_loss = (charge_bus - stored_charge) + (stored_discharge - demand * off_h)
    stored_change = stored_charge - stored_discharge

    accounted = (
        propulsive_work
        + engine_thermal
        + source_loss
        + demand_loss
        + propeller_loss
        + battery_loss
        + stored_change
    )
    residual = fuel_chemical - accounted
    scale = max(abs(fuel_chemical), 1.0e-30)
    return CycleEnergyBalance(
        fuel_chemical_in_kwh=fuel_chemical,
        propulsive_work_out_kwh=propulsive_work,
        engine_thermal_loss_kwh=engine_thermal,
        source_chain_loss_kwh=source_loss,
        demand_chain_loss_kwh=demand_loss,
        propeller_loss_kwh=propeller_loss,
        battery_ohmic_loss_kwh=battery_loss,
        battery_stored_energy_change_kwh=stored_change,
        residual_kwh=residual,
        residual_fraction=abs(residual) / scale,
    )
