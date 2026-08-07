"""Altitude/mass feasibility maps for the analytical cycle diagnostics."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.analysis.cycle_model import (
    DiagnosticParameters,
    OperatingRegime,
    battery_energy_for_equal_duration_kwh,
    classify_regime,
    continuous_fuel_rate_kg_h,
    optimal_battery_assisted_power,
    willans_coefficients,
)
from src.models import aerodynamics
from src.models.atmosphere import atmosphere, g0
from src.models.battery import BatteryPack
from src.models.engine import Turboshaft
from src.models.mass import MassBreakdown, build_mass_budget
from src.models.powertrain import SeriesPowertrain

__all__ = [
    "BatteryEfficiencies",
    "CycleMapPoint",
    "DischargeRegionSummary",
    "LoiterFeasibilitySummary",
    "PackTradePoint",
    "ReferenceMapSummary",
    "battery_efficiencies",
    "build_pack_size_trade",
    "build_feasibility_map",
    "discharge_condition_removal",
    "discharge_crossing_mass_kg",
    "find_power_boundary_altitude_m",
    "generate_reference_feasibility_artifacts",
    "loiter_bus_demand_kw",
    "loiter_feasibility_summary",
    "minimum_discharge_c_rate",
    "minimum_pack_capacity_kwh",
    "plot_feasibility_map",
    "plot_pack_sizing_contours",
    "plot_pack_trade",
    "write_feasibility_csv",
    "write_pack_sizing_csv",
    "write_pack_trade_csv",
]


@dataclass(frozen=True)
class BatteryEfficiencies:
    """Constant-efficiency approximation at two stated bus powers."""

    charge: float
    discharge: float
    round_trip: float
    soc: float
    charge_bus_kw: float
    discharge_bus_kw: float


def battery_efficiencies(
    battery: BatteryPack,
    *,
    charge_bus_kw: float,
    discharge_bus_kw: float,
    soc: float,
) -> BatteryEfficiencies:
    """Evaluate separate terminal efficiencies from the Rint model."""
    charge_kw = float(charge_bus_kw)
    discharge_kw = float(discharge_bus_kw)
    state_of_charge = float(soc)
    if charge_kw < 0.0 or discharge_kw < 0.0:
        raise ValueError("charge_bus_kw and discharge_bus_kw are positive magnitudes")
    if not 0.0 <= state_of_charge <= 1.0:
        raise ValueError(f"soc must lie in [0, 1], got {soc!r}")

    v_oc = float(battery.open_circuit_voltage(state_of_charge))
    if charge_kw == 0.0:
        eta_charge = 1.0
    else:
        charge_current = float(
            battery.current_from_power(-charge_kw, state_of_charge)
        )
        stored_charge_kw = abs(v_oc * charge_current / 1000.0)
        eta_charge = stored_charge_kw / charge_kw
    if discharge_kw == 0.0:
        eta_discharge = 1.0
    else:
        discharge_current = float(
            battery.current_from_power(discharge_kw, state_of_charge)
        )
        stored_discharge_kw = v_oc * discharge_current / 1000.0
        eta_discharge = discharge_kw / stored_discharge_kw
    return BatteryEfficiencies(
        charge=eta_charge,
        discharge=eta_discharge,
        round_trip=eta_charge * eta_discharge,
        soc=state_of_charge,
        charge_bus_kw=charge_kw,
        discharge_bus_kw=discharge_kw,
    )


def loiter_bus_demand_kw(
    altitude_m: float,
    mass_kg: float,
    *,
    wing_area_m2: float,
    aspect_ratio: float,
    oswald_efficiency: float,
    cd0: float,
    cl_max: float,
    propeller_efficiency: float,
    powertrain: SeriesPowertrain,
    climb_rate_mps: float = 0.0,
) -> float:
    """Minimum-power/stall-margin DC-bus demand from the aerodynamic model."""
    atmospheric = atmosphere(float(altitude_m))
    weight_n = float(mass_kg) * g0
    speed_mps, _ = aerodynamics.loiter_speed(
        weight_n,
        atmospheric.density_kg_m3,
        wing_area_m2,
        cd0,
        aspect_ratio,
        oswald_efficiency,
        cl_max,
    )
    state = aerodynamics.evaluate(
        weight_n,
        atmospheric.density_kg_m3,
        speed_mps,
        wing_area_m2,
        cd0,
        aspect_ratio,
        oswald_efficiency,
        propeller_efficiency,
        climb_rate_mps=climb_rate_mps,
    )
    return float(powertrain.bus_power_required(float(state.shaft_power_w) / 1000.0))


def minimum_pack_capacity_kwh(
    demand_bus_kw: float, discharge_c_rate: float
) -> float:
    """Minimum pack energy [kWh] whose C-rate proxy can carry engine-OFF load."""
    demand = float(demand_bus_kw)
    c_rate = float(discharge_c_rate)
    if not math.isfinite(demand) or demand <= 0.0:
        raise ValueError("demand_bus_kw must be finite and positive")
    if not math.isfinite(c_rate) or c_rate <= 0.0:
        raise ValueError("discharge_c_rate must be finite and positive")
    return demand / c_rate


def minimum_discharge_c_rate(
    demand_bus_kw: float, capacity_kwh: float
) -> float:
    """Minimum bus-power C-rate proxy that carries engine-OFF load."""
    demand = float(demand_bus_kw)
    capacity = float(capacity_kwh)
    if not math.isfinite(demand) or demand <= 0.0:
        raise ValueError("demand_bus_kw must be finite and positive")
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("capacity_kwh must be finite and positive")
    return demand / capacity


def discharge_crossing_mass_kg(
    loiter_start_mass_kg: float,
    loiter_start_demand_bus_kw: float,
    discharge_limit_bus_kw: float,
) -> float:
    """Mass [kg] where a constant-lift-coefficient demand reaches a power cap."""
    start_mass = float(loiter_start_mass_kg)
    start_demand = float(loiter_start_demand_bus_kw)
    limit = float(discharge_limit_bus_kw)
    if not math.isfinite(start_mass) or start_mass <= 0.0:
        raise ValueError("loiter_start_mass_kg must be finite and positive")
    if not math.isfinite(start_demand) or start_demand <= 0.0:
        raise ValueError("loiter_start_demand_bus_kw must be finite and positive")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("discharge_limit_bus_kw must be finite and positive")

    # Minimum-power demand at fixed lift coefficient scales with W**1.5.
    return start_mass * (limit / start_demand) ** (2.0 / 3.0)


@dataclass(frozen=True)
class LoiterFeasibilitySummary:
    """Discharge crossing and logged-time share for one loiter trajectory."""

    crossing_mass_kg: float
    fuel_burned_to_crossing_kg: float
    elapsed_to_crossing_h: float
    loiter_duration_h: float
    feasible_duration_h: float
    loiter_time_power_feasible_fraction: float


def loiter_feasibility_summary(
    loiter_start_mass_kg: float,
    loiter_start_demand_bus_kw: float,
    discharge_limit_bus_kw: float,
    loiter_duration_h: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    source_efficiency: float,
    *,
    integration_points: int = 4097,
) -> LoiterFeasibilitySummary:
    """Compute the continuous-operation path to engine-OFF feasibility."""
    start_mass = float(loiter_start_mass_kg)
    start_demand = float(loiter_start_demand_bus_kw)
    duration = float(loiter_duration_h)
    slope = float(willans_a_kg_kwh)
    intercept = float(willans_b_kg_h)
    source = float(source_efficiency)
    if duration <= 0.0 or not math.isfinite(duration):
        raise ValueError("loiter_duration_h must be finite and positive")
    if slope <= 0.0 or intercept < 0.0 or not 0.0 < source <= 1.0:
        raise ValueError("fuel coefficients and source_efficiency are invalid")
    if integration_points < 3:
        raise ValueError("integration_points must be at least three")

    crossing = min(
        start_mass,
        discharge_crossing_mass_kg(
            start_mass, start_demand, discharge_limit_bus_kw
        ),
    )
    masses = np.linspace(crossing, start_mass, integration_points)
    demands = start_demand * (masses / start_mass) ** 1.5
    fuel_rates = slope * demands / source + intercept
    elapsed_h = float(np.trapezoid(1.0 / fuel_rates, masses))
    feasible_h = max(duration - elapsed_h, 0.0)
    return LoiterFeasibilitySummary(
        crossing_mass_kg=crossing,
        fuel_burned_to_crossing_kg=start_mass - crossing,
        elapsed_to_crossing_h=elapsed_h,
        loiter_duration_h=duration,
        feasible_duration_h=feasible_h,
        loiter_time_power_feasible_fraction=feasible_h / duration,
    )


@dataclass(frozen=True)
class CycleMapPoint:
    """One altitude/mass point in the feasibility data product."""

    altitude_m: float
    mass_kg: float
    demand_bus_kw: float
    engine_max_shaft_kw: float
    engine_bus_available_kw: float
    surplus_bus_kw: float
    regime: str
    cycling_blocker: str
    charge_side_cycling_candidate: bool
    discharge_side_feasible: bool
    cycling_feasible: bool
    minimum_pack_capacity_kwh: float
    minimum_discharge_c_rate: float
    eta_charge: float
    eta_discharge: float
    round_trip_efficiency: float
    optimal_engine_on_kw: float
    duty_cycle: float
    cycle_to_continuous_fuel_ratio: float
    cycle_benefit_fraction: float
    cycle_active_bound: str
    battery_assist_bus_kw: float
    battery_assist_rate_feasible: bool
    assisted_optimal_engine_kw: float
    assisted_endurance_h: float
    assisted_fuel_duration_h: float
    assisted_battery_duration_h: float
    assisted_active_bound: str
    assisted_limiting_source: str
    equal_duration_bus_energy_kwh: float
    fuel_available_kg: float
    idle_fuel_fraction: float
    charge_c_rate: float
    discharge_c_rate: float
    restart_fuel_kg: float
    minimum_on_time_s: float
    minimum_off_time_s: float
    efficiency_evaluation_soc: float


def _nan() -> float:
    return float("nan")


def build_feasibility_map(
    altitudes_m: Sequence[float],
    masses_kg: Sequence[float],
    *,
    wing_area_m2: float,
    aspect_ratio: float,
    oswald_efficiency: float,
    cd0: float,
    cl_max: float,
    propeller_efficiency: float,
    dry_mass_kg: float,
    engine: Turboshaft,
    battery: BatteryPack,
    powertrain: SeriesPowertrain,
    parameters: DiagnosticParameters,
    efficiency_evaluation_soc: float,
    usable_soc_low: float,
    usable_soc_high: float,
) -> tuple[CycleMapPoint, ...]:
    """Evaluate regimes and constrained optima over a rectangular grid."""
    altitudes = tuple(float(value) for value in altitudes_m)
    masses = tuple(float(value) for value in masses_kg)
    if not altitudes or not masses:
        raise ValueError("altitudes_m and masses_kg must both be non-empty")
    if not 0.0 <= usable_soc_low < usable_soc_high <= 1.0:
        raise ValueError("usable SoC bounds must satisfy 0 <= low < high <= 1")
    if not 0.0 <= efficiency_evaluation_soc <= 1.0:
        raise ValueError("efficiency_evaluation_soc must lie in [0, 1]")

    a, b = willans_coefficients(
        engine.rated_power_kw,
        engine.sfc_rated_kg_kwh,
        parameters.idle_fuel_fraction,
    )
    charge_limit_kw = parameters.charge_c_rate * battery.capacity_kwh
    discharge_limit_kw = parameters.discharge_c_rate * battery.capacity_kwh
    usable_stored_kwh = (
        usable_soc_high - usable_soc_low
    ) * battery.capacity_kwh
    source_efficiency = powertrain.source_chain_efficiency
    points: list[CycleMapPoint] = []

    for mass_kg in masses:
        if mass_kg < dry_mass_kg:
            raise ValueError("masses_kg must not fall below dry_mass_kg")
        fuel_available_kg = mass_kg - dry_mass_kg
        for altitude_m in altitudes:
            demand_kw = loiter_bus_demand_kw(
                altitude_m,
                mass_kg,
                wing_area_m2=wing_area_m2,
                aspect_ratio=aspect_ratio,
                oswald_efficiency=oswald_efficiency,
                cd0=cd0,
                cl_max=cl_max,
                propeller_efficiency=propeller_efficiency,
                powertrain=powertrain,
            )
            sigma = float(atmosphere(altitude_m).density_ratio)
            engine_max_kw = engine.max_power_kw(sigma)
            engine_bus_kw = source_efficiency * engine_max_kw
            surplus_kw = engine_bus_kw - demand_kw

            charge_power_kw = min(max(surplus_kw, 0.0), charge_limit_kw)
            efficiencies = battery_efficiencies(
                battery,
                charge_bus_kw=charge_power_kw,
                discharge_bus_kw=demand_kw,
                soc=efficiency_evaluation_soc,
            )
            classification = None
            for _ in range(4):
                classification = classify_regime(
                    demand_kw,
                    engine_max_kw,
                    charge_limit_kw,
                    discharge_limit_kw,
                    a,
                    b,
                    source_efficiency,
                    efficiencies.charge,
                    efficiencies.discharge,
                )
                optimum = classification.cycle_optimum
                next_charge_kw = (
                    max(source_efficiency * optimum.engine_on_kw - demand_kw, 0.0)
                    if optimum is not None
                    else 0.0
                )
                if abs(next_charge_kw - charge_power_kw) <= 1.0e-10:
                    break
                charge_power_kw = next_charge_kw
                efficiencies = battery_efficiencies(
                    battery,
                    charge_bus_kw=charge_power_kw,
                    discharge_bus_kw=demand_kw,
                    soc=efficiency_evaluation_soc,
                )
            assert classification is not None

            optimum = classification.cycle_optimum
            charge_candidate = bool(
                optimum is not None
                and optimum.duty_cycle < 1.0 - 1.0e-7
                and optimum.benefit_fraction > 1.0e-12
            )
            discharge_feasible = demand_kw <= discharge_limit_kw + 1.0e-7
            cycling_feasible = (
                classification.regime is OperatingRegime.CYCLING_FEASIBLE
            )
            reported_optimum = optimum if cycling_feasible else None
            x_kw = (
                reported_optimum.engine_on_kw
                if reported_optimum is not None
                else _nan()
            )
            delta = (
                reported_optimum.duty_cycle
                if reported_optimum is not None
                else _nan()
            )
            ratio = (
                reported_optimum.cycle_fuel_kg_h
                / reported_optimum.continuous_fuel_kg_h
                if reported_optimum is not None
                else _nan()
            )
            benefit = (
                reported_optimum.benefit_fraction
                if reported_optimum is not None
                else _nan()
            )
            cycle_bound = (
                reported_optimum.active_bound
                if reported_optimum is not None
                else ""
            )

            assist_kw = max(demand_kw - engine_bus_kw, 0.0)
            assist_rate_feasible = assist_kw <= discharge_limit_kw + 1.0e-10
            assisted_power = assisted_endurance = assisted_fuel = _nan()
            assisted_battery = equal_energy = _nan()
            assisted_bound = assisted_limiter = ""
            if classification.regime is OperatingRegime.BATTERY_ASSISTED:
                assisted_eta = battery_efficiencies(
                    battery,
                    charge_bus_kw=0.0,
                    discharge_bus_kw=assist_kw,
                    soc=efficiency_evaluation_soc,
                ).discharge
                usable_bus_kwh = usable_stored_kwh * assisted_eta
                assisted = optimal_battery_assisted_power(
                    demand_kw,
                    0.0,
                    engine_max_kw,
                    source_efficiency,
                    fuel_available_kg,
                    usable_bus_kwh,
                    a,
                    b,
                )
                assisted_power = assisted.engine_power_kw
                assisted_endurance = assisted.endurance_h
                assisted_fuel = assisted.fuel_duration_h
                assisted_battery = assisted.battery_duration_h
                assisted_bound = assisted.active_bound
                assisted_limiter = assisted.limiting_source
                equal_energy = battery_energy_for_equal_duration_kwh(
                    engine_max_kw,
                    demand_kw,
                    source_efficiency,
                    fuel_available_kg,
                    a,
                    b,
                )

            points.append(
                CycleMapPoint(
                    altitude_m=altitude_m,
                    mass_kg=mass_kg,
                    demand_bus_kw=demand_kw,
                    engine_max_shaft_kw=engine_max_kw,
                    engine_bus_available_kw=engine_bus_kw,
                    surplus_bus_kw=surplus_kw,
                    regime=classification.regime.value,
                    cycling_blocker=classification.cycling_blocker or "",
                    charge_side_cycling_candidate=charge_candidate,
                    discharge_side_feasible=discharge_feasible,
                    cycling_feasible=cycling_feasible,
                    minimum_pack_capacity_kwh=minimum_pack_capacity_kwh(
                        demand_kw, parameters.discharge_c_rate
                    ),
                    minimum_discharge_c_rate=minimum_discharge_c_rate(
                        demand_kw, battery.capacity_kwh
                    ),
                    eta_charge=efficiencies.charge,
                    eta_discharge=efficiencies.discharge,
                    round_trip_efficiency=efficiencies.round_trip,
                    optimal_engine_on_kw=x_kw,
                    duty_cycle=delta,
                    cycle_to_continuous_fuel_ratio=ratio,
                    cycle_benefit_fraction=benefit,
                    cycle_active_bound=cycle_bound,
                    battery_assist_bus_kw=assist_kw,
                    battery_assist_rate_feasible=assist_rate_feasible,
                    assisted_optimal_engine_kw=assisted_power,
                    assisted_endurance_h=assisted_endurance,
                    assisted_fuel_duration_h=assisted_fuel,
                    assisted_battery_duration_h=assisted_battery,
                    assisted_active_bound=assisted_bound,
                    assisted_limiting_source=assisted_limiter,
                    equal_duration_bus_energy_kwh=equal_energy,
                    fuel_available_kg=fuel_available_kg,
                    idle_fuel_fraction=parameters.idle_fuel_fraction,
                    charge_c_rate=parameters.charge_c_rate,
                    discharge_c_rate=parameters.discharge_c_rate,
                    restart_fuel_kg=parameters.restart_fuel_kg,
                    minimum_on_time_s=parameters.minimum_on_time_s,
                    minimum_off_time_s=parameters.minimum_off_time_s,
                    efficiency_evaluation_soc=efficiency_evaluation_soc,
                )
            )
    return tuple(points)


@dataclass(frozen=True)
class DischargeRegionSummary:
    """Sampled design-grid point count, not a mission-time or area metric."""

    charge_side_candidate_points: int
    removed_points: int
    retained_points: int
    sampled_grid_point_removed_fraction: float


def discharge_condition_removal(
    points: Sequence[CycleMapPoint],
) -> DischargeRegionSummary:
    """Count charge-only candidates rejected by the discharge power cap."""
    candidates = sum(point.charge_side_cycling_candidate for point in points)
    removed = sum(
        point.charge_side_cycling_candidate and not point.discharge_side_feasible
        for point in points
    )
    retained = candidates - removed
    fraction = removed / candidates if candidates else 0.0
    return DischargeRegionSummary(candidates, removed, retained, fraction)


def find_power_boundary_altitude_m(
    mass_kg: float,
    target_surplus_bus_kw: float,
    *,
    wing_area_m2: float,
    aspect_ratio: float,
    oswald_efficiency: float,
    cd0: float,
    cl_max: float,
    propeller_efficiency: float,
    engine: Turboshaft,
    powertrain: SeriesPowertrain,
    climb_rate_mps: float = 0.0,
    lower_altitude_m: float = 0.0,
    upper_altitude_m: float = 10_000.0,
) -> float:
    """Bisection altitude where engine bus surplus reaches a target."""
    target = float(target_surplus_bus_kw)

    def residual(altitude_m: float) -> float:
        demand_kw = loiter_bus_demand_kw(
            altitude_m,
            mass_kg,
            wing_area_m2=wing_area_m2,
            aspect_ratio=aspect_ratio,
            oswald_efficiency=oswald_efficiency,
            cd0=cd0,
            cl_max=cl_max,
            propeller_efficiency=propeller_efficiency,
            powertrain=powertrain,
            climb_rate_mps=climb_rate_mps,
        )
        sigma = float(atmosphere(altitude_m).density_ratio)
        available_kw = float(
            powertrain.bus_power_from_engine(engine.max_power_kw(sigma))
        )
        return available_kw - demand_kw - target

    lower, upper = float(lower_altitude_m), float(upper_altitude_m)
    lower_value, upper_value = residual(lower), residual(upper)
    if lower_value == 0.0:
        return lower
    if upper_value == 0.0:
        return upper
    if lower_value * upper_value > 0.0:
        raise ValueError("requested power boundary is not bracketed")
    for _ in range(80):
        trial = 0.5 * (lower + upper)
        if residual(trial) > 0.0:
            lower = trial
        else:
            upper = trial
    return 0.5 * (lower + upper)


def write_feasibility_csv(
    points: Sequence[CycleMapPoint], output_path: str | Path
) -> Path:
    """Write the complete map and repeated scenario metadata as CSV."""
    if not points:
        raise ValueError("points must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(point) for point in points]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_pack_sizing_csv(
    points: Sequence[CycleMapPoint], output_path: str | Path
) -> Path:
    """Write the altitude/mass discharge sizing thresholds as CSV."""
    if not points:
        raise ValueError("points must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "altitude_m",
        "mass_kg",
        "demand_bus_kw",
        "minimum_pack_capacity_kwh",
        "minimum_discharge_c_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: getattr(point, name) for name in fieldnames} for point in points
        )
    return path


def plot_feasibility_map(
    points: Sequence[CycleMapPoint],
    output_path: str | Path,
    *,
    parameters: DiagnosticParameters,
    loiter_start_mass_kg: float,
    efficiency_evaluation_soc: float,
    loiter_altitude_m: float | None = None,
    loiter_end_mass_kg: float | None = None,
    discharge_crossing_mass_kg: float | None = None,
) -> "matplotlib.figure.Figure":
    """Plot regime, power, optimum, duty, and fuel-benefit diagnostics."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if not points:
        raise ValueError("points must not be empty")
    altitudes = np.array(sorted({point.altitude_m for point in points}))
    masses = np.array(sorted({point.mass_kg for point in points}))
    lookup = {(point.mass_kg, point.altitude_m): point for point in points}
    regime_codes = {
        OperatingRegime.ENGINE_LIMITED_CONTINUOUS.value: 0,
        OperatingRegime.CYCLING_FEASIBLE.value: 1,
        OperatingRegime.BATTERY_ASSISTED.value: 2,
    }
    regime_grid = np.array(
        [
            [regime_codes[lookup[(mass, altitude)].regime] for altitude in altitudes]
            for mass in masses
        ]
    )
    reference_mass = masses[int(np.argmin(abs(masses - loiter_start_mass_kg)))]
    reference = [lookup[(reference_mass, altitude)] for altitude in altitudes]

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    regime_axis, power_axis, optimum_axis, benefit_axis = axes.flat
    cmap = ListedColormap(["#d9d9d9", "#5abf90", "#df8f44"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    regime_axis.pcolormesh(
        altitudes / 1000.0,
        masses,
        regime_grid,
        shading="nearest",
        cmap=cmap,
        norm=norm,
    )
    regime_axis.set_title("Operating regime")
    regime_axis.set_ylabel("Aircraft mass [kg]")
    regime_axis.text(
        0.02,
        0.03,
        "grey: continuous · green: cycling · orange: battery-assisted",
        transform=regime_axis.transAxes,
        fontsize=8,
    )
    trajectory_values = (
        loiter_altitude_m,
        loiter_end_mass_kg,
        discharge_crossing_mass_kg,
    )
    if any(value is not None for value in trajectory_values):
        if any(value is None for value in trajectory_values):
            raise ValueError("all loiter trajectory marker values must be supplied")
        altitude_km = float(loiter_altitude_m) / 1000.0
        end_mass = float(loiter_end_mass_kg)
        crossing_mass = float(discharge_crossing_mass_kg)
        regime_axis.plot(
            [altitude_km, altitude_km],
            [loiter_start_mass_kg, end_mass],
            color="black",
            linewidth=1.4,
            label="logged loiter mass trajectory",
        )
        regime_axis.scatter(
            [altitude_km],
            [crossing_mass],
            marker="X",
            s=45,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
            label="OFF-feasibility crossing",
        )
        regime_axis.legend(loc="upper right", fontsize=7)

    demand = np.array([point.demand_bus_kw for point in reference])
    available = np.array([point.engine_bus_available_kw for point in reference])
    power_axis.plot(altitudes / 1000.0, demand, label="bus demand D")
    power_axis.plot(altitudes / 1000.0, available, label="g·Pmax")
    power_axis.set_title(f"Power at {reference_mass:.1f} kg")
    power_axis.set_ylabel("DC-bus power [kW]")
    power_axis.legend(fontsize=8)

    optimum = np.array([point.optimal_engine_on_kw for point in reference])
    maximum = np.array([point.engine_max_shaft_kw for point in reference])
    optimum_axis.plot(altitudes / 1000.0, optimum, color="0.55", label="numerical x*")
    for bound, color, label in (
        ("charge_ceiling", "tab:blue", "x*: charge ceiling"),
        ("engine_ceiling", "tab:red", "x*: engine ceiling"),
        ("engine_and_charge_ceiling", "tab:purple", "x*: both ceilings"),
    ):
        mask = np.array([point.cycle_active_bound == bound for point in reference])
        optimum_axis.scatter(
            altitudes[mask] / 1000.0,
            optimum[mask],
            s=10,
            color=color,
            label=label,
            zorder=3,
        )
    optimum_axis.plot(altitudes / 1000.0, maximum, linestyle="--", label="Pmax")
    optimum_axis.set_title("Cycling ON power and ceiling")
    optimum_axis.set_ylabel("Engine shaft power [kW]")
    optimum_axis.legend(fontsize=8)

    ratio = np.array([point.cycle_to_continuous_fuel_ratio for point in reference])
    duty = np.array([point.duty_cycle for point in reference])
    benefit_axis.plot(altitudes / 1000.0, ratio, label="cycled/continuous fuel")
    benefit_axis.plot(altitudes / 1000.0, duty, label="duty cycle")
    benefit_axis.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    benefit_axis.set_title("Fuel ratio and duty cycle")
    benefit_axis.set_ylabel("Fraction [-]")
    benefit_axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Geopotential altitude [km]")
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "Cycle feasibility over sampled altitude–mass design-grid points — "
        + parameters.label()
        + f"; Rint efficiencies at SoC={efficiency_evaluation_soc:.2f}\n"
        "restart and dwell values are recorded placeholders, not modelled in this milestone",
        fontsize=10,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    return figure


def plot_pack_sizing_contours(
    points: Sequence[CycleMapPoint],
    output_path: str | Path,
    *,
    discharge_c_rate: float,
    battery_capacity_kwh: float,
) -> "matplotlib.figure.Figure":
    """Plot engine-OFF capacity and C-rate thresholds over altitude and mass."""
    import matplotlib.pyplot as plt

    if not points:
        raise ValueError("points must not be empty")
    altitudes = np.array(sorted({point.altitude_m for point in points}))
    masses = np.array(sorted({point.mass_kg for point in points}))
    lookup = {(point.mass_kg, point.altitude_m): point for point in points}
    capacity_grid = np.array(
        [
            [lookup[(mass, altitude)].minimum_pack_capacity_kwh for altitude in altitudes]
            for mass in masses
        ]
    )
    c_rate_grid = np.array(
        [
            [lookup[(mass, altitude)].minimum_discharge_c_rate for altitude in altitudes]
            for mass in masses
        ]
    )
    x_grid, y_grid = np.meshgrid(altitudes / 1000.0, masses)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    capacity_plot = axes[0].contourf(x_grid, y_grid, capacity_grid, levels=15)
    axes[0].contour(
        x_grid,
        y_grid,
        capacity_grid,
        levels=[battery_capacity_kwh],
        colors="white",
        linewidths=1.2,
    )
    figure.colorbar(capacity_plot, ax=axes[0], label="Minimum capacity [kWh]")
    axes[0].set_title(f"Capacity threshold at {discharge_c_rate:g}C")

    c_rate_plot = axes[1].contourf(x_grid, y_grid, c_rate_grid, levels=15)
    axes[1].contour(
        x_grid,
        y_grid,
        c_rate_grid,
        levels=[discharge_c_rate],
        colors="white",
        linewidths=1.2,
    )
    figure.colorbar(c_rate_plot, ax=axes[1], label="Minimum discharge C-rate [C]")
    axes[1].set_title(f"C-rate threshold at {battery_capacity_kwh:g} kWh")
    for axis in axes:
        axis.set_xlabel("Geopotential altitude [km]")
        axis.set_ylabel("Aircraft mass [kg]")
    figure.suptitle(
        "Engine-OFF discharge sizing: capacity × C-rate must exceed bus demand",
        fontsize=10,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    return figure


@dataclass(frozen=True)
class PackTradePoint:
    """One fixed-MTOW pack-size result relative to the reference capacity."""

    capacity_kwh: float
    idle_fuel_fraction: float
    pack_mass_delta_kg: float
    fuel_capacity_cost_kg: float
    discharge_enabling_benefit_kg: float
    charge_ceiling_benefit_kg: float
    gross_operational_benefit_kg: float
    net_fuel_benefit_kg: float
    loiter_endurance_h: float
    endurance_change_h: float


def _cycle_policy_fuel_rate_kg_h(
    demand_bus_kw: float,
    *,
    engine_max_kw: float,
    battery: BatteryPack,
    charge_limit_bus_kw: float,
    discharge_limit_bus_kw: float,
    willans_a_kg_kwh: float,
    willans_b_kg_h: float,
    source_efficiency: float,
    efficiency_evaluation_soc: float,
) -> float:
    demand = float(demand_bus_kw)
    if source_efficiency * engine_max_kw < demand - 1.0e-9:
        raise ValueError("pack trade requires an engine-continuous loiter point")
    charge_power_kw = min(
        max(source_efficiency * engine_max_kw - demand, 0.0),
        charge_limit_bus_kw,
    )
    efficiencies = battery_efficiencies(
        battery,
        charge_bus_kw=charge_power_kw,
        discharge_bus_kw=demand,
        soc=efficiency_evaluation_soc,
    )
    classification = None
    for _ in range(4):
        classification = classify_regime(
            demand,
            engine_max_kw,
            charge_limit_bus_kw,
            discharge_limit_bus_kw,
            willans_a_kg_kwh,
            willans_b_kg_h,
            source_efficiency,
            efficiencies.charge,
            efficiencies.discharge,
        )
        optimum = classification.cycle_optimum
        next_charge_kw = (
            max(source_efficiency * optimum.engine_on_kw - demand, 0.0)
            if optimum is not None
            else 0.0
        )
        if abs(next_charge_kw - charge_power_kw) <= 1.0e-10:
            break
        charge_power_kw = next_charge_kw
        efficiencies = battery_efficiencies(
            battery,
            charge_bus_kw=charge_power_kw,
            discharge_bus_kw=demand,
            soc=efficiency_evaluation_soc,
        )
    if (
        classification is not None
        and classification.regime is OperatingRegime.CYCLING_FEASIBLE
    ):
        optimum = classification.cycle_optimum
        if optimum is None:
            raise RuntimeError("cycling classification has no optimum")
        return optimum.cycle_fuel_kg_h
    return continuous_fuel_rate_kg_h(
        demand,
        willans_a_kg_kwh,
        willans_b_kg_h,
        source_efficiency,
    )


def _duration_curve(
    loiter_start_mass_kg: float,
    lower_mass_kg: float,
    fuel_rate: Callable[[float], float],
    integration_points: int,
    break_masses_kg: Sequence[float] = (),
) -> tuple[np.ndarray, np.ndarray]:
    uniform_masses = np.linspace(
        loiter_start_mass_kg, lower_mass_kg, integration_points
    )
    internal_breaks = [
        float(mass)
        for mass in break_masses_kg
        if lower_mass_kg < mass < loiter_start_mass_kg
    ]
    masses = np.array(
        sorted(set(uniform_masses).union(internal_breaks), reverse=True)
    )
    mass_steps = masses[:-1] - masses[1:]
    midpoints = 0.5 * (masses[:-1] + masses[1:])
    rates = np.array([fuel_rate(float(mass)) for mass in midpoints])
    segment_h = mass_steps / rates
    cumulative_h = np.concatenate(([0.0], np.cumsum(segment_h)))
    return loiter_start_mass_kg - masses, cumulative_h


def _time_at_mass(
    loiter_start_mass_kg: float,
    end_mass_kg: float,
    fuel_burn_axis_kg: np.ndarray,
    time_axis_h: np.ndarray,
) -> float:
    burn_kg = loiter_start_mass_kg - end_mass_kg
    if burn_kg < 0.0 or burn_kg > fuel_burn_axis_kg[-1]:
        raise ValueError("end mass lies outside the integrated pack-trade curve")
    return float(np.interp(burn_kg, fuel_burn_axis_kg, time_axis_h))


def _mass_at_time(
    loiter_start_mass_kg: float,
    elapsed_h: float,
    fuel_burn_axis_kg: np.ndarray,
    time_axis_h: np.ndarray,
) -> float:
    if elapsed_h < 0.0 or elapsed_h > time_axis_h[-1]:
        raise ValueError("elapsed time lies outside the integrated pack-trade curve")
    burned_kg = float(np.interp(elapsed_h, time_axis_h, fuel_burn_axis_kg))
    return loiter_start_mass_kg - burned_kg


def build_pack_size_trade(
    capacities_kwh: Sequence[float],
    idle_fuel_fractions: Sequence[float],
    *,
    reference_capacity_kwh: float,
    loiter_start_mass_kg: float,
    loiter_fuel_floor_kg: float,
    loiter_start_demand_bus_kw: float,
    altitude_m: float,
    wing_area_m2: float,
    aspect_ratio: float,
    peak_bus_kw: float,
    engine: Turboshaft,
    powertrain: SeriesPowertrain,
    charge_c_rate: float,
    discharge_c_rate: float,
    efficiency_evaluation_soc: float,
    integration_points: int = 801,
) -> tuple[PackTradePoint, ...]:
    """Compute pack mass cost and analytical cycling benefit at fixed MTOW."""
    capacities = tuple(float(value) for value in capacities_kwh)
    idle_fractions = tuple(float(value) for value in idle_fuel_fractions)
    if not capacities or not idle_fractions:
        raise ValueError("capacity and idle-fraction sweeps must be non-empty")
    if any(value <= 0.0 for value in capacities):
        raise ValueError("capacities_kwh must all be positive")
    if any(not 0.0 <= value < 1.0 for value in idle_fractions):
        raise ValueError("idle_fuel_fractions must lie in [0, 1)")
    if integration_points < 101:
        raise ValueError("integration_points must be at least 101")

    engine_bus_rating_kw = float(
        powertrain.bus_power_from_engine(engine.rated_power_kw)
    )
    required_peak_battery_kw = max(float(peak_bus_kw) - engine_bus_rating_kw, 0.0)
    minimum_peak_capacity_kwh = (
        minimum_pack_capacity_kwh(required_peak_battery_kw, discharge_c_rate)
        if required_peak_battery_kw > 0.0
        else 0.0
    )
    if any(value < minimum_peak_capacity_kwh - 1.0e-10 for value in capacities):
        raise ValueError(
            "capacity sweep crosses the peak-power feasibility boundary at "
            f"{minimum_peak_capacity_kwh:g} kWh"
        )

    all_capacities = set(capacities)
    all_capacities.add(float(reference_capacity_kwh))
    budgets: dict[float, MassBreakdown] = {
        capacity: build_mass_budget(
            engine.rated_power_kw,
            capacity,
            peak_bus_kw,
            wing_area_m2,
            aspect_ratio,
        )
        for capacity in all_capacities
    }
    reference_capacity = float(reference_capacity_kwh)
    reference_budget = budgets[reference_capacity]
    reference_end_mass = reference_budget.dry_kg + float(loiter_fuel_floor_kg)
    minimum_end_mass = min(
        budget.dry_kg + float(loiter_fuel_floor_kg) for budget in budgets.values()
    )
    lower_mass = max(0.5 * loiter_start_mass_kg, minimum_end_mass - 100.0)
    sigma = float(atmosphere(float(altitude_m)).density_ratio)
    engine_max_kw = engine.max_power_kw(sigma)
    source = powertrain.source_chain_efficiency
    points: list[PackTradePoint] = []

    for idle_fraction in idle_fractions:
        a, b = willans_coefficients(
            engine.rated_power_kw,
            engine.sfc_rated_kg_kwh,
            idle_fraction,
        )

        def make_curve(
            capacity: float,
            charge_limit_kw: float,
            discharge_limit_kw: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            battery = BatteryPack(
                capacity,
                charge_c_rate=charge_c_rate,
                discharge_c_rate=discharge_c_rate,
            )

            def fuel_rate(mass_kg: float) -> float:
                demand_kw = loiter_start_demand_bus_kw * (
                    mass_kg / loiter_start_mass_kg
                ) ** 1.5
                return _cycle_policy_fuel_rate_kg_h(
                    demand_kw,
                    engine_max_kw=engine_max_kw,
                    battery=battery,
                    charge_limit_bus_kw=charge_limit_kw,
                    discharge_limit_bus_kw=discharge_limit_kw,
                    willans_a_kg_kwh=a,
                    willans_b_kg_h=b,
                    source_efficiency=source,
                    efficiency_evaluation_soc=efficiency_evaluation_soc,
                )

            return _duration_curve(
                loiter_start_mass_kg,
                lower_mass,
                fuel_rate,
                integration_points,
                break_masses_kg=(
                    discharge_crossing_mass_kg(
                        loiter_start_mass_kg,
                        loiter_start_demand_bus_kw,
                        discharge_limit_kw,
                    ),
                ),
            )

        reference_curve = make_curve(
            reference_capacity,
            charge_c_rate * reference_capacity,
            discharge_c_rate * reference_capacity,
        )
        reference_duration_h = _time_at_mass(
            loiter_start_mass_kg,
            reference_end_mass,
            *reference_curve,
        )

        for capacity in capacities:
            full_curve = make_curve(
                capacity,
                charge_c_rate * capacity,
                discharge_c_rate * capacity,
            )
            discharge_only_curve = make_curve(
                capacity,
                charge_c_rate * reference_capacity,
                discharge_c_rate * capacity,
            )
            full_mass_at_reference_time = _mass_at_time(
                loiter_start_mass_kg,
                reference_duration_h,
                *full_curve,
            )
            discharge_mass_at_reference_time = _mass_at_time(
                loiter_start_mass_kg,
                reference_duration_h,
                *discharge_only_curve,
            )
            budget = budgets[capacity]
            end_mass = budget.dry_kg + float(loiter_fuel_floor_kg)
            endurance_h = _time_at_mass(
                loiter_start_mass_kg,
                end_mass,
                *full_curve,
            )
            pack_mass_delta = budget.battery_kg - reference_budget.battery_kg
            fuel_cost = reference_budget.fuel_kg - budget.fuel_kg
            discharge_benefit = (
                discharge_mass_at_reference_time - reference_end_mass
            )
            charge_benefit = (
                full_mass_at_reference_time - discharge_mass_at_reference_time
            )
            gross_benefit = full_mass_at_reference_time - reference_end_mass
            net_benefit = gross_benefit - fuel_cost
            if abs(capacity - reference_capacity) <= 1.0e-12:
                pack_mass_delta = fuel_cost = 0.0
                discharge_benefit = charge_benefit = gross_benefit = 0.0
                net_benefit = 0.0
                endurance_h = reference_duration_h
            points.append(
                PackTradePoint(
                    capacity_kwh=capacity,
                    idle_fuel_fraction=idle_fraction,
                    pack_mass_delta_kg=pack_mass_delta,
                    fuel_capacity_cost_kg=fuel_cost,
                    discharge_enabling_benefit_kg=discharge_benefit,
                    charge_ceiling_benefit_kg=charge_benefit,
                    gross_operational_benefit_kg=gross_benefit,
                    net_fuel_benefit_kg=net_benefit,
                    loiter_endurance_h=endurance_h,
                    endurance_change_h=endurance_h - reference_duration_h,
                )
            )
    return tuple(points)


def write_pack_trade_csv(
    points: Sequence[PackTradePoint], output_path: str | Path
) -> Path:
    """Write the fixed-MTOW pack-size sweep as CSV."""
    if not points:
        raise ValueError("points must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(point) for point in points]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_pack_trade(
    points: Sequence[PackTradePoint], output_path: str | Path
) -> "matplotlib.figure.Figure":
    """Plot net fuel-equivalent value and endurance versus pack capacity."""
    import matplotlib.pyplot as plt

    if not points:
        raise ValueError("points must not be empty")
    idle_fractions = sorted({point.idle_fuel_fraction for point in points})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for idle_fraction in idle_fractions:
        selection = sorted(
            (
                point
                for point in points
                if point.idle_fuel_fraction == idle_fraction
            ),
            key=lambda point: point.capacity_kwh,
        )
        capacities = [point.capacity_kwh for point in selection]
        label = f"idle={idle_fraction:.2f}"
        axes[0].plot(
            capacities,
            [point.net_fuel_benefit_kg for point in selection],
            label=label,
        )
        axes[1].plot(
            capacities,
            [point.endurance_change_h for point in selection],
            label=label,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("Net fuel-equivalent benefit [kg]")
    axes[0].set_title("Operational saving minus fixed-MTOW fuel cost")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    axes[1].set_ylabel("Loiter endurance change [h]")
    axes[1].set_title("Analytical charge-sustaining endurance")
    for axis in axes:
        axis.set_xlabel("Battery capacity [kWh]")
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    figure.suptitle(
        "Peak-power-feasible pack sweep; 3C discharge remains a conditional placeholder",
        fontsize=10,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    return figure


@dataclass(frozen=True)
class ReferenceMapSummary:
    """Headline values accompanying the reference data and figure."""

    csv_path: Path
    figure_path: Path
    sizing_csv_path: Path
    sizing_figure_path: Path
    trade_csv_path: Path
    trade_figure_path: Path
    mtow_demand_3km_kw: float
    loiter_start_demand_3km_kw: float
    loiter_crossing_mass_kg: float
    loiter_fuel_to_crossing_kg: float
    loiter_time_to_crossing_h: float
    loiter_time_power_feasible_fraction: float
    loiter_start_minimum_capacity_kwh: float
    loiter_start_minimum_discharge_c_rate: float
    charge_side_candidate_points: int
    discharge_removed_points: int
    sampled_grid_points_removed_fraction: float
    mtow_engine_only_boundary_m: float
    loiter_start_engine_only_boundary_m: float
    mtow_charge_to_engine_boundary_m: float
    loiter_start_charge_to_engine_boundary_m: float
    service_ceiling_m: float
    absolute_ceiling_m: float


def generate_reference_feasibility_artifacts(
    output_directory: str | Path,
    *,
    parameters: DiagnosticParameters,
    loiter_start_mass_kg: float,
    loiter_end_mass_kg: float,
    loiter_duration_h: float,
    efficiency_evaluation_soc: float,
    usable_soc_low: float,
    usable_soc_high: float,
) -> ReferenceMapSummary:
    """Generate the band-interpretation aircraft's milestone data products."""
    # Band-interpretation reference design; see assumptions.md O-12.
    wing_area_m2 = 7.59175537062125
    aspect_ratio = 16.0
    oswald_efficiency = 0.78
    cd0 = 0.028
    cl_max = 1.5
    propeller_efficiency = 0.85
    dry_mass_kg = 711.3898016890586
    engine_power_kw = 86.7791369750147
    battery_capacity_kwh = 10.0
    powertrain = SeriesPowertrain()
    # Explicit placeholder calibration; see assumptions.md E-02, E-04 and E-05.
    engine = Turboshaft(
        rated_power_kw=engine_power_kw,
        sfc_rated_kg_kwh=0.45,
        idle_fuel_fraction=parameters.idle_fuel_fraction,
        lapse_exponent=0.8,
        min_power_fraction=0.15,
        allow_shutdown=True,
        restart_fuel_kg=parameters.restart_fuel_kg,
    )
    battery = BatteryPack(
        capacity_kwh=battery_capacity_kwh,
        discharge_c_rate=parameters.discharge_c_rate,
        charge_c_rate=parameters.charge_c_rate,
    )
    altitudes = np.linspace(0.0, 10_000.0, 101)
    masses = np.unique(
        np.append(
            np.linspace(dry_mass_kg + 5.0, 1000.0, 49),
            float(loiter_start_mass_kg),
        )
    )
    points = build_feasibility_map(
        altitudes,
        masses,
        wing_area_m2=wing_area_m2,
        aspect_ratio=aspect_ratio,
        oswald_efficiency=oswald_efficiency,
        cd0=cd0,
        cl_max=cl_max,
        propeller_efficiency=propeller_efficiency,
        dry_mass_kg=dry_mass_kg,
        engine=engine,
        battery=battery,
        powertrain=powertrain,
        parameters=parameters,
        efficiency_evaluation_soc=efficiency_evaluation_soc,
        usable_soc_low=usable_soc_low,
        usable_soc_high=usable_soc_high,
    )
    directory = Path(output_directory)
    csv_path = write_feasibility_csv(
        points, directory / "cycle_feasibility_map.csv"
    )
    sizing_csv_path = write_pack_sizing_csv(
        points, directory / "cycle_pack_sizing_contours.csv"
    )
    loiter_demand = loiter_bus_demand_kw(
        3000.0,
        loiter_start_mass_kg,
        wing_area_m2=wing_area_m2,
        aspect_ratio=aspect_ratio,
        oswald_efficiency=oswald_efficiency,
        cd0=cd0,
        cl_max=cl_max,
        propeller_efficiency=propeller_efficiency,
        powertrain=powertrain,
    )
    crossing_mass = discharge_crossing_mass_kg(
        loiter_start_mass_kg,
        loiter_demand,
        parameters.discharge_c_rate * battery_capacity_kwh,
    )
    figure_path = directory / "cycle_feasibility_map.png"
    figure = plot_feasibility_map(
        points,
        figure_path,
        parameters=parameters,
        loiter_start_mass_kg=loiter_start_mass_kg,
        efficiency_evaluation_soc=efficiency_evaluation_soc,
        loiter_altitude_m=3000.0,
        loiter_end_mass_kg=loiter_end_mass_kg,
        discharge_crossing_mass_kg=crossing_mass,
    )
    import matplotlib.pyplot as plt

    plt.close(figure)
    sizing_figure_path = directory / "cycle_pack_sizing_contours.png"
    sizing_figure = plot_pack_sizing_contours(
        points,
        sizing_figure_path,
        discharge_c_rate=parameters.discharge_c_rate,
        battery_capacity_kwh=battery_capacity_kwh,
    )
    plt.close(sizing_figure)

    peak_bus_kw = float(powertrain.bus_power_from_engine(engine_power_kw)) + (
        parameters.discharge_c_rate * battery_capacity_kwh
    )
    threshold_capacity_kwh = minimum_pack_capacity_kwh(
        loiter_demand, parameters.discharge_c_rate
    )
    peak_boost_kw = peak_bus_kw - float(
        powertrain.bus_power_from_engine(engine_power_kw)
    )
    peak_power_capacity_kwh = minimum_pack_capacity_kwh(
        peak_boost_kw, parameters.discharge_c_rate
    )
    trade_capacities = np.unique(
        np.append(
            np.linspace(peak_power_capacity_kwh, 20.0, 41),
            threshold_capacity_kwh,
        )
    )
    trade_points = build_pack_size_trade(
        trade_capacities,
        np.linspace(0.05, 0.35, 7),
        reference_capacity_kwh=battery_capacity_kwh,
        loiter_start_mass_kg=loiter_start_mass_kg,
        loiter_fuel_floor_kg=loiter_end_mass_kg - dry_mass_kg,
        loiter_start_demand_bus_kw=loiter_demand,
        altitude_m=3000.0,
        wing_area_m2=wing_area_m2,
        aspect_ratio=aspect_ratio,
        peak_bus_kw=peak_bus_kw,
        engine=engine,
        powertrain=powertrain,
        charge_c_rate=parameters.charge_c_rate,
        discharge_c_rate=parameters.discharge_c_rate,
        efficiency_evaluation_soc=efficiency_evaluation_soc,
    )
    trade_csv_path = write_pack_trade_csv(
        trade_points, directory / "cycle_pack_trade.csv"
    )
    trade_figure_path = directory / "cycle_pack_trade.png"
    trade_figure = plot_pack_trade(trade_points, trade_figure_path)
    plt.close(trade_figure)

    boundary_options = dict(
        wing_area_m2=wing_area_m2,
        aspect_ratio=aspect_ratio,
        oswald_efficiency=oswald_efficiency,
        cd0=cd0,
        cl_max=cl_max,
        propeller_efficiency=propeller_efficiency,
        engine=engine,
        powertrain=powertrain,
    )
    charge_limit_kw = parameters.charge_c_rate * battery_capacity_kwh
    mtow_engine = find_power_boundary_altitude_m(
        1000.0, 0.0, **boundary_options
    )
    loiter_engine = find_power_boundary_altitude_m(
        loiter_start_mass_kg, 0.0, **boundary_options
    )
    mtow_charge = find_power_boundary_altitude_m(
        1000.0, charge_limit_kw, **boundary_options
    )
    loiter_charge = find_power_boundary_altitude_m(
        loiter_start_mass_kg, charge_limit_kw, **boundary_options
    )
    service_ceiling = find_power_boundary_altitude_m(
        1000.0, 0.0, climb_rate_mps=0.5, **boundary_options
    )
    mtow_demand = loiter_bus_demand_kw(
        3000.0,
        1000.0,
        powertrain=powertrain,
        **{key: value for key, value in boundary_options.items() if key not in ("engine", "powertrain")},
    )
    a, b = willans_coefficients(
        engine.rated_power_kw,
        engine.sfc_rated_kg_kwh,
        parameters.idle_fuel_fraction,
    )
    loiter_summary = loiter_feasibility_summary(
        loiter_start_mass_kg,
        loiter_demand,
        parameters.discharge_c_rate * battery_capacity_kwh,
        loiter_duration_h,
        a,
        b,
        powertrain.source_chain_efficiency,
    )
    removal = discharge_condition_removal(points)
    return ReferenceMapSummary(
        csv_path=csv_path,
        figure_path=figure_path,
        sizing_csv_path=sizing_csv_path,
        sizing_figure_path=sizing_figure_path,
        trade_csv_path=trade_csv_path,
        trade_figure_path=trade_figure_path,
        mtow_demand_3km_kw=mtow_demand,
        loiter_start_demand_3km_kw=loiter_demand,
        loiter_crossing_mass_kg=loiter_summary.crossing_mass_kg,
        loiter_fuel_to_crossing_kg=loiter_summary.fuel_burned_to_crossing_kg,
        loiter_time_to_crossing_h=loiter_summary.elapsed_to_crossing_h,
        loiter_time_power_feasible_fraction=(
            loiter_summary.loiter_time_power_feasible_fraction
        ),
        loiter_start_minimum_capacity_kwh=minimum_pack_capacity_kwh(
            loiter_demand, parameters.discharge_c_rate
        ),
        loiter_start_minimum_discharge_c_rate=minimum_discharge_c_rate(
            loiter_demand, battery_capacity_kwh
        ),
        charge_side_candidate_points=removal.charge_side_candidate_points,
        discharge_removed_points=removal.removed_points,
        sampled_grid_points_removed_fraction=(
            removal.sampled_grid_point_removed_fraction
        ),
        mtow_engine_only_boundary_m=mtow_engine,
        loiter_start_engine_only_boundary_m=loiter_engine,
        mtow_charge_to_engine_boundary_m=mtow_charge,
        loiter_start_charge_to_engine_boundary_m=loiter_charge,
        service_ceiling_m=service_ceiling,
        absolute_ceiling_m=mtow_engine,
    )
