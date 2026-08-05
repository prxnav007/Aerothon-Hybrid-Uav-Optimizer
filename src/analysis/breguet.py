"""Closed-form Breguet checks for the simulator's pure-thermal loiter path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from src.models.aerodynamics import drag_coefficient
from src.models.atmosphere import atmosphere
from src.models.engine import Turboshaft
from src.models.powertrain import SeriesPowertrain

__all__ = [
    "BreguetResult",
    "breguet_endurance",
    "breguet_loiter_estimate",
    "breguet_range",
    "effective_sfc",
    "endurance_optimal_cl",
]

GRAVITY_MPS2 = 9.80665
JOULES_PER_KWH = 3_600_000.0

SfcMode = Literal["fixed", "mean_operating_point"]


class _BreguetAircraft(Protocol):
    """Aircraft data needed by the analytical loiter estimate."""

    dry_mass_kg: float
    wing_area_m2: float
    cd0: float
    aspect_ratio: float
    oswald_e: float
    cl_max: float
    eta_prop: float
    engine: Turboshaft
    powertrain: SeriesPowertrain
    stall_margin: float


@dataclass(frozen=True)
class BreguetResult:
    """Analytical result and the operating point used to obtain it."""

    endurance_s: float
    cl: float
    cd: float
    lift_to_drag: float
    speed_initial_mps: float
    speed_final_mps: float
    sfc_used_kg_kwh: float
    effective_sfc_kg_kwh: float
    stall_limited: bool


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def _efficiency(name: str, value: float) -> float:
    result = _positive(name, value)
    if result > 1.0:
        raise ValueError(f"{name} must lie in (0, 1], got {value!r}")
    return result


def _weights(weight_initial_n: float, weight_final_n: float) -> tuple[float, float]:
    initial = _positive("weight_initial_n", weight_initial_n)
    final = _positive("weight_final_n", weight_final_n)
    if final > initial:
        raise ValueError(
            "weight_final_n must not exceed weight_initial_n, "
            f"got {final!r} > {initial!r}"
        )
    return initial, final


def _sfc_kg_j(sfc_kg_kwh: float) -> float:
    """Convert the public kg/kWh SFC boundary to the equation's kg/J."""
    return _positive("sfc_kg_kwh", sfc_kg_kwh) / JOULES_PER_KWH


def breguet_endurance(
    weight_initial_n: float,
    weight_final_n: float,
    rho: float,
    wing_area_m2: float,
    cl: float,
    cd: float,
    sfc_kg_kwh: float,
    eta_prop: float,
) -> float:
    """Propeller-aircraft endurance [s] at constant altitude and lift coefficient."""
    initial, final = _weights(weight_initial_n, weight_final_n)
    density = _positive("rho", rho)
    area = _positive("wing_area_m2", wing_area_m2)
    lift_coefficient = _positive("cl", cl)
    drag_coefficient_value = _positive("cd", cd)
    sfc_kg_j = _sfc_kg_j(sfc_kg_kwh)
    propeller_efficiency = _efficiency("eta_prop", eta_prop)

    # From dW/dt = -g*sfc*D*V/eta with V=sqrt(2W/(rho*S*CL)) and D=W/(L/D),
    # integrating W**-1.5 gives the W**-0.5 difference below.
    aerodynamic_factor = lift_coefficient**1.5 / drag_coefficient_value
    weight_integral = final**-0.5 - initial**-0.5
    return (
        propeller_efficiency
        / (GRAVITY_MPS2 * sfc_kg_j)
        * aerodynamic_factor
        * math.sqrt(2.0 * density * area)
        * weight_integral
    )


def breguet_range(
    weight_initial_n: float,
    weight_final_n: float,
    cl: float,
    cd: float,
    sfc_kg_kwh: float,
    eta_prop: float,
) -> float:
    """Propeller-aircraft range [m] at constant lift coefficient and SFC."""
    initial, final = _weights(weight_initial_n, weight_final_n)
    lift_coefficient = _positive("cl", cl)
    drag_coefficient_value = _positive("cd", cd)
    sfc_kg_j = _sfc_kg_j(sfc_kg_kwh)
    propeller_efficiency = _efficiency("eta_prop", eta_prop)

    return (
        propeller_efficiency
        / (GRAVITY_MPS2 * sfc_kg_j)
        * lift_coefficient
        / drag_coefficient_value
        * math.log(initial / final)
    )


def effective_sfc(sfc_engine_kg_kwh: float, powertrain: SeriesPowertrain) -> float:
    """Fuel per propeller-shaft kWh after all series-chain losses [kg/kWh]."""
    sfc = _positive("sfc_engine_kg_kwh", sfc_engine_kg_kwh)
    return sfc / powertrain.chain_efficiency


def endurance_optimal_cl(
    cd0: float,
    aspect_ratio: float,
    oswald_e: float,
    cl_max: float,
    stall_margin: float = 1.2,
) -> float:
    """Return the minimum-power lift coefficient subject to the stall margin."""
    parasite_drag = _positive("cd0", cd0)
    aspect_ratio_value = _positive("aspect_ratio", aspect_ratio)
    span_efficiency = _positive("oswald_e", oswald_e)
    maximum_lift = _positive("cl_max", cl_max)
    margin = _positive("stall_margin", stall_margin)
    if margin < 1.0:
        raise ValueError(f"stall_margin must be at least 1, got {stall_margin!r}")

    unconstrained = math.sqrt(
        3.0 * parasite_drag * math.pi * aspect_ratio_value * span_efficiency
    )
    return min(unconstrained, maximum_lift / margin**2)


def _loiter_speed(weight_n: float, rho: float, area_m2: float, cl: float) -> float:
    return math.sqrt(2.0 * weight_n / (rho * area_m2 * cl))


def _engine_power_kw(
    weight_n: float,
    speed_mps: float,
    cl: float,
    cd: float,
    eta_prop: float,
    powertrain: SeriesPowertrain,
) -> float:
    thrust_power_kw = weight_n * (cd / cl) * speed_mps / 1000.0
    propeller_shaft_kw = thrust_power_kw / eta_prop
    return propeller_shaft_kw / powertrain.chain_efficiency


def breguet_loiter_estimate(
    aircraft: _BreguetAircraft,
    altitude_m: float,
    fuel_available_kg: float,
    sfc_mode: SfcMode = "mean_operating_point",
) -> BreguetResult:
    """Estimate pure-thermal loiter endurance for an aircraft and usable fuel."""
    fuel = float(fuel_available_kg)
    if not math.isfinite(fuel) or fuel < 0.0:
        raise ValueError(
            f"fuel_available_kg must be finite and non-negative, got {fuel_available_kg!r}"
        )
    if sfc_mode not in ("fixed", "mean_operating_point"):
        raise ValueError(
            "sfc_mode must be 'fixed' or 'mean_operating_point', "
            f"got {sfc_mode!r}"
        )
    if aircraft.powertrain.load_dependent:
        raise ValueError(
            "Breguet requires constant conversion efficiencies; "
            "load-dependent powertrain losses have no single effective SFC"
        )

    dry_mass = _positive("aircraft.dry_mass_kg", aircraft.dry_mass_kg)
    area = _positive("aircraft.wing_area_m2", aircraft.wing_area_m2)
    eta_prop = _efficiency("aircraft.eta_prop", aircraft.eta_prop)
    state = atmosphere(float(altitude_m))
    rho = float(state.density_kg_m3)

    cl = endurance_optimal_cl(
        aircraft.cd0,
        aircraft.aspect_ratio,
        aircraft.oswald_e,
        aircraft.cl_max,
        aircraft.stall_margin,
    )
    cd = float(drag_coefficient(cl, aircraft.cd0, aircraft.aspect_ratio, aircraft.oswald_e))
    unconstrained_cl = math.sqrt(
        3.0 * aircraft.cd0 * math.pi * aircraft.aspect_ratio * aircraft.oswald_e
    )
    stall_limited = cl < unconstrained_cl

    weight_final_n = dry_mass * GRAVITY_MPS2
    weight_initial_n = (dry_mass + fuel) * GRAVITY_MPS2
    speed_initial = _loiter_speed(weight_initial_n, rho, area, cl)
    speed_final = _loiter_speed(weight_final_n, rho, area, cl)

    if sfc_mode == "fixed":
        sfc_used = aircraft.engine.sfc_rated_kg_kwh
    else:
        initial_power = _engine_power_kw(
            weight_initial_n,
            speed_initial,
            cl,
            cd,
            eta_prop,
            aircraft.powertrain,
        )
        final_power = _engine_power_kw(
            weight_final_n,
            speed_final,
            cl,
            cd,
            eta_prop,
            aircraft.powertrain,
        )
        sfc_used = float(aircraft.engine.sfc_kg_kwh(0.5 * (initial_power + final_power)))

    chain_sfc = effective_sfc(sfc_used, aircraft.powertrain)
    endurance = breguet_endurance(
        weight_initial_n,
        weight_final_n,
        rho,
        area,
        cl,
        cd,
        chain_sfc,
        eta_prop,
    )
    return BreguetResult(
        endurance_s=endurance,
        cl=cl,
        cd=cd,
        lift_to_drag=cl / cd,
        speed_initial_mps=speed_initial,
        speed_final_mps=speed_final,
        sfc_used_kg_kwh=sfc_used,
        effective_sfc_kg_kwh=chain_sfc,
        stall_limited=stall_limited,
    )
