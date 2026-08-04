"""Point-mass performance model for a fixed-wing UAV: drag, power, and speeds.

Sits directly above :mod:`src.models.atmosphere` and directly below the mission
simulator. Given a flight condition (density, speed, weight) and a geometry
(wing area, aspect ratio, drag polar), it returns drag, power required, and the
characteristic speeds that define the flight envelope.

Model
-----
Point-mass (3-DoF) performance with a **parabolic drag polar**::

    C_D = C_D0 + C_L**2 / (pi * AR * e)

Lift is taken equal to weight (``C_L = W / (q*S)``), which is steady level
flight with the small-climb-angle approximation ``cos(gamma) ~ 1``. For this
mission that is comfortably valid: at 65 m/s and the ~3 m/s climb rates flown
here, gamma < 3 deg and cos(gamma) > 0.9986, a <0.15 % error on C_L.

Assumptions and validity limits
-------------------------------
* **Incompressible flow.** No Mach or compressibility correction is applied.
  Valid because this vehicle cruises at M ~ 0.22 (69.44 m/s at 6 km, where
  a = 316.4 m/s), well under the M = 0.3 threshold. This is *asserted*, not
  assumed - see ``test_cruise_mach_justifies_incompressible_model`` in
  ``tests/test_aerodynamics.py``, which computes it from the atmosphere module.
* **Parabolic polar.** C_D0 and e are treated as constant with C_L. Real polars
  bend up near C_Lmax (separation) and vary with Reynolds number; this model is
  a conceptual-design approximation and should not be trusted within a few
  percent of stall.
* **Small climb angle**, as above. Not valid for steep climbs or dives.
* **Propeller modelled as a scalar efficiency** ``eta_prop``. No advance-ratio
  or blade-element modelling, no windmilling drag.
* Out of scope: ground effect, trim drag, stability and control, structural
  loads, compressibility.

Unit convention - READ THIS
---------------------------
**SI throughout, and weight is in NEWTONS, not kilograms.** Mixing N and kg is
by far the most likely integration bug against this module: a 1000 kg MTOW
vehicle has ``weight_n = 9810.0``, not ``1000.0``. Passing kg would understate
drag by a factor of ~9.81 in the induced term and silently produce a wildly
optimistic endurance.

    weight        newtons  [N]
    density       kg/m^3
    speed         m/s
    area          m^2
    power         watts    [W]
    rate of climb m/s      (positive up, negative in descent)
    angles        radians
    C_D0, e, AR, C_L, C_D, eta_prop, safety_margin: dimensionless

Verification approach
---------------------
Unlike the ISA atmosphere module there is *no authoritative table* to check
against: C_D0, Oswald efficiency and C_Lmax are engineering estimates, not
standards. Nothing here is presented as an authoritative value. Verification is
**mathematical self-consistency**: every closed-form result is confirmed in the
tests by independent numerical computation from the same drag polar - the
speeds that numerically minimise ``D*V`` and ``D`` over a fine velocity sweep
must land on the analytic ``speed_min_power`` and ``speed_best_ld``.

References consulted for the *forms* of the equations (August 2026)
-------------------------------------------------------------------
* Raymer, *Aircraft Design: A Conceptual Approach*, sec. 12.6 - Oswald
  efficiency correlations and the equivalent skin-friction (C_fe) method. The
  book itself is not available online; the straight-wing correlation
  ``e = 1.78(1 - 0.045*AR**0.68) - 0.64`` was confirmed verbatim against
  secondary design references, and Wikipedia's "Oswald efficiency number"
  article cites Raymer sec. 12.6 for it and gives the typical range e = 0.7-0.85
  for conventional aircraft.
* Anderson, *Aircraft Performance and Design* - parabolic drag polar, maximum
  L/D and minimum-power conditions. Not available online; the conditions were
  confirmed independently (below).
* Gudmundsson, *General Aviation Aircraft Design* - drag buildup conventions.
  Not available online; cited for completeness, not relied upon.
* MIT 16.unified aircraft-performance notes and equivalent courseware -
  confirmed the minimum-power condition ``C_Di = 3*C_D0`` (so ``C_D = 4*C_D0``),
  the resulting ``V_minpower = 3**(-1/4) * V_bestLD ~ 0.760 * V_bestLD``, and
  ``(L/D)_minpower = sqrt(3)/2 * (L/D)_max``. All three are re-derived as
  executable assertions in the test suite.

Every relation in this module is additionally checked numerically in
``tests/test_aerodynamics.py``, which is the verification that actually matters.

Notes
-----
Pure and stateless: no module-level configuration, no import from
``src/config.py``, and no import of the atmosphere module - density is always an
argument. Weight, geometry and coefficients are always arguments, which is what
lets wing area and aspect ratio become genetic-algorithm design variables later
without touching this file.

Scalar input (anything with ``ndim == 0``) returns a Python ``float``/``bool``/
``str``; if any input is an ndarray the result is an ndarray of the broadcast
shape.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "AeroState",
    "HighAspectRatioWarning",
    "drag_coefficient",
    "drag_force",
    "dynamic_pressure",
    "evaluate",
    "induced_drag_factor",
    "lift_coefficient",
    "lift_to_drag",
    "loiter_speed",
    "max_lift_to_drag",
    "oswald_efficiency",
    "parasite_drag_from_wetted_area",
    "rate_of_climb",
    "shaft_power_required",
    "speed_best_ld",
    "speed_min_power",
    "stall_speed",
    "thrust_power_required",
]

FloatOrArray = float | npt.NDArray[np.float64]
BoolOrArray = bool | npt.NDArray[np.bool_]
StrOrArray = str | npt.NDArray[np.str_]

# Aspect ratio above which the Raymer Oswald correlation is extrapolating. See
# `oswald_efficiency` for why this matters inside a GA.
RAYMER_AR_LIMIT: float = 12.0

# Raymer's swept-wing correlation is fitted for leading-edge sweep beyond this.
RAYMER_SWEEP_LIMIT_RAD: float = np.deg2rad(30.0)


class HighAspectRatioWarning(UserWarning):
    """Raised by :func:`oswald_efficiency` when extrapolating past AR = 12.

    Its own category (rather than a bare ``UserWarning``) so callers can filter
    it, promote it to an error, or assert on it in tests without catching
    unrelated warnings.
    """


# ---------------------------------------------------------------------------
# Argument validation
#
# These exist because every quantity below appears in a denominator somewhere.
# A GA mutation that produces AR = 0 or e = 0 would otherwise yield inf or a
# NEGATIVE induced drag factor, and a negative drag would look to the optimiser
# like free thrust - the single most dangerous silent failure in this module.
# ---------------------------------------------------------------------------


def _require_positive(name: str, value: FloatOrArray) -> npt.NDArray[np.float64]:
    """Coerce to float64 and raise ``ValueError`` unless every element is > 0.

    Returns the coerced array so callers can validate and convert in one step.
    """
    arr = np.asarray(value, dtype=np.float64)
    ok = np.isfinite(arr) & (arr > 0.0)
    if not bool(np.all(ok)):
        raise ValueError(
            f"{name} must be finite and strictly positive, got {_describe(arr, ok)}. "
            f"It appears in a denominator; a non-positive value would produce "
            f"infinite or negative drag rather than an error."
        )
    return arr


def _require_finite(name: str, value: FloatOrArray) -> npt.NDArray[np.float64]:
    """Coerce to float64 and raise unless every element is finite (sign free).

    Used for quantities that are legitimately negative, such as climb rate.
    """
    arr = np.asarray(value, dtype=np.float64)
    ok = np.isfinite(arr)
    if not bool(np.all(ok)):
        raise ValueError(f"{name} must be finite, got {_describe(arr, ok)}.")
    return arr


def _require_efficiency(name: str, value: FloatOrArray) -> npt.NDArray[np.float64]:
    """Raise unless every element is in (0, 1]. For propulsive efficiencies."""
    arr = np.asarray(value, dtype=np.float64)
    ok = np.isfinite(arr) & (arr > 0.0) & (arr <= 1.0)
    if not bool(np.all(ok)):
        raise ValueError(
            f"{name} must be a finite efficiency in (0, 1], got "
            f"{_describe(arr, ok)}. Values above 1 would extract more shaft "
            f"power than the propeller converts."
        )
    return arr


def _describe(arr: npt.NDArray[np.float64], ok: npt.NDArray[np.bool_]) -> str:
    """Format the offending elements of a failed validation for the message."""
    bad = np.atleast_1d(arr)[~np.atleast_1d(ok)]
    shown = ", ".join(f"{float(v)!r}" for v in bad[:5])
    if bad.size > 5:
        shown += f", ... ({bad.size} offending values)"
    return shown


def _restore(result: npt.NDArray[np.float64], *inputs: FloatOrArray) -> FloatOrArray:
    """Return a Python float when every input was scalar, else the ndarray.

    NumPy turns scalar arithmetic into 0-d arrays; without this they would leak
    into the dataclass fields and into downstream simulator arithmetic.
    """
    if all(np.ndim(x) == 0 for x in inputs):
        return float(result)
    return result


# ---------------------------------------------------------------------------
# Basic relations
# ---------------------------------------------------------------------------


def dynamic_pressure(rho: FloatOrArray, v_mps: FloatOrArray) -> FloatOrArray:
    """Free-stream dynamic pressure.

    Equation: ``q = 0.5 * rho * V**2``.

    Args:
        rho: Air density [kg/m^3], > 0.
        v_mps: True airspeed [m/s], > 0. Zero is rejected rather than returning
            q = 0, because every consumer of q divides by it; a V = 0 flight
            condition is a simulator bug, not a physical state this model covers.

    Returns:
        Dynamic pressure [Pa].

    Raises:
        ValueError: If rho or v_mps is non-positive or non-finite.

    Example:
        >>> round(dynamic_pressure(0.9091, 69.44), 1)
        2191.8
    """
    r = _require_positive("rho", rho)
    v = _require_positive("v_mps", v_mps)
    # Dynamic pressure
    return _restore(0.5 * r * v**2, rho, v_mps)


def lift_coefficient(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    v_mps: FloatOrArray,
    wing_area_m2: FloatOrArray,
) -> FloatOrArray:
    """Lift coefficient required for steady level flight.

    Equation: ``C_L = W / (q * S)``, from vertical equilibrium L = W.

    ASSUMPTION: small climb angle. The exact relation is
    ``C_L = W*cos(gamma) / (q*S)``; we take cos(gamma) ~ 1. At this mission's
    climb rates (< 3 m/s at ~65 m/s) gamma < 3 deg and the error is < 0.15 %.

    Args:
        weight_n: Aircraft weight [N] - NEWTONS, not kg.
        rho: Air density [kg/m^3], > 0.
        v_mps: True airspeed [m/s], > 0.
        wing_area_m2: Reference wing area [m^2], > 0.

    Returns:
        Lift coefficient [-].

    Raises:
        ValueError: If any argument is non-positive or non-finite.

    Example:
        >>> round(lift_coefficient(9810.0, 0.9091, 69.44, 10.0), 4)
        0.4476
    """
    w = _require_positive("weight_n", weight_n)
    r = _require_positive("rho", rho)
    v = _require_positive("v_mps", v_mps)
    s = _require_positive("wing_area_m2", wing_area_m2)
    # Dynamic pressure
    q = 0.5 * r * v**2
    # Lift coefficient from vertical equilibrium, L = W
    return _restore(w / (q * s), weight_n, rho, v_mps, wing_area_m2)


def induced_drag_factor(
    aspect_ratio: FloatOrArray, oswald_e: FloatOrArray
) -> FloatOrArray:
    """Induced drag factor k of the parabolic polar.

    Equation: ``k = 1 / (pi * AR * e)``, so that ``C_Di = k * C_L**2``.

    Args:
        aspect_ratio: Wing aspect ratio b^2/S [-], > 0.
        oswald_e: Oswald span efficiency [-], > 0. Values <= 0 are rejected:
            they would make k negative, turning induced drag into thrust, which
            a genetic algorithm would exploit enthusiastically.

    Returns:
        Induced drag factor [-].

    Raises:
        ValueError: If aspect_ratio or oswald_e is non-positive or non-finite.

    Example:
        >>> round(induced_drag_factor(16.0, 0.78), 6)
        0.025506
    """
    ar = _require_positive("aspect_ratio", aspect_ratio)
    e = _require_positive("oswald_e", oswald_e)
    # Induced drag factor, k = 1 / (pi * AR * e)
    return _restore(1.0 / (np.pi * ar * e), aspect_ratio, oswald_e)


def drag_coefficient(
    c_l: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
) -> FloatOrArray:
    """Total drag coefficient from the parabolic drag polar.

    Equation: ``C_D = C_D0 + k * C_L**2`` with ``k = 1/(pi*AR*e)``.

    Args:
        c_l: Lift coefficient [-]. May be negative (pushover); only C_L**2
            enters, so the polar is symmetric about C_L = 0.
        cd0: Zero-lift (parasite) drag coefficient [-], > 0.
        aspect_ratio: Wing aspect ratio [-], > 0.
        oswald_e: Oswald span efficiency [-], > 0.

    Returns:
        Total drag coefficient [-].

    Raises:
        ValueError: If cd0, aspect_ratio or oswald_e is non-positive.

    Example:
        >>> round(drag_coefficient(0.447577, 0.028, 16.0, 0.78), 6)
        0.033109
    """
    cl = _require_finite("c_l", c_l)
    cd_0 = _require_positive("cd0", cd0)
    ar = _require_positive("aspect_ratio", aspect_ratio)
    e = _require_positive("oswald_e", oswald_e)
    # Parabolic drag polar
    cd = cd_0 + cl**2 / (np.pi * ar * e)
    return _restore(cd, c_l, cd0, aspect_ratio, oswald_e)


def drag_force(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    v_mps: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
) -> FloatOrArray:
    """Total drag in steady level flight.

    Equations: ``q = 0.5*rho*V**2``; ``C_L = W/(q*S)``;
    ``C_D = C_D0 + k*C_L**2``; ``D = q*S*C_D``.

    Args:
        weight_n: Weight [N]. rho: Density [kg/m^3]. v_mps: TAS [m/s].
        wing_area_m2: Wing area [m^2]. cd0: Parasite drag coefficient [-].
        aspect_ratio: Aspect ratio [-]. oswald_e: Oswald efficiency [-].
        All must be strictly positive.

    Returns:
        Drag force [N].

    Raises:
        ValueError: If any argument is non-positive or non-finite.

    Example:
        >>> round(drag_force(9810.0, 0.9091, 69.44, 10.0, 0.028, 16.0, 0.78), 1)
        725.7
    """
    inputs = (weight_n, rho, v_mps, wing_area_m2, cd0, aspect_ratio, oswald_e)
    _, _, _, d, _ = _polar_one_pass(*inputs)
    return _restore(d, *inputs)


def lift_to_drag(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    v_mps: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
) -> FloatOrArray:
    """Lift-to-drag ratio at this flight condition.

    Equation: ``L/D = C_L / C_D``. In steady level flight this equals ``W/D``,
    since L = W.

    Args: as :func:`drag_force`. All strictly positive.

    Returns:
        Lift-to-drag ratio [-].

    Raises:
        ValueError: If any argument is non-positive or non-finite.

    Example:
        >>> round(lift_to_drag(9810.0, 0.9091, 69.44, 10.0, 0.028, 16.0, 0.78), 2)
        13.52
    """
    inputs = (weight_n, rho, v_mps, wing_area_m2, cd0, aspect_ratio, oswald_e)
    _, cl, cd, _, _ = _polar_one_pass(*inputs)
    return _restore(cl / cd, *inputs)


def _polar_one_pass(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    v_mps: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Validate once and evaluate the whole polar. Internal.

    Returns ``(q, C_L, C_D, D, V)`` as ndarrays. Every public function that
    needs more than one of these goes through here, so the polar is written out
    exactly once in this module.
    """
    w = _require_positive("weight_n", weight_n)
    r = _require_positive("rho", rho)
    v = _require_positive("v_mps", v_mps)
    s = _require_positive("wing_area_m2", wing_area_m2)
    cd_0 = _require_positive("cd0", cd0)
    ar = _require_positive("aspect_ratio", aspect_ratio)
    e = _require_positive("oswald_e", oswald_e)

    # Dynamic pressure
    q = 0.5 * r * v**2
    # Lift coefficient from vertical equilibrium, L = W (small climb angle)
    cl = w / (q * s)
    # Parabolic drag polar
    cd = cd_0 + cl**2 / (np.pi * ar * e)
    # Total drag
    d = q * s * cd
    return q, cl, cd, d, v


# ---------------------------------------------------------------------------
# Power required
# ---------------------------------------------------------------------------


def thrust_power_required(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    v_mps: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
    climb_rate_mps: FloatOrArray = 0.0,
) -> FloatOrArray:
    """Propulsive (thrust) power required, before propeller losses.

    Equation: ``P_thrust = D*V + W*ROC`` - drag power plus the rate of gain of
    potential energy.

    **This value may be NEGATIVE.** In a descent, ``W*ROC`` is negative and can
    exceed the drag power, meaning gravity supplies more energy than drag
    dissipates. That is physically meaningful (it is what sets the power-off
    glide), so this function returns it unclamped. It is
    :func:`shaft_power_required` that applies the clamp, because a real
    propeller-driven UAV cannot regenerate.

    Args:
        weight_n .. oswald_e: as :func:`drag_force`, all strictly positive.
        climb_rate_mps: Rate of climb [m/s], positive up. May be negative.

    Returns:
        Thrust power required [W]; negative in a sufficiently steep descent.

    Raises:
        ValueError: If any argument other than climb_rate_mps is non-positive.

    Example:
        Level flight at the cruise point needs ~50.4 kW of thrust power.

        >>> p = thrust_power_required(9810.0, 0.9091, 69.44, 10.0, 0.028, 16.0, 0.78)
        >>> round(p / 1000.0, 1)
        50.4
    """
    inputs = (weight_n, rho, v_mps, wing_area_m2, cd0, aspect_ratio, oswald_e)
    _, _, _, d, v = _polar_one_pass(*inputs)
    w = np.asarray(weight_n, dtype=np.float64)
    roc = _require_finite("climb_rate_mps", climb_rate_mps)
    # Thrust power = drag power + rate of potential energy gain
    p_thrust = d * v + w * roc
    return _restore(p_thrust, *inputs, climb_rate_mps)


def shaft_power_required(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    v_mps: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
    eta_prop: FloatOrArray,
    climb_rate_mps: FloatOrArray = 0.0,
) -> tuple[FloatOrArray, BoolOrArray]:
    """Shaft power required at the propeller, clamped at zero.

    Equation: ``P_shaft = max(0, D*V + W*ROC) / eta_prop``.

    WHY THE CLAMP: a propeller-driven UAV cannot regenerate. When the descent is
    steep enough that ``P_thrust <= 0``, the aircraft is windmilling or at idle
    and the correct shaft power demand is zero, not a negative number. Returning
    a negative value here would feed the energy-management controller a
    *negative* demand, which it would happily integrate as battery recharge -
    inventing energy and inflating endurance. The boolean flag exists so the
    simulator handles that segment explicitly instead of inferring it from a
    zero that could equally mean "unpowered" or "coincidentally zero drag".

    Args:
        weight_n .. oswald_e: as :func:`drag_force`, all strictly positive.
        eta_prop: Propeller efficiency [-], in (0, 1].
        climb_rate_mps: Rate of climb [m/s], positive up. May be negative.

    Returns:
        ``(shaft_power_w, power_off)``. Power is [W] and always >= 0.
        ``power_off`` is True where the unclamped thrust power was <= 0, i.e.
        the aircraft is in an idle/windmilling descent.

    Raises:
        ValueError: If any argument is out of range.

    Example:
        Level cruise at eta_prop = 0.8 needs ~63.0 kW of shaft power.

        >>> p, off = shaft_power_required(
        ...     9810.0, 0.9091, 69.44, 10.0, 0.028, 16.0, 0.78, 0.8)
        >>> round(p / 1000.0, 1), off
        (63.0, False)

        A steep descent demands no shaft power at all.

        >>> p, off = shaft_power_required(
        ...     9810.0, 0.9091, 65.0, 10.0, 0.028, 16.0, 0.78, 0.8,
        ...     climb_rate_mps=-6.33)
        >>> p, off
        (0.0, True)
    """
    inputs = (weight_n, rho, v_mps, wing_area_m2, cd0, aspect_ratio, oswald_e)
    _, _, _, d, v = _polar_one_pass(*inputs)
    w = np.asarray(weight_n, dtype=np.float64)
    roc = _require_finite("climb_rate_mps", climb_rate_mps)
    eta = _require_efficiency("eta_prop", eta_prop)

    # Thrust power = drag power + rate of potential energy gain
    p_thrust = d * v + w * roc
    # Power-off (windmilling / idle descent): gravity alone exceeds drag losses.
    power_off = p_thrust <= 0.0
    # Shaft power at the propeller, clamped: no regeneration is possible.
    p_shaft = np.where(power_off, 0.0, p_thrust / eta)

    all_inputs = (*inputs, eta_prop, climb_rate_mps)
    if all(np.ndim(x) == 0 for x in all_inputs):
        return float(p_shaft), bool(power_off)
    return p_shaft, power_off


# ---------------------------------------------------------------------------
# Characteristic speeds
# ---------------------------------------------------------------------------


def stall_speed(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cl_max: FloatOrArray,
) -> FloatOrArray:
    """1g stall speed.

    Condition: the slowest speed at which the wing can still generate L = W, so
    C_L is at its maximum. Setting ``W = 0.5*rho*V**2*S*C_Lmax`` and solving:

        ``V_stall = sqrt(2*W / (rho * S * C_Lmax))``

    Args:
        weight_n: Weight [N]. rho: Density [kg/m^3].
        wing_area_m2: Wing area [m^2]. cl_max: Maximum lift coefficient [-].
        All strictly positive.

    Returns:
        1g stall speed [m/s].

    Raises:
        ValueError: If any argument is non-positive or non-finite.

    Example:
        >>> round(stall_speed(9810.0, 0.9091, 10.0, 1.5), 2)
        37.93
    """
    w = _require_positive("weight_n", weight_n)
    r = _require_positive("rho", rho)
    s = _require_positive("wing_area_m2", wing_area_m2)
    clm = _require_positive("cl_max", cl_max)
    # Stall speed, 1g: V = sqrt(2W / (rho * S * C_Lmax))
    v = np.sqrt(2.0 * w / (r * s * clm))
    return _restore(v, weight_n, rho, wing_area_m2, cl_max)


def max_lift_to_drag(
    cd0: FloatOrArray, aspect_ratio: FloatOrArray, oswald_e: FloatOrArray
) -> FloatOrArray:
    """Maximum lift-to-drag ratio of the polar.

    Condition: ``C_Di == C_D0``. Maximising ``C_L/(C_D0 + k*C_L**2)`` gives
    ``C_L = sqrt(C_D0/k)``, at which the induced and parasite contributions are
    equal and ``C_D = 2*C_D0``. Hence

        ``(L/D)_max = 1 / (2*sqrt(C_D0*k)) = 0.5*sqrt(pi*AR*e / C_D0)``

    Note this depends only on the polar - not on weight, density or speed. It is
    a property of the airframe, which is why it is the natural figure of merit
    for the GA's wing sizing.

    Args:
        cd0: Parasite drag coefficient [-], > 0.
        aspect_ratio: Aspect ratio [-], > 0.
        oswald_e: Oswald efficiency [-], > 0.

    Returns:
        Maximum lift-to-drag ratio [-].

    Raises:
        ValueError: If any argument is non-positive or non-finite.

    Example:
        >>> round(max_lift_to_drag(0.028, 16.0, 0.78), 2)
        18.71
    """
    cd_0 = _require_positive("cd0", cd0)
    ar = _require_positive("aspect_ratio", aspect_ratio)
    e = _require_positive("oswald_e", oswald_e)
    # Maximum L/D condition: C_Di = C_D0  =>  (L/D)max = 0.5*sqrt(pi*AR*e/C_D0)
    ld = 0.5 * np.sqrt(np.pi * ar * e / cd_0)
    return _restore(ld, cd0, aspect_ratio, oswald_e)


def speed_best_ld(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
) -> FloatOrArray:
    """Speed for maximum L/D - best glide, and best range for a propeller UAV.

    Condition: ``C_Di == C_D0``, i.e. induced drag exactly equals parasite drag.
    That gives ``C_L_bestLD = sqrt(C_D0/k) = sqrt(C_D0*pi*AR*e)``; substituting
    into ``V = sqrt(2W/(rho*S*C_L))``:

        ``V_bestLD = sqrt(2W/(rho*S)) * (C_D0*pi*AR*e)**-0.25``

    This minimises *drag*, hence maximises range for a propeller aircraft (whose
    fuel/energy flow scales with power, but whose range integrand scales with
    L/D). It is NOT the endurance speed - see :func:`speed_min_power`.

    Args:
        weight_n: Weight [N]. rho: Density [kg/m^3].
        wing_area_m2: Wing area [m^2]. cd0: Parasite drag coefficient [-].
        aspect_ratio: Aspect ratio [-]. oswald_e: Oswald efficiency [-].
        All strictly positive.

    Returns:
        Best-L/D speed [m/s].

    Raises:
        ValueError: If any argument is non-positive or non-finite.

    Example:
        >>> round(speed_best_ld(9810.0, 0.9091, 10.0, 0.028, 16.0, 0.78), 2)
        45.39
    """
    w = _require_positive("weight_n", weight_n)
    r = _require_positive("rho", rho)
    s = _require_positive("wing_area_m2", wing_area_m2)
    cd_0 = _require_positive("cd0", cd0)
    ar = _require_positive("aspect_ratio", aspect_ratio)
    e = _require_positive("oswald_e", oswald_e)
    # Maximum L/D condition: C_Di = C_D0
    v = np.sqrt(2.0 * w / (r * s)) * np.power(cd_0 * np.pi * ar * e, -0.25)
    return _restore(v, weight_n, rho, wing_area_m2, cd0, aspect_ratio, oswald_e)


def speed_min_power(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
) -> FloatOrArray:
    """Speed for minimum power required - maximum endurance for a propeller UAV.

    Condition: ``C_Di == 3*C_D0`` (so ``C_D = 4*C_D0``). Minimising power
    ``P = D*V`` rather than drag ``D`` shifts the optimum slower, because power
    carries an extra factor of V: writing ``P ~ C_D/C_L**1.5`` and setting
    dP/dC_L = 0 gives ``C_D0 = k*C_L**2/3``, i.e. induced drag three times
    parasite drag. That factor 3 is where the 3 in the formula comes from:

        ``C_L_minpower = sqrt(3*C_D0*pi*AR*e)``
        ``V_minpower   = sqrt(2W/(rho*S)) * (3*C_D0*pi*AR*e)**-0.25``
                       = V_bestLD * 3**-0.25 ~ 0.7598 * V_bestLD

    This is THE speed for this mission, whose objective is maximum endurance.
    But note it is frequently below stall for a high-AR wing - use
    :func:`loiter_speed`, which applies the stall-margin cap.

    Args: as :func:`speed_best_ld`, all strictly positive.

    Returns:
        Minimum-power speed [m/s].

    Raises:
        ValueError: If any argument is non-positive or non-finite.

    Example:
        >>> round(speed_min_power(9810.0, 0.9091, 10.0, 0.028, 16.0, 0.78), 2)
        34.49
    """
    w = _require_positive("weight_n", weight_n)
    r = _require_positive("rho", rho)
    s = _require_positive("wing_area_m2", wing_area_m2)
    cd_0 = _require_positive("cd0", cd0)
    ar = _require_positive("aspect_ratio", aspect_ratio)
    e = _require_positive("oswald_e", oswald_e)
    # Minimum power condition: C_Di = 3 * C_D0
    v = np.sqrt(2.0 * w / (r * s)) * np.power(3.0 * cd_0 * np.pi * ar * e, -0.25)
    return _restore(v, weight_n, rho, wing_area_m2, cd0, aspect_ratio, oswald_e)


def loiter_speed(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
    cl_max: FloatOrArray,
    safety_margin: FloatOrArray = 1.2,
) -> tuple[FloatOrArray, StrOrArray]:
    """Achievable loiter speed, capping the min-power condition at stall margin.

    The unconstrained endurance optimum is :func:`speed_min_power`, but for a
    high-aspect-ratio wing the C_L it demands is often **above any achievable
    C_Lmax**. For the reference configuration here (W = 9810 N, S = 10 m^2,
    AR = 16, e = 0.78, C_D0 = 0.028) it asks for C_L = 1.81, which no
    unflapped wing delivers - and V_minpower = 34.5 m/s sits *below* the
    37.9 m/s stall speed. Flying the theoretical optimum would mean stalling.

    So the achievable condition is::

        C_L_allowed = C_Lmax / safety_margin**2
        C_L_loiter  = min(C_L_minpower, C_L_allowed)
        V_loiter    = sqrt(2W / (rho * S * C_L_loiter))

    ``safety_margin`` is a margin on *speed*, hence the square when converted to
    a C_L cap: flying at 1.2*V_stall corresponds to C_L = C_Lmax/1.44.

    Args:
        weight_n .. oswald_e: as :func:`speed_best_ld`, all strictly positive.
        cl_max: Maximum lift coefficient [-], > 0.
        safety_margin: Speed margin over stall [-], >= 1. Default 1.2, the
            conventional loiter margin.

    Returns:
        ``(v_loiter_mps, active_constraint)`` where ``active_constraint`` is
        ``"min_power"`` when the aerodynamic optimum is achievable, or
        ``"stall_margin"`` when the stall cap binds. The simulator needs to know
        which, and so does the report: a stall-margin-limited loiter means the
        design is wing-loading limited, not drag limited.

    Raises:
        ValueError: If any argument is out of range, or safety_margin < 1.

    Example:
        The reference configuration is stall-margin limited, not at its
        aerodynamic optimum.

        >>> v, active = loiter_speed(
        ...     9810.0, 0.9091, 10.0, 0.028, 16.0, 0.78, cl_max=1.5)
        >>> round(v, 2), active
        (45.52, 'stall_margin')
    """
    w = _require_positive("weight_n", weight_n)
    r = _require_positive("rho", rho)
    s = _require_positive("wing_area_m2", wing_area_m2)
    cd_0 = _require_positive("cd0", cd0)
    ar = _require_positive("aspect_ratio", aspect_ratio)
    e = _require_positive("oswald_e", oswald_e)
    clm = _require_positive("cl_max", cl_max)
    margin = _require_positive("safety_margin", safety_margin)
    if not bool(np.all(margin >= 1.0)):
        raise ValueError(
            f"safety_margin must be >= 1 (it is a multiplier on stall speed), "
            f"got {float(np.min(margin))!r}."
        )

    # Minimum power condition: C_Di = 3 * C_D0
    cl_minpower = np.sqrt(3.0 * cd_0 * np.pi * ar * e)
    # Stall margin expressed as a C_L cap: flying at m*V_stall => C_Lmax/m**2
    cl_allowed = clm / margin**2

    cl_loiter = np.minimum(cl_minpower, cl_allowed)
    v = np.sqrt(2.0 * w / (r * s * cl_loiter))
    is_min_power = cl_minpower <= cl_allowed
    constraint = np.where(is_min_power, "min_power", "stall_margin")

    all_inputs = (
        weight_n, rho, wing_area_m2, cd0, aspect_ratio, oswald_e,
        cl_max, safety_margin,
    )
    if all(np.ndim(x) == 0 for x in all_inputs):
        return float(v), str(constraint)
    return v, constraint


# ---------------------------------------------------------------------------
# Oswald efficiency correlations
# ---------------------------------------------------------------------------


def oswald_efficiency(
    aspect_ratio: FloatOrArray,
    sweep_rad: FloatOrArray = 0.0,
    method: str = "raymer_straight",
    constant_value: float | None = None,
) -> FloatOrArray:
    """Estimate the Oswald span efficiency factor e.

    Methods:

    ``"raymer_straight"`` (default) - Raymer's straight-wing correlation::

        e = 1.78 * (1 - 0.045*AR**0.68) - 0.64

    ``"raymer_swept"`` - Raymer's swept-wing correlation, fitted for leading-edge
    sweep beyond 30 deg::

        e = 4.61 * (1 - 0.045*AR**0.68) * cos(sweep_LE)**0.15 - 3.1

    ``"constant"`` - returns ``constant_value`` unchanged, for treating e as an
    explicit, documented design assumption.

    CAVEAT - READ BEFORE USING THIS IN THE GA. Both Raymer correlations are
    fitted to general-aviation aircraft and become badly pessimistic at high
    aspect ratio. The straight-wing form gives e ~ 0.76 at AR 10, ~0.61 at AR 16
    and ~0.53 at AR 20, whereas real sailplanes at AR 20+ achieve e ~ 0.8. Used
    unmodified as a GA fitness input this **artificially suppresses high aspect
    ratio**: the optimiser sees induced drag rising with AR faster than it
    really does and converges on a stubbier wing than the physics warrants. That
    is a modelling artefact, not a design finding, and it must not be reported as
    one. Above AR = 12 this function emits :class:`HighAspectRatioWarning`; if
    you intend to explore high AR, use ``method="constant"`` with a justified
    value and state it as an assumption.

    No default is applied silently: ``method`` is always explicit in the
    signature, and ``"constant"`` requires ``constant_value``.

    Args:
        aspect_ratio: Wing aspect ratio [-], > 0.
        sweep_rad: Leading-edge sweep [rad]. Used only by ``"raymer_swept"``.
        method: One of ``"raymer_straight"``, ``"raymer_swept"``, ``"constant"``.
        constant_value: Required (and only used) when ``method="constant"``.

    Returns:
        Oswald efficiency [-], guaranteed > 0.

    Raises:
        ValueError: If aspect_ratio is non-positive, if ``method`` is unknown,
            if ``constant_value`` is missing or non-positive for the constant
            method, or if a correlation produces e <= 0 (which the swept form
            does above AR ~ 19 at zero sweep). A non-positive e would make the
            induced drag factor negative, i.e. induced drag would become thrust.

    Warns:
        HighAspectRatioWarning: If a Raymer correlation is used above AR = 12.

    Example:
        >>> round(oswald_efficiency(10.0), 4)
        0.7566
        >>> round(oswald_efficiency(8.0, method="constant", constant_value=0.85), 2)
        0.85
    """
    if method == "constant":
        if constant_value is None:
            raise ValueError(
                "method='constant' requires constant_value. It exists so e can "
                "be a documented design assumption rather than a correlation "
                "output - state the value and its justification."
            )
        e_const = _require_positive("constant_value", constant_value)
        return _restore(e_const, constant_value)

    ar = _require_positive("aspect_ratio", aspect_ratio)

    if method not in ("raymer_straight", "raymer_swept"):
        raise ValueError(
            f"Unknown method {method!r}. Expected one of 'raymer_straight', "
            f"'raymer_swept', 'constant'."
        )

    # Both correlations extrapolate badly past their GA-aircraft calibration
    # range; warn rather than silently suppressing high aspect ratio in the GA.
    if bool(np.any(ar > RAYMER_AR_LIMIT)):
        warnings.warn(
            f"Raymer Oswald correlation used at AR > {RAYMER_AR_LIMIT:g} "
            f"(max requested {float(np.max(ar)):g}). The correlation is fitted "
            f"to general-aviation aircraft and is pessimistic here: it gives "
            f"e ~ 0.61 at AR 16 where real high-AR wings achieve ~0.8. Inside a "
            f"genetic algorithm this artificially penalises high aspect ratio. "
            f"Consider method='constant' with a justified value.",
            HighAspectRatioWarning,
            stacklevel=2,
        )

    if method == "raymer_straight":
        # Raymer straight-wing Oswald correlation
        e = 1.78 * (1.0 - 0.045 * np.power(ar, 0.68)) - 0.64
        inputs: tuple[FloatOrArray, ...] = (aspect_ratio,)
    else:
        sweep = _require_finite("sweep_rad", sweep_rad)
        if bool(np.any(np.abs(sweep) < RAYMER_SWEEP_LIMIT_RAD)):
            warnings.warn(
                f"Raymer swept-wing correlation used below "
                f"{np.rad2deg(RAYMER_SWEEP_LIMIT_RAD):g} deg leading-edge "
                f"sweep, outside its fitted range; use method='raymer_straight' "
                f"for near-straight wings.",
                HighAspectRatioWarning,
                stacklevel=2,
            )
        # Raymer swept-wing Oswald correlation (fitted for sweep > 30 deg)
        e = (
            4.61
            * (1.0 - 0.045 * np.power(ar, 0.68))
            * np.power(np.cos(sweep), 0.15)
            - 3.1
        )
        inputs = (aspect_ratio, sweep_rad)

    if not bool(np.all(e > 0.0)):
        raise ValueError(
            f"The {method!r} correlation produced a non-positive Oswald "
            f"efficiency (min {float(np.min(e)):.4f}) at aspect ratio up to "
            f"{float(np.max(ar)):g}. This is the correlation breaking down "
            f"outside its fitted range, not a physical result; a non-positive e "
            f"would make induced drag negative. Use method='constant' with a "
            f"justified value at this aspect ratio."
        )

    return _restore(e, *inputs)


# ---------------------------------------------------------------------------
# Parasite drag buildup
# ---------------------------------------------------------------------------


def parasite_drag_from_wetted_area(
    s_wet_m2: FloatOrArray, s_ref_m2: FloatOrArray, c_fe: FloatOrArray = 0.0055
) -> FloatOrArray:
    """Parasite drag coefficient by the equivalent skin-friction method.

    Equation: ``C_D0 = C_fe * S_wet / S_ref``.

    WHY THIS EXISTS. C_D0 is referenced to wing area, so it is **not independent
    of S**. If wing area becomes a GA design variable and C_D0 is held fixed,
    the optimiser gets an unphysical free lunch: shrinking the wing cuts drag
    ``D = q*S*C_D0`` linearly while the parasite drag of the fuselage, tail and
    booms - which has not changed at all - silently shrinks with it. Routing
    C_D0 through this function ties it to actual wetted area, so shrinking S
    raises C_D0 and the total parasite drag stays put.

    It is deliberately **not wired in automatically**: doing so would require
    this module to know the vehicle's wetted-area breakdown, which is a
    configuration choice, not aerodynamics. The caller must opt in.

    ``c_fe = 0.0055`` is an ASSUMPTION, not a fact - Raymer's representative
    value for a clean light aircraft. It varies roughly 0.003 (a very clean
    composite sailplane) to 0.008 (a light aircraft with fixed gear and struts).
    Treat it as a design assumption to be stated and, ideally, swept.

    Args:
        s_wet_m2: Total wetted area [m^2], > 0.
        s_ref_m2: Reference (wing) area [m^2], > 0.
        c_fe: Equivalent skin friction coefficient [-], > 0. Default 0.0055.

    Returns:
        Zero-lift drag coefficient [-].

    Raises:
        ValueError: If any argument is non-positive or non-finite.

    Example:
        A wetted area 5.1x the wing area lands essentially on the C_D0 = 0.028
        used as the reference configuration throughout this project.

        >>> round(parasite_drag_from_wetted_area(51.0, 10.0), 5)
        0.02805
    """
    s_wet = _require_positive("s_wet_m2", s_wet_m2)
    s_ref = _require_positive("s_ref_m2", s_ref_m2)
    cfe = _require_positive("c_fe", c_fe)
    # Equivalent skin friction method: C_D0 = C_fe * S_wet / S_ref
    return _restore(cfe * s_wet / s_ref, s_wet_m2, s_ref_m2, c_fe)


# ---------------------------------------------------------------------------
# Climb performance
# ---------------------------------------------------------------------------


def rate_of_climb(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    v_mps: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
    eta_prop: FloatOrArray,
    shaft_power_available_w: FloatOrArray,
) -> FloatOrArray:
    """Rate of climb available from excess power.

    Equation: ``ROC = (P_shaft_available * eta_prop - D*V) / W`` - the specific
    excess power. The propeller converts shaft power to thrust power, drag power
    is spent staying aloft, and whatever remains buys potential energy.

    ASSUMPTION: small climb angle, consistent with the rest of this module, so
    drag is evaluated at the level-flight C_L. May return a negative value,
    which is the sink rate when available power cannot sustain level flight -
    that is meaningful, so it is not clamped.

    Args:
        weight_n .. oswald_e: as :func:`drag_force`, all strictly positive.
        eta_prop: Propeller efficiency [-], in (0, 1].
        shaft_power_available_w: Shaft power available [W], > 0.

    Returns:
        Rate of climb [m/s]; positive up, negative if power is insufficient.

    Raises:
        ValueError: If any argument is out of range.

    Example:
        100 kW of shaft power at the cruise point gives ~3.0 m/s of climb:
        80 kW reaches the air, 50.4 kW is spent on drag, and the remaining
        29.6 kW lifts 9810 N.

        >>> roc = rate_of_climb(
        ...     9810.0, 0.9091, 69.44, 10.0, 0.028, 16.0, 0.78, 0.8, 100000.0)
        >>> round(roc, 2)
        3.02
    """
    inputs = (weight_n, rho, v_mps, wing_area_m2, cd0, aspect_ratio, oswald_e)
    _, _, _, d, v = _polar_one_pass(*inputs)
    w = np.asarray(weight_n, dtype=np.float64)
    eta = _require_efficiency("eta_prop", eta_prop)
    p_avail = _require_positive("shaft_power_available_w", shaft_power_available_w)
    # Rate of climb from excess power: ROC = (P_shaft*eta_prop - D*V) / W
    roc = (p_avail * eta - d * v) / w
    return _restore(roc, *inputs, eta_prop, shaft_power_available_w)


# ---------------------------------------------------------------------------
# Bundled state - the hot path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AeroState:
    """Complete aerodynamic state at one flight condition.

    Frozen, so a state can be handed to the energy-management controller with no
    risk of mutation.

    Note on typing: fields are annotated with the union types rather than plain
    ``float``/``bool`` because :func:`evaluate` is vectorized - array input
    returns a state whose fields are arrays of the broadcast shape.

    Attributes:
        velocity_mps: True airspeed [m/s].
        dynamic_pressure_pa: q [Pa].
        lift_coefficient: C_L [-].
        drag_coefficient: C_D [-].
        drag_n: Total drag [N].
        lift_to_drag: L/D [-].
        thrust_power_w: Thrust power required [W], unclamped (may be negative).
        shaft_power_w: Shaft power required [W], clamped at 0.
        power_off: True where the aircraft is in an idle/windmilling descent.
    """

    velocity_mps: FloatOrArray
    dynamic_pressure_pa: FloatOrArray
    lift_coefficient: FloatOrArray
    drag_coefficient: FloatOrArray
    drag_n: FloatOrArray
    lift_to_drag: FloatOrArray
    thrust_power_w: FloatOrArray
    shaft_power_w: FloatOrArray
    power_off: BoolOrArray


def evaluate(
    weight_n: FloatOrArray,
    rho: FloatOrArray,
    v_mps: FloatOrArray,
    wing_area_m2: FloatOrArray,
    cd0: FloatOrArray,
    aspect_ratio: FloatOrArray,
    oswald_e: FloatOrArray,
    eta_prop: FloatOrArray,
    climb_rate_mps: FloatOrArray = 0.0,
) -> AeroState:
    """Full aerodynamic state in one pass. The simulator's hot path.

    Computes q, C_L, C_D, D, L/D and both powers with a single validation pass
    and a single evaluation of the polar. Prefer this over calling the
    individual functions: inside the GA's mission simulator this runs 1e6-1e7
    times, and the separate functions would re-validate and recompute q and C_L
    for every quantity.

    As with the atmosphere module, the vectorized path is dramatically cheaper
    per point than the scalar path - call it once with an array of flight
    conditions where the mission segment allows, rather than per timestep.

    Args:
        weight_n: Weight [N] - NEWTONS. rho: Density [kg/m^3].
        v_mps: TAS [m/s]. wing_area_m2: Wing area [m^2].
        cd0: Parasite drag coefficient [-]. aspect_ratio: Aspect ratio [-].
        oswald_e: Oswald efficiency [-]. All strictly positive.
        eta_prop: Propeller efficiency [-], in (0, 1].
        climb_rate_mps: Rate of climb [m/s], positive up. May be negative.

    Returns:
        :class:`AeroState`. Scalar input gives scalar fields; array input gives
        arrays of the broadcast shape.

    Raises:
        ValueError: If any argument is out of range.

    Example:
        The cruise point: 250 km/h at 6 km on the reference configuration.

        >>> st = evaluate(9810.0, 0.9091, 69.44, 10.0, 0.028, 16.0, 0.78, 0.8)
        >>> round(st.lift_coefficient, 4), round(st.drag_n, 1)
        (0.4476, 725.7)
        >>> round(st.lift_to_drag, 2), round(st.shaft_power_w / 1000, 1)
        (13.52, 63.0)
    """
    inputs = (weight_n, rho, v_mps, wing_area_m2, cd0, aspect_ratio, oswald_e)
    q, cl, cd, d, v = _polar_one_pass(*inputs)
    w = np.asarray(weight_n, dtype=np.float64)
    eta = _require_efficiency("eta_prop", eta_prop)
    roc = _require_finite("climb_rate_mps", climb_rate_mps)

    # Lift-to-drag ratio (equals W/D in level flight)
    ld = cl / cd
    # Thrust power = drag power + rate of potential energy gain
    p_thrust = d * v + w * roc
    # Power-off (windmilling / idle descent): no regeneration is possible.
    power_off = p_thrust <= 0.0
    # Shaft power at the propeller, clamped at zero
    p_shaft = np.where(power_off, 0.0, p_thrust / eta)

    all_inputs = (*inputs, eta_prop, climb_rate_mps)
    scalar = all(np.ndim(x) == 0 for x in all_inputs)
    if scalar:
        return AeroState(
            velocity_mps=float(v),
            dynamic_pressure_pa=float(q),
            lift_coefficient=float(cl),
            drag_coefficient=float(cd),
            drag_n=float(d),
            lift_to_drag=float(ld),
            thrust_power_w=float(p_thrust),
            shaft_power_w=float(p_shaft),
            power_off=bool(power_off),
        )
    # Broadcast every field to the common shape so array consumers can index
    # them together without worrying about which inputs were scalar.
    shape = np.broadcast_shapes(*(np.shape(x) for x in all_inputs))
    return AeroState(
        velocity_mps=np.broadcast_to(v, shape).copy(),
        dynamic_pressure_pa=np.broadcast_to(q, shape).copy(),
        lift_coefficient=np.broadcast_to(cl, shape).copy(),
        drag_coefficient=np.broadcast_to(cd, shape).copy(),
        drag_n=np.broadcast_to(d, shape).copy(),
        lift_to_drag=np.broadcast_to(ld, shape).copy(),
        thrust_power_w=np.broadcast_to(p_thrust, shape).copy(),
        shaft_power_w=np.broadcast_to(p_shaft, shape).copy(),
        power_off=np.broadcast_to(power_off, shape).copy(),
    )
