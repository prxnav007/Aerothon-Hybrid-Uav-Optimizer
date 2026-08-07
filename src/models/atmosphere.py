"""
International Standard Atmosphere model for altitudes from 0 to 20 km.

This module calculates atmospheric properties such as:

- temperature
- pressure
- air density
- speed of sound
- dynamic viscosity

These values are used by the UAV simulator for aerodynamic, engine, Reynolds
number and Mach number calculations.

The model follows the International Standard Atmosphere and contains two layers:

1. Troposphere: 0 to 11 km
   Temperature decreases by 6.5 K for every kilometre of altitude.

2. Lower stratosphere: 11 to 20 km
   Temperature remains constant at 216.65 K.

The altitude input `h_m` must be geopotential altitude in metres. If geometric
altitude is available, convert it first using
`geometric_to_geopotential()`.

This model does not include weather, wind, humidity, non-standard temperatures
or atmospheric layers above 20 km.

All functions support either a single altitude or a NumPy array of altitudes.
A single input returns a float, while an array input returns an array with the
same shape.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "AtmosphericState",
    "atmosphere",
    "density",
    "density_ratio",
    "dynamic_viscosity",
    "geometric_to_geopotential",
    "geopotential_to_geometric",
    "mach_number",
    "pressure",
    "speed_of_sound",
    "temperature",
]

# Every public function accepts and returns either of these.
## This only accepts a List of Float numbers or an Array of Floats(Numpy array).
FloatOrArray = float | npt.NDArray[np.float64]
# This is the datatype indicator.

# ---------------------------------------------------------------------------
# Primary ISA constants (definitions - these are fixed by the standard)
# ---------------------------------------------------------------------------

g0: float = 9.80665
"""Standard acceleration of free fall [m/s^2]. PRIMARY (ISO 2533 g_n)."""

R_STAR: float = 8.31432
# Universal Gas Constant [J/(mol*k)]
M_AIR: float = 0.02896442
# Molar mass of dry air [kg/mol]

GAMMA: float = 1.4
"""Ratio of specific heats of air, cp/cv [-]. PRIMARY (ISO 2533 kappa)."""

T0: float = 288.15
"""Sea-level standard temperature [K] (15 degC). PRIMARY."""

P0: float = 101325.0
"""Sea-level standard pressure [Pa] (1 atm). PRIMARY."""

L_TROPO: float = 0.0065
"""Troposphere temperature lapse rate [K/m], stored as a POSITIVE decay rate."""

H_TROPOPAUSE: float = 11000.0
"""Geopotential altitude of the tropopause [m]. PRIMARY."""

R_EARTH: float = 6356766.0
"""Effective Earth radius used by ISA for the geometric/geopotential map [m]."""

SUTHERLAND_BETA: float = 1.458e-6
"""Sutherland's law coefficient for air [kg/(m*s*K^0.5)]. PRIMARY (ISO 2533 beta_s)."""

SUTHERLAND_S: float = 110.4
"""Sutherland's constant for air [K]. PRIMARY (ISO 2533 S)."""

H_MIN: float = 0.0
"""Lower geopotential-altitude validity limit of this module [m]."""

H_MAX: float = 20000.0
"""Upper geopotential-altitude validity limit of this module [m]."""

# ---------------------------------------------------------------------------
# Derived ISA constants
# ---------------------------------------------------------------------------

R_AIR: float = R_STAR / M_AIR
"""Specific gas constant of dry air [J/(kg*K)]. DERIVED as R* / M."""

RHO0: float = P0 / (R_AIR * T0)
"""Sea-level standard density [kg/m^3]. DERIVED from the ideal gas law."""

BARO_EXPONENT: float = g0 / (L_TROPO * R_AIR)
"""Exponent of the linear-lapse barometric formula [-]. DERIVED = g0/(L*R)."""

T_TROPOPAUSE: float = T0 - L_TROPO * H_TROPOPAUSE
"""Temperature at the tropopause [K]. DERIVED = 216.65 (-56.5 degC)."""

P_TROPOPAUSE: float = P0 * (T_TROPOPAUSE / T0) ** BARO_EXPONENT
"""Pressure at the tropopause [Pa]. DERIVED = 22632.04 (table value: 22632.1)."""

RHO_TROPOPAUSE: float = P_TROPOPAUSE / (R_AIR * T_TROPOPAUSE)
"""Density at the tropopause [kg/m^3]. DERIVED = 0.3639178 (table: 0.363918)."""

SCALE_HEIGHT_STRATO: float = R_AIR * T_TROPOPAUSE / g0
"""Pressure scale height of the isothermal layer [m]. DERIVED = 6341.62."""


# ---------------------------------------------------------------------------
# Import-time verification of the derived constants
# ---------------------------------------------------------------------------


def _check_constant(name: str, computed: float, reference: float, rel_tol: float) -> None:
    error = abs(computed - reference) / abs(reference)
    if not error < rel_tol:
        raise AssertionError(
            f"ISA constant {name} does not match its published value: "
            f"derived {computed!r}, standard {reference!r} "
            f"(relative error {error:.3e}, tolerance {rel_tol:.1e}). "
            "A primary constant is wrong - fix the primary constant, do NOT "
            "loosen this tolerance."
        )
_check_constant("R_AIR = R*/M", R_AIR, 287.05287, 1e-7)
_check_constant("RHO0 = P0/(R*T0)", RHO0, 1.225, 1e-8)
_check_constant("BARO_EXPONENT = g0/(L*R)", BARO_EXPONENT, 5.2559, 1e-4)
_check_constant("T_TROPOPAUSE = T0 - L*11000", T_TROPOPAUSE, 216.65, 1e-12)
_check_constant("P_TROPOPAUSE", P_TROPOPAUSE, 22632.1, 1e-4)
_check_constant("RHO_TROPOPAUSE", RHO_TROPOPAUSE, 0.363918, 1e-4)
_check_constant("SCALE_HEIGHT_STRATO = R*T/g0", SCALE_HEIGHT_STRATO, 6341.6, 1e-4)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _as_array(x: FloatOrArray) -> npt.NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def _restore_scalar(result: npt.NDArray[np.float64], original: FloatOrArray) -> FloatOrArray:
    return float(result) if np.ndim(original) == 0 else result


def _validate_altitude(h: npt.NDArray[np.float64]) -> None:
    in_range = np.isfinite(h) & (h >= H_MIN) & (h <= H_MAX)
    if bool(np.all(in_range)):
        return

    offenders = np.atleast_1d(h)[~np.atleast_1d(in_range)]
    shown = ", ".join(f"{float(v)!r}" for v in offenders[:5])
    if offenders.size > 5:
        shown += f", ... ({offenders.size} offending values in total)"
    raise ValueError(
        f"Geopotential altitude out of range for this ISA model: valid range is "
        f"{H_MIN:.0f} m <= h <= {H_MAX:.0f} m (offending value(s): {shown}). "
        "Altitudes are deliberately not clamped; fix the caller's bounds."
    )


def _checked(h_m: FloatOrArray) -> npt.NDArray[np.float64]:
    """Validate and coerce an altitude argument in one step."""
    h = _as_array(h_m)
    _validate_altitude(h)
    return h


# ---------------------------------------------------------------------------
# Altitude conversions
# ---------------------------------------------------------------------------


def geometric_to_geopotential(z_m: FloatOrArray) -> FloatOrArray:
    z = _as_array(z_m)
    # Geometric -> geopotential (ISO 2533). h < z always, by ~0.16 % at 10 km.
    h = (R_EARTH * z) / (R_EARTH + z)
    return _restore_scalar(h, z_m)


def geopotential_to_geometric(h_m: FloatOrArray) -> FloatOrArray:
    """Convert geopotential altitude to geometric (true) altitude.

    Equation: ``z = R_EARTH * h / (R_EARTH - h)``. Exact inverse of
    :func:`geometric_to_geopotential`.

    Args:
        h_m: Geopotential altitude [m]. Scalar or ndarray. Not range-checked,
            for the same reason as the forward conversion. Singular at
            ``h = R_EARTH`` (6 356 766 m), which is ~318x this module's ceiling
            and so unreachable by any caller of the atmosphere functions.

    Returns:
        Geometric altitude [m]; float for scalar input, ndarray otherwise.

    Example:
        >>> round(geopotential_to_geometric(9984.293), 1)
        10000.0
    """
    h = _as_array(h_m)
    # Geopotential -> geometric (ISO 2533).
    z = (R_EARTH * h) / (R_EARTH - h)
    return _restore_scalar(z, h_m)


# ---------------------------------------------------------------------------
# Core ISA state - single pass over both layers
# ---------------------------------------------------------------------------


def _temperature_pressure_density(
    h: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute T, p, rho for an already-validated altitude array. Internal.

    This is the one place where the layer branch lives; everything public is a
    thin wrapper. Returns (T [K], p [Pa], rho [kg/m^3]) as ndarrays.
    """
    in_troposphere = h < H_TROPOPAUSE

    # -- Layer 1: troposphere ------------------------------------------------
    # Linear temperature lapse law (ISA troposphere).
    t_tropo = T0 - L_TROPO * h
    # Barometric formula, linear-lapse layer form (power law).
    p_tropo = P0 * np.power(t_tropo / T0, BARO_EXPONENT)

    # -- Layer 2: lower stratosphere ----------------------------------------
    # Barometric formula, isothermal layer form (EXPONENTIAL, not a power law).
    #
    # WHY the two layers use structurally different formulas: integrating
    # hydrostatic balance dp/dh = -rho*g0 together with the ideal gas law gives
    # the power law only while the lapse rate L is non-zero - its exponent is
    # g0/(L*R), which is singular as L -> 0. Above the tropopause L IS exactly
    # zero, so the integral collapses to a pure exponential with scale height
    # R*T/g0. Reusing the troposphere power law above 11 km is the single most
    # common bug in ISA implementations; it silently under-predicts stratospheric
    # density and would flatter this UAV's high-altitude cruise performance.
    decay = np.exp(-(h - H_TROPOPAUSE) / SCALE_HEIGHT_STRATO)
    p_strato = P_TROPOPAUSE * decay

    # Both branches are evaluated everywhere and then selected. That is safe
    # here because neither is singular anywhere in [0, H_MAX]: the power-law
    # base T0 - L*h stays positive all the way to 20 km (158.15 K), and the
    # exponential's argument stays small and bounded. The range guard in
    # `_validate_altitude` is what makes that guarantee hold, which is why no
    # `np.errstate` juggling is needed in this hot path.
    temperature_k = np.where(in_troposphere, t_tropo, T_TROPOPAUSE)
    pressure_pa = np.where(in_troposphere, p_tropo, p_strato)

    # Ideal gas law, p = rho * R * T. Used for BOTH layers in preference to the
    # equivalent direct forms (RHO0*(T/T0)**(n-1) and RHO_TROPOPAUSE*exp(...)),
    # so density can never drift out of thermodynamic consistency with pressure.
    # The tests check this closed form against both direct forms.
    density_kg_m3 = pressure_pa / (R_AIR * temperature_k)

    return temperature_k, pressure_pa, density_kg_m3


def _temperature_only(h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Temperature for an already-validated altitude array. Internal.

    Split out because a, mu and T themselves depend on temperature alone, and
    computing pressure for them would be wasted work.
    """
    # Linear temperature lapse law below the tropopause, isothermal above it.
    return np.where(h < H_TROPOPAUSE, T0 - L_TROPO * h, T_TROPOPAUSE)


# ---------------------------------------------------------------------------
# Public scalar/array property functions
# ---------------------------------------------------------------------------


def temperature(h_m: FloatOrArray) -> FloatOrArray:
    """ISA static air temperature.

    Equation: ``T = T0 - L_TROPO * h`` below 11 km (linear lapse law);
    ``T = T_TROPOPAUSE = 216.65 K`` from 11 km to 20 km (isothermal).

    Args:
        h_m: Geopotential altitude [m], 0 to 20 000.

    Returns:
        Temperature [K]; float for scalar input, ndarray otherwise.

    Raises:
        ValueError: If any altitude is outside [0, 20 000] m or is not finite.

    Example:
        At the UAV's 6 km cruise altitude, T = 249.15 K (-24 degC).

        >>> round(temperature(6000.0), 2)
        249.15
    """
    h = _checked(h_m)
    return _restore_scalar(_temperature_only(h), h_m)


def pressure(h_m: FloatOrArray) -> FloatOrArray:
    """ISA static air pressure.

    Equation: ``p = P0 * (T/T0)**(g0/(L*R))`` below 11 km (barometric formula,
    linear-lapse form); ``p = P_TROPOPAUSE * exp(-(h - 11000)/H_s)`` from 11 km
    to 20 km (barometric formula, isothermal form), with
    ``H_s = R*T_TROPOPAUSE/g0``.

    Args:
        h_m: Geopotential altitude [m], 0 to 20 000.

    Returns:
        Pressure [Pa]; float for scalar input, ndarray otherwise.

    Raises:
        ValueError: If any altitude is outside [0, 20 000] m or is not finite.

    Example:
        Reference ISA value at 6 km is 47 181 Pa.

        >>> round(pressure(6000.0), 1)
        47181.0
    """
    h = _checked(h_m)
    _, p, _ = _temperature_pressure_density(h)
    return _restore_scalar(p, h_m)


def density(h_m: FloatOrArray) -> FloatOrArray:
    """ISA air density.

    Equation: ``rho = p / (R_AIR * T)`` (ideal gas law), with p and T from the
    layer-appropriate formulas. Equivalent closed forms, verified in the tests:
    ``rho = RHO0 * (T/T0)**(g0/(L*R) - 1)`` in the troposphere and
    ``rho = RHO_TROPOPAUSE * exp(-(h - 11000)/H_s)`` in the stratosphere.

    Args:
        h_m: Geopotential altitude [m], 0 to 20 000.

    Returns:
        Density [kg/m^3]; float for scalar input, ndarray otherwise.

    Raises:
        ValueError: If any altitude is outside [0, 20 000] m or is not finite.

    Example:
        Reference ISA value at 6 km is 0.659697 kg/m^3.

        >>> round(density(6000.0), 6)
        0.659697
    """
    h = _checked(h_m)
    _, _, rho = _temperature_pressure_density(h)
    return _restore_scalar(rho, h_m)


def speed_of_sound(h_m: FloatOrArray) -> FloatOrArray:
    """Speed of sound in ISA air.

    Equation: ``a = sqrt(GAMMA * R_AIR * T)`` (ideal-gas acoustic speed). It
    depends on temperature alone, so it is constant at 295.07 m/s throughout the
    isothermal layer.

    Args:
        h_m: Geopotential altitude [m], 0 to 20 000.

    Returns:
        Speed of sound [m/s]; float for scalar input, ndarray otherwise.

    Raises:
        ValueError: If any altitude is outside [0, 20 000] m or is not finite.

    Example:
        Sea-level ISA speed of sound is 340.29 m/s.

        >>> round(speed_of_sound(0.0), 2)
        340.29
    """
    h = _checked(h_m)
    # Ideal-gas acoustic speed, a = sqrt(gamma * R * T).
    a = np.sqrt(GAMMA * R_AIR * _temperature_only(h))
    return _restore_scalar(a, h_m)


def density_ratio(h_m: FloatOrArray) -> FloatOrArray:
    """Density ratio sigma = rho / RHO0.

    Equation: ``sigma = rho(h) / RHO0``. This is the quantity the engine model
    consumes for power lapse (shaft power falls roughly with sigma) and the
    quantity that links true and equivalent airspeed
    (``V_EAS = V_TAS * sqrt(sigma)``).

    Args:
        h_m: Geopotential altitude [m], 0 to 20 000.

    Returns:
        Density ratio [-]: 1.0 at sea level, falling to ~0.0719 at 20 km;
        float for scalar input, ndarray otherwise.

    Raises:
        ValueError: If any altitude is outside [0, 20 000] m or is not finite.

    Example:
        At 6 km the air is a little over half sea-level density.

        >>> round(density_ratio(6000.0), 4)
        0.5385
    """
    h = _checked(h_m)
    _, _, rho = _temperature_pressure_density(h)
    # Density ratio sigma = rho / rho_sea_level.
    return _restore_scalar(rho / RHO0, h_m)


def dynamic_viscosity(h_m: FloatOrArray) -> FloatOrArray:
    """Dynamic (absolute) viscosity of air via Sutherland's law.

    Equation: ``mu = beta_s * T**1.5 / (T + S)`` with
    ``beta_s = 1.458e-6 kg/(m*s*K^0.5)`` and ``S = 110.4 K`` (ISO 2533). Needed
    downstream for the Reynolds number that sets the wing's skin-friction drag.

    Like the speed of sound this is a function of temperature only, so it is
    constant above the tropopause.

    Args:
        h_m: Geopotential altitude [m], 0 to 20 000.

    Returns:
        Dynamic viscosity [Pa*s]; float for scalar input, ndarray otherwise.

    Raises:
        ValueError: If any altitude is outside [0, 20 000] m or is not finite.

    Example:
        Sea-level ISA viscosity is 1.789e-5 Pa*s (PDAS table: 17.89e-6).

        >>> float(f"{dynamic_viscosity(0.0):.4g}")
        1.789e-05
    """
    h = _checked(h_m)
    t = _temperature_only(h)
    # Sutherland's law for dynamic viscosity of air.
    mu = SUTHERLAND_BETA * np.power(t, 1.5) / (t + SUTHERLAND_S)
    return _restore_scalar(mu, h_m)


def mach_number(v_mps: FloatOrArray, h_m: FloatOrArray) -> FloatOrArray:
    """Flight Mach number.

    Equation: ``M = V / a(h)``, with ``a = sqrt(GAMMA * R_AIR * T)``.

    Args:
        v_mps: True airspeed [m/s]. Scalar or ndarray, broadcast against `h_m`.
        h_m: Geopotential altitude [m], 0 to 20 000.

    Returns:
        Mach number [-]; float when both inputs are scalar, ndarray otherwise.

    Raises:
        ValueError: If any altitude is outside [0, 20 000] m or is not finite.

    Example:
        The UAV cruises at 250 km/h (69.44 m/s) at 6 km, i.e. M = 0.22 - well
        below 0.3, which is why incompressible aerodynamics is valid here.

        >>> round(mach_number(69.44, 6000.0), 3)
        0.219
    """
    h = _checked(h_m)
    v = _as_array(v_mps)
    # Mach number, M = V / a.
    mach = v / np.sqrt(GAMMA * R_AIR * _temperature_only(h))
    # Scalar out only if BOTH inputs were scalar, since v may broadcast h up.
    if np.ndim(v_mps) == 0 and np.ndim(h_m) == 0:
        return float(mach)
    return mach


# ---------------------------------------------------------------------------
# Bundled state - the main entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtmosphericState:
    """Complete ISA state at one altitude (or, elementwise, at an array of them).

    Frozen so a state can be passed around the mission simulator with no risk of
    a downstream model mutating it.

    Note on typing: the fields are annotated ``FloatOrArray`` rather than
    ``float`` because :func:`atmosphere` is vectorized - passing an ndarray of
    altitudes returns one state whose fields are ndarrays of matching shape.

    Attributes:
        altitude_m: Geopotential altitude [m].
        temperature_k: Static temperature [K].
        pressure_pa: Static pressure [Pa].
        density_kg_m3: Density [kg/m^3].
        speed_of_sound_mps: Speed of sound [m/s].
        density_ratio: rho / RHO0 [-].
        dynamic_viscosity_pa_s: Dynamic viscosity [Pa*s].
    """

    altitude_m: FloatOrArray
    temperature_k: FloatOrArray
    pressure_pa: FloatOrArray
    density_kg_m3: FloatOrArray
    speed_of_sound_mps: FloatOrArray
    density_ratio: FloatOrArray
    dynamic_viscosity_pa_s: FloatOrArray


def atmosphere(h_m: FloatOrArray) -> AtmosphericState:
    """Full ISA state at a geopotential altitude. Main entry point.

    Computes every field in one pass - temperature is evaluated once and reused
    for pressure, density, speed of sound and viscosity. Prefer this over
    calling the individual functions in sequence: inside the GA's mission
    simulator this runs 1e6-1e7 times, and the per-function versions would redo
    the layer branch and the temperature lapse each time.

    Performance note: measured at ~67 ns per altitude when called once with a
    1e6-element array, but ~30 us per call when called with a scalar - roughly
    450x worse per point, because NumPy's fixed per-operation overhead then
    dominates the arithmetic. At the GA's 1e6-1e7 evaluations that is the
    difference between under a second and several minutes. Callers in the
    time-marching simulator should therefore evaluate a whole altitude profile
    in one call wherever the mission segment allows it, rather than calling this
    per timestep with a float.

    Args:
        h_m: Geopotential altitude [m], 0 to 20 000. Convert from geometric
            altitude with :func:`geometric_to_geopotential` first.

    Returns:
        :class:`AtmosphericState`. Scalar input gives float fields; ndarray
        input gives ndarray fields of matching shape.

    Raises:
        ValueError: If any altitude is outside [0, 20 000] m or is not finite.

    Example:
        Sea level recovers the ISA definitions exactly.

        >>> sl = atmosphere(0.0)
        >>> round(sl.temperature_k, 2), round(sl.pressure_pa, 1)
        (288.15, 101325.0)
        >>> round(sl.density_kg_m3, 4), round(sl.speed_of_sound_mps, 2)
        (1.225, 340.29)
    """
    h = _checked(h_m)
    t, p, rho = _temperature_pressure_density(h)

    # Ideal-gas acoustic speed, a = sqrt(gamma * R * T).
    a = np.sqrt(GAMMA * R_AIR * t)
    # Sutherland's law for dynamic viscosity of air.
    mu = SUTHERLAND_BETA * np.power(t, 1.5) / (t + SUTHERLAND_S)
    # Density ratio sigma = rho / rho_sea_level, consumed by the engine model.
    sigma = rho / RHO0

    return AtmosphericState(
        altitude_m=_restore_scalar(h, h_m),
        temperature_k=_restore_scalar(t, h_m),
        pressure_pa=_restore_scalar(p, h_m),
        density_kg_m3=_restore_scalar(rho, h_m),
        speed_of_sound_mps=_restore_scalar(a, h_m),
        density_ratio=_restore_scalar(sigma, h_m),
        dynamic_viscosity_pa_s=_restore_scalar(mu, h_m),
    )
