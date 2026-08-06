"""Turboshaft fuel consumption and altitude power lapse for a series hybrid UAV.

Fuel flow is affine in shaft power (Willans line), so specific fuel consumption
rises as power falls - the part-load penalty that gives load-levelling, and
therefore hybridization, any value at all. Power is kW, fuel flow kg/s and SFC
kg/kWh at the API boundary; the Willans arithmetic runs in kg/h internally.
Density ratio arrives as an argument, never an altitude. Rationale for every
default is in ``docs/assumptions.md`` (E-01..E-05, M-06); engine *mass* belongs
to ``mass.py`` (M-04) and is deliberately absent here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["EngineState", "Turboshaft"]

FloatOrArray = float | npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Fuel properties
# ---------------------------------------------------------------------------

LHV_KJ_KG = 43100.0
SECONDS_PER_HOUR = 3600.0

# ---------------------------------------------------------------------------
# Turboshaft calibration defaults
# ---------------------------------------------------------------------------

SFC_RATED_KG_KWH = 0.45
IDLE_FUEL_FRACTION = 0.20
LAPSE_EXPONENT = 0.8
MIN_POWER_FRACTION = 0.15
RESTART_FUEL_KG = 0.0


def _as_array(x: FloatOrArray) -> npt.NDArray[np.float64]:
    """Coerce to a float64 ndarray (possibly 0-d) without copying if able."""
    return np.asarray(x, dtype=np.float64)


def _restore_scalar(result: npt.NDArray[np.float64], original: FloatOrArray) -> FloatOrArray:
    """Return a Python float when the caller passed a scalar, else the ndarray."""
    return float(result) if np.ndim(original) == 0 else result


def _validate_sigma(sigma: float) -> None:
    """Raise ``ValueError`` unless the density ratio lies in (0, 1]."""
    if not 0.0 < sigma <= 1.0:
        raise ValueError(f"sigma must lie in (0, 1], got {sigma!r}")


@dataclass(frozen=True)
class EngineState:
    """What the engine actually did in response to one power command."""

    commanded_kw: float
    delivered_kw: float
    fuel_flow_kg_s: float
    sfc_kg_kwh: float
    thermal_efficiency: float
    load_fraction: float
    at_idle: bool
    shut_down: bool
    power_limited: bool


@dataclass(frozen=True)
class Turboshaft:
    """Willans-line turboshaft with a density-ratio power lapse.

    Args:
        rated_power_kw: Sea-level rated shaft power [kW].
        sfc_rated_kg_kwh: Specific fuel consumption at rated power [kg/kWh].
        idle_fuel_fraction: Idle fuel flow as a fraction of rated fuel flow [-].
        lapse_exponent: Exponent on density ratio in the power lapse [-].
        min_power_fraction: Lowest commandable power as a fraction of rated [-].
        allow_shutdown: Stop below the floor instead of idling - see O-04.
        restart_fuel_kg: Fuel charged for one restart [kg]; reported, never
            accumulated - see assumptions.md E-05.
    """

    rated_power_kw: float
    sfc_rated_kg_kwh: float = SFC_RATED_KG_KWH
    idle_fuel_fraction: float = IDLE_FUEL_FRACTION
    lapse_exponent: float = LAPSE_EXPONENT
    min_power_fraction: float = MIN_POWER_FRACTION
    allow_shutdown: bool = True
    restart_fuel_kg: float = RESTART_FUEL_KG

    def __post_init__(self) -> None:
        if not self.rated_power_kw > 0.0:
            raise ValueError(f"rated_power_kw must be positive, got {self.rated_power_kw!r}")
        if not self.sfc_rated_kg_kwh > 0.0:
            raise ValueError(f"sfc_rated_kg_kwh must be positive, got {self.sfc_rated_kg_kwh!r}")
        if not 0.0 <= self.idle_fuel_fraction < 1.0:
            raise ValueError(
                f"idle_fuel_fraction must lie in [0, 1), got {self.idle_fuel_fraction!r}"
            )

    # -- Willans calibration ------------------------------------------------

    @property
    def rated_fuel_flow_kg_h(self) -> float:
        """Fuel flow at rated power [kg/h]."""
        return self.sfc_rated_kg_kwh * self.rated_power_kw

    @property
    def willans_b(self) -> float:
        """Parasitic fuel flow, the Willans intercept [kg/h]."""
        return self.idle_fuel_fraction * self.rated_fuel_flow_kg_h

    @property
    def willans_a(self) -> float:
        """Marginal fuel cost, the Willans slope [kg/kWh]."""
        return (self.rated_fuel_flow_kg_h - self.willans_b) / self.rated_power_kw

    @property
    def marginal_efficiency(self) -> float:
        """Indicated thermal efficiency implied by the Willans slope [-]."""
        return SECONDS_PER_HOUR / (self.willans_a * LHV_KJ_KG)

    @property
    def idle_power_kw(self) -> float:
        """Lowest sea-level shaft power the engine can hold [kW]."""
        return self.min_power_fraction * self.rated_power_kw

    # -- Altitude -----------------------------------------------------------

    def max_power_kw(self, sigma: float) -> float:
        """Maximum available shaft power [kW] at a given density ratio."""
        _validate_sigma(sigma)
        # Turbine power lapse with density ratio.
        return self.rated_power_kw * sigma**self.lapse_exponent

    # -- Fuel, vectorized over power ----------------------------------------

    def _fuel_flow_kg_h(self, power: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Willans line, kg/h from kW. Internal; no floors applied."""
        return self.willans_a * power + self.willans_b

    def fuel_flow_kg_s(self, power_kw: FloatOrArray, sigma: float = 1.0) -> FloatOrArray:
        """Raw Willans fuel flow [kg/s]; no idle floor and no power limit applied.

        Args:
            power_kw: Shaft power [kW]. Scalar or ndarray.
            sigma: Density ratio [-]. Validated, but fuel flow at a given
                absolute power is altitude-independent - see assumptions.md E-04.
        """
        _validate_sigma(sigma)
        power = _as_array(power_kw)
        return _restore_scalar(self._fuel_flow_kg_h(power) / SECONDS_PER_HOUR, power_kw)

    def sfc_kg_kwh(self, power_kw: FloatOrArray) -> FloatOrArray:
        """Specific fuel consumption [kg/kWh]; ``inf`` at zero power."""
        power = _as_array(power_kw)
        with np.errstate(divide="ignore", invalid="ignore"):
            # SFC(P) = a + b/P, the Willans line divided through by power.
            sfc = np.where(power != 0.0, self.willans_a + self.willans_b / power, np.inf)
        return _restore_scalar(sfc, power_kw)

    def thermal_efficiency(self, power_kw: FloatOrArray) -> FloatOrArray:
        """Instantaneous thermal efficiency, shaft power over chemical power [-]."""
        power = _as_array(power_kw)
        chemical_kw = self._fuel_flow_kg_h(power) / SECONDS_PER_HOUR * LHV_KJ_KG
        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency = np.where(chemical_kw > 0.0, power / chemical_kw, 0.0)
        return _restore_scalar(efficiency, power_kw)

    # -- Operating point ----------------------------------------------------

    def operate(self, commanded_kw: float, sigma: float = 1.0) -> EngineState:
        """Resolve a power command into an achievable operating point.

        Applies the altitude power limit, then the low-power floor, and reports
        which of them bound. Never raises on an unachievable command.
        """
        available_kw = self.max_power_kw(sigma)

        power_limited = commanded_kw > available_kw
        delivered_kw = min(float(commanded_kw), available_kw)

        # Below the floor the engine either idles or stops - see O-04.
        at_idle = False
        shut_down = False
        if delivered_kw < self.idle_power_kw:
            if self.allow_shutdown:
                delivered_kw = 0.0
                shut_down = True
            else:
                delivered_kw = min(self.idle_power_kw, available_kw)
                at_idle = True

        return EngineState(
            commanded_kw=float(commanded_kw),
            delivered_kw=delivered_kw,
            fuel_flow_kg_s=0.0 if shut_down else float(self.fuel_flow_kg_s(delivered_kw)),
            sfc_kg_kwh=float(self.sfc_kg_kwh(delivered_kw)),
            thermal_efficiency=float(self.thermal_efficiency(delivered_kw)),
            load_fraction=delivered_kw / available_kw,
            at_idle=at_idle,
            shut_down=shut_down,
            power_limited=power_limited,
        )
