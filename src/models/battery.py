"""Rint battery pack on the DC bus of a series hybrid UAV.

Positive power is discharge and negative is charge throughout. Current is
solved from the quadratic P = V_oc*I - I^2*R rather than P/V_oc, so ohmic loss
is the only efficiency term the model needs. State of charge is passed in and
returned, never held here, so the controller can price candidate power splits
without committing to them.

Available power is energy-limited as well as rate-limited: it is the power the
pack can sustain for the *whole* of a step, so it depends on the step length.
That is what keeps coulomb counting from overshooting the cutoff within a step
and handing the bus energy the pack does not have. Rationale for every default
is in ``docs/assumptions.md`` (B-03..B-06, O-05); pack *mass* belongs to
``mass.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["BatteryPack", "BatteryState", "SOC_EPS"]

FloatOrArray = float | npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

WATTS_PER_KW = 1000.0
SECONDS_PER_HOUR = 3600.0

# ---------------------------------------------------------------------------
# Rint calibration defaults
# ---------------------------------------------------------------------------

V_MIN_V = 300.0
V_MAX_V = 400.0
R_REF_OHM = 0.05
R_REF_CAPACITY_KWH = 10.0
SCALE_RESISTANCE = True

# ---------------------------------------------------------------------------
# Operating limits
# ---------------------------------------------------------------------------

DISCHARGE_C_RATE = 3.0
CHARGE_C_RATE = 1.0
SOC_MIN = 0.05
SOC_SAFE_MIN = 0.20

# Headroom below which a bound counts as reached. Limiting a step to the charge
# above the cutoff lands state of charge on the bound to within rounding, so an
# exact comparison would decide `at_cutoff` on floating-point dust and let a
# mission loop creep toward the floor in ever-smaller steps without arriving.
# Nine orders below anything physically meaningful, seven above the dust.
SOC_EPS = 1.0e-9


def _as_array(x: FloatOrArray) -> npt.NDArray[np.float64]:
    """Coerce to a float64 ndarray (possibly 0-d) without copying if able."""
    return np.asarray(x, dtype=np.float64)


def _restore_scalar(result: npt.NDArray[np.float64], *originals: FloatOrArray) -> FloatOrArray:
    """Return a Python float when every caller argument was a scalar."""
    if all(np.ndim(original) == 0 for original in originals):
        return float(result)
    return result


def _require_positive(**values: float) -> None:
    """Raise ``ValueError`` naming the first argument that is not > 0."""
    for name, value in values.items():
        if not value > 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True)
class BatteryState:
    """What the pack actually did over one integration step.

    The two limit flags separate failures with different remedies.
    ``rate_limited`` means the pack could not deliver that power for an instant,
    let alone a step - a sizing problem. ``energy_limited`` means it could have,
    but lacks the charge to hold it for the whole step - an energy-management
    problem, or a timestep-resolution one. Both can be true at once.
    """

    soc: float
    power_kw: float
    commanded_kw: float
    current_a: float
    open_circuit_voltage_v: float
    terminal_voltage_v: float
    ohmic_loss_kw: float
    at_cutoff: bool
    rate_limited: bool
    energy_limited: bool
    below_safe_floor: bool

    @property
    def power_limited(self) -> bool:
        """True when the delivered power fell short of the command, either cause."""
        return self.rate_limited or self.energy_limited


@dataclass(frozen=True)
class BatteryPack:
    """Rint lithium-ion pack sized by energy capacity.

    Args:
        capacity_kwh: Rated pack energy [kWh].
        v_min_v: Open-circuit voltage at zero state of charge [V].
        v_max_v: Open-circuit voltage at full state of charge [V].
        r_ref_ohm: Internal resistance quoted at ``r_ref_capacity_kwh`` [ohm].
        r_ref_capacity_kwh: Capacity that reference resistance belongs to [kWh].
        scale_resistance: Scale resistance inversely with capacity - see O-05.
        discharge_c_rate: Continuous discharge limit [1/h].
        charge_c_rate: Continuous charge limit [1/h].
        soc_min: Hard discharge cutoff [-].
        soc_safe_min: Recommended operating floor [-]; reported, never enforced.
    """

    capacity_kwh: float
    v_min_v: float = V_MIN_V
    v_max_v: float = V_MAX_V
    r_ref_ohm: float = R_REF_OHM
    r_ref_capacity_kwh: float = R_REF_CAPACITY_KWH
    scale_resistance: bool = SCALE_RESISTANCE
    discharge_c_rate: float = DISCHARGE_C_RATE
    charge_c_rate: float = CHARGE_C_RATE
    soc_min: float = SOC_MIN
    soc_safe_min: float = SOC_SAFE_MIN

    def __post_init__(self) -> None:
        _require_positive(
            capacity_kwh=self.capacity_kwh,
            v_min_v=self.v_min_v,
            r_ref_ohm=self.r_ref_ohm,
            r_ref_capacity_kwh=self.r_ref_capacity_kwh,
            discharge_c_rate=self.discharge_c_rate,
            charge_c_rate=self.charge_c_rate,
        )
        if not self.v_min_v < self.v_max_v:
            raise ValueError(
                f"v_min_v must be below v_max_v, got {self.v_min_v!r} and {self.v_max_v!r}"
            )
        if not 0.0 <= self.soc_min < 1.0:
            raise ValueError(f"soc_min must lie in [0, 1), got {self.soc_min!r}")

    # -- Pack constants -----------------------------------------------------

    @property
    def nominal_voltage_v(self) -> float:
        """Mean open-circuit voltage over a full state-of-charge sweep [V]."""
        return 0.5 * (self.v_min_v + self.v_max_v)

    @property
    def charge_capacity_ah(self) -> float:
        """Charge capacity [Ah]; recovers the rated energy at nominal voltage."""
        return self.capacity_kwh * WATTS_PER_KW / self.nominal_voltage_v

    @property
    def internal_resistance_ohm(self) -> float:
        """Pack internal resistance [ohm]."""
        if not self.scale_resistance:
            return self.r_ref_ohm
        # Capacity is added as parallel strings at fixed voltage - see O-05.
        return self.r_ref_ohm * self.r_ref_capacity_kwh / self.capacity_kwh

    @property
    def max_discharge_kw(self) -> float:
        """C-rate discharge limit [kW] - see assumptions.md B-04."""
        return self.discharge_c_rate * self.capacity_kwh

    @property
    def max_charge_kw(self) -> float:
        """C-rate charge limit [kW], as a positive magnitude."""
        return self.charge_c_rate * self.capacity_kwh

    # -- Raw evaluations, vectorized over power -----------------------------

    def open_circuit_voltage(self, soc: FloatOrArray) -> FloatOrArray:
        """Open-circuit voltage [V], linear in state of charge."""
        s = _as_array(soc)
        return _restore_scalar(self.v_min_v + (self.v_max_v - self.v_min_v) * s, soc)

    def ohmic_power_ceiling_kw(self, soc: FloatOrArray) -> FloatOrArray:
        """Greatest bus power [kW] the equivalent circuit can transfer."""
        v_oc = _as_array(self.open_circuit_voltage(soc))
        ceiling = v_oc * v_oc / (4.0 * self.internal_resistance_ohm) / WATTS_PER_KW
        return _restore_scalar(ceiling, soc)

    def current_from_power(self, power_kw: FloatOrArray, soc: FloatOrArray) -> FloatOrArray:
        """Pack current [A] for a commanded bus power; positive on discharge.

        No C-rate or state-of-charge limit is applied. A power beyond the ohmic
        ceiling is clamped to it so the result stays finite rather than NaN.
        """
        v_oc = _as_array(self.open_circuit_voltage(soc))
        r = self.internal_resistance_ohm
        power_w = _as_array(power_kw) * WATTS_PER_KW

        # R*I^2 - V_oc*I + P_bus = 0, lower root; signed correctly either side of
        # zero power, so charging falls out of the same expression.
        discriminant = np.maximum(v_oc * v_oc - 4.0 * r * power_w, 0.0)
        current = (v_oc - np.sqrt(discriminant)) / (2.0 * r)
        return _restore_scalar(current, power_kw, soc)

    def terminal_voltage(self, power_kw: FloatOrArray, soc: FloatOrArray) -> FloatOrArray:
        """Terminal voltage [V]; below OCV on discharge, above it on charge."""
        v_oc = _as_array(self.open_circuit_voltage(soc))
        current = _as_array(self.current_from_power(power_kw, soc))
        # Rint terminal voltage.
        return _restore_scalar(v_oc - current * self.internal_resistance_ohm, power_kw, soc)

    def ohmic_loss_kw(self, power_kw: FloatOrArray, soc: FloatOrArray) -> FloatOrArray:
        """Internal resistive loss [kW] at a commanded bus power; no limits applied."""
        current = _as_array(self.current_from_power(power_kw, soc))
        loss = current * current * self.internal_resistance_ohm / WATTS_PER_KW
        return _restore_scalar(loss, power_kw, soc)

    # -- Limits -------------------------------------------------------------
    #
    # Two independent ceilings bound every command. The *rate* ceiling is what
    # the pack can push at all, and does not depend on the step length. The
    # *energy* ceiling is what the charge between here and the cutoff can
    # sustain for the whole step, and falls as the step grows. Availability is
    # the lesser; which one bound it is the difference between a pack that is
    # too small and a controller that asked for too much - see B-06.

    def _max_power_current_a(self, soc: float) -> float:
        """Current [A] at the ohmic ceiling, where dP/dI changes sign."""
        return float(self.open_circuit_voltage(soc)) / (2.0 * self.internal_resistance_ohm)

    def _bus_power_kw(self, current_a: float, soc: float) -> float:
        """Bus power [kW] from a signed current - the inverse of the quadratic."""
        v_oc = float(self.open_circuit_voltage(soc))
        r = self.internal_resistance_ohm
        return (v_oc * current_a - current_a * current_a * r) / WATTS_PER_KW

    def _coulomb_limited_current_a(self, soc_headroom: float, dt_s: float) -> float:
        """Constant current [A] that consumes ``soc_headroom`` in exactly ``dt_s``.

        Exact regardless of voltage: coulomb counting is linear in current.
        """
        return soc_headroom * self.charge_capacity_ah * SECONDS_PER_HOUR / dt_s

    def _discharge_rate_ceiling_kw(self, soc: float) -> float:
        """Discharge power [kW] the pack can push, with no regard for how long."""
        return min(self.max_discharge_kw, float(self.ohmic_power_ceiling_kw(soc)))

    def _discharge_energy_ceiling_kw(self, soc: float, dt_s: float) -> float:
        """Discharge power [kW] landing state of charge exactly on ``soc_min``.

        Infinite when the charge above the cutoff would support more current
        than the pack can push through its own resistance: the coulomb budget
        then constrains nothing, and evaluating the quadratic there would run
        back down the far side of the ohmic ceiling and report a *negative*
        power. Open-circuit voltage is taken at the start of the step; see B-06
        for the resulting bias.
        """
        headroom = soc - self.soc_min
        if headroom <= SOC_EPS:
            return 0.0
        current = self._coulomb_limited_current_a(headroom, dt_s)
        if current >= self._max_power_current_a(soc):
            return math.inf
        return self._bus_power_kw(current, soc)

    def _charge_energy_ceiling_kw(self, soc: float, dt_s: float) -> float:
        """Charge power [kW], positive magnitude, landing exactly on soc = 1.

        Charging has no ohmic ceiling to guard: the bus supplies the loss as
        well as the stored energy, so |P| = V_oc*|I| + I^2*R rises without
        bound in |I|. The same quadratic gives it, with the current signed.
        """
        headroom = 1.0 - soc
        if headroom <= SOC_EPS:
            return 0.0
        current = -self._coulomb_limited_current_a(headroom, dt_s)
        return -self._bus_power_kw(current, soc)

    def available_discharge_kw(self, soc: float, dt_s: float | None = None) -> float:
        """Bus power [kW] deliverable, for the whole of ``dt_s`` when given.

        Zero at or below the hard cutoff. With a step length it also respects
        the charge above the cutoff, so it is non-increasing in ``dt_s``;
        without one it is the rate limit alone, which a controller sizing a
        step cannot rely on.
        """
        if soc - self.soc_min <= SOC_EPS:
            return 0.0
        rate_ceiling_kw = self._discharge_rate_ceiling_kw(soc)
        if dt_s is None:
            return rate_ceiling_kw
        _require_positive(dt_s=dt_s)
        return min(rate_ceiling_kw, self._discharge_energy_ceiling_kw(soc, dt_s))

    def available_charge_kw(self, soc: float, dt_s: float | None = None) -> float:
        """Absorbable bus power [kW], positive magnitude, for the whole of ``dt_s``.

        Zero at full charge. The ``dt_s`` behaviour mirrors the discharge side.
        """
        if 1.0 - soc <= SOC_EPS:
            return 0.0
        if dt_s is None:
            return self.max_charge_kw
        _require_positive(dt_s=dt_s)
        return min(self.max_charge_kw, self._charge_energy_ceiling_kw(soc, dt_s))

    def round_trip_efficiency(self, power_kw: float, soc: float) -> float:
        """Discharge-to-charge terminal voltage ratio [-] at equal current magnitude."""
        current = float(self.current_from_power(abs(power_kw), soc))
        sag_v = current * self.internal_resistance_ohm
        v_oc = float(self.open_circuit_voltage(soc))
        return (v_oc - sag_v) / (v_oc + sag_v)

    # -- Integration --------------------------------------------------------

    def step(self, soc: float, power_kw: float, dt_s: float) -> BatteryState:
        """Integrate one timestep under a commanded bus power.

        Limits the command to what the pack can sustain for the whole step -
        C-rate, ohmic ceiling, and the charge left this side of the boundary -
        then coulomb-counts at the limited current and reports the bus power
        that current actually produces. An unachievable command is limited and
        flagged, never raised.

        Nothing is clamped after the fact. The three limits are compared as
        currents, so state of charge arrives on the boundary rather than past
        it, and the energy leaving the pack is the energy the bus receives.
        """
        _require_positive(dt_s=dt_s)

        commanded_kw = float(power_kw)
        charging = commanded_kw < 0.0
        if charging:
            boundary_soc, headroom = 1.0, 1.0 - soc
            rate_ceiling_kw = self.max_charge_kw
            energy_ceiling_kw = self._charge_energy_ceiling_kw(soc, dt_s)
        else:
            boundary_soc, headroom = self.soc_min, soc - self.soc_min
            rate_ceiling_kw = self._discharge_rate_ceiling_kw(soc)
            energy_ceiling_kw = self._discharge_energy_ceiling_kw(soc, dt_s)

        rate_limited = abs(commanded_kw) > rate_ceiling_kw
        energy_limited = abs(commanded_kw) > energy_ceiling_kw

        # All three limits as current magnitudes, so the binding one can be
        # integrated directly. The commanded and rate currents come from the
        # same quadratic used everywhere else.
        commanded_a = abs(float(self.current_from_power(commanded_kw, soc)))
        rate_a = abs(
            float(self.current_from_power(math.copysign(rate_ceiling_kw, commanded_kw), soc))
        )
        energy_a = self._coulomb_limited_current_a(max(headroom, 0.0), dt_s)
        current = math.copysign(min(commanded_a, rate_a, energy_a), commanded_kw)

        r = self.internal_resistance_ohm
        v_oc = float(self.open_circuit_voltage(soc))
        power = (
            self._bus_power_kw(current, soc)
            if rate_limited or energy_limited
            else commanded_kw
        )

        if current == 0.0:
            new_soc = soc
        elif energy_a <= min(commanded_a, rate_a):
            # The energy limit is defined by where it lands, so land there.
            new_soc = boundary_soc
        else:
            # Coulomb counting: state of charge integrates current, not power.
            new_soc = soc - current * dt_s / (self.charge_capacity_ah * SECONDS_PER_HOUR)

        return BatteryState(
            soc=new_soc,
            power_kw=power,
            commanded_kw=commanded_kw,
            current_a=current,
            open_circuit_voltage_v=v_oc,
            terminal_voltage_v=v_oc - current * r,
            ohmic_loss_kw=current * current * r / WATTS_PER_KW,
            at_cutoff=new_soc <= self.soc_min + SOC_EPS,
            rate_limited=rate_limited,
            energy_limited=energy_limited,
            below_safe_floor=new_soc < self.soc_safe_min,
        )
