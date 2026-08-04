"""Rint battery pack on the DC bus of a series hybrid UAV.

Positive power is discharge and negative is charge throughout. Current is
solved from the quadratic P = V_oc*I - I^2*R rather than P/V_oc, so ohmic loss
is the only efficiency term the model needs. State of charge is passed in and
returned, never held here, so the controller can price candidate power splits
without committing to them. Rationale for every default is in
``docs/assumptions.md`` (B-03..B-06, O-05); pack *mass* belongs to ``mass.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["BatteryPack", "BatteryState"]

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
    """What the pack actually did over one integration step."""

    soc: float
    power_kw: float
    commanded_kw: float
    current_a: float
    open_circuit_voltage_v: float
    terminal_voltage_v: float
    ohmic_loss_kw: float
    at_cutoff: bool
    power_limited: bool
    below_safe_floor: bool


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

    def available_discharge_kw(self, soc: float) -> float:
        """Deliverable bus power [kW]; zero at or below the hard cutoff."""
        if soc <= self.soc_min:
            return 0.0
        return min(self.max_discharge_kw, float(self.ohmic_power_ceiling_kw(soc)))

    def available_charge_kw(self, soc: float) -> float:
        """Absorbable bus power [kW] as a positive magnitude; zero at full charge."""
        if soc >= 1.0:
            return 0.0
        return self.max_charge_kw

    def round_trip_efficiency(self, power_kw: float, soc: float) -> float:
        """Discharge-to-charge terminal voltage ratio [-] at equal current magnitude."""
        current = float(self.current_from_power(abs(power_kw), soc))
        sag_v = current * self.internal_resistance_ohm
        v_oc = float(self.open_circuit_voltage(soc))
        return (v_oc - sag_v) / (v_oc + sag_v)

    # -- Integration --------------------------------------------------------

    def step(self, soc: float, power_kw: float, dt_s: float) -> BatteryState:
        """Integrate one timestep under a commanded bus power.

        Applies the C-rate limit, the state-of-charge lockouts and the ohmic
        ceiling, then coulomb-counts. An unachievable command is clamped and
        flagged, never raised.
        """
        _require_positive(dt_s=dt_s)

        commanded_kw = float(power_kw)
        power = min(
            max(commanded_kw, -self.available_charge_kw(soc)),
            self.available_discharge_kw(soc),
        )
        power_limited = power != commanded_kw

        r = self.internal_resistance_ohm
        v_oc = float(self.open_circuit_voltage(soc))
        current = float(self.current_from_power(power, soc))

        # Coulomb counting: state of charge integrates current, not bus power.
        integrated_soc = soc - current * dt_s / (self.charge_capacity_ah * SECONDS_PER_HOUR)
        new_soc = min(max(integrated_soc, 0.0), 1.0)
        # A clamp here means the command was unsustainable for the whole step.
        power_limited = power_limited or new_soc != integrated_soc

        return BatteryState(
            soc=new_soc,
            power_kw=power,
            commanded_kw=commanded_kw,
            current_a=current,
            open_circuit_voltage_v=v_oc,
            terminal_voltage_v=v_oc - current * r,
            ohmic_loss_kw=current * current * r / WATTS_PER_KW,
            at_cutoff=new_soc <= self.soc_min,
            power_limited=power_limited,
            below_safe_floor=new_soc < self.soc_safe_min,
        )
