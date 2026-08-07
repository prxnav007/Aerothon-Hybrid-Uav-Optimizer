"""Rint battery pack on the DC bus of a series hybrid UAV.

Positive power is discharge and negative is charge throughout. Current is
solved from the quadratic P = V_oc*I - I^2*R rather than P/V_oc, so ohmic loss
is the only efficiency term the model needs. State of charge is passed in and
returned, never held here, so the controller can price candidate power splits
without committing to them.

Legacy mode preserves the original bus-power proxy and start-of-step OCV
integration. Physical mode requires explicit current and terminal-voltage
limits and uses midpoint OCV. Available power remains energy-aware in both.
Rationale is in ``docs/assumptions.md`` (B-03..B-07, O-05); pack mass belongs
to ``mass.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

__all__ = [
    "BatteryAvailability",
    "BatteryMode",
    "BatteryPack",
    "BatteryState",
    "SOC_EPS",
]

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


class BatteryMode(str, Enum):
    """Selectable battery limiting and integration convention."""

    LEGACY = "legacy"
    PHYSICAL = "physical"


@dataclass(frozen=True)
class BatteryAvailability:
    """Power capability and the constraint that establishes it."""

    power_kw: float
    current_a: float
    constraint_terminal_voltage_v: float
    binding_limit: str


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
    current_limited: bool = False
    voltage_limited: bool = False
    active_limit: str = "none"
    constraint_terminal_voltage_v: float | None = None

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
        mode: ``legacy`` bus-power proxy or explicit-limit ``physical`` mode.
        i_charge_max_a: Physical-mode charge-current magnitude [A].
        i_discharge_max_a: Physical-mode discharge-current magnitude [A].
        terminal_voltage_min_v: Physical-mode discharge terminal floor [V].
        terminal_voltage_max_v: Physical-mode charge terminal ceiling [V].
        q_nominal_ah: Physical-mode nominal charge capacity [Ah].
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
    mode: BatteryMode | str = BatteryMode.LEGACY
    i_charge_max_a: float | None = None
    i_discharge_max_a: float | None = None
    terminal_voltage_min_v: float | None = None
    terminal_voltage_max_v: float | None = None
    q_nominal_ah: float | None = None

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
        try:
            BatteryMode(self.mode)
        except (TypeError, ValueError) as error:
            raise ValueError(f"mode must be 'legacy' or 'physical', got {self.mode!r}") from error
        physical_values = {
            "i_charge_max_a": self.i_charge_max_a,
            "i_discharge_max_a": self.i_discharge_max_a,
            "terminal_voltage_min_v": self.terminal_voltage_min_v,
            "terminal_voltage_max_v": self.terminal_voltage_max_v,
            "q_nominal_ah": self.q_nominal_ah,
        }
        if self.battery_mode is BatteryMode.PHYSICAL:
            if any(value is None for value in physical_values.values()):
                missing = [name for name, value in physical_values.items() if value is None]
                raise ValueError(
                    "physical mode requires explicit parameters: " + ", ".join(missing)
                )
            _require_positive(
                **{name: float(value) for name, value in physical_values.items()}
            )
            if not float(self.terminal_voltage_min_v) < float(
                self.terminal_voltage_max_v
            ):
                raise ValueError("physical terminal voltage limits are reversed")
        elif any(value is not None for value in physical_values.values()):
            raise ValueError("physical limit parameters require mode='physical'")

    # -- Pack constants -----------------------------------------------------

    @property
    def battery_mode(self) -> BatteryMode:
        """Validated limiting and integration mode."""
        return BatteryMode(self.mode)

    @property
    def nominal_voltage_v(self) -> float:
        """Mean open-circuit voltage over a full state-of-charge sweep [V]."""
        return 0.5 * (self.v_min_v + self.v_max_v)

    @property
    def charge_capacity_ah(self) -> float:
        """Charge capacity [Ah]; recovers the rated energy at nominal voltage."""
        if self.battery_mode is BatteryMode.PHYSICAL:
            return float(self.q_nominal_ah)
        return self.capacity_kwh * WATTS_PER_KW / self.nominal_voltage_v

    @property
    def internal_resistance_ohm(self) -> float:
        """Pack internal resistance [ohm]."""
        if not self.scale_resistance:
            return self.r_ref_ohm
        # Capacity is added as parallel strings at fixed voltage - see O-05.
        return self.r_ref_ohm * self.r_ref_capacity_kwh / self.capacity_kwh

    def stored_energy_kwh(self, soc: FloatOrArray) -> FloatOrArray:
        """Open-circuit stored energy [kWh] between zero and the given SoC."""
        state = _as_array(soc)
        if not bool(np.all(np.isfinite(state) & (state >= 0.0) & (state <= 1.0))):
            raise ValueError(f"soc must lie in [0, 1], got {soc!r}")
        voltage_integral_v = (
            self.v_min_v * state
            + 0.5 * (self.v_max_v - self.v_min_v) * state * state
        )
        # Integral of V_oc dq for the linear-OCV pack; see assumptions.md B-03.
        energy = self.charge_capacity_ah * voltage_integral_v / WATTS_PER_KW
        return _restore_scalar(energy, soc)

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

    def _effective_resistance_ohm(self, dt_s: float) -> float:
        """Midpoint-OCV quadratic coefficient for a constant-current step."""
        slope_v = self.v_max_v - self.v_min_v
        midpoint_term = slope_v * dt_s / (
            2.0 * self.charge_capacity_ah * SECONDS_PER_HOUR
        )
        return self.internal_resistance_ohm + midpoint_term

    def current_from_power_over_step(
        self, power_kw: float, soc: float, dt_s: float
    ) -> float:
        """Constant current [A] producing bus power over one integration step."""
        _require_positive(dt_s=dt_s)
        if self.battery_mode is BatteryMode.LEGACY:
            return float(self.current_from_power(power_kw, soc))
        v_start = float(self.open_circuit_voltage(soc))
        r_eff = self._effective_resistance_ohm(dt_s)
        power_w = float(power_kw) * WATTS_PER_KW
        discriminant = max(v_start * v_start - 4.0 * r_eff * power_w, 0.0)
        return (v_start - math.sqrt(discriminant)) / (2.0 * r_eff)

    def integration_ocv_v(self, soc: float, current_a: float, dt_s: float) -> float:
        """OCV [V] used by the selected step integration convention."""
        if self.battery_mode is BatteryMode.LEGACY:
            return float(self.open_circuit_voltage(soc))
        midpoint_soc = soc - current_a * dt_s / (
            2.0 * self.charge_capacity_ah * SECONDS_PER_HOUR
        )
        return float(self.open_circuit_voltage(midpoint_soc))

    def internal_power_kw(self, power_kw: float, soc: float, dt_s: float) -> float:
        """Open-circuit power [kW] under the selected integration convention."""
        current = self.current_from_power_over_step(power_kw, soc, dt_s)
        return self.integration_ocv_v(soc, current, dt_s) * current / WATTS_PER_KW

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

    @staticmethod
    def _binding_limit(candidates: tuple[tuple[str, float], ...]) -> tuple[float, str]:
        current = min(value for _, value in candidates)
        tolerance = max(1.0e-10, abs(current) * 1.0e-10)
        names = [
            name for name, value in candidates if abs(value - current) <= tolerance
        ]
        return current, "_and_".join(names)

    def _physical_availability(
        self, soc: float, dt_s: float | None, *, charging: bool
    ) -> BatteryAvailability:
        if charging:
            headroom = 1.0 - soc
            if headroom <= SOC_EPS:
                return BatteryAvailability(
                    0.0,
                    0.0,
                    float(self.open_circuit_voltage(soc)),
                    "soc_boundary",
                )
            current_limit = float(self.i_charge_max_a)
            voltage_headroom = float(self.terminal_voltage_max_v) - float(
                self.open_circuit_voltage(soc)
            )
        else:
            headroom = soc - self.soc_min
            if headroom <= SOC_EPS:
                return BatteryAvailability(
                    0.0,
                    0.0,
                    float(self.open_circuit_voltage(soc)),
                    "soc_boundary",
                )
            current_limit = float(self.i_discharge_max_a)
            voltage_headroom = float(self.open_circuit_voltage(soc)) - float(
                self.terminal_voltage_min_v
            )

        slope_v = self.v_max_v - self.v_min_v
        endpoint_term = (
            slope_v * dt_s / (self.charge_capacity_ah * SECONDS_PER_HOUR)
            if dt_s is not None
            else 0.0
        )
        voltage_current = max(
            voltage_headroom / (self.internal_resistance_ohm + endpoint_term),
            0.0,
        )
        candidates: list[tuple[str, float]] = [
            ("current", current_limit),
            ("voltage", voltage_current),
        ]
        if not charging:
            effective_r = (
                self._effective_resistance_ohm(dt_s)
                if dt_s is not None
                else self.internal_resistance_ohm
            )
            candidates.append(
                (
                    "ohmic",
                    float(self.open_circuit_voltage(soc)) / (2.0 * effective_r),
                )
            )
        if dt_s is not None:
            _require_positive(dt_s=dt_s)
            candidates.append(
                (
                    "energy",
                    self._coulomb_limited_current_a(headroom, dt_s),
                )
            )
        magnitude_a, binding = self._binding_limit(tuple(candidates))
        signed_current = -magnitude_a if charging else magnitude_a
        if dt_s is None:
            power_kw = self._bus_power_kw(signed_current, soc)
            terminal_v = float(self.open_circuit_voltage(soc)) - (
                signed_current * self.internal_resistance_ohm
            )
        else:
            v_mid = self.integration_ocv_v(soc, signed_current, dt_s)
            power_kw = (
                v_mid * signed_current
                - signed_current * signed_current * self.internal_resistance_ohm
            ) / WATTS_PER_KW
            end_soc = soc - signed_current * dt_s / (
                self.charge_capacity_ah * SECONDS_PER_HOUR
            )
            terminal_v = float(self.open_circuit_voltage(end_soc)) - (
                signed_current * self.internal_resistance_ohm
            )
        return BatteryAvailability(abs(power_kw), magnitude_a, terminal_v, binding)

    def discharge_availability(
        self, soc: float, dt_s: float | None = None
    ) -> BatteryAvailability:
        """Detailed positive discharge capability and active constraint."""
        if self.battery_mode is BatteryMode.PHYSICAL:
            return self._physical_availability(soc, dt_s, charging=False)
        if soc - self.soc_min <= SOC_EPS:
            return BatteryAvailability(
                0.0,
                0.0,
                float(self.open_circuit_voltage(soc)),
                "soc_boundary",
            )
        rate_kw = self._discharge_rate_ceiling_kw(soc)
        rate_name = (
            "legacy_power"
            if self.max_discharge_kw <= float(self.ohmic_power_ceiling_kw(soc))
            else "ohmic"
        )
        candidates = [(rate_name, rate_kw)]
        if dt_s is not None:
            _require_positive(dt_s=dt_s)
            candidates.append(("energy", self._discharge_energy_ceiling_kw(soc, dt_s)))
        binding, power_kw = min(candidates, key=lambda item: item[1])
        current = abs(float(self.current_from_power(power_kw, soc)))
        terminal = float(self.terminal_voltage(power_kw, soc))
        return BatteryAvailability(power_kw, current, terminal, binding)

    def charge_availability(
        self, soc: float, dt_s: float | None = None
    ) -> BatteryAvailability:
        """Detailed positive charge-power magnitude and active constraint."""
        if self.battery_mode is BatteryMode.PHYSICAL:
            return self._physical_availability(soc, dt_s, charging=True)
        if 1.0 - soc <= SOC_EPS:
            return BatteryAvailability(
                0.0,
                0.0,
                float(self.open_circuit_voltage(soc)),
                "soc_boundary",
            )
        candidates = [("legacy_power", self.max_charge_kw)]
        if dt_s is not None:
            _require_positive(dt_s=dt_s)
            candidates.append(("energy", self._charge_energy_ceiling_kw(soc, dt_s)))
        binding, power_kw = min(candidates, key=lambda item: item[1])
        current = abs(float(self.current_from_power(-power_kw, soc)))
        terminal = float(self.terminal_voltage(-power_kw, soc))
        return BatteryAvailability(power_kw, current, terminal, binding)

    def available_discharge_kw(self, soc: float, dt_s: float | None = None) -> float:
        """Bus power [kW] deliverable, for the whole of ``dt_s`` when given.

        Zero at or below the hard cutoff. With a step length it also respects
        the charge above the cutoff, so it is non-increasing in ``dt_s``;
        without one it is the rate limit alone, which a controller sizing a
        step cannot rely on.
        """
        if self.battery_mode is BatteryMode.PHYSICAL:
            return self.discharge_availability(soc, dt_s).power_kw
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
        if self.battery_mode is BatteryMode.PHYSICAL:
            return self.charge_availability(soc, dt_s).power_kw
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
        """Integrate one timestep under the selected battery convention."""
        if self.battery_mode is BatteryMode.PHYSICAL:
            return self._step_physical(soc, power_kw, dt_s)
        return self._step_legacy(soc, power_kw, dt_s)

    def _step_legacy(self, soc: float, power_kw: float, dt_s: float) -> BatteryState:
        """Integrate the frozen bus-power-proxy convention.

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
            active_limit=(
                (
                    self.charge_availability(soc, dt_s).binding_limit
                    if charging
                    else self.discharge_availability(soc, dt_s).binding_limit
                )
                if abs(commanded_kw)
                >= (
                    self.available_charge_kw(soc, dt_s)
                    if charging
                    else self.available_discharge_kw(soc, dt_s)
                )
                - 1.0e-10
                and abs(commanded_kw) > 0.0
                else "none"
            ),
            constraint_terminal_voltage_v=v_oc - current * r,
        )

    def _step_physical(self, soc: float, power_kw: float, dt_s: float) -> BatteryState:
        """Integrate a midpoint-OCV step under current and terminal limits."""
        _require_positive(dt_s=dt_s)
        commanded_kw = float(power_kw)
        charging = commanded_kw < 0.0
        availability = (
            self.charge_availability(soc, dt_s)
            if charging
            else self.discharge_availability(soc, dt_s)
        )
        commanded_a = abs(
            self.current_from_power_over_step(commanded_kw, soc, dt_s)
        )
        tolerance_a = max(1.0e-10, availability.current_a * 1.0e-10)
        limited = commanded_a > availability.current_a + tolerance_a
        current_magnitude = min(commanded_a, availability.current_a)
        current = -current_magnitude if charging else current_magnitude
        active = (
            availability.binding_limit
            if commanded_a >= availability.current_a - tolerance_a
            and abs(commanded_kw) > 0.0
            else "none"
        )
        current_limited = limited and "current" in availability.binding_limit
        voltage_limited = limited and "voltage" in availability.binding_limit
        energy_limited = limited and any(
            name in availability.binding_limit for name in ("energy", "soc_boundary")
        )
        rate_limited = limited and any(
            name in availability.binding_limit
            for name in ("current", "voltage", "ohmic")
        )

        v_mid = self.integration_ocv_v(soc, current, dt_s)
        r = self.internal_resistance_ohm
        power = (
            (v_mid * current - current * current * r) / WATTS_PER_KW
            if limited
            else commanded_kw
        )
        if current == 0.0:
            new_soc = soc
        elif energy_limited:
            new_soc = 1.0 if charging else self.soc_min
        else:
            new_soc = soc - current * dt_s / (
                self.charge_capacity_ah * SECONDS_PER_HOUR
            )
        terminal_mid = v_mid - current * r
        constraint_terminal = float(self.open_circuit_voltage(new_soc)) - current * r

        return BatteryState(
            soc=new_soc,
            power_kw=power,
            commanded_kw=commanded_kw,
            current_a=current,
            open_circuit_voltage_v=v_mid,
            terminal_voltage_v=terminal_mid,
            ohmic_loss_kw=current * current * r / WATTS_PER_KW,
            at_cutoff=new_soc <= self.soc_min + SOC_EPS,
            rate_limited=rate_limited,
            energy_limited=energy_limited,
            below_safe_floor=new_soc < self.soc_safe_min,
            current_limited=current_limited,
            voltage_limited=voltage_limited,
            active_limit=active,
            constraint_terminal_voltage_v=constraint_terminal,
        )
