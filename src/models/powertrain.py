"""Series hybrid conversion chain between the propeller shaft and the DC bus.

Every stage between fuel and thrust that is not the engine, the battery or the
propeller lives here: generator and rectifier on the source side, cabling on the
bus, inverter and motor on the demand side. Power is kW throughout and flows are
non-negative; battery bus power is positive on discharge, matching ``battery.py``.
The power split is ``control/``, fuel flow is ``engine.py``, propeller efficiency
is ``aerodynamics.py``. Rationale for every default is in ``docs/assumptions.md``
(P-01..P-05, O-11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["PowertrainState", "SeriesPowertrain"]

FloatOrArray = float | npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Stage efficiency defaults - see assumptions.md P-01, P-04
# ---------------------------------------------------------------------------

ETA_GENERATOR = 0.95
ETA_RECTIFIER = 0.95
ETA_INVERTER = 0.95
ETA_MOTOR = 0.95
ETA_CABLING = 0.99

# ---------------------------------------------------------------------------
# Load-dependent loss model - see assumptions.md O-11
# ---------------------------------------------------------------------------

NOLOAD_LOSS_FRACTION = 0.30

# ---------------------------------------------------------------------------
# Bus balance tolerance
# ---------------------------------------------------------------------------

BALANCE_TOLERANCE_REL = 1.0e-6
BALANCE_TOLERANCE_ABS_KW = 1.0e-6


def _as_array(x: FloatOrArray) -> npt.NDArray[np.float64]:
    """Coerce to a float64 ndarray (possibly 0-d) without copying if able."""
    return np.asarray(x, dtype=np.float64)


def _restore_scalar(result: npt.NDArray[np.float64], original: FloatOrArray) -> FloatOrArray:
    """Return a Python float when the caller passed a scalar, else the ndarray."""
    return float(result) if np.ndim(original) == 0 else result


def _validate_efficiency(**values: float) -> None:
    """Raise ``ValueError`` naming the first efficiency outside (0, 1]."""
    for name, value in values.items():
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must lie in (0, 1], got {value!r}")


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio of two powers [-]; zero when the denominator does not flow."""
    return numerator / denominator if denominator > 0.0 else 0.0


@dataclass(frozen=True)
class PowertrainState:
    """What the conversion chain did for one demand and one candidate split."""

    shaft_demand_kw: float
    bus_demand_kw: float
    engine_shaft_kw: float
    bus_from_engine_kw: float
    battery_bus_kw: float
    bus_residual_kw: float
    balanced: bool
    source_losses_kw: float
    demand_losses_kw: float
    total_losses_kw: float
    source_efficiency: float
    demand_efficiency: float


@dataclass(frozen=True)
class SeriesPowertrain:
    """Efficiency chain of a series hybrid, with an optional load-dependent mode.

    Args:
        eta_generator: Engine shaft to generator terminals [-].
        eta_rectifier: Generator terminals to DC bus [-].
        eta_inverter: DC bus to motor terminals [-].
        eta_motor: Motor terminals to propeller shaft [-].
        eta_cabling: DC distribution [-] - see assumptions.md P-04.
        load_dependent: Derate each machine off its own rating - see O-11.
        noload_loss_fraction: Share of rated loss that is load-independent [-].
        rated_engine_kw: Generator and rectifier rating [kW]; required when
            ``load_dependent``.
        rated_bus_kw: Inverter and motor rating [kW]; required when
            ``load_dependent``.
    """

    eta_generator: float = ETA_GENERATOR
    eta_rectifier: float = ETA_RECTIFIER
    eta_inverter: float = ETA_INVERTER
    eta_motor: float = ETA_MOTOR
    eta_cabling: float = ETA_CABLING
    load_dependent: bool = False
    noload_loss_fraction: float = NOLOAD_LOSS_FRACTION
    rated_engine_kw: float | None = None
    rated_bus_kw: float | None = None

    def __post_init__(self) -> None:
        _validate_efficiency(
            eta_generator=self.eta_generator,
            eta_rectifier=self.eta_rectifier,
            eta_inverter=self.eta_inverter,
            eta_motor=self.eta_motor,
            eta_cabling=self.eta_cabling,
        )
        if not 0.0 <= self.noload_loss_fraction < 1.0:
            raise ValueError(
                f"noload_loss_fraction must lie in [0, 1), got {self.noload_loss_fraction!r}"
            )
        if self.load_dependent:
            for name in ("rated_engine_kw", "rated_bus_kw"):
                rating = getattr(self, name)
                if rating is None or not rating > 0.0:
                    raise ValueError(
                        f"{name} must be positive when load_dependent, got {rating!r}"
                    )

    # -- Rated chain efficiencies -------------------------------------------

    @property
    def source_chain_efficiency(self) -> float:
        """Engine shaft to DC bus at rated load [-]."""
        return self.eta_generator * self.eta_rectifier

    @property
    def demand_chain_efficiency(self) -> float:
        """DC bus to propeller shaft at rated load [-]."""
        return self.eta_inverter * self.eta_motor * self.eta_cabling

    @property
    def chain_efficiency(self) -> float:
        """Engine shaft to propeller shaft at rated load [-]."""
        return self.source_chain_efficiency * self.demand_chain_efficiency

    # -- Single-stage conversion --------------------------------------------
    #
    # Cabling is resistive with no excitation to sustain, so it carries no
    # no-load term and stays constant in both modes - see assumptions.md P-04.

    def _loss_coefficients(self, eta_rated: float, rated_kw: float) -> tuple[float, float]:
        """No-load [kW] and load-squared [1/kW] loss coefficients for one stage."""
        # Split the rated loss so efficiency recovers eta_rated at the rating.
        loss_rated_kw = rated_kw * (1.0 / eta_rated - 1.0)
        no_load_kw = self.noload_loss_fraction * loss_rated_kw
        quadratic = (1.0 - self.noload_loss_fraction) * loss_rated_kw / (rated_kw * rated_kw)
        return no_load_kw, quadratic

    def _stage_input(
        self, output: npt.NDArray[np.float64], eta_rated: float, rated_kw: float | None
    ) -> npt.NDArray[np.float64]:
        """Input power [kW] a stage draws to produce ``output``."""
        if not self.load_dependent or rated_kw is None:
            return output / eta_rated
        # P_in = P_out + a + b*P_out^2
        no_load_kw, quadratic = self._loss_coefficients(eta_rated, rated_kw)
        throughput = np.maximum(output, 0.0)
        return throughput + no_load_kw + quadratic * throughput * throughput

    def _stage_output(
        self, power_in: npt.NDArray[np.float64], eta_rated: float, rated_kw: float | None
    ) -> npt.NDArray[np.float64]:
        """Output power [kW] a stage produces from ``power_in``."""
        if not self.load_dependent or rated_kw is None:
            return power_in * eta_rated
        # Positive root of b*P_out^2 + P_out - (P_in - a) = 0, written so it
        # stays exact as b approaches zero. Input below the no-load loss turns
        # into no output at all.
        no_load_kw, quadratic = self._loss_coefficients(eta_rated, rated_kw)
        net = np.maximum(power_in - no_load_kw, 0.0)
        return 2.0 * net / (1.0 + np.sqrt(1.0 + 4.0 * quadratic * net))

    def stage_efficiency(
        self, output_kw: FloatOrArray, eta_rated: float, rated_kw: float | None
    ) -> FloatOrArray:
        """Efficiency [-] of a single stage delivering ``output_kw``."""
        output = _as_array(output_kw)
        power_in = self._stage_input(output, eta_rated, rated_kw)
        return _restore_scalar(np.where(power_in > 0.0, output / power_in, 0.0), output_kw)

    # -- Chain conversion, vectorized ---------------------------------------

    def bus_power_required(self, shaft_power_kw: FloatOrArray) -> FloatOrArray:
        """DC bus power [kW] drawn to deliver shaft power at the propeller."""
        shaft = _as_array(shaft_power_kw)
        motor_in = self._stage_input(shaft, self.eta_motor, self.rated_bus_kw)
        inverter_in = self._stage_input(motor_in, self.eta_inverter, self.rated_bus_kw)
        return _restore_scalar(inverter_in / self.eta_cabling, shaft_power_kw)

    def shaft_power_from_bus(self, bus_kw: FloatOrArray) -> FloatOrArray:
        """Propeller shaft power [kW] delivered by a DC bus draw."""
        bus = _as_array(bus_kw)
        inverter_in = bus * self.eta_cabling
        motor_in = self._stage_output(inverter_in, self.eta_inverter, self.rated_bus_kw)
        return _restore_scalar(
            self._stage_output(motor_in, self.eta_motor, self.rated_bus_kw), bus_kw
        )

    def bus_power_from_engine(self, engine_shaft_kw: FloatOrArray) -> FloatOrArray:
        """DC bus power [kW] contributed by engine shaft power."""
        shaft = _as_array(engine_shaft_kw)
        generator_out = self._stage_output(shaft, self.eta_generator, self.rated_engine_kw)
        return _restore_scalar(
            self._stage_output(generator_out, self.eta_rectifier, self.rated_engine_kw),
            engine_shaft_kw,
        )

    def engine_power_for_bus(self, bus_kw: FloatOrArray) -> FloatOrArray:
        """Engine shaft power [kW] needed for a DC bus contribution."""
        bus = _as_array(bus_kw)
        rectifier_in = self._stage_input(bus, self.eta_rectifier, self.rated_engine_kw)
        return _restore_scalar(
            self._stage_input(rectifier_in, self.eta_generator, self.rated_engine_kw), bus_kw
        )

    # -- Bus balance --------------------------------------------------------

    def solve(
        self, shaft_demand_kw: float, engine_shaft_kw: float, battery_bus_kw: float
    ) -> PowertrainState:
        """Resolve one candidate split against the DC bus balance.

        An imbalance is reported, never raised; the caller decides whether it
        constitutes mission failure.
        """
        shaft_demand_kw = float(shaft_demand_kw)
        engine_shaft_kw = float(engine_shaft_kw)
        battery_bus_kw = float(battery_bus_kw)

        bus_demand_kw = float(self.bus_power_required(shaft_demand_kw))
        bus_from_engine_kw = float(self.bus_power_from_engine(engine_shaft_kw))

        # P_bus_from_engine + P_battery - P_bus_demand
        residual_kw = bus_from_engine_kw + battery_bus_kw - bus_demand_kw
        tolerance_kw = max(BALANCE_TOLERANCE_REL * abs(bus_demand_kw), BALANCE_TOLERANCE_ABS_KW)

        source_losses_kw = engine_shaft_kw - bus_from_engine_kw
        demand_losses_kw = bus_demand_kw - shaft_demand_kw

        return PowertrainState(
            shaft_demand_kw=shaft_demand_kw,
            bus_demand_kw=bus_demand_kw,
            engine_shaft_kw=engine_shaft_kw,
            bus_from_engine_kw=bus_from_engine_kw,
            battery_bus_kw=battery_bus_kw,
            bus_residual_kw=residual_kw,
            balanced=abs(residual_kw) <= tolerance_kw,
            source_losses_kw=source_losses_kw,
            demand_losses_kw=demand_losses_kw,
            total_losses_kw=source_losses_kw + demand_losses_kw,
            source_efficiency=_safe_ratio(bus_from_engine_kw, engine_shaft_kw),
            demand_efficiency=_safe_ratio(shaft_demand_kw, bus_demand_kw),
        )

    # -- Reported metrics ---------------------------------------------------

    def system_efficiency(
        self, propulsive_kw: float, fuel_chemical_kw: float, battery_bus_kw: float
    ) -> float:
        """Propulsive output over energy input [-] - see assumptions.md P-05."""
        # Charging is not an input to the system, so only discharge counts.
        input_kw = float(fuel_chemical_kw) + max(float(battery_bus_kw), 0.0)
        return _safe_ratio(float(propulsive_kw), input_kw)

    def peak_bus_power(self, max_shaft_kw: float) -> float:
        """DC bus power [kW] at the worst-case shaft demand, for M-04 sizing."""
        return float(self.bus_power_required(float(max_shaft_kw)))
