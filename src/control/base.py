"""Shared interface and behavioural contract for energy-management controllers.

Controllers only price battery energy; the Hamiltonian and power-split search
belong to ``power_split.py``. This module is dependency-free so simulation can
consume the interface without coupling control to any physics implementation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "ControlContext",
    "EMSController",
    "S_ABSOLUTE_MAX",
    "S_ABSOLUTE_MIN",
    "assert_controller_contract",
    "neutral_equivalence_factor",
]

# Equivalence-factor sanity rails, not controller tuning bounds; see C-01.
S_ABSOLUTE_MIN = 0.5
S_ABSOLUTE_MAX = 20.0
KJ_PER_KWH = 3600.0


def _require_finite(name: str, value: float) -> float:
    """Return ``value`` as a float or raise when it is not finite."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def neutral_equivalence_factor(
    sfc_kg_kwh: float,
    source_chain_efficiency: float,
    lhv_kj_kg: float,
    round_trip_efficiency: float = 1.0,
) -> float:
    """Break-even ECMS equivalence factor for one engine operating point.

    Args:
        sfc_kg_kwh: Engine specific fuel consumption [kg/kWh].
        source_chain_efficiency: Engine-shaft to DC-bus efficiency [-].
        lhv_kj_kg: Fuel lower heating value [kJ/kg].
        round_trip_efficiency: Charge-discharge efficiency [-]; ``1.0``
            disables the optional correction — see assumptions.md C-02.
    """
    sfc = _require_finite("sfc_kg_kwh", sfc_kg_kwh)
    source_efficiency = _require_finite(
        "source_chain_efficiency", source_chain_efficiency
    )
    lhv = _require_finite("lhv_kj_kg", lhv_kj_kg)
    round_trip = _require_finite("round_trip_efficiency", round_trip_efficiency)

    if sfc <= 0.0:
        raise ValueError(f"sfc_kg_kwh must be positive, got {sfc_kg_kwh!r}")
    if not 0.0 < source_efficiency <= 1.0:
        raise ValueError(
            "source_chain_efficiency must lie in (0, 1], "
            f"got {source_chain_efficiency!r}"
        )
    if lhv <= 0.0:
        raise ValueError(f"lhv_kj_kg must be positive, got {lhv_kj_kg!r}")
    if not 0.0 < round_trip <= 1.0:
        raise ValueError(
            f"round_trip_efficiency must lie in (0, 1], got {round_trip_efficiency!r}"
        )

    # Equality of engine fuel per bus kWh and equivalent battery fuel.
    neutral = sfc / (source_efficiency * KJ_PER_KWH / lhv)
    return neutral / round_trip


@dataclass(frozen=True)
class ControlContext:
    """Immutable inputs available to a controller at one simulation step.

    A controller consuming ``phase`` is overfit to this mission profile and its
    results must be interpreted accordingly.
    """

    soc: float
    bus_demand_kw: float
    max_bus_kw: float
    neutral_s: float
    switching_s: float
    time_s: float
    phase: str

    def __post_init__(self) -> None:
        soc = _require_finite("soc", self.soc)
        bus_demand = _require_finite("bus_demand_kw", self.bus_demand_kw)
        max_bus = _require_finite("max_bus_kw", self.max_bus_kw)
        neutral = _require_finite("neutral_s", self.neutral_s)
        switching = _require_finite("switching_s", self.switching_s)
        time = _require_finite("time_s", self.time_s)

        if not 0.0 <= soc <= 1.0:
            raise ValueError(f"soc must lie in [0, 1], got {self.soc!r}")
        if max_bus < 0.0:
            raise ValueError(f"max_bus_kw must be non-negative, got {self.max_bus_kw!r}")
        if neutral <= 0.0:
            raise ValueError(f"neutral_s must be positive, got {self.neutral_s!r}")
        if switching <= 0.0:
            raise ValueError(f"switching_s must be positive, got {self.switching_s!r}")
        if time < 0.0:
            raise ValueError(f"time_s must be non-negative, got {self.time_s!r}")

        object.__setattr__(self, "soc", soc)
        object.__setattr__(self, "bus_demand_kw", bus_demand)
        object.__setattr__(self, "max_bus_kw", max_bus)
        object.__setattr__(self, "neutral_s", neutral)
        object.__setattr__(self, "switching_s", switching)
        object.__setattr__(self, "time_s", time)

    @property
    def demand_normalised(self) -> float:
        """DC-bus demand divided by its normalising maximum, in [0, 1]."""
        if self.max_bus_kw == 0.0:
            return 0.0
        return min(max(self.bus_demand_kw / self.max_bus_kw, 0.0), 1.0)


class EMSController(ABC):
    """Common interface for every energy-management strategy.

    Controllers are stateless; their output must be a pure function of context.
    """

    @abstractmethod
    def equivalence_factor(self, ctx: ControlContext) -> float:
        """Return the controller's unconstrained equivalence factor."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable human-readable strategy name."""

    def clamped_equivalence_factor(self, ctx: ControlContext) -> float:
        """Return the equivalence factor within the absolute sanity rails."""
        value = float(self.equivalence_factor(ctx))
        return min(max(value, S_ABSOLUTE_MIN), S_ABSOLUTE_MAX)


def _contract_failure(number: int, detail: str) -> None:
    raise AssertionError(f"controller contract assertion {number} failed: {detail}")


def assert_controller_contract(
    controller: EMSController,
    tolerance: float = 1.0e-9,
    *,
    skip_assertions: Iterable[int] = (),
) -> None:
    """Assert the shared behavioural contract over state-of-charge and demand.

    ``skip_assertions`` explicitly exempts assertions 3–5 for experimental
    control groups such as fixed-s ECMS; safety assertions 1–2 cannot be waived.
    """
    tolerance = _require_finite("tolerance", tolerance)
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance!r}")

    skipped = frozenset(skip_assertions)
    invalid_skips = skipped.difference({3, 4, 5})
    if invalid_skips:
        raise ValueError(
            "skip_assertions may contain only 3, 4, and 5; "
            f"got {sorted(invalid_skips)!r}"
        )

    soc_grid = tuple(index / 10.0 for index in range(11))
    demand_grid = tuple(index / 10.0 for index in range(11))
    switching_reference = 5.0

    # Assertions 3–5 guard the flat-controller failure in AGENTS.md known traps.
    for demand in demand_grid:
        outputs: list[float] = []
        for soc in soc_grid:
            ctx = ControlContext(
                soc=soc,
                bus_demand_kw=demand,
                max_bus_kw=1.0,
                neutral_s=8.0,
                switching_s=switching_reference,
                time_s=0.0,
                phase="contract",
            )
            first = controller.clamped_equivalence_factor(ctx)
            if not math.isfinite(first) or not S_ABSOLUTE_MIN <= first <= S_ABSOLUTE_MAX:
                _contract_failure(1, f"non-finite or unbounded output {first!r}")

            second = controller.clamped_equivalence_factor(ctx)
            if first != second:
                _contract_failure(2, f"non-deterministic outputs {first!r} and {second!r}")
            outputs.append(first)

        if 3 not in skipped:
            for lower_soc, higher_soc in zip(outputs, outputs[1:]):
                if higher_soc > lower_soc + tolerance:
                    _contract_failure(
                        3,
                        f"output rose with state of charge at demand {demand:.1f}",
                    )

        if 4 not in skipped and max(outputs) - min(outputs) <= tolerance:
            _contract_failure(
                4,
                f"output is flat in state of charge at demand {demand:.1f}",
            )

        if 5 not in skipped:
            straddles = any(
                value < switching_reference - tolerance for value in outputs
            ) and any(
                value > switching_reference + tolerance for value in outputs
            )
            if not straddles:
                _contract_failure(
                    5,
                    f"output does not straddle switching_s at demand {demand:.1f}",
                )
