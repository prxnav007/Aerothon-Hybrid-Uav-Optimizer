"""Opt-in fuzzy adaptation of the ECMS equivalence factor.

The controller maps state of charge and normalized DC-bus demand to a ratio
around the marginal ``switching_s`` reference.  It is deliberately stateless
and is not selected by the mission simulator or any optimization workflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.control.base import ControlContext, EMSController

__all__ = ["FuzzyECMS"]

# Experimental fuzzy defaults are documented in assumptions.md C-06.
SOC_LOW_DEFAULT = 0.30
SOC_HIGH_DEFAULT = 0.70
S_MIN_RATIO_DEFAULT = 0.75
S_MAX_RATIO_DEFAULT = 1.25


def _finite_parameter(name: str, value: float) -> float:
    """Return a finite float or raise ``ValueError`` naming the parameter."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _three_set_partition(
    value: float,
    low_shoulder_end: float,
    high_shoulder_start: float,
) -> tuple[float, float, float]:
    """Return a complete low/medium/high piecewise-linear partition."""
    midpoint = 0.5 * (low_shoulder_end + high_shoulder_start)
    if value <= low_shoulder_end:
        return 1.0, 0.0, 0.0
    if value < midpoint:
        medium = (value - low_shoulder_end) / (midpoint - low_shoulder_end)
        return 1.0 - medium, medium, 0.0
    if value < high_shoulder_start:
        high = (value - midpoint) / (high_shoulder_start - midpoint)
        return 0.0, 1.0 - high, high
    return 0.0, 0.0, 1.0


@dataclass(frozen=True)
class FuzzyECMS(EMSController):
    """Experimental fuzzy ECMS controller with no workflow activation.

    ``soc_low`` and ``soc_high`` locate the saturated low/high SoC shoulders.
    The consequent range is expressed as ratios around ``switching_s`` so it
    follows the current engine and source-chain operating point.
    """

    soc_low: float = SOC_LOW_DEFAULT
    soc_high: float = SOC_HIGH_DEFAULT
    s_min_ratio: float = S_MIN_RATIO_DEFAULT
    s_max_ratio: float = S_MAX_RATIO_DEFAULT

    def __post_init__(self) -> None:
        soc_low = _finite_parameter("soc_low", self.soc_low)
        soc_high = _finite_parameter("soc_high", self.soc_high)
        s_min_ratio = _finite_parameter("s_min_ratio", self.s_min_ratio)
        s_max_ratio = _finite_parameter("s_max_ratio", self.s_max_ratio)

        if not 0.0 < soc_low < soc_high < 1.0:
            raise ValueError(
                "soc thresholds must satisfy 0 < soc_low < soc_high < 1, "
                f"got {soc_low!r} and {soc_high!r}"
            )
        if not 0.0 < s_min_ratio < 1.0:
            raise ValueError(
                "s_min_ratio must lie in (0, 1), "
                f"got {s_min_ratio!r}"
            )
        if not s_max_ratio > 1.0:
            raise ValueError(
                "s_max_ratio must exceed 1, "
                f"got {s_max_ratio!r}"
            )

        object.__setattr__(self, "soc_low", soc_low)
        object.__setattr__(self, "soc_high", soc_high)
        object.__setattr__(self, "s_min_ratio", s_min_ratio)
        object.__setattr__(self, "s_max_ratio", s_max_ratio)

    def _equivalence_ratio(self, ctx: ControlContext) -> float:
        soc_membership = _three_set_partition(
            ctx.soc,
            self.soc_low,
            self.soc_high,
        )
        demand_membership = _three_set_partition(
            ctx.demand_normalised,
            0.0,
            1.0,
        )

        low = self.s_min_ratio
        medium_low = 0.5 * (self.s_min_ratio + 1.0)
        medium_high = 0.5 * (1.0 + self.s_max_ratio)
        high = self.s_max_ratio
        consequents = (
            (high, high, high),
            (medium_high, 1.0, medium_low),
            (low, low, low),
        )

        # Zero-order Sugeno inference with product rule activation.
        weighted = 0.0
        total = 0.0
        for soc_index, soc_weight in enumerate(soc_membership):
            for demand_index, demand_weight in enumerate(demand_membership):
                activation = soc_weight * demand_weight
                weighted += activation * consequents[soc_index][demand_index]
                total += activation
        if total <= 0.0:
            raise RuntimeError("fuzzy membership partition has no active rule")
        return weighted / total

    def equivalence_factor(self, ctx: ControlContext) -> float:
        """Return the raw fuzzy equivalence factor for one control context."""
        return ctx.switching_s * self._equivalence_ratio(ctx)

    @property
    def name(self) -> str:
        """Stable label encoding all experimental fuzzy parameters."""
        return (
            f"fuzzy_soc={self.soc_low:.2f}-{self.soc_high:.2f}_"
            f"r={self.s_min_ratio:.2f}-{self.s_max_ratio:.2f}"
        )

    def reachable_range(self, switching_s: float) -> tuple[float, float]:
        """Raw equivalence-factor range over the full SoC domain."""
        switching = _finite_parameter("switching_s", switching_s)
        if switching <= 0.0:
            raise ValueError(f"switching_s must be positive, got {switching_s!r}")
        return self.s_min_ratio * switching, self.s_max_ratio * switching

    def straddles_switching(self, switching_s: float) -> bool:
        """Whether the configured consequent range brackets ``switching_s``."""
        switching = _finite_parameter("switching_s", switching_s)
        if switching <= 0.0:
            raise ValueError(f"switching_s must be positive, got {switching_s!r}")
        lower, upper = self.reachable_range(switching)
        return lower < switching < upper
