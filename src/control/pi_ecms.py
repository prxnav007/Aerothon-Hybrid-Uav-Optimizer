"""Proportional state-of-charge feedback for adaptive ECMS.

The SoC law is the canonical adaptive-ECMS approximation of the battery-energy
costate from Pontryagin's Minimum Principle. Integral action is deliberately
absent; a future accumulator belongs in ``ControlContext``, never in this
stateless controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.control.base import ControlContext, EMSController

__all__ = ["PIECMS"]

# Proposed assumptions entries C-03 and C-04 are included in the task handoff.
KP_DEFAULT = 5.0
SOC_REF_DEFAULT = 0.6
S0_RATIO_DEFAULT = 1.0


def _finite_parameter(name: str, value: float) -> float:
    """Return a finite float or raise ``ValueError`` naming the parameter."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


@dataclass(frozen=True)
class PIECMS(EMSController):
    """Adaptive ECMS with proportional state-of-charge feedback.

    ``soc_ref`` sets where the feedback correction is zero; it is the
    charge-neutral point only when the selected anchor is neutral. Exactly one
    anchor is active: absolute ``s0`` or operating-point-relative ``s0_ratio``.
    The ratio default tracks the changing engine break-even point, while
    ``soc_ref`` is intended to stay fixed rather than become an optimizer gene.
    Despite the class name, this implementation has no integral term.
    """

    kp: float = KP_DEFAULT
    soc_ref: float = SOC_REF_DEFAULT
    s0: float | None = None
    s0_ratio: float | None = S0_RATIO_DEFAULT

    def __post_init__(self) -> None:
        if (self.s0 is None) == (self.s0_ratio is None):
            raise ValueError("exactly one of s0 and s0_ratio must be set")

        kp = _finite_parameter("kp", self.kp)
        soc_ref = _finite_parameter("soc_ref", self.soc_ref)
        if kp < 0.0:
            raise ValueError(f"kp must be non-negative, got {self.kp!r}")
        if not 0.0 <= soc_ref <= 1.0:
            raise ValueError(f"soc_ref must lie in [0, 1], got {self.soc_ref!r}")

        object.__setattr__(self, "kp", kp)
        object.__setattr__(self, "soc_ref", soc_ref)
        if self.s0 is not None:
            object.__setattr__(self, "s0", _finite_parameter("s0", self.s0))
        if self.s0_ratio is not None:
            object.__setattr__(
                self,
                "s0_ratio",
                _finite_parameter("s0_ratio", self.s0_ratio),
            )

    def _anchor(self, neutral_s: float) -> float:
        """Return the active anchor at one engine operating point."""
        neutral = _finite_parameter("neutral_s", neutral_s)
        if neutral <= 0.0:
            raise ValueError(f"neutral_s must be positive, got {neutral_s!r}")
        if self.s0 is not None:
            return self.s0
        ratio = self.s0_ratio
        if ratio is None:
            raise RuntimeError("validated controller has no equivalence-factor anchor")
        return ratio * neutral

    def equivalence_factor(self, ctx: ControlContext) -> float:
        """Return the raw proportional-feedback equivalence factor."""
        # Proportional SoC feedback around the selected neutral anchor.
        return self._anchor(ctx.neutral_s) + self.kp * (self.soc_ref - ctx.soc)

    @property
    def name(self) -> str:
        """Stable name encoding the feedback gain and active anchor."""
        if self.s0 is not None:
            return f"pi_kp={self.kp:.2f}_s0={self.s0:.2f}"
        ratio = self.s0_ratio
        if ratio is None:
            raise RuntimeError("validated controller has no equivalence-factor anchor")
        return f"pi_kp={self.kp:.2f}_r={ratio:.2f}"

    def reachable_range(self, neutral_s: float) -> tuple[float, float]:
        """Raw equivalence-factor range from full to empty state of charge."""
        anchor = self._anchor(neutral_s)
        at_full = anchor + self.kp * (self.soc_ref - 1.0)
        at_empty = anchor + self.kp * self.soc_ref
        return at_full, at_empty

    def straddles_neutral(self, neutral_s: float) -> bool:
        """Whether the raw SoC sweep reaches both sides of ``neutral_s``."""
        neutral = _finite_parameter("neutral_s", neutral_s)
        if neutral <= 0.0:
            raise ValueError(f"neutral_s must be positive, got {neutral_s!r}")
        lower, upper = self.reachable_range(neutral)
        return lower < neutral < upper
