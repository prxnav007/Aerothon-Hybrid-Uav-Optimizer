"""Fixed-equivalence-factor ECMS control groups."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.control.base import (
    ControlContext,
    EMSController,
    S_ABSOLUTE_MAX,
    S_ABSOLUTE_MIN,
)

__all__ = ["FixedECMS"]


@dataclass(frozen=True)
class FixedECMS(EMSController):
    """Return either a fixed absolute factor or a fixed neutral-factor ratio."""

    s: float | None = None
    s_ratio: float | None = None

    def __post_init__(self) -> None:
        if (self.s is None) == (self.s_ratio is None):
            raise ValueError("exactly one of s and s_ratio must be set")

        field_name = "s" if self.s is not None else "s_ratio"
        value = float(getattr(self, field_name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name} must be finite and positive, got {value!r}")
        object.__setattr__(self, field_name, value)

    def equivalence_factor(self, ctx: ControlContext) -> float:
        """Return the configured factor without state-of-charge feedback."""
        if self.s is not None:
            return self.s
        return float(self.s_ratio) * ctx.neutral_s

    @property
    def name(self) -> str:
        """Stable benchmark label including the selected parameter value."""
        if self.s is not None:
            return f"fixed_s={self.s:.2f}"
        return f"fixed_ratio={self.s_ratio:.2f}"

    @classmethod
    def pure_thermal(cls) -> "FixedECMS":
        """Prefer the engine up to the absolute factor ceiling.

        The battery is not discharged while the engine can meet bus demand; it
        still supplies mandatory shortfall above the engine's lapsed rating.
        """
        return cls(s=S_ABSOLUTE_MAX)

    @classmethod
    def battery_first(cls) -> "FixedECMS":
        """Prefer the battery down to the absolute factor floor.

        The engine still runs when battery rate or energy availability binds.
        """
        return cls(s=S_ABSOLUTE_MIN)

    @classmethod
    def at_neutral(cls) -> "FixedECMS":
        """Track the operating-point neutral factor without SoC feedback."""
        return cls(s_ratio=1.0)
