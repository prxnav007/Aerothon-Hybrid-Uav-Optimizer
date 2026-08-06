"""Immutable mission-profile data for the time-marching simulator.

This module describes phase targets, termination conditions and mission-level
reserves.  It deliberately contains no aircraft physics or flyability checks;
those belong to ``simulator.py`` and ``feasibility.py`` respectively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "AltitudeMode",
    "DESCENT_LANDING_FUEL_KG",
    "MissionProfile",
    "Phase",
    "SpeedMode",
    "Termination",
    "ps1_mission",
]

DESCENT_LANDING_FUEL_KG = 4.7  # measured with margin; see assumptions.md S-04


class SpeedMode(Enum):
    """How the simulator obtains the airspeed target for a phase."""

    FIXED = "fixed"
    MIN_POWER = "min_power"
    BEST_LD = "best_ld"


class AltitudeMode(Enum):
    """How altitude changes during a phase."""

    HOLD = "hold"
    CLIMB_TO = "climb_to"
    DESCEND_TO = "descend_to"


class Termination(Enum):
    """Condition that ends a phase."""

    DURATION = "duration"
    ALTITUDE = "altitude"
    RESOURCE = "resource"


def _is_finite(value: object) -> bool:
    """Whether ``value`` can be represented as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Phase:
    """One internally consistent segment of a mission profile."""

    name: str
    speed_mode: SpeedMode
    altitude_mode: AltitudeMode
    termination: Termination
    target_altitude_m: float
    speed_mps: float | None = None
    climb_rate_mps: float | None = None
    duration_s: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when the phase is internally inconsistent."""
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError(f"name must be a non-empty identifier, got {self.name!r}")
        if not isinstance(self.speed_mode, SpeedMode):
            raise ValueError(f"speed_mode must be a SpeedMode, got {self.speed_mode!r}")
        if not isinstance(self.altitude_mode, AltitudeMode):
            raise ValueError(
                f"altitude_mode must be an AltitudeMode, got {self.altitude_mode!r}"
            )
        if not isinstance(self.termination, Termination):
            raise ValueError(
                f"termination must be a Termination, got {self.termination!r}"
            )
        if not _is_finite(self.target_altitude_m) or self.target_altitude_m < 0.0:
            raise ValueError(
                "target_altitude_m must be finite and non-negative, "
                f"got {self.target_altitude_m!r}"
            )

        if self.speed_mode is SpeedMode.FIXED:
            if not _is_finite(self.speed_mps) or float(self.speed_mps) <= 0.0:
                raise ValueError(
                    f"speed_mps must be finite and positive for FIXED speed, got {self.speed_mps!r}"
                )
        elif self.speed_mps is not None:
            raise ValueError(
                f"speed_mps must be unset when speed_mode is {self.speed_mode.name}, "
                f"got {self.speed_mps!r}"
            )

        if self.altitude_mode is AltitudeMode.HOLD:
            if self.climb_rate_mps is not None:
                raise ValueError(
                    "climb_rate_mps must be unset when altitude_mode is HOLD, "
                    f"got {self.climb_rate_mps!r}"
                )
        else:
            if not _is_finite(self.climb_rate_mps) or float(self.climb_rate_mps) == 0.0:
                raise ValueError(
                    "climb_rate_mps must be finite and non-zero when altitude changes, "
                    f"got {self.climb_rate_mps!r}"
                )
            if (
                self.altitude_mode is AltitudeMode.CLIMB_TO
                and float(self.climb_rate_mps) < 0.0
            ):
                raise ValueError(
                    f"climb_rate_mps must be positive for CLIMB_TO, got {self.climb_rate_mps!r}"
                )
            if (
                self.altitude_mode is AltitudeMode.DESCEND_TO
                and float(self.climb_rate_mps) > 0.0
            ):
                raise ValueError(
                    f"climb_rate_mps must be negative for DESCEND_TO, got {self.climb_rate_mps!r}"
                )

        if self.termination is Termination.DURATION:
            if not _is_finite(self.duration_s) or float(self.duration_s) <= 0.0:
                raise ValueError(
                    "duration_s must be finite and positive for DURATION termination, "
                    f"got {self.duration_s!r}"
                )
        elif self.duration_s is not None:
            raise ValueError(
                f"duration_s must be unset when termination is {self.termination.name}, "
                f"got {self.duration_s!r}"
            )


@dataclass(frozen=True)
class MissionProfile:
    """Ordered phases and resource limits for one complete mission."""

    phases: tuple[Phase, ...]
    initial_altitude_m: float
    fuel_reserve_kg: float
    descent_landing_fuel_kg: float
    min_usable_fuel_kg: float
    max_mission_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when the profile is internally inconsistent."""
        if not self.phases:
            raise ValueError("phases must contain at least one phase")
        for phase in self.phases:
            if not isinstance(phase, Phase):
                raise ValueError(f"phases must contain only Phase instances, got {phase!r}")
            phase.validate()

        names = self.phase_names
        if len(set(names)) != len(names):
            raise ValueError(f"phase name values must be unique, got {names!r}")
        if not _is_finite(self.initial_altitude_m) or self.initial_altitude_m < 0.0:
            raise ValueError(
                "initial_altitude_m must be finite and non-negative, "
                f"got {self.initial_altitude_m!r}"
            )
        for field_name in (
            "fuel_reserve_kg",
            "max_mission_time_s",
            "min_usable_fuel_kg",
        ):
            value = getattr(self, field_name)
            if not _is_finite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive, got {value!r}")
        if (
            not _is_finite(self.descent_landing_fuel_kg)
            or self.descent_landing_fuel_kg < 0.0
        ):
            raise ValueError(
                "descent_landing_fuel_kg must be finite and non-negative, "
                f"got {self.descent_landing_fuel_kg!r}"
            )
        if self.fuel_reserve_kg >= self.min_usable_fuel_kg:
            raise ValueError(
                "fuel_reserve_kg must be less than min_usable_fuel_kg, "
                f"got {self.fuel_reserve_kg!r} and {self.min_usable_fuel_kg!r}"
            )

        resource_indices = tuple(
            index
            for index, phase in enumerate(self.phases)
            if phase.termination is Termination.RESOURCE
        )
        if len(resource_indices) != 1:
            raise ValueError(
                "phases must contain exactly one RESOURCE termination, "
                f"got {len(resource_indices)}"
            )
        resource_index = resource_indices[0]
        if resource_index == len(self.phases) - 1:
            raise ValueError("RESOURCE phase must not be the last phase")
        if not any(
            phase.altitude_mode is AltitudeMode.DESCEND_TO
            and phase.target_altitude_m == 0.0
            for phase in self.phases[resource_index + 1 :]
        ):
            raise ValueError(
                "RESOURCE phase must be followed by a DESCEND_TO phase targeting ground"
            )

        previous_name = "initial_altitude_m"
        start_altitude_m = self.initial_altitude_m
        for phase in self.phases:
            if phase.altitude_mode is AltitudeMode.HOLD:
                if phase.target_altitude_m != start_altitude_m:
                    raise ValueError(
                        f"altitude discontinuity between {previous_name!r} and {phase.name!r}: "
                        f"HOLD target_altitude_m {phase.target_altitude_m!r} does not equal "
                        f"starting altitude {start_altitude_m!r}"
                    )
            elif phase.altitude_mode is AltitudeMode.CLIMB_TO:
                if phase.target_altitude_m <= start_altitude_m:
                    raise ValueError(
                        f"target_altitude_m for phase {phase.name!r} must exceed the altitude "
                        f"after {previous_name!r}"
                    )
            elif phase.target_altitude_m >= start_altitude_m:
                raise ValueError(
                    f"target_altitude_m for phase {phase.name!r} must be below the altitude "
                    f"after {previous_name!r}"
                )
            start_altitude_m = phase.target_altitude_m
            previous_name = phase.name

        if self.phases[-1].target_altitude_m != 0.0:
            raise ValueError(
                "final phase target_altitude_m must be zero, "
                f"got {self.phases[-1].target_altitude_m!r}"
            )

    @property
    def endurance_phase_index(self) -> int:
        """Index of the unique resource-terminated endurance phase."""
        return next(
            index
            for index, phase in enumerate(self.phases)
            if phase.termination is Termination.RESOURCE
        )

    @property
    def loiter_fuel_floor_kg(self) -> float:
        """Fuel retained at loiter exit for the remaining mission and reserve."""
        return self.descent_landing_fuel_kg + self.fuel_reserve_kg

    @property
    def phase_names(self) -> tuple[str, ...]:
        """Phase names in mission order."""
        return tuple(phase.name for phase in self.phases)

    def phase_by_name(self, name: str) -> Phase:
        """Return a phase by name or raise ``KeyError`` when it is absent."""
        for phase in self.phases:
            if phase.name == name:
                return phase
        raise KeyError(f"unknown phase name {name!r}")


def ps1_mission(
    *,
    cruise_altitude_m: float = 3000.0,
    cruise_speed_mps: float = 250.0 * 1000.0 / 3600.0,
    loiter_speed_mode: SpeedMode = SpeedMode.MIN_POWER,
    loiter_speed_mps: float | None = None,
    climb_rate_mps: float = 2.0,
    descent_rate_mps: float = 3.0,
    initial_altitude_m: float = 0.0,
    takeoff_speed_mps: float = 50.0,
    climb_speed_mps: float = 65.0,
    descent_speed_mps: float = 65.0,
    landing_speed_mps: float = 45.0,
    takeoff_duration_s: float = 120.0,
    cruise_duration_s: float = 3600.0,
    landing_duration_s: float = 120.0,
    fuel_reserve_kg: float = 5.0,
    descent_landing_fuel_kg: float = DESCENT_LANDING_FUEL_KG,
    min_usable_fuel_kg: float = 20.0,
    max_mission_time_s: float = 24.0 * 3600.0,
) -> MissionProfile:
    """Build the six-phase Aerothon PS1 mission profile."""
    if not _is_finite(descent_rate_mps) or descent_rate_mps <= 0.0:
        raise ValueError(
            "descent_rate_mps must be a finite positive downward-rate magnitude, "
            f"got {descent_rate_mps!r}"
        )

    phases = (
        Phase(
            name="takeoff",
            speed_mode=SpeedMode.FIXED,
            altitude_mode=AltitudeMode.HOLD,
            termination=Termination.DURATION,
            target_altitude_m=initial_altitude_m,
            speed_mps=takeoff_speed_mps,
            duration_s=takeoff_duration_s,
        ),
        Phase(
            name="climb",
            speed_mode=SpeedMode.FIXED,
            altitude_mode=AltitudeMode.CLIMB_TO,
            termination=Termination.ALTITUDE,
            target_altitude_m=cruise_altitude_m,
            speed_mps=climb_speed_mps,
            climb_rate_mps=climb_rate_mps,
        ),
        Phase(
            name="cruise",
            speed_mode=SpeedMode.FIXED,
            altitude_mode=AltitudeMode.HOLD,
            termination=Termination.DURATION,
            target_altitude_m=cruise_altitude_m,
            speed_mps=cruise_speed_mps,
            duration_s=cruise_duration_s,
        ),
        Phase(
            name="loiter",
            speed_mode=loiter_speed_mode,
            altitude_mode=AltitudeMode.HOLD,
            termination=Termination.RESOURCE,
            target_altitude_m=cruise_altitude_m,
            speed_mps=loiter_speed_mps,
        ),
        Phase(
            name="descent",
            speed_mode=SpeedMode.FIXED,
            altitude_mode=AltitudeMode.DESCEND_TO,
            termination=Termination.ALTITUDE,
            target_altitude_m=0.0,
            speed_mps=descent_speed_mps,
            climb_rate_mps=-float(descent_rate_mps),
        ),
        Phase(
            name="landing",
            speed_mode=SpeedMode.FIXED,
            altitude_mode=AltitudeMode.HOLD,
            termination=Termination.DURATION,
            target_altitude_m=0.0,
            speed_mps=landing_speed_mps,
            duration_s=landing_duration_s,
        ),
    )
    return MissionProfile(
        phases=phases,
        initial_altitude_m=initial_altitude_m,
        fuel_reserve_kg=fuel_reserve_kg,
        descent_landing_fuel_kg=descent_landing_fuel_kg,
        min_usable_fuel_kg=min_usable_fuel_kg,
        max_mission_time_s=max_mission_time_s,
    )
