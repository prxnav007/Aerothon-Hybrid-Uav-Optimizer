"""Immutable normalized chromosome for plant--thermostat optimization.

Only independent design decisions belong here. ``C_D0`` is derived from
geometry, fuel follows fixed-MTOW mass closure, and restart fuel, dwell,
uncertainty parameters and mission settings are external scenarios. The
thermostat already selects maximum feasible ON power, while GA operators belong
in ``ga.py``. No model is constructed and no simulation is imported or run.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

__all__ = [
    "CHROMOSOME_SCHEMA_VERSION",
    "DESIGN_SPACE_SCHEMA_VERSION",
    "EXCLUDED_GENE_NAMES",
    "GENE_NAMES",
    "DecodedPlantThermostatDesign",
    "NormalizedChromosome",
    "PhysicalGeneBound",
    "PlantThermostatDesignSpace",
    "decode_chromosome",
    "encode_design",
    "ideal_restart_fuel_seed",
    "practical_thermostat_seed",
]

CHROMOSOME_SCHEMA_VERSION = 1
DESIGN_SPACE_SCHEMA_VERSION = 1
GENE_NAMES = (
    "wing_area",
    "aspect_ratio",
    "engine_rating",
    "battery_capacity",
    "thermostat_soc_low",
    "thermostat_soc_gap",
)
EXCLUDED_GENE_NAMES = (
    "cd0",
    "fuel_capacity",
    "fuel_mass",
    "cruise_altitude",
    "restart_fuel",
    "minimum_on_dwell",
    "minimum_off_dwell",
    "engine_idle_fuel_fraction",
    "battery_c_rate",
    "battery_chemistry",
    "battery_specific_energy",
    "cl_max",
    "oswald_efficiency",
    "equivalence_factor",
    "thermostat_on_power",
    "mission_timestep",
    "ga_hyperparameters",
)


class _BatteryWithSocFloor(Protocol):
    soc_min: float


def _finite(name: str, value: Any) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return 0.0 if result == 0.0 else result


def _canonical_digest(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class PhysicalGeneBound:
    """Named physical interval with fixed dimensional metadata."""

    gene_name: str
    lower: float
    upper: float
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.gene_name, str) or not self.gene_name:
            raise ValueError("gene_name must be a non-empty string")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("unit must be a non-empty string")
        lower = _finite(f"{self.gene_name}.lower", self.lower)
        upper = _finite(f"{self.gene_name}.upper", self.upper)
        if not lower < upper:
            raise ValueError(
                f"{self.gene_name} lower bound must be smaller than its upper bound"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_dict(self) -> dict[str, str | float]:
        """Return stable JSON-compatible bound metadata."""
        return {
            "gene_name": self.gene_name,
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
        }


# Initial, provisional search intervals; see assumptions.md OPT-01.
_INITIAL_WING_AREA = PhysicalGeneBound("wing_area", 6.0, 16.0, "m^2")
_INITIAL_ASPECT_RATIO = PhysicalGeneBound("aspect_ratio", 10.0, 24.0, "1")
_INITIAL_ENGINE_RATING = PhysicalGeneBound("engine_rating", 60.0, 140.0, "kW")
_INITIAL_BATTERY_CAPACITY = PhysicalGeneBound("battery_capacity", 5.0, 30.0, "kWh")


@dataclass(frozen=True)
class PlantThermostatDesignSpace:
    """Explicit bounds and metadata for the six-gene design space."""

    soc_floor: float
    wing_area: PhysicalGeneBound = _INITIAL_WING_AREA
    aspect_ratio: PhysicalGeneBound = _INITIAL_ASPECT_RATIO
    engine_rating: PhysicalGeneBound = _INITIAL_ENGINE_RATING
    battery_capacity: PhysicalGeneBound = _INITIAL_BATTERY_CAPACITY
    soc_low_upper: float = 0.60
    soc_high_upper: float = 0.95
    minimum_soc_gap: float = 0.05

    def __post_init__(self) -> None:
        expected = (
            ("wing_area", self.wing_area, "m^2"),
            ("aspect_ratio", self.aspect_ratio, "1"),
            ("engine_rating", self.engine_rating, "kW"),
            ("battery_capacity", self.battery_capacity, "kWh"),
        )
        for attribute, bound, unit in expected:
            if not isinstance(bound, PhysicalGeneBound):
                raise ValueError(f"{attribute} must be a PhysicalGeneBound")
            if bound.gene_name != attribute or bound.unit != unit:
                raise ValueError(
                    f"{attribute} must describe gene {attribute!r} in unit {unit!r}"
                )

        floor = _finite("soc_floor", self.soc_floor)
        low_upper = _finite("soc_low_upper", self.soc_low_upper)
        high_upper = _finite("soc_high_upper", self.soc_high_upper)
        gap = _finite("minimum_soc_gap", self.minimum_soc_gap)
        if not 0.0 <= floor < low_upper < high_upper <= 1.0:
            raise ValueError(
                "threshold bounds must satisfy 0 <= soc_floor < soc_low_upper "
                "< soc_high_upper <= 1"
            )
        if gap <= 0.0 or low_upper + gap >= high_upper:
            raise ValueError(
                "threshold range must accommodate a positive minimum gap at "
                "the maximum lower threshold"
            )
        object.__setattr__(self, "soc_floor", floor)
        object.__setattr__(self, "soc_low_upper", low_upper)
        object.__setattr__(self, "soc_high_upper", high_upper)
        object.__setattr__(self, "minimum_soc_gap", gap)

    @classmethod
    def from_battery(
        cls,
        battery: _BatteryWithSocFloor,
        **overrides: Any,
    ) -> PlantThermostatDesignSpace:
        """Construct bounds from the supplied pack's authoritative SoC floor."""
        return cls(soc_floor=battery.soc_min, **overrides)

    @property
    def identifier(self) -> str:
        """Cross-process identity using exact hexadecimal float encodings."""
        payload = {
            "schema_version": DESIGN_SPACE_SCHEMA_VERSION,
            "gene_names": list(GENE_NAMES),
            "linear_bounds": [
                {
                    "gene_name": bound.gene_name,
                    "lower_hex": bound.lower.hex(),
                    "upper_hex": bound.upper.hex(),
                    "unit": bound.unit,
                }
                for bound in self.linear_bounds
            ],
            "threshold_bounds": {
                "soc_floor_hex": self.soc_floor.hex(),
                "soc_low_upper_hex": self.soc_low_upper.hex(),
                "soc_high_upper_hex": self.soc_high_upper.hex(),
                "minimum_soc_gap_hex": self.minimum_soc_gap.hex(),
                "unit": "1",
            },
        }
        return _canonical_digest(
            f"plant-thermostat-design-space-v{DESIGN_SPACE_SCHEMA_VERSION}", payload
        )

    @property
    def linear_bounds(self) -> tuple[PhysicalGeneBound, ...]:
        """The first four bounds in chromosome order."""
        return (
            self.wing_area,
            self.aspect_ratio,
            self.engine_rating,
            self.battery_capacity,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return complete, deterministic report metadata."""
        return {
            "schema_version": DESIGN_SPACE_SCHEMA_VERSION,
            "design_space_id": self.identifier,
            "gene_names": list(GENE_NAMES),
            "linear_bounds": [bound.to_dict() for bound in self.linear_bounds],
            "threshold_bounds": {
                "soc_floor": self.soc_floor,
                "soc_low_upper": self.soc_low_upper,
                "soc_high_upper": self.soc_high_upper,
                "minimum_soc_gap": self.minimum_soc_gap,
                "unit": "1",
                "encoding": "dependent_lower_and_gap",
            },
        }


@dataclass(frozen=True)
class DecodedPlantThermostatDesign:
    """Physical independent decisions decoded from one chromosome."""

    wing_area_m2: float
    aspect_ratio: float
    engine_rating_kw: float
    battery_capacity_kwh: float
    soc_low: float
    soc_high: float

    def __post_init__(self) -> None:
        for name in (
            "wing_area_m2",
            "aspect_ratio",
            "engine_rating_kw",
            "battery_capacity_kwh",
            "soc_low",
            "soc_high",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in (
            "wing_area_m2",
            "aspect_ratio",
            "engine_rating_kw",
            "battery_capacity_kwh",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.soc_low < self.soc_high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= soc_low < soc_high <= 1")

    def to_dict(self) -> dict[str, float]:
        """Return physical values without constructing any model."""
        return {
            "wing_area_m2": self.wing_area_m2,
            "aspect_ratio": self.aspect_ratio,
            "engine_rating_kw": self.engine_rating_kw,
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "soc_low": self.soc_low,
            "soc_high": self.soc_high,
        }


@dataclass(frozen=True)
class NormalizedChromosome:
    """Exactly six finite genes stored immutably in the closed unit interval."""

    genes: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            raw_genes: Sequence[Any] = tuple(self.genes)
        except TypeError as error:
            raise ValueError("genes must be an iterable of six real values") from error
        if len(raw_genes) != len(GENE_NAMES):
            raise ValueError("exactly six genes are required")
        genes = tuple(_finite(name, value) for name, value in zip(GENE_NAMES, raw_genes))
        for name, value in zip(GENE_NAMES, genes):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
        object.__setattr__(self, "genes", genes)

    def cache_key(self, *, bounds: PlantThermostatDesignSpace) -> str:
        """Return a versioned SHA-256 key with lossless binary64 canonicalization."""
        payload = {
            "schema_version": CHROMOSOME_SCHEMA_VERSION,
            "design_space_id": bounds.identifier,
            "gene_names": list(GENE_NAMES),
            "genes_hex": [value.hex() for value in self.genes],
        }
        return _canonical_digest(
            f"plant-thermostat-chromosome-v{CHROMOSOME_SCHEMA_VERSION}", payload
        )

    def to_dict(self, *, bounds: PlantThermostatDesignSpace) -> dict[str, Any]:
        """Return deterministic JSON-compatible normalized serialization."""
        return {
            "schema_version": CHROMOSOME_SCHEMA_VERSION,
            "gene_names": list(GENE_NAMES),
            "genes": list(self.genes),
            "design_space_id": bounds.identifier,
        }

    @classmethod
    def from_dict(
        cls,
        record: Mapping[str, Any],
        *,
        bounds: PlantThermostatDesignSpace,
    ) -> NormalizedChromosome:
        """Deserialize after validating schema, order and design-space identity."""
        if not isinstance(record, Mapping):
            raise ValueError("chromosome record must be a mapping")
        version = record.get("schema_version")
        if type(version) is not int or version != CHROMOSOME_SCHEMA_VERSION:
            raise ValueError(f"unsupported chromosome schema version {version!r}")
        if tuple(record.get("gene_names", ())) != GENE_NAMES:
            raise ValueError("serialized gene order does not match the stable schema")
        if record.get("design_space_id") != bounds.identifier:
            raise ValueError("serialized design-space identity does not match bounds")
        genes = record.get("genes")
        if isinstance(genes, (str, bytes)) or not isinstance(genes, Sequence):
            raise ValueError("serialized genes must be a six-value sequence")
        return cls(genes=tuple(genes))


def _decode_linear(value: float, bound: PhysicalGeneBound) -> float:
    if value == 0.0:
        return bound.lower
    if value == 1.0:
        return bound.upper
    return bound.lower + value * (bound.upper - bound.lower)


def decode_chromosome(
    chromosome: NormalizedChromosome,
    *,
    bounds: PlantThermostatDesignSpace,
) -> DecodedPlantThermostatDesign:
    """Decode normalized genes without clipping or evaluating physics."""
    if not isinstance(chromosome, NormalizedChromosome):
        raise ValueError("chromosome must be a NormalizedChromosome")
    wing, aspect, engine, battery, low_gene, gap_gene = chromosome.genes
    linear = tuple(
        _decode_linear(value, bound)
        for value, bound in zip((wing, aspect, engine, battery), bounds.linear_bounds)
    )
    soc_low = _decode_linear(
        low_gene,
        PhysicalGeneBound(
            "thermostat_soc_low", bounds.soc_floor, bounds.soc_low_upper, "1"
        ),
    )
    if gap_gene == 0.0:
        soc_high = soc_low + bounds.minimum_soc_gap
    elif gap_gene == 1.0:
        soc_high = bounds.soc_high_upper
    else:
        available_gap = bounds.soc_high_upper - soc_low - bounds.minimum_soc_gap
        soc_high = soc_low + bounds.minimum_soc_gap + gap_gene * available_gap
    return DecodedPlantThermostatDesign(
        wing_area_m2=linear[0],
        aspect_ratio=linear[1],
        engine_rating_kw=linear[2],
        battery_capacity_kwh=linear[3],
        soc_low=soc_low,
        soc_high=soc_high,
    )


def _encode_linear(name: str, value: float, bound: PhysicalGeneBound) -> float:
    if not bound.lower <= value <= bound.upper:
        raise ValueError(
            f"{name} must lie in [{bound.lower}, {bound.upper}] {bound.unit}"
        )
    if value == bound.lower:
        return 0.0
    if value == bound.upper:
        return 1.0
    return (value - bound.lower) / (bound.upper - bound.lower)


def encode_design(
    design: DecodedPlantThermostatDesign,
    *,
    bounds: PlantThermostatDesignSpace,
) -> NormalizedChromosome:
    """Apply the exact inverse transform, rejecting invalid physical designs."""
    if not isinstance(design, DecodedPlantThermostatDesign):
        raise ValueError("design must be a DecodedPlantThermostatDesign")
    linear_values = (
        design.wing_area_m2,
        design.aspect_ratio,
        design.engine_rating_kw,
        design.battery_capacity_kwh,
    )
    normalized = [
        _encode_linear(name, value, bound)
        for name, value, bound in zip(GENE_NAMES[:4], linear_values, bounds.linear_bounds)
    ]
    if not bounds.soc_floor <= design.soc_low <= bounds.soc_low_upper:
        raise ValueError(
            f"soc_low must lie in [{bounds.soc_floor}, {bounds.soc_low_upper}]"
        )
    if design.soc_high > bounds.soc_high_upper:
        raise ValueError(f"soc_high must not exceed {bounds.soc_high_upper}")
    physical_gap = design.soc_high - design.soc_low
    if physical_gap < bounds.minimum_soc_gap:
        raise ValueError(
            f"soc_high - soc_low must be at least {bounds.minimum_soc_gap}"
        )
    low_gene = _encode_linear(
        "thermostat_soc_low",
        design.soc_low,
        PhysicalGeneBound(
            "thermostat_soc_low", bounds.soc_floor, bounds.soc_low_upper, "1"
        ),
    )
    available_gap = bounds.soc_high_upper - design.soc_low - bounds.minimum_soc_gap
    if design.soc_high == design.soc_low + bounds.minimum_soc_gap:
        gap_gene = 0.0
    elif design.soc_high == bounds.soc_high_upper:
        gap_gene = 1.0
    else:
        gap_gene = (physical_gap - bounds.minimum_soc_gap) / available_gap
    return NormalizedChromosome(genes=tuple((*normalized, low_gene, gap_gene)))


# Exact stored band-aircraft values, not claims of optimality; see OPT-01.
_REFERENCE_WING_AREA_M2 = 7.59175537062125
_REFERENCE_ASPECT_RATIO = 16.0
_REFERENCE_ENGINE_RATING_KW = 86.7791369750147
_REFERENCE_BATTERY_CAPACITY_KWH = 10.0


def practical_thermostat_seed(
    *, bounds: PlantThermostatDesignSpace
) -> NormalizedChromosome:
    """Encode the locally retuned 0.1 kg/start thermostat anchor."""
    return encode_design(
        DecodedPlantThermostatDesign(
            wing_area_m2=_REFERENCE_WING_AREA_M2,
            aspect_ratio=_REFERENCE_ASPECT_RATIO,
            engine_rating_kw=_REFERENCE_ENGINE_RATING_KW,
            battery_capacity_kwh=_REFERENCE_BATTERY_CAPACITY_KWH,
            soc_low=0.225,
            soc_high=0.350,
        ),
        bounds=bounds,
    )


def ideal_restart_fuel_seed(
    *, bounds: PlantThermostatDesignSpace
) -> NormalizedChromosome:
    """Encode the selected zero-restart-fuel thermostat anchor."""
    return encode_design(
        DecodedPlantThermostatDesign(
            wing_area_m2=_REFERENCE_WING_AREA_M2,
            aspect_ratio=_REFERENCE_ASPECT_RATIO,
            engine_rating_kw=_REFERENCE_ENGINE_RATING_KW,
            battery_capacity_kwh=_REFERENCE_BATTERY_CAPACITY_KWH,
            soc_low=0.225,
            soc_high=0.300,
        ),
        bounds=bounds,
    )
