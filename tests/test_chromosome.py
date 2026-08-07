"""Focused contracts for the normalized plant--thermostat chromosome."""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest

from src.models.battery import BatteryPack
from src.optimization.chromosome import (
    CHROMOSOME_SCHEMA_VERSION,
    EXCLUDED_GENE_NAMES,
    GENE_NAMES,
    DecodedPlantThermostatDesign,
    NormalizedChromosome,
    PhysicalGeneBound,
    PlantThermostatDesignSpace,
    decode_chromosome,
    encode_design,
    ideal_restart_fuel_seed,
    practical_thermostat_seed,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PRACTICAL_GENES = (
    0.159175537062125,
    0.42857142857142855,
    0.33473921218768365,
    0.2,
    0.31818181818181818,
    0.11111111111111108,
)
EXPECTED_IDEAL_GENES = (*EXPECTED_PRACTICAL_GENES[:5], 0.037037037037037014)


@pytest.fixture
def design_space() -> PlantThermostatDesignSpace:
    return PlantThermostatDesignSpace.from_battery(BatteryPack(10.0))


@pytest.mark.parametrize("count", [0, 1, 5, 7])
def test_exactly_six_genes_are_required(count: int) -> None:
    with pytest.raises(ValueError, match="exactly six genes"):
        NormalizedChromosome(genes=(0.5,) * count)


def test_gene_order_is_stable_and_serialized_explicitly(
    design_space: PlantThermostatDesignSpace,
) -> None:
    expected = (
        "wing_area",
        "aspect_ratio",
        "engine_rating",
        "battery_capacity",
        "thermostat_soc_low",
        "thermostat_soc_gap",
    )
    assert GENE_NAMES == expected
    record = NormalizedChromosome((0.5,) * 6).to_dict(bounds=design_space)
    assert tuple(record["gene_names"]) == expected


@pytest.mark.parametrize("index,value", [(0, -1.0e-12), (5, 1.0 + 1.0e-12)])
def test_normalized_values_outside_the_unit_interval_are_rejected(
    index: int, value: float
) -> None:
    genes = [0.5] * 6
    genes[index] = value
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NormalizedChromosome(tuple(genes))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_normalized_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        NormalizedChromosome((value, 0.5, 0.5, 0.5, 0.5, 0.5))


def test_linear_decoding_reaches_exact_plant_bounds(
    design_space: PlantThermostatDesignSpace,
) -> None:
    lower = decode_chromosome(NormalizedChromosome((0.0,) * 6), bounds=design_space)
    upper = decode_chromosome(NormalizedChromosome((1.0,) * 6), bounds=design_space)
    assert lower.to_dict() == {
        "wing_area_m2": 6.0,
        "aspect_ratio": 10.0,
        "engine_rating_kw": 60.0,
        "battery_capacity_kwh": 5.0,
        "soc_low": 0.05,
        "soc_high": 0.10,
    }
    assert upper.wing_area_m2 == 16.0
    assert upper.aspect_ratio == 24.0
    assert upper.engine_rating_kw == 140.0
    assert upper.battery_capacity_kwh == 30.0


def test_threshold_decoding_satisfies_every_invariant_over_a_dense_grid(
    design_space: PlantThermostatDesignSpace,
) -> None:
    samples = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0)
    for low_gene, gap_gene in product(samples, repeat=2):
        chromosome = NormalizedChromosome((0.5, 0.5, 0.5, 0.5, low_gene, gap_gene))
        design = decode_chromosome(chromosome, bounds=design_space)
        assert design_space.soc_floor <= design.soc_low <= design_space.soc_low_upper
        assert design.soc_low < design.soc_high <= design_space.soc_high_upper
        assert design.soc_high - design.soc_low + 1.0e-15 >= design_space.minimum_soc_gap


@pytest.mark.parametrize(
    "low_gene,gap_gene,expected",
    [
        (0.0, 0.0, (0.05, 0.10)),
        (0.0, 1.0, (0.05, 0.95)),
        (1.0, 0.0, (0.60, 0.65)),
        (1.0, 1.0, (0.60, 0.95)),
    ],
)
def test_threshold_boundary_gene_pairs_map_to_the_defined_edges(
    design_space: PlantThermostatDesignSpace,
    low_gene: float,
    gap_gene: float,
    expected: tuple[float, float],
) -> None:
    chromosome = NormalizedChromosome((0.5, 0.5, 0.5, 0.5, low_gene, gap_gene))
    design = decode_chromosome(chromosome, bounds=design_space)
    assert (design.soc_low, design.soc_high) == pytest.approx(expected, abs=1.0e-15)


def test_physical_to_normalized_to_physical_round_trip(
    design_space: PlantThermostatDesignSpace,
) -> None:
    original = DecodedPlantThermostatDesign(
        wing_area_m2=11.25,
        aspect_ratio=17.5,
        engine_rating_kw=93.25,
        battery_capacity_kwh=18.75,
        soc_low=0.275,
        soc_high=0.725,
    )
    recovered = decode_chromosome(encode_design(original, bounds=design_space), bounds=design_space)
    for name, value in original.to_dict().items():
        assert recovered.to_dict()[name] == pytest.approx(value, abs=1.0e-14)


@pytest.mark.parametrize(
    "genes",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (0.123456789, 0.75, 0.333, 0.9, 0.42, 0.81),
    ],
)
def test_normalized_to_physical_to_normalized_round_trip(
    design_space: PlantThermostatDesignSpace,
    genes: tuple[float, ...],
) -> None:
    original = NormalizedChromosome(genes)
    recovered = encode_design(
        decode_chromosome(original, bounds=design_space), bounds=design_space
    )
    assert recovered.genes == pytest.approx(original.genes, abs=1.0e-14)


@pytest.mark.parametrize(
    "low,high,message",
    [
        (0.04, 0.20, "soc_low"),
        (0.61, 0.80, "soc_low"),
        (0.30, 0.34, "at least"),
        (0.30, 0.96, "soc_high"),
    ],
)
def test_invalid_physical_threshold_pairs_are_rejected_without_repair(
    design_space: PlantThermostatDesignSpace,
    low: float,
    high: float,
    message: str,
) -> None:
    design = DecodedPlantThermostatDesign(10.0, 16.0, 90.0, 10.0, low, high)
    with pytest.raises(ValueError, match=message):
        encode_design(design, bounds=design_space)


def test_reversed_physical_thresholds_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="soc_low < soc_high"):
        DecodedPlantThermostatDesign(10.0, 16.0, 90.0, 10.0, 0.4, 0.4)


def test_practical_reference_seed_uses_authoritative_plant_precision(
    design_space: PlantThermostatDesignSpace,
) -> None:
    seed = practical_thermostat_seed(bounds=design_space)
    assert seed.genes == pytest.approx(EXPECTED_PRACTICAL_GENES, abs=1.0e-15)
    design = decode_chromosome(seed, bounds=design_space)
    assert design.to_dict() == pytest.approx(
        {
            "wing_area_m2": 7.59175537062125,
            "aspect_ratio": 16.0,
            "engine_rating_kw": 86.7791369750147,
            "battery_capacity_kwh": 10.0,
            "soc_low": 0.225,
            "soc_high": 0.350,
        },
        abs=1.0e-14,
    )


def test_ideal_restart_fuel_seed_decodes_to_the_selected_design(
    design_space: PlantThermostatDesignSpace,
) -> None:
    seed = ideal_restart_fuel_seed(bounds=design_space)
    assert seed.genes == pytest.approx(EXPECTED_IDEAL_GENES, abs=1.0e-15)
    design = decode_chromosome(seed, bounds=design_space)
    assert design.soc_low == pytest.approx(0.225, abs=1.0e-15)
    assert design.soc_high == pytest.approx(0.300, abs=1.0e-15)


def test_cache_key_is_deterministic_and_includes_design_space_identity(
    design_space: PlantThermostatDesignSpace,
) -> None:
    chromosome = NormalizedChromosome((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    same_bounds = PlantThermostatDesignSpace(soc_floor=0.05)
    changed_bounds = PlantThermostatDesignSpace(soc_floor=0.06)
    assert chromosome.cache_key(bounds=design_space) == chromosome.cache_key(
        bounds=same_bounds
    )
    assert chromosome.cache_key(bounds=design_space) != chromosome.cache_key(
        bounds=changed_bounds
    )


def test_cache_key_is_identical_in_a_fresh_python_process(
    design_space: PlantThermostatDesignSpace,
) -> None:
    chromosome = NormalizedChromosome((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    code = (
        "from src.optimization.chromosome import NormalizedChromosome, "
        "PlantThermostatDesignSpace; "
        "c=NormalizedChromosome((0.1,0.2,0.3,0.4,0.5,0.6)); "
        "b=PlantThermostatDesignSpace(soc_floor=0.05); "
        "print(c.cache_key(bounds=b))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == chromosome.cache_key(bounds=design_space)


def test_cache_keys_distinguish_adjacent_binary64_gene_values(
    design_space: PlantThermostatDesignSpace,
) -> None:
    nearby = math.nextafter(0.5, 1.0)
    first = NormalizedChromosome((0.5,) * 6)
    second = NormalizedChromosome((nearby, 0.5, 0.5, 0.5, 0.5, 0.5))
    assert first.cache_key(bounds=design_space) != second.cache_key(bounds=design_space)


def test_serialization_round_trip_is_json_compatible_and_deterministic(
    design_space: PlantThermostatDesignSpace,
) -> None:
    original = practical_thermostat_seed(bounds=design_space)
    record = original.to_dict(bounds=design_space)
    serialized = json.dumps(record, separators=(",", ":"), sort_keys=False)
    recovered = NormalizedChromosome.from_dict(json.loads(serialized), bounds=design_space)
    assert recovered == original
    assert recovered.to_dict(bounds=design_space) == record


def test_unknown_schema_version_is_rejected(
    design_space: PlantThermostatDesignSpace,
) -> None:
    record = practical_thermostat_seed(bounds=design_space).to_dict(bounds=design_space)
    record["schema_version"] = CHROMOSOME_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="unsupported chromosome schema version"):
        NormalizedChromosome.from_dict(record, bounds=design_space)


def test_bounds_use_the_actual_configured_battery_soc_floor() -> None:
    battery = BatteryPack(10.0, soc_min=0.12)
    bounds = PlantThermostatDesignSpace.from_battery(battery)
    assert bounds.soc_floor == 0.12
    assert decode_chromosome(NormalizedChromosome((0.0,) * 6), bounds=bounds).soc_low == 0.12


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"wing_area": PhysicalGeneBound("wing_area", 6.0, 16.0, "kW")}, "unit"),
        ({"soc_low_upper": math.inf}, "finite"),
        ({"soc_low_upper": 0.92, "soc_high_upper": 0.95}, "accommodate"),
        ({"minimum_soc_gap": 0.0}, "positive minimum gap"),
    ],
)
def test_inconsistent_or_impossible_design_space_metadata_is_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PlantThermostatDesignSpace(soc_floor=0.05, **kwargs)


def test_reversed_and_nonfinite_physical_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="smaller"):
        PhysicalGeneBound("wing_area", 16.0, 6.0, "m^2")
    with pytest.raises(ValueError, match="finite"):
        PhysicalGeneBound("wing_area", 6.0, math.nan, "m^2")


def test_chromosome_and_decoded_design_are_immutable(
    design_space: PlantThermostatDesignSpace,
) -> None:
    chromosome = practical_thermostat_seed(bounds=design_space)
    design = decode_chromosome(chromosome, bounds=design_space)
    with pytest.raises(dataclasses.FrozenInstanceError):
        chromosome.genes = (0.5,) * 6
    with pytest.raises(dataclasses.FrozenInstanceError):
        design.soc_low = 0.3


def test_explicit_non_genes_exclude_derived_and_scenario_quantities() -> None:
    required = {
        "cd0",
        "fuel_mass",
        "restart_fuel",
        "minimum_on_dwell",
        "oswald_efficiency",
        "thermostat_on_power",
        "mission_timestep",
        "ga_hyperparameters",
    }
    assert required <= set(EXCLUDED_GENE_NAMES)
    assert required.isdisjoint(GENE_NAMES)


def test_module_import_does_not_import_the_mission_simulator() -> None:
    code = (
        "import sys; import src.optimization.chromosome; "
        "print('src.simulation.simulator' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"
