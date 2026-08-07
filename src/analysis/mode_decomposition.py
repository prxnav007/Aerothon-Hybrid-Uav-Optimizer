"""Energy-ledger decomposition of logged engine and battery operating modes."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.models.battery import BatteryPack
from src.simulation.simulator import TimeStep

__all__ = [
    "ModeDecomposition",
    "ModeEnergy",
    "PostCrossingWindow",
    "decompose_modes",
    "select_post_crossing_window",
    "write_mode_decomposition_csv",
    "write_mode_decompositions_csv",
]


@dataclass(frozen=True)
class PostCrossingWindow:
    """Logged loiter steps strictly after the first discharge-feasible endpoint."""

    crossing_step: TimeStep
    steps: tuple[TimeStep, ...]
    initial_soc: float


@dataclass(frozen=True)
class ModeEnergy:
    """Integrated time and signed battery flows for one exclusive mode."""

    mode: str
    point_count: int
    elapsed_h: float
    time_fraction: float
    battery_bus_energy_kwh: float
    battery_internal_energy_kwh: float
    ohmic_loss_kwh: float
    battery_endpoint_energy_change_kwh: float


@dataclass(frozen=True)
class ModeDecomposition:
    """Four-mode ledger and energy-derived OFF-time allocation."""

    modes: tuple[ModeEnergy, ...]
    battery_mode: str
    timestep_s: float
    neutral_band_kw: float
    window_duration_h: float
    initial_soc: float
    terminal_soc: float
    delta_soc: float
    endpoint_battery_energy_change_kwh: float
    internal_ledger_energy_change_kwh: float
    euler_ledger_residual_kwh: float
    charge_bus_energy_in_kwh: float
    discharge_bus_energy_out_kwh: float
    stored_charge_energy_kwh: float
    stored_discharge_energy_kwh: float
    recirculated_internal_energy_kwh: float
    recirculated_charge_bus_energy_kwh: float
    recirculated_discharge_bus_energy_kwh: float
    recirculated_round_trip_loss_kwh: float
    engine_off_total_h: float
    engine_off_cyclic_h: float
    engine_off_depletion_h: float
    engine_off_total_fraction: float
    engine_off_cyclic_fraction: float
    engine_off_depletion_fraction: float
    off_time_allocation: str
    endpoint_stored_charge_energy_kwh: float
    endpoint_stored_discharge_energy_kwh: float
    endpoint_recirculated_energy_kwh: float
    engine_off_cyclic_endpoint_h: float
    engine_off_depletion_endpoint_h: float
    engine_off_cyclic_endpoint_fraction: float
    engine_off_depletion_endpoint_fraction: float
    cyclic_off_fraction_uncertainty_low: float
    cyclic_off_fraction_uncertainty_high: float
    cyclic_off_fraction_uncertainty_width: float


_MODE_ORDER = (
    "on_charging",
    "on_battery_neutral",
    "on_battery_assisting",
    "off_discharging",
)


def select_post_crossing_window(
    log: Sequence[TimeStep],
    discharge_limit_kw: float,
    *,
    phase: str = "loiter",
    tolerance_kw: float = 1.0e-10,
) -> PostCrossingWindow:
    """Select steps after the logged endpoint where demand first meets the cap."""
    limit = float(discharge_limit_kw)
    tolerance = float(tolerance_kw)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("discharge_limit_kw must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_kw must be finite and non-negative")
    phase_steps = tuple(step for step in log if step.phase == phase)
    if not phase_steps:
        raise ValueError(f"log contains no {phase!r} steps")
    try:
        crossing_index = next(
            index
            for index, step in enumerate(phase_steps)
            if step.bus_demand_kw <= limit + tolerance
        )
    except StopIteration as error:
        raise ValueError("discharge-feasibility crossing is absent from log") from error
    crossing = phase_steps[crossing_index]
    steps = phase_steps[crossing_index + 1 :]
    if not steps:
        raise ValueError("no complete logged steps remain after the crossing")
    return PostCrossingWindow(crossing, steps, crossing.soc)


def _mode_for(step: TimeStep, neutral_band_kw: float) -> str:
    power = step.battery_bus_kw
    if step.engine_shut_down:
        if power <= neutral_band_kw:
            raise ValueError("engine-OFF step is not materially discharging")
        return "off_discharging"
    if power < -neutral_band_kw:
        return "on_charging"
    if power > neutral_band_kw:
        return "on_battery_assisting"
    return "on_battery_neutral"


def decompose_modes(
    steps: Sequence[TimeStep],
    battery: BatteryPack,
    *,
    initial_soc: float,
    neutral_band_kw: float = 1.0e-6,
) -> ModeDecomposition:
    """Integrate four exclusive modes and split OFF time on an energy basis."""
    entries = tuple(steps)
    band = float(neutral_band_kw)
    start_soc = float(initial_soc)
    if not entries:
        raise ValueError("steps must not be empty")
    if not math.isfinite(band) or band < 0.0:
        raise ValueError("neutral_band_kw must be finite and non-negative")
    if not 0.0 <= start_soc <= 1.0:
        raise ValueError("initial_soc must lie in [0, 1]")

    duration_h = sum(step.dt_s for step in entries) / 3600.0
    accumulators = {
        mode: [0, 0.0, 0.0, 0.0, 0.0, 0.0] for mode in _MODE_ORDER
    }
    for step in entries:
        mode = _mode_for(step, band)
        values = accumulators[mode]
        duration = step.dt_s / 3600.0
        values[0] += 1
        values[1] += duration
        values[2] += step.battery_bus_kw * duration
        values[3] += step.battery_internal_kw * duration
        values[4] += step.battery_ohmic_loss_kw * duration
        values[5] += step.battery_stored_energy_change_kwh
    modes = tuple(
        ModeEnergy(
            mode=mode,
            point_count=int(values[0]),
            elapsed_h=values[1],
            time_fraction=values[1] / duration_h,
            battery_bus_energy_kwh=values[2],
            battery_internal_energy_kwh=values[3],
            ohmic_loss_kwh=values[4],
            battery_endpoint_energy_change_kwh=values[5],
        )
        for mode, values in accumulators.items()
    )

    charge_bus = -sum(min(row.battery_bus_energy_kwh, 0.0) for row in modes)
    discharge_bus = sum(max(row.battery_bus_energy_kwh, 0.0) for row in modes)
    stored_charge = -sum(
        min(row.battery_internal_energy_kwh, 0.0) for row in modes
    )
    stored_discharge = sum(
        max(row.battery_internal_energy_kwh, 0.0) for row in modes
    )
    recirculated = min(stored_charge, stored_discharge)
    endpoint_stored_charge = sum(
        max(row.battery_endpoint_energy_change_kwh, 0.0) for row in modes
    )
    endpoint_stored_discharge = -sum(
        min(row.battery_endpoint_energy_change_kwh, 0.0) for row in modes
    )
    endpoint_recirculated = min(
        endpoint_stored_charge, endpoint_stored_discharge
    )
    recirculated_charge_bus = (
        charge_bus * recirculated / stored_charge if stored_charge > 0.0 else 0.0
    )
    recirculated_discharge_bus = (
        discharge_bus * recirculated / stored_discharge
        if stored_discharge > 0.0
        else 0.0
    )
    round_trip_loss = recirculated_charge_bus - recirculated_discharge_bus

    lookup = {row.mode: row for row in modes}
    off = lookup["off_discharging"]
    cyclic_off_h = (
        off.elapsed_h * recirculated / stored_discharge
        if stored_discharge > 0.0
        else 0.0
    )
    depletion_off_h = off.elapsed_h - cyclic_off_h
    cyclic_endpoint_h = (
        off.elapsed_h * endpoint_recirculated / endpoint_stored_discharge
        if endpoint_stored_discharge > 0.0
        else 0.0
    )
    depletion_endpoint_h = off.elapsed_h - cyclic_endpoint_h
    internal_change = stored_charge - stored_discharge
    terminal_soc = entries[-1].soc
    endpoint_change = float(battery.stored_energy_kwh(terminal_soc)) - float(
        battery.stored_energy_kwh(start_soc)
    )
    cyclic_internal_fraction = cyclic_off_h / duration_h
    cyclic_endpoint_fraction = cyclic_endpoint_h / duration_h
    uncertainty_low = min(cyclic_internal_fraction, cyclic_endpoint_fraction)
    uncertainty_high = max(cyclic_internal_fraction, cyclic_endpoint_fraction)
    return ModeDecomposition(
        modes=modes,
        battery_mode=battery.battery_mode.value,
        timestep_s=max(step.dt_s for step in entries),
        neutral_band_kw=band,
        window_duration_h=duration_h,
        initial_soc=start_soc,
        terminal_soc=terminal_soc,
        delta_soc=terminal_soc - start_soc,
        endpoint_battery_energy_change_kwh=endpoint_change,
        internal_ledger_energy_change_kwh=internal_change,
        euler_ledger_residual_kwh=endpoint_change - internal_change,
        charge_bus_energy_in_kwh=charge_bus,
        discharge_bus_energy_out_kwh=discharge_bus,
        stored_charge_energy_kwh=stored_charge,
        stored_discharge_energy_kwh=stored_discharge,
        recirculated_internal_energy_kwh=recirculated,
        recirculated_charge_bus_energy_kwh=recirculated_charge_bus,
        recirculated_discharge_bus_energy_kwh=recirculated_discharge_bus,
        recirculated_round_trip_loss_kwh=round_trip_loss,
        engine_off_total_h=off.elapsed_h,
        engine_off_cyclic_h=cyclic_off_h,
        engine_off_depletion_h=depletion_off_h,
        engine_off_total_fraction=off.elapsed_h / duration_h,
        engine_off_cyclic_fraction=cyclic_internal_fraction,
        engine_off_depletion_fraction=depletion_off_h / duration_h,
        off_time_allocation=(
            "legacy fields use the integrated internal-power ledger; endpoint "
            "fields use per-step stored-energy changes; both allocate "
            "recirculated discharge pro rata across all discharging modes"
        ),
        endpoint_stored_charge_energy_kwh=endpoint_stored_charge,
        endpoint_stored_discharge_energy_kwh=endpoint_stored_discharge,
        endpoint_recirculated_energy_kwh=endpoint_recirculated,
        engine_off_cyclic_endpoint_h=cyclic_endpoint_h,
        engine_off_depletion_endpoint_h=depletion_endpoint_h,
        engine_off_cyclic_endpoint_fraction=cyclic_endpoint_fraction,
        engine_off_depletion_endpoint_fraction=depletion_endpoint_h / duration_h,
        cyclic_off_fraction_uncertainty_low=uncertainty_low,
        cyclic_off_fraction_uncertainty_high=uncertainty_high,
        cyclic_off_fraction_uncertainty_width=(uncertainty_high - uncertainty_low),
    )


def write_mode_decomposition_csv(
    decomposition: ModeDecomposition, output_path: str | Path
) -> Path:
    """Write four mode rows with repeated window-level ledger metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = asdict(decomposition)
    summary.pop("modes")
    rows = [asdict(mode) | summary for mode in decomposition.modes]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_mode_decompositions_csv(
    decompositions: Sequence[ModeDecomposition], output_path: str | Path
) -> Path:
    """Write multiple mode and timestep ledgers to one table."""
    rows = []
    for decomposition in decompositions:
        summary = asdict(decomposition)
        summary.pop("modes")
        rows.extend(asdict(mode) | summary for mode in decomposition.modes)
    if not rows:
        raise ValueError("decompositions must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
