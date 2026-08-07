"""Read-only audit of battery constraint observability in mission logs."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.models.battery import BatteryMode, BatteryPack
from src.simulation.simulator import TimeStep

__all__ = [
    "BatteryModePowerPoint",
    "ConstraintAudit",
    "ConstraintAuditPoint",
    "DirectionMarginSummary",
    "audit_constraint_margins",
    "build_battery_mode_comparison",
    "plot_battery_mode_comparison",
    "plot_constraint_margins",
    "write_constraint_margins_csv",
    "write_battery_mode_comparison_csv",
]


@dataclass(frozen=True)
class BatteryModePowerPoint:
    """Legacy and physical instantaneous power limits at one SoC."""

    scenario: str
    soc: float
    legacy_charge_max_kw: float
    physical_charge_max_kw: float
    charge_binding_limit: str
    charge_permissiveness: str
    legacy_discharge_max_kw: float
    physical_discharge_max_kw: float
    discharge_binding_limit: str
    discharge_permissiveness: str
    i_charge_max_a: float
    i_discharge_max_a: float
    terminal_voltage_min_v: float
    terminal_voltage_max_v: float
    q_nominal_ah: float


@dataclass(frozen=True)
class ConstraintAuditPoint:
    """Observed battery point with physical margins left undefined."""

    direction: str
    log_index: int
    time_s: float
    phase: str
    start_soc: float
    battery_bus_kw: float
    current_magnitude_a: float
    open_circuit_voltage_v: float
    terminal_voltage_v: float
    resistance_ohm: float
    physical_current_limit_a: float | None
    physical_current_margin_a: float | None
    physical_current_limit_source: str
    physical_terminal_voltage_limit_v: float | None
    physical_voltage_margin_v: float | None
    physical_voltage_limit_source: str
    ocv_endpoint_difference_v: float
    bus_power_proxy_limit_kw: float
    bus_power_proxy_margin_kw: float


@dataclass(frozen=True)
class DirectionMarginSummary:
    """Distribution summary for charge or discharge observations."""

    direction: str
    point_count: int
    physical_current_margin_min_a: float | None
    physical_voltage_margin_min_v: float | None
    physical_limit_proximity_fraction: float | None
    physical_current_limit_source: str
    physical_voltage_limit_source: str
    observed_current_min_a: float
    observed_current_median_a: float
    observed_current_p95_a: float
    observed_current_max_a: float
    observed_terminal_voltage_min_v: float
    observed_terminal_voltage_median_v: float
    observed_terminal_voltage_p95_v: float
    observed_terminal_voltage_max_v: float
    minimum_ocv_endpoint_difference_v: float
    endpoint_difference_soc: float
    endpoint_difference_bus_power_kw: float
    minimum_bus_power_proxy_margin_kw: float
    proxy_margin_soc: float
    proxy_margin_bus_power_kw: float
    fraction_at_bus_power_proxy: float
    bus_power_proxy_tolerance_kw: float


@dataclass(frozen=True)
class ConstraintAudit:
    """Charge and discharge audit plus all non-neutral logged observations."""

    charge: DirectionMarginSummary
    discharge: DirectionMarginSummary
    points: tuple[ConstraintAuditPoint, ...]
    neutral_band_kw: float
    physical_margin_status: str


def _summary(
    direction: str,
    points: Sequence[ConstraintAuditPoint],
    proxy_tolerance_kw: float,
) -> DirectionMarginSummary:
    if not points:
        raise ValueError(f"mission log contains no {direction} points")
    currents = np.array([point.current_magnitude_a for point in points])
    voltages = np.array([point.terminal_voltage_v for point in points])
    endpoint = min(points, key=lambda point: point.ocv_endpoint_difference_v)
    proxy = min(points, key=lambda point: point.bus_power_proxy_margin_kw)
    at_proxy = sum(
        point.bus_power_proxy_margin_kw <= proxy_tolerance_kw for point in points
    ) / len(points)
    current_source = _CURRENT_LIMIT_SOURCE
    voltage_source = _VOLTAGE_LIMIT_SOURCE
    return DirectionMarginSummary(
        direction=direction,
        point_count=len(points),
        physical_current_margin_min_a=None,
        physical_voltage_margin_min_v=None,
        physical_limit_proximity_fraction=None,
        physical_current_limit_source=current_source,
        physical_voltage_limit_source=voltage_source,
        observed_current_min_a=float(np.min(currents)),
        observed_current_median_a=float(np.median(currents)),
        observed_current_p95_a=float(np.quantile(currents, 0.95)),
        observed_current_max_a=float(np.max(currents)),
        observed_terminal_voltage_min_v=float(np.min(voltages)),
        observed_terminal_voltage_median_v=float(np.median(voltages)),
        observed_terminal_voltage_p95_v=float(np.quantile(voltages, 0.95)),
        observed_terminal_voltage_max_v=float(np.max(voltages)),
        minimum_ocv_endpoint_difference_v=endpoint.ocv_endpoint_difference_v,
        endpoint_difference_soc=endpoint.start_soc,
        endpoint_difference_bus_power_kw=endpoint.battery_bus_kw,
        minimum_bus_power_proxy_margin_kw=proxy.bus_power_proxy_margin_kw,
        proxy_margin_soc=proxy.start_soc,
        proxy_margin_bus_power_kw=proxy.battery_bus_kw,
        fraction_at_bus_power_proxy=at_proxy,
        bus_power_proxy_tolerance_kw=proxy_tolerance_kw,
    )


_CURRENT_LIMIT_SOURCE = (
    "undefined: B-04 and BatteryPack expose a bus-power C-rate proxy, "
    "not a physical current limit"
)
_VOLTAGE_LIMIT_SOURCE = (
    "undefined: B-03 v_min_v/v_max_v are open-circuit calibration "
    "endpoints, not terminal-voltage limits"
)


def audit_constraint_margins(
    log: Sequence[TimeStep],
    battery: BatteryPack,
    *,
    initial_soc: float,
    neutral_band_kw: float = 1.0e-6,
    bus_power_proxy_tolerance_kw: float = 1.0e-6,
) -> ConstraintAudit:
    """Reconstruct observed electrical points without inventing physical limits."""
    entries = tuple(log)
    start_soc = float(initial_soc)
    band = float(neutral_band_kw)
    proxy_tolerance = float(bus_power_proxy_tolerance_kw)
    if not entries:
        raise ValueError("log must not be empty")
    if not 0.0 <= start_soc <= 1.0:
        raise ValueError("initial_soc must lie in [0, 1]")
    if band < 0.0 or proxy_tolerance < 0.0:
        raise ValueError("numerical tolerances must be non-negative")

    points: list[ConstraintAuditPoint] = []
    soc = start_soc
    for index, step in enumerate(entries):
        power = step.battery_bus_kw
        if abs(power) > band:
            direction = "charge" if power < 0.0 else "discharge"
            v_oc = float(battery.open_circuit_voltage(soc))
            signed_current = step.battery_internal_kw * 1000.0 / v_oc
            current = abs(signed_current)
            resistance = battery.internal_resistance_ohm
            terminal = v_oc - signed_current * resistance
            if not math.isclose(
                terminal * signed_current / 1000.0,
                power,
                rel_tol=0.0,
                abs_tol=2.0e-10,
            ):
                raise ValueError("logged battery power is inconsistent with Rint fields")
            if direction == "charge":
                endpoint_difference = battery.v_max_v - terminal
                proxy_limit = battery.max_charge_kw
                proxy_margin = proxy_limit - abs(power)
            else:
                endpoint_difference = terminal - battery.v_min_v
                proxy_limit = battery.max_discharge_kw
                proxy_margin = proxy_limit - power
            points.append(
                ConstraintAuditPoint(
                    direction=direction,
                    log_index=index,
                    time_s=step.time_s,
                    phase=step.phase,
                    start_soc=soc,
                    battery_bus_kw=power,
                    current_magnitude_a=current,
                    open_circuit_voltage_v=v_oc,
                    terminal_voltage_v=terminal,
                    resistance_ohm=resistance,
                    physical_current_limit_a=None,
                    physical_current_margin_a=None,
                    physical_current_limit_source=_CURRENT_LIMIT_SOURCE,
                    physical_terminal_voltage_limit_v=None,
                    physical_voltage_margin_v=None,
                    physical_voltage_limit_source=_VOLTAGE_LIMIT_SOURCE,
                    ocv_endpoint_difference_v=endpoint_difference,
                    bus_power_proxy_limit_kw=proxy_limit,
                    bus_power_proxy_margin_kw=proxy_margin,
                )
            )
        soc = step.soc

    charge_points = tuple(point for point in points if point.direction == "charge")
    discharge_points = tuple(
        point for point in points if point.direction == "discharge"
    )
    return ConstraintAudit(
        charge=_summary("charge", charge_points, proxy_tolerance),
        discharge=_summary("discharge", discharge_points, proxy_tolerance),
        points=tuple(points),
        neutral_band_kw=band,
        physical_margin_status=(
            "undefined: the current model contains neither physical current "
            "limits nor terminal-voltage limits"
        ),
    )


def write_constraint_margins_csv(
    audit: ConstraintAudit, output_path: str | Path
) -> Path:
    """Write all observed charge and discharge points and explicit undefineds."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(point) for point in audit.points]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_constraint_margins(
    audit: ConstraintAudit, output_path: str | Path
) -> "matplotlib.figure.Figure":
    """Plot observed electrical distributions and bus-power proxy margins."""
    import matplotlib.pyplot as plt

    charge = [point for point in audit.points if point.direction == "charge"]
    discharge = [point for point in audit.points if point.direction == "discharge"]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    for points, color, label in (
        (charge, "tab:blue", "charge"),
        (discharge, "tab:orange", "discharge"),
    ):
        axes[0, 0].hist(
            [point.current_magnitude_a for point in points],
            bins=24,
            alpha=0.55,
            color=color,
            label=label,
        )
        axes[0, 1].hist(
            [point.terminal_voltage_v for point in points],
            bins=24,
            alpha=0.55,
            color=color,
            label=label,
        )
        axes[1, 0].hist(
            [point.bus_power_proxy_margin_kw for point in points],
            bins=24,
            alpha=0.55,
            color=color,
            label=label,
        )
        axes[1, 1].scatter(
            [point.start_soc for point in points],
            [point.terminal_voltage_v for point in points],
            s=7,
            alpha=0.45,
            color=color,
            label=label,
        )
    axes[0, 0].set(xlabel="Observed current magnitude [A]", ylabel="Count")
    axes[0, 1].set(xlabel="Observed terminal voltage [V]", ylabel="Count")
    axes[1, 0].set(xlabel="Bus-power proxy headroom [kW]", ylabel="Count")
    axes[1, 1].set(xlabel="Start-of-step SoC [-]", ylabel="Terminal voltage [V]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Battery observations; physical current and terminal-voltage margins undefined",
        fontsize=10,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    return figure


def _permissiveness(physical_kw: float, legacy_kw: float) -> str:
    tolerance = max(1.0e-10, abs(legacy_kw) * 1.0e-10)
    if physical_kw > legacy_kw + tolerance:
        return "physical_more_permissive"
    if physical_kw < legacy_kw - tolerance:
        return "physical_less_permissive"
    return "equal"


def build_battery_mode_comparison(
    soc_values: Sequence[float],
    legacy_battery: BatteryPack,
    physical_battery: BatteryPack,
    *,
    scenario: str,
) -> tuple[BatteryModePowerPoint, ...]:
    """Evaluate instantaneous charge and discharge limits over SoC."""
    if legacy_battery.battery_mode is not BatteryMode.LEGACY:
        raise ValueError("legacy_battery must use legacy mode")
    if physical_battery.battery_mode is not BatteryMode.PHYSICAL:
        raise ValueError("physical_battery must use physical mode")
    if not scenario:
        raise ValueError("scenario must not be empty")

    rows: list[BatteryModePowerPoint] = []
    for value in soc_values:
        soc = float(value)
        legacy_charge = legacy_battery.charge_availability(soc)
        physical_charge = physical_battery.charge_availability(soc)
        legacy_discharge = legacy_battery.discharge_availability(soc)
        physical_discharge = physical_battery.discharge_availability(soc)
        rows.append(
            BatteryModePowerPoint(
                scenario=scenario,
                soc=soc,
                legacy_charge_max_kw=legacy_charge.power_kw,
                physical_charge_max_kw=physical_charge.power_kw,
                charge_binding_limit=physical_charge.binding_limit,
                charge_permissiveness=_permissiveness(
                    physical_charge.power_kw, legacy_charge.power_kw
                ),
                legacy_discharge_max_kw=legacy_discharge.power_kw,
                physical_discharge_max_kw=physical_discharge.power_kw,
                discharge_binding_limit=physical_discharge.binding_limit,
                discharge_permissiveness=_permissiveness(
                    physical_discharge.power_kw, legacy_discharge.power_kw
                ),
                i_charge_max_a=float(physical_battery.i_charge_max_a),
                i_discharge_max_a=float(physical_battery.i_discharge_max_a),
                terminal_voltage_min_v=float(
                    physical_battery.terminal_voltage_min_v
                ),
                terminal_voltage_max_v=float(
                    physical_battery.terminal_voltage_max_v
                ),
                q_nominal_ah=float(physical_battery.q_nominal_ah),
            )
        )
    if not rows:
        raise ValueError("soc_values must not be empty")
    return tuple(rows)


def write_battery_mode_comparison_csv(
    points: Sequence[BatteryModePowerPoint], output_path: str | Path
) -> Path:
    """Write battery-mode power-limit curves and scenario metadata."""
    rows = [asdict(point) for point in points]
    if not rows:
        raise ValueError("points must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_battery_mode_comparison(
    points: Sequence[BatteryModePowerPoint],
    output_path: str | Path,
    *,
    scenario: str,
) -> "matplotlib.figure.Figure":
    """Plot legacy and physical instantaneous power limits for one scenario."""
    import matplotlib.pyplot as plt

    selected = [point for point in points if point.scenario == scenario]
    if not selected:
        raise ValueError(f"scenario {scenario!r} is absent from points")
    selected.sort(key=lambda point: point.soc)
    soc = [point.soc for point in selected]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for axis, direction in zip(axes, ("charge", "discharge")):
        legacy = [getattr(point, f"legacy_{direction}_max_kw") for point in selected]
        physical = [
            getattr(point, f"physical_{direction}_max_kw") for point in selected
        ]
        axis.plot(soc, legacy, "--", label="legacy bus-power proxy")
        axis.plot(soc, physical, label="physical current/voltage mode")
        axis.set(
            xlabel="State of charge [-]",
            ylabel=f"Maximum {direction} bus power [kW]",
            title=direction.capitalize(),
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"Battery power limits: {scenario}", fontsize=10)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    return figure
