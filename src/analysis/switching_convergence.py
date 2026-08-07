"""Fuel and switching convergence for endpoint-energy replay policies."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.analysis.replay_comparison import EndpointEnergyComparison, StrategyReplay

__all__ = [
    "SwitchingConvergenceRow",
    "build_switching_convergence_rows",
    "pi_restart_frequency_grows_monotonically",
    "plot_switching_convergence",
    "write_switching_convergence_csv",
]


@dataclass(frozen=True)
class SwitchingConvergenceRow:
    """One point strategy or unequal-energy PI policy at one timestep."""

    comparison: str
    battery_mode: str
    timestep_s: float
    strategy: str
    policy_role: str
    s0_ratio: float | None
    target_kwh: float
    endpoint_energy_residual_kwh: float
    fuel_consumed_kg: float
    fuel_convergence_order: float | None
    fuel_convergence_status: str
    pi_to_continuous_gap_kg: float | None
    pi_to_ideal_gap_kg: float | None
    restart_count: int | None
    restarts_per_flight_hour: float | None
    restart_frequency_grows_monotonically: bool | None


def _duration_h(result: StrategyReplay) -> float:
    return result.fuel_consumed_kg / result.average_fuel_rate_kg_h


def _raw_rows(
    comparisons: Sequence[EndpointEnergyComparison],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        strategies = [
            ("point", comparison.continuous),
            ("relaxed_point", comparison.ideal_relaxed),
        ]
        interval = comparison.pi_endpoint_policy_interval
        if interval.exact_match:
            strategies.append(("point", interval.lower_energy_policy))
        else:
            strategies.extend(
                (
                    ("lower_energy_policy", interval.lower_energy_policy),
                    ("upper_energy_policy", interval.upper_energy_policy),
                )
            )
        for role, result in strategies:
            restart_rate = (
                result.restart_count / _duration_h(result)
                if result.restart_count is not None
                else None
            )
            rows.append(
                {
                    "comparison": comparison.comparison,
                    "battery_mode": comparison.battery_mode,
                    "timestep_s": comparison.timestep_s,
                    "strategy": result.strategy,
                    "policy_role": role,
                    "s0_ratio": result.calibration_parameter_value,
                    "target_kwh": comparison.target_battery_energy_change_kwh,
                    "endpoint_energy_residual_kwh": (
                        result.terminal_energy_shortfall_kwh
                    ),
                    "fuel_consumed_kg": result.fuel_consumed_kg,
                    "pi_to_continuous_gap_kg": comparison.pi_to_continuous_gap_kg,
                    "pi_to_ideal_gap_kg": comparison.pi_to_ideal_gap_kg,
                    "restart_count": result.restart_count,
                    "restarts_per_flight_hour": restart_rate,
                }
            )
    return rows


def _observed_order(values: Sequence[tuple[float, float]]) -> tuple[float | None, str]:
    ordered = sorted(values, reverse=True)
    if len(ordered) != 3 or [item[0] for item in ordered] != [60.0, 30.0, 15.0]:
        return None, "undefined; requires the 60/30/15 s series"
    coarse, medium, fine = (item[1] for item in ordered)
    first = coarse - medium
    second = medium - fine
    if abs(first) <= 1.0e-12 or abs(second) <= 1.0e-12:
        return None, "roundoff-limited or unchanged across timestep"
    if first * second <= 0.0:
        return None, "undefined; successive fuel differences change sign"
    return math.log(abs(first / second), 2.0), "observed three-level order"


def build_switching_convergence_rows(
    comparisons: Sequence[EndpointEnergyComparison],
) -> tuple[SwitchingConvergenceRow, ...]:
    """Flatten comparisons and add fuel-order and PI restart-trend diagnostics."""
    raw = _raw_rows(comparisons)
    fuel_series: dict[tuple[str, str, str, str], list[tuple[float, float]]] = {}
    for row in raw:
        key = (
            str(row["comparison"]),
            str(row["battery_mode"]),
            str(row["strategy"]),
            str(row["policy_role"]),
        )
        fuel_series.setdefault(key, []).append(
            (float(row["timestep_s"]), float(row["fuel_consumed_kg"]))
        )

    monotonic: dict[tuple[str, str], bool] = {}
    for comparison in {str(row["comparison"]) for row in raw}:
        for mode in {str(row["battery_mode"]) for row in raw}:
            rates_by_dt: dict[float, list[float]] = {}
            for row in raw:
                if (
                    row["comparison"] == comparison
                    and row["battery_mode"] == mode
                    and row["strategy"] == "pi_ecms"
                    and row["restarts_per_flight_hour"] is not None
                ):
                    rates_by_dt.setdefault(float(row["timestep_s"]), []).append(
                        float(row["restarts_per_flight_hour"])
                    )
            if sorted(rates_by_dt, reverse=True) == [60.0, 30.0, 15.0]:
                intervals = [
                    (min(rates_by_dt[dt]), max(rates_by_dt[dt]))
                    for dt in (60.0, 30.0, 15.0)
                ]
                monotonic[(comparison, mode)] = (
                    intervals[1][0] > intervals[0][1]
                    and intervals[2][0] > intervals[1][1]
                )

    output = []
    for row in raw:
        key = (
            str(row["comparison"]),
            str(row["battery_mode"]),
            str(row["strategy"]),
            str(row["policy_role"]),
        )
        order, status = _observed_order(fuel_series[key])
        trend = (
            monotonic.get((str(row["comparison"]), str(row["battery_mode"])))
            if row["strategy"] == "pi_ecms"
            else None
        )
        output.append(
            SwitchingConvergenceRow(
                **row,
                fuel_convergence_order=order,
                fuel_convergence_status=status,
                restart_frequency_grows_monotonically=trend,
            )
        )
    return tuple(output)


def pi_restart_frequency_grows_monotonically(
    rows: Sequence[SwitchingConvergenceRow], *, comparison: str, battery_mode: str
) -> bool:
    """Return the strict interval-separated 60/30/15 s PI restart trend."""
    selected = [
        row
        for row in rows
        if row.comparison == comparison
        and row.battery_mode == battery_mode
        and row.strategy == "pi_ecms"
    ]
    values = {row.restart_frequency_grows_monotonically for row in selected}
    if not selected or None in values or len(values) != 1:
        raise ValueError("complete PI 60/30/15 s series is required")
    return bool(values.pop())


def write_switching_convergence_csv(
    rows: Sequence[SwitchingConvergenceRow], output_path: str | Path
) -> Path:
    """Write the switching and fuel convergence audit."""
    entries = tuple(rows)
    if not entries:
        raise ValueError("rows must not be empty")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(entries[0].__dict__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in entries)
    return path


def plot_switching_convergence(
    rows: Sequence[SwitchingConvergenceRow], output_path: str | Path
) -> Path:
    """Plot PI restart-rate ranges across unequal-energy policies."""
    import matplotlib.pyplot as plt

    pi_rows = [row for row in rows if row.strategy == "pi_ecms"]
    if not pi_rows:
        raise ValueError("rows contain no PI replay")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, comparison in zip(axes, ("charge_sustaining", "mission_depleting")):
        for mode, colour in (("legacy", "tab:blue"), ("physical", "tab:orange")):
            selected = [
                row
                for row in pi_rows
                if row.comparison == comparison and row.battery_mode == mode
            ]
            by_dt = {}
            for row in selected:
                by_dt.setdefault(row.timestep_s, []).append(
                    float(row.restarts_per_flight_hour)
                )
            dts = sorted(by_dt, reverse=True)
            lows = [min(by_dt[dt]) for dt in dts]
            highs = [max(by_dt[dt]) for dt in dts]
            axis.plot(dts, lows, marker="o", color=colour, label=mode)
            axis.fill_between(dts, lows, highs, color=colour, alpha=0.18)
        axis.set_title(comparison.replace("_", " ").title())
        axis.set_xlabel("Timestep (s)")
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    axes[0].set_ylabel("PI restarts per flight hour")
    axes[1].legend()
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
