"""Point-performance sizing for the hybrid-electric UAV.

The constraint diagram answers whether a wing loading and installed sea-level
engine rating can meet steady stall, cruise, climb, and ceiling requirements.
It does not establish endurance, battery energy sufficiency over a mission, or
mass/fuel-volume closure; those remain separate simulator and sizing checks.

No take-off-distance curve is included because the problem statement gives no
runway-length requirement. Transient climb curves may include a finite battery
boost, expressed at the DC bus, while sustained cruise and ceiling curves use
the engine alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.models.aerodynamics import loiter_speed
from src.models.atmosphere import atmosphere
from src.models.mass import build_mass_budget
from src.models.powertrain import SeriesPowertrain

__all__ = [
    "Airframe",
    "ConstraintCase",
    "DesignPoint",
    "constraint_curves",
    "feasible_design_point",
    "fuel_contours",
    "plot_constraint_diagram",
    "power_loading_required",
    "stall_wing_loading_limit",
]

FloatArray = npt.NDArray[np.float64]
STALL_SPEED_MARGIN = 1.2  # see assumptions.md CD-01


def _positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


@dataclass(frozen=True)
class ConstraintCase:
    """One point-performance condition used to build a power curve.

    ``battery_boost_kw`` is electrical power delivered at the DC bus. A zero
    value defines a sustained, engine-only curve.
    """

    name: str
    altitude_m: float
    speed_mps: float | None
    climb_rate_mps: float
    battery_boost_kw: float
    weight_n: float

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not np.isfinite(self.altitude_m) or not 0.0 <= self.altitude_m <= 20_000.0:
            raise ValueError("altitude_m must be finite and in [0, 20000]")
        if self.speed_mps is not None:
            _positive("speed_mps", self.speed_mps)
        if not np.isfinite(self.climb_rate_mps):
            raise ValueError("climb_rate_mps must be finite")
        if not np.isfinite(self.battery_boost_kw) or self.battery_boost_kw < 0.0:
            raise ValueError("battery_boost_kw must be finite and non-negative")
        _positive("weight_n", self.weight_n)


@dataclass(frozen=True)
class Airframe:
    """Aerodynamic inputs shared by every constraint case."""

    aspect_ratio: float
    oswald_efficiency: float
    cd0: float
    cl_max: float
    propeller_efficiency: float

    def __post_init__(self) -> None:
        for name in ("aspect_ratio", "oswald_efficiency", "cd0", "cl_max"):
            _positive(name, getattr(self, name))
        _positive("propeller_efficiency", self.propeller_efficiency)
        if self.propeller_efficiency > 1.0:
            raise ValueError("propeller_efficiency must not exceed 1")


@dataclass(frozen=True)
class DesignPoint:
    """A margin-adjusted design selected from a sampled feasible envelope."""

    wing_loading_pa: float
    power_loading_w_per_n: float
    wing_area_m2: float
    engine_power_sl_kw: float
    binding_constraint: str


class _CurveSet(dict[str, FloatArray]):
    """Dictionary of curves with the coordinates needed by design selection."""

    def __init__(self, wing_loading_pa: FloatArray, weight_n: float) -> None:
        super().__init__()
        self.wing_loading_pa = wing_loading_pa
        self.weight_n = weight_n


def _wing_loading_array(wing_loading_pa: Any) -> FloatArray:
    values = np.asarray(wing_loading_pa, dtype=np.float64)
    if not bool(np.all(np.isfinite(values) & (values > 0.0))):
        raise ValueError("wing_loading_pa must contain only finite positive values")
    return values


def stall_wing_loading_limit(
    airframe: Airframe, v_stall_max_mps: float, rho_sl: float
) -> float:
    """Maximum wing loading [N/m²] permitted by the sea-level stall speed."""
    _positive("v_stall_max_mps", v_stall_max_mps)
    _positive("rho_sl", rho_sl)
    return 0.5 * rho_sl * v_stall_max_mps**2 * airframe.cl_max


def power_loading_required(
    wing_loading_pa: Any,
    case: ConstraintCase,
    airframe: Airframe,
    powertrain: SeriesPowertrain,
    lapse_exponent: float,
) -> float | FloatArray:
    """Required installed sea-level engine power loading ``P_SL/W`` [W/N].

    The result is vectorized over ``wing_loading_pa``. Battery boost is first
    subtracted at the DC bus, so it bypasses source-chain losses but not the
    inverter/motor/cabling losses already included in bus demand.
    """
    wing_loading = _wing_loading_array(wing_loading_pa)
    if not np.isfinite(lapse_exponent) or lapse_exponent < 0.0:
        raise ValueError("lapse_exponent must be finite and non-negative")

    state = atmosphere(case.altitude_m)
    rho = float(state.density_kg_m3)
    sigma = float(state.density_ratio)
    wing_area_m2 = case.weight_n / wing_loading

    if case.speed_mps is None:
        speed, _ = loiter_speed(
            case.weight_n,
            rho,
            wing_area_m2,
            airframe.cd0,
            airframe.aspect_ratio,
            airframe.oswald_efficiency,
            airframe.cl_max,
            safety_margin=STALL_SPEED_MARGIN,
        )
        speed = np.asarray(speed, dtype=np.float64)
    else:
        speed = np.full_like(wing_loading, case.speed_mps)

    q = 0.5 * rho * speed**2
    drag_over_weight = (
        q * airframe.cd0 / wing_loading
        + wing_loading
        / (q * np.pi * airframe.aspect_ratio * airframe.oswald_efficiency)
    )
    shaft_w_per_n = (
        case.climb_rate_mps + speed * drag_over_weight
    ) / airframe.propeller_efficiency

    bus_w_per_n = shaft_w_per_n / powertrain.demand_chain_efficiency
    boost_w_per_n = case.battery_boost_kw * 1000.0 / case.weight_n
    engine_bus_w_per_n = np.maximum(bus_w_per_n - boost_w_per_n, 0.0)
    rating_w_per_n = engine_bus_w_per_n / (
        powertrain.source_chain_efficiency * sigma**lapse_exponent
    )
    if np.ndim(wing_loading_pa) == 0:
        return float(rating_w_per_n)
    return rating_w_per_n


def constraint_curves(
    wing_loading_pa: Any,
    cases: Sequence[ConstraintCase],
    airframe: Airframe,
    powertrain: SeriesPowertrain,
    lapse_exponent: float,
) -> dict[str, FloatArray]:
    """Evaluate named sustained and transient constraints on one common grid."""
    wing_loading = _wing_loading_array(wing_loading_pa)
    if wing_loading.ndim != 1 or wing_loading.size == 0:
        raise ValueError("wing_loading_pa must be a non-empty one-dimensional grid")
    if not cases:
        raise ValueError("cases must contain at least one constraint")
    weights = {case.weight_n for case in cases}
    if len(weights) != 1:
        raise ValueError("all cases must use the same weight_n")

    curves = _CurveSet(wing_loading.copy(), cases[0].weight_n)
    for case in cases:
        if case.name in curves:
            raise ValueError(f"duplicate constraint name {case.name!r}")
        curves[case.name] = np.asarray(
            power_loading_required(
                wing_loading, case, airframe, powertrain, lapse_exponent
            ),
            dtype=np.float64,
        )
    return curves


def feasible_design_point(
    curves: Mapping[str, npt.ArrayLike],
    stall_limit_pa: float,
    margin: float = 1.10,
) -> DesignPoint:
    """Select the sampled point with the lowest margin-adjusted power loading."""
    _positive("stall_limit_pa", stall_limit_pa)
    if not np.isfinite(margin) or margin < 1.0:
        raise ValueError("margin must be finite and at least 1")
    if not curves:
        raise ValueError("curves must contain at least one constraint")
    if not hasattr(curves, "wing_loading_pa") or not hasattr(curves, "weight_n"):
        raise ValueError(
            "curves must come from constraint_curves so wing-loading and weight metadata exist"
        )

    wing_loading = np.asarray(curves.wing_loading_pa, dtype=np.float64)  # type: ignore[attr-defined]
    names = tuple(curves)
    rows = np.vstack([np.asarray(curves[name], dtype=np.float64) for name in names])
    if rows.shape[1:] != wing_loading.shape:
        raise ValueError("every curve must match the wing_loading_pa grid")
    allowed = np.flatnonzero(wing_loading <= stall_limit_pa)
    if allowed.size == 0:
        raise ValueError("wing-loading grid contains no point inside the stall limit")

    envelope = np.max(rows, axis=0)
    local_index = int(np.argmin(envelope[allowed]))
    index = int(allowed[local_index])
    binding_index = int(np.argmax(rows[:, index]))
    required = float(envelope[index] * margin)
    weight_n = float(curves.weight_n)  # type: ignore[attr-defined]
    selected_wing_loading = float(wing_loading[index])
    return DesignPoint(
        wing_loading_pa=selected_wing_loading,
        power_loading_w_per_n=required,
        wing_area_m2=weight_n / selected_wing_loading,
        engine_power_sl_kw=required * weight_n / 1000.0,
        binding_constraint=names[binding_index],
    )


def fuel_contours(
    wing_loading_grid: Any,
    power_loading_grid: Any,
    airframe: Airframe,
    mtow_kg: float,
    *,
    battery_kwh: float,
    peak_bus_kw: Any,
    **mass_options: float,
) -> FloatArray:
    """Map a sizing grid to fuel mass [kg] using the fixed-MTOW mass closure.

    Battery capacity and peak bus rating are explicit because ``W/S`` and
    ``P/W`` alone do not size the battery, inverter, or motor.
    """
    _positive("mtow_kg", mtow_kg)
    if not np.isfinite(battery_kwh) or battery_kwh < 0.0:
        raise ValueError("battery_kwh must be finite and non-negative")
    wing_loading = _wing_loading_array(wing_loading_grid)
    power_loading = _wing_loading_array(power_loading_grid)
    peak_bus = _wing_loading_array(peak_bus_kw)
    wing_loading, power_loading, peak_bus = np.broadcast_arrays(
        wing_loading, power_loading, peak_bus
    )
    weight_n = mtow_kg * 9.80665

    result = np.empty(wing_loading.shape, dtype=np.float64)
    for index in np.ndindex(result.shape):
        budget = build_mass_budget(
            engine_kw=float(power_loading[index] * weight_n / 1000.0),
            battery_kwh=battery_kwh,
            peak_bus_kw=float(peak_bus[index]),
            wing_area_m2=float(weight_n / wing_loading[index]),
            aspect_ratio=airframe.aspect_ratio,
            mtow_kg=mtow_kg,
            **mass_options,
        )
        result[index] = budget.fuel_kg
    return result


def plot_constraint_diagram(
    wing_loading_pa: Any,
    cases: Sequence[ConstraintCase],
    airframe: Airframe,
    powertrain: SeriesPowertrain,
    lapse_exponent: float,
    *,
    v_stall_max_mps: float,
    margin: float = 1.10,
    design_point: DesignPoint | None = None,
    fuel_contour_data: tuple[Any, Any, Any] | None = None,
    output_path: str | Path = "deliverables/figures/constraint_diagram.png",
) -> "matplotlib.figure.Figure":
    """Create and save the report constraint diagram."""
    import matplotlib.pyplot as plt

    wing_loading = _wing_loading_array(wing_loading_pa)
    curves = constraint_curves(
        wing_loading, cases, airframe, powertrain, lapse_exponent
    )
    rho_sl = float(atmosphere(0.0).density_kg_m3)
    stall_limit = stall_wing_loading_limit(airframe, v_stall_max_mps, rho_sl)
    point = design_point or feasible_design_point(curves, stall_limit, margin)

    figure, axis = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, case in enumerate(cases):
        family = "sustained" if case.battery_boost_kw == 0.0 else "transient"
        axis.plot(
            wing_loading,
            curves[case.name],
            linestyle="-" if family == "sustained" else "--",
            color=colors[index % len(colors)],
            label="_nolegend_",
        )

    rows = np.vstack(list(curves.values()))
    envelope = margin * np.max(rows, axis=0)
    mask = wing_loading <= stall_limit
    y_max = max(float(np.max(envelope[mask])) * 1.22, point.power_loading_w_per_n * 1.15)
    axis.fill_between(
        wing_loading[mask], envelope[mask], y_max, color="tab:green", alpha=0.13,
        label="_nolegend_",
    )
    axis.axvline(stall_limit, color="black", linestyle=":", label="_nolegend_")
    axis.scatter(
        [point.wing_loading_pa], [point.power_loading_w_per_n], marker="*", s=150,
        color="black", zorder=5, label="_nolegend_",
    )
    axis.annotate(
        "selected design",
        (point.wing_loading_pa, point.power_loading_w_per_n),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
    )

    if fuel_contour_data is not None:
        x_grid, y_grid, fuel_grid = fuel_contour_data
        contours = axis.contour(x_grid, y_grid, fuel_grid, colors="0.35", linewidths=0.8)
        axis.clabel(contours, fmt="%g kg fuel", fontsize=8)

    axis.set_xlim(float(np.min(wing_loading)), float(np.max(wing_loading)))
    axis.set_ylim(0.0, y_max)
    axis.set_xlabel("Wing loading, W/S [N/m²]")
    axis.set_ylabel("Installed sea-level engine power loading, P_SL/W [W/N]")
    axis.set_title(
        f"Solid: sustained · dashed: transient · shaded: feasible with {margin - 1.0:.0%} margin",
        fontsize=10,
    )
    axis.grid(True, alpha=0.25)

    label_x = float(wing_loading[-1])
    for index, case in enumerate(cases):
        axis.annotate(
            case.name,
            (label_x, float(curves[case.name][-1])),
            xytext=(-4, 3),
            textcoords="offset points",
            ha="right",
            color=colors[index % len(colors)],
            fontsize=7,
        )
    axis.annotate(
        "stall limit", (stall_limit, 0.0), xytext=(4, 8),
        textcoords="offset points", rotation=90, va="bottom", fontsize=8,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    return figure
