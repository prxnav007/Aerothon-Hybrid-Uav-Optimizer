from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.analysis.constraint_diagram import (
    Airframe,
    ConstraintCase,
    constraint_curves,
    feasible_design_point,
    fuel_contours,
    plot_constraint_diagram,
    power_loading_required,
    stall_wing_loading_limit,
)
from src.models.aerodynamics import (
    HighAspectRatioWarning,
    max_lift_to_drag,
    oswald_efficiency,
)
from src.models.atmosphere import atmosphere
from src.models.engine import LAPSE_EXPONENT
from src.models.mass import build_mass_budget
from src.models.powertrain import SeriesPowertrain

WEIGHT_N = 1000.0 * 9.80665
CRUISE_MPS = 250.0 * 1000.0 / 3600.0
REFERENCE_WING_LOADING = WEIGHT_N / 10.0
AIRFRAME = Airframe(
    aspect_ratio=16.0,
    oswald_efficiency=0.78,
    cd0=0.028,
    cl_max=1.5,
    propeller_efficiency=0.85,
)
POWERTRAIN = SeriesPowertrain()


def case(
    name: str = "cruise_3km",
    *,
    altitude_m: float = 3000.0,
    speed_mps: float | None = CRUISE_MPS,
    climb_rate_mps: float = 0.0,
    battery_boost_kw: float = 0.0,
) -> ConstraintCase:
    return ConstraintCase(
        name=name,
        altitude_m=altitude_m,
        speed_mps=speed_mps,
        climb_rate_mps=climb_rate_mps,
        battery_boost_kw=battery_boost_kw,
        weight_n=WEIGHT_N,
    )


def design_cases() -> tuple[ConstraintCase, ...]:
    return (
        case("cruise_3km"),
        case(
            "climb_3km_transient",
            speed_mps=None,
            climb_rate_mps=2.0,
            battery_boost_kw=30.0,
        ),
        case(
            "ceiling_10km",
            altitude_m=10_000.0,
            speed_mps=None,
            climb_rate_mps=0.5,
        ),
    )


def stall_limit(airframe: Airframe = AIRFRAME) -> float:
    return stall_wing_loading_limit(
        airframe, 45.0 / 1.2, float(atmosphere(0.0).density_kg_m3)
    )


def test_cruise_curve_minimum_matches_the_closed_form_best_ld_condition() -> None:
    rho = float(atmosphere(3000.0).density_kg_m3)
    q = 0.5 * rho * CRUISE_MPS**2
    expected = q * np.sqrt(
        AIRFRAME.cd0 * np.pi * AIRFRAME.aspect_ratio * AIRFRAME.oswald_efficiency
    )
    grid = np.linspace(0.6 * expected, 1.4 * expected, 20_001)
    curve = power_loading_required(grid, case(), AIRFRAME, POWERTRAIN, LAPSE_EXPONENT)
    actual = grid[int(np.argmin(curve))]
    assert actual == pytest.approx(expected, abs=np.diff(grid).max())


def test_cruise_minimum_has_the_aerodynamics_modules_maximum_lift_to_drag() -> None:
    rho = float(atmosphere(3000.0).density_kg_m3)
    q = 0.5 * rho * CRUISE_MPS**2
    wing_loading = q * np.sqrt(
        AIRFRAME.cd0 * np.pi * AIRFRAME.aspect_ratio * AIRFRAME.oswald_efficiency
    )
    cl = wing_loading / q
    cd = AIRFRAME.cd0 + cl**2 / (
        np.pi * AIRFRAME.aspect_ratio * AIRFRAME.oswald_efficiency
    )
    assert cl / cd == pytest.approx(
        max_lift_to_drag(
            AIRFRAME.cd0, AIRFRAME.aspect_ratio, AIRFRAME.oswald_efficiency
        ),
        rel=1e-13,
    )


def test_cruise_curve_decreases_then_increases_strictly() -> None:
    rho = float(atmosphere(3000.0).density_kg_m3)
    q = 0.5 * rho * CRUISE_MPS**2
    optimum = q * np.sqrt(
        AIRFRAME.cd0 * np.pi * AIRFRAME.aspect_ratio * AIRFRAME.oswald_efficiency
    )
    below = np.linspace(0.3 * optimum, 0.999 * optimum, 1000)
    above = np.linspace(1.001 * optimum, 2.0 * optimum, 1000)
    assert np.all(np.diff(power_loading_required(
        below, case(), AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )) < 0.0)
    assert np.all(np.diff(power_loading_required(
        above, case(), AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )) > 0.0)


def test_stall_limit_is_a_vertical_boundary_and_linear_in_clmax() -> None:
    rho_sl = float(atmosphere(0.0).density_kg_m3)
    first = stall_wing_loading_limit(AIRFRAME, 37.5, rho_sl)
    doubled = stall_wing_loading_limit(
        Airframe(16.0, 0.78, 0.028, 3.0, 0.85), 37.5, rho_sl
    )
    assert first == pytest.approx(1291.9921893489611)
    assert doubled == pytest.approx(2.0 * first)


def test_dc_bus_battery_boost_strictly_reduces_installed_engine_loading() -> None:
    grid = np.linspace(500.0, 1800.0, 301)
    sustained = power_loading_required(
        grid, case(), AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )
    transient = power_loading_required(
        grid, case("boost", battery_boost_kw=10.0), AIRFRAME, POWERTRAIN,
        LAPSE_EXPONENT,
    )
    assert np.all(transient < sustained)


def test_hybrid_reduction_matches_a_hand_bus_side_conversion() -> None:
    sustained_case = case()
    boosted_case = case("boost", battery_boost_kw=10.0)
    sustained = power_loading_required(
        REFERENCE_WING_LOADING, sustained_case, AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )
    transient = power_loading_required(
        REFERENCE_WING_LOADING, boosted_case, AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )
    sigma = float(atmosphere(3000.0).density_ratio)
    expected = (10_000.0 / WEIGHT_N) / (
        POWERTRAIN.source_chain_efficiency * sigma**LAPSE_EXPONENT
    )
    assert sustained - transient == pytest.approx(expected, rel=1e-13)


def test_solved_climb_speed_removes_the_old_65mps_drag_penalty() -> None:
    solved = case("solved_climb", speed_mps=None, climb_rate_mps=2.0)
    fixed = case("fixed_climb", speed_mps=65.0, climb_rate_mps=2.0)
    solved_rating = power_loading_required(
        REFERENCE_WING_LOADING, solved, AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )
    fixed_rating = power_loading_required(
        REFERENCE_WING_LOADING, fixed, AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )
    assert solved_rating < fixed_rating


def test_cruise_rating_altitude_sweep_exposes_the_nonflat_high_altitude_tail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ratings_kw = []
    for altitude_m in range(0, 10_001, 1000):
        rating = power_loading_required(
            REFERENCE_WING_LOADING,
            case(f"cruise_{altitude_m}", altitude_m=altitude_m),
            AIRFRAME,
            POWERTRAIN,
            LAPSE_EXPONENT,
        ) * WEIGHT_N / 1000.0
        ratings_kw.append(rating)
    print("cruise altitude sweep [kW]:", [round(v, 3) for v in ratings_kw])
    assert int(np.argmin(ratings_kw)) == 1
    assert ratings_kw[-1] > 1.35 * min(ratings_kw)
    assert "cruise altitude sweep" in capsys.readouterr().out


def test_greater_lapse_exponent_requires_more_rating_above_sea_level() -> None:
    grid = np.linspace(500.0, 1600.0, 101)
    low = power_loading_required(grid, case(), AIRFRAME, POWERTRAIN, 0.5)
    high = power_loading_required(grid, case(), AIRFRAME, POWERTRAIN, 1.0)
    assert np.all(high > low)


def test_reference_airframe_requires_93kw_and_the_75kw_point_fails_cruise() -> None:
    required_w_per_n = power_loading_required(
        REFERENCE_WING_LOADING, case(), AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )
    required_kw = required_w_per_n * WEIGHT_N / 1000.0
    available_w_per_n = 75_000.0 / WEIGHT_N
    assert required_kw == pytest.approx(93.33675192840187, rel=1e-12)
    assert required_kw > 85.0
    assert available_w_per_n < required_w_per_n


def test_design_point_satisfies_all_curves_and_downward_perturbation_does_not() -> None:
    grid = np.linspace(150.0, 1800.0, 6601)
    curves = constraint_curves(
        grid, design_cases(), AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )
    point = feasible_design_point(curves, stall_limit())
    index = int(np.argmin(abs(grid - point.wing_loading_pa)))
    required = max(curve[index] for curve in curves.values())
    assert point.wing_loading_pa <= stall_limit()
    assert point.power_loading_w_per_n >= required
    assert point.power_loading_w_per_n / 1.10 == pytest.approx(required)
    perturbed = point.power_loading_w_per_n / 1.10 * (1.0 - 1e-12)
    assert any(perturbed < curve[index] for curve in curves.values())


@pytest.mark.parametrize(
    ("cases", "expected"),
    [
        ((case("cruise"), case(
            "climb", speed_mps=None, climb_rate_mps=2.0, battery_boost_kw=30.0
        )), "cruise"),
        ((case("cruise"), case(
            "climb", speed_mps=None, climb_rate_mps=8.0
        )), "climb"),
    ],
)
def test_binding_constraint_names_the_active_curve(
    cases: tuple[ConstraintCase, ...], expected: str
) -> None:
    grid = np.linspace(300.0, 1400.0, 2201)
    curves = constraint_curves(grid, cases, AIRFRAME, POWERTRAIN, LAPSE_EXPONENT)
    assert feasible_design_point(curves, stall_limit()).binding_constraint == expected


def test_fuel_contours_match_direct_mass_closure() -> None:
    wing_loading, power_loading = np.meshgrid(
        np.array([700.0, 1000.0]), np.array([8.0, 11.0])
    )
    peak_bus = power_loading * WEIGHT_N / 1000.0 * POWERTRAIN.source_chain_efficiency + 30.0
    fuel = fuel_contours(
        wing_loading,
        power_loading,
        AIRFRAME,
        1000.0,
        battery_kwh=10.0,
        peak_bus_kw=peak_bus,
    )
    direct = build_mass_budget(
        engine_kw=power_loading[0, 0] * WEIGHT_N / 1000.0,
        battery_kwh=10.0,
        peak_bus_kw=peak_bus[0, 0],
        wing_area_m2=WEIGHT_N / wing_loading[0, 0],
        aspect_ratio=16.0,
    )
    assert fuel.shape == wing_loading.shape
    assert fuel[0, 0] == pytest.approx(direct.fuel_kg)


def test_report_sensitivities_have_the_expected_directions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    grid = np.linspace(150.0, 1800.0, 6601)

    def power(airframe: Airframe) -> float:
        curves = constraint_curves(
            grid, design_cases(), airframe, POWERTRAIN, LAPSE_EXPONENT
        )
        return feasible_design_point(curves, stall_limit(airframe)).engine_power_sl_kw

    ar_results = {ar: power(Airframe(ar, 0.78, 0.028, 1.5, 0.85)) for ar in (12, 16, 20)}
    cd0_results = {
        cd0: power(Airframe(16.0, 0.78, cd0, 1.5, 0.85))
        for cd0 in (0.022, 0.028, 0.034)
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HighAspectRatioWarning)
        raymer_e = float(oswald_efficiency(16.0, method="raymer_straight"))
    fixed = power(AIRFRAME)
    raymer = power(Airframe(16.0, raymer_e, 0.028, 1.5, 0.85))
    print("AR sensitivity [kW]:", ar_results)
    print("CD0 sensitivity [kW]:", cd0_results)
    print("Oswald sensitivity [kW]:", {"fixed_0.78": fixed, "Raymer": raymer})
    assert ar_results[12] > ar_results[16] > ar_results[20]
    assert cd0_results[0.022] < cd0_results[0.028] < cd0_results[0.034]
    assert raymer > fixed
    assert "Oswald sensitivity" in capsys.readouterr().out


def test_report_reference_selection_and_60kw_cleanliness_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    grid = np.linspace(150.0, 1800.0, 6601)

    def point_at(cd0: float):
        airframe = Airframe(16.0, 0.78, cd0, 1.5, 0.85)
        curves = constraint_curves(
            grid, design_cases(), airframe, POWERTRAIN, LAPSE_EXPONENT
        )
        return feasible_design_point(curves, stall_limit(airframe))

    reference = point_at(0.028)
    lower_cd0, upper_cd0 = 0.0001, 0.028
    for _ in range(50):
        trial = 0.5 * (lower_cd0 + upper_cd0)
        if point_at(trial).engine_power_sl_kw <= 60.0:
            lower_cd0 = trial
        else:
            upper_cd0 = trial
    threshold = point_at(lower_cd0)
    print("selected point:", reference)
    print("60 kW CD0 threshold:", lower_cd0, threshold)
    assert reference.engine_power_sl_kw == pytest.approx(133.27046796049953)
    assert reference.binding_constraint == "ceiling_10km"
    assert lower_cd0 == pytest.approx(0.0050698368813567, rel=1e-8)
    assert threshold.engine_power_sl_kw == pytest.approx(60.0, rel=1e-12)
    assert threshold.wing_area_m2 > 35.0
    assert "60 kW CD0 threshold" in capsys.readouterr().out


def test_band_interpretation_omits_the_imposed_ceiling_and_recovers_fuel() -> None:
    grid = np.linspace(150.0, 1800.0, 6601)
    curves = constraint_curves(
        grid, design_cases()[:2], AIRFRAME, POWERTRAIN, LAPSE_EXPONENT
    )
    point = feasible_design_point(curves, stall_limit())
    peak_bus_kw = float(POWERTRAIN.bus_power_from_engine(point.engine_power_sl_kw)) + 30.0
    masses = build_mass_budget(
        engine_kw=point.engine_power_sl_kw,
        battery_kwh=10.0,
        peak_bus_kw=peak_bus_kw,
        wing_area_m2=point.wing_area_m2,
        aspect_ratio=AIRFRAME.aspect_ratio,
    )
    assert point.binding_constraint == "cruise_3km"
    assert point.wing_area_m2 == pytest.approx(7.59175537062125)
    assert point.engine_power_sl_kw == pytest.approx(86.7791369750147)
    assert masses.dry_kg == pytest.approx(711.3898016890586)
    assert masses.fuel_kg == pytest.approx(288.6101983109414)


def test_models_and_result_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        AIRFRAME.cd0 = 0.02  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        case().altitude_m = 5000.0  # type: ignore[misc]


def test_plot_saves_the_report_figure_with_both_curve_families(tmp_path: Path) -> None:
    path = tmp_path / "constraint.png"
    figure = plot_constraint_diagram(
        np.linspace(300.0, 1800.0, 601),
        design_cases(),
        AIRFRAME,
        POWERTRAIN,
        LAPSE_EXPONENT,
        v_stall_max_mps=37.5,
        output_path=path,
    )
    line_styles = {line.get_linestyle() for line in figure.axes[0].lines}
    assert path.exists() and path.stat().st_size > 0
    assert "-" in line_styles and "--" in line_styles and ":" in line_styles


@pytest.mark.parametrize(
    "bad_case",
    [
        dict(name="", altitude_m=3000.0, speed_mps=50.0, climb_rate_mps=0.0,
             battery_boost_kw=0.0, weight_n=WEIGHT_N),
        dict(name="x", altitude_m=-1.0, speed_mps=50.0, climb_rate_mps=0.0,
             battery_boost_kw=0.0, weight_n=WEIGHT_N),
        dict(name="x", altitude_m=3000.0, speed_mps=0.0, climb_rate_mps=0.0,
             battery_boost_kw=0.0, weight_n=WEIGHT_N),
        dict(name="x", altitude_m=3000.0, speed_mps=50.0, climb_rate_mps=0.0,
             battery_boost_kw=-1.0, weight_n=WEIGHT_N),
    ],
)
def test_invalid_constraint_cases_raise(bad_case: dict[str, float | str]) -> None:
    with pytest.raises(ValueError):
        ConstraintCase(**bad_case)  # type: ignore[arg-type]
