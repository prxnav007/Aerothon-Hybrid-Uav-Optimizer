"""Tests for :mod:`src.models.aerodynamics` (point-mass performance model).

There is no authoritative table for C_D0, Oswald efficiency or C_Lmax - they are
engineering estimates, not standards. So verification here is **mathematical
self-consistency**, in five tiers:

1. **Analytical identities** - the closed forms satisfy the conditions they were
   derived from (``C_Di = C_D0`` at best L/D, ``C_Di = 3*C_D0`` at min power).
2. **Numerical vs closed form** - THE PRIMARY VERIFICATION. A fine velocity
   sweep is minimised by brute force over the same drag polar, and the argmin
   must land on the analytic speed. This is independent of how the closed form
   was derived, so it catches an algebra error that tier 1 could not.
3. **Structural / limiting behaviour** - U-shaped power curve, drag asymptotes,
   W**2 scaling of induced drag, monotonic improvement with aspect ratio.
4. **Behavioural** - descent clamping, loiter-constraint switching, Oswald
   warnings, vectorization, argument guards.
5. **Hand-checked worked example** - a fully independent recomputation.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from src.models import aerodynamics as aero
from src.models import atmosphere as atm
from src.models.aerodynamics import AeroState, HighAspectRatioWarning

# ---------------------------------------------------------------------------
# Reference configuration, used throughout. NOT a design recommendation - it is
# a self-consistent test fixture. Weight is in NEWTONS (1000 kg MTOW).
# ---------------------------------------------------------------------------
REF = {
    "weight_n": 9810.0,
    "rho": 0.9091,
    "wing_area_m2": 10.0,
    "cd0": 0.028,
    "aspect_ratio": 16.0,
    "oswald_e": 0.78,
}
REF_V = 69.44  # 250 km/h cruise
REF_CL_MAX = 1.5
REF_ETA_PROP = 0.8

# Polar-only subset (no flight condition), for the speed functions.
POLAR = {k: REF[k] for k in ("cd0", "aspect_ratio", "oswald_e")}


def _polar_args() -> tuple[float, float, float, float, float, float]:
    """(W, rho, S, cd0, AR, e) in the order the speed functions expect."""
    return (
        REF["weight_n"],
        REF["rho"],
        REF["wing_area_m2"],
        REF["cd0"],
        REF["aspect_ratio"],
        REF["oswald_e"],
    )


def _random_configs(n: int = 6, seed: int = 20260803) -> list[dict[str, float]]:
    """Physically sensible random (W, S, AR, e, C_D0, rho) sets, fixed seed.

    Ranges are chosen so that V_minpower and V_bestLD both land inside the
    5-150 m/s sweep used by the numerical tests.
    """
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append(
            {
                "weight_n": float(rng.uniform(5000.0, 15000.0)),
                "rho": float(rng.uniform(0.40, 1.225)),
                "wing_area_m2": float(rng.uniform(8.0, 20.0)),
                "cd0": float(rng.uniform(0.020, 0.050)),
                "aspect_ratio": float(rng.uniform(8.0, 20.0)),
                "oswald_e": float(rng.uniform(0.60, 0.90)),
            }
        )
    return configs


RANDOM_CONFIGS = _random_configs()


def _speed_args(cfg: dict[str, float]) -> tuple[float, ...]:
    return (
        cfg["weight_n"],
        cfg["rho"],
        cfg["wing_area_m2"],
        cfg["cd0"],
        cfg["aspect_ratio"],
        cfg["oswald_e"],
    )


def _induced_cd(cfg: dict[str, float], v: float) -> float:
    """C_Di at a given speed, computed straight from the definitions."""
    cl = aero.lift_coefficient(
        cfg["weight_n"], cfg["rho"], v, cfg["wing_area_m2"]
    )
    k = aero.induced_drag_factor(cfg["aspect_ratio"], cfg["oswald_e"])
    return k * cl**2


# ===========================================================================
# 1. Analytical identity tests
# ===========================================================================


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_best_ld_speed_satisfies_induced_equals_parasite(cfg) -> None:
    """At V_bestLD the maximum-L/D condition C_Di == C_D0 must hold exactly."""
    v = aero.speed_best_ld(*_speed_args(cfg))
    assert _induced_cd(cfg, v) == pytest.approx(cfg["cd0"], rel=1e-9)


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_min_power_speed_satisfies_induced_equals_three_times_parasite(cfg) -> None:
    """At V_minpower the minimum-power condition C_Di == 3*C_D0 must hold.

    This is the factor 3 that puts the 3 in ``(3*C_D0*pi*AR*e)**-0.25``.
    """
    v = aero.speed_min_power(*_speed_args(cfg))
    assert _induced_cd(cfg, v) == pytest.approx(3.0 * cfg["cd0"], rel=1e-9)


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_min_power_speed_is_three_to_the_minus_quarter_of_best_ld(cfg) -> None:
    v_mp = aero.speed_min_power(*_speed_args(cfg))
    v_ld = aero.speed_best_ld(*_speed_args(cfg))
    assert v_mp == pytest.approx(v_ld * 3.0**-0.25, rel=1e-12)
    # ~0.7598, the textbook "76 % of the minimum-drag speed".
    assert v_mp / v_ld == pytest.approx(0.759836, rel=1e-5)


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_lift_to_drag_at_best_ld_speed_equals_max_lift_to_drag(cfg) -> None:
    v = aero.speed_best_ld(*_speed_args(cfg))
    achieved = aero.lift_to_drag(
        cfg["weight_n"], cfg["rho"], v, cfg["wing_area_m2"],
        cfg["cd0"], cfg["aspect_ratio"], cfg["oswald_e"],
    )
    expected = aero.max_lift_to_drag(
        cfg["cd0"], cfg["aspect_ratio"], cfg["oswald_e"]
    )
    assert achieved == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_cl_at_best_ld_equals_sqrt_cd0_over_k(cfg) -> None:
    """C_L_bestLD == sqrt(C_D0/k), the closed form the speed derives from."""
    v = aero.speed_best_ld(*_speed_args(cfg))
    cl = aero.lift_coefficient(
        cfg["weight_n"], cfg["rho"], v, cfg["wing_area_m2"]
    )
    k = aero.induced_drag_factor(cfg["aspect_ratio"], cfg["oswald_e"])
    assert cl == pytest.approx(np.sqrt(cfg["cd0"] / k), rel=1e-12)


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_drag_coefficient_doubles_at_best_ld_and_quadruples_at_min_power(cfg) -> None:
    """C_D == 2*C_D0 at best L/D and C_D == 4*C_D0 at minimum power.

    Direct consequences of C_Di = C_D0 and C_Di = 3*C_D0 respectively; both are
    standard textbook results, so they pin the conditions from a second angle.
    """
    for speed_fn, factor in (
        (aero.speed_best_ld, 2.0),
        (aero.speed_min_power, 4.0),
    ):
        v = speed_fn(*_speed_args(cfg))
        cl = aero.lift_coefficient(
            cfg["weight_n"], cfg["rho"], v, cfg["wing_area_m2"]
        )
        cd = aero.drag_coefficient(
            cl, cfg["cd0"], cfg["aspect_ratio"], cfg["oswald_e"]
        )
        assert cd == pytest.approx(factor * cfg["cd0"], rel=1e-9)


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_lift_to_drag_at_min_power_is_sqrt3_over_2_of_max(cfg) -> None:
    """(L/D)_minpower == sqrt(3)/2 * (L/D)_max ~ 0.866.

    Independently confirmed against published performance courseware; a useful
    check because it mixes the two conditions rather than testing each alone.
    """
    v = aero.speed_min_power(*_speed_args(cfg))
    achieved = aero.lift_to_drag(
        cfg["weight_n"], cfg["rho"], v, cfg["wing_area_m2"],
        cfg["cd0"], cfg["aspect_ratio"], cfg["oswald_e"],
    )
    ld_max = aero.max_lift_to_drag(
        cfg["cd0"], cfg["aspect_ratio"], cfg["oswald_e"]
    )
    assert achieved == pytest.approx(np.sqrt(3.0) / 2.0 * ld_max, rel=1e-9)


def test_max_lift_to_drag_is_independent_of_weight_density_and_area() -> None:
    """(L/D)max is a property of the polar alone - it takes no flight condition."""
    ld = aero.max_lift_to_drag(**POLAR)
    for w, rho, s in [(5000.0, 0.4, 8.0), (15000.0, 1.225, 25.0)]:
        v = aero.speed_best_ld(w, rho, s, **POLAR)
        achieved = aero.lift_to_drag(w, rho, v, s, **POLAR)
        assert achieved == pytest.approx(ld, rel=1e-9)


# ===========================================================================
# 2. Numerical-vs-closed-form tests  (the strongest verification)
# ===========================================================================

# 5 to 150 m/s in 0.01 m/s steps.
SWEEP_STEP = 0.01
V_SWEEP = np.arange(5.0, 150.0 + SWEEP_STEP, SWEEP_STEP)


def _sweep_drag_and_power(cfg: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Drag and thrust power over the full velocity sweep, vectorized."""
    d = aero.drag_force(
        cfg["weight_n"], cfg["rho"], V_SWEEP, cfg["wing_area_m2"],
        cfg["cd0"], cfg["aspect_ratio"], cfg["oswald_e"],
    )
    return d, d * V_SWEEP


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_numerical_min_of_power_matches_speed_min_power(cfg) -> None:
    """Brute-force argmin of D*V must land on the analytic V_minpower.

    Fully independent of the derivation: it only uses the drag polar and a grid
    search. If the closed form's algebra were wrong, this would catch it.
    """
    _, power = _sweep_drag_and_power(cfg)
    v_numerical = V_SWEEP[int(np.argmin(power))]
    v_closed_form = aero.speed_min_power(*_speed_args(cfg))
    assert abs(v_numerical - v_closed_form) <= SWEEP_STEP


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_numerical_min_of_drag_matches_speed_best_ld(cfg) -> None:
    """Brute-force argmin of D must land on the analytic V_bestLD."""
    drag, _ = _sweep_drag_and_power(cfg)
    v_numerical = V_SWEEP[int(np.argmin(drag))]
    v_closed_form = aero.speed_best_ld(*_speed_args(cfg))
    assert abs(v_numerical - v_closed_form) <= SWEEP_STEP


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_numerical_max_lift_to_drag_matches_closed_form(cfg) -> None:
    """Brute-force max of C_L/C_D over the sweep must match (L/D)max."""
    ld = aero.lift_to_drag(
        cfg["weight_n"], cfg["rho"], V_SWEEP, cfg["wing_area_m2"],
        cfg["cd0"], cfg["aspect_ratio"], cfg["oswald_e"],
    )
    expected = aero.max_lift_to_drag(
        cfg["cd0"], cfg["aspect_ratio"], cfg["oswald_e"]
    )
    assert float(np.max(ld)) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_argmin_of_drag_and_of_power_are_distinct_and_correctly_ordered(cfg) -> None:
    """V_minpower < V_bestLD, numerically - the two optima really do differ.

    Guards against a copy-paste error that made both formulas identical, which
    the individual argmin tests would each still pass if the *same* wrong
    formula were used in both.
    """
    drag, power = _sweep_drag_and_power(cfg)
    v_min_power = V_SWEEP[int(np.argmin(power))]
    v_min_drag = V_SWEEP[int(np.argmin(drag))]
    assert v_min_power < v_min_drag
    assert v_min_power / v_min_drag == pytest.approx(3.0**-0.25, rel=1e-3)


# ===========================================================================
# 3. Structural / limiting-behaviour tests
# ===========================================================================


@pytest.mark.parametrize("cfg", RANDOM_CONFIGS, ids=lambda c: f"AR{c['aspect_ratio']:.1f}")
def test_power_curve_is_u_shaped_with_exactly_one_interior_minimum(cfg) -> None:
    """P(V) strictly decreases below V_minpower and strictly increases above it."""
    _, power = _sweep_drag_and_power(cfg)
    v_min = aero.speed_min_power(*_speed_args(cfg))

    below = V_SWEEP < v_min
    above = V_SWEEP > v_min
    assert np.all(np.diff(power[below]) < 0.0), "power must fall below V_minpower"
    assert np.all(np.diff(power[above]) > 0.0), "power must rise above V_minpower"

    # Exactly one sign change in the derivative => exactly one interior minimum.
    sign_changes = np.count_nonzero(np.diff(np.sign(np.diff(power))) != 0)
    assert sign_changes == 1


def test_drag_diverges_at_both_ends_of_the_speed_range() -> None:
    """Induced-dominated as V -> 0, parasite-dominated as V -> large."""
    slow = aero.drag_force(REF["weight_n"], REF["rho"], 1.0, REF["wing_area_m2"],
                           REF["cd0"], REF["aspect_ratio"], REF["oswald_e"])
    middle = aero.drag_force(REF["weight_n"], REF["rho"], 45.39, REF["wing_area_m2"],
                             REF["cd0"], REF["aspect_ratio"], REF["oswald_e"])
    fast = aero.drag_force(REF["weight_n"], REF["rho"], 400.0, REF["wing_area_m2"],
                           REF["cd0"], REF["aspect_ratio"], REF["oswald_e"])
    assert slow > 100.0 * middle
    assert fast > 10.0 * middle

    # And the composition flips: induced dominates slow, parasite dominates fast.
    k = aero.induced_drag_factor(REF["aspect_ratio"], REF["oswald_e"])
    for v, induced_should_dominate in ((1.0, True), (400.0, False)):
        cl = aero.lift_coefficient(REF["weight_n"], REF["rho"], v, REF["wing_area_m2"])
        assert (k * cl**2 > REF["cd0"]) is induced_should_dominate


def test_induced_drag_scales_as_weight_squared_at_fixed_dynamic_pressure() -> None:
    """Doubling weight quadruples the induced drag term (C_L ~ W, C_Di ~ C_L**2)."""
    k = aero.induced_drag_factor(REF["aspect_ratio"], REF["oswald_e"])
    q = aero.dynamic_pressure(REF["rho"], REF_V)

    def induced_drag(w: float) -> float:
        cl = aero.lift_coefficient(w, REF["rho"], REF_V, REF["wing_area_m2"])
        return q * REF["wing_area_m2"] * k * cl**2

    base = induced_drag(REF["weight_n"])
    assert induced_drag(2.0 * REF["weight_n"]) == pytest.approx(4.0 * base, rel=1e-12)
    assert induced_drag(3.0 * REF["weight_n"]) == pytest.approx(9.0 * base, rel=1e-12)


def test_parasite_drag_is_independent_of_weight() -> None:
    q = aero.dynamic_pressure(REF["rho"], REF_V)
    parasite = q * REF["wing_area_m2"] * REF["cd0"]
    for w in (1000.0, 9810.0, 50000.0):
        total = aero.drag_force(w, REF["rho"], REF_V, REF["wing_area_m2"],
                                REF["cd0"], REF["aspect_ratio"], REF["oswald_e"])
        cl = aero.lift_coefficient(w, REF["rho"], REF_V, REF["wing_area_m2"])
        k = aero.induced_drag_factor(REF["aspect_ratio"], REF["oswald_e"])
        induced = q * REF["wing_area_m2"] * k * cl**2
        assert total - induced == pytest.approx(parasite, rel=1e-12)


def test_increasing_aspect_ratio_strictly_reduces_induced_drag() -> None:
    ars = np.array([6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0])
    k = aero.induced_drag_factor(ars, REF["oswald_e"])
    assert np.all(np.diff(k) < 0.0)

    cl = aero.lift_coefficient(REF["weight_n"], REF["rho"], REF_V, REF["wing_area_m2"])
    induced_cd = k * cl**2
    assert np.all(np.diff(induced_cd) < 0.0)

    # ... and therefore strictly improves (L/D)max.
    ld_max = aero.max_lift_to_drag(REF["cd0"], ars, REF["oswald_e"])
    assert np.all(np.diff(ld_max) > 0.0)


def test_increasing_cd0_strictly_reduces_max_lift_to_drag() -> None:
    cd0s = np.array([0.015, 0.020, 0.028, 0.035, 0.050])
    ld_max = aero.max_lift_to_drag(cd0s, REF["aspect_ratio"], REF["oswald_e"])
    assert np.all(np.diff(ld_max) < 0.0)


# ===========================================================================
# 4. Behavioural tests
# ===========================================================================

# --- Descent / power clamping ----------------------------------------------


def test_steep_descent_returns_zero_shaft_power_and_power_off_flag() -> None:
    """W=9810 N, V=65 m/s, ROC=-6.33 m/s at 3000 m: gravity outpaces drag.

    Density comes from the ISA module at 3000 m, so this exercises the real
    interface the mission simulator will use.
    """
    rho_3km = atm.density(3000.0)

    p_thrust = aero.thrust_power_required(
        REF["weight_n"], rho_3km, 65.0, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], climb_rate_mps=-6.33,
    )
    # Unclamped thrust power is genuinely negative here (~ -18.8 kW).
    assert p_thrust < 0.0

    p_shaft, power_off = aero.shaft_power_required(
        REF["weight_n"], rho_3km, 65.0, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], REF_ETA_PROP,
        climb_rate_mps=-6.33,
    )
    assert p_shaft == 0.0
    assert power_off is True


def test_no_negative_shaft_power_across_a_descent_sweep() -> None:
    """Across every descent rate, shaft power is >= 0 and the flag is consistent."""
    climb_rates = np.linspace(-25.0, 10.0, 701)
    p_shaft, power_off = aero.shaft_power_required(
        REF["weight_n"], REF["rho"], 65.0, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], REF_ETA_PROP,
        climb_rate_mps=climb_rates,
    )
    assert np.all(p_shaft >= 0.0), "shaft power must never be negative"

    # The flag must mark exactly the clamped points, and nothing else.
    p_thrust = aero.thrust_power_required(
        REF["weight_n"], REF["rho"], 65.0, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"],
        climb_rate_mps=climb_rates,
    )
    np.testing.assert_array_equal(power_off, p_thrust <= 0.0)
    assert np.all(p_shaft[power_off] == 0.0)
    assert np.all(p_shaft[~power_off] > 0.0)
    # Sanity: the sweep actually straddles the boundary.
    assert power_off.any() and (~power_off).any()


def test_shaft_power_is_thrust_power_over_efficiency_when_powered() -> None:
    p_thrust = aero.thrust_power_required(
        REF["weight_n"], REF["rho"], REF_V, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"],
    )
    p_shaft, power_off = aero.shaft_power_required(
        REF["weight_n"], REF["rho"], REF_V, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], REF_ETA_PROP,
    )
    assert power_off is False
    assert p_shaft == pytest.approx(p_thrust / REF_ETA_PROP, rel=1e-12)
    assert p_shaft > p_thrust  # eta < 1 means more shaft power than thrust power


def test_climb_costs_exactly_the_potential_energy_rate() -> None:
    """P_thrust(climb) - P_thrust(level) == W * ROC, by construction."""
    level = aero.thrust_power_required(
        REF["weight_n"], REF["rho"], REF_V, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], climb_rate_mps=0.0,
    )
    climbing = aero.thrust_power_required(
        REF["weight_n"], REF["rho"], REF_V, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], climb_rate_mps=3.0,
    )
    assert climbing - level == pytest.approx(REF["weight_n"] * 3.0, rel=1e-12)


def test_rate_of_climb_is_consistent_with_power_required() -> None:
    """The ROC that a given power buys, fed back, reproduces that power."""
    p_avail = 100_000.0
    roc = aero.rate_of_climb(
        REF["weight_n"], REF["rho"], REF_V, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], REF_ETA_PROP, p_avail,
    )
    p_shaft, _ = aero.shaft_power_required(
        REF["weight_n"], REF["rho"], REF_V, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], REF_ETA_PROP,
        climb_rate_mps=roc,
    )
    assert p_shaft == pytest.approx(p_avail, rel=1e-12)


def test_rate_of_climb_goes_negative_when_power_is_insufficient() -> None:
    roc = aero.rate_of_climb(
        REF["weight_n"], REF["rho"], REF_V, REF["wing_area_m2"],
        REF["cd0"], REF["aspect_ratio"], REF["oswald_e"], REF_ETA_PROP, 10_000.0,
    )
    assert roc < 0.0  # a sink rate, deliberately not clamped


# --- Loiter constraint switching -------------------------------------------


def test_loiter_is_stall_margin_limited_for_the_high_ar_reference_config() -> None:
    """AR=16, C_D0=0.028, C_Lmax=1.5: the min-power C_L of 1.81 is unreachable."""
    v, active = aero.loiter_speed(*_polar_args(), cl_max=REF_CL_MAX)
    assert active == "stall_margin"

    # The cap really is what set the speed: C_L == C_Lmax / 1.2**2.
    cl = aero.lift_coefficient(REF["weight_n"], REF["rho"], v, REF["wing_area_m2"])
    assert cl == pytest.approx(REF_CL_MAX / 1.2**2, rel=1e-12)

    # And the unconstrained optimum was indeed out of reach.
    cl_min_power = np.sqrt(3.0 * REF["cd0"] * np.pi * REF["aspect_ratio"] * REF["oswald_e"])
    assert cl_min_power > REF_CL_MAX


def test_loiter_is_min_power_limited_for_a_low_ar_high_drag_config() -> None:
    """Low AR, high C_D0, high C_Lmax: the aerodynamic optimum is achievable."""
    cfg = {"weight_n": 9810.0, "rho": 0.9091, "wing_area_m2": 10.0,
           "cd0": 0.040, "aspect_ratio": 5.0, "oswald_e": 0.75}
    cl_max = 2.4

    v, active = aero.loiter_speed(*_speed_args(cfg), cl_max=cl_max)
    assert active == "min_power"

    # It genuinely switched: min-power C_L is below the stall cap here.
    cl_min_power = np.sqrt(3.0 * cfg["cd0"] * np.pi * cfg["aspect_ratio"] * cfg["oswald_e"])
    assert cl_min_power < cl_max / 1.2**2
    # ... and the loiter speed equals the unconstrained min-power speed.
    assert v == pytest.approx(aero.speed_min_power(*_speed_args(cfg)), rel=1e-12)


def test_loiter_speed_never_below_the_stall_margin() -> None:
    """Whichever constraint binds, the result respects the margin over stall."""
    for cfg, cl_max in [
        (REF | {"aspect_ratio": 16.0}, 1.5),
        ({"weight_n": 9810.0, "rho": 0.9091, "wing_area_m2": 10.0,
          "cd0": 0.040, "aspect_ratio": 5.0, "oswald_e": 0.75}, 2.4),
    ]:
        args = _speed_args(cfg)
        v, _ = aero.loiter_speed(*args, cl_max=cl_max)
        v_stall = aero.stall_speed(cfg["weight_n"], cfg["rho"],
                                   cfg["wing_area_m2"], cl_max)
        assert v >= v_stall * 1.2 - 1e-9


def test_loiter_speed_is_vectorized_over_the_constraint_switch() -> None:
    """Array input must return an array of constraint labels, not one label."""
    ars = np.array([5.0, 8.0, 12.0, 16.0, 20.0])
    v, active = aero.loiter_speed(
        REF["weight_n"], REF["rho"], REF["wing_area_m2"], 0.040, ars, 0.75,
        cl_max=2.4,
    )
    assert isinstance(active, np.ndarray)
    assert active.shape == ars.shape
    # Low AR can reach its optimum; high AR is stall-capped.
    assert active[0] == "min_power"
    assert active[-1] == "stall_margin"
    # Once capped, the speed stops varying with AR (it is set by C_Lmax alone).
    capped = active == "stall_margin"
    assert np.allclose(v[capped], v[capped][0])


def test_loiter_safety_margin_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="safety_margin"):
        aero.loiter_speed(*_polar_args(), cl_max=REF_CL_MAX, safety_margin=0.9)


# --- Oswald efficiency ------------------------------------------------------


def test_raymer_straight_correlation_values() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HighAspectRatioWarning)
        assert aero.oswald_efficiency(10.0) == pytest.approx(0.76, abs=0.005)
        assert aero.oswald_efficiency(16.0) == pytest.approx(0.61, abs=0.005)
        assert aero.oswald_efficiency(20.0) == pytest.approx(0.53, abs=0.005)


def test_high_aspect_ratio_warning_fires_above_twelve_only() -> None:
    """AR 16 warns that the correlation is extrapolating; AR 10 does not."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        aero.oswald_efficiency(10.0)  # must NOT warn

    with pytest.warns(HighAspectRatioWarning, match="pessimistic"):
        aero.oswald_efficiency(16.0)


def test_high_aspect_ratio_warning_fires_for_arrays_containing_high_ar() -> None:
    with pytest.warns(HighAspectRatioWarning):
        aero.oswald_efficiency(np.array([8.0, 10.0, 18.0]))


def test_constant_method_returns_the_supplied_value_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # No warning even at AR 20 - the caller has taken responsibility.
        e = aero.oswald_efficiency(20.0, method="constant", constant_value=0.82)
    assert e == 0.82


def test_constant_method_requires_a_value() -> None:
    with pytest.raises(ValueError, match="constant_value"):
        aero.oswald_efficiency(12.0, method="constant")


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        aero.oswald_efficiency(10.0, method="schrenk")


def test_swept_correlation_raises_rather_than_returning_negative_e() -> None:
    """At AR 20 and zero sweep the swept correlation goes negative.

    A negative e would flip the sign of induced drag - the optimiser would find
    a wing that produces thrust by lifting. It must raise, not return.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HighAspectRatioWarning)
        with pytest.raises(ValueError, match="non-positive Oswald"):
            aero.oswald_efficiency(20.0, sweep_rad=0.0, method="raymer_swept")


def test_swept_correlation_is_usable_in_its_fitted_range() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HighAspectRatioWarning)
        e = aero.oswald_efficiency(
            8.0, sweep_rad=np.deg2rad(35.0), method="raymer_swept"
        )
    assert 0.0 < e < 1.0


def test_oswald_efficiency_feeds_the_polar_consistently() -> None:
    """End-to-end: a correlation value flows into k without special handling."""
    e = aero.oswald_efficiency(10.0)
    k = aero.induced_drag_factor(10.0, e)
    assert k == pytest.approx(1.0 / (np.pi * 10.0 * e), rel=1e-12)


# --- Parasite drag buildup --------------------------------------------------


def test_wetted_area_method_ties_cd0_to_geometry() -> None:
    """Shrinking the wing at fixed wetted area RAISES C_D0 - no free lunch."""
    cd0_big = aero.parasite_drag_from_wetted_area(51.0, 10.0)
    cd0_small = aero.parasite_drag_from_wetted_area(51.0, 5.0)
    assert cd0_small == pytest.approx(2.0 * cd0_big, rel=1e-12)

    # And the parasite drag force is then unchanged by the wing area change,
    # which is exactly the unphysical free lunch this exists to prevent.
    q = aero.dynamic_pressure(REF["rho"], REF_V)
    assert q * 10.0 * cd0_big == pytest.approx(q * 5.0 * cd0_small, rel=1e-12)


def test_wetted_area_method_matches_the_reference_cd0() -> None:
    assert aero.parasite_drag_from_wetted_area(51.0, 10.0) == pytest.approx(
        0.02805, rel=1e-12
    )


# --- Vectorization and typing ----------------------------------------------


def test_scalar_inputs_return_python_scalars() -> None:
    st = aero.evaluate(**REF, v_mps=REF_V, eta_prop=REF_ETA_PROP)
    assert isinstance(st, AeroState)
    for name, value in vars(st).items():
        expected = bool if name == "power_off" else float
        assert isinstance(value, expected), f"{name} should be {expected.__name__}"
        assert not isinstance(value, np.ndarray)


def test_array_input_returns_arrays_of_matching_shape() -> None:
    v = np.linspace(30.0, 120.0, 25)
    st = aero.evaluate(**REF, v_mps=v, eta_prop=REF_ETA_PROP)
    for name, value in vars(st).items():
        assert isinstance(value, np.ndarray), name
        assert value.shape == v.shape, name


def test_evaluate_broadcasts_mixed_scalar_and_array_arguments() -> None:
    """Scalar fields are broadcast up so consumers can index every field alike."""
    v = np.linspace(40.0, 90.0, 7)
    st = aero.evaluate(**REF, v_mps=v, eta_prop=REF_ETA_PROP, climb_rate_mps=2.0)
    for value in vars(st).values():
        assert np.shape(value) == v.shape


@pytest.mark.parametrize(
    "func",
    [aero.drag_force, aero.lift_to_drag, aero.thrust_power_required],
    ids=lambda f: f.__name__,
)
def test_vectorized_equals_elementwise(func) -> None:
    v = np.linspace(20.0, 140.0, 61)
    vectorized = func(REF["weight_n"], REF["rho"], v, REF["wing_area_m2"],
                      REF["cd0"], REF["aspect_ratio"], REF["oswald_e"])
    elementwise = np.array([
        func(REF["weight_n"], REF["rho"], float(x), REF["wing_area_m2"],
             REF["cd0"], REF["aspect_ratio"], REF["oswald_e"])
        for x in v
    ])
    np.testing.assert_allclose(vectorized, elementwise, rtol=0.0, atol=0.0)


def test_evaluate_vectorized_equals_elementwise() -> None:
    v = np.linspace(20.0, 140.0, 41)
    batch = aero.evaluate(**REF, v_mps=v, eta_prop=REF_ETA_PROP)
    for i, x in enumerate(v):
        single = aero.evaluate(**REF, v_mps=float(x), eta_prop=REF_ETA_PROP)
        assert batch.drag_n[i] == single.drag_n
        assert batch.shaft_power_w[i] == single.shaft_power_w
        assert bool(batch.power_off[i]) == single.power_off


def test_evaluate_agrees_with_the_individual_functions() -> None:
    v = np.linspace(30.0, 120.0, 37)
    st = aero.evaluate(**REF, v_mps=v, eta_prop=REF_ETA_PROP)
    args = (REF["weight_n"], REF["rho"], v, REF["wing_area_m2"],
            REF["cd0"], REF["aspect_ratio"], REF["oswald_e"])
    np.testing.assert_allclose(
        st.dynamic_pressure_pa, aero.dynamic_pressure(REF["rho"], v), rtol=0, atol=0)
    np.testing.assert_allclose(
        st.lift_coefficient,
        aero.lift_coefficient(REF["weight_n"], REF["rho"], v, REF["wing_area_m2"]),
        rtol=0, atol=0)
    np.testing.assert_allclose(st.drag_n, aero.drag_force(*args), rtol=0, atol=0)
    np.testing.assert_allclose(st.lift_to_drag, aero.lift_to_drag(*args), rtol=0, atol=0)
    np.testing.assert_allclose(
        st.thrust_power_w, aero.thrust_power_required(*args), rtol=0, atol=0)


def test_aero_state_is_frozen() -> None:
    st = aero.evaluate(**REF, v_mps=REF_V, eta_prop=REF_ETA_PROP)
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.drag_n = 0.0  # type: ignore[misc]


# --- Argument guards --------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_arg", "bad_value"),
    [
        ("v_mps", 0.0),
        ("wing_area_m2", 0.0),
        ("aspect_ratio", 0.0),
        ("oswald_e", 0.0),
        ("rho", 0.0),
        ("weight_n", 0.0),
        ("cd0", 0.0),
        ("v_mps", -1.0),
        ("oswald_e", -0.5),
        ("rho", np.nan),
    ],
)
def test_evaluate_rejects_non_positive_arguments(bad_arg, bad_value) -> None:
    kwargs = {**REF, "v_mps": REF_V, "eta_prop": REF_ETA_PROP, bad_arg: bad_value}
    with pytest.raises(ValueError, match=bad_arg):
        aero.evaluate(**kwargs)


def test_error_message_names_the_offending_argument_and_value() -> None:
    with pytest.raises(ValueError) as excinfo:
        aero.evaluate(**{**REF, "aspect_ratio": 0.0}, v_mps=REF_V,
                      eta_prop=REF_ETA_PROP)
    message = str(excinfo.value)
    assert "aspect_ratio" in message
    assert "0.0" in message


@pytest.mark.parametrize("bad_eta", [0.0, -0.1, 1.5, np.nan])
def test_efficiency_outside_unit_interval_is_rejected(bad_eta) -> None:
    with pytest.raises(ValueError, match="eta_prop"):
        aero.evaluate(**REF, v_mps=REF_V, eta_prop=bad_eta)


def test_non_finite_climb_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="climb_rate_mps"):
        aero.evaluate(**REF, v_mps=REF_V, eta_prop=REF_ETA_PROP,
                      climb_rate_mps=np.inf)


def test_one_bad_element_in_an_array_raises() -> None:
    with pytest.raises(ValueError, match="v_mps"):
        aero.evaluate(**REF, v_mps=np.array([50.0, 0.0, 70.0]),
                      eta_prop=REF_ETA_PROP)


# ===========================================================================
# 5. Hand-checked worked example  (independently recomputed)
# ===========================================================================


def test_worked_example_flight_condition() -> None:
    """W=9810 N, S=10 m^2, AR=16, e=0.78, C_D0=0.028, rho=0.9091, V=69.44 m/s.

    Every figure below was recomputed from first principles before being
    written down, and agrees with the values supplied for this task to within
    their quoted precision (largest deviation 1.6e-4 relative, on P_thrust,
    which is 3-significant-figure rounding of 50.392 kW to "50.4 kW").
    """
    st = aero.evaluate(**REF, v_mps=REF_V, eta_prop=REF_ETA_PROP)

    assert st.dynamic_pressure_pa == pytest.approx(2191.8, rel=1e-4)
    assert st.lift_coefficient == pytest.approx(0.4476, rel=1e-3)
    assert st.drag_coefficient == pytest.approx(0.033110, rel=1e-3)
    assert st.drag_n == pytest.approx(725.7, rel=1e-3)
    assert st.lift_to_drag == pytest.approx(13.52, rel=1e-3)
    assert st.thrust_power_w == pytest.approx(50.4e3, rel=1e-3)

    k = aero.induced_drag_factor(REF["aspect_ratio"], REF["oswald_e"])
    assert k == pytest.approx(0.025506, rel=1e-4)


def test_worked_example_characteristic_speeds() -> None:
    assert aero.max_lift_to_drag(**POLAR) == pytest.approx(18.71, rel=1e-3)
    assert aero.speed_best_ld(*_polar_args()) == pytest.approx(45.39, rel=1e-3)
    assert aero.speed_min_power(*_polar_args()) == pytest.approx(34.49, rel=1e-3)
    assert aero.stall_speed(
        REF["weight_n"], REF["rho"], REF["wing_area_m2"], REF_CL_MAX
    ) == pytest.approx(37.93, rel=1e-3)

    # C_L at best L/D
    v = aero.speed_best_ld(*_polar_args())
    cl = aero.lift_coefficient(REF["weight_n"], REF["rho"], v, REF["wing_area_m2"])
    assert cl == pytest.approx(1.0478, rel=1e-3)


def test_min_power_speed_lies_below_stall_for_the_reference_configuration() -> None:
    """The diagnostic that motivates the whole loiter solver.

    V_minpower (34.49 m/s) < V_stall (37.93 m/s), so the theoretical endurance
    optimum is unflyable for this wing and the stall-margin cap MUST bind.
    """
    v_min_power = aero.speed_min_power(*_polar_args())
    v_stall = aero.stall_speed(
        REF["weight_n"], REF["rho"], REF["wing_area_m2"], REF_CL_MAX
    )
    assert v_min_power < v_stall

    _, active = aero.loiter_speed(*_polar_args(), cl_max=REF_CL_MAX)
    assert active == "stall_margin"


# ===========================================================================
# Cross-module assumption check
# ===========================================================================


def test_cruise_mach_justifies_incompressible_model() -> None:
    """Assert - not assume - that the incompressible model is valid here.

    This module applies no compressibility correction. That is only legitimate
    below M ~ 0.3. Computed from the ISA module at the 6 km cruise altitude
    rather than hard-coded, so a future change to cruise altitude or speed that
    invalidates the assumption fails here instead of silently biasing the drag.
    """
    mach = atm.mach_number(REF_V, 6000.0)
    assert mach == pytest.approx(0.22, abs=0.005)
    assert mach < 0.3, "incompressible aerodynamics no longer valid"

    # Valid across the whole 3-10 km cruise band at this speed, too.
    machs = atm.mach_number(REF_V, np.linspace(3000.0, 10000.0, 50))
    assert np.all(machs < 0.3)
