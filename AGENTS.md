# AGENTS.md

Context for coding agents and new contributors. It records what you **cannot** learn by reading the
code. Everything here that the source can tell you faster has been left out on purpose — no API
listings, no paraphrased docstrings, no background on genetic algorithms or ECMS.

## 1. What this project is

Aerothon (IIT Indore with HAL), Problem Statement 1: design and optimize a hybrid-electric
propulsion system for a 1000 kg fixed-wing UAV. Objective is **maximum endurance**, and everything
in the repo exists to serve it. The architecture is a **series hybrid**: a turboshaft drives a
generator only, a battery sits on the DC bus, and all thrust is electric. A genetic algorithm sizes
the hardware and tunes the energy-management controller together, scored by a time-marching mission
simulation whose loiter phase runs until fuel reserve or SoC cutoff.

## 2. Orientation

```
src/models/       physics primitives — atmosphere, aerodynamics, mass, engine, battery, powertrain
src/control/      energy-management controllers (fixed / PI / fuzzy ECMS) and the power split
src/simulation/   mission profile, time-marching simulator, feasibility rollup
src/optimization/ chromosome, fitness, GA driver, sensitivity
src/analysis/     Breguet cross-check, constraint diagram, baseline-vs-optimized benchmark
src/dashboard/    Streamlit app  ← visualization track, see §8
docs/assumptions.md   the single source of truth for every assumed constant
configs/          mission, UAV specs, GA settings, named scenarios (YAML)
tests/            one test module per source module
notebooks/        exploratory only; nothing imports from here
data/, deliverables/   empty with .gitkeep; outputs land here, nothing is committed
```

**Dependency order.** `models/` → `control/` → `simulation/` → `optimization/` → `analysis/` and
`dashboard/`. Build in that order; nothing lower may import anything higher.

**Modules under `src/models/` do not import each other.** Every coupling arrives as an argument:
`engine.py` takes a density ratio, never an altitude; `aerodynamics.py` takes a density, never an
`AtmosphericState`; `mass.py` takes a peak bus power that `powertrain.py` computed. Keep it that
way — it is what lets each model be tested standalone and what keeps the GA free to make wing area,
aspect ratio and altitude design variables without touching a physics file. None of them import
`src/config.py` either.

## 3. Build status — as of 2026-08-05

**Verify this against the filesystem before trusting it; it is the first section to go stale.**

| Area | State |
|---|---|
| `models/atmosphere.py`, `aerodynamics.py`, `mass.py`, `engine.py`, `battery.py`, `powertrain.py` | complete, with test modules |
| `control/base.py`, `tests/test_control_base.py` | complete; shared controller interface and reusable contract suite |
| `tests/test_controllers.py`, `tests/test_mission.py` | **empty files** |
| Remaining `control/`, plus `simulation/`, `optimization/`, `analysis/`, `dashboard/`, `src/config.py` | **empty files** — directory skeleton only |
| `configs/*.yaml`, `run.sh`, `README.md` | **empty files** |
| `notebooks/*.ipynb` | placeholder JSON |

Caveat on "passes tests": pytest is **not installed** in the project venv (`aeroprjct/`, Python
3.13), and the committed `.pyc` artifacts show the suite was last run under a Python 3.14
environment that is not on this machine. What was verified directly here is that all six model
modules import cleanly and reproduce the figures quoted in `docs/assumptions.md` — wing mass 119.58
vs 118.72 kg for the two independent laws (M-03/M-03b), chain efficiency 0.80636 and 37.204 kW of
engine shaft power for 30 kW of demand (P-01), 86.79 A at the battery reference condition (B-03),
45.71 kW lapsed rating at 6 km (E-04). Install pytest before claiming the suite is green.

## 4. Non-negotiable conventions

These took iteration to settle and will be violated by default.

### Documentation style

Roughly **one comment line per 8–10 lines of code**. One short comment per equation block, naming
the equation and nothing else. No worked examples, no validity ranges, no derivations, and no source
comparisons in docstrings. Assumption *rationale* lives in `docs/assumptions.md` and is referenced
by ID in a one-line comment (`# see assumptions.md M-04`) — never restated in the code. Prefer a
named constant to a comment explaining a number.

An earlier iteration of this project produced files at roughly ten comment lines per two code lines
and they had to be stripped by hand. **`atmosphere.py` and `aerodynamics.py` predate the rule and
still carry the old style** — they are the counterexample, not the template. Copy the density of
`mass.py`, `engine.py`, `battery.py` or `powertrain.py`.

### Units and sign conventions

SI at every API boundary. The silent-failure risks, in order:

- **Weight is in newtons, not kilograms.** A 1000 kg vehicle is `weight_n = 9810.0`. Passing kg
  understates induced drag ~9.81× and produces a wildly optimistic endurance with no error.
- **`aerodynamics.py` returns power in WATTS. `engine.py`, `battery.py`, `powertrain.py` and
  `mass.py` all work in kW.** The aero↔powertrain seam is the one place a factor of 1000 can cross
  silently; convert explicitly at the call site.
- **Altitude is geopotential** everywhere in `atmosphere.py` and in the mission profile. Convert
  geometric altitude with `geometric_to_geopotential` first. Mixing them is a 0.16 % error at 10 km
  that never raises.
- **Battery bus power is positive on discharge, negative on charge**, in `battery.py` and
  `powertrain.py` alike. Battery *current* carries the same sign.
- Fuel flow is kg/s at the boundary and SFC is kg/kWh; the Willans arithmetic runs in kg/h
  internally. Climb rate is positive up. Angles are radians. Density ratio σ is dimensionless in
  (0, 1].

### Purity

Every model is a **frozen dataclass or a free function**, and every state object is frozen. No
mutable state is carried between calls: state of charge is passed in and returned, never stored on
`BatteryPack`. The reason is the energy-management controller — it evaluates candidate power splits
to price them, and rejects most of them. A model that mutated itself on evaluation would corrupt its
own search, which is exactly the bug the previous stateful `Battery` class had.

### Vectorization

`atmosphere.atmosphere`, `aerodynamics.evaluate`, and the raw evaluation functions in `engine.py`,
`battery.py` and `powertrain.py` must accept ndarrays and return arrays of the broadcast shape;
scalar input returns a Python `float`/`bool`/`str`. They sit in the GA's inner loop and run 1e6–1e7
times, where the scalar path costs ~450× more per point than the array path. The `_as_array` /
`_restore_scalar` pair in each module is the shared idiom. Step and solve methods
(`battery.step`, `engine.operate`, `powertrain.solve`) are deliberately scalar — they are per-timestep.

### Clamp and flag, never raise

Models report **infeasibility** through boolean fields on their returned state; the caller decides
whether it constitutes mission failure, and the GA needs the gradient either way. Current flags:

| Module | Flags |
|---|---|
| `engine.EngineState` | `at_idle`, `shut_down`, `power_limited` |
| `battery.BatteryState` | `at_cutoff`, `rate_limited`, `energy_limited`, `below_safe_floor`, `power_limited` (their disjunction) |
| `powertrain.PowertrainState` | `balanced`, with `bus_residual_kw` carrying the magnitude |
| `aerodynamics.AeroState` | `power_off`; `loiter_speed` also returns `"min_power"` / `"stall_margin"` |
| `mass` | `is_feasible()`; `fuel_volume_check` returns a `fits` bool |

Two things this rule does **not** cover. Bad *arguments* still raise `ValueError` — a non-positive
aspect ratio or an out-of-range altitude is a caller bug, and clamping it would hand the GA a flat
region of the search space to exploit. And infeasibility is never *clipped*: `build_mass_budget`
returns a negative `fuel_kg` unchanged so fitness can grade how infeasible a candidate is (M-09).

### `docs/assumptions.md` discipline

Every numeric constant not derived from first principles and not mandated by the problem statement
has an entry with an ID (`A-01`, `AE-07`, `M-03`, `E-04`, `B-06`, `P-01`, `S-03`, `O-11`). Code
references the ID in a one-line comment and does not repeat the reasoning. Status markers:
`MANDATED` (given by the PS, not a choice), `VERIFIED` (confirmed against a cited source),
`UNVERIFIED` (reasonable estimate, source not yet confirmed), `PLACEHOLDER` (must be replaced before
final submission), `OPEN` (see §9 of that file). Entries also record a **bias direction** —
*optimistic* means it flatters the design.

**Changing a default value means editing its entry and adding a change-log row in the same commit.**
Adding a constant without an entry, or silently promoting a `PLACEHOLDER`, is a defect.

## 5. Testing conventions

Three categories are in use across the existing modules, plus a fourth that emerged from the
rebuild. Follow the naming style already there: long, sentence-like test names that state the
claim.

1. **Reference values** — assert against an authoritative published table.
   `test_temperature_matches_isa_table`, `test_dynamic_viscosity_matches_pdas_table`.
   Available for `atmosphere.py` and nowhere else.
2. **Internal consistency** — assert that two independent routes to the same number agree.
   `test_algebraic_substitution_matches_a_fixed_point_iteration` (mass closure vs. a fixed-point
   solve), `test_no_phantom_energy_in_either_direction` (coulomb count vs. reported bus power over a
   randomized sweep), the aerodynamics tests that minimize `D*V` numerically over a velocity sweep
   and require the result to land on the analytic `speed_min_power`.
3. **Behavioural** — assert the direction of a response, or that a limit fires and is flagged.
   `test_bigger_engine_costs_fuel`, `test_command_above_the_discharge_limit_is_clamped_and_flagged`,
   `test_infeasibility_is_graded_not_clipped`, `test_altitudes_are_not_silently_clamped`.
4. **Regression guards against the previous implementation** — pin the corrected number *and* the
   wrong one, so the old behaviour cannot come back.
   `test_engine_power_against_the_previous_implementation` (31.58 vs 37.20 kW),
   `test_current_exceeds_the_naive_power_over_ocv_estimate`, `PREVIOUS_MODEL_SFC = 0.30`.

**Internal consistency is the strongest verification available for every module except
`atmosphere.py`** — C_D0, Oswald efficiency, specific powers and cell energy density are engineering
estimates, not standards, so there is nothing authoritative to check them against. Write more of
category 2 rather than pinning an estimate to four decimals and calling it verified.

`conftest.py` at the repo root puts the project on `sys.path`; tests import `src.models.x` directly.
Randomized sweeps use `np.random.default_rng(<fixed seed>)`.

## 6. Known traps

Each of these was already made once. The symptom is what you will recognise.

- **Raymer's `N_z` is the ULTIMATE load factor, not the limit factor.** Symptom: wing mass ~22 %
  light (98 kg where the independent scaling law says 119 kg), and the optimizer buys a bigger wing
  than it should. `mass.py` takes 3.8 as the *limit* factor and multiplies by 1.5 internally.
- **Generator and rectifier size on engine rated power; inverter and motor size on peak bus power.**
  Generator and rectifier are upstream of the DC bus, so they never see battery discharge; the
  inverter and motor carry the take-off peak, which is engine plus battery. A spec that said
  otherwise was wrong. Symptom: powertrain mass insensitive to battery power sizing.
- **The whole conversion chain, not just the motor.** The previous implementation divided shaft
  power by motor efficiency alone and then treated the generator's *bus* output as engine *shaft*
  power, dropping the generator, rectifier, inverter and cabling. Symptom: 31.58 kW of engine power
  called for where 37.20 kW is required at 30 kW of shaft demand — fuel burn understated ~16 %.
- **`I = P / V_oc` makes the pack implicitly lossless.** The previous battery computed current that
  way and never applied the resulting loss. Symptom: internal resistance is configured but changing
  it moves no output; ~0.377 kW unaccounted at the reference condition. Solve the quadratic.
- **A SoC clamp to `[0.0, 1.0]` lets the pack discharge through its own floor, unflagged.** Symptom:
  a mission ends with SoC below `soc_min` and no limit flag set; a 5 kWh pack delivers 0.25 kWh in a
  step where it held 0.0435 kWh. Limit in *current* space before integrating so the step lands on
  the boundary; do not clamp afterwards.
- **Fuzzy output membership functions anchored to the universe edges.** The previous controller
  built the `s_factor` MFs across the fixed 1.0–6.0 universe rather than around `s_min`/`s_max`.
  Symptom: the GA's controller genes barely move the defuzzified output, so those genes show a
  near-flat fitness response and the "adaptive" claim is not actually earned.
- **Constant SFC structurally guarantees the battery loses.** With no part-load penalty,
  load-levelling has zero value, so the battery's only remaining function is enabling a smaller
  engine — a trade it loses on mass at every size. Symptom: the optimizer drives battery capacity to
  its lower bound "on its own". That is the model deciding, not the physics. `engine.py` uses a
  Willans line for exactly this reason (E-01).
- **The bus balance check that cannot fail.** The previous shortfall test compared
  `p_gen + p_batt` against demand, but `p_batt` was *defined* as demand minus `p_gen`, so the check
  could only fire on the all-zero fallback path. Symptom: zero reported power shortfalls, ever.
  Compute the residual from the chain, do not reconstruct it from the split.
- **Battery power limit taken from engine rated power.** A 5 kWh pack with a 75 kW cap implies 15C,
  which no energy cell sustains. The rate limit must come from capacity × C-rate (B-04).
- **The atmosphere's two layers use structurally different formulas.** Reusing the troposphere power
  law above 11 km under-predicts stratospheric density and flatters high-altitude cruise. The
  previous model instead capped altitude at 11 km silently, which does the same thing.
- **`docs/presentation/ppt_brief.md` describes the deleted implementation.** Its headline
  12.25 h → 19.25 h and its 6-gene chromosome come from the code that contained the traps above.
  Treat it as a record of intent and slide structure, not as a result to reproduce or cite.

## 7. Open decisions

They are listed in **`docs/assumptions.md` §9 (O-01 … O-11)** and are not duplicated here.

The rule: an open decision means **both options are implemented behind a flag**, with the current
default recorded in the entry — `allow_shutdown` (O-04), `scale_resistance` (O-05),
`load_dependent` (O-11), `oswald_efficiency(method=...)` (O-01). That is what lets the decision be
resolved later by sweeping the flag rather than by rewriting the model. **Do not silently resolve
one**, and do not delete the branch you think is wrong.

## 8. How to work on this

- **Evaluate a specification before implementing it.** Two specs in this project have contained
  errors caught at implementation time (the `N_z` row, and the component sizing basis). If a spec is
  wrong, say so and implement the better version — then record the correction in the
  `docs/assumptions.md` change log.
- **Never tune a constant to make a test pass.** Report the discrepancy instead. If the two wing-mass
  laws disagree by 17 %, that is a finding.
- **Do not resolve an open decision without being asked.**
- **Two tracks, one repo.** The modelling track owns `src/models/`, `src/control/`,
  `src/simulation/`, `src/optimization/`, `src/analysis/`, `tests/` and `docs/assumptions.md`. The
  visualization and deliverables track owns `src/dashboard/`, `.streamlit/`, `deliverables/` and
  `docs/presentation/`. `configs/` and `README.md` are shared. Editing across the seam is allowed
  but must be flagged explicitly in the change description — do not quietly restyle a dashboard or
  retune a model that belongs to the other track.
- Run the venv interpreter as `aeroprjct/Scripts/python.exe`.

---

**What makes this file stale:** §3 build status goes stale first, then §7 as open decisions are
resolved, then §6 as new traps are found. A change to a convention belongs in this file **in the
same commit that changes the code**. If a section grows past what fits here, move it to `docs/` and
link it — this file targets 250 lines.
