# Design Assumptions

**Project:** Hybrid-Electric Propulsion Optimization for a Fixed-Wing UAV
**Problem statement:** AEROTHON — IIT Indore in collaboration with Hindustan Aeronautics Limited, PS1

This document is the single source of truth for every assumed value in the codebase. Code modules
reference entries here by ID rather than restating rationale in comments. Every numeric constant
that is not derived from first principles or mandated by the problem statement must appear here.

## Status legend

| Marker | Meaning |
|---|---|
| `MANDATED` | Fixed by the problem statement — not a design choice |
| `VERIFIED` | Value confirmed against a cited published source |
| `UNVERIFIED` | Value is a reasonable engineering estimate; source not yet confirmed |
| `PLACEHOLDER` | Working value only; must be replaced before final submission |
| `OPEN` | Decision not yet made — see §9 |

## Bias convention

Where an assumption is known to skew results, the entry records the direction:
*optimistic* means it makes the design look better than reality, *conservative* means the opposite.
Neutral assumptions carry no bias note.

---

## 1. Problem statement constraints

These are not assumptions. They are given.

| Parameter | Value | Status |
|---|---|---|
| Maximum take-off weight | 1000 kg | `MANDATED` |
| Payload | 200 kg | `MANDATED` |
| Cruise speed | 250 km/h (69.44 m/s) | `MANDATED` |
| Cruise altitude band | 3–10 km | `MANDATED` |
| Preferred prime mover | ~60 kW turboshaft | `MANDATED` (stated as preference, not constraint) |
| Mission phases | take-off, climb, cruise, loiter, descent, landing | `MANDATED` |
| Objective | maximize endurance | `MANDATED` |

Problem statement Note 1 explicitly excludes CFD, structural, and thermal simulation. Note 2
directs the work toward system-level design space exploration. Both are cited below as
justification for the fidelity level chosen.

---

## 2. Atmosphere — `models/atmosphere.py`

### A-01 — International Standard Atmosphere, no temperature offset
**Value:** ISA per ISO 2533 / ICAO Doc 7488. `VERIFIED`
**Rationale:** The problem statement specifies no operating location, season, or climate. ISA is
the universal baseline in conceptual design, so all candidate designs are compared on identical
atmospheric conditions.
**Bias:** Optimistic for hot-day operation. ISA+20 reduces density by roughly 7%, which lapses
turboshaft power and raises the lift coefficient required at a given speed. A hot-and-high case is
noted as future work.

### A-02 — Dry air, constant composition, ideal gas
**Value:** R = 287.05287 J/(kg·K), γ = 1.4. `VERIFIED`
**Rationale:** Humidity changes density by well under 1% at realistic values. Compressibility
factor is 1.000 to four decimals across the mission envelope.

### A-03 — Two-layer model, valid 0–20 km
**Value:** Troposphere (linear lapse, 0–11 km) and lower isothermal stratosphere (11–20 km).
`VERIFIED`
**Rationale:** Mission ceiling is 10 km. The stratosphere layer is implemented as a guard against
an out-of-range altitude produced by optimizer mutation, not because it is flown.

### A-04 — Geopotential altitude convention
**Value:** All altitudes in `MISSION_PROFILE` are geopotential. `VERIFIED`
**Rationale:** ISA is defined on geopotential altitude. The difference from geometric altitude is
0.16% at 10 km. Either convention is valid provided it is applied consistently; mixing them is a
silent error.

### A-05 — No wind
**Value:** Still air. `VERIFIED`
**Rationale:** For a *range* mission wind is first-order. For an *endurance* mission it is not:
time aloft is set by fuel flow and power required, both functions of airspeed rather than
groundspeed. This justification is specific to the stated objective.

---

## 3. Aerodynamics — `models/aerodynamics.py`

### AE-01 — Point-mass performance model
**Value:** Aircraft treated as a particle; force balance only, no rotational degrees of freedom.
`VERIFIED`
**Rationale:** Directly justified by problem statement Notes 1 and 2.

### AE-02 — Parabolic drag polar
**Value:** C_D = C_D0 + C_L² / (π·AR·e). `VERIFIED`
**Rationale:** Standard conceptual-design approximation; exact for an elliptically loaded wing in
inviscid flow and empirically adequate at moderate lift coefficient.
**Bias:** Optimistic near C_Lmax, where flow separation makes drag rise faster than quadratic.
Because loiter is stall-margin-limited (see AE-05), the vehicle operates closer to that region
than a conventional cruise point, so this flatters the phase that dominates endurance.

### AE-03 — Zero-lift drag coefficient
**Value:** C_D0 = 0.028. `PLACEHOLDER`
**Rationale:** Representative of a clean UAV of this class. Must be replaced with a component
buildup (see AE-04) before final submission.
**Bias:** Unknown until the buildup is done.

### AE-04 — C_D0 is referenced to wing area and therefore not independent of it
**Value:** Equivalent skin friction method, C_D0 = C_fe · S_wet / S_ref, with C_fe = 0.0055.
`UNVERIFIED`
**Rationale:** Fuselage and empennage wetted area do not shrink when the wing does. Holding C_D0
fixed while wing area varies would give the optimizer an unphysical free lunch from shrinking the
wing. Only relevant if wing area becomes a design variable (see O-02).

### AE-05 — Maximum lift coefficient
**Value:** C_Lmax = 1.5, clean configuration. `PLACEHOLDER`
**Rationale:** No airfoil has been selected. Because the minimum-power speed falls below stall
speed for high-aspect-ratio wings, the loiter condition is set by stall margin rather than by the
minimum-power condition — which makes this parameter a stronger driver of endurance than most
propulsion parameters. It is therefore treated as a sensitivity study, not a fixed number.

### AE-06 — Stall margin at loiter
**Value:** V_loiter ≥ 1.2 · V_stall, equivalently C_L ≤ C_Lmax / 1.44. `VERIFIED`
**Rationale:** Conventional manoeuvre margin. Without this cap the solver returns a loiter lift
coefficient above C_Lmax, which is physically unflyable.

### AE-07 — Oswald efficiency treatment
**Value:** `OPEN` — see O-01.
**Rationale:** Raymer's straight-wing correlation, e = 1.78(1 − 0.045·AR^0.68) − 0.64, is fitted
to general-aviation aircraft and becomes pessimistic above roughly AR 12: it gives e ≈ 0.76 at
AR 10, 0.61 at AR 16, and 0.53 at AR 20, whereas sailplanes at AR 20+ achieve approximately 0.8.
Applied unmodified inside an optimizer it suppresses high aspect ratio for reasons that are not
physical.
**Bias:** The correlation is conservative on aspect ratio; a fixed value may be optimistic.
This choice determines whether the aspect-ratio result is a physical optimum or a modelling
artifact, and must be resolved with a sensitivity sweep either way.

### AE-08 — Incompressible flow
**Value:** Mach-independent drag polar. `VERIFIED`
**Rationale:** Maximum Mach number is approximately 0.22 at cruise, well below the conventional
0.3 threshold and far from critical Mach. Asserted in code rather than assumed.

### AE-09 — Small climb angle, L ≈ W
**Value:** cos γ ≈ 1. `VERIFIED`
**Rationale:** At 2 m/s climb and 65 m/s airspeed, γ = 1.8° and cos γ = 0.9995 — a 0.05% error in
required lift.

### AE-10 — Quasi-steady flight
**Value:** Accelerations neglected; each timestep is an equilibrium point. `VERIFIED`
**Rationale:** Phase transitions are gentle and the timestep is far longer than any relevant
dynamic time constant.
**Bias:** Optimistic for take-off, which is genuinely not an equilibrium condition. Take-off power
is treated as an estimate.

### AE-11 — Propeller efficiency
**Value:** η_prop = 0.85, constant. `OPEN` — see O-03.
**Rationale:** Propeller efficiency varies with advance ratio J = V/nD. A fixed-pitch propeller
optimized for cruise falls to roughly 0.5–0.6 at take-off.
**Bias:** Optimistic. Holding η_prop at its cruise value understates peak power demand, which
understates the transient peak the battery must shave — directly flattering any conclusion that a
small battery is sufficient.

### AE-12 — Straight-and-level loiter
**Value:** Wings level, no turn. `VERIFIED`
**Rationale:** Simplification of a real orbit or racetrack pattern.
**Bias:** Optimistic. At 15° bank the load factor is 1.035, raising induced drag by roughly 7% on
the phase that occupies the majority of mission time.

### AE-13 — Clean configuration throughout
**Value:** No drag increment for landing gear, flaps, or control deflection. `VERIFIED`
**Bias:** Optimistic for take-off and landing. Real ΔC_D0 for extended gear plus flaps can reach
0.02–0.05, comparable to the entire clean C_D0. Ground effect and trim drag are likewise neglected;
ground effect neglect is conservative, trim drag neglect is optimistic.

---

## 4. Mass and sizing — `models/mass.py`

### M-01 — Fixed MTOW collapses the mass snowball
**Value:** Fuel is the algebraic residual of the mass budget. `VERIFIED`
**Rationale:** Conceptual design normally requires an iterative solve because empty weight depends
on MTOW, which depends on fuel, which depends on empty weight. Because the problem statement fixes
MTOW at 1000 kg, this loop collapses to direct subtraction. This is also the mechanism that makes
every gram of powertrain mass trade directly against fuel mass.

### M-02 — Fixed-mass group
**Value:** 250 kg, decomposed as:

| Component | Mass | Basis |
|---|---|---|
| Fuselage structure | 100 kg | ~10% MTOW |
| Landing gear | 50 kg | ~5% MTOW |
| Empennage | 25 kg | ~2.5% MTOW |
| Avionics, autopilot, sensors | 50 kg | ~5% MTOW |
| Propeller and hub | 25 kg | |

`UNVERIFIED`
**Rationale:** These components do not respond to any design variable, and MTOW is fixed, so
fuselage load paths and landing gear sizing are fixed with it. Note this figure **excludes** the
wing and the entire electrical chain, both of which are computed explicitly — an earlier lumped
figure of 450 kg contained them implicitly and would double-count if reused.

### M-03 — Wing mass regression
**Value:** Raymer general-aviation wing weight equation, with:

| Parameter | Value |
|---|---|
| Construction factor (composite) | 0.87 |
| Design limit load factor | 3.8 |
| Ultimate factor | 1.5 × limit |
| **N_z fed to regression** | **5.7 (ultimate)** |
| Taper ratio | 0.5 |
| Thickness-to-chord | 0.15 |
| Sweep | 0° |
| Fuel in wing W_fw (nominal) | 250 kg |
| Cruise dynamic pressure q | 1590.5 Pa (69.44 m/s at 6 km ISA) |

`VERIFIED` (equation form, coefficients, exponents and units), `UNVERIFIED` (parameter values —
composite factor, taper, t/c, load factor)
**Rationale:** Raymer's regressions are fitted to manned general-aviation aircraft — pressurized,
certified, metal construction. A 1000 kg UAV has none of those characteristics. Application outside
the calibration set is standard practice at conceptual level for want of a better model, and the
composite factor is the conventional correction.
**N_z is the ULTIMATE load factor, not the limit load factor.** Raymer's chapter 15 statistical
group weights are defined on ultimate load factor throughout. Feeding the limit value instead
under-predicts wing mass by 1.5^0.49 = 22%. This was confirmed against two independent
transcriptions of the same equation during the `mass.py` build, including the
Forrester/Sóbester/Keane wing weight benchmark, which lists N_z as "ultimate load factor" over
[2.5, 6]. An earlier revision of this entry recorded a single row `N_z = 3.8`, which caused exactly
that error. The code takes 3.8 as the limit factor and multiplies by 1.5 before evaluating the
regression. See O-09 — whether 3.8 is the right *limit* factor for this vehicle is still open.
**W_fw and q are effectively inert.** Their exponents are 0.0035 and 0.006, so a factor-of-two error
in either moves wing mass by under 0.3% and 0.5% respectively. W_fw is fixed at a nominal 250 kg
rather than being solved iteratively against the fuel residual (M-01), which would be a circularity
bought for no accuracy.
**Bias:** Direction unknown. Wing mass scales as (N_z · W)^0.49, so the load factor alone moves the
result materially — 3.8 is the civil normal-category value and a loiter-optimized UAV could justify
lower. This term is what creates an interior optimum for wing area and aspect ratio instead of the
optimizer running to a bound.

### M-03b — Simplified wing mass cross-check
**Value:** m = 4.0 · S^0.75 · AR^0.6. `UNVERIFIED`
**Rationale:** A second, structurally independent scaling law used only to sanity-check M-03, never
in the mission simulation. At the reference configuration (S = 10 m², AR = 16) it gives 118.7 kg
against the regression's 119.6 kg — 0.7% apart, which is the strongest available corroboration that
the ultimate-load-factor reading above is the correct one. Read against the limit factor the
regression returns 98.0 kg, 17% below this law.

### M-04 — Propulsion component specific powers
**Value:**

| Component | kW/kg | Sized on |
|---|---|---|
| Turboshaft engine | 3.5 | Engine rated power |
| Generator | 3.0 | Engine rated power |
| Rectifier | 15.0 | Engine rated power |
| Inverter | 15.0 | Peak bus power |
| Electric motor | 7.0 | Peak bus power |
| Cabling and cooling | 15% of electrical subtotal | — |

`UNVERIFIED`
**Rationale:** Technology-level assumptions representing current state of the art. The inverter and
motor are sized on peak bus power, not engine rated power, because they carry the take-off peak
which is engine output plus battery discharge.
**Bias:** The engine figure supersedes an earlier 1.5 kW/kg. Published small turboshafts run
3–4.5 kW/kg; an over-heavy engine makes engine growth look more expensive than it is and distorts
the central engine-versus-battery trade in the battery's favour.
**Note:** Without the electrical chain masses, the mass penalty of selecting a series architecture
is unaccounted and the architecture choice cannot be defended. This is the most consequential
addition to the mass model.

### M-05 — Fuel system mass
**Value:** 7% of fuel mass. `UNVERIFIED`
**Rationale:** Tanks, pumps, plumbing, and vents typically run 5–10% of fuel mass for this class.
Creates a circularity with M-01 which is resolved algebraically, not iteratively.

### M-06 — Fuel properties and volume
**Value:** Jet A-1, density 0.804 kg/L, LHV 43.1 MJ/kg. Usable tank volume taken as 50% of gross
wing volume. `VERIFIED` (fuel properties), `PLACEHOLDER` (volume fraction)
**Rationale:** Wing internal volume scales as S · chord · (t/c), and chord = √(S/AR), so tank
volume falls as both wing area and aspect ratio change.
**Wing area is the sensitive variable, not aspect ratio.** An earlier revision of this entry claimed
the volume check was "a second, independent penalty on high aspect ratio"; that is wrong. At the
reference fuel load the constraint binds below S ≈ 7 m². The aspect-ratio crossover is AR ≈ 46,
which is physically irrelevant for this vehicle, so the check does not constrain any realistic
design. It is cheap insurance against a runaway optimizer, not an active constraint, and it is not
a mechanism that limits aspect ratio.
**The 0.5 usable fraction is unsourced and optimistic.** Real wing tanks are limited to the box
between front and rear spars, typically 0.30–0.35 of gross wing volume. At 0.30 the reference
design has approximately 356 L available against 348 L required — marginal rather than comfortable,
which would move this check from inert to nearly binding. See O-10.

### M-07 — No mass growth allowance
**Value:** 0% contingency. `UNVERIFIED`
**Bias:** Optimistic. Real programmes carry 5–15% contingency at conceptual stage.

### M-08 — No centre-of-gravity or balance constraint
**Value:** Not modelled. `VERIFIED`
**Rationale:** Out of scope at this fidelity per problem statement Note 1. Battery placement, fuel
burn, and payload location all shift CG in a real design.

### M-09 — Minimum usable fuel and unclipped infeasibility
**Value:** 20 kg. `UNVERIFIED`
**Rationale:** Below roughly 20 kg the fuel load cannot cover taxi, take-off, climb and reserves,
so the design is treated as infeasible. The mass model does **not** clamp or raise on such designs:
it returns the computed residual even when negative, so the GA fitness function can grade *how*
infeasible a candidate is and keep a gradient toward the feasible region. Clipping at zero would
flatten that region and let the optimizer wander in it.

---

## 5. Engine — `models/engine.py`

### E-01 — Willans line part-load fuel model
**Value:** ṁ_f · LHV = P_shaft / η_i + P_0, equivalently SFC(P) = a + b/P. `VERIFIED` (form)
**Rationale:** Gas turbines are optimized for a single design point; away from it, compressor
pressure ratio falls, turbine inlet temperature drops, component isentropic efficiencies degrade,
and fixed parasitic losses become a larger fraction of a shrinking output. The Willans line captures
this with two physically interpretable parameters and is the standard treatment in the
hybrid-powertrain literature.
**Significance:** This is the single most consequential assumption in the project. A constant SFC
gives load-levelling zero value, which reduces the battery's only function to enabling a smaller
engine — a trade that loses on mass at every battery size. Without a part-load penalty the model
has already decided that hybridization does not pay.

### E-02 — Willans calibration
**Value:** SFC at rated power = 0.45 kg/kWh; idle fuel flow = 20% of maximum fuel flow.
`PLACEHOLDER`
**Rationale:** Must be fitted to published data for a 60–100 kW class turboshaft before final
submission. Candidate sources: Rolls-Royce/Allison 250 series, PBS TS100.
**Bias:** The earlier constant value of 0.30 kg/kWh is a large-turboprop figure and is optimistic
for an engine of this size, independently of the part-load shape.

### E-03 — Willans parameter scaling with engine size
**Value:** Marginal efficiency term `a` invariant within the size class; parasitic term `b` scales
with rated power. `UNVERIFIED`
**Rationale:** Marginal thermal efficiency does not vary strongly across a narrow size range, while
parasitic losses grow with the machine. A consequence worth noting: an oversized engine carries a
larger absolute idle penalty, which is precisely the mechanism that makes load-levelling valuable.

### E-04 — Altitude power lapse
**Value:** P_max(h) = P_max,SL · σ^n with n = 0.8. `UNVERIFIED`
**Rationale:** Turbine power falls approximately with density ratio. Only *maximum available*
power lapses: the Willans coefficients a and b are held altitude-invariant, so SFC at a given
absolute shaft power is treated as independent of altitude. Parasitic losses are largely mechanical
and do not fall with air density, and holding a fixed is the same statement that marginal thermal
efficiency is set by the cycle rather than by ambient conditions.
**Bias:** Conservative. The mild SFC improvement from colder inlet air at altitude is neglected, so
predicted cruise fuel burn is slightly higher than a real engine would deliver. The endurance
result therefore errs on the pessimistic side, which is the right direction for a design claim.
**Consequence for sizing:** a 75 kW engine delivers 59.1 kW at 3 km (σ = 0.742) and 45.7 kW at 6 km
(σ = 0.538). If shaft power demand at the chosen cruise altitude exceeds the lapsed rating the
mission is infeasible there and the engine or the altitude must change — this couples directly to
O-07.

### E-05 — Low-power operating floor
**Value:** Floor at 15% of rated power. Mode `OPEN` — see O-04. Restart fuel 0.0 kg `PLACEHOLDER`.
**Rationale:** The Willans line predicts SFC → ∞ as power → 0, which is correct in the limit but
must not be extrapolated below roughly 15–20% of rated power. Whether the engine idles (burning the
parasitic term, producing near-zero output) or shuts off (burning nothing, with a restart penalty)
is a modelling decision with real consequences for the energy-management strategy.
**Implementation:** both modes are implemented behind `allow_shutdown` rather than one being
chosen, so O-04 can be resolved by sweeping the flag. In idle mode a below-floor command delivers
15% of rated power and the corresponding fuel flow; the surplus generation is handed back to the
caller to dispose of into the battery. In shutdown mode it delivers zero power and zero fuel.
**Bias:** Optimistic. A zero-cost restart flatters shutdown mode — a real turboshaft start consumes
fuel, takes seconds of spool-up during which no power is available, and consumes engine life. The
model is stateless by design, so it reports `shut_down` in the returned state and leaves the caller
to detect the off-to-on transition and charge `restart_fuel_kg` against it.

---

## 6. Energy storage — `models/battery.py`

### B-01 — Chemistry fixed at lithium-ion NMC
**Value:** Not a design variable. `VERIFIED` (as a scoping decision)
**Rationale:** The problem statement lists battery chemistry as an optional design variable. It is
excluded here because the model's fidelity cannot distinguish chemistries beyond specific energy
and C-rate, so it would be a variable with no penalty gradient — the optimizer would move it
arbitrarily. Excluding variables the model cannot resolve is a deliberate scoping choice, stated
rather than omitted.

### B-02 — Pack-level specific energy
**Value:** 250 Wh/kg at cell level × 0.75 packaging factor = 187.5 Wh/kg at pack level.
`UNVERIFIED`
**Rationale:** The 250 Wh/kg figure is a cell specification. Battery management, contactors,
cooling, structure, and wiring reduce this substantially at pack level.
**Bias:** Any error here favours the battery. If the earlier cell-level figure produced a design
that still rejected large batteries, correcting it strengthens that conclusion rather than
weakening it.

### B-03 — Rint equivalent circuit model
**Value:** Linear open-circuit voltage, 300 V at 0% SoC to 400 V at 100% SoC; internal resistance
0.05 Ω at a 10 kWh reference capacity, scaled as R(E) = 0.05 · (10 / E_kWh). `PLACEHOLDER` (values),
`VERIFIED` (equation form and the energy-balance consistency below)
**Rationale:** Simplest model that captures ohmic loss and voltage sag.
**Current is solved from the quadratic, not approximated.** Discharge is R·I² − V_oc·I + P = 0 and
the physical branch is the lower root, which is signed correctly on both sides of zero power, so
charging (V_t = V_oc + |I|·R) falls out of the same expression. Approximating I = P/V_oc leaves
internal resistance decorative and makes the pack implicitly 100% efficient. At 30 kW on a 10 kWh
pack at 50% SoC the solved current is 86.79 A against the naive 85.71 A — 1.26% higher — and the
0.377 kW of ohmic loss that the naive form never charged anywhere.
**Charge capacity is consistent by construction.** Q_nom = E_rated / V_nominal with
V_nominal = (V_min + V_max)/2. Because the OCV is linear, ∫V_oc dq over a full sweep is exactly
V_nominal · Q_nom, so coulomb counting on current recovers the rated energy at the bus. This is
asserted in `tests/test_battery.py` rather than assumed.
**Resistance scales inversely with capacity.** Added capacity is added parallel cell strings at
fixed pack voltage, so resistance falls as 1/E: 0.10 Ω at 5 kWh, 0.05 Ω at 10 kWh, 0.01 Ω at 50 kWh.
Holding it fixed at 0.05 Ω would penalize large packs for resistance they do not have. As with E-05,
both behaviours are implemented behind `scale_resistance` rather than one being chosen, so O-05 can
be resolved by sweeping the flag; the scaled model is the default.
**Ohmic power ceiling.** The quadratic has no real root above V_oc²/(4R) — 612.5 kW for the 10 kWh
pack at 50% SoC. The C-rate limit (B-04) binds two decades earlier at every realistic pack size, so
the guard is never active in practice, but it is enforced rather than allowed to produce NaN.
**Bias:** Optimistic. A single fixed resistance neglects the rise at low SoC, at low temperature,
and with age; real packs are worst exactly where the take-off peak is drawn from a depleted pack.

### B-04 — Charge and discharge rate limits
**Value:** 3C continuous discharge, 1C continuous charge. `UNVERIFIED`
**Rationale:** Representative for high-energy-density cells; charge acceptance is the lower of the
two for this chemistry. The pack power limit must derive from capacity and C-rate, not from engine
rated power — a 5 kWh pack with a 75 kW limit implies 15C, which no energy cell sustains. Under
this model that pack caps at 15 kW, a factor of five below what the previous limit allowed.
**Consequence for the architecture:** the C-rate limit, not the energy capacity, is what sizes the
pack for take-off peak shaving. A pack large enough to shave the peak may carry far more energy
than the mission needs, and that surplus energy is dead mass under the fixed-MTOW closure (M-01).

### B-05 — No thermal or ageing model
**Value:** Not modelled. `VERIFIED`
**Rationale:** Out of scope at this fidelity. Capacity fade, temperature-dependent resistance, and
Peukert effects are noted as future work.

### B-06 — State of charge limits
**Value:** Hard cutoff 5%; recommended operating floor 20%. `UNVERIFIED`
**Rationale:** Protects against deep discharge. Discharge is locked out below the hard cutoff in
simulation; charging is not, so the pack can always recover. The 20% floor is reported on every
returned state and never enforced — it is advice to the energy-management controller (O-08), not a
constraint, so the controller is free to trade against it and be graded on having done so.
**Implementation:** a power command the pack cannot meet is clamped and flagged rather than raised,
for the same reason M-09 does not clamp negative fuel — the caller decides whether an unmet demand
constitutes mission failure, and the optimizer needs the gradient either way.

---

## 7. Powertrain — `models/powertrain.py`

### P-01 — Series architecture efficiency chain
**Value:**

| Stage | Efficiency |
|---|---|
| Generator | 0.95 |
| Rectifier | 0.95 |
| Inverter | 0.95 |
| Electric motor | 0.95 |
| Propeller | see AE-11 |

`UNVERIFIED`
**Rationale:** Representative values for the component classes. Source side (engine → bus) is
η_gen · η_rect = 0.9025; demand side (bus → shaft) is η_inv · η_motor = 0.9025. Including the
propeller, every kilowatt of drag power costs approximately 1.45 kW of engine shaft power.

### P-02 — Efficiencies constant with load
**Value:** No part-load derating. `UNVERIFIED`
**Bias:** Optimistic during low-demand phases. Real converter and machine efficiencies fall at low
power fractions.

### P-03 — Excluded design variables
**Value:** Number of propulsion motors and generator architecture (AC/DC) are fixed, not optimized.
`VERIFIED` (as a scoping decision)
**Rationale:** The problem statement lists both as optional design variables. Neither is modelled
at a fidelity that would produce a penalty gradient — distributed propulsion aerodynamic benefit
and AC/DC topology differences are not represented — so including them would add dimensions the
optimizer cannot resolve. As with B-01, this exclusion is stated deliberately.

---

## 8. Mission and simulation — `simulation/`

### S-01 — Fill-to-MTOW fuel policy
**Value:** Fuel loaded = MTOW − dry mass. `VERIFIED`
**Rationale:** Makes the powertrain-versus-fuel trade explicit and direct. Every kilogram spent on
engine, battery, or structure is a kilogram of fuel not carried.

### S-02 — Loiter is the endurance phase
**Value:** Loiter extended until fuel reserve or SoC cutoff is reached; total mission time is the
endurance being maximized. `VERIFIED`
**Rationale:** Directly implements the stated objective while guaranteeing descent and landing
remain completable.

### S-03 — Integration scheme and timestep
**Value:** Explicit Euler, Δt = 60 s. `UNVERIFIED`
**Rationale:** First-order accuracy is adequate for quasi-steady phases. A shorter step during
climb, or an adaptive step, is noted as future work.
**Bias:** Unknown; a step-size convergence study should be run to quantify it.

### S-04 — Reserves and limits
**Value:** Landing fuel reserve 5 kg; minimum usable fuel for feasibility 20 kg; maximum mission
time 24 h. `UNVERIFIED`
**Rationale:** The reserve guarantees descent and landing can be completed. The 24 h cap is a
safety bound on the simulation loop, not a physical limit.

---

## 9. Open decisions

These must be resolved and this document updated before final submission.

| ID | Decision | Options | Blocks |
|---|---|---|---|
| O-01 | Oswald efficiency treatment | Raymer AR correlation vs fixed documented value with sensitivity sweep | AE-07, aspect-ratio result validity |
| O-02 | Wing area and aspect ratio as design variables | Optimize both vs freeze with written justification | AE-04, M-03, gene set |
| O-03 | Propeller efficiency model | Constant 0.85 vs phase-dependent vs variable-pitch assumption | AE-11, peak power demand |
| O-04 | Engine low-power behaviour | Idle (burn parasitic term) vs shutdown with restart penalty | E-05, energy-management strategy |
| O-05 | Internal resistance scaling | Fixed 0.05 Ω vs scaled with pack capacity (both implemented; scaled is the default) | B-03 |
| O-06 | Loiter speed | Solved from minimum-power/stall-margin condition each step vs fixed | AE-05, AE-06, mission profile |
| O-07 | Cruise altitude | Fixed at a chosen value vs treated as a design variable | E-04, mission profile |
| O-08 | Energy-management controller | Fuzzy adaptive vs PI feedback vs fixed equivalence factor | Gene set, innovation claim |
| O-09 | Design limit load factor | 3.8 (FAR 23 normal category, manned) vs 2.5–3.0 (typical MALE-class UAV) | M-03, wing mass |
| O-10 | Usable tank volume fraction | 0.5 (current) vs 0.30–0.35 (inter-spar box) | M-06, whether the volume check binds |

**O-09.** 3.8 is a manned-aircraft certification value from FAR 23 normal category with no direct
applicability to an unmanned vehicle. Dropping the limit factor from 3.8 to 2.5 scales wing mass by
(2.5/3.8)^0.49 = 0.815 — roughly 23 kg at the reference configuration, which converts directly into
fuel under the fixed-MTOW closure (M-01). This is the single largest unforced assumption in the
mass model.

---

## 10. References

To be completed as sources are verified. Required before final submission.

| Ref | Source | Used for |
|---|---|---|
| R1 | ISO 2533:1975, *Standard Atmosphere* | A-01 to A-04 |
| R2 | ICAO Doc 7488, *Manual of the ICAO Standard Atmosphere* | A-01 to A-04 |
| R3 | Raymer, D. P., *Aircraft Design: A Conceptual Approach* | AE-07, M-03 |
| R4 | Anderson, J. D., *Aircraft Performance and Design* | AE-02, AE-05, AE-06 |
| R5 | Guzzella, L. and Sciarretta, A., *Vehicle Propulsion Systems* | E-01, ECMS formulation |
| R6 | *(pending)* Published turboshaft performance data, 60–100 kW class | E-02, M-04 |
| R7 | *(pending)* Lithium-ion cell and pack specifications | B-02, B-04 |
| R8 | *(pending)* Electrical machine and power electronics specific power data | M-04, P-01 |

Author names and titles above are recorded from working notes and must be confirmed against the
actual editions before citation in the technical report.

---

## 11. Change log

| Date | Change |
|---|---|
| — | Initial version. Fixed-mass group revised from 450 kg lumped to 250 kg itemized (M-02) following explicit modelling of wing and electrical chain masses. Engine specific power revised from 1.5 to 3.5 kW/kg (M-04). Constant SFC replaced by Willans line (E-01). |
| 2026-08-04 | **`mass.py` build.** M-03: N_z corrected from limit to ultimate load factor — the entry previously listed a single row `N_z = 3.8`, which fed the regression the limit value and under-predicted wing mass by 22%; equation form and units promoted to `VERIFIED`, parameter values remain `UNVERIFIED`. M-06: the claim that the fuel volume check penalizes high aspect ratio corrected — wing *area* is the sensitive variable (binds below S ≈ 7 m²), the aspect-ratio crossover at AR ≈ 46 is physically irrelevant; usable-volume fraction downgraded to `PLACEHOLDER`. O-09 (limit load factor) and O-10 (tank volume fraction) opened. |
| 2026-08-04 | **`engine.py` build.** E-04: recorded that the Willans coefficients are held altitude-invariant and that only maximum power lapses; neglecting the colder-inlet SFC gain is conservative. E-05: floor fixed at 15% of rated, both idle and shutdown modes implemented behind a flag pending O-04, restart fuel marked `PLACEHOLDER` at 0.0 kg and flagged as optimistic. |
| 2026-08-04 | **`battery.py` build.** B-03: internal resistance scaling model recorded as R(E) = 0.05 · (10 / E_kWh), implemented behind `scale_resistance` with the scaled form as default pending O-05; the quadratic current solution and the Q_nom = E/V_nominal consistency promoted to `VERIFIED` in form; the naive P/V_oc error quantified at 1.26% in current and 0.377 kW in unbilled loss at the reference condition; ohmic power ceiling V_oc²/(4R) recorded as an enforced guard. B-04: retitled to cover both rate limits, charge C-rate of 1C recorded, and the previous engine-rated power limit identified as an implied 15C on a 5 kWh pack. B-06: clamp-and-flag behaviour recorded, and the 20% floor documented as reported-not-enforced. |
