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
**Value:** Floor at 15% of rated power; shutdown is the resolved default (O-04). Restart fuel
0.0 kg `PLACEHOLDER`.
**Rationale:** The Willans line predicts SFC → ∞ as power → 0, which is correct in the limit but
must not be extrapolated below roughly 15–20% of rated power. Whether the engine idles (burning the
parasitic term, producing near-zero output) or shuts off (burning nothing, with a restart penalty)
is a modelling decision with real consequences for the energy-management strategy.
**Implementation:** both modes remain implemented behind `allow_shutdown` for sensitivity. In idle
mode a below-floor command delivers
15% of rated power and the corresponding fuel flow; the surplus generation is handed back to the
caller to dispose of into the battery. In shutdown mode it delivers zero power and zero fuel; the
simulator detects each off-to-on transition and charges `restart_fuel_kg`.
**Bias:** Optimistic. A zero-cost restart flatters shutdown mode — a real turboshaft start consumes
fuel, takes seconds of spool-up during which no power is available, and consumes engine life. The
engine model is stateless by design, so it reports `shut_down` in the returned state; the simulator
is the caller that detects the off-to-on transition and charges `restart_fuel_kg` against it.

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

### B-06 — State of charge limits and energy-limited availability
**Value:** Hard cutoff 5%; recommended operating floor 20%; boundary tolerance 1×10⁻⁹ SoC.
`UNVERIFIED` (the two limits), `VERIFIED` (the availability model in form)
**Rationale:** Protects against deep discharge. Discharge is locked out below the hard cutoff in
simulation; charging is not, so the pack can always recover. The 20% floor is reported on every
returned state and never enforced — it is advice to the energy-management controller (O-08), not a
constraint, so the controller is free to trade against it and be graded on having done so.
**Implementation:** a power command the pack cannot meet is limited and flagged rather than raised,
for the same reason M-09 does not clamp negative fuel — the caller decides whether an unmet demand
constitutes mission failure, and the optimizer needs the gradient either way.

**Available power is energy-limited as well as rate-limited.** Checking the cutoff *before*
integrating and clamping SoC *after* lets a step that starts above the floor integrate straight
through it, so the bus is handed energy the pack does not hold. Availability is therefore the power
the pack can sustain for the **whole** step, not the power it can deliver at the instant the step
begins. Two ceilings bound every command:

| Ceiling | Discharge | Charge (magnitude) | Depends on Δt |
|---|---|---|---|
| Rate | min(C_dis·E, V_oc²/4R) | C_chg·E | no |
| Energy | V_oc·I − I²R at I = (SoC − SoC_min)·Q_nom·3600/Δt | V_oc·I + I²R at I = (1 − SoC)·Q_nom·3600/Δt | yes |

The limiting current is exact irrespective of voltage, because coulomb counting is linear in
current; only the conversion to power needs a voltage. Availability is the lesser of the two
ceilings and is **non-increasing in Δt** — a longer step must be sustained from the same charge.

**The energy ceiling must be capped at the ohmic ceiling before the quadratic is evaluated.**
P(I) = V_oc·I − I²R turns over at I = V_oc/2R, so a coulomb budget larger than that current lands on
the descending branch and returns a *negative* power: at SoC = 1.0 over a 1 s step the 10 kWh pack's
budget of 97.7 kA evaluates to −438 MW, which as an availability would command the pack to charge at
full power. Above that current the coulomb budget constrains nothing — every achievable power draws
less — so it is reported as imposing no limit at all.

**Charging is symmetric but not identical.** The same signed quadratic gives both directions: with
I negative it returns −(V_oc·|I| + I²R), so the bus pays the loss on top of the stored energy, which
is correct. Charging has no analogue of the ohmic ceiling, because |P| rises without bound in |I|,
so no cap is needed on that side. SoC overshooting 1.0 at the top of the range is the same defect as
overshooting the floor at the bottom and is fixed by the same construction.

**Open-circuit voltage is taken at the start of the step.** For constant current over a step, the
average OCV is the value at the midpoint SoC, so a midpoint evaluation would be exact. It is not
adopted. Being exact requires the *same* voltage in `step()` as in the availability functions, and
solving for the current implied by the step-average voltage folds the OCV slope into the quadratic
as an effective resistance, R_eff = R + (V_max − V_min)·Δt/(2·Q_nom·3600) — 0.029 Ω against a true
0.05 Ω at Δt = 60 s on the 10 kWh pack. That makes the reported current, terminal voltage and
resistive loss functions of the timestep, and integrates the pack to second order while every other
state in the simulation is explicit Euler at first order (S-03). The cost is not worth the gain,
which is bounded by (V_max − V_min)·ΔSoC_step / 2V_oc:

| Condition | ΔSoC over the step | Overstatement of delivered energy |
|---|---|---|
| 3C for 60 s (the worst case the C-rate allows) | 0.050 | 0.71% |
| 0.5C loiter draw for 60 s | 0.008 | 0.12% |
| Final step onto the cutoff, worked case below | 0.010 | 0.16% |

**Bias:** Optimistic, and it falls on the endurance objective directly. Start-of-step OCV overstates
the step-average, so each step delivers marginally more bus energy than the charge it removes is
worth. Summed over a full discharge the model conserves ∫V_oc dq to 1.2×10⁻³ relative at the S-03
60 s step and 1.2×10⁻⁴ at 6 s — first order in Δt, halving when the step halves, which confirms the
residual is this bias and nothing else. It is three orders of magnitude below the defect it
replaces, and it shrinks with the timestep, which the alternative would not.

**Limiting is done in current space and nothing is clamped.** The commanded current, the C-rate
current and the coulomb-budget current are compared as magnitudes, the smallest is integrated, and
the reported bus power is V_oc·I − I²R at *that* current rather than the command. Clamping SoC after
integrating is what broke conservation: it reported the full commanded power as delivered while
pinning SoC at the floor, and the difference was energy the bus received that the pack never held.
Because the binding limit is known before integrating, an energy-limited step is assigned its
boundary directly and arrives on 5% or 100% exactly rather than within rounding of it, so no clamp
is needed to keep SoC in range and none remains. A command that binds no limit is still reported
verbatim, so a controller comparing delivered against commanded sees no floating-point dust.

**The clamp question is closed.** A randomized sweep over pack size, SoC, command and timestep
asserts two identities on every step: the coulomb count matches the integrated current, and the
reported bus power is the quadratic at that same current. Landing *on* a bound rather
than past it makes an exact comparison decide `at_cutoff` on floating-point dust — and would let a
mission loop creep toward the floor in ever-halving steps without arriving — so both the cutoff test
and the availability functions treat headroom below 1×10⁻⁹ SoC as zero. That is nine orders below
anything physically meaningful and seven above the rounding, and it strands at most 10⁻⁸ kWh.

**Availability depends on the timestep, and that coupling is accepted.** Any energy-aware limit must
know the horizon it is being asked to sustain; a pack can deliver 3C for a second and not for an
hour at the same SoC. The dependence is on a scalar duration passed in, not on the simulator, so the
model stays standalone. `available_discharge_kw` and `available_charge_kw` take `dt_s` as optional:
given one they return the sustainable power, omitting it they return the rate limit alone. The
energy-management controller (O-08) **must pass it** — searching candidate splits against the rate
limit alone will propose power the pack cannot hold, `step()` will reduce it, and the DC bus balance
the controller believed it had (P-01) will not close. `step()` applies every limit itself regardless,
so a caller that ignores the availability functions still cannot create energy; what it loses is the
guarantee that its own bus balance holds.

**Two limit flags, not one.** `rate_limited` means the pack could not deliver that power for an
instant — a sizing problem, fixed by a bigger pack. `energy_limited` means it could have, but lacks
the charge to hold it for the step — an energy-management problem, or a timestep-resolution one.
Both can be true at once. `power_limited` remains available as their disjunction and is exactly
equivalent to the delivered power differing from the command.

**Worked case.** A 5 kWh pack at SoC = 0.06 commanded 15 kW (its full 3C rating) for 60 s holds
0.0435 kWh above the floor. Unlimited, it delivered 0.25 kWh — 5.7× what it had — and landed at
SoC = 0.0019, *below* the hard cutoff, with no flag raised, because the clamp is to [0, 1] and never
fired. Limited, the charge above the floor supports 8.571 A, which at V_oc = 306 V is **2.616 kW**
against the 15 kW commanded; it delivers 0.0436 kWh, lands on 0.05 exactly, and reports
`energy_limited` with `rate_limited` false. Loiter terminates on this cutoff, so these are the last
steps of every mission and the phantom energy went straight into the objective.

---

## 7. Powertrain — `models/powertrain.py`

### P-01 — Series architecture efficiency chain
**Value:**

| Stage | Efficiency |
|---|---|
| Generator | 0.95 |
| Rectifier | 0.95 |
| DC bus cabling | 0.99 — see P-04 |
| Inverter | 0.95 |
| Electric motor | 0.95 |
| Propeller | see AE-11 |

`UNVERIFIED`
**Rationale:** Representative values for the component classes. Source side (engine → bus) is
η_gen · η_rect = 0.9025; demand side (bus → shaft) is η_inv · η_motor · η_cable = 0.8935; engine
shaft to propeller shaft is 0.8064. Including the propeller, every kilowatt of drag power costs
approximately 1.46 kW of engine shaft power.
**This chain is why `powertrain.py` exists.** The previous implementation divided shaft demand by
motor efficiency alone and treated the generator's bus output as engine shaft power, dropping the
inverter, generator, rectifier and cabling stages entirely. For 30 kW of shaft demand it called for
31.58 kW of engine shaft power where the full chain requires 37.20 kW — a factor of 1.178, so the
old figure understated engine power, and with it fuel burn, by 15.1%.

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

### P-04 — DC bus cabling and distribution efficiency
**Value:** 0.99. `UNVERIFIED`
**Rationale:** High-voltage DC distribution over a short airframe run. Previously neglected
entirely; it is made explicit here so that it is a stated assumption rather than a silent omission.
It is the fifth stage in the chain and the only one that is purely resistive — there is no
excitation to sustain — so it carries no no-load loss term and stays constant in both modes of O-11.
**Bias:** Mildly optimistic if cable runs are long or the bus sits at the low end of the 300–400 V
range (B-03), since distribution loss scales with the square of current at fixed power.

### P-05 — System efficiency metric definition
**Value:** η_system = P_propulsive / (P_fuel,chemical + P_battery,discharge). `VERIFIED` (as a
definition)
**Rationale:** Matches deliverable 8.3. Propulsive power is thrust power — drag power plus the rate
of potential energy gain — and arrives from `aerodynamics.py`; fuel chemical power is mass flow
times lower heating value, from `engine.py`. Battery **charging is excluded from the denominator**:
it is not an energy input to the system, and admitting negative values would let a charging step
inflate the metric without bound as the denominator passed through zero. The denominator is
guarded, returning zero rather than raising.

---

## 8. Control, mission and simulation — `control/`, `simulation/`

### C-01 — Absolute equivalence-factor sanity bounds
**Value:** 0.5 ≤ s ≤ 20.0. `UNVERIFIED`
**Rationale:** These are last-resort rails against a mutated controller gene or a failed inference,
not tuning bounds and not a claim about the useful controller range. The simulator calls the
clamped base-class method, so no strategy can send an extreme equivalence factor into the ECMS
Hamiltonian. A controller that reaches either rail during an ordinary mission should be treated as
misconfigured even though the value remains numerically safe.

### C-02 — Round-trip correction to the neutral equivalence factor
**Value:** `round_trip_efficiency = 1.0` by default, so the correction is disabled. `VERIFIED`
(correction form), `UNVERIFIED` (default)
**Rationale:** Equating engine fuel per DC-bus kWh with the Hamiltonian's equivalent battery fuel
defines the neutral factor. If discharged battery energy must later be replenished through a cycle
with efficiency η_rt, the break-even factor is divided by η_rt. The function accepts the efficiency
explicitly because the battery model makes it operating-point dependent; 1.0 preserves the direct
engine-versus-bus comparison until the controller integration supplies that operating point.
**Bias:** The default under-prices discharged battery energy relative to a lossy replacement cycle.

### C-03 — Marginal ECMS switching equivalence factor
**Value:** s_switch = a·LHV/(3600·η_source), where `a` is the Willans marginal slope. For the
default calibration, a = 0.36 kg/kWh and η_source = 0.9025, giving s_switch = 4.776. `VERIFIED`
(algebraic form under constant source efficiency and negligible battery resistance)
**Rationale:** Differentiating the Hamiltonian with respect to engine shaft power removes the
Willans intercept `b`, because that parasitic fuel flow is already sunk while the engine is on.
The resulting modulation threshold therefore uses marginal fuel consumption, not average SFC.
Controllers therefore anchor ratio parameterizations on `switching_s`; `neutral_s` remains a
logged average-cost diagnostic and is retained for the explicit `FixedECMS.at_neutral()` comparison.
This differs from `base.py`'s neutral factor: using the default rated SFC of 0.45 kg/kWh gives
s_neutral = 5.970. The neutral factor prices an average bus kWh at a stated operating point;
s_switch determines whether another engine kW lowers the Hamiltonian. Neither is the separate
engine on/off threshold, where the intercept and restart policy matter.
**Exact-model qualification:** Rint loss is quadratic in current, not bus power. Its exact
open-circuit power contains the battery model's square root and makes the on-state Hamiltonian
convex, allowing an interior minimum. Load-dependent source efficiency under O-11 also replaces
the affine source relation, so the closed form is a reference threshold rather than the exact
switching surface in either case.

### C-04 — Power-split numerical tolerances
**Value:** Production power tolerance 0.01 kW; one-sided derivative probe 0.001 kW at the default;
fine-grid oracle resolution 0.05 kW; internal bound-comparison epsilon 1×10⁻¹⁰ kW. `VERIFIED`
**Rationale:** Ten watts is already far below meaningful engine control and component-model
resolution. The continuous engine-on Hamiltonian is tested at its endpoints first; golden-section
search runs only when the finite-difference derivative changes sign. In a fixed-seed 1000-case
benchmark this reduced the mean from 30.6 to 3.842 Hamiltonian evaluations and made the production
solver 4.81× faster than the former 2 kW grid (median of five paired timings). A 75 kW interval
needs about 19 golden contractions at 0.01 kW, versus roughly 29 at the deleted 1×10⁻⁴ kW setting.
**O-11 check:** With load-dependent source efficiency, all 500 fixed-seed cases agreed with the
0.05 kW oracle on feasibility and engine power. Sampled adjacent slopes were non-decreasing over
491 non-degenerate feasible intervals, so the implemented O-11 Hamiltonian remains convex. The
literal 1×10⁻⁹ relative Hamiltonian-agreement requirement failed in 14 interior cases (worst
8.68×10⁻⁹): the continuous solver was lower than the nearest 0.05 kW grid point, so this is oracle
quantisation rather than a local-minimum failure.

### C-05 — PI-ECMS reference state and default gain
**Value:** `soc_ref = 0.6`, `kp = 5.0`, and `s0_ratio = 1.0` relative to `switching_s`.
`UNVERIFIED`
**Rationale:** The reference SoC fixes the point where proportional correction is zero and is held
out of the genetic chromosome to avoid an extra search dimension. The gain is a reproducible
starting point, not a tuned claim; Part E sweeps it from 0 to 20 together with the anchor ratio.
**Bias:** Unknown. Larger gain retains more charge at low SoC but can increase engine cycling and
part-load operation; the direction depends on the mission and restart model.

### C-06 — Experimental fuzzy-ECMS inference and defaults
**Value:** `soc_low = 0.30`, `soc_high = 0.70`, `s_min_ratio = 0.75`, and
`s_max_ratio = 1.25`, with the consequent ratios applied to `switching_s`. `UNVERIFIED` (tuning),
`VERIFIED` (complete rule coverage and monotonic SoC response)
**Rationale:** The controller is an opt-in benchmark for O-08, not the active mission strategy and
not an innovation claim. A zero-order Sugeno system uses complete low/medium/high piecewise-linear
partitions for SoC and normalized bus demand. Low SoC selects the high ratio at every demand and
high SoC selects the low ratio. At medium SoC the ratio falls from `(1 + s_max_ratio)/2` through
`1.0` to `(s_min_ratio + 1)/2` as demand rises, allowing peak assistance only when charge is
available. All nine rules are defined, and invalid thresholds are rejected rather than sorted or
clipped.
**Implementation:** Consequents are derived from the configured ratio bounds rather than anchored
to fixed universe edges, so changing either bound moves the saturated output directly. Product
rule activation and weighted-average defuzzification make inference deterministic and stateless.
The module is not imported by the simulator, mission factory, or optimization skeleton; it runs
only when a caller explicitly injects `FuzzyECMS` through the existing controller interface.
**Bias:** Unknown. These values are reproducible test defaults, not calibrated controller gains.

### C-07 — Terminal-energy recovery tolerances and work cap
**Value:** terminal-target tolerance 1×10⁻⁶ kWh for continuously adjustable replay solves;
maximum 32 backward inductions per finite-horizon scenario. `VERIFIED` (numerical policy)
**Rationale:** The independently measured 1×10⁻¹² kWh ledger-closure tolerance diagnoses
roundoff and is not an optimisation tolerance. Discrete policies can jump across a finite energy
interval, so scalar bisection terminates as soon as it repeats the same adjacent policy pair. The
induction cap is a runtime guard and does not turn an unconverged result into a bound.

### C-08 — Full-mission thermostat integration reference
**Value:** causal scheduling; SoC thresholds 0.4/0.6; minimum ON/OFF dwell 60/60 s; restart fuel
0 kg/start; computed ON power; 60 s mission timestep. `UNVERIFIED` (uncalibrated reference)
**Rationale:** This named configuration exercises the replay-tested thermostat through the complete
six-phase plant simulation without changing the ECMS default or tuning thresholds. The initial
state is explicitly engine ON with 60 s elapsed, so power is available at mission start, minimum
ON dwell is already satisfied, and initial availability is not miscounted as a restart.
**Implementation:** `run_mission` accepts the thermostat only through opt-in keyword arguments and
rejects horizon-aware scheduling. The scheduler emits an engine command and explicit next state;
the simulator alone evaluates and integrates the engine, battery, fuel and restart accounting.
Thermostat and engine restart-fuel values must match to prevent competing ownership.
**Bias:** Optimistic. Zero restart fuel omits start energy, spool delay and engine-life cost. The
thresholds are an integration fixture, not a calibrated or optimal controller claim.

### C-09 — Global thermostat threshold-search resolution
**Value:** search over the actual usable SoC interval [0.05, 1.0]; minimum threshold separation
0.05; 56-point deterministic triangular coarse mesh including (0.4, 0.6); four retained coarse
regions; 0.025 local refinement step; maximum 72 completed full-mission evaluations. `VERIFIED`
(bounded numerical policy), `UNVERIFIED` (resolution sufficiency)
**Rationale:** Two controller variables do not justify stochastic optimisation. The mesh includes
the battery floor, narrow and wide bands, upper thresholds through 1.0 and the untuned reference.
The nonzero separation prevents threshold equality and same-bound numerical chatter at the 60 s
control resolution. Every completed mission is appended and flushed to CSV before the next begins;
resume skips threshold pairs by a stable decimal key.
**Selection:** Endurance is the only primary objective. Infeasible missions are excluded rather
than penalised into the ranking. Exact event-time ties within 1×10⁻⁶ s are disclosed and broken by
restart count, transition-duration margin and remaining resources, in that order. No terminal-SoC
reward or depletion penalty is applied.
**Bias:** The result is the best feasible pair found inside these bounds and this resolution, not a
proof of continuous global optimality. Zero restart fuel retains the optimistic bias from C-08.

### C-10 — Frozen-aircraft controller comparison and restart sensitivity
**Value:** fixed-ECMS local ratios 1.0/1.1/1.2; PI local ratios 1.2/1.3/1.4 and gains
0/2.5/5.0; restart-fuel sensitivities 0/0.1/0.5 kg per OFF-to-ON transition; optional 0.1 kg/start
retuning capped at 18 full missions. For the practical recommendation only, retaining at least 99%
of the best zero-cost endurance is treated as close. `VERIFIED` (bounded numerical policy),
`UNVERIFIED` (99% recommendation threshold), `PLACEHOLDER` (positive restart-fuel values)
**Rationale:** The historical fixed/PI headline settings are not accompanied by executable sweep
code or checkpoints in the current tree. The stated neighbourhood reconstructs only the adjacent
historical resolution needed to detect a stale selected point. Controller parameters are selected
at zero restart cost, then frozen for the three common simulator reruns. A changed 0.1 kg/start
ranking alone opens the bounded retuning gate; it does not promote that uncalibrated value to a
physical claim. Figures read their plotted values back from source CSV and retain the same
controller colour identity across PNG, SVG and PDF exports.
**Bias:** Zero restart cost favours memoryless ECMS switching. Positive values expose sensitivity
but cannot identify the physically correct transition penalty without calibration.

### S-01 — Fill-to-MTOW fuel policy
**Value:** Fuel loaded = MTOW − dry mass. `VERIFIED`
**Rationale:** Makes the powertrain-versus-fuel trade explicit and direct. Every kilogram spent on
engine, battery, or structure is a kilogram of fuel not carried.

### S-02 — Loiter is the endurance phase
**Value:** Loiter extended until fuel reserve or SoC cutoff is reached; total mission time is the
endurance being maximized. `VERIFIED`
**Rationale:** Directly implements the stated objective while retaining descent and landing after
the resource-terminated phase. The profile guarantees that those phases are present; whether a
particular aircraft can complete them on the reserve belongs to mission feasibility.

### S-03 — Integration scheme and timestep
**Value:** Explicit Euler, Δt = 60 s by default; per-phase override supported. `VERIFIED` for the
reference mission convergence study.
**Rationale:** Reference endurance was 10 808.007 s, 10 817.894 s, 10 822.985 s and 10 825.527 s
at 120, 60, 30 and 15 s respectively. The 30-to-15 s change is 0.0235%, and the 60 s result is
0.0705% below the 15 s result. This is comfortably below the modelling uncertainty elsewhere.
Using 300 s only in loiter changed endurance by 0.0383% relative to uniform 60 s and ran 2.41×
faster in the test benchmark, so the override is appropriate for large GA runs.
**Bias:** The reference sequence converges upward as the step is shortened, so 60 s is slightly
conservative for endurance in this case. This measured bound is configuration-specific, not a
proof for every controller or design.

### S-04 — Reserves and limits
**Value:** Post-landing reserve 5.0 kg; descent-and-landing allocation 4.7 kg; minimum usable fuel
for feasibility 20 kg; maximum mission time 24 h. `UNVERIFIED`
**Rationale:** On the 86.779 kW band-interpretation aircraft, the resolved shutdown policy and
default PI controller consumed 4.257 kg after loiter. A phase-only cutoff-SoC check consumed
4.108 kg. The default uses the larger measurement, adds 10%, and rounds upward to 0.1 kg. Loiter
therefore stops at 9.7 kg, descent and landing may spend 4.7 kg, and 5.0 kg must remain after
landing. Falling below that post-landing reserve is reported as `fuel_reserve_shortfall`. The 24 h
cap is a loop safety bound rather than a physical limit.

### S-05 — Default cruise altitude and vertical rates
**Value:** Cruise altitude 3000 m; target climb rate +2.0 m/s; target descent rate −3.0 m/s.
`OPEN` (altitude — see O-07), `UNVERIFIED` (rates)
**Rationale:** 3000 m is the lower edge of the mandated 3–10 km cruise band and is the reproducible
factory default while altitude remains open. It replaces the previous 6000 m default, where the
power lapse of a 60 kW-class turboshaft is already material. The rates are mission targets rather
than guaranteed performance: climb duration is derived from the achieved rate, and the simulator
must limit the target against available excess power and report any shortfall. The public factory
takes descent rate as a positive downward magnitude; the phase stores −3.0 m/s under the
positive-up sign convention.
**Bias:** Unknown. Altitude trades lower-density aerodynamic effects against engine power lapse, so
the direction cannot be assigned without the O-07 sweep.

### S-06 — Default fixed phase targets
**Value:**

| Phase | Fixed airspeed | Fixed duration | Altitude behaviour |
|---|---:|---:|---|
| Take-off | 50 m/s | 120 s | Hold ground reference |
| Climb | 65 m/s | — | Climb to cruise altitude |
| Cruise | 69.44 m/s | 3600 s | Hold cruise altitude |
| Loiter | solved by minimum power | open-ended | Hold cruise altitude |
| Descent | 65 m/s | — | Descend to zero altitude |
| Landing | 45 m/s | 120 s | Hold zero altitude |

`MANDATED` (cruise speed and phase order), `UNVERIFIED` (remaining fixed targets)
**Rationale:** Cruise speed is the problem-statement value of 250 km/h. The other fixed speeds and
durations are conceptual mission inputs retained as explicit factory arguments, not aircraft
performance claims. Climb and descent terminate on altitude rather than time. Loiter defaults to
the solved minimum-power condition but retains fixed-speed and best-L/D modes for O-06.

### S-07 — Pre-dispatch equivalence-factor reference point
**Value:** Engine-only, demand-following shaft power, clamped to the current lapsed engine range;
source-chain efficiency is evaluated at that same power for both `neutral_s` and `switching_s`.
`UNVERIFIED` (control-policy choice)
**Rationale:** The actual engine SFC is an outcome of the ECMS split. It therefore cannot be used to
compute `neutral_s` before the controller and split solver run without a fixed-point iteration or a
previous-step state. The engine-only operating point is deterministic on the first step, remains
load-dependent under O-11, and introduces no controller history into the simulator. The actual SFC
is still logged after the split. `switching_s` uses the Willans slope at that same reference and is
the controller anchor; `neutral_s` is diagnostic only.
**Bias:** Under O-11, using average source-chain efficiency at the reference power only
approximates the exact marginal bus-power derivative. Iterating the controller and split, or using
that derivative directly, is a distinct policy rather than a neutral implementation detail.

### CD-01 — Maximum stall speed for constraint sizing
**Value:** V_stall,max = V_landing / 1.2 = 45 / 1.2 = 37.5 m/s at sea level. `UNVERIFIED`
**Rationale:** The problem statement does not supply a stall speed. The constraint diagram derives
one from the mission profile's 45 m/s landing speed and the standard 1.2 speed margin already used
by the aerodynamic model. With C_Lmax = 1.5 this limits W/S to 1292.0 N/m². Landing speed remains
an assumed mission input, so the derived limit is not promoted beyond it.
**Bias:** Unknown. A lower certified landing margin or high-lift landing configuration would permit
greater wing loading; handling-quality or field constraints could demand less.

### CD-02 — Service-ceiling residual climb rate
**Value:** ROC = +0.5 m/s at 10 000 m. `UNVERIFIED`
**Rationale:** The maximum operating altitude is checked with a small positive residual climb rate,
rather than defining ceiling at exactly zero excess power. The 0.5 m/s convention is a sizing input,
not a problem-statement requirement.
**Bias:** Conservative relative to a zero-rate absolute ceiling; optimistic if operational climb
performance at 10 km must exceed 0.5 m/s.

### CD-03 — No take-off-distance constraint
**Value:** Omitted because no runway length, obstacle height, or ground-roll target is specified.
`UNVERIFIED`
**Rationale:** A take-off-distance curve cannot be derived from take-off speed alone. The transient
climb curve still sizes airborne peak power, but it is not a substitute for field performance.
**Bias:** Optimistic. A later runway requirement can only shrink the feasible region.

### CD-04 — Constraint-diagram selection margin and reference case
**Value:** 10% installed-power margin; reference transient boost = 30 kW at the DC bus. `UNVERIFIED`
**Rationale:** The margin is applied after taking the envelope of all sampled power constraints.
The reference boost is the 10 kWh pack's 3C discharge rating from B-04; it bypasses generator and
rectifier losses but still supplies the motor-side demand chain. The report design case combines
cruise at the 3000 m default, +2 m/s transient climb at 3000 m, and a +0.5 m/s service-ceiling check
at 10 000 m under the ceiling-required interpretation. The band interpretation in O-12 omits that
imposed ceiling constraint. These point constraints do not prove that the battery can sustain the
boost or that the mass and fuel-volume budgets close.
**Bias:** The power margin is conservative; treating the full 3C pack rating as continuously
available throughout climb is optimistic until mission energy is checked.

### CD-05 — Measured altitude trend of the reference cruise constraint
**Value:** Installed rating at W/S = 980.665 N/m² is 92.211, 92.206, 92.551, 93.337, 94.678,
96.725, 99.674, 103.781, 109.384, 116.936 and 127.042 kW at 0–10 km in 1 km steps. `VERIFIED`
**Rationale:** The shallow minimum is at 1 km, but the curve is not approximately flat over the
whole altitude band: its 10 km value is 37.8% above its minimum. Density reduction initially trims
parasite power, after which induced power and the turboshaft lapse dominate. This corrects the
constraint-diagram specification's unsupported “roughly flat across 0–10 km” claim.

---

## 8a. Optimization — `optimization/chromosome.py`

### OPT-01 — Initial plant–thermostat chromosome design space
**Value:** Six normalized binary64 genes in the fixed order `wing_area`, `aspect_ratio`,
`engine_rating`, `battery_capacity`, `thermostat_soc_low`, `thermostat_soc_gap`. Physical plant
bounds are 6–16 m², 10–24, 60–140 kW and 5–30 kWh. The lower thermostat threshold spans the
configured battery cutoff to 0.60; the dependent upper threshold spans from 0.05 above that lower
threshold to 0.95. `UNVERIFIED` (physical search bounds), `VERIFIED` (encoding and inverse)
**Rationale:** The dependent gap coordinate makes every decoded pair ordered and separated without
repair after crossover or mutation. Bounds remain an explicit immutable input to every transform;
their versioned identity and each normalized binary64 value's exact hexadecimal representation
enter the SHA-256 cache key, so no decimal rounding merges distinct candidates.
**Seeds:** The two deterministic anchors use the stored band-aircraft precision of 7.59175537062125
m² and 86.7791369750147 kW, with aspect ratio 16 and battery capacity 10 kWh. These differ from the
rounded prompt values by −0.00024462937875 m² and +0.0001369750147 kW. The practical threshold pair
is 0.225/0.350 and the ideal restart-fuel pair is 0.225/0.300. They initialize a later search; they
are not production defaults or global-optimality claims.
**Exclusions:** C_D0 is a geometry-derived quantity; fuel follows fixed-MTOW mass closure; restart
fuel and dwell are external physical scenarios; aerodynamic, battery and engine uncertainty inputs
belong in sensitivity analysis; the thermostat computes maximum feasible ON power; mission inputs
and GA operators are not aircraft genes. Cruise altitude remains excluded while O-07 is open.
**Integration gate:** The current mission `Aircraft` still stores and forwards a fixed C_D0 and the
reference builder still sets 0.028. OPT-02 now resolves the wing-area gate only for the opt-in GA
static path; no ordinary mission or controller-study path uses its calibrated wetted-area model.
The reference path similarly stores Oswald efficiency 0.78 independently of aspect ratio. Aspect
ratio still changes induced drag through 1/(π·AR·e) and wing mass through the Raymer AR^0.6 term,
but no Oswald correlation is called automatically; conclusions remain conditional on O-01.
**Bias:** Unknown. Wide provisional bounds expose model behaviour but do not validate the physical
extremes, and fixed C_D0 would artificially reward wing-area reduction if the integration gate were
ignored.

### OPT-02 — Static plant resolution and reference-calibrated wetted area
**Value:** `optimization/feasibility.py` is an opt-in, immutable algebraic screen between chromosome
decoding and later mission fitness. Its nominal policy is `reference_calibrated_wetted_area`, with
C_fe = 0.0055, t/c = 0.15, reference S = 7.59175537062125 m² and reference C_D0 = 0.028.
The inferred fixed fuselage, empennage, boom and other non-wing aggregate is
22.896044038214548 m². `PROVISIONAL` (aggregate geometry), `VERIFIED` (implementation identities)

The wing approximation and total buildup are

```
S_wet,wing = 2 S (1 + 0.25 t/c)
S_wet,total = 22.896044038214548 + S_wet,wing
C_D0 = C_fe S_wet,total / S
```

The final line is evaluated by `parasite_drag_from_wetted_area`, not reimplemented in the
optimization layer. At S = 6, 7.59175537062125, 10 and 16 m² it gives total wetted areas of
35.346044, 38.648936, 43.646044 and 56.096044 m², and C_D0 values of 0.03240054, exactly 0.028,
0.02400532 and 0.01928302. Although C_D0 falls, C_D0 S rises monotonically over those points, so
enlarging the wing does not receive free parasite-drag-area reduction. The fixed aggregate is an
inference that preserves the controller-study reference aircraft, not measured component geometry;
later sensitivity must vary or replace it.

**Scenario and policy isolation:** The scenario serializes mission point conditions, mass-model
inputs, battery branch and rates, powertrain efficiencies and load-dependence branch, engine lapse
and shutdown branch, thermostat bounds, fuel/tank inputs, wetted-area calibration and fixed Oswald
policy. A SHA-256 identity uses lossless binary64 values. Evaluation writes no files and does not
import the simulator, thermostat, fitness or GA. The default reference-aircraft C_D0 and the 51 m²
aerodynamics formula fixture are unchanged.

**Nominal aerodynamic policy:** e = 0.78 is fixed and recorded as `fixed_reference`; aspect ratio
still changes k = 1/(π AR e) and the authoritative wing-mass regression. Activating aspect ratio in
a later fitness calculation is therefore mechanically consistent but conclusions remain
conditional on this fixed-e choice and O-01.

**Hard constraints:** finite positive wing area, aspect ratio, engine rating and battery capacity;
nonnegative component masses; thermostat floor, lower/upper limits and minimum gap; fixed-MTOW
closure; minimum usable fuel; tank volume; landing-derived stall margin; 3 km cruise rating with
the CD-04 10% margin; 3 km mission-speed climb rating with sustainable battery assistance and the
same margin; take-off point combined engine/battery power with that margin; and battery discharge
sustainable over the 60 s static-screen step. The 10 km, +0.5 m/s service-ceiling point is advisory
under the active O-12 3 km band interpretation and becomes hard only through an explicit scenario
flag. No take-off-distance surrogate is added.

Each record carries the physical quantity, bound, signed margin, units, source and a dimensionless
violation. Lower-bound violations use `max(0, required - available) / max(|required|, ε)` and
upper-bound violations use `max(0, actual - allowed) / max(|allowed|, ε)`. Binary64 discrepancies
at or below ε = 10⁻¹² relative are recorded as zero violation. Only hard violations enter their
exact sum, maximum and count; advisory constraints and warnings cannot make a candidate infeasible.

**Static reference audit:** The practical seed resolves to span 11.021256 m, chord 0.688829 m,
C_D0 = 0.028, e = 0.78 and stall speed 37.496485 m/s against 37.5 m/s allowed. Component masses
are: fixed group 250.000000 kg, payload 200.000000 kg, wing 97.041791 kg, engine 24.794039 kg,
generator 28.926379 kg, rectifier 5.785276 kg, inverter 7.221211 kg, motor 15.474024 kg,
cabling/cooling 8.611034 kg, battery 53.333333 kg and fuel system 20.202714 kg. Dry mass is
711.389802 kg and the un-clipped fuel residual is 288.610198 kg. Required/available tank volumes
are 358.967908/392.206313 L, a +33.238405 L margin.

The reference engine rating is 86.779137 kW sea-level shaft and 68.360327 kW at 3 km. Required
sea-level-equivalent ratings after the 10% margin are 86.779137 kW cruise, 69.881003 kW climb and
6.485925 kW at the take-off point, giving numerical-zero, +16.898134 and +80.293212 kW margins.
The battery can sustain 30.000000 kW for the screen step against 13.469851 kW required, a
+16.530149 kW bus margin. Every hard record passes and total hard violation is zero. The advisory
10 km ceiling requires 174.499453 kW and misses by 87.720316 kW. Warnings identify the inferred
wetted-area aggregate, fixed-e conditionality and the fact that static passage does not prove the
six-phase mission.

**Bias:** Unknown. The reference identity is exact by calibration, but extrapolation across wing
area assumes the entire non-wing wetted area stays fixed and the wing thickness ratio stays fixed.
Static feasibility also omits fuel burn, SoC trajectory, phase transitions, reserves, restart fuel,
dwell behaviour and time-varying power balance; those remain the responsibility of later mission
fitness.

### OPT-03 — Single-candidate full-mission fitness contract
**Value:** The nominal scenario is the existing six-phase 3 km mission at its 60 s default step,
1000 kg MTOW, 200 kg payload, 0.1 kg fuel per OFF-to-ON start and 60 s hard ON/OFF dwell. The
thermostat is causal, uses one global SoC band and retains the simulator's maximum-feasible ON-power
rule. The objective is loiter duration alone. `PLACEHOLDER` (restart fuel), `UNVERIFIED` (dwell),
`VERIFIED` (integration and deterministic reference reproduction)

**Scope and isolation:** `optimization/fitness.py` decodes and statically screens one normalized
chromosome, constructs fresh immutable plant/controller inputs from its `ResolvedPlantDesign`, and
calls `run_mission` at most once. Static failure skips simulation. The result and cache identity
include the chromosome, design-space, static-scenario, fitness-scenario and result-schema identities;
no process-randomized hash or persistent cache is used. The 0.1 kg/start scenario is an uncalibrated
practical sensitivity, not a validated start model.

**Plant and feasibility contract:** Fuel is the single fixed-MTOW residual, 1000 kg minus resolved
dry mass; component masses are not added again. Wing area supplies reference area, calibrated C_D0
and wing mass; aspect ratio supplies induced drag and wing mass; engine rating supplies lapsed power
and propulsion mass; battery capacity supplies stored energy, mass, resistance and rate limits; the
two thermostat genes supply the lower and decoded upper thresholds. Static constraints cover the
algebraic design screen. Dynamic feasibility separately requires all mandatory phases including
descent and landing, reserves and battery floor, feasible controller/plant delivery, hard dwell and
restart accounting, and closed bus, fuel and discrete battery-energy ledgers. Normal fuel-reserve
loiter exit is allowed when descent and landing subsequently complete. Terminal SoC equality is not
required in this variable-duration problem, and unused fuel or energy is diagnostic only.

Active engine or battery limiting is not by itself a power failure when delivered bus power still
balances demand; the authoritative reference exercises both limits. The endpoint battery-energy
residual retains the documented explicit-Euler OCV integration bias and is reported as a warning;
the independently reconstructed discrete-energy residual is the hard closure gate. Physical
normalization scales are distinct from numerical ledger tolerances.

**Failure policy and provisional assumptions:** Only the fitness runner adapter's documented
candidate-physical exception is converted to structured infeasibility. Unexpected exceptions,
assertions, schema errors and nonfinite simulator output propagate. The reference-calibrated
wetted-area extrapolation and fixed e = 0.78 remain provisional under OPT-02 and O-01. No GA,
population, sweep, sensitivity analysis or optimization result is part of this milestone.

**Reference gate:** The practical reference chromosome at S = 7.59175537062125 m², AR = 16,
86.7791369750147 kW, 10 kWh and SoC band 0.225/0.350 reproduces the stored 0.1 kg/start comparison:
54876.880130153375 s total, 48536.880130153375 s loiter and 50 restarts, with the stored fuel, SoC,
OFF-fraction, completion and ledger diagnostics within their established deterministic tolerances.

### OPT-04 — Deterministic constrained genetic-algorithm method
**Value:** One run uses 64 individuals, 40 total evaluated populations including generation zero,
two elites, tournament size three, pairwise crossover probability 0.90, bounded simulated binary
crossover with eta_c = 15, independent bounded polynomial mutation with probability 1/6 per gene
and eta_m = 20, and ten-generation early-stop patience at 10^-4 relative material improvement.
The deterministic default development seed is 20260808; production seed sets remain an external
orchestration choice. Exact-duplicate construction retries are capped at 32 before the documented
fresh-uniform/final-duplicate fallback, and a 10^-12 combined-violation tolerance only absorbs the
fitness layer's existing normalization roundoff.
`UNVERIFIED` (initial hyperparameters), `VERIFIED` (deterministic implementation and resume)

**Initialization and replacement:** Forty-eight normalized chromosomes use independently permuted
six-dimensional Latin-hypercube strata. The remaining sixteen are the exact practical and ideal
reference seeds plus seven reflected normal-space perturbations around each at initial standard
deviation 0.05 per gene. Full generational replacement retains two unchanged elites and creates 62
offspring positions. All operators act on the normalized lower-threshold/gap encoding; no decoded
threshold repair, physical-unit operator or adaptive mutation is used. These values are justified
starting settings for six variables and have not been hyperparameter-tuned.

**Ordering and stopping:** Deb ordering is authoritative: feasible beats infeasible, feasible
candidates maximize loiter seconds, infeasible candidates minimize combined normalized violation,
and exact ties use deterministic evaluation identity only. Resource slack, restart count and final
SoC are not secondary objectives. A sub-threshold objective increase updates the exact best-found
record without resetting patience. Stagnation is unavailable until a feasible result exists, and
means only that the finite stochastic search stopped—not mathematical convergence or global
optimality.

**Budget and recovery:** With generation zero plus 39 offspring generations, the pre-cache upper
bound is 64 + 39(64 - 2) = 2482 candidate placements per seed, or 7446 for three seeds. Exact
chromosome/evaluation identities cache feasible and infeasible results without rounded matching.
Every new evaluator result is immediately appended to a checksummed JSONL ledger and `fsync`ed;
every completed generation atomically replaces a checksummed JSON checkpoint containing population,
history, counters, stagnation state and complete NumPy PCG64 state. Resume validates chromosome,
bounds, GA, scenario and codec identities, ignores and removes only an interrupted final JSONL tail,
and reuses committed post-checkpoint evaluations before replaying the same stochastic operations.

**Interpretation:** Outputs must be called a `best_found`, `best_feasible_found` or search result.
No production UAV population, multiple-seed comparison or GA hyperparameter optimization has been
run. Sensitivity analysis remains later work because GA convergence diagnostics do not measure
model-input uncertainty or resolve the provisional assumptions in OPT-02/OPT-03.

**Bias:** Unknown. The selected operators and finite budget can miss better feasible regions, while
the two reference-centred seed clouds can bias early sampling toward the current reference plant.

---

## 9. Open decisions

These must be resolved and this document updated before final submission.

| ID | Decision | Options | Blocks |
|---|---|---|---|
| O-01 | Oswald efficiency treatment | Raymer AR correlation vs fixed documented value with sensitivity sweep | AE-07, aspect-ratio result validity |
| O-02 | Wing area and aspect ratio as design variables | Optimize both vs freeze with written justification | AE-04, M-03, gene set |
| O-03 | Propeller efficiency model | Constant 0.85 vs phase-dependent vs variable-pitch assumption | AE-11, peak power demand |
| O-05 | Internal resistance scaling | Fixed 0.05 Ω vs scaled with pack capacity (both implemented; scaled is the default) | B-03 |
| O-06 | Loiter speed | Solved from minimum-power/stall-margin condition each step vs fixed | AE-05, AE-06, mission profile |
| O-07 | Cruise altitude | Fixed at a chosen value vs treated as a design variable | E-04, mission profile |
| O-08 | Energy-management controller | Fuzzy adaptive vs PI feedback vs fixed equivalence factor | Gene set, innovation claim |
| O-09 | Design limit load factor | 3.8 (FAR 23 normal category, manned) vs 2.5–3.0 (typical MALE-class UAV) | M-03, wing mass |
| O-10 | Usable tank volume fraction | 0.5 (current) vs 0.30–0.35 (inter-spar box) | M-06, whether the volume check binds |
| O-11 | Load-dependent component efficiency | Constant stage efficiencies (current default) vs a no-load-plus-load-squared loss model (both implemented behind `load_dependent`) | P-01, P-02, loiter power demand |
| O-12 | Meaning of the stated 3–10 km cruise-altitude band | Require a 10 km service ceiling vs select cruise altitude within the band and report achievable ceiling | CD-04, engine rating, fuel mass |

**O-04 (resolved 2026-08-05).** Engine shutdown is the default because the series-hybrid battery
can carry low bus demand while the turboshaft is off; idle mode remains available for sensitivity.
On the band-interpretation aircraft with default PI control, shutdown increased simulated time
from 14.203 h to 15.606 h and shut the engine down for 21.91% of loiter. The idle case missed the
post-landing reserve by 0.170 kg under the shutdown-sized descent allocation. Free restarts remain
optimistic: 0.1 kg per restart reduced endurance to 14.610 h over 163 restarts, while 0.5 kg reduced
it to 12.028 h over 120 restarts and caused a reserve shortfall. These sensitivities also expose
timestep-dependent chatter until a dwell-time or start-transient model is added.

**O-12.** Both readings are retained for presentation. Imposing a 10 km service ceiling selects
14.045 m² and 133.270 kW, bound by `ceiling_10km`, with 807.120 kg dry mass and 192.880 kg fuel.
Treating 3–10 km as the selectable cruise band and sizing only at the chosen 3 km cruise/climb
conditions selects 7.592 m² and 86.779 kW, bound by `cruise_3km`, with 711.390 kg dry mass and
288.610 kg fuel. The latter design's achievable 0.5 m/s service ceiling is 5.842 km and its
zero-rate absolute ceiling is 6.825 km. No interpretation is silently preferred for judging; the
band design is used for the controller study because the mission flies at 3 km.

**O-06.** The mission profile now represents speed as a mode. `MIN_POWER` is the current loiter
default, `FIXED` accepts an explicit airspeed, and `BEST_LD` is available for range-oriented
comparisons. This structural support does not resolve which loiter policy should be used in the
final design; the simulator must resolve solved modes from the current aircraft state each step.

**O-07.** `ps1_mission(cruise_altitude_m=...)` propagates its argument through climb, cruise and
loiter, so fixed-altitude studies and a GA-supplied altitude use the same profile structure. The
3000 m factory default in S-05 is a baseline value, not a resolution of the open decision.

**O-11.** The vehicle loiters at roughly a third of the inverter and motor ratings, which is where a
constant-efficiency assumption is least accurate, and four stages compound. Under the loss model
P_loss = a + b·P_out², calibrated so each stage recovers its rated efficiency at its rating with 30%
of the rated loss load-independent, the demand chain falls from 0.8935 to 0.8833 at 30 kW of shaft
demand on a 90 kW bus rating — 1.15% more bus power for the same thrust. The compounded chain moves
less, 0.8064 to 0.7995, because the generator and rectifier sit nearer their own best point at that
condition. Note that the model peaks *below* rated, at √(f/(1−f)) = 65.5% of the rating where the
no-load and load losses are equal, and falls back to the rated value at 100% — efficiency is not
monotonic in load. Sizing each stage is therefore not a free choice: rating a machine far above its
cruise load costs efficiency at cruise.

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
| 2026-08-08 | **GA Milestone 3 deterministic engine.** Added OPT-04 and `optimization/ga.py`: 48-point Latin-hypercube plus 16-member reference-seeded initialization, Deb ranking, tournament/SBX/polynomial operators, two-elite replacement, exact evaluation caching, immutable histories/results, material-improvement stopping, checksummed JSONL evaluation ledger, atomic JSON generation checkpoints and exact PCG64 resume. The 40-total-population convention caps one seed at 2482 placements. Only analytical/mock objectives were optimized; no production mission population or hyperparameter tuning was run. |
| 2026-08-08 | **GA Milestone 2 full-mission fitness.** Added OPT-03 and `optimization/fitness.py`: explicit immutable 3 km/0.1 kg-start scenario, authoritative resolved-plant conversion, fixed-MTOW fuel closure, single-call mission evaluation, loiter-only objective, structured static/dynamic violations, fresh-state isolation and deterministic evaluation identity. The one practical reference mission reproduced its stored artifact; no GA, sweep or optimization was run. |
| 2026-08-08 | **GA Milestone 1 static plant feasibility.** Added OPT-02 and `optimization/feasibility.py`: immutable resolved geometry, component mass/fixed-MTOW fuel closure, tank and point-power capability, Deb-compatible normalized hard violations, advisory 10 km ceiling handling and deterministic scenario identity. Authorised the opt-in `reference_calibrated_wetted_area` policy, which infers 22.896044038214548 m² fixed non-wing wetted area and exactly preserves C_D0 = 0.028 at the reference wing. Fixed e = 0.78 remains explicit and conditional; ordinary mission aircraft, controller results, fitness and GA are unchanged. |
| 2026-08-08 | **Plant–thermostat chromosome foundation.** Added OPT-01 for the immutable six-gene normalized representation, explicit battery-floor-derived design space, dependent threshold inverse, lossless binary64 cache canonicalization, deterministic serialization and exact practical/ideal reference seeds. Recorded that current mission C_D0 is fixed as area changes and that Oswald efficiency is stored independently of aspect ratio; no fitness, mission, GA operator or optimization result was added. |
| 2026-08-07 | **Frozen-aircraft controller comparison.** Added C-10 for bounded reconstruction of the missing fixed/PI local sweeps, exact zero-cost controller reruns, common authoritative restart-fuel sensitivities, conditional 0.1 kg/start local retuning, resumable mission checkpoints and CSV-backed presentation figures. The practical recommendation defines "close" as retaining at least 99% of the best ideal endurance and leaves that judgment threshold unverified. Plant values, controller defaults, battery physics, DP, fuzzy ECMS and GA remain unchanged. |
| 2026-08-07 | **Controller-only thermostat threshold study.** Added C-09 for the deterministic 72-evaluation coarse-to-fine search, append-only resume checkpoints, phase fuel/battery ledgers, feasibility-first ranking, exact winner repeat, one-step loiter extension check and conditional phase-dependent gate. Plant parameters, dwell, restart, ON-power physics and ECMS defaults remain frozen. |
| 2026-08-07 | **Opt-in full-mission thermostat integration.** Added C-08, separated the pure thermostat command from replay plant evaluation, carried explicit dwell/transition state through all mission phases, retained `run_mission` as the single plant and restart-fuel accounting owner, and added the named uncalibrated six-phase reference report. Existing ECMS dispatch remains the ordinary path. |
| 2026-08-07 | **Milestones 2/3 numerical-validity recovery.** Added C-07, separated ledger and terminal-target residuals, replaced raw opposite-energy fuel intervals with endpoint-policy data, added a 208-policy exhaustive oracle, Lagrangian dual lower bounds and exact discrete-target upper bounds, made dwell hard in thermostat and DP comparisons, and added caching, policy hashes, induction caps, incremental checkpoints and resume support. Historical production artifacts are retained but reclassified as exploratory in `docs/numerical_validity_recovery.md`. |
| 2026-08-07 | **Opt-in `control/fuzzy_ecms.py` benchmark.** Added C-06 for a complete 3×3 zero-order Sugeno controller using SoC and normalized bus demand, with consequent bounds expressed as ratios around the marginal `switching_s` reference. The controller remains absent from mission defaults and the optimization skeleton, so O-08 stays open. Added contract, rule-coverage, gene-responsiveness, validation and power-split boundary tests. |
| 2026-08-05 | **Marginal controller anchor, reserve semantics, and sizing interpretations.** Added `switching_s` to the controller context and made it the fixed-ratio and PI-ratio anchor; `neutral_s` remains an average-cost diagnostic. Resolved O-04 to engine shutdown by default while retaining idle sensitivity, and added simulator accounting for off-to-on restart fuel. Split S-04 into 4.7 kg for measured descent/landing consumption and a separate 5.0 kg post-landing reserve, with explicit reserve-shortfall reporting. Opened O-12 and recorded both constraint results: 133.270 kW with a required 10 km ceiling versus 86.779 kW when the stated altitude is treated as a selectable cruise band. |
| — | Initial version. Fixed-mass group revised from 450 kg lumped to 250 kg itemized (M-02) following explicit modelling of wing and electrical chain masses. Engine specific power revised from 1.5 to 3.5 kW/kg (M-04). Constant SFC replaced by Willans line (E-01). |
| 2026-08-04 | **`mass.py` build.** M-03: N_z corrected from limit to ultimate load factor — the entry previously listed a single row `N_z = 3.8`, which fed the regression the limit value and under-predicted wing mass by 22%; equation form and units promoted to `VERIFIED`, parameter values remain `UNVERIFIED`. M-06: the claim that the fuel volume check penalizes high aspect ratio corrected — wing *area* is the sensitive variable (binds below S ≈ 7 m²), the aspect-ratio crossover at AR ≈ 46 is physically irrelevant; usable-volume fraction downgraded to `PLACEHOLDER`. O-09 (limit load factor) and O-10 (tank volume fraction) opened. |
| 2026-08-04 | **`engine.py` build.** E-04: recorded that the Willans coefficients are held altitude-invariant and that only maximum power lapses; neglecting the colder-inlet SFC gain is conservative. E-05: floor fixed at 15% of rated, both idle and shutdown modes implemented behind a flag pending O-04, restart fuel marked `PLACEHOLDER` at 0.0 kg and flagged as optimistic. |
| 2026-08-05 | **`powertrain.py` build, and `battery.py` limiting moved into current space.** P-01: cabling added as a fifth stage, the compounded chain recorded as 0.8064 engine shaft to propeller shaft, and the previous implementation's omission quantified — 31.58 kW of engine shaft power called for against 37.20 kW required at 30 kW of shaft demand, understating fuel burn by 15.1%. P-04 opened for DC distribution at 0.99, previously neglected entirely. P-05 opened to fix the system efficiency metric, with battery charging excluded from the denominator. O-11 opened for load-dependent stage efficiency; recorded that the model peaks at 65.5% of rating rather than rising monotonically to it, and that the demand chain falls 1.15% at the loiter condition while the compounded chain falls 0.86%. B-06: limiting moved from power space into current space and the SoC clamp removed outright rather than retained as a guard — an energy-limited step now lands on its boundary exactly, and the reported bus power is the quadratic at the integrated current rather than the command. `available_discharge_kw` and `available_charge_kw` take `dt_s` as optional rather than required, restoring the rate-limit-only query; recorded that the controller must pass it or its own bus balance will not close. |
| 2026-08-05 | **`battery.py` revision — energy-limited discharge.** B-06: rewritten and retitled. Available power is now the power sustainable for the whole step, not the instantaneous capability, closing a path by which a step starting above the cutoff integrated through it and delivered energy the pack did not hold — 0.25 kWh against 0.0435 kWh available in the worked 5 kWh case, and the resulting SoC of 0.0019 sat below the hard floor with no flag, because the clamp is to [0, 1] and never fired. Availability model promoted to `VERIFIED` in form. Recorded: the energy ceiling must be capped at the ohmic ceiling before the quadratic is evaluated, or it returns large negative powers at short timesteps; start-of-step OCV retained over a midpoint evaluation, with the R_eff closed form for the midpoint case and the resulting optimistic bias tabulated (0.71% worst case per step, 1.2×10⁻³ over a full discharge at Δt = 60 s, first order in Δt); the SoC clamp is no longer reachable and a 1×10⁻⁹ boundary tolerance added so `at_cutoff` is not decided on rounding; the Δt coupling of availability accepted and justified. `power_limited` split into `rate_limited` and `energy_limited` with the old name kept as their disjunction. B-04 deliberately **not** changed: the C-rate limit remains a bus-power cap of C·E_kWh, not a current cap of C·Q_nom, which would have silently restated the 10 kWh pack's 30 kW limit as 29.63 kW. |
| 2026-08-04 | **`battery.py` build.** B-03: internal resistance scaling model recorded as R(E) = 0.05 · (10 / E_kWh), implemented behind `scale_resistance` with the scaled form as default pending O-05; the quadratic current solution and the Q_nom = E/V_nominal consistency promoted to `VERIFIED` in form; the naive P/V_oc error quantified at 1.26% in current and 0.377 kW in unbilled loss at the reference condition; ohmic power ceiling V_oc²/(4R) recorded as an enforced guard. B-04: retitled to cover both rate limits, charge C-rate of 1C recorded, and the previous engine-rated power limit identified as an implied 15C on a 5 kWh pack. B-06: clamp-and-flag behaviour recorded, and the 20% floor documented as reported-not-enforced. |
| 2026-08-05 | **`control/base.py` build.** Added C-01 absolute equivalence-factor sanity rails at 0.5 and 20.0, applied by the concrete base-class entry point rather than by individual controllers. Added C-02 with the neutral-factor round-trip correction divided by η_rt and recorded 1.0 as the uncorrected default. The neutral factor remains operating-point dependent because SFC arrives from the Willans engine model. |
| 2026-08-05 | **`simulation/mission.py` build.** Replaced fixed-duration climb and descent with altitude termination and signed target rates; changed loiter from a fixed 70 m/s to a selectable speed mode with minimum power as the default; made every mission input a factory argument; and added immutable phase/profile validation. Recorded the 3000 m cruise-altitude default and the remaining fixed phase targets in S-05/S-06 without resolving O-06 or O-07. |
| 2026-08-05 | **`control/power_split.py` build.** Added feasible-bound intersection, separate off/idle candidates, golden-section ECMS minimisation, explicit infeasible decisions, active-bound diagnostics, and a fine-grid test oracle. C-03 distinguishes the Willans marginal switching factor (4.776) from the average-SFC neutral factor (5.970). Corrected the specification's description of Rint loss: it is quadratic in current, only approximately quadratic in bus power, while the exact Hamiltonian remains convex. The solver uses the powertrain forward/inverse APIs so O-11's load-dependent source chain remains reachable. |
| 2026-08-05 | **Power-split fast path and `simulation/simulator.py` integration.** C-04 updated from a 1×10⁻⁴ to 0.01 kW production tolerance and finite-difference endpoint screening; measured 3.842 Hamiltonian evaluations per solve and 4.81× speedup over the 2 kW grid. The 500-case O-11 sweep found a convex continuous Hamiltonian and power agreement within the 0.05 kW oracle resolution; documented the oracle's 14 quantisation-level failures of the over-tight 1×10⁻⁹ relative Hamiltonian criterion. Added the one-step whole-stack integration contract and the deterministic six-phase simulator with injected controllers/split solvers, partial altitude steps, resource-boundary shortening, diagnostics, optional logging and dataframe export. S-03 promoted from `UNVERIFIED` with the 120/60/30/15 s convergence series; S-04 clarified that the loiter-exit reserve is subsequently available to descent and landing; S-07 records the non-circular neutral-factor operating point. |
| 2026-08-05 | **`analysis/constraint_diagram.py` build.** Added vectorized sustained and battery-boosted cruise/climb/ceiling power constraints, stall boundary, margin-adjusted design selection, fixed-MTOW fuel contours, and the report figure. CD-01 records the landing-derived 37.5 m/s maximum stall speed; CD-02 the 0.5 m/s service-ceiling rate; CD-03 the deliberate omission of an undefined take-off-distance constraint; and CD-04 the 10% sizing margin and DC-bus boost convention. CD-05 corrects the specification's “roughly flat” altitude claim with the measured 0–10 km cruise-rating series. Matplotlib was advanced from 3.10.0 to 3.10.7 because 3.10.0 recursively fails while drawing even a basic plot on the repository's Python 3.14 runtime. |
