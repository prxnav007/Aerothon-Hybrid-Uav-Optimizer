# PPT Input Brief — Hybrid-Electric Propulsion Optimization for a Fixed-Wing UAV

**Hackathon:** Aerothon — IIT Indore in collaboration with Hindustan Aeronautics Limited (HAL), Problem Statement 1 (PS1)
**Repo:** Aerothon-Hybrid-Uav-Optimizer (Python)
**Headline result: endurance improved from 12.25 h (baseline) to 19.25 h (GA-optimized) — a +57.1 % improvement, with all mission phases completed and every constraint satisfied.**

---

## 1. The Problem (from the PS)

Design and optimize a hybrid-electric propulsion system for a fixed-wing UAV:
- MTOW ~1000 kg, payload ~200 kg, cruise ~250 km/h, cruise altitude 3–10 km
- Preferred prime mover: ~60 kW-class turboshaft engine
- Mission: take-off → climb → cruise → loiter → descent → landing
- Objective: **maximize endurance** subject to MTOW, payload, mission-completion, propulsion-power, and battery-safety constraints.

## 2. Our Propulsion Architecture

**Series hybrid-electric (turboelectric + battery):**

```
Fuel tank → Turboshaft engine → Generator → Rectifier (AC→DC, η=0.95) ─┐
                                                                        ├→ DC bus → Inverter (η=0.95) → Electric motor (η=0.95) → Propeller (η=0.85)
Li-ion battery pack (250 Wh/kg, 300–400 V) ←──(charge/discharge)───────┘
```

- The turboshaft never drives the propeller mechanically — it only generates electricity. All propulsion is electric.
- The battery sits on the DC bus: it **peak-shaves** high-power phases (take-off, climb) and is **recharged in flight** by the engine during low-demand phases.
- Why series: decouples engine speed from prop speed (engine can run at efficient set-points), simple power management, enables distributed electric propulsion later.

## 3. The Core Method — Two Nested Optimization Loops

This is the key "optimization quality + innovation" story:

**Inner loop (every 60 s of simulated flight): Fuzzy-Adaptive ECMS**
- ECMS = Equivalent Consumption Minimization Strategy. At each time step we choose the engine/battery power split that minimizes total *equivalent fuel mass*:
  `cost = fuel_burned_kg + s × (battery_energy_kWh × 0.08 kg/kWh)`
- The equivalence factor **s** "prices" battery energy in fuel terms. High s → protect the battery, run the engine. Low s → spend the battery.
- **Innovation:** s is not a constant. A **fuzzy-logic controller** (scikit-fuzzy, 7 rules, Mamdani-style, centroid defuzzification) computes s continuously from two inputs: battery SoC and normalized power demand. Example rules: "SoC low → s high (save battery)", "SoC high AND demand low → s low (spend battery)", "SoC medium AND demand high → s medium (balanced)".
- The power split search discretizes generator power in 2 kW steps from 0 to engine max, subject to battery charge/discharge limits and a SoC-based discharge lockout.

**Outer loop: Genetic Algorithm (DEAP) co-optimizes plant AND controller**
- 6-gene chromosome (design variables):
  1. Turboshaft engine size (kW), bounds [30, 100]
  2. Battery capacity (kWh), bounds [5, 50]
  3. Fuzzy SoC-low threshold, [0.10, 0.40]
  4. Fuzzy SoC-high threshold, [0.60, 0.90]
  5. ECMS s-factor min, [1.0, 3.0]
  6. ECMS s-factor max, [3.0, 5.5]
- Genes 1–2 size the hardware; genes 3–6 tune the fuzzy energy-management controller. **Sizing and power-management strategy are optimized simultaneously** — a design is only as good as the controller flying it.
- Fitness = mission endurance (s) from a full time-marching mission simulation. Designs that fail to complete all phases get fitness × 0.1 (heavy penalty but with gradient so GA can climb back to feasibility). Out-of-bounds genes get fitness 0.
- GA settings: population 30, 10 generations, blend crossover (α=0.5, p=0.7), polynomial bounded mutation (η=20, p=0.2 — strictly enforces physical bounds), tournament selection (size 3), hall-of-fame elitism. 208 total design evaluations, 172 feasible.

## 4. The Mission Simulator (fitness evaluator)

Time-marching simulation, Δt = 60 s, flying the full PS mission profile:

| Phase | Duration | Altitude target | Speed |
|---|---|---|---|
| Take-off | 120 s | 300 m | 50 m/s |
| Climb | 2850 s | 6000 m (≈2 m/s climb rate) | 65 m/s |
| Cruise | 3600 s | 6000 m | 69.4 m/s (= 250 km/h, per PS) |
| **Loiter** | **open-ended = endurance phase** | 6000 m | 70 m/s |
| Descent | 900 s | 300 m | 65 m/s |
| Landing | 120 s | 0 m | 45 m/s |

- **Loiter is extended until the fuel reserve (5 kg) or battery cutoff (5 % SoC) is reached** — so total mission time IS the endurance being maximized, and descent + landing can always be completed afterward.
- Cruise altitude 6000 m chosen inside the PS's 3–10 km band — keeps cruise power within a 60 kW-class engine.
- **Fuel policy:** the tank is filled up to MTOW (fuel = 1000 kg − dry system weight). This makes the GA trade engine/battery mass *directly* against fuel mass — the central design trade-off.
- Aircraft weight decreases as fuel burns, which reduces drag/power over the mission (properly modeled).

**Physics models (each is a documented assumption):**
- International Standard Atmosphere (ISA) density model up to 11 km: ρ = ρ₀(T/T₀)^4.255
- Parabolic drag polar: C_D = C_D0 + C_L²/(π·AR·e), with C_D0 = 0.025, AR = 10, Oswald e = 0.8, wing area 5 m²
- Shaft power = (Drag·V + Weight·climb_rate)/η_prop
- Battery: coulomb-counting SoC model with internal resistance (0.05 Ω) and linear OCV (300 V at 0 % → 400 V at 100 % SoC); energy density 250 Wh/kg (Li-ion)
- Turboshaft: constant SFC 0.3 kg/kWh (typical turboshaft), power-to-weight 1.5 kW/kg
- Efficiencies: propeller 0.85, motor 0.95, rectifier 0.95, inverter 0.95
- Weight build-up: 450 kg empty airframe + 200 kg payload + engine mass + battery mass + fuel (to MTOW)
- Fuel LHV 43.1 MJ/kg (jet fuel) — used for the system-efficiency metric

**Constraint handling:**
- MTOW = 1000 kg (hard, via the fill-to-MTOW fuel policy); min 20 kg usable fuel or design is infeasible
- Power balance: engine + battery must meet electric demand at every step or the design fails ("power shortfall")
- Battery safety: hard SoC cutoff 5 % (simulation never discharges below it; discharge is locked out), recommended safe floor 20 % shown on dashboard
- 5 kg landing fuel reserve; 24 h safety cap on mission time
- Payload 200 kg always carried

## 5. Results

**Baseline (reference, non-optimized):** the PS-preferred 60 kW turboshaft + 30 kWh battery + hand-picked fuzzy parameters (SoC thresholds 0.3/0.7, s-range 3.0–5.0) → **12.25 h endurance**.

**GA-optimized design → 19.25 h endurance (+57.1 %):**

| Design variable | Baseline | Optimized |
|---|---|---|
| Engine size | 60 kW | **75.65 kW** |
| Battery capacity | 30 kWh | **5.10 kWh** |
| Fuzzy SoC low threshold | 0.30 | 0.319 |
| Fuzzy SoC high threshold | 0.70 | 0.728 |
| ECMS s min | 3.0 | 2.26 |
| ECMS s max | 5.0 | 3.10 |

Optimized mission facts: take-off weight exactly 1000 kg (MTOW), dry weight 720.8 kg, fuel loaded 279.2 kg, fuel consumed 274.4 kg (lands with reserve), final SoC 33.6 % (well above both floors), mission complete, loiter lasted ≈17.1 h of the 19.25 h total.

**GA convergence:** best design 18.45 h at generation 0 → 19.05 h (gen 3) → 19.25 h by generation 8; population average climbs from 6.3 h to ~16.6 h. Clean convergence plot available.

**The key engineering insight (great trade-off slide):**
The GA *shrank* the battery from 30 kWh (120 kg) to 5.1 kWh (20 kg) and *grew* the engine from 60 to 75.7 kW (+10 kg). Net ≈ 89 kg of powertrain mass converted into extra fuel. Why: jet fuel stores ~43 MJ/kg vs ~0.9 MJ/kg for Li-ion — roughly **48× the energy per kilogram**. So for an endurance mission the battery's optimal role is NOT energy storage; it is a **transient power buffer** — peak-shaving take-off/climb (where demand hits ~76 kW+, briefly exceeding what a small engine alone could give) and absorbing recharge during low-demand phases. The bigger engine both covers cruise/loiter demand at reasonable load and recharges the small battery quickly. The GA discovered this fuel-vs-battery trade-off autonomously; the design-space scatter plot (engine kW vs battery kWh, colored by endurance, infeasible designs marked ×) shows the feasibility boundary and the optimum star.

**Endurance improvement criterion (10 % of score): +7.00 h absolute, +57.1 % relative, vs a like-for-like baseline flying the identical mission with the identical simulator — only sizing and controller tuning differ.**

## 6. Simulation Dashboard (Streamlit + Plotly)

Run with `streamlit run src/dashboard/app.py`. Sections map 1:1 to deliverable 8.3:
1. **Mission profile & flight phases** — altitude and airspeed traces with color-coded phase bands (takeoff/climb/cruise/loiter/descent/landing)
2. **Power distribution thermal vs electric** — stacked area of engine/generator vs battery discharge, battery charging shown below zero, dashed total-demand line
3. **Battery SoC** — with 20 % safe-floor and 5 % hard-cutoff lines
4. **Fuel consumption** — remaining + consumed, landing-reserve line
5. **Engine & motor operating conditions** — engine load % of rated (with 100 % line) and motor electric power
6. **System efficiency** — instantaneous trace + mean per flight phase; defined as propulsive power out ÷ (fuel chemical power + battery power in)
7. **Endurance estimation** — KPI header: endurance, take-off weight vs MTOW, fuel consumed, min SoC, avg system efficiency, peak power demand; feasibility banner
8. **Optimization results & trade-offs** — GA convergence curve, design-space scatter (feasible/infeasible/optimum), baseline-vs-optimized endurance bar, optimal-genes table
- Bonus: dedicated **power-management strategy chart** showing the fuzzy equivalence factor s evolving in flight (rises as SoC falls — the controller visibly re-pricing battery energy)
- **Interactive:** sidebar scenario selector — Optimized (GA result) / Baseline / **Custom design with live sliders for all 6 genes that re-flies the whole mission in real time** (cached simulation). Full data table + CSV download included.

## 7. Tech Stack & Repo Structure

- Python; libraries: **DEAP** (genetic algorithm), **scikit-fuzzy** (fuzzy ECMS controller), NumPy/SciPy/pandas, **Streamlit + Plotly** (dashboard)
- `src/config.py` — UAV specs, mission profile, mission/energy settings, ISA constants (all assumptions centralized and documented)
- `src/physics.py` — ISA atmosphere, drag polar, shaft-power-required
- `src/components.py` — Battery model, TurboshaftEngine model, system weight build-up
- `src/optimization.py` — FuzzyECMS controller (robust: input validation, gene sorting/clamping, defuzzification fallback so the GA can never crash it)
- `src/ga_optimization.py` — ECMS power-split optimizer, mission simulator, GA driver; saves baseline results, best-design results, convergence history, all 208 evaluations, and a summary JSON to `data/`
- `src/dashboard/app.py` — the 8-section Streamlit dashboard
- Reproducible pipeline: `python -m src.ga_optimization` → then `streamlit run src/dashboard/app.py`

## 8. Mapping to Evaluation Criteria

- **Mission feasibility (20 %):** full 6-phase time-marching simulation; explicit feasibility flags; fuel reserve + SoC cutoff guarantee descent/landing completion; MTOW and payload hard-enforced.
- **Optimization quality (25 %):** nested GA × ECMS co-optimization of hardware and controller; bounded polynomial mutation keeps designs physical; penalty-with-gradient constraint handling; convergence and full design-space evidence saved and visualized.
- **Engineering justification (20 %):** every assumption documented with source rationale (ISA, drag polar, SFC 0.3 kg/kWh, 250 Wh/kg, efficiency chain); fill-to-MTOW policy makes the mass trade explicit; battery-as-power-buffer conclusion follows from energy-density physics.
- **Innovation (15 %):** fuzzy-adaptive equivalence factor (adaptive ECMS) instead of fixed rule-based splitting, AND letting the GA tune the fuzzy controller itself — the optimizer designs both the aircraft and its energy-management brain.
- **Endurance improvement (10 %):** +57.1 % (12.25 h → 19.25 h) on a like-for-like mission.
- **Presentation & visualization (10 %):** 8-section interactive dashboard with live what-if sliders, consistent color-per-entity design system, phase-band context on every chart.

## 9. Suggested Slide Flow (~12 slides)

1. Title + team + headline number (19.25 h, +57 %)
2. Problem statement recap & requirements
3. Proposed architecture: series hybrid diagram (engine→generator→DC bus→motor→prop, battery on bus)
4. Methodology overview: nested loops diagram (GA outer / mission sim / fuzzy ECMS inner)
5. Modeling & assumptions (physics, battery, engine, weights, efficiencies)
6. Mission profile & simulation logic (loiter-as-endurance, fill-to-MTOW)
7. Power management: fuzzy adaptive ECMS (rules table + s-factor-in-flight chart)
8. Optimization setup: 6 genes, bounds, GA operators, constraint handling
9. Results: convergence curve + design-space trade-off scatter
10. Optimized vs baseline table + endurance bar (+57.1 %)
11. Key insight: fuel vs battery energy density → battery as power buffer, not energy store
12. Dashboard demo screenshots + limitations & future work (variable SFC maps, battery thermal/degradation, multi-objective Pareto, higher-resolution time step, altitude/speed as design variables)

## 10. Honest Limitations (good for Q&A slide)

- Constant SFC (real turboshafts have part-load SFC penalty — would further favor the battery for load-leveling)
- Simplified battery (no thermal, aging, or C-rate detail beyond a power cap)
- 60 s time step; point-mass performance model; no CFD/structures (explicitly out of scope per PS)
- Single-objective GA (endurance); multi-objective (endurance vs battery margin vs engine life) is a natural extension
