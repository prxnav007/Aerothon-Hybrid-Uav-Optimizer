# Hybrid-Electric UAV Endurance Optimizer

A physics-based plant–controller co-design framework for a 1000 kg series-hybrid fixed-wing UAV.

## Headline result

The best feasible design found by one genetic-algorithm (GA) seed increased simulated loiter from
48,536.880 s to 54,927.874 s on the 3 km design mission: a **6,390.994 s (106.517 min,
13.17%) improvement**. The thermostat-based energy-management system completed all six mission
phases with a 60 s simulation step, 60 s minimum ON/OFF dwell, and an assumed restart cost of
0.1 kg of fuel per start.

This is a best-found result for the stated conceptual simulation—not a global-optimality, full
3–10 km envelope, or flight-validation claim. A separate 15 s replay retained a 13.13% loiter
improvement; [the detailed results](docs/final_results.md#15-second-validation) keep that validation
separate from the 60 s optimization result.

## Why this is a coupled problem

Maximum take-off weight (MTOW) is fixed at 1000 kg, so every kilogram of dry hardware displaces a
kilogram of fuel. A larger battery adds energy and power capability but also mass. A larger engine
adds capability but reduces fuel capacity. Wing area and aspect ratio change both drag and
structural mass. Finally, controller decisions determine when fuel is converted to electricity and
when battery energy is charged or discharged.

The simulator tracks battery state of charge (SoC), fuel, mass, altitude, power flow, limits, and
phase completion over time. The controller study includes fixed and adaptive equivalent
consumption minimization strategy (ECMS) controllers, which price electrical energy as an
equivalent fuel cost, plus a separate rule-based thermostat scheduler.

## Series-hybrid architecture

```text
Fuel → turboshaft → generator/rectifier → DC bus → inverter/motor → propeller
                                             ↕
                                          battery
```

Only the electric motor drives the propeller. The turboshaft drives a generator, while the battery
connects bidirectionally at the DC bus and acts primarily as an energy and power buffer.

## Approach

```text
Mission constraints
→ physics-based sizing
→ time-marching mission simulation
→ controller comparison
→ practical thermostat selection
→ plant–controller GA
→ 15-second replay of the best-found design
```

The six co-design variables are wing area, aspect ratio, engine rating, battery capacity, thermostat
lower SoC threshold, and thermostat SoC gap. Fuel is not a gene: it is the residual that closes the
fixed-MTOW mass budget.

## Key results

All values below are from the 60 s, 3 km, 0.1 kg/start GA scenario.

| Quantity | Practical reference | Best feasible design found |
|---|---:|---:|
| Wing area | 7.5918 m² | 9.0227 m² |
| Aspect ratio | 16.0000 | 22.6445 |
| Engine rating | 86.7791 kW | 83.4021 kW |
| Battery capacity | 10.0000 kWh | 8.3431 kWh |
| Thermostat SoC band | 0.225–0.350 | 0.208–0.627 |
| Initial fuel | 288.6102 kg | 264.3921 kg |
| Loiter duration | 48,536.880 s (13.482 h) | 54,927.874 s (15.258 h) |
| Total mission time | 54,876.880 s (15.244 h) | 61,267.874 s (17.019 h) |
| Engine restarts | 50 | 25 |
| Final / minimum SoC | 0.364 / 0.178 | 0.191 / 0.165 |

The best-found aircraft completed take-off, climb, cruise, loiter, descent, and landing, then ended
normally at the fuel-reserve boundary with 2.111 kg of reserve slack. See
[the authoritative results document](docs/final_results.md) for exact provenance, feasibility
margins, GA accounting, and the 15 s validation.

## Controller decision

On the frozen reference aircraft with idealized zero restart fuel, adaptive PI-ECMS achieved the
greatest endurance. Fixed-(s) ECMS was only 22.606 s behind, while the optimized thermostat was
342.191 s (5.703 min) behind but reduced starts from 182 to 81. With positive restart-fuel
assumptions, fewer starts became materially valuable; a retuned thermostat was therefore selected
for plant–controller co-design.

The thermostat is a rule-based energy-management system, not ECMS. Its practical selection does
not imply mathematical superiority over every ECMS formulation.

![Controller endurance and restart trade-off](deliverables/figures/controller_endurance_restart_tradeoff.png)

*Frozen reference aircraft, 3 km mission, 60 s step, and zero restart fuel. This figure supports
controller selection; it is not the GA aircraft comparison.*

## Repository structure

```text
src/
  models/          atmosphere, aerodynamics, mass, engine, battery, powertrain
  control/         fixed/PI/fuzzy ECMS, power split, thermostat scheduling
  simulation/      six-phase mission definition and time-marching simulator
  optimization/    chromosome, static feasibility, fitness, GA, production runner
  analysis/        sizing, controller studies, cross-checks, exploratory DP tools
  dashboard/       Streamlit mission replay and 15-second validation path
tests/              unit, consistency, regression, integration, and GA tests
docs/               assumptions, conventions, studies, dashboard, final results
deliverables/
  figures/          curated judge-facing figures and retained workflow checkpoints
  figure_sources/   editable controller-figure sources and plotting data
  optimization/     completed one-seed GA artifacts
  validation/       compact 15-second mission validation
  archive/          preserved intermediate and exploratory artifacts
requirements.txt    pinned runtime, dashboard, plotting, and test dependencies
```

## Installation

Run from the repository root. The project is source-based rather than an installable package.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment using your platform’s normal command, then keep the repository root as
the working directory. The pinned requirements include NumPy, pandas, Matplotlib, Pillow, Plotly,
Streamlit, and pytest.

## Running the project

The completed GA and validation artifacts are present; viewing results does not require rerunning
the expensive search.

```bash
# Tests
python -m pytest tests -q -p no:cacheprovider

# Frozen PI reference regression
python -m pytest tests/test_baseline_regression.py -q -s -p no:cacheprovider

# Named zero-restart thermostat mission
python -m src.analysis.thermostat_mission

# One controller-comparison stage; use a scratch output directory
python -m src.analysis.controller_comparison zero --output-dir build/controller_comparison

# Streamlit dashboard
python -m streamlit run src/dashboard/app.py
```

The practical 0.1 kg/start reference gate is intentionally opt-in because it runs a real mission:

```bash
AEROTHON_RUN_REFERENCE_FITNESS_GATE=1 python -m pytest tests/test_fitness.py -q -k authoritative_practical_reference
```

In PowerShell, set the variable first with
`$env:AEROTHON_RUN_REFERENCE_FITNESS_GATE = "1"`, then run the pytest command.

The production runner resumes from its checkpoint and evaluation ledger when both are present:

```bash
python -u -m src.optimization.ga_runner
```

Do not rerun it merely to inspect the result; use the linked artifacts below. The exact 15 s
validation command is documented in [the dashboard guide](docs/dashboard.md).

## Results and artifacts

- [Authoritative final results](docs/final_results.md)
- [Curated final figure set](deliverables/figures/README.md)
- [Best GA design](deliverables/optimization/ga_production_seed_20260808/best_found.json)
- [GA run summary](deliverables/optimization/ga_production_seed_20260808/ga_run_summary.md)
- [Generation history](deliverables/optimization/ga_production_seed_20260808/generation_history.csv)
- [15-second validation](deliverables/validation/mission_15s_validation.json)
- [Dashboard guide](docs/dashboard.md)
- [Assumptions and evidence status](docs/assumptions.md)
- [Power and energy conventions](docs/conventions.md)

## Assumptions and limitations

The result is conditional on one GA seed, a 3 km primary mission, fixed Oswald efficiency
`e = 0.78`, a provisional wetted-area calibration, assumed restart fuel and dwell, and documented
engine/battery placeholders. The high aspect ratio has not received detailed structural or
manufacturing optimization. The 10 km service-ceiling check is advisory in this scenario and is
not satisfied by the selected design. See [Limitations](docs/final_results.md#11-limitations) for the
complete list.

## Verification status

The release-cleanup verification completed the full suite on 2026-08-08 with **1,149 passed and
1 skipped** using pytest 9.1.1. Bytecode and pytest cache writing were disabled for that run. No
mission, GA, controller sweep, or numerical artifact was regenerated.
