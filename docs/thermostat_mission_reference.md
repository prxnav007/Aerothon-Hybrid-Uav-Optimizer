# Full-mission thermostat integration reference

This experiment runs the existing causal thermostat through the same six-phase `run_mission`
plant used by fixed and PI-ECMS. It is an integration reference only: its thresholds were not
calibrated or optimised, and the reference aircraft is not presented as a GA result.

## Named configuration

| Item | Value |
|---|---:|
| SoC lower / upper threshold | 0.4 / 0.6 |
| Minimum ON / OFF dwell | 60 s / 60 s |
| Restart fuel | 0 kg/start |
| Engine-ON power | Existing feasible ON-power selection |
| Terminal strategy | Causal |
| Mission timestep | 60 s |
| Initial thermostat state | Engine ON, 60 s elapsed, zero restarts |

The initial ON state makes the engine explicitly available for the initial mission demand. Its
dwell is already satisfied, and initial availability is not counted as an OFF-to-ON restart.
No future demand, phase duration, time-to-go or future resource use is passed to this causal path.

## Ownership and behavior

The thermostat emits requested ON/OFF state, requested shaft power, regime, transition information
and the next immutable scheduler state. `run_mission` remains the authority that evaluates the real
turboshaft and series powertrain, steps the battery, updates fuel and mass, and enforces feasibility.
The simulator charges the engine-model restart amount once per actual OFF-to-ON transition; it
rejects a thermostat configuration whose restart amount differs from the engine model.

Hard dwell is never bypassed for an early safety restart. If an OFF dwell prevents a required start
and the battery cannot meet demand, the mission terminates with `hard_dwell_infeasible`,
`controller_infeasible` and `power_shortfall` flags.

## Reference aircraft and frozen comparator

The explicit aircraft has 1000 kg MTOW, 7.591755 m² wing area, 86.779137 kW sea-level engine,
10 kWh battery, 711.389802 kg dry mass and 288.610198 kg initial fuel. It flies the default 3 km
mission profile.

The unchanged frozen PI-ECMS comparator is:

| Metric | Frozen PI-ECMS | Thermostat reference |
|---|---:|---:|
| Total time | 56,182.854870 s | 55,614.325399 s |
| Loiter time | 49,842.854870 s | 49,274.325399 s |
| Running fuel | 283.167276 kg | 282.497732 kg |
| Restart fuel | 0 kg | 0 kg |
| Fuel remaining | 5.442923 kg | 6.112466 kg |
| Final SoC | 0.102409 | 0.518779 |
| Minimum SoC | 0.050005 | 0.353230 |
| Restarts | 180 | 33 |
| Termination | `fuel_reserve` | `fuel_reserve` |

The thermostat completed all six phases with zero reserve shortfall, zero hard-dwell violations and
no controller or plant infeasibility. Its logged `engine_power_limited` and `battery_rate_limited`
flags identify real boundary encounters during battery-assisted climb and charge-limited cycling;
they did not terminate the mission. The maximum stepwise bus-balance residual was
2.03×10⁻¹³ kW and the reconstructed fuel residual was −2.27×10⁻¹³ kg.

Run `python -m src.analysis.thermostat_mission` to execute the one named experiment and write:

- `deliverables/figures/thermostat_mission_reference.json` — complete machine-readable report;
- `deliverables/figures/thermostat_mission_phase_regime.csv` — compact phase/regime summary.

The report is descriptive. A single untuned thermostat run does not establish superiority or
optimality relative to PI-ECMS.
