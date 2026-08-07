# Frozen-aircraft full-mission controller comparison

This study compares controller parameters frozen at their zero-restart-fuel selections. Restart costs of 0.1 and 0.5 kg/start are sensitivity assumptions, not calibrated measurements. No plant variable, controller default, battery model, DP, fuzzy controller or GA was changed.

## Specification audit

- The supplied phrase `zero-restart thermostat result` is interpreted as zero restart *fuel*; the mission contains 81 measured OFF-to-ON transitions.
- No executable historical complete-mission fixed/PI sweep or checkpoint is present. The verification therefore reconstructs a 3-point fixed neighbourhood and a 3 x 3 PI neighbourhood at the historical 0.1 ratio and 2.5 gain resolution.
- Fixed ratios 1.1 and 1.2 produce the same event-time endurance to 1e-9 s. Ratio 1.1 is retained as the historical lower-ratio representative, not called a unique optimum.
- The documented `aeroprjct/Scripts/python.exe` is absent; execution used the configured bundled Python with the repository dependency directory.
- The existing pure-thermal control group is not a feasible full-mission continuous comparator for this aircraft, so it is not included.

## Exact selected configurations

The full records are in `controller_configuration_record.json`. Fixed-(s) uses the switching-ratio parameter with adaptation disabled. PI uses a switching-relative base, proportional SoC feedback, no integral state and the shared [0.5, 20] clamp. Thermostat uses causal 60/60 s hard dwell and the existing maximum-feasible ON-power rule.

## Zero restart-fuel result

| Controller | Endurance (h) | Delta (min) | Final SoC | Starts | Fuel above reserve (kg) |
|---|---:|---:|---:|---:|---:|
| Tuned adaptive PI-ECMS | 15.6768 | +0.000 | 0.0767 | 182 | 0.5609 |
| Tuned fixed-(s) ECMS | 15.6705 | -0.377 | 0.0960 | 179 | 0.4464 |
| Optimised thermostat | 15.5818 | -5.703 | 0.2204 | 81 | 0.5078 |

Zero-cost Pareto frontier: fixed_s_ecms, adaptive_pi_ecms, optimised_thermostat.

## Parameter-frozen restart sensitivity

| Cost (kg/start) | Controller | Endurance (h) | Starts | Running fuel (kg) | Restart fuel (kg) | Final fuel (kg) | Final SoC | Feasible | Termination |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0.0 | Tuned adaptive PI-ECMS | 15.6768 | 182 | 283.0493 | 0.0000 | 5.5609 | 0.0767 | True | `fuel_reserve` |
| 0.0 | Tuned fixed-(s) ECMS | 15.6705 | 179 | 283.1638 | 0.0000 | 5.4464 | 0.0960 | True | `fuel_reserve` |
| 0.0 | Optimised thermostat | 15.5818 | 81 | 283.1024 | 0.0000 | 5.5078 | 0.2204 | True | `fuel_reserve` |
| 0.1 | Tuned adaptive PI-ECMS | 14.6778 | 165 | 267.1913 | 16.5000 | 4.9189 | 0.0696 | False | `fuel_reserve_shortfall` |
| 0.1 | Tuned fixed-(s) ECMS | 14.6768 | 161 | 267.2397 | 16.1000 | 5.2705 | 0.0612 | True | `fuel_reserve` |
| 0.1 | Optimised thermostat | 15.0925 | 77 | 275.5380 | 7.7000 | 5.3722 | 0.2588 | True | `fuel_reserve` |
| 0.5 | Tuned adaptive PI-ECMS | 12.0611 | 121 | 225.6554 | 60.5000 | 2.4548 | 0.0718 | False | `fuel_reserve_shortfall` |
| 0.5 | Tuned fixed-(s) ECMS | 12.1240 | 118 | 226.7735 | 59.0000 | 2.8367 | 0.0675 | False | `fuel_reserve_shortfall` |
| 0.5 | Optimised thermostat | 13.5778 | 66 | 251.2329 | 33.0000 | 4.3773 | 0.2308 | False | `fuel_reserve_shortfall` |

- 0 kg/start ranking: adaptive_pi_ecms, fixed_s_ecms, optimised_thermostat.
- 0.1 kg/start ranking: optimised_thermostat, fixed_s_ecms.
- 0.5 kg/start ranking: no feasible controller.
- Optional 0.1 kg/start retuning required: True.
- Bounded 0.1 kg/start retuning used 18 missions and no 0.5 kg/start retuning.
- Best retuned thermostat: `thermostat:low=0.225:high=0.350`, 54876.880 s with 50 starts.
- Best feasible fixed/PI neighbourhood result: 52836.620 s.

## PPT-ready statements

**Numerical statement.** Tuned adaptive PI-ECMS achieved the greatest ideal zero-restart-fuel endurance at 15.6768 h.

**Engineering recommendation.** The optimised thermostat retained 99.394% of the best ideal endurance while reducing starts by 101 relative to adaptive PI-ECMS. Its positive-restart-cost sensitivity and two-threshold chromosome support its selection for plant-controller co-optimisation.

## Verdict

Recommended controller: `optimised_thermostat`. The recommendation is practical, and is bounded by this frozen aircraft, timestep, local controller searches and uncalibrated restart-cost sensitivity. It is not a global-optimality claim.

## Reproducibility

Machine-readable tuned comparison: `C:/Aerothon-Hybrid-Uav-Optimizer/deliverables/figures/controller_comparison.csv`. Every plotted value is read back from its figure-specific CSV before rendering. Full-suite status: 983 passed in 102.22 s on 2026-08-07.
