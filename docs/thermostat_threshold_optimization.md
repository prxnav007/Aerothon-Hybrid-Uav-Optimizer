# Frozen-aircraft thermostat threshold study

This controller-only study tunes the global `soc_low` and `soc_high` thresholds on the existing
1000 kg reference aircraft. It does not change the aircraft, battery physics, 60 s hard dwell,
zero restart-fuel assumption, causal scheduling, maximum-feasible ON-power rule, ECMS defaults or
mission. It is not plant co-design and does not use GA, DP or fuzzy ECMS.

## Search definition

The actual battery interval is `0.05 <= SoC <= 1.0`. Candidate bands require
`soc_high - soc_low >= 0.05`. The deterministic search evaluates a 56-point triangular coarse mesh,
retains four feasible regions and refines locally at 0.025 SoC resolution, with a hard cap of 72
completed missions. The mesh explicitly contains the untuned `(0.4, 0.6)` pair, the battery floor,
narrow and wide bands, and upper thresholds through 1.0.

Every completed mission is appended and flushed to
`deliverables/figures/thermostat_threshold_search.csv`. Re-running the study reads this checkpoint
and skips existing decimal threshold keys. Each evaluation rebuilds the frozen aircraft, mission,
controller parameters and explicit initial thermostat state, preventing state leakage.

Endurance is the sole primary objective. There is no terminal-SoC reward or depletion penalty.
Candidates must complete all six phases and close the existing fuel, power and discrete-energy
ledgers. Exact event-time ties are disclosed; restart count, transition-duration margin and
remaining resources are secondary diagnostics only.

## Phase-dependent gate

Global thresholds are evaluated first. A separate loiter/non-loiter band is permitted only if a
mandatory non-loiter phase demonstrably conflicts with the best feasible global band. High final
SoC or a conservative fuel-allocation result alone does not establish such a controller conflict.
No six-phase threshold table is permitted.

The direct slack check reproduces the winning loiter through its exact fractional final step, adds
one distinct 60 s loiter interval, then verifies descent, landing and the post-landing reserve.

## Artifacts

- `thermostat_threshold_phase_ledger.csv`: frozen PI and untuned thermostat phase endpoints;
- `thermostat_threshold_search.csv`: resumable per-candidate checkpoint;
- `thermostat_threshold_best.json`: bounds, winner, comparison, extension check and phase gate.

Results are reported as the best feasible thermostat found within the stated bounds and search
resolution. They do not establish superiority under realistic restart physics because restart fuel
remains zero.

## Completed result

The 72-evaluation search found `(soc_low, soc_high) = (0.225, 0.300)` as the unique best feasible
pair at the evaluated resolution. Neither threshold is pinned to the battery bound, and the 0.075
band is wider than the 0.05 minimum separation.

| Metric | Frozen PI-ECMS | Untuned thermostat | Best global thermostat |
|---|---:|---:|---:|
| Total mission time | 56,182.854870 s | 55,614.325399 s | 56,094.325399 s |
| Loiter time | 49,842.854870 s | 49,274.325399 s | 49,754.325399 s |
| Final SoC | 0.102409 | 0.518779 | 0.220358 |
| Minimum SoC | 0.050005 | 0.353230 | 0.179662 |
| Fuel remaining | 5.442923 kg | 6.112466 kg | 5.507815 kg |
| Restarts | 180 | 33 | 81 |
| Termination | `fuel_reserve` | `fuel_reserve` | `fuel_reserve` |

The best pair recovers 480.000 s (8.000 min, 84.43%) of the untuned controller's 568.529 s
deficit. It remains 88.529 s (1.475 min) below frozen PI-ECMS while retaining 99 fewer restarts.

Both comparison controllers leave loiter at exactly 9.7 kg. The untuned thermostat enters loiter at
SoC 0.669501 and leaves at 0.574930, using only 0.978729 kWh while burning fuel at 17.053191 kg/h.
PI-ECMS enters at SoC 0.593188 and leaves at 0.063675, using 5.035568 kWh while averaging
16.879594 kg/h. The thermostat therefore reaches the same fuel threshold earlier because it burns
fuel faster during loiter while preserving substantially more battery energy.

The winning mission is deterministic on an exact repeat. It ends with 1.958150 kWh stored, of
which 1.526007 kWh lies above the battery floor, and 0.507815 kg above the post-landing reserve.
Forcing one additional 60 s loiter step consumes 0.343398 kg and still completes descent and
landing, but finishes at 4.899968 kg: a 0.100032 kg reserve shortfall. The additional interval is
therefore not feasible under the stated reserve constraint.

No infeasible candidate achieved more loiter time than the selected global pair, and the winner
completed climb, descent and landing without controller, plant or dwell infeasibility. There is no
demonstrated loiter-versus-non-loiter threshold conflict, so phase-dependent bands are not justified
and were not implemented.
