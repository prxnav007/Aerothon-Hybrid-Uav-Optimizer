# Milestones 2/3 numerical-validity recovery

**Status:** recovery gate passed for the tiny oracle and the discretised conditional-loiter
problem. Milestones 2 and 3 remain incomplete. No full production sweep is validated by this
note.

## What was invalid

For a policy π, fuel F(π), terminal stored energy E(π), and target E*, two policies selected at
different terminal shadow prices can end on opposite sides of E*. Their fuel values do not, by
that fact alone, bound the exact-target optimum. The same applies to two thermostat threshold
policies surrounding E*. Consequently:

- the old finite-horizon `fuel_interval_kg` and thermostat fuel-interval interpretation are
  withdrawn;
- the previous 0.27–0.34% thermostat-to-DP point-gap interpretation is invalid;
- the reported 99.50–99.61% captured-benefit interpretation is provisional and is not emitted for
  an unequal-energy PI pair;
- the old 60/30/15 s DP CSVs and the coarse restart/dwell sensitivity remain historical,
  exploratory outputs.

The two residuals are now independent. `ledger_residual_kwh` checks a single trajectory's energy
accounting. `terminal_target_residual_kwh` checks whether that trajectory reaches the requested
comparison target. Roundoff closure of the first says nothing about the second.

## Exact oracle and valid bounds

`tiny_dp_oracle.py` enumerates all 208 admissible policies in a six-step problem with battery
energy, engine state, hard ON/OFF dwell, OFF plus two ON actions, restart fuel, and an attainable
terminal target. The exact target optimum consumes 0.470 kg. The adjacent shadow-supported
policies end at 0.1 and 0.5 kWh and consume 0.380 and 0.440 kg: both raw fuels lie below the exact
target optimum, which is the regression counterexample to the deleted fuel-bracket claim.

The valid Lagrangian quantity is

`g(λ) = minπ [F(π) + λ(E(π) − E*)]`.

Maximising it on the oracle gives a 0.425 kg lower bound. Exhaustive enumeration supplies the
0.470 kg deterministic feasible upper bound, leaving a 0.045 kg gap.

The production finite-horizon solver now defines a finite discretised problem with one stored
successor index for every feasible grid transition. Both backward induction and discrete replay
use that same successor. A terminal-state-constrained induction constructs a deterministic policy
at the exact inserted target grid state when reachable. The shadow-price solve reports the maximum
dual value separately. Those bounds apply only to the discretised SoC/action model. Fixed-action
continuous replay is reported alongside it, including its target residual and feasibility, but is
not covered by the discrete bounds.

## Dwell and physical interpretation

Minimum ON/OFF dwell is hard in both thermostat and DP comparisons. An early safety restart is not
allowed. Finite-horizon DP and the explicitly horizon-aware thermostat may inspect future replay
demand. The causal thermostat never does: it conservatively assumes the current demand persists
for the complete prospective OFF dwell. This current-information check cannot guarantee
survivability against an unknown future demand increase, which remains an explicit limitation of
causal hard-dwell control. If the engine must remain ON near the upper SoC boundary,
load-following or the battery's exact charge limit prevents overcharge.

The selected thermostat ON power is the maximum feasible engine power when the active battery
charge constraint bounds a monotonically improving cycle calculation. It is not evidence that
higher engine power is intrinsically suboptimal. Likewise, `soc_high = 0.99` in the historical
study is a bound-pinned numerical result under the legacy battery model. The model has no validated
CC–CV taper, high-SoC charge-acceptance curve, thermal limit, or pack terminal-voltage limit, so
0.99 is not a validated physical threshold.

All replay results here are conditional loiter results over an exogenous demand/mass trajectory,
not full-mission endurance results. Restart/dwell results from the 31-SoC/5-action sensitivity grid
are directional only and are not quantitatively comparable with finer-grid headline outputs.

## Milestone status

- **Milestone 2:** provisional. Exact-point comparisons remain usable; unequal-energy PI and
  thermostat policy pairs provide endpoint and raw-policy data only. The captured-benefit claim is
  withdrawn pending valid equal-energy bounds.
- **Milestone 3A:** the matched idealised subproblem now has valid logic: the analytical relaxed
  optimum is a lower bound and an integer-step schedule with continuously repaired ON power is an
  exactly periodic feasible upper bound. The old 120 s result is excluded from its former
  close-agreement claim; its −0.528 kWh mismatch was not an equal-energy verification.
- **Milestone 3B:** exploratory. Non-monotonic timestep fuel, SoC-grid phase changes, and wide
  endpoint-policy intervals prevent a converged continuous exact-energy benchmark claim.

## Historical artifact classification

| Artifact family | Classification |
|---|---|
| `deliverables/archive/intermediate_figures/periodic_dp_convergence.csv` | Historical grid replay; terminal mismatch must be read explicitly |
| `deliverables/archive/intermediate_figures/finite_horizon_dp_15/30/60.csv` | Exploratory unequal-energy supported policies |
| `deliverables/archive/intermediate_figures/finite_horizon_dp_grid*.csv` | Exploratory grid-phase evidence |
| `deliverables/archive/intermediate_figures/finite_horizon_dp_sensitivity.csv` | Directional only; coarse 31-SoC/5-action grid |
| `deliverables/archive/intermediate_figures/thermostat_equal_energy.csv` | Misnamed historical unequal-energy policy data; no fuel bounds |
| `deliverables/archive/intermediate_figures/thermostat_sensitivity.csv` | Conditional-loiter directional sensitivity only |

New CSV writers use `policy_role`, `endpoint_energy_interval_width_kwh`,
`policy_fuel_values_kg`, `ledger_residual_kwh`, `terminal_target_residual_kwh`, policy hashes,
separate kernel/solve time, and explicit bound scope. Scenario runners append each completed
scenario immediately and skip completed scenario labels on resume.

## Bounded 60 s demonstration

The one permitted production-code demonstration used a 12-step synthetic conditional-loiter
demand trace at 60 s, a requested 21-point SoC grid expanded to 23 points by initial/target state
insertion, and five engine actions. Its terminal energy-change target was 0.0 kWh. For the
discretised nearest-grid problem, the dual lower bound was 2.4538227147 kg and the deterministic
exact-grid-target upper bound was 2.6938227147 kg, a 0.2400000000 kg gap.

Continuous replay of the discrete upper policy consumed 2.6938227147 kg, incurred no engine,
battery, SoC, dwell, or restart-accounting violations, and had policy hash
`93f78535743c5f6b`. It nevertheless missed the continuous terminal target by -0.9629447202 kWh;
its ledger residual was 0.0090809416 kWh. It is therefore not a continuous exact-target feasible
upper bound, and no continuous optimality or thermostat-to-DP gap is inferred. Kernel construction
took 0.0848901220 s and policy solution took 0.0116115050 s across six backward inductions,
terminating because a shadow-supported policy reached the inserted discrete target exactly.
