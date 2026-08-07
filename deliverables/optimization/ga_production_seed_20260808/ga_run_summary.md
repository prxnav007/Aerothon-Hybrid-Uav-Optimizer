# Production plant-thermostat GA — seed 20260808

This records the **best feasible design found by this one-seed GA run**; it is not a claim of global optimality.

## Search outcome

- Generation-zero gate: **passed**.
- Termination: `max_generations` after 40 completed evaluated populations.
- Runtime: 2069.321 s.
- Placements / unique evaluations / missions: 2482 / 2384 / 1809.
- Static-infeasible / dynamic-infeasible / feasible: 575 / 198 / 1611.
- Cache hits: 98.
- Stagnation stop: false.

## Best design

- Wing: 9.02273742033 m², AR 22.6444813909, span 14.2938871413 m, C_D0 0.0253692668152.
- Engine / battery: 83.4020599829 kW / 8.34314114486 kWh.
- Thermostat: 0.208416283678–0.627422294769 (gap 0.41900601109).
- Dry mass / initial fuel: 735.607870261 kg / 264.392129739 kg.

## Mission

- Loiter / total: 54927.8740646 s / 61267.8740646 s.
- Gain over practical reference: 6390.99393441 s (106.516565573 min, 13.1672945%).
- Final fuel / reserve slack: 7.11084404579 kg / 2.11084404579 kg.
- Final / minimum SoC: 0.190881696274 / 0.164957471107.
- Terminal usable battery energy: 1.04792951615 kWh.
- Restarts / restart fuel: 25 / 2.5 kg.
- Overall / loiter engine-OFF fraction: 0.267024117448 / 0.287285831989.

## Bounds and recovery

- Genes within 1% of a normalized bound: `[]`.
- Checkpoint and evaluation ledger are the authoritative recovery state.
- Resume command: `python -u -m src.optimization.ga_runner`.
