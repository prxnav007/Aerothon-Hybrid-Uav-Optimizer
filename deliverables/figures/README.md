# Final figure set

This folder contains the five existing visual subjects that directly support the hackathon story.
No missing architecture, GA-convergence, optimized-aircraft-comparison or mission-profile figure was
fabricated during curation. The controller plots are frozen-aircraft controller-selection evidence;
they are not plots of the later GA-optimized aircraft.

| Figure | Meaning | Source | Use | Important assumption |
|---|---|---|---|---|
| `constraint_diagram.png` | Shows the aircraft sizing feasibility region and selected reference design point. | `src.analysis.constraint_diagram` | Main presentation or report | Constraint-screening view; the shown point predates the final GA result and the 10 km ceiling remains advisory. |
| `controller_endurance_comparison.png` | Compares full-mission endurance for the selected fixed ECMS, adaptive PI-ECMS and thermostat controllers. | `src.analysis.controller_comparison` | Main presentation | Frozen 1000 kg aircraft, 3 km mission and idealized zero restart fuel. |
| `controller_endurance_restart_tradeoff.png` | Shows the endurance-versus-engine-start trade-off used to justify the thermostat for co-design. | `src.analysis.controller_comparison` | Main presentation | Frozen 1000 kg aircraft, 3 km mission and idealized zero restart fuel. |
| `controller_restart_sensitivity.png` | Shows how assumed restart fuel changes endurance and feasibility across controller families. | `src.analysis.controller_comparison` | Backup slide or report | Restart costs of 0, 0.1 and 0.5 kg/start are uncalibrated sensitivity assumptions. |
| `controller_terminal_resources.png` | Compares terminal SoC and post-landing fuel slack for the zero-cost controller comparison. | `src.analysis.controller_comparison` | Backup slide or report | Frozen 1000 kg aircraft, 3 km mission and idealized zero restart fuel. |

Editable SVGs, PDF duplicates and final-plot CSVs are under
`deliverables/figure_sources/controller_comparison/`. Development, DP, battery, cycle and superseded
thermostat artifacts are preserved under `deliverables/archive/intermediate_figures/`.

Several nonvisual controller and thermostat CSV/JSON checkpoints remain here because existing
controller resume stages and the reference fitness regression read those exact paths. They are
workflow evidence, not additional visual subjects.
