"""Run the two authorized 15-second validation missions and write one summary."""

from __future__ import annotations

from src.dashboard.data import (
    load_dashboard_scenarios,
    run_validation_scenario,
    validation_summary,
    write_validation_summary,
)


def main() -> int:
    scenarios = load_dashboard_scenarios()
    bundles = []
    for key in ("practical_reference", "ga_selected"):
        print(f"validation_start scenario={key} timestep_s=15", flush=True)
        bundle = run_validation_scenario(scenarios[key])
        bundles.append(bundle)
        summary = validation_summary(bundle)
        print(
            " ".join(
                (
                    f"validation_complete scenario={key}",
                    f"feasible={summary['dynamically_feasible']}",
                    f"total_s={summary['total_mission_seconds']:.12g}",
                    f"loiter_s={summary['loiter_seconds']:.12g}",
                    f"restarts={summary['restart_count']}",
                )
            ),
            flush=True,
        )
    path = write_validation_summary(bundles)
    print(f"validation_summary={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
