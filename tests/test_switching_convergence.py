"""Switching-frequency regression for endpoint-energy PI policies."""

from __future__ import annotations

from dataclasses import replace

from src.analysis.mode_decomposition import select_post_crossing_window
from src.analysis.replay_comparison import replay_pi_ecms, resample_replay_steps
from tests.test_baseline_regression import _reproduce_baseline


def test_endpoint_policy_restart_frequency_grows_with_sample_rate() -> None:
    _, _, result, _, aircraft, controller = _reproduce_baseline()
    assert result.log is not None
    window = select_post_crossing_window(
        result.log, aircraft.battery.max_discharge_kw
    )
    lower_bracket_ratios = {
        60.0: 1.5657709765434262,
        30.0: 1.567304408550262,
        15.0: 1.5613532471656804,
    }
    restart_counts = []
    duration_h = sum(step.dt_s for step in window.steps) / 3600.0
    for dt_s, ratio in lower_bracket_ratios.items():
        replay = replay_pi_ecms(
            resample_replay_steps(window.steps, dt_s),
            aircraft,
            replace(controller, s0_ratio=ratio),
            initial_soc=window.initial_soc,
            initial_engine_shut_down=window.crossing_step.engine_shut_down,
        )
        restart_counts.append(replay.restart_count)
    assert restart_counts == [175, 349, 695]
    rates = [count / duration_h for count in restart_counts]
    assert rates[0] < rates[1] < rates[2]
    assert rates[2] > 3.9 * rates[0]
