"""Streamlit application import and optional-chart behavior tests."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import pandas as pd
import pytest

import src.dashboard.app as dashboard_app
import src.dashboard.data as dashboard_data
from src.dashboard.data import DashboardDataError


def test_dashboard_module_import_does_not_launch_a_mission(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard_data,
        "run_validation_scenario",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mission ran")),
    )
    reloaded = importlib.reload(dashboard_app)
    assert callable(reloaded.main)


def test_missing_plotly_is_an_actionable_error_without_fabricated_chart(monkeypatch) -> None:
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("plotly unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(DashboardDataError, match="require Plotly"):
        dashboard_app._plotly()


def test_power_and_operating_figures_use_supported_series_hybrid_telemetry() -> None:
    frame = pd.DataFrame(
        {
            "mission_time_h": (0.0, 15.0 / 3600.0),
            "time_start_s": (0.0, 15.0),
            "time_s": (15.0, 30.0),
            "phase": ("takeoff", "climb"),
            "motor_electrical_input_kw": (20.0, 22.0),
            "rectifier_bus_output_kw": (16.0, 18.0),
            "battery_bus_kw": (4.0, 4.0),
            "bus_demand_kw": (20.0, 22.0),
            "engine_on": (True, True),
            "restart_event": (True, False),
            "requested_engine_shaft_kw": (18.0, 20.0),
            "engine_shaft_kw": (18.0, 20.0),
            "engine_available_kw": (80.0, 79.0),
            "generator_electrical_output_kw": (17.0, 19.0),
            "motor_shaft_output_kw": (18.0, 20.0),
            "battery_current_a": (11.0, 12.0),
            "battery_terminal_voltage_v": (380.0, 379.0),
            "engine_thermal_efficiency": (0.24, 0.25),
            "engine_source_efficiency": (0.90, 0.90),
            "demand_path_efficiency": (0.90, 0.91),
            "battery_terminal_efficiency": (0.98, 0.98),
        }
    )
    power = dashboard_app._power_figure(frame, frame, 15.0)
    operating = dashboard_app._operating_figure(frame, frame, 15.0)
    assert {trace.name for trace in power.data} >= {"Engine ON", "Restart"}
    assert {trace.name for trace in operating.data} >= {
        "Generator electrical output", "Battery bus power",
        "Motor electrical input", "Motor shaft output",
    }
    assert power.layout.paper_bgcolor == "#FFFFFF"
    assert operating.layout.plot_bgcolor == "#FFFFFF"


def test_missing_optional_metric_is_labelled_unavailable() -> None:
    assert dashboard_app._optional_metric(None, "A") == "Unavailable"
    assert dashboard_app._optional_metric(float("nan"), "V") == "Unavailable"


def test_mission_timeline_uses_the_installed_streamlit_slider_api() -> None:
    assert "format_func" not in inspect.signature(dashboard_app.st.slider).parameters
    source = textwrap.dedent(inspect.getsource(dashboard_app._playback_controls))
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    incompatible = [
        call for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "slider"
        and any(keyword.arg == "format_func" for keyword in call.keywords)
    ]
    assert incompatible == []
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "select_slider"
        for call in calls
    )


def test_playback_queue_advances_and_stops_at_the_last_sample(monkeypatch) -> None:
    state = {"playing_test": True, "position_test": 2}
    monkeypatch.setattr(dashboard_app, "st", SimpleNamespace(session_state=state))
    frame = pd.DataFrame(index=range(10))
    dashboard_app._queue_playback_update(frame, "test", 4)
    assert state["pending_position_test"] == 6
    assert state["playing_test"] is True
    state["position_test"] = 8
    dashboard_app._queue_playback_update(frame, "test", 4)
    assert state["pending_position_test"] == 9
    assert state["playing_test"] is False
