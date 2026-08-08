"""Streamlit dashboard for 15-second mission replay and completed GA results."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.data import (
    ARCHITECTURE_LABEL,
    DashboardDataError,
    FrozenDashboardScenario,
    GAArtifacts,
    PHASE_ORDER,
    PRODUCTION_GA_DIRECTORY,
    ValidationBundle,
    ga_candidates_dataframe,
    ga_history_dataframe,
    ga_tradeoff_csv_bytes,
    load_dashboard_scenarios,
    load_ga_artifacts,
    load_validation_summary,
    run_validation_scenario,
    telemetry_csv_bytes,
    telemetry_dataframe,
    validation_summary_json_bytes,
)
from src.dashboard.theme import (
    BATTERY_COLOR,
    DASHBOARD_CSS,
    DEMAND_COLOR,
    ENGINE_COLOR,
    FUEL_COLOR,
    MOTOR_COLOR,
    PHASE_COLORS,
    SOC_COLOR,
    STATUS_COLORS,
)

APP_TITLE = "Hybrid-Electric UAV Mission Simulator"
REPLAY_LABEL = "Accelerated replay of a deterministic 15-second time-marching simulation."


def _plotly() -> tuple[Any, Any, Any]:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise DashboardDataError(
            "Interactive charts require Plotly. Install the pinned requirements.txt."
        ) from error
    return px, go, make_subplots


@st.cache_data(show_spinner=False)
def _cached_validation(
    scenario_key: str,
    evaluation_key: str,
    best_found_path: str,
) -> ValidationBundle:
    scenarios = load_dashboard_scenarios(best_found_path)
    scenario = scenarios[scenario_key]
    if scenario.evaluation_key != evaluation_key:
        raise DashboardDataError("Simulation cache identity does not match the frozen scenario")
    return run_validation_scenario(scenario)


@st.cache_data(show_spinner=False)
def _cached_ga_artifacts(directory: str) -> GAArtifacts:
    return load_ga_artifacts(directory)


def _phase_intervals(frame: pd.DataFrame) -> list[tuple[str, float, float]]:
    intervals = []
    for phase in PHASE_ORDER:
        rows = frame.loc[frame["phase"] == phase]
        if rows.empty:
            continue
        intervals.append(
            (phase, float(rows["time_start_s"].min()), float(rows["time_s"].max()))
        )
    return intervals


def _shade_phases(figure: Any, frame: pd.DataFrame) -> None:
    for phase, start, end in _phase_intervals(frame):
        figure.add_vrect(
            x0=start / 3600.0,
            x1=end / 3600.0,
            fillcolor=PHASE_COLORS[phase],
            opacity=0.15,
            line_width=0,
            layer="below",
            annotation_text=phase.title(),
            annotation_position="top left",
            row="all",
            col="all",
        )


def _display_frame(frame: pd.DataFrame, stride: int, position: int) -> pd.DataFrame:
    indices = set(range(0, len(frame), max(stride, 1)))
    indices.update({position, len(frame) - 1})
    return frame.iloc[sorted(indices)]


def _mission_profile_figure(
    frame: pd.DataFrame, display: pd.DataFrame, current_time_s: float
) -> Any:
    _, go, make_subplots = _plotly()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=display["mission_time_h"], y=display["altitude_m"] / 1000.0,
            name="Altitude", line={"color": "#0369A1", "width": 2},
            hovertemplate="%{x:.3f} h<br>%{y:.3f} km<extra>Altitude</extra>",
        ), secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=display["mission_time_h"], y=display["speed_mps"],
            name="Airspeed", line={"color": MOTOR_COLOR, "width": 2},
            hovertemplate="%{x:.3f} h<br>%{y:.2f} m/s<extra>Airspeed</extra>",
        ), secondary_y=True,
    )
    _shade_phases(figure, frame)
    figure.add_vline(x=current_time_s / 3600.0, line_color="#DC2626", line_width=2)
    figure.update_layout(
        title="Mission altitude and airspeed",
        template="plotly_white",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"color": "#0F172A"},
        hovermode="x unified", height=470, margin={"l": 30, "r": 30, "t": 70, "b": 35},
        legend={"orientation": "h", "y": 1.12},
    )
    figure.update_xaxes(title="Simulated mission time (h)")
    figure.update_yaxes(title="Altitude (km)", secondary_y=False)
    figure.update_yaxes(title="Airspeed (m/s)", secondary_y=True)
    return figure


def _phase_duration_figure(bundle: ValidationBundle) -> Any:
    _, go, _ = _plotly()
    durations = bundle.mission_result.phase_durations_s
    values = [durations.get(name, 0.0) / 60.0 for name in PHASE_ORDER]
    figure = go.Figure(
        go.Bar(
            x=[name.title() for name in PHASE_ORDER], y=values,
            marker_color=[PHASE_COLORS[name] for name in PHASE_ORDER],
            text=[f"{value:.1f}" for value in values], textposition="outside",
            hovertemplate="%{x}<br>%{y:.2f} min<extra></extra>",
        )
    )
    figure.update_layout(
        title="Phase duration summary", yaxis_title="Duration (min)",
        template="plotly_white",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"color": "#0F172A"},
        height=390, margin={"l": 30, "r": 20, "t": 60, "b": 30},
    )
    return figure


def _power_figure(frame: pd.DataFrame, display: pd.DataFrame, current_time_s: float) -> Any:
    _, go, make_subplots = _plotly()
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=(0.80, 0.20), vertical_spacing=0.08,
        subplot_titles=("DC-bus power balance", "Engine state and restarts"),
    )
    traces = (
        ("Motor electrical input", "motor_electrical_input_kw", MOTOR_COLOR),
        ("Generator contribution", "rectifier_bus_output_kw", ENGINE_COLOR),
        ("Battery bus power", "battery_bus_kw", BATTERY_COLOR),
        ("DC-bus demand", "bus_demand_kw", DEMAND_COLOR),
    )
    for label, column, color in traces:
        figure.add_trace(
            go.Scatter(
                x=display["mission_time_h"], y=display[column], name=label,
                line={"color": color, "width": 2},
                hovertemplate=f"%{{x:.3f}} h<br>%{{y:.3f}} kW<extra>{label}</extra>",
            ), row=1, col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=display["mission_time_h"], y=display["engine_on"].astype(int),
            name="Engine ON", line={"color": ENGINE_COLOR, "width": 2, "shape": "hv"},
            hovertemplate="%{x:.3f} h<br>Engine state: %{y}<extra></extra>",
        ), row=2, col=1,
    )
    restarts = display.loc[display["restart_event"]]
    if not restarts.empty:
        figure.add_trace(
            go.Scatter(
                x=restarts["mission_time_h"], y=[1.0] * len(restarts), mode="markers",
                name="Restart", marker={"color": "#DC2626", "size": 9, "symbol": "x"},
                hovertemplate="%{x:.3f} h<extra>Engine restart</extra>",
            ), row=2, col=1,
        )
    _shade_phases(figure, frame)
    figure.add_hline(y=0.0, line_color="#64748B", line_width=1, row=1, col=1)
    figure.add_vline(
        x=current_time_s / 3600.0, line_color="#DC2626", line_width=2,
        row="all", col="all",
    )
    figure.update_layout(
        title="Series-hybrid electrical power flow",
        template="plotly_white",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"color": "#0F172A"},
        hovermode="x unified", height=620,
        legend={"orientation": "h", "y": 1.10},
        margin={"l": 30, "r": 20, "t": 90, "b": 35},
    )
    figure.update_yaxes(title_text="Power (kW)", row=1, col=1)
    figure.update_yaxes(
        title_text="State", tickmode="array", tickvals=(0, 1), ticktext=("OFF", "ON"),
        range=(-0.15, 1.2), row=2, col=1,
    )
    figure.update_xaxes(title_text="Simulated mission time (h)", row=2, col=1)
    return figure


def _resources_figure(frame: pd.DataFrame, display: pd.DataFrame, current_time_s: float) -> Any:
    _, go, make_subplots = _plotly()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(x=display["mission_time_h"], y=display["soc"], name="Battery SoC",
                   line={"color": SOC_COLOR, "width": 2}), secondary_y=False,
    )
    for column, label, color, dash in (
        ("thermostat_soc_low", "Thermostat low", "#166534", "dash"),
        ("thermostat_soc_high", "Thermostat high", "#22C55E", "dash"),
        ("battery_soc_floor", "Battery floor", "#991B1B", "dot"),
    ):
        figure.add_trace(
            go.Scatter(x=display["mission_time_h"], y=display[column], name=label,
                       line={"color": color, "dash": dash, "width": 1.5}),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(x=display["mission_time_h"], y=display["fuel_remaining_kg"],
                   name="Fuel remaining", line={"color": FUEL_COLOR, "width": 2}),
        secondary_y=True,
    )
    _shade_phases(figure, frame)
    figure.add_vline(x=current_time_s / 3600.0, line_color="#DC2626", line_width=2)
    figure.update_layout(
        title="Stored energy and fuel", hovermode="x unified", height=470,
        template="plotly_white",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"color": "#0F172A"},
        legend={"orientation": "h", "y": 1.13},
        margin={"l": 30, "r": 30, "t": 75, "b": 35},
    )
    figure.update_xaxes(title="Simulated mission time (h)")
    figure.update_yaxes(title="State of charge (-)", range=[0, 1.02], secondary_y=False)
    figure.update_yaxes(title="Fuel remaining (kg)", secondary_y=True)
    return figure


def _fuel_ledger_figure(display: pd.DataFrame) -> Any:
    _, go, _ = _plotly()
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=display["mission_time_h"], y=display["cumulative_running_fuel_kg"],
        name="Running fuel", line={"color": FUEL_COLOR, "width": 2}, stackgroup="fuel",
    ))
    figure.add_trace(go.Scatter(
        x=display["mission_time_h"], y=display["cumulative_restart_fuel_kg"],
        name="Restart fuel", line={"color": ENGINE_COLOR, "width": 2}, stackgroup="fuel",
    ))
    figure.update_layout(
        title="Cumulative physical fuel ledger", xaxis_title="Simulated mission time (h)",
        template="plotly_white",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"color": "#0F172A"},
        yaxis_title="Fuel consumed (kg)", hovermode="x unified", height=390,
        legend={"orientation": "h", "y": 1.13},
        margin={"l": 30, "r": 20, "t": 70, "b": 35},
    )
    return figure


def _operating_figure(frame: pd.DataFrame, display: pd.DataFrame, current_time_s: float) -> Any:
    _, go, make_subplots = _plotly()
    figure = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        specs=[[{}], [{}], [{"secondary_y": True}], [{}]],
        subplot_titles=(
            "Engine shaft operation", "Series-hybrid component power",
            "Battery electrical state", "Component efficiency",
        ),
    )
    for label, column, color in (
        ("Requested engine shaft", "requested_engine_shaft_kw", "#F59E0B"),
        ("Delivered engine shaft", "engine_shaft_kw", ENGINE_COLOR),
        ("Available engine shaft", "engine_available_kw", "#7C2D12"),
    ):
        figure.add_trace(go.Scatter(
            x=display["mission_time_h"], y=display[column], name=label,
            line={"color": color, "width": 2}, connectgaps=False,
        ), row=1, col=1)
    for label, column, color in (
        ("Generator electrical output", "generator_electrical_output_kw", ENGINE_COLOR),
        ("Battery bus power", "battery_bus_kw", BATTERY_COLOR),
        ("Motor electrical input", "motor_electrical_input_kw", MOTOR_COLOR),
        ("Motor shaft output", "motor_shaft_output_kw", "#6D28D9"),
    ):
        figure.add_trace(go.Scatter(
            x=display["mission_time_h"], y=display[column], name=label,
            line={"color": color, "width": 1.8}, connectgaps=False,
        ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=display["mission_time_h"], y=display["battery_current_a"],
        name="Battery current", line={"color": BATTERY_COLOR, "width": 2},
    ), row=3, col=1, secondary_y=False)
    figure.add_trace(go.Scatter(
        x=display["mission_time_h"], y=display["battery_terminal_voltage_v"],
        name="Terminal voltage", line={"color": "#0F766E", "width": 2},
    ), row=3, col=1, secondary_y=True)
    for label, column, color in (
        ("Engine thermal", "engine_thermal_efficiency", ENGINE_COLOR),
        ("Engine shaft to DC bus", "engine_source_efficiency", "#EA580C"),
        ("DC bus to motor shaft", "demand_path_efficiency", MOTOR_COLOR),
        ("Battery terminal", "battery_terminal_efficiency", BATTERY_COLOR),
    ):
        figure.add_trace(go.Scatter(
            x=display["mission_time_h"], y=display[column], name=label,
            line={"color": color, "width": 1.8}, connectgaps=False,
        ), row=4, col=1)
    _shade_phases(figure, frame)
    figure.add_vline(
        x=current_time_s / 3600.0, line_color="#DC2626", line_width=2,
        row="all", col="all",
    )
    figure.update_yaxes(title_text="Power (kW)", row=1, col=1)
    figure.update_yaxes(title_text="Power (kW)", row=2, col=1)
    figure.update_yaxes(title_text="Current (A)", row=3, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Voltage (V)", row=3, col=1, secondary_y=True)
    figure.update_yaxes(title_text="Efficiency (-)", range=[0, 1.02], row=4, col=1)
    figure.update_xaxes(title_text="Simulated mission time (h)", row=4, col=1)
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"color": "#0F172A"},
        height=1050, hovermode="x unified", legend={"orientation": "h", "y": 1.06},
        margin={"l": 40, "r": 30, "t": 90, "b": 35},
    )
    return figure


def _architecture_html() -> str:
    return """
    <div class="architecture">
      <span class="box">Fuel</span><span>→</span><span class="box">Engine</span><span>→</span>
      <span class="box">Generator</span><span>→</span><span class="box">DC bus</span><span>→</span>
      <span class="box">Motor</span><span>→</span><span class="box">Propeller</span>
      <span style="margin-left:.7rem">↕</span><span class="box battery">Battery</span>
    </div>
    """


def _mission_tab(
    bundle: ValidationBundle, frame: pd.DataFrame, display: pd.DataFrame, position: int
) -> None:
    row = frame.iloc[position]
    total = bundle.mission_result.endurance_s
    metrics = st.columns(5)
    metrics[0].metric("Mission time", f"{row['time_s'] / 3600:.3f} h")
    metrics[1].metric("Phase", str(row["phase"]).title())
    metrics[2].metric("Altitude", f"{row['altitude_m'] / 1000:.2f} km")
    metrics[3].metric("Fuel", f"{row['fuel_remaining_kg']:.2f} kg")
    metrics[4].metric("Battery SoC", f"{row['soc'] * 100:.1f}%")
    metrics = st.columns(5)
    metrics[0].metric("Airspeed", f"{row['speed_mps']:.2f} m/s")
    metrics[1].metric("Aircraft mass", f"{row['mass_kg']:.2f} kg")
    metrics[2].metric("Engine", "ON" if row["engine_on"] else "OFF")
    metrics[3].metric("Elapsed", f"{row['time_s'] / 60:.1f} min")
    metrics[4].metric("Remaining", f"{max(total - row['time_s'], 0) / 60:.1f} min")
    st.plotly_chart(
        _mission_profile_figure(frame, display, float(row["time_s"])),
        width="stretch", theme=None, config={"displaylogo": False},
    )
    left, right = st.columns([1.45, 1.0])
    left.plotly_chart(
        _phase_duration_figure(bundle), width="stretch", theme=None,
        config={"displaylogo": False},
    )
    loiter = bundle.mission_result.phase_durations_s.get("loiter", 0.0)
    with right:
        st.subheader("Completed mission result")
        st.metric("Total endurance", f"{total / 3600:.4f} h")
        st.metric("Loiter endurance", f"{loiter / 3600:.4f} h")
        st.write(f"Termination: `{bundle.mission_result.termination_reason}`")
        st.write(f"All six phases complete: **{bundle.mission_result.mission_complete}**")
        st.caption(bundle.scenario.claim)


def _energy_tab(
    bundle: ValidationBundle, frame: pd.DataFrame, display: pd.DataFrame, position: int
) -> None:
    row = frame.iloc[position]
    st.markdown(_architecture_html(), unsafe_allow_html=True)
    st.caption(ARCHITECTURE_LABEL)
    st.info(
        "Battery bus power is positive on discharge and negative on charge. "
        "All contributions on the power-flow chart are electrical DC-bus or explicitly labelled motor-input quantities."
    )
    st.plotly_chart(
        _power_figure(frame, display, float(row["time_s"])), width="stretch",
        theme=None, config={"displaylogo": False},
    )
    restarts = frame.loc[frame["restart_event"]]
    if not restarts.empty:
        st.caption(
            "Restart samples (mission minutes): "
            + ", ".join(f"{value / 60:.1f}" for value in restarts["time_s"].head(20))
            + (" …" if len(restarts) > 20 else "")
        )
    left, right = st.columns(2)
    left.plotly_chart(
        _resources_figure(frame, display, float(row["time_s"])), width="stretch",
        theme=None, config={"displaylogo": False},
    )
    right.plotly_chart(
        _fuel_ledger_figure(display), width="stretch", theme=None,
        config={"displaylogo": False},
    )


def _optional_metric(value: Any, unit: str, decimals: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "Unavailable"
    return f"{float(value):.{decimals}f} {unit}".rstrip()


def _efficiency_tab(frame: pd.DataFrame, display: pd.DataFrame, position: int) -> None:
    row = frame.iloc[position]
    st.plotly_chart(
        _operating_figure(frame, display, float(row["time_s"])), width="stretch",
        theme=None, config={"displaylogo": False},
    )
    cols = st.columns(4)
    cols[0].metric("Engine load", f"{row['engine_load_fraction'] * 100:.1f}%")
    cols[1].metric("Battery current", _optional_metric(row["battery_current_a"], "A"))
    cols[2].metric("Terminal voltage", _optional_metric(row["battery_terminal_voltage_v"], "V"))
    cols[3].metric("SFC", f"{row['sfc_kg_kwh']:.3f} kg/kWh" if math.isfinite(row['sfc_kg_kwh']) else "Engine OFF")
    cols = st.columns(4)
    cols[0].metric(
        "Useful propulsion energy",
        f"{row['cumulative_useful_propulsion_energy_kwh']:.2f} kWh",
    )
    cols[1].metric(
        "Battery bus throughput",
        f"{row['cumulative_battery_bus_throughput_kwh']:.2f} kWh",
    )
    cols[2].metric("Power-limit samples", int(frame["power_limited"].sum()))
    cols[3].metric(
        "Battery-limit samples", int((frame["battery_active_limit"] != "none").sum())
    )
    st.markdown(
        "**Authoritative definitions.** Engine thermal efficiency is shaft power divided by fuel-LHV power. "
        "Source efficiency is rectified DC-bus output divided by engine shaft power. Demand-path efficiency "
        "is motor shaft output divided by DC-bus demand. Battery terminal efficiency is bus output/internal "
        "withdrawal on discharge and stored input/bus input on charge."
    )
    st.caption(
        "Generator, rectifier, cabling, inverter and motor use the documented constant-efficiency branch in "
        "these frozen scenarios. No torque, RPM, temperature, generator voltage/current or efficiency maps exist "
        "in the model, and there is no independent motor-limit flag, so the dashboard does not invent them."
    )


def _ga_history_figure(history: pd.DataFrame) -> Any:
    _, go, make_subplots = _plotly()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(
        x=history["generation"], y=history["best_feasible_objective"] / 3600.0,
        name="Best feasible loiter", line={"color": SOC_COLOR, "width": 3},
    ), secondary_y=False)
    for label, column, color in (
        ("Feasible", "feasible_count", STATUS_COLORS["Feasible"]),
        ("Static infeasible", "static_infeasible_count", STATUS_COLORS["Static infeasible"]),
        ("Dynamic infeasible", "dynamic_infeasible_count", STATUS_COLORS["Dynamic infeasible"]),
    ):
        figure.add_trace(go.Bar(
            x=history["generation"], y=history[column], name=label,
            marker_color=color, opacity=0.55,
        ), secondary_y=True)
    figure.update_layout(
        title="GA convergence and population feasibility", barmode="stack", height=500,
        template="plotly_white",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"color": "#0F172A"},
        legend={"orientation": "h", "y": 1.02, "x": 0.04}, hovermode="x unified",
        margin={"l": 30, "r": 30, "t": 100, "b": 35},
    )
    figure.update_xaxes(title="Generation (generation zero included)")
    figure.update_yaxes(title="Best feasible loiter (h)", secondary_y=False)
    figure.update_yaxes(title="Candidates in population", secondary_y=True)
    return figure


def _optimization_tab(artifacts: GAArtifacts) -> None:
    best = artifacts.best_found
    history = ga_history_dataframe(artifacts)
    candidates = ga_candidates_dataframe(artifacts)
    st.warning("Optimization evaluation — 60-second timestep. This is not a 15-second validation result.")
    st.caption("Best feasible design found by one completed GA seed; no global-optimum claim is made.")
    st.plotly_chart(
        _ga_history_figure(history), width="stretch", theme=None,
        config={"displaylogo": False},
    )
    run = best["run"]
    statistics = run["statistics"]
    cols = st.columns(5)
    cols[0].metric("Populations", run["completed_generation_count"])
    cols[1].metric("Unique evaluations", statistics["unique_fitness_evaluations"])
    cols[2].metric("Full missions", statistics["mission_calls"])
    cols[3].metric("Cache hits", statistics["cache_hits"])
    cols[4].metric("Termination", run["termination_reason"])

    st.subheader("Best design versus practical reference")
    selected = best["aircraft"]
    reference_row = candidates.loc[
        candidates["candidate_context"] == "practical_seed"
    ].iloc[0]
    reference = {
        name: reference_row[name]
        for name in (
            "wing_area_m2", "aspect_ratio", "engine_rating_kw",
            "battery_capacity_kwh", "soc_low", "soc_high",
            "dry_mass_kg", "initial_fuel_kg",
        )
    }
    comparison = pd.DataFrame(
        {
            "Variable": list(reference),
            "Practical reference": [reference[name] for name in reference],
            "GA-selected": [selected[name] for name in reference],
        }
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    x_options = {
        "Wing area (m²)": "wing_area_m2",
        "Aspect ratio": "aspect_ratio",
        "Engine rating (kW)": "engine_rating_kw",
        "Battery capacity (kWh)": "battery_capacity_kwh",
        "Thermostat low SoC": "soc_low",
        "Thermostat high SoC": "soc_high",
    }
    y_options = {
        "Loiter endurance (h)": "objective_loiter_hours",
        "Dry mass (kg)": "dry_mass_kg",
        "Initial fuel allocation (kg)": "initial_fuel_kg",
        "Restart count": "restart_count",
        "Combined normalized violation": "combined_normalized_violation",
    }
    controls = st.columns(2)
    x_label = controls[0].selectbox("Trade-off x-axis", tuple(x_options), index=0)
    y_label = controls[1].selectbox("Trade-off y-axis", tuple(y_options), index=0)
    x_name, y_name = x_options[x_label], y_options[y_label]
    px, _, _ = _plotly()
    hover = [
        name for name in (
            "wing_area_m2", "aspect_ratio", "engine_rating_kw", "battery_capacity_kwh",
            "soc_low", "soc_high", "dry_mass_kg", "initial_fuel_kg",
            "objective_loiter_hours", "restart_count", "feasibility_status",
        ) if name in candidates.columns
    ]
    plotted = candidates.loc[candidates[x_name].notna() & candidates[y_name].notna()].copy()
    figure = px.scatter(
        plotted, x=x_name, y=y_name, color="feasibility_status",
        symbol="feasibility_status", hover_data=hover,
        color_discrete_map=STATUS_COLORS,
        category_orders={"feasibility_status": ["Feasible", "Dynamic infeasible", "Static infeasible"]},
    )
    anchors = plotted.loc[plotted["candidate_context"].isin(("practical_seed", "ideal_seed"))]
    if not anchors.empty:
        figure.add_scatter(
            x=anchors[x_name], y=anchors[y_name], mode="markers+text",
            text=anchors["candidate_context"], textposition="top center",
            marker={"size": 14, "color": "#111827", "symbol": "diamond-open", "line": {"width": 2}},
            name="Reference anchors",
        )
    best_y = {
        "objective_loiter_hours": best["mission"]["objective_hours"],
        "dry_mass_kg": selected["dry_mass_kg"],
        "initial_fuel_kg": selected["initial_fuel_kg"],
        "restart_count": best["mission"]["controller_behavior"]["restart_count"],
        "combined_normalized_violation": best["constraints"]["combined_normalized_violation"],
    }
    figure.add_scatter(
        x=[selected[x_name] if x_name in selected else best["mission"].get(x_name)],
        y=[best_y[y_name]],
        mode="markers", marker={"size": 17, "color": "#DC2626", "symbol": "star"},
        name="Best feasible found",
    )
    figure.update_layout(
        title=f"Candidate trade-off: {x_label} vs {y_label}", height=570,
        template="plotly_white",
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font={"color": "#0F172A"},
        xaxis_title=x_label, yaxis_title=y_label,
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 30, "r": 20, "t": 75, "b": 40},
    )
    st.plotly_chart(
        figure, width="stretch", theme=None, config={"displaylogo": False}
    )
    for warning in artifacts.warnings:
        st.warning(warning)
    st.download_button(
        "Download selected GA trade-off data (CSV)",
        ga_tradeoff_csv_bytes(artifacts),
        file_name="ga_evaluated_candidates_dashboard.csv", mime="text/csv",
    )


def _scenario_configuration(scenario: FrozenDashboardScenario) -> None:
    design = scenario.decoded_design
    st.sidebar.caption(scenario.claim)
    st.sidebar.dataframe(
        pd.DataFrame(
            {
                "Input": ("Wing area", "Aspect ratio", "Engine rating", "Battery", "SoC low", "SoC high"),
                "Value": (
                    f"{design['wing_area_m2']:.9g} m²", f"{design['aspect_ratio']:.9g}",
                    f"{design['engine_rating_kw']:.9g} kW", f"{design['battery_capacity_kwh']:.9g} kWh",
                    f"{design['soc_low']:.6f}", f"{design['soc_high']:.6f}",
                ),
            }
        ),
        hide_index=True, width="stretch",
    )


def _playback_controls(frame: pd.DataFrame, scenario_key: str) -> tuple[int, int, int]:
    position_key = f"position_{scenario_key}"
    playing_key = f"playing_{scenario_key}"
    slider_key = f"slider_{scenario_key}"
    pending_key = f"pending_position_{scenario_key}"
    st.session_state.setdefault(position_key, 0)
    st.session_state.setdefault(playing_key, False)
    if pending_key in st.session_state:
        pending = min(int(st.session_state.pop(pending_key)), len(frame) - 1)
        st.session_state[position_key] = pending
        st.session_state[slider_key] = pending
    st.session_state.setdefault(slider_key, int(st.session_state[position_key]))
    speed = st.sidebar.select_slider("Playback speed", options=(1, 2, 5, 10, 20), value=5,
                                     format_func=lambda value: f"{value} samples/update")
    buttons = st.sidebar.columns(3)
    play_clicked = buttons[0].button("Play", use_container_width=True)
    pause_clicked = buttons[1].button("Pause", use_container_width=True)
    reset_clicked = buttons[2].button("Reset", use_container_width=True)
    if play_clicked:
        st.session_state[playing_key] = True
        st.rerun()
    if pause_clicked:
        st.session_state[playing_key] = False
        st.rerun()
    if reset_clicked:
        st.session_state[playing_key] = False
        st.session_state[position_key] = 0
        st.session_state[slider_key] = 0
        st.rerun()
    phase = st.sidebar.selectbox("Jump to phase", PHASE_ORDER, format_func=str.title)
    if st.sidebar.button("Go to phase", use_container_width=True):
        matches = frame.index[frame["phase"] == phase]
        if len(matches):
            st.session_state[position_key] = int(matches[0])
            st.session_state[slider_key] = int(matches[0])
    position = st.sidebar.select_slider(
        "Mission-time sample", options=range(len(frame)),
        key=slider_key,
        format_func=lambda index: f"{frame.iloc[index]['time_s'] / 3600:.3f} h",
    )
    st.session_state[position_key] = position
    stride = st.sidebar.select_slider(
        "Chart display downsampling", options=(1, 2, 4, 8, 16), value=2,
        format_func=lambda value: "All samples" if value == 1 else f"Every {value} samples",
    )
    return position, stride, speed


def _queue_playback_update(
    frame: pd.DataFrame, scenario_key: str, speed: int
) -> None:
    playing_key = f"playing_{scenario_key}"
    if not st.session_state[playing_key]:
        return
    position_key = f"position_{scenario_key}"
    pending_key = f"pending_position_{scenario_key}"
    updated = min(int(st.session_state[position_key]) + int(speed), len(frame) - 1)
    st.session_state[pending_key] = updated
    if updated >= len(frame) - 1:
        st.session_state[playing_key] = False


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="✈️", layout="wide")
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption(REPLAY_LABEL)
    best_path = PRODUCTION_GA_DIRECTORY / "best_found.json"
    try:
        scenarios = load_dashboard_scenarios(best_path)
    except DashboardDataError as error:
        st.error(str(error))
        st.stop()
    labels = {key: value.label for key, value in scenarios.items()}
    selected_key = st.sidebar.selectbox(
        "Aircraft scenario", tuple(scenarios), format_func=labels.get,
    )
    scenario = scenarios[selected_key]
    _scenario_configuration(scenario)
    st.sidebar.write("**Mission:** 3 km · six phases · 15 s integration")
    st.sidebar.write("**Restart assumption:** 0.1 kg/start · 60 s hard ON/OFF dwell")
    run_clicked = st.sidebar.button("Run simulation", type="primary", use_container_width=True)
    bundle_key = f"validation_{selected_key}"
    if run_clicked:
        try:
            with st.spinner("Running one deterministic 15-second mission…"):
                st.session_state[bundle_key] = _cached_validation(
                    selected_key, scenario.evaluation_key, str(best_path)
                )
        except (DashboardDataError, ValueError, RuntimeError) as error:
            st.error(f"Simulation could not be completed: {error}")
    bundle = st.session_state.get(bundle_key)
    try:
        artifacts = _cached_ga_artifacts(str(PRODUCTION_GA_DIRECTORY))
    except DashboardDataError as error:
        artifacts = None
        st.warning(f"GA results are unavailable: {error}")

    tabs = st.tabs(("Mission overview", "Energy & power flow", "Operating conditions", "Optimization results"))
    if bundle is None:
        for tab in tabs[:3]:
            with tab:
                st.info("Select the aircraft and click **Run simulation** to load its 15-second telemetry.")
    else:
        frame = telemetry_dataframe(bundle)
        with tabs[0]:
            mission_slot = st.empty()
        with tabs[1]:
            energy_slot = st.empty()
        with tabs[2]:
            operating_slot = st.empty()
        replay_interval = (
            0.5 if st.session_state.get(f"playing_{selected_key}", False) else None
        )
        epoch_key = f"replay_epoch_{selected_key}"
        replay_epoch = int(st.session_state.get(epoch_key, 0)) + 1
        st.session_state[epoch_key] = replay_epoch

        @st.fragment(run_every=replay_interval)
        def replay_telemetry() -> None:
            if st.session_state.get(epoch_key) != replay_epoch:
                return
            position, stride, speed = _playback_controls(frame, selected_key)
            display = _display_frame(frame, stride, position)
            with mission_slot.container():
                _mission_tab(bundle, frame, display, position)
            with energy_slot.container():
                _energy_tab(bundle, frame, display, position)
            with operating_slot.container():
                _efficiency_tab(frame, display, position)
            st.sidebar.download_button(
                "Download telemetry CSV", telemetry_csv_bytes(bundle),
                file_name=f"{selected_key}_telemetry_15s.csv", mime="text/csv",
                use_container_width=True,
            )
            st.sidebar.download_button(
                "Download summary JSON", validation_summary_json_bytes(bundle),
                file_name=f"{selected_key}_summary_15s.json", mime="application/json",
                use_container_width=True,
            )
            _queue_playback_update(frame, selected_key, speed)

        replay_telemetry()
    with tabs[3]:
        if artifacts is None:
            st.error("Completed GA artifacts could not be loaded; no fallback values were substituted.")
        else:
            try:
                _optimization_tab(artifacts)
            except (DashboardDataError, KeyError, TypeError, ValueError) as error:
                st.error(f"GA charts could not be rendered: {error}")
        try:
            validation = load_validation_summary()
        except DashboardDataError:
            validation = None
        if validation is not None:
            st.subheader("Separate 15-second validation outcomes")
            rows = []
            for key, item in validation["scenarios"].items():
                rows.append(
                    {
                        "Scenario": item["scenario_label"],
                        "Total endurance (h)": item["total_mission_seconds"] / 3600.0,
                        "Loiter endurance (h)": item["loiter_seconds"] / 3600.0,
                        "Restarts": item["restart_count"],
                        "Feasible": item["dynamically_feasible"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.metric(
                "Validated GA loiter improvement",
                f"{validation['validated_loiter_improvement_percent']:.3f}%",
            )


if __name__ == "__main__":
    main()
