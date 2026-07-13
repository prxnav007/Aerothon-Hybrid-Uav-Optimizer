# src/dashboard/app.py
"""
Interactive Simulation Dashboard (Deliverable 8.3 of the problem statement).

Sections map 1:1 to the required visualizations:
  1. Mission profile and flight phases
  2. Power distribution between thermal and electric propulsion
  3. Battery State of Charge (SoC)
  4. Fuel consumption throughout the mission
  5. Engine and motor operating conditions
  6. System efficiency
  7. Endurance estimation (KPI header + comparison)
  8. Optimization results and design trade-offs

Run from the project root:  streamlit run src/dashboard/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import UAV_SPECS, MISSION_SETTINGS                # noqa: E402
from src.ga_optimization import (run_simulation, logs_to_dataframe,  # noqa: E402
                                 BASELINE_GENES)

DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------------
# Palette (validated reference instance from the data-viz design method).
# Color follows the entity across every chart: engine=blue, battery=aqua,
# demand/motor=yellow, fuel=green, s-factor=violet, efficiency=orange.
# --------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_ENGINE, C_BATT, C_DEMAND = "#2a78d6", "#1baf7a", "#eda100"
C_FUEL, C_SFACTOR, C_EFF = "#008300", "#4a3aa7", "#eb6834"
S_WARN, S_CRIT = "#fab219", "#d03b3b"
C_ENGINE_LIGHT = "#86b6ef"  # lighter step of the same blue ramp (baseline bar / GA avg)
PHASE_COLORS = {"takeoff": "#2a78d6", "climb": "#1baf7a", "cruise": "#eda100",
                "loiter": "#008300", "descent": "#4a3aa7", "landing": "#e34948"}
SEQ_BLUES = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"],
             [0.75, "#1c5cab"], [1.0, "#0d366b"]]

st.set_page_config(page_title="Hybrid-Electric UAV Optimizer", page_icon="🛩️",
                   layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def style(fig, y_title, x_title="Mission time (h)", height=340, unified=True):
    """Shared chart chrome: recessive grid/axes, ink text, top legend, hover layer."""
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family='system-ui, "Segoe UI", sans-serif', color=INK, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified" if unified else "closest",
        hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor=GRID),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     title_font_color=INK2, tickfont_color=MUTED)
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     title_font_color=INK2, tickfont_color=MUTED)
    return fig


def phase_segments(df):
    """Contiguous (phase, t_start, t_end) segments from the log."""
    phases, times = df["phase"].tolist(), df["time_h"].tolist()
    segments, start = [], 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            t_end = times[i] if i < len(phases) else times[-1]
            segments.append((phases[start], times[start], t_end))
            start = i
    return segments


def add_phase_bands(fig, df, labels=False):
    """Soft per-phase background washes; identity is carried by the label text."""
    for k, (phase, t0, t1) in enumerate(phase_segments(df)):
        fig.add_vrect(
            x0=t0, x1=t1, fillcolor=PHASE_COLORS.get(phase, MUTED), opacity=0.10,
            line_width=0,
            annotation_text=phase.capitalize() if labels else None,
            annotation_position="top left" if k % 2 == 0 else "bottom left",
            annotation_font=dict(size=11, color=INK2),
        )


@st.cache_data(show_spinner="Flying the mission simulation…")
def simulate(genes):
    endurance_s, logs = run_simulation(list(genes), record_logs=True)
    return logs_to_dataframe(logs), logs["meta"]


def load_summary():
    path = DATA_DIR / "optimization_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


summary = load_summary()

# --------------------------------------------------------------------------
# Sidebar — scenario selection (interactivity: every chart below re-renders)
# --------------------------------------------------------------------------
st.sidebar.title("🛩️ Design scenario")
options = ["Baseline design (60 kW / 30 kWh)", "Custom design (live simulation)"]
if summary:
    options.insert(0, "Optimized design (GA result)")
else:
    st.sidebar.info("No GA results found yet — run `python -m src.ga_optimization` "
                    "to unlock the optimized design and Section 8.")
scenario = st.sidebar.radio("Simulate and visualize:", options)

if scenario.startswith("Optimized"):
    genes = summary["best_genes"]
elif scenario.startswith("Baseline"):
    genes = BASELINE_GENES
else:
    st.sidebar.caption("Design variables (GA genes) — the mission is re-flown live:")
    genes = [
        st.sidebar.slider("Turboshaft engine size (kW)", 30.0, 100.0, 60.0, 1.0),
        st.sidebar.slider("Battery capacity (kWh)", 5.0, 50.0, 30.0, 1.0),
        st.sidebar.slider("Fuzzy SoC low threshold", 0.10, 0.40, 0.30, 0.01),
        st.sidebar.slider("Fuzzy SoC high threshold", 0.60, 0.90, 0.70, 0.01),
        st.sidebar.slider("ECMS s-factor min", 1.0, 3.0, 3.0, 0.1),
        st.sidebar.slider("ECMS s-factor max", 3.0, 5.5, 5.0, 0.1),
    ]

df, meta = simulate(tuple(genes))

st.sidebar.divider()
st.sidebar.markdown(
    f"**Current design**\n\n"
    f"- Engine: **{meta['engine_kw']:.1f} kW** turboshaft\n"
    f"- Battery: **{meta['battery_kwh']:.1f} kWh** Li-ion\n"
    f"- Fuel loaded: **{meta['initial_fuel_kg']:.0f} kg** (fills to MTOW)\n"
    f"- Payload: **{UAV_SPECS['payload_kg']} kg**\n"
    f"- Take-off weight: **{meta['takeoff_weight_kg']:.0f} kg** "
    f"(MTOW {UAV_SPECS['mtow_kg']} kg)"
)

# --------------------------------------------------------------------------
# Header + Section 7: Endurance estimation & mission feasibility
# --------------------------------------------------------------------------
st.title("Hybrid-Electric Propulsion Optimization — Fixed-Wing UAV")
st.caption("Simulation dashboard · turboshaft + battery hybrid · "
           "fuzzy-adaptive ECMS power management · GA-optimized sizing")

if meta["mission_complete"]:
    st.success(f"✅ **Mission feasible** — all phases completed. "
               f"{meta['termination_reason']}.")
else:
    st.error(f"⚠️ **Mission NOT feasible** — {meta['termination_reason']}.")

if df.empty:
    st.stop()

powered = df[df["p_req_kw"] > 1.0]
baseline_h = summary["baseline_endurance_h"] if summary else None
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Endurance", f"{meta['endurance_h']:.2f} h",
          delta=(f"{meta['endurance_h'] - baseline_h:+.2f} h vs baseline"
                 if baseline_h and scenario.startswith("Optimized") else None))
k2.metric("Take-off weight", f"{meta['takeoff_weight_kg']:.0f} kg",
          help=f"MTOW limit: {UAV_SPECS['mtow_kg']} kg")
k3.metric("Fuel consumed", f"{meta['fuel_used_kg']:.1f} kg")
k4.metric("Minimum SoC", f"{df['soc'].min() * 100:.0f} %")
k5.metric("Avg system efficiency", f"{powered['system_efficiency'].mean() * 100:.1f} %")
k6.metric("Peak power demand", f"{df['p_req_kw'].max():.0f} kW")

# --------------------------------------------------------------------------
# Section 1: Mission profile and flight phases
# --------------------------------------------------------------------------
st.header("1 · Mission profile & flight phases")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["time_h"], y=df["altitude_m"], name="Altitude",
    mode="lines", line=dict(color=INK, width=2),
    fill="tozeroy", fillcolor="rgba(137,135,129,0.08)",
    hovertemplate="%{y:.0f} m"))
add_phase_bands(fig, df, labels=True)
st.plotly_chart(style(fig, "Altitude (m)"), width="stretch", theme=None)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["time_h"], y=df["speed_mps"] * 3.6, name="Airspeed",
    mode="lines", line=dict(color=INK2, width=2), hovertemplate="%{y:.0f} km/h"))
add_phase_bands(fig, df)
st.plotly_chart(style(fig, "Airspeed (km/h)", height=220), width="stretch",
                theme=None)

# --------------------------------------------------------------------------
# Section 2: Power distribution — thermal vs electric
# --------------------------------------------------------------------------
st.header("2 · Power distribution — thermal vs. electric")
st.caption("Stacked areas show who supplies the power at every instant; the dashed "
           "line is total electric demand. Battery area below zero = recharging "
           "from the engine.")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["time_h"], y=df["p_gen_kw"], name="Engine / generator",
    mode="lines", line=dict(color=C_ENGINE, width=0.5), stackgroup="supply",
    fillcolor="rgba(42,120,214,0.55)", hovertemplate="%{y:.1f} kW"))
fig.add_trace(go.Scatter(
    x=df["time_h"], y=df["p_batt_kw"].clip(lower=0), name="Battery (discharge)",
    mode="lines", line=dict(color=C_BATT, width=0.5), stackgroup="supply",
    fillcolor="rgba(27,175,122,0.55)", hovertemplate="%{y:.1f} kW"))
fig.add_trace(go.Scatter(
    x=df["time_h"], y=df["p_batt_kw"].clip(upper=0), name="Battery (charging)",
    mode="lines", line=dict(color=C_BATT, width=1, dash="dot"),
    fill="tozeroy", fillcolor="rgba(27,175,122,0.25)", hovertemplate="%{y:.1f} kW"))
fig.add_trace(go.Scatter(
    x=df["time_h"], y=df["p_req_kw"], name="Total electric demand",
    mode="lines", line=dict(color=C_DEMAND, width=2, dash="dash"),
    hovertemplate="%{y:.1f} kW"))
add_phase_bands(fig, df)
st.plotly_chart(style(fig, "Power (kW)", height=400), width="stretch",
                theme=None)

# --------------------------------------------------------------------------
# Sections 3 & 4: Battery SoC + fuel consumption
# --------------------------------------------------------------------------
c_soc, c_fuel = st.columns(2)
with c_soc:
    st.header("3 · Battery State of Charge")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_h"], y=df["soc"] * 100, name="SoC",
        mode="lines", line=dict(color=C_BATT, width=2), hovertemplate="%{y:.1f} %"))
    fig.add_hline(y=MISSION_SETTINGS["soc_safe_min"] * 100, line_dash="dash",
                  line_color=S_WARN, line_width=1.5,
                  annotation_text="⚠ safe operating floor (20 %)",
                  annotation_font_color=INK2)
    fig.add_hline(y=MISSION_SETTINGS["soc_cutoff"] * 100, line_dash="dash",
                  line_color=S_CRIT, line_width=1.5,
                  annotation_text="✖ hard cutoff (5 %)", annotation_font_color=INK2)
    add_phase_bands(fig, df)
    fig.update_yaxes(range=[0, 105])
    st.plotly_chart(style(fig, "State of Charge (%)"), width="stretch",
                    theme=None)

with c_fuel:
    st.header("4 · Fuel consumption")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_h"], y=df["fuel_kg"], name="Fuel remaining",
        mode="lines", line=dict(color=C_FUEL, width=2),
        fill="tozeroy", fillcolor="rgba(0,131,0,0.10)", hovertemplate="%{y:.1f} kg"))
    fig.add_trace(go.Scatter(
        x=df["time_h"], y=df["fuel_used_kg"], name="Fuel consumed",
        mode="lines", line=dict(color=MUTED, width=1.5, dash="dot"),
        hovertemplate="%{y:.1f} kg"))
    fig.add_hline(y=MISSION_SETTINGS["fuel_reserve_kg"], line_dash="dash",
                  line_color=S_WARN, line_width=1.5,
                  annotation_text="landing reserve", annotation_font_color=INK2)
    add_phase_bands(fig, df)
    st.plotly_chart(style(fig, "Fuel (kg)"), width="stretch", theme=None)

# --------------------------------------------------------------------------
# Section 5: Engine and motor operating conditions
# --------------------------------------------------------------------------
st.header("5 · Engine & motor operating conditions")
c_eng, c_mot = st.columns(2)
with c_eng:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_h"], y=df["engine_load_pct"], name="Engine load",
        mode="lines", line=dict(color=C_ENGINE, width=2), hovertemplate="%{y:.0f} %"))
    fig.add_hline(y=100, line_dash="dash", line_color=S_CRIT, line_width=1.5,
                  annotation_text="rated power", annotation_font_color=INK2)
    add_phase_bands(fig, df)
    fig.update_yaxes(range=[0, 115])
    st.plotly_chart(style(fig, f"Engine load (% of {meta['engine_kw']:.0f} kW rated)"),
                    width="stretch", theme=None)
with c_mot:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_h"], y=df["motor_power_kw"], name="Motor electric power",
        mode="lines", line=dict(color=C_DEMAND, width=2), hovertemplate="%{y:.1f} kW"))
    add_phase_bands(fig, df)
    st.plotly_chart(style(fig, "Motor electric input (kW)"), width="stretch",
                    theme=None)

# --------------------------------------------------------------------------
# Section 6: System efficiency
# --------------------------------------------------------------------------
st.header("6 · System efficiency")
st.caption("Propulsive power out ÷ (fuel chemical power + battery power) in. "
           "Descent/landing read near zero because almost no propulsive power "
           "is needed.")
c_eff1, c_eff2 = st.columns([3, 2])
with c_eff1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_h"], y=df["system_efficiency"] * 100, name="System efficiency",
        mode="lines", line=dict(color=C_EFF, width=2), hovertemplate="%{y:.1f} %"))
    add_phase_bands(fig, df)
    st.plotly_chart(style(fig, "Efficiency (%)"), width="stretch", theme=None)
with c_eff2:
    order = list(dict.fromkeys(df["phase"]))
    phase_eff = df.groupby("phase", sort=False)["system_efficiency"].mean() * 100
    fig = go.Figure(go.Bar(
        x=[p.capitalize() for p in order], y=[phase_eff[p] for p in order],
        marker=dict(color=C_EFF, cornerradius=4), width=0.55,
        text=[f"{phase_eff[p]:.1f}%" for p in order],
        textposition="outside", textfont=dict(color=INK2, size=12),
        hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig = style(fig, "Mean efficiency (%)", x_title="Flight phase", unified=False)
    st.plotly_chart(fig, width="stretch", theme=None)

# --------------------------------------------------------------------------
# Power management strategy (supports Section 2 + innovation criterion)
# --------------------------------------------------------------------------
st.header("Power-management strategy — fuzzy adaptive equivalence factor")
st.caption("The fuzzy controller continuously re-prices battery energy: a high "
           "s-factor protects the battery (engine preferred); a low s-factor "
           "spends it. Watch it rise as SoC falls.")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["time_h"], y=df["s_factor"], name="Equivalence factor s",
    mode="lines", line=dict(color=C_SFACTOR, width=2), hovertemplate="s = %{y:.2f}"))
add_phase_bands(fig, df)
st.plotly_chart(style(fig, "Equivalence factor s (–)", height=260),
                width="stretch", theme=None)

# --------------------------------------------------------------------------
# Section 8: Optimization results and design trade-offs
# --------------------------------------------------------------------------
st.header("8 · Optimization results & design trade-offs")
conv_path = DATA_DIR / "ga_convergence.csv"
evals_path = DATA_DIR / "ga_evaluations.csv"

if summary and conv_path.exists() and evals_path.exists():
    conv = pd.read_csv(conv_path)
    evals = pd.read_csv(evals_path)

    c_conv, c_trade = st.columns(2)
    with c_conv:
        st.subheader("GA convergence")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=conv["gen"], y=conv["max"], name="Best design",
            mode="lines+markers", line=dict(color=C_ENGINE, width=2),
            marker=dict(size=8), hovertemplate="%{y:.2f} h"))
        fig.add_trace(go.Scatter(
            x=conv["gen"], y=conv["avg"], name="Population average",
            mode="lines+markers", line=dict(color=C_ENGINE_LIGHT, width=2, dash="dot"),
            marker=dict(size=8), hovertemplate="%{y:.2f} h"))
        fig = style(fig, "Endurance (h)", x_title="Generation")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch", theme=None)

    with c_trade:
        st.subheader("Design-space trade-off")
        ok = evals[evals["mission_complete"]]
        bad = evals[~evals["mission_complete"]]
        fig = go.Figure()
        if not bad.empty:
            fig.add_trace(go.Scatter(
                x=bad["engine_kw"], y=bad["battery_kwh"], name="Infeasible design",
                mode="markers",
                marker=dict(symbol="x-thin", size=8, line=dict(color=MUTED, width=1.5)),
                hovertemplate="Engine %{x:.0f} kW · Battery %{y:.0f} kWh · "
                              "mission failed<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=ok["engine_kw"], y=ok["battery_kwh"], name="Feasible design",
            mode="markers",
            marker=dict(size=9, color=ok["endurance_h"], colorscale=SEQ_BLUES,
                        colorbar=dict(title="Endurance (h)", outlinewidth=0),
                        line=dict(color=SURFACE, width=1)),
            customdata=ok[["endurance_h"]],
            hovertemplate="Engine %{x:.1f} kW · Battery %{y:.1f} kWh · "
                          "%{customdata[0]:.2f} h<extra></extra>"))
        bg = summary["best_genes"]
        fig.add_trace(go.Scatter(
            x=[bg[0]], y=[bg[1]], name="Optimal design", mode="markers+text",
            marker=dict(symbol="star", size=16, color=C_DEMAND,
                        line=dict(color=INK, width=1)),
            text=["  optimum"], textposition="middle right",
            textfont=dict(color=INK2, size=12),
            hovertemplate=f"Optimum: {bg[0]:.1f} kW · {bg[1]:.1f} kWh<extra></extra>"))
        fig = style(fig, "Battery capacity (kWh)", x_title="Engine size (kW)",
                    unified=False)
        st.plotly_chart(fig, width="stretch", theme=None)

    st.subheader("Endurance improvement (baseline → optimized)")
    best_h = summary["best_meta"]["endurance_h"]
    base_h = summary["baseline_endurance_h"]
    fig = go.Figure(go.Bar(
        y=["Baseline (60 kW / 30 kWh)", "GA-optimized"], x=[base_h, best_h],
        orientation="h", width=0.5,
        marker=dict(color=[C_ENGINE_LIGHT, C_ENGINE], cornerradius=4),
        text=[f"{base_h:.2f} h", f"{best_h:.2f} h  ({summary['endurance_improvement_pct']:+.1f} %)"],
        textposition="outside", textfont=dict(color=INK2, size=13),
        hovertemplate="%{y}: %{x:.2f} h<extra></extra>"))
    fig = style(fig, "", x_title="Endurance (h)", height=220, unified=False)
    fig.update_xaxes(range=[0, best_h * 1.35])
    st.plotly_chart(fig, width="stretch", theme=None)

    with st.expander("Optimal design variables (GA genes)"):
        st.table(pd.DataFrame({
            "Design variable": ["Engine size (kW)", "Battery capacity (kWh)",
                                "Fuzzy SoC low threshold", "Fuzzy SoC high threshold",
                                "ECMS s-factor min", "ECMS s-factor max"],
            "Optimized value": [f"{v:.2f}" for v in summary["best_genes"]],
            "Baseline value": [f"{v:.2f}" for v in summary["baseline_genes"]],
        }))
else:
    st.info("Optimization artifacts not found. Run `python -m src.ga_optimization` "
            "from the project root to generate the GA convergence history, "
            "design-space evaluations, and baseline comparison.")

# --------------------------------------------------------------------------
# Accessibility: full data table + download
# --------------------------------------------------------------------------
with st.expander("📋 Table view — full mission time-series data"):
    st.dataframe(df, width="stretch", height=320)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       file_name="mission_simulation.csv", mime="text/csv")

st.caption("Assumptions: ISA atmosphere · parabolic drag polar · coulomb-counting "
           "battery with linear OCV · constant turboshaft SFC 0.3 kg/kWh · fuel "
           "LHV 43.1 MJ/kg · tank filled to MTOW after payload and powertrain mass.")
