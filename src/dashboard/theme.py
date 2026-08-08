"""Shared visual identity for the Streamlit dashboard."""

ENGINE_COLOR = "#D97706"
BATTERY_COLOR = "#2563EB"
MOTOR_COLOR = "#7C3AED"
DEMAND_COLOR = "#1F2937"
SOC_COLOR = "#15803D"
FUEL_COLOR = "#92400E"

PHASE_COLORS = {
    "takeoff": "#FDE68A",
    "climb": "#BFDBFE",
    "cruise": "#C7D2FE",
    "loiter": "#BBF7D0",
    "descent": "#FED7AA",
    "landing": "#E9D5FF",
}

STATUS_COLORS = {
    "Feasible": "#15803D",
    "Static infeasible": "#B91C1C",
    "Dynamic infeasible": "#D97706",
}

DASHBOARD_CSS = """
<style>
    :root {color-scheme: light;}
    .stApp {background: #FCFCFB; color: #0F172A;}
    [data-testid="stHeader"] {background: rgba(252, 252, 251, 0.92);}
    [data-testid="stSidebar"] {background: #F4F5F7;}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {color: #0F172A;}
    [data-baseweb="select"] > div {background: #FFFFFF; color: #0F172A;}
    [data-testid="stTab"], [data-testid="stTab"] p {color: #334155 !important;}
    [data-testid="stTab"][aria-selected="true"],
    [data-testid="stTab"][aria-selected="true"] p {color: #DC2626 !important;}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 0.7rem; padding: 0.7rem;}
    [data-testid="stMetric"] * {color: #0F172A;}
    [data-testid="stAlert"] p {color: #0F172A !important;}
    .architecture {display: flex; align-items: center; gap: .45rem; flex-wrap: wrap;
        background: #F8FAFC; border: 1px solid #E2E8F0; padding: .9rem;
        border-radius: .7rem; margin: .4rem 0 1rem;}
    .architecture .box {background: white; border: 1px solid #CBD5E1;
        border-radius: .45rem; padding: .35rem .65rem; font-weight: 600;}
    .architecture .battery {border-color: #2563EB; color: #1D4ED8;}
    .caption {color: #475569; font-size: .9rem;}
</style>
"""
