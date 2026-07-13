# src/config.py

# Baseline UAV Specifications
UAV_SPECS = {
    "mtow_kg": 1000,
    "payload_kg": 200,
    "cruise_speed_mps": 250 * (1000 / 3600), # Convert km/h to m/s
    "cruise_altitude_m": 6000, # Within the 3-10 km band; keeps cruise power within a 60 kW-class engine
    "wing_area_m2": 5.0, # Assumption for a 1000kg UAV
    "cd0": 0.025, # Zero-lift drag coefficient (typical for clean UAV)
    "aspect_ratio": 10,
    "oswald_efficiency": 0.8,
    "propeller_efficiency": 0.85,
    "motor_efficiency": 0.95,
    "rectifier_efficiency": 0.95,
    "inverter_efficiency": 0.95
}

# Mission Profile (per problem statement: take-off, climb, cruise, loiter, descent, landing)
# altitude_m is the TARGET altitude at the end of the phase; climb rate is derived from
# (target - current altitude) / duration. Rates are sized so that peak power stays within
# engine + battery capability of a 60 kW-class hybrid (climb ~2 m/s needs ~76 kW electric).
# The loiter phase has duration_s = None: it is the ENDURANCE phase and is extended until
# the fuel reserve or battery cutoff is reached. Total endurance = full mission time.
MISSION_PROFILE = [
    {"phase": "takeoff", "duration_s": 120,  "altitude_m": 300,  "speed_mps": 50},
    {"phase": "climb",   "duration_s": 2850, "altitude_m": 6000, "speed_mps": 65},
    {"phase": "cruise",  "duration_s": 3600, "altitude_m": 6000, "speed_mps": UAV_SPECS["cruise_speed_mps"]},
    {"phase": "loiter",  "duration_s": None, "altitude_m": 6000, "speed_mps": 70},
    {"phase": "descent", "duration_s": 900,  "altitude_m": 300,  "speed_mps": 65},
    {"phase": "landing", "duration_s": 120,  "altitude_m": 0,    "speed_mps": 45}
]

# Mission / energy-management settings
MISSION_SETTINGS = {
    "fuel_lhv_kj_kg": 43100,   # Lower heating value of jet fuel (kJ/kg) - for system efficiency
    "fuel_reserve_kg": 5.0,    # Loiter ends here so descent + landing can always be completed
    "min_fuel_kg": 20.0,       # A design must lift at least this much fuel to be feasible
    "soc_cutoff": 0.05,        # Hard battery floor - simulation never discharges below this
    "soc_safe_min": 0.20,      # Recommended safe operating floor (shown on the dashboard)
    "max_mission_s": 86400     # 24 h safety cap on total mission time
}

# Constants
ISA_CONSTANTS = {
    "rho0": 1.225, # kg/m^3 at sea level
    "temp0": 288.15, # K
    "temp_lapse_rate": 0.0065, # K/m
    "gravity": 9.81 # m/s^2
}