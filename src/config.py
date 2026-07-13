# src/config.py

# Baseline UAV Specifications
UAV_SPECS = {
    "mtow_kg": 1000,
    "payload_kg": 200,
    "cruise_speed_mps": 250 * (1000 / 3600), # Convert km/h to m/s
    "cruise_altitude_m": 8000, # Average of 3-10 km range
    "wing_area_m2": 5.0, # Assumption for a 1000kg UAV
    "cd0": 0.025, # Zero-lift drag coefficient (typical for clean UAV)
    "aspect_ratio": 10,
    "oswald_efficiency": 0.8,
    "propeller_efficiency": 0.85,
    "motor_efficiency": 0.95,
    "rectifier_efficiency": 0.95,
    "inverter_efficiency": 0.95
}

# Mission Profile (Time in seconds, Altitude in meters, Phase name)
# This is a simplified profile for testing. You can expand it later.
MISSION_PROFILE = [
    {"phase": "takeoff", "duration_s": 60, "altitude_m": 0, "speed_mps": 60},
    {"phase": "climb", "duration_s": 300, "altitude_m": 8000, "speed_mps": 80},
    {"phase": "cruise", "duration_s": 5000, "altitude_m": 8000, "speed_mps": UAV_SPECS["cruise_speed_mps"]},
    {"phase": "loiter", "duration_s": 3000, "altitude_m": 8000, "speed_mps": 120},
    {"phase": "descent", "duration_s": 600, "altitude_m": 0, "speed_mps": 80},
    {"phase": "landing", "duration_s": 60, "altitude_m": 0, "speed_mps": 30}
]

# Constants
ISA_CONSTANTS = {
    "rho0": 1.225, # kg/m^3 at sea level
    "temp0": 288.15, # K
    "temp_lapse_rate": 0.0065, # K/m
    "gravity": 9.81 # m/s^2
}