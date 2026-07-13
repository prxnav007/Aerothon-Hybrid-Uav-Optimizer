# src/physics.py
import math
import numpy as np
from src.config import UAV_SPECS, ISA_CONSTANTS

def calculate_air_density(altitude_m):
    """Calculates air density using the International Standard Atmosphere (ISA) model up to 11km."""
    if altitude_m < 0:
        altitude_m = 0
    elif altitude_m > 11000:
        altitude_m = 11000 # Cap at tropopause for simplification
        
    temp = ISA_CONSTANTS["temp0"] - (ISA_CONSTANTS["temp_lapse_rate"] * altitude_m)
    rho = ISA_CONSTANTS["rho0"] * (temp / ISA_CONSTANTS["temp0"])**4.255
    return rho

def calculate_drag(speed_mps, altitude_m, weight_kg):
    """Calculates drag force using simplified aerodynamic equations."""
    rho = calculate_air_density(altitude_m)
    q = 0.5 * rho * (speed_mps ** 2) # Dynamic pressure
    
    # Calculate Lift Coefficient (Cl)
    weight_n = weight_kg * ISA_CONSTANTS["gravity"]
    cl = weight_n / (q * UAV_SPECS["wing_area_m2"])
    
    # Calculate Drag Coefficient (Cd) using parabolic drag polar: Cd = Cd0 + Cl^2 / (pi * AR * e)
    cd = UAV_SPECS["cd0"] + (cl ** 2) / (math.pi * UAV_SPECS["aspect_ratio"] * UAV_SPECS["oswald_efficiency"])
    
    drag_force = cd * q * UAV_SPECS["wing_area_m2"]
    return drag_force, cl, cd

def calculate_shaft_power_req(speed_mps, altitude_m, weight_kg, climb_rate_mps=0):
    """Calculates required shaft power for a given flight condition."""
    drag_force, _, _ = calculate_drag(speed_mps, altitude_m, weight_kg)
    
    # If climbing, add the power required to gain altitude (P = mg * vertical_speed)
    weight_n = weight_kg * ISA_CONSTANTS["gravity"]
    power_required_thrust = (drag_force * speed_mps) + (weight_n * climb_rate_mps)
    
    # Account for propeller efficiency
    shaft_power_req = power_required_thrust / UAV_SPECS["propeller_efficiency"]
    
    return max(shaft_power_req, 0) # Ensure power is not negative