from src.physics import calculate_air_density, calculate_shaft_power_req

# Test ISA model at 8000m (Cruise Altitude)
rho_8km = calculate_air_density(8000)
print(f"Air Density at 8000m: {rho_8km:.4f} kg/m^3")

# Test Power required at Cruise (8000m, 250 km/h, 1000kg MTOW)
p_req = calculate_shaft_power_req(speed_mps=69.44, altitude_m=8000, weight_kg=1000)
print(f"Required Shaft Power at Cruise: {p_req:.2f} Watts ({p_req/1000:.2f} kW)")