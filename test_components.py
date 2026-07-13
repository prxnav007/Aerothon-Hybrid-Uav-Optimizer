from src.components import Battery, TurboshaftEngine, calculate_system_weight
from src.physics import calculate_shaft_power_req

# 1. Test System Weight Constraint
# Let's try a 60kW engine and a 20kWh battery
total_weight = calculate_system_weight(engine_kw=60, battery_kwh=20)
print(f"Total System Weight (60kW eng, 20kWh batt): {total_weight:.2f} kg")
print(f"Within 1000kg MTOW? {total_weight <= 1000}\n")

# 2. Test Battery Discharge
battery = Battery(capacity_kwh=20, max_power_kw=60)
engine = TurboshaftEngine(max_power_kw=60)

# Get required power at cruise
p_req_cruise = calculate_shaft_power_req(speed_mps=69.44, altitude_m=8000, weight_kg=1000)
print(f"Power required at cruise: {p_req_cruise/1000:.2f} kW")

# Simulate 1 minute of flight using ONLY battery power
dt = 60 # 60 seconds
p_req_kw = p_req_cruise / 1000

battery.update(power_kw=p_req_kw, dt_s=dt)
fuel_flow = engine.get_fuel_flow_kg_s(0) # Engine off

print(f"Battery SoC after 1 min of pure electric cruise: {battery.soc*100:.2f}%")
print(f"Fuel flow (engine off): {fuel_flow:.4f} kg/s\n")

# Simulate 1 minute of flight using ONLY engine power (charging battery)
# Engine provides exactly what is needed, plus a bit extra to charge
p_engine_kw = p_req_kw + 10 # 10kW extra for charging
battery.update(power_kw=-10, dt_s=dt) # -10 means charging
fuel_flow = engine.get_fuel_flow_kg_s(p_engine_kw)

print(f"Battery SoC after 1 min of engine charging: {battery.soc*100:.2f}%")
print(f"Fuel flow (engine on): {fuel_flow:.5f} kg/s")