# src/components.py
import numpy as np
from src.config import UAV_SPECS



class Battery:
    """Robust Coulomb-Counting Battery Model with Internal Resistance."""
    def __init__(self, capacity_kwh, max_power_kw, energy_density_wh_kg=250):
        self.capacity_wh = capacity_kwh * 1000
        self.max_power_kw = max_power_kw
        self.energy_density = energy_density_wh_kg
        
        # State variables
        self.soc = 1.0  # Start at 100% SoC
        
        # Electrical parameters
        self.v_ocv = 400  # Open Circuit Voltage (V) at 100% SoC
        self.r_int = 0.05 # Internal Resistance (Ohms)

    def update(self, power_kw, dt_s):
        """Updates SoC and voltage based on power draw over time dt."""
        # Estimate terminal voltage (simplified: sags slightly based on power draw)
        # Positive power = discharging, Negative power = charging
        current = (power_kw * 1000) / self.v_ocv if self.v_ocv != 0 else 0
        
        # Coulomb counting for SoC
        # (current * dt) / capacity in Amp-hours
        delta_soc = (current * dt_s) / (self.capacity_wh / self.v_ocv * 3600)
        self.soc -= delta_soc
        self.soc = max(0.0, min(1.0, self.soc)) # Clamp between 0 and 1
        
        # Update OCV based on SoC (Linear approximation: 300V at 0%, 400V at 100%)
        self.v_ocv = 300 + (100 * self.soc)
        
        # Update terminal voltage considering internal resistance sag
        self.voltage = self.v_ocv - (current * self.r_int)

    def get_weight_kg(self):
        """Calculates battery weight based on capacity and energy density."""
        return self.capacity_wh / self.energy_density

class TurboshaftEngine:
    """Models the gas turbine engine and generator."""
    def __init__(self, max_power_kw, power_to_weight_kw_kg=1.5):
        self.max_power_kw = max_power_kw
        self.power_to_weight = power_to_weight_kw_kg
        
        # Specific Fuel Consumption (kg/kW/s) - simplified constant
        # Typical turboshaft SFC is ~0.3 kg/kW/hr -> convert to seconds
        self.sfc = 0.3 / 3600 

    def get_fuel_flow_kg_s(self, power_kw):
        """Returns fuel flow rate for a given power output."""
        # Prevent negative fuel flow (engine braking not modeled)
        effective_power = max(0, power_kw)
        return effective_power * self.sfc

    def get_weight_kg(self):
        """Calculates engine weight based on max power."""
        return self.max_power_kw / self.power_to_weight

def calculate_system_weight(engine_kw, battery_kwh, payload_kg=200, empty_airframe_kg=450):
    """Calculates total UAV weight to check against MTOW constraint."""
    engine = TurboshaftEngine(engine_kw)
    battery = Battery(battery_kwh, engine_kw) # Max battery power approximated to engine power
    
    total_weight = empty_airframe_kg + payload_kg + engine.get_weight_kg() + battery.get_weight_kg()
    return total_weight