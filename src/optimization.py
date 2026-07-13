# src/optimization.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyECMS:
    """
    Dynamic Fuzzy Logic Controller for Adaptive ECMS.
    Tunable via Genetic Algorithm genes: soc_low, soc_high, s_min, s_max.
    """
    def __init__(self, soc_low_thresh=0.3, soc_high_thresh=0.7, s_min=2.0, s_max=4.0):
        # 1. Strict Input Validation & Sorting Check
        # Convert to float to prevent integer division issues
        soc_low_thresh = float(soc_low_thresh)
        soc_high_thresh = float(soc_high_thresh)
        s_min = float(s_min)
        s_max = float(s_max)

        # Swap if inverted to prevent membership function inversion
        if soc_low_thresh > soc_high_thresh:
            soc_low_thresh, soc_high_thresh = soc_high_thresh, soc_low_thresh
        if s_min > s_max:
            s_min, s_max = s_max, s_min

        # Clamp to safe absolute boundaries
        self.soc_low = np.clip(soc_low_thresh, 0.01, 0.99)
        self.soc_high = np.clip(soc_high_thresh, 0.02, 0.995)
        self.s_min = np.clip(s_min, 1.0, 5.9)
        self.s_max = np.clip(s_max, self.s_min + 0.1, 6.0) # Ensure s_max is strictly > s_min

        # 2. Define Antecedents and Consequent
        self.soc = ctrl.Antecedent(np.arange(0.0, 1.01, 0.01), 'soc')
        self.p_req_norm = ctrl.Antecedent(np.arange(0.0, 1.01, 0.01), 'p_req_norm')
        self.s_factor = ctrl.Consequent(np.arange(1.0, 6.01, 0.01), 's_factor')

        # 3. Dynamic Membership Functions
        # SOC Membership Functions (Triangle peaks at boundaries)
        self.soc['low'] = fuzz.trimf(self.soc.universe, [0.0, 0.0, self.soc_low])
        self.soc['medium'] = fuzz.trimf(self.soc.universe, [self.soc_low, (self.soc_low + self.soc_high) / 2.0, self.soc_high])
        self.soc['high'] = fuzz.trimf(self.soc.universe, [self.soc_high, 1.0, 1.0])

        # Power Demand Membership Functions
        self.p_req_norm.automf(3, names=['low', 'medium', 'high'])

        # S_Factor Membership Functions (Tessellated across 1.0 to 6.0)
        mid_s = (self.s_min + self.s_max) / 2.0
        self.s_factor['low'] = fuzz.trimf(self.s_factor.universe, [1.0, self.s_min, mid_s])
        self.s_factor['medium'] = fuzz.trimf(self.s_factor.universe, [self.s_min, mid_s, self.s_max])
        self.s_factor['high'] = fuzz.trimf(self.s_factor.universe, [mid_s, self.s_max, 6.0])

        # 4. Fuzzy Rules (3 requested + 4 robustness rules to cover 3x3 state space)
        rule1 = ctrl.Rule(self.soc['low'], self.s_factor['high'])                                      # Save battery
        rule2 = ctrl.Rule(self.soc['high'] & self.p_req_norm['low'], self.s_factor['low'])             # Maximize battery usage
        rule3 = ctrl.Rule(self.soc['medium'] & self.p_req_norm['high'], self.s_factor['medium'])       # Balanced approach
        
        # Additional rules to prevent sparse rulebase defuzzification errors
        rule4 = ctrl.Rule(self.soc['high'] & self.p_req_norm['high'], self.s_factor['low'])            # High battery, high demand -> use battery
        rule5 = ctrl.Rule(self.soc['high'] & self.p_req_norm['medium'], self.s_factor['low'])          # High battery, medium demand -> use battery
        rule6 = ctrl.Rule(self.soc['medium'] & self.p_req_norm['medium'], self.s_factor['medium'])     # Medium state -> balanced
        rule7 = ctrl.Rule(self.soc['medium'] & self.p_req_norm['low'], self.s_factor['medium'])        # Medium battery, low demand -> balanced

        # 5. Create Control System
        self.ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])

    def compute_s_factor(self, current_soc, current_p_req, max_power):
        """
        Calculates the crisp equivalence factor (s) based on current flight state.
        Safely handles inputs and catches defuzzification runtime errors.
        """
        # Instantiate a fresh simulation per call to prevent state bleeding
        sim = ctrl.ControlSystemSimulation(self.ctrl_system)

        # Safe input clipping and normalization
        safe_soc = float(np.clip(current_soc, 0.0, 1.0))
        if max_power <= 0:
            safe_p_norm = 0.0
        else:
            safe_p_norm = float(np.clip(current_p_req / max_power, 0.0, 1.0))

        sim.input['soc'] = safe_soc
        sim.input['p_req_norm'] = safe_p_norm

        try:
            sim.compute()
            result = float(sim.output['s_factor'])
            
            # Final NaN check
            if np.isnan(result):
                return (self.s_min + self.s_max) / 2.0
                
            return result
            
        except Exception as e:
            # Fallback to safe mid-range baseline value if defuzzification fails
            # (e.g., if GA genes create an impossibly sparse intersection)
            return (self.s_min + self.s_max) / 2.0

# Example usage block for testing
if __name__ == "__main__":
    # Simulate genes from GA (e.g., purposely inverted to test sorting logic)
    fuzzy_controller = FuzzyECMS(soc_low_thresh=0.8, soc_high_thresh=0.2, s_min=4.5, s_max=1.5)
    
    print(f"Corrected Gene Bounds -> Low: {fuzzy_controller.soc_low}, High: {fuzzy_controller.soc_high}")
    print(f"Corrected S Bounds -> Min: {fuzzy_controller.s_min}, Max: {fuzzy_controller.s_max}\n")
    
    # Test cases
    s_high_soc_high_power = fuzzy_controller.compute_s_factor(current_soc=0.9, current_p_req=50, max_power=60)
    print(f"SoC: 0.9, Power: 50kW/60kW -> s_factor: {s_high_soc_high_power:.2f} (Expect LOW)")
    
    s_low_soc_high_power = fuzzy_controller.compute_s_factor(current_soc=0.1, current_p_req=50, max_power=60)
    print(f"SoC: 0.1, Power: 50kW/60kW -> s_factor: {s_low_soc_high_power:.2f} (Expect HIGH)")
    
    s_mid_soc_mid_power = fuzzy_controller.compute_s_factor(current_soc=0.5, current_p_req=30, max_power=60)
    print(f"SoC: 0.5, Power: 30kW/60kW -> s_factor: {s_mid_soc_mid_power:.2f} (Expect MEDIUM)")