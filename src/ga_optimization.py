# src/ga_optimization.py
import numpy as np
import random
from deap import base, creator, tools, algorithms
from src.config import UAV_SPECS, MISSION_PROFILE
from src.physics import calculate_shaft_power_req
from src.components import Battery, TurboshaftEngine, calculate_system_weight
from src.optimization import FuzzyECMS

# --- 1. Helper: Power Split Optimizer (FIXED MATH) ---
def optimize_power_split(p_req_elec_kw, battery, engine, s_factor, dt_s):
    """
    Finds the optimal Generator power to minimize ECMS cost.
    Cost is calculated in total mass (kg) consumed over the time step dt_s.
    """
    best_cost = float('inf')
    best_p_gen = 0.0
    best_p_batt = 0.0
    best_fuel_flow = 0.0
    
    # Discretize engine power search space (0 to max in 2kW steps for speed)
    for p_gen in np.arange(0, engine.max_power_kw + 1, 2.0):
        p_batt = p_req_elec_kw - p_gen
        
        # Constraints
        if p_batt > battery.max_power_kw or p_batt < -battery.max_power_kw:
            continue
            
        # Instantaneous fuel flow rate (kg/s)
        fuel_flow = engine.get_fuel_flow_kg_s(p_gen)
        
        # 1. Actual engine fuel burned during this time step (kg)
        fuel_burned_kg = fuel_flow * dt_s
        
        # 2. Actual battery energy consumed during this time step (kWh)
        batt_energy_kwh = p_batt * (dt_s / 3600.0)
        
        # 3. Virtual battery equivalent fuel mass (kg)
        # (Assuming 0.08 kg of jet fuel per kWh of equivalent energy)
        batt_eq_fuel_kg = batt_energy_kwh * 0.08
        
        # Total ECMS Cost (kg) = Engine Fuel + (Equivalence Factor * Battery Eq Fuel)
        cost = fuel_burned_kg + (s_factor * batt_eq_fuel_kg)
        
        if cost < best_cost:
            best_cost = cost
            best_p_gen = p_gen
            best_p_batt = p_batt
            best_fuel_flow = fuel_flow # Keep returning rate (kg/s) for state updates
            
    return best_p_gen, best_p_batt, best_fuel_flow

# --- 2. The Simulation Engine (Fitness Evaluator) ---
# src/ga_optimization.py (Update the run_simulation function)

def run_simulation(genes, record_logs=False):
    """
    Runs a full flight simulation based on GA genes.
    Returns: endurance_time_s (float), logs (dict)
    """
    # Unpack genes
    eng_kw, batt_kwh, soc_low, soc_high, s_min, s_max = genes
    
    # Initialize components
    engine = TurboshaftEngine(max_power_kw=eng_kw)
    battery = Battery(capacity_kwh=batt_kwh, max_power_kw=eng_kw)
    fuzzy_ecms = FuzzyECMS(soc_low, soc_high, s_min, s_max)
    
    # Hard Constraint: Check MTOW
    system_weight = calculate_system_weight(engine_kw=eng_kw, battery_kwh=batt_kwh)
    initial_fuel_kg = 150.0
    total_weight = system_weight + initial_fuel_kg # MUST include fuel!
    
    if total_weight > UAV_SPECS["mtow_kg"]:
        return 0.0, {"error": "MTOW Exceeded"} # Penalize heavily
    
    # Simulation State
    current_weight = total_weight
    current_fuel = initial_fuel_kg
    time_s = 0
    dt = 60.0 # 60-second time step speed optimization
    
    # Logs for visualization
    logs = {"time": [], "soc": [], "p_gen": [], "p_batt": [], "fuel": [], "weight": []}

    # Simple Continuous Loiter Mission
    altitude = UAV_SPECS["cruise_altitude_m"]
    speed = UAV_SPECS["cruise_speed_mps"]
    
    # Safety break (max 24 hours)
    while time_s < 86400 and current_fuel > 0.1 and battery.soc > 0.05:
        # 1. Calculate Power Demand
        p_shaft_req = calculate_shaft_power_req(speed, altitude, current_weight)
        p_elec_req = (p_shaft_req / UAV_SPECS["motor_efficiency"]) / 1000 # Convert to kW
        
        # 2. Get Fuzzy Equivalence Factor
        s_factor = fuzzy_ecms.compute_s_factor(battery.soc, p_elec_req, engine.max_power_kw)
        
        # 3. Optimize Power Split
        p_gen, p_batt, fuel_flow = optimize_power_split(p_elec_req, battery, engine, s_factor, dt)
        
        # 4. Update States
        battery.update(power_kw=p_batt, dt_s=dt)
        fuel_burned = fuel_flow * dt
        current_fuel -= fuel_burned
        current_weight -= fuel_burned
        
        # 5. Log data
        if record_logs and time_s % 300 == 0: # Log every 5 minutes
            logs["time"].append(time_s / 3600) # Store in hours for easier plotting
            logs["soc"].append(battery.soc)
            logs["p_gen"].append(p_gen)
            logs["p_batt"].append(p_batt)
            logs["fuel"].append(current_fuel)
            logs["weight"].append(current_weight)
            
        time_s += dt

    return time_s, logs

# --- 3. The Genetic Algorithm (Outer Loop) ---
# src/ga_optimization.py (Replace just the run_ga function)

def run_ga():
    # Safe DEAP class cleanup to prevent duplicate definition crashes
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize Endurance
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Gene definitions [min, max]
    toolbox.register("attr_eng", random.uniform, 30, 100)
    toolbox.register("attr_batt", random.uniform, 5, 50)
    toolbox.register("attr_soc_low", random.uniform, 0.1, 0.4)
    toolbox.register("attr_soc_high", random.uniform, 0.6, 0.9)
    toolbox.register("attr_s_min", random.uniform, 1.0, 3.0)
    toolbox.register("attr_s_max", random.uniform, 3.0, 5.5)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_eng, toolbox.attr_batt, toolbox.attr_soc_low, 
                      toolbox.attr_soc_high, toolbox.attr_s_min, toolbox.attr_s_max), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        # Add a hard penalty if genes are somehow out of bounds
        eng, batt, soc_l, soc_h, s_mn, s_mx = individual
        if not (30 <= eng <= 100 and 5 <= batt <= 50 and 0.1 <= soc_l <= 0.4 and 
                0.6 <= soc_h <= 0.9 and 1.0 <= s_mn <= 3.0 and 3.0 <= s_mx <= 5.5):
            return (0.0,) # Instant death for invalid genes
            
        endurance, _ = run_simulation(individual)
        return (endurance,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    # FIX: Use Polynomial Bounded Mutation to strictly enforce physical limits
    # eta=20.0 controls how tightly it searches near the bounds
    toolbox.register("mutate", tools.mutPolynomialBounded, 
                     eta=20.0, 
                     low=[30, 5, 0.1, 0.6, 1.0, 3.0], 
                     up=[100, 50, 0.4, 0.9, 3.0, 5.5], 
                     indpb=0.2)
                     
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=30) # Population size 30
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    print("Starting GA Optimization...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
    
    best_genes = hof[0]
    print(f"\nBest Endurance: {hof[0].fitness.values[0]/3600:.2f} hours")
    print(f"Optimal Engine: {best_genes[0]:.2f} kW, Battery: {best_genes[1]:.2f} kWh")
    
    return best_genes

if __name__ == "__main__":
    # Run a quick single test first
    print("Testing Single Simulation...")
    test_genes = [60, 20, 0.3, 0.7, 2.0, 4.0]
    endurance, logs = run_simulation(test_genes, record_logs=True)
    print(f"Test Endurance: {endurance/3600:.2f} hours")
    
    # Run the GA
    print("\nStarting GA Optimization...")
    best = run_ga()
    
    print("\nRunning Final Best Simulation for Logs...")
    best_endurance, best_logs = run_simulation(best, record_logs=True)
    print(f"Final Best Endurance: {best_endurance/3600:.2f} hours")
    print("Logs ready for visualization!")
    
    
    print(f"Final Best Endurance: {best_endurance/3600:.2f} hours")
    print("Logs ready for visualization!")
    
    # --- ADD THIS: Export logs to CSV ---
    import pandas as pd
    df = pd.DataFrame(best_logs)
    df.to_csv("data/best_simulation_results.csv", index=False)
    print("Logs successfully saved to data/best_simulation_results.csv")