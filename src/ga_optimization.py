# src/ga_optimization.py
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from src.config import UAV_SPECS, MISSION_PROFILE, MISSION_SETTINGS
from src.physics import calculate_shaft_power_req
from src.components import Battery, TurboshaftEngine, calculate_system_weight
from src.optimization import FuzzyECMS

# Reference (non-optimized) configuration used to quantify endurance improvement:
# the problem statement's preferred ~60 kW turboshaft with a mid-size battery and
# untuned (hand-picked) fuzzy ECMS parameters.
BASELINE_GENES = [60.0, 30.0, 0.3, 0.7, 3.0, 5.0]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# --- 1. Helper: Power Split Optimizer ---
def optimize_power_split(p_req_elec_kw, battery, engine, s_factor, dt_s, max_discharge_kw=None):
    """
    Finds the optimal Generator power to minimize ECMS cost.
    Cost is calculated in total mass (kg) consumed over the time step dt_s.
    max_discharge_kw caps battery discharge (e.g. 0 when SoC is at the safety cutoff).
    """
    if max_discharge_kw is None:
        max_discharge_kw = battery.max_power_kw

    best_cost = float('inf')
    best_p_gen = 0.0
    best_p_batt = 0.0
    best_fuel_flow = 0.0

    # Discretize engine power search space (0 to max in 2kW steps for speed)
    for p_gen in np.arange(0, engine.max_power_kw + 1, 2.0):
        p_batt = p_req_elec_kw - p_gen

        # Constraints: discharge limit (safety-aware) and charge limit
        if p_batt > max_discharge_kw or p_batt < -battery.max_power_kw:
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
            best_fuel_flow = fuel_flow  # Keep returning rate (kg/s) for state updates

    return best_p_gen, best_p_batt, best_fuel_flow

# --- 2. The Simulation Engine (Fitness Evaluator) ---
def run_simulation(genes, record_logs=False, dt=60.0):
    """
    Flies the full mission profile (takeoff -> climb -> cruise -> loiter -> descent -> landing).
    The loiter phase is extended until the fuel reserve / battery cutoff is reached, so the
    returned mission time IS the endurance being maximized.

    Fuel policy: the tank is filled up to MTOW (fuel = MTOW - dry system weight), so the GA
    trades engine/battery mass directly against fuel mass.

    Returns: endurance_time_s (float), logs (dict of lists + logs["meta"] dict)
    """
    eng_kw, batt_kwh, soc_low, soc_high, s_min, s_max = genes

    engine = TurboshaftEngine(max_power_kw=eng_kw)
    battery = Battery(capacity_kwh=batt_kwh, max_power_kw=eng_kw)
    fuzzy_ecms = FuzzyECMS(soc_low, soc_high, s_min, s_max)

    lhv_kj_kg = MISSION_SETTINGS["fuel_lhv_kj_kg"]
    reserve_kg = MISSION_SETTINGS["fuel_reserve_kg"]
    soc_cutoff = MISSION_SETTINGS["soc_cutoff"]
    max_time_s = MISSION_SETTINGS["max_mission_s"]

    # Hard Constraint: MTOW. Fuel fills the remaining weight allowance.
    dry_weight = calculate_system_weight(engine_kw=eng_kw, battery_kwh=batt_kwh)
    initial_fuel_kg = UAV_SPECS["mtow_kg"] - dry_weight

    meta = {
        "genes": list(genes), "engine_kw": eng_kw, "battery_kwh": batt_kwh,
        "dry_weight_kg": dry_weight, "initial_fuel_kg": max(initial_fuel_kg, 0.0),
        "takeoff_weight_kg": min(dry_weight + max(initial_fuel_kg, 0.0), UAV_SPECS["mtow_kg"]),
        "mission_complete": False, "termination_reason": "", "endurance_s": 0.0,
    }

    if initial_fuel_kg < MISSION_SETTINGS["min_fuel_kg"]:
        meta["termination_reason"] = "Infeasible: MTOW leaves no usable fuel allowance"
        return 0.0, {"meta": meta}

    # Simulation state
    current_weight = dry_weight + initial_fuel_kg
    current_fuel = initial_fuel_kg
    time_s = 0.0
    altitude = 0.0

    log_keys = ["time_h", "phase", "altitude_m", "speed_mps", "p_req_kw", "p_gen_kw",
                "p_batt_kw", "soc", "s_factor", "fuel_kg", "fuel_used_kg", "weight_kg",
                "engine_load_pct", "motor_power_kw", "system_efficiency"]
    logs = {k: [] for k in log_keys}

    failed = False
    for phase in MISSION_PROFILE:
        phase_name = phase["phase"]
        speed = phase["speed_mps"]
        target_alt = phase["altitude_m"]
        duration = phase["duration_s"]
        is_loiter = duration is None

        climb_rate = 0.0 if is_loiter else (target_alt - altitude) / duration
        phase_elapsed = 0.0

        while True:
            # Phase exit conditions (checked BEFORE the step so the fuel reserve holds)
            if is_loiter:
                if (current_fuel <= reserve_kg or battery.soc <= soc_cutoff
                        or time_s >= max_time_s):
                    break
            elif phase_elapsed >= duration:
                break

            # 1. Power demand at the current flight condition
            p_shaft_w = calculate_shaft_power_req(speed, altitude, current_weight, climb_rate)
            p_elec_kw = (p_shaft_w / UAV_SPECS["motor_efficiency"]) / 1000.0

            # 2. Fuzzy adaptive equivalence factor
            s_factor = fuzzy_ecms.compute_s_factor(battery.soc, p_elec_kw, engine.max_power_kw)

            # 3. Optimal engine/battery power split (battery locked out below SoC cutoff)
            max_dis = battery.max_power_kw if battery.soc > soc_cutoff else 0.0
            p_gen, p_batt, fuel_flow = optimize_power_split(
                p_elec_kw, battery, engine, s_factor, dt, max_discharge_kw=max_dis)

            # Constraint: propulsion power requirement must be satisfied
            if p_gen + p_batt < p_elec_kw - 0.5:
                meta["termination_reason"] = f"Power shortfall during {phase_name}"
                failed = True
                break

            # 4. Update states
            battery.update(power_kw=p_batt, dt_s=dt)
            fuel_burned = fuel_flow * dt
            current_fuel -= fuel_burned
            current_weight -= fuel_burned

            if current_fuel <= 0 and not is_loiter and phase_name not in ("descent", "landing"):
                meta["termination_reason"] = f"Fuel exhausted during {phase_name}"
                failed = True
                break

            # 5. Log data (every step)
            if record_logs:
                fuel_power_kw = fuel_flow * lhv_kj_kg          # chemical power in (kW)
                batt_out_kw = max(p_batt, 0.0)                 # battery power in (kW)
                p_prop_kw = (p_shaft_w * UAV_SPECS["propeller_efficiency"]) / 1000.0
                denom = fuel_power_kw + batt_out_kw
                efficiency = p_prop_kw / denom if denom > 1e-6 else 0.0

                logs["time_h"].append(time_s / 3600.0)
                logs["phase"].append(phase_name)
                logs["altitude_m"].append(altitude)
                logs["speed_mps"].append(speed)
                logs["p_req_kw"].append(p_elec_kw)
                logs["p_gen_kw"].append(p_gen)
                logs["p_batt_kw"].append(p_batt)
                logs["soc"].append(battery.soc)
                logs["s_factor"].append(s_factor)
                logs["fuel_kg"].append(current_fuel)
                logs["fuel_used_kg"].append(initial_fuel_kg - current_fuel)
                logs["weight_kg"].append(current_weight)
                logs["engine_load_pct"].append(100.0 * p_gen / engine.max_power_kw)
                logs["motor_power_kw"].append(p_elec_kw)
                logs["system_efficiency"].append(efficiency)

            altitude += climb_rate * dt
            time_s += dt
            phase_elapsed += dt

        if failed:
            break
        altitude = target_alt  # remove interpolation drift at phase boundaries

    if not failed:
        meta["mission_complete"] = True
        meta["termination_reason"] = "Mission completed (loiter ended at fuel reserve/battery limit)"

    meta["endurance_s"] = time_s
    meta["endurance_h"] = time_s / 3600.0
    meta["fuel_used_kg"] = initial_fuel_kg - current_fuel
    meta["final_soc"] = battery.soc
    logs["meta"] = meta
    return time_s, logs

def logs_to_dataframe(logs):
    """Converts the time-series part of the simulation logs to a DataFrame."""
    return pd.DataFrame({k: v for k, v in logs.items() if k != "meta"})

# --- 3. The Genetic Algorithm (Outer Loop) ---
def run_ga(pop_size=30, ngen=10, save_results=True):
    # Safe DEAP class cleanup to prevent duplicate definition crashes
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize Endurance
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

    # Every evaluated design is recorded for the trade-off visualization
    eval_history = []

    def evaluate(individual):
        eng, batt, soc_l, soc_h, s_mn, s_mx = individual
        if not (30 <= eng <= 100 and 5 <= batt <= 50 and 0.1 <= soc_l <= 0.4 and
                0.6 <= soc_h <= 0.9 and 1.0 <= s_mn <= 3.0 and 3.0 <= s_mx <= 5.5):
            return (0.0,)  # Instant death for invalid genes

        endurance, logs = run_simulation(individual)
        complete = logs["meta"]["mission_complete"]
        # Designs that cannot complete all mission phases are heavily penalized,
        # but keep a gradient so the GA can climb toward feasibility.
        fitness = endurance if complete else endurance * 0.1

        eval_history.append({
            "engine_kw": eng, "battery_kwh": batt, "soc_low": soc_l, "soc_high": soc_h,
            "s_min": s_mn, "s_max": s_mx, "endurance_h": endurance / 3600.0,
            "mission_complete": complete, "fitness_h": fitness / 3600.0,
        })
        return (fitness,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    # Polynomial Bounded Mutation strictly enforces physical limits
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     eta=20.0,
                     low=[30, 5, 0.1, 0.6, 1.0, 3.0],
                     up=[100, 50, 0.4, 0.9, 3.0, 5.5],
                     indpb=0.2)

    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    print("Starting GA Optimization...")
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen,
                                       stats=stats, halloffame=hof, verbose=True)

    best_genes = list(hof[0])
    print(f"\nBest Endurance: {hof[0].fitness.values[0] / 3600:.2f} hours")
    print(f"Optimal Engine: {best_genes[0]:.2f} kW, Battery: {best_genes[1]:.2f} kWh")

    if save_results:
        DATA_DIR.mkdir(exist_ok=True)
        # GA convergence history (for the optimization-results plot)
        conv = pd.DataFrame(logbook)
        conv[["max", "avg", "min"]] = conv[["max", "avg", "min"]] / 3600.0  # hours
        conv.to_csv(DATA_DIR / "ga_convergence.csv", index=False)
        # Every evaluated design (for the design trade-off plot)
        pd.DataFrame(eval_history).to_csv(DATA_DIR / "ga_evaluations.csv", index=False)
        print(f"GA history saved to {DATA_DIR}")

    return best_genes

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Baseline (non-optimized) mission - the reference for endurance improvement
    print("Running Baseline Simulation (60 kW engine, 20 kWh battery)...")
    base_endurance, base_logs = run_simulation(BASELINE_GENES, record_logs=True)
    print(f"Baseline Endurance: {base_endurance / 3600:.2f} hours "
          f"({base_logs['meta']['termination_reason']})")
    logs_to_dataframe(base_logs).to_csv(DATA_DIR / "baseline_simulation_results.csv", index=False)

    # 2. GA optimization
    best = run_ga()

    # 3. Re-fly the optimal design with full logging for the dashboard
    print("\nRunning Final Best Simulation for Logs...")
    best_endurance, best_logs = run_simulation(best, record_logs=True)
    print(f"Final Best Endurance: {best_endurance / 3600:.2f} hours")
    logs_to_dataframe(best_logs).to_csv(DATA_DIR / "best_simulation_results.csv", index=False)

    # 4. Summary for the dashboard KPI header
    improvement = 100.0 * (best_endurance - base_endurance) / base_endurance if base_endurance else 0.0
    summary = {
        "best_genes": list(best),
        "best_meta": best_logs["meta"],
        "baseline_genes": BASELINE_GENES,
        "baseline_endurance_h": base_endurance / 3600.0,
        "endurance_improvement_pct": improvement,
    }
    with open(DATA_DIR / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Endurance improvement over baseline: {improvement:+.1f}%")
    print(f"All results saved to {DATA_DIR} - launch the dashboard with:")
    print("  streamlit run src/dashboard/app.py")
