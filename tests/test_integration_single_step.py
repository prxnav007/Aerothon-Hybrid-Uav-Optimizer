"""One-step integration contract across the complete physics/control stack."""

import math

import pytest

from src.control.base import neutral_equivalence_factor
from src.control.power_split import solve_split
from src.models.aerodynamics import evaluate
from src.models.atmosphere import atmosphere, g0
from src.models.battery import BatteryPack
from src.models.engine import LHV_KJ_KG, Turboshaft
from src.models.powertrain import SeriesPowertrain


def test_reference_aircraft_single_step_interfaces_align() -> None:
    altitude_m = 3000.0
    speed_mps = 45.0
    mass_kg = 1000.0
    wing_area_m2 = 10.0
    aspect_ratio = 16.0
    oswald_efficiency = 0.78
    cd0 = 0.028
    propeller_efficiency = 0.85
    soc = 0.6
    dt_s = 60.0

    engine = Turboshaft(75.0)
    battery = BatteryPack(10.0)
    powertrain = SeriesPowertrain()

    atmospheric = atmosphere(altitude_m)
    assert 0.8 < atmospheric.density_kg_m3 < 1.0
    assert 0.6 < atmospheric.density_ratio < 0.9

    # The integration boundary that most needs an executable assertion: the
    # aerodynamic model receives force, not mass.
    weight_n = mass_kg * g0
    assert weight_n == pytest.approx(mass_kg * g0, rel=0.0, abs=0.0)
    assert weight_n == pytest.approx(9806.65, rel=1.0e-12)

    aerodynamic = evaluate(
        weight_n,
        atmospheric.density_kg_m3,
        speed_mps,
        wing_area_m2,
        cd0,
        aspect_ratio,
        oswald_efficiency,
        propeller_efficiency,
    )
    assert 0.5 < aerodynamic.lift_coefficient < 1.5
    assert 100.0 < aerodynamic.drag_n < 1500.0
    assert 1.0 < aerodynamic.shaft_power_w / 1000.0 < 100.0
    assert not aerodynamic.power_off

    shaft_power_kw = aerodynamic.shaft_power_w / 1000.0
    bus_demand_kw = float(powertrain.bus_power_required(shaft_power_kw))
    assert shaft_power_kw < bus_demand_kw < 100.0

    # A split-dependent SFC cannot be known before solving the split.  The
    # neutral factor therefore uses the deterministic engine-only operating
    # point that would meet this timestep's bus demand.
    reference_engine_kw = float(powertrain.engine_power_for_bus(bus_demand_kw))
    reference_engine_state = engine.operate(
        reference_engine_kw, atmospheric.density_ratio
    )
    neutral_s = neutral_equivalence_factor(
        reference_engine_state.sfc_kg_kwh,
        powertrain.source_chain_efficiency,
        LHV_KJ_KG,
    )
    assert 0.0 < reference_engine_state.load_fraction < 1.0
    assert 0.4 < reference_engine_state.sfc_kg_kwh < 1.0
    assert 1.0 < neutral_s < 20.0

    split = solve_split(
        bus_demand_kw,
        engine,
        battery,
        powertrain,
        5.0,
        soc,
        atmospheric.density_ratio,
        dt_s,
    )
    assert split.feasible
    assert all(
        math.isfinite(value)
        for value in (
            split.engine_shaft_kw,
            split.bus_from_engine_kw,
            split.battery_bus_kw,
            split.battery_internal_kw,
            split.fuel_flow_kg_s,
            split.hamiltonian_kg_s,
        )
    )
    assert split.bus_from_engine_kw + split.battery_bus_kw == pytest.approx(
        bus_demand_kw, abs=1.0e-9
    )

    engine_state = engine.operate(split.engine_shaft_kw, atmospheric.density_ratio)
    battery_state = battery.step(soc, split.battery_bus_kw, dt_s)
    fuel_burned_kg = engine_state.fuel_flow_kg_s * dt_s
    assert engine_state.delivered_kw == pytest.approx(split.engine_shaft_kw)
    assert engine_state.fuel_flow_kg_s > 0.0
    assert 0.0 < fuel_burned_kg < 1.0
    assert battery_state.power_kw == pytest.approx(split.battery_bus_kw, abs=1.0e-9)

    # At s=5 the default no-shutdown engine charges: s is above the marginal
    # switching factor even though it is below the average-SFC neutral factor.
    assert split.battery_bus_kw < 0.0
    assert battery_state.soc > soc

    # Exercise the requested discharge sign convention separately; lowering s
    # makes pack energy cheap and produces a positive battery command.
    discharge_split = solve_split(
        bus_demand_kw,
        engine,
        battery,
        powertrain,
        0.5,
        soc,
        atmospheric.density_ratio,
        dt_s,
    )
    discharge_state = battery.step(soc, discharge_split.battery_bus_kw, dt_s)
    assert discharge_split.battery_bus_kw > 0.0
    assert discharge_state.soc < soc

    print("single-step integration trace")
    print(f"altitude_m={altitude_m:.6f}")
    print(f"density_kg_m3={atmospheric.density_kg_m3:.12f}")
    print(f"density_ratio={atmospheric.density_ratio:.12f}")
    print(f"mass_kg={mass_kg:.6f}")
    print(f"weight_n={weight_n:.6f}")
    print(f"speed_mps={speed_mps:.6f}")
    print(f"lift_coefficient={aerodynamic.lift_coefficient:.12f}")
    print(f"drag_n={aerodynamic.drag_n:.12f}")
    print(f"shaft_power_kw={shaft_power_kw:.12f}")
    print(f"bus_demand_kw={bus_demand_kw:.12f}")
    print(f"neutral_reference_engine_kw={reference_engine_state.delivered_kw:.12f}")
    print(f"reference_sfc_kg_kwh={reference_engine_state.sfc_kg_kwh:.12f}")
    print(f"neutral_s={neutral_s:.12f}")
    print(f"equivalence_factor={5.0:.12f}")
    print(f"engine_shaft_kw={split.engine_shaft_kw:.12f}")
    print(f"engine_load_fraction={engine_state.load_fraction:.12f}")
    print(f"actual_sfc_kg_kwh={engine_state.sfc_kg_kwh:.12f}")
    print(f"engine_thermal_efficiency={engine_state.thermal_efficiency:.12f}")
    print(f"bus_from_engine_kw={split.bus_from_engine_kw:.12f}")
    print(f"battery_bus_kw={split.battery_bus_kw:.12f}")
    print(f"battery_internal_kw={split.battery_internal_kw:.12f}")
    print(f"hamiltonian_kg_s={split.hamiltonian_kg_s:.12f}")
    print(
        "bus_balance_residual_kw="
        f"{split.bus_from_engine_kw + split.battery_bus_kw - bus_demand_kw:.12f}"
    )
    print(f"fuel_flow_kg_s={engine_state.fuel_flow_kg_s:.12f}")
    print(f"fuel_burned_kg={fuel_burned_kg:.12f}")
    print(f"soc_initial={soc:.12f}")
    print(f"soc_final={battery_state.soc:.12f}")
    print(f"battery_current_a={battery_state.current_a:.12f}")
    print(f"battery_terminal_voltage_v={battery_state.terminal_voltage_v:.12f}")
    print(f"battery_ohmic_loss_kw={battery_state.ohmic_loss_kw:.12f}")
    print(f"discharge_check_bus_kw={discharge_split.battery_bus_kw:.12f}")
    print(f"discharge_check_soc_final={discharge_state.soc:.12f}")
