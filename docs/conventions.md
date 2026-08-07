# Power and energy conventions

This is the measurement-point ledger for the series-hybrid model. It records
what each power name means; it does not change any model or controller
behaviour. SI is used throughout, except that the aerodynamic layer reports W
while the propulsion, control, simulation, and cycle-analysis layers report kW.

The physical path is:

`fuel → engine shaft → generator → rectifier → DC bus → cabling → inverter → motor → propeller shaft → thrust`

The battery connects at the DC bus. Its one signed convention is positive when
the pack supplies the bus and negative when the bus charges the pack.

## Quantity ledger

| Code quantity | Measurement point | Sign convention | Unit |
|---|---|---|---|
| `aerodynamics.thrust_power_required`, `AeroState.thrust_power_w` | Flight-path thrust work, `D·V + W·ROC`, downstream of the propeller | Positive when propulsion must do work; may be negative when gravity more than covers drag in descent | W |
| `aerodynamics.shaft_power_required`, `AeroState.shaft_power_w` | Mechanical input to the propeller | Non-negative demand; clamped to zero when thrust power is non-positive | W |
| `rate_of_climb(..., shaft_power_available_w)` | Mechanical power available at the propeller shaft | Positive supply | W |
| `Turboshaft.rated_power_kw`, `max_power_kw`, `idle_power_kw` | Turboshaft output shaft | Non-negative capability | kW |
| `EngineState.commanded_kw`, `delivered_kw` | Turboshaft output shaft | Non-negative command/supply | kW |
| `BatteryPack.max_discharge_kw`, `max_charge_kw` | Legacy C-rate-derived bus-power proxies, retained as named constants in both modes | Positive capability magnitude | kW |
| `BatteryPack.available_discharge_kw`, `available_charge_kw` | Battery terminals at the DC bus; legacy mode uses the proxies, physical mode uses explicit current and terminal-voltage limits | Positive capability magnitude, although actual charge power is negative | kW |
| `BatteryState.commanded_kw`, `power_kw` | Battery terminals at the DC bus | Positive discharge, negative charge | kW |
| `BatteryState.current_a` | Through the Rint pack | Positive discharge, negative charge | A |
| `BatteryState.terminal_voltage_v` | Legacy: start-of-step terminal voltage; physical: constant-current step-average terminal voltage | Below OCV on discharge, above OCV on charge | V |
| `BatteryState.constraint_terminal_voltage_v` | Physical-mode worst endpoint used for the terminal constraint; diagnostic only in legacy mode | Charge ceiling or discharge floor convention | V |
| `BatteryState.ohmic_loss_kw`, `BatteryPack.ohmic_loss_kw` | Internal resistance | Always non-negative dissipation | kW |
| `BatteryPack.stored_energy_kwh` | Open-circuit energy obtained by integrating the linear OCV curve from zero SoC | Non-negative state quantity | kWh |
| `SplitDecision.battery_internal_kw`, logged `battery_internal_kw` | Open-circuit source, `V_oc·I` | Positive stored-energy withdrawal, negative stored-energy addition | kW |
| `SeriesPowertrain.bus_power_required`, `PowertrainState.bus_demand_kw` | DC bus upstream of cabling/inverter/motor | Non-negative load | kW |
| `SeriesPowertrain.bus_power_from_engine`, `PowertrainState.bus_from_engine_kw` | Rectifier output at the DC bus | Non-negative supply | kW |
| `SeriesPowertrain.engine_power_for_bus` | Turboshaft output shaft required for a requested bus supply | Non-negative demand on the engine shaft | kW |
| `SeriesPowertrain.shaft_power_from_bus`, `PowertrainState.shaft_demand_kw` | Propeller-shaft output of cabling/inverter/motor | Non-negative output/demand | kW |
| `PowertrainState.battery_bus_kw` | Battery connection at the DC bus | Positive discharge, negative charge | kW |
| `PowertrainState.bus_residual_kw` | DC-bus balance, `engine bus + battery bus − bus demand` | Positive excess, negative shortfall | kW |
| `PowertrainState.source_losses_kw` | Generator plus rectifier, engine shaft to DC bus | Non-negative in supported forward flow | kW |
| `PowertrainState.demand_losses_kw` | Cabling plus inverter plus motor, DC bus to propeller shaft | Non-negative in supported forward flow | kW |
| `PowertrainState.total_losses_kw` | Source and demand conversion chains combined | Non-negative in supported forward flow | kW |
| `SeriesPowertrain.peak_bus_power` | Peak DC-bus demand used to size inverter and motor | Positive rating | kW |
| `ControlContext.bus_demand_kw` | DC-bus load seen by the controller | Non-negative demand | kW |
| `ControlContext.max_bus_kw` | Engine maximum bus supply plus available battery discharge | Non-negative capability | kW |
| `SplitDecision.engine_shaft_kw` | Selected turboshaft output shaft | Non-negative supply | kW |
| `SplitDecision.bus_from_engine_kw` | Selected rectifier output at the bus | Non-negative supply | kW |
| `SplitDecision.battery_bus_kw` | Selected battery terminal power | Positive discharge, negative charge | kW |
| `TimeStep.shaft_power_kw`, `bus_demand_kw`, `engine_shaft_kw`, `bus_from_engine_kw`, `battery_bus_kw` | Same points as their model counterparts above | Same conventions as above | kW |
| `TimeStep.thrust_power_kw` | Non-negative propulsion-delivered thrust work; the simulator records zero during a power-off descent | Non-negative output | kW |
| `TimeStep.engine_thermal_loss_kw` | Fuel chemical power less engine shaft power | Non-negative loss for the present calibration | kW |
| `TimeStep.source_losses_kw`, `demand_losses_kw`, `propeller_losses_kw`, `battery_ohmic_loss_kw` | Named conversion segment | Non-negative loss | kW |
| `TimeStep.battery_stored_energy_change_kwh` | Endpoint change from pre-step to post-step SoC using the integrated OCV curve | Positive charge, negative discharge | kWh |
| `MissionResult.peak_bus_kw`, `peak_engine_kw` | Greatest logged bus demand and engine shaft output | Non-negative magnitude | kW |
| `SeriesPowertrain.system_efficiency(..., fuel_chemical_kw, battery_bus_kw)` | Whole-system metric input | Chemical power is positive input; only positive battery discharge enters the denominator | kW |
| `ConstraintCase.battery_boost_kw` | Battery supply injected at the DC bus | Positive discharge capability | kW |
| `power_loading_required` result | Installed sea-level engine shaft rating divided by weight | Positive requirement | W/N |
| `build_mass_budget.engine_kw` | Engine shaft nameplate used for engine mass | Positive rating | kW |
| `build_mass_budget.peak_bus_kw` | Bus nameplate used for inverter/motor mass | Positive rating | kW |
| `generator_mass(power_kw)`, `rectifier_mass(power_kw)` | Numerical proxy based on engine shaft nameplate, not the actual electrical flow at their terminals | Positive rating | kW |
| `cycle_model.demand_bus_kw` (`D`) | Constant DC-bus demand over an analytical cycle | Positive load | kW |
| `cycle_model.engine_on_kw` (`x`) | Turboshaft output while ON | Positive supply | kW |
| `cycle_model.charge_limit_bus_kw`, `discharge_limit_bus_kw` | Battery terminal power limits | Positive magnitudes | kW |
| `cycle_model.battery_usable_bus_kwh` | Energy the pack can deliver at the DC bus at the stated discharge efficiency | Positive usable energy | kWh |

The component-mass functions use rating proxies rather than measured flows.
In particular, generator and rectifier mass are both sized with the numerical
engine shaft rating, while inverter and motor mass use peak bus power. That is
the explicit M-04 sizing convention; the generator/rectifier number is not a
claim that shaft and electrical nameplate kW are the same physical quantity.

## Battery boundary and loss identity

The Rint model uses, for either current direction,

`P_bus = V_oc·I − I²R = P_internal − P_ohmic`.

Therefore `P_internal = P_bus + P_ohmic` in both directions. On discharge,
`P_internal > P_bus > 0`. On charge, both powers are negative and
`|P_bus| > |P_internal|`; the bus supplies stored energy plus the ohmic loss.
`SplitDecision.battery_internal_kw` and the simulator log compute `V_oc·I`, so
they obey this relation for positive discharge and negative charge. Legacy
mode uses start-of-step OCV. Physical mode uses the midpoint OCV over a
constant-current step; because the implemented OCV curve is linear, its
integral is exact and the endpoint stored-energy change equals the internal-
power ledger to floating-point roundoff.

`available_discharge_kw` and `available_charge_kw` are bus-terminal limits.
With `dt_s`, they are further limited to what can be held for the complete
step; without it, they report only the instantaneous rate ceiling.

### Legacy C-rate ambiguity retained for baseline preservation

The present implementation computes `C × capacity_kWh` and interprets the
result as a bus-power cap. A physical cell C-rate normally constrains current
relative to Ah capacity, whose bus power varies with OCV and terminal voltage.
The names therefore suggest a current limit while the implementation and B-04
define a terminal-power proxy. Its use is internally consistent—availability,
the split solver, and `BatteryPack.step` all enforce the same bus-side number—
but its physical interpretation is ambiguous. This milestone does not change
it because doing so would move the baseline.

The configured 300 V and 400 V values are OCV-curve endpoints, not terminal
cutoffs. Physical mode therefore names its distinct explicit inputs
`terminal_voltage_min_v` and `terminal_voltage_max_v`; it never reinterprets
the legacy OCV fields. No physical-limit values are defaults, so selecting
physical mode without all five explicit inputs raises `ValueError`.

## Efficiency direction and application

| Stage | Forward direction | Forward operation | Inverse operation |
|---|---|---|---|
| Generator | Engine shaft → generator terminals | Multiply by `eta_generator` | Divide by it |
| Rectifier | Generator terminals → DC bus | Multiply by `eta_rectifier` | Divide by it |
| Cabling | DC bus → inverter input | Multiply by `eta_cabling` | Divide by it |
| Inverter | Inverter input → motor terminals | Multiply by `eta_inverter` | Divide by it |
| Motor | Motor terminals → propeller shaft | Multiply by `eta_motor` | Divide by it |
| Propeller | Propeller shaft → thrust work | Multiply by `eta_prop` conceptually; code obtains shaft demand by dividing positive thrust demand by it | Not part of `SeriesPowertrain` |

The constant-efficiency path applies every stage once. Generator and rectifier
appear only on the engine-source path; battery discharge enters downstream of
them. Cabling, inverter, and motor appear once for both engine and battery bus
energy. Propeller efficiency appears only in `aerodynamics.py`, so it is neither
omitted nor duplicated in `powertrain.py`.

There is one ambiguity in the optional O-11 load-dependent branch:
`rated_engine_kw` is documented as the engine/generator input-side rating, but
the generic stage-loss calibration treats its `rated_kw` argument as stage
output. The same number is passed through generator and rectifier despite their
different terminal powers. The default constant-efficiency branch is
unaffected. This is recorded for a later behaviour-changing milestone.

## Round-trip efficiency

No single calibrated round-trip efficiency controls the current simulator.

- `BatteryPack.round_trip_efficiency(power_kw, soc)` is a diagnostic terminal-
  voltage ratio at equal charge/discharge current magnitude. For the 10 kWh
  pack at 30 kW discharge and SoC 0.5 it is 0.9755064388. It is used by battery
  tests, not by dispatch or mission integration.
- `neutral_equivalence_factor(..., round_trip_efficiency=1.0)` accepts one
  scalar, but the simulator leaves the 1.0 correction disabled.
- The cycle analysis accepts `eta_charge` and `eta_discharge` separately. At
  the actual 10 kW charge and 33.631429 kW discharge powers, evaluated at SoC
  0.55 on the 10 kWh pack, they are 0.996063706 and 0.986473886, giving
  0.982590835 round trip. Applying the old 0.9755 value to this asymmetric
  operating point would overstate cycle loss.

## Energy accounting

The simulator ledger defines stored-energy change from its discrete update as
`ΔE_stored = −Σ(V_oc·I·Δt)`. The checked identity is

`fuel chemical = thrust work + engine thermal loss + source-chain loss + demand-chain loss + propeller loss + battery ohmic loss + ΔE_stored`.

Propeller loss must be explicit if “propulsive work” means thrust work. The
milestone prompt's written list omitted it even though the aerodynamic model
applies propeller efficiency. With endpoint stored energy inferred from SoC,
`mission_energy_balance` has a 1.405×10⁻³ relative residual at the 60 s step on
the battery-preferring regression mission, falling to 7.022×10⁻⁴ at 30 s and
3.533×10⁻⁴ at 15 s. This first-order scaling locates it in the documented
start-of-step-OCV explicit-Euler bias rather than an omitted flow. Using the
discrete `−V_oc,start·I·Δt` stored-energy update closes at 5.67×10⁻¹⁶ relative.
The analytical charge-sustaining cycle closes below 1×10⁻¹⁴ relative.

## Terminal-energy optimisation

`ledger_residual_kwh` is an accounting diagnostic for one trajectory.
`terminal_target_residual_kwh` is the signed miss against a requested comparison target. The
ledger roundoff tolerance is never used as a discrete-policy optimisation tolerance.

Opposite-energy deterministic policies define an endpoint-energy interval and two raw policy fuel
values. They do not define a fuel or optimality interval without a separate proof. A valid lower
bound is the maximised Lagrangian dual value; a valid upper bound is the fuel of a deterministic
policy feasible at the same terminal target. Current finite-horizon bounds apply to the explicitly
snapped discrete SoC/action model, while continuous replay is a separately labelled diagnostic.

## Feasibility metric labels

The discharge-feasibility study uses two unrelated percentages. The
"sampled design-grid point removal fraction" counts altitude–mass samples
rejected by the discharge condition; it is neither an area measure nor a
mission measure. The "logged loiter time power-feasible fraction" integrates
logged loiter duration at demand no greater than the discharge proxy. CSV
headers, fixtures, captions and prose must retain those full labels.

## Frozen baseline

No pre-existing reproduction command was present: `run.sh`, the configs, and
README were empty. The executable regression command added by this milestone
is:

```bash
PYTHONDONTWRITEBYTECODE=1 env/bin/pytest -q -s tests/test_baseline_regression.py
```

The pre-edit capture was 56,182.854870286 s (15.606348575 h), 283.167275528 kg
fuel used, final/minimum SoC 0.102409322/0.050005272, 180 restarts, and
`fuel_reserve` termination. The pre-edit full suite command was
`PYTHONDONTWRITEBYTECODE=1 env/bin/pytest -q`, with 812 passing tests. The exact
values are stored in `tests/fixtures/milestone1_baseline.json`.

That fixture now also pins the interpretation of the 180 restarts: the controller
is the stateless `PIECMS(kp=5, soc_ref=0.6, s0_ratio=1)`, the engine permits
shutdown with a 15% floor and zero restart fuel, and no minimum ON/OFF dwell or
start transient is active. The count is therefore a free-transition baseline,
not evidence that the later transition constraints have been exercised.
