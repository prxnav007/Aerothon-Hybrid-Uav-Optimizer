"""Pure thermostat scheduling with explicit engine and dwell state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.models.battery import BatteryPack, SOC_EPS
from src.models.engine import LHV_KJ_KG, Turboshaft
from src.models.powertrain import SeriesPowertrain

__all__ = [
    "TerminalStrategy",
    "ThermostatDecision",
    "ThermostatParameters",
    "ThermostatRegime",
    "ThermostatState",
    "select_engine_on_power",
    "thermostat_step",
]

_POWER_TOLERANCE_KW = 1.0e-7


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


class TerminalStrategy(str, Enum):
    """Future-information available to the scheduler."""

    CAUSAL = "causal"
    HORIZON_AWARE = "horizon_aware"


class ThermostatRegime(str, Enum):
    """Physical operating regime selected for one step."""

    CYCLING = "cycling"
    CONTINUOUS = "continuous_engine"
    BATTERY_ASSISTED = "battery_assisted"
    TERMINAL_DEPLETION = "terminal_depletion"


@dataclass(frozen=True)
class ThermostatParameters:
    """Explicit scheduling and transition sensitivity inputs."""

    soc_low: float
    soc_high: float
    minimum_on_time_s: float
    minimum_off_time_s: float
    restart_fuel_kg: float
    engine_on_power_kw: float | None
    terminal_strategy: TerminalStrategy | str

    def __post_init__(self) -> None:
        low = _finite("soc_low", self.soc_low)
        high = _finite("soc_high", self.soc_high)
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("soc thresholds must satisfy 0 <= soc_low < soc_high <= 1")
        for name in ("minimum_on_time_s", "minimum_off_time_s"):
            if _finite(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if _finite("restart_fuel_kg", self.restart_fuel_kg) < 0.0:
            raise ValueError("restart_fuel_kg must be non-negative")
        if self.engine_on_power_kw is not None:
            if _finite("engine_on_power_kw", self.engine_on_power_kw) <= 0.0:
                raise ValueError("engine_on_power_kw must be positive when supplied")
        try:
            strategy = TerminalStrategy(self.terminal_strategy)
        except (TypeError, ValueError) as error:
            raise ValueError("terminal_strategy must be causal or horizon_aware") from error
        object.__setattr__(self, "terminal_strategy", strategy)


@dataclass(frozen=True)
class ThermostatState:
    """All scheduler history carried explicitly between calls."""

    engine_on: bool
    elapsed_in_state_s: float
    restart_count: int = 0
    terminal_depletion: bool = False

    def __post_init__(self) -> None:
        if _finite("elapsed_in_state_s", self.elapsed_in_state_s) < 0.0:
            raise ValueError("elapsed_in_state_s must be non-negative")
        if self.restart_count < 0:
            raise ValueError("restart_count must be non-negative")


@dataclass(frozen=True)
class ThermostatDecision:
    """One scheduled split and the explicit next scheduler state."""

    engine_shaft_kw: float
    bus_from_engine_kw: float
    battery_bus_kw: float
    battery_internal_kw: float
    fuel_flow_kg_s: float
    restart_fuel_kg: float
    engine_off: bool
    regime: ThermostatRegime
    regime_reason: str
    active_constraint: str
    transitioned: bool
    next_state: ThermostatState
    feasible: bool


@dataclass(frozen=True)
class _OnPowerSelection:
    power_kw: float
    average_fuel_kg_h: float
    active_constraint: str
    cycling_beneficial: bool


def select_engine_on_power(
    *,
    demand_bus_kw: float,
    soc: float,
    sigma: float,
    dt_s: float,
    engine: Turboshaft,
    battery: BatteryPack,
    powertrain: SeriesPowertrain,
    explicit_power_kw: float | None = None,
    action_count: int = 129,
) -> _OnPowerSelection:
    """Minimise average sustaining-cycle fuel over the feasible ON interval."""
    demand = _finite("demand_bus_kw", demand_bus_kw)
    duration = _finite("dt_s", dt_s)
    if demand < 0.0 or duration <= 0.0:
        raise ValueError("demand must be non-negative and dt_s positive")
    if action_count < 3:
        raise ValueError("action_count must be at least three")
    maximum = engine.max_power_kw(sigma)
    required = float(powertrain.engine_power_for_bus(demand))
    lower = max(required, engine.idle_power_kw)
    charge_limit = battery.available_charge_kw(soc, duration)
    upper = min(
        maximum,
        float(powertrain.engine_power_for_bus(demand + charge_limit)),
    )
    if upper < lower - _POWER_TOLERANCE_KW:
        return _OnPowerSelection(required, math.inf, "no_charge_surplus", False)
    upper = max(upper, lower)
    discharge_internal = battery.internal_power_kw(demand, soc, duration)

    def average_fuel(power_kw: float) -> float:
        surplus = max(float(powertrain.bus_power_from_engine(power_kw)) - demand, 0.0)
        charge_internal = -battery.internal_power_kw(-surplus, soc, duration)
        if charge_internal <= 0.0:
            return float(engine.fuel_flow_kg_s(power_kw, sigma)) * 3600.0
        duty = discharge_internal / (discharge_internal + charge_internal)
        return duty * float(engine.fuel_flow_kg_s(power_kw, sigma)) * 3600.0

    if explicit_power_kw is None:
        probes = (lower, 0.5 * (lower + upper), upper)
        probe_values = tuple(average_fuel(power) for power in probes)
        if probe_values[0] >= probe_values[1] >= probe_values[2]:
            selected, value = upper, probe_values[2]
        elif probe_values[0] <= probe_values[1] <= probe_values[2]:
            selected, value = lower, probe_values[0]
        else:
            powers = np.linspace(lower, upper, action_count)
            values = np.asarray([average_fuel(float(power)) for power in powers])
            index = int(np.argmin(values))
            selected = float(powers[index])
            value = float(values[index])
    else:
        selected = min(max(float(explicit_power_kw), lower), upper)
        value = average_fuel(selected)
    tolerance = max(_POWER_TOLERANCE_KW, (upper - lower) / max(action_count - 1, 1))
    if abs(selected - maximum) <= tolerance and abs(selected - upper) <= tolerance:
        active = "engine_max"
    elif abs(selected - upper) <= tolerance:
        active = "battery_charge_limit"
    elif abs(selected - lower) <= tolerance:
        active = "continuous_lower_bound"
    else:
        active = "explicit_or_interior"
    continuous_fuel = float(engine.fuel_flow_kg_s(required, sigma)) * 3600.0
    return _OnPowerSelection(
        selected,
        value,
        active,
        value < continuous_fuel - 1.0e-12,
    )


def _terminal_triggered(
    parameters: ThermostatParameters,
    state: ThermostatState,
    *,
    battery: BatteryPack,
    soc: float,
    demand_bus_kw: float,
    dt_s: float,
    time_to_go_s: float | None,
    terminal_energy_target_kwh: float | None,
) -> bool:
    if parameters.terminal_strategy is TerminalStrategy.CAUSAL:
        return False
    if time_to_go_s is None or terminal_energy_target_kwh is None:
        raise ValueError("horizon-aware scheduling requires time-to-go and terminal energy")
    current = float(battery.stored_energy_kwh(soc))
    if terminal_energy_target_kwh < float(battery.stored_energy_kwh(battery.soc_min)):
        raise ValueError("terminal energy target lies below the battery cutoff")
    if state.terminal_depletion:
        return True
    discharge_kw = battery.internal_power_kw(demand_bus_kw, soc, dt_s)
    available_h = max(current - terminal_energy_target_kwh, 0.0) / max(
        discharge_kw, 1.0e-12
    )
    return time_to_go_s <= available_h * 3600.0 + dt_s


def thermostat_step(
    parameters: ThermostatParameters,
    state: ThermostatState,
    *,
    demand_bus_kw: float,
    soc: float,
    sigma: float,
    dt_s: float,
    engine: Turboshaft,
    battery: BatteryPack,
    powertrain: SeriesPowertrain,
    time_to_go_s: float | None = None,
    terminal_energy_target_kwh: float | None = None,
) -> ThermostatDecision:
    """Advance a thermostat schedule without mutating controller or plant state."""
    demand = _finite("demand_bus_kw", demand_bus_kw)
    duration = _finite("dt_s", dt_s)
    if demand < 0.0 or duration <= 0.0:
        raise ValueError("demand must be non-negative and dt_s positive")
    if not battery.soc_min <= soc <= 1.0:
        raise ValueError("soc lies outside the usable battery interval")
    engine_max = engine.max_power_kw(sigma)
    engine_bus_max = float(powertrain.bus_power_from_engine(engine_max))
    discharge_limit = battery.available_discharge_kw(soc, duration)
    off_feasible = engine.allow_shutdown and discharge_limit + _POWER_TOLERANCE_KW >= demand
    terminal = _terminal_triggered(
        parameters,
        state,
        battery=battery,
        soc=soc,
        demand_bus_kw=demand,
        dt_s=duration,
        time_to_go_s=time_to_go_s,
        terminal_energy_target_kwh=terminal_energy_target_kwh,
    )

    if engine_bus_max < demand - _POWER_TOLERANCE_KW:
        regime = ThermostatRegime.BATTERY_ASSISTED
        reason = "demand_above_engine_bus_ceiling"
        requested_on = True
        engine_kw = engine_max
        active = "engine_max"
    else:
        selection = select_engine_on_power(
            demand_bus_kw=demand,
            soc=soc,
            sigma=sigma,
            dt_s=duration,
            engine=engine,
            battery=battery,
            powertrain=powertrain,
            explicit_power_kw=parameters.engine_on_power_kw,
        )
        cycling_available = (
            off_feasible
            and selection.cycling_beneficial
            and selection.power_kw
            > float(powertrain.engine_power_for_bus(demand)) + _POWER_TOLERANCE_KW
        )
        if terminal:
            regime = ThermostatRegime.TERMINAL_DEPLETION
            reason = "preview_terminal_energy_target"
            target = float(terminal_energy_target_kwh)
            above_target = float(battery.stored_energy_kwh(soc)) > target + 1.0e-12
            requested_on = not (above_target and off_feasible)
            engine_kw = float(powertrain.engine_power_for_bus(demand))
            active = "terminal_target"
        elif not cycling_available:
            regime = ThermostatRegime.CONTINUOUS
            if not off_feasible:
                reason = "battery_cannot_carry_full_demand"
            elif not selection.cycling_beneficial:
                reason = "cycling_not_economic"
            else:
                reason = "no_feasible_charge_surplus"
            requested_on = True
            engine_kw = max(
                float(powertrain.engine_power_for_bus(demand)), engine.idle_power_kw
            )
            active = reason
        else:
            regime = ThermostatRegime.CYCLING
            reason = "soc_hysteresis"
            requested_on = state.engine_on
            engine_kw = selection.power_kw
            active = selection.active_constraint
            if state.engine_on:
                can_stop = state.elapsed_in_state_s >= parameters.minimum_on_time_s
                if soc >= parameters.soc_high - SOC_EPS and can_stop:
                    requested_on = False
            else:
                can_start = state.elapsed_in_state_s >= parameters.minimum_off_time_s
                if soc <= parameters.soc_low + SOC_EPS and can_start:
                    requested_on = True

    safety_restart = not state.engine_on and not off_feasible
    if safety_restart:
        requested_on = True
    if state.engine_on and not requested_on:
        dwell_met = state.elapsed_in_state_s >= parameters.minimum_on_time_s
        requested_on = not dwell_met
    if not state.engine_on and requested_on and not safety_restart:
        dwell_met = state.elapsed_in_state_s >= parameters.minimum_off_time_s
        if terminal and terminal_energy_target_kwh is not None:
            at_target = (
                float(battery.stored_energy_kwh(soc))
                <= terminal_energy_target_kwh + 1.0e-12
            )
        else:
            at_target = False
        requested_on = dwell_met or at_target

    if requested_on:
        if regime is ThermostatRegime.CYCLING:
            command_kw = engine_kw
        elif regime is ThermostatRegime.BATTERY_ASSISTED:
            command_kw = engine_max
        else:
            command_kw = min(max(engine_kw, engine.idle_power_kw), engine_max)
    else:
        command_kw = 0.0
    engine_state = engine.operate(command_kw, sigma)
    if not requested_on and not engine_state.shut_down:
        raise RuntimeError("thermostat OFF was converted into engine idling")
    bus_from_engine = float(powertrain.bus_power_from_engine(engine_state.delivered_kw))
    battery_kw = demand - bus_from_engine
    charge_limit = battery.available_charge_kw(soc, duration)
    feasible = (
        battery_kw <= discharge_limit + _POWER_TOLERANCE_KW
        and battery_kw >= -charge_limit - _POWER_TOLERANCE_KW
    )
    battery_internal = battery.internal_power_kw(battery_kw, soc, duration)
    transitioned = requested_on != state.engine_on
    restarted = not state.engine_on and requested_on
    next_state = ThermostatState(
        engine_on=requested_on,
        elapsed_in_state_s=duration if transitioned else state.elapsed_in_state_s + duration,
        restart_count=state.restart_count + int(restarted),
        terminal_depletion=terminal,
    )
    return ThermostatDecision(
        engine_shaft_kw=engine_state.delivered_kw,
        bus_from_engine_kw=bus_from_engine,
        battery_bus_kw=battery_kw,
        battery_internal_kw=battery_internal,
        fuel_flow_kg_s=engine_state.fuel_flow_kg_s,
        restart_fuel_kg=parameters.restart_fuel_kg if restarted else 0.0,
        engine_off=engine_state.shut_down,
        regime=regime,
        regime_reason=reason,
        active_constraint=active,
        transitioned=transitioned,
        next_state=next_state,
        feasible=feasible,
    )
