"""
Episode runner — wraps the in-process DcOpsEnvironment for data generation.

Provides:
  - reset_episode(scenario_key, seed) -> (env, dashboard, action_result, steps_remaining, meta)
  - step_episode(env, command) -> (dashboard, action_result, reward, done, steps_remaining)
  - format_user_turn(action_result, steps_remaining, dashboard)
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass

# Either pip-install the dc_ops_env package, or set DC_OPS_ENV_PATH to the
# directory that *contains* the dc_ops_env folder.
_DC_OPS_PATH = os.environ.get(
    "DC_OPS_ENV_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dc_ops_environment-main")),
)
if _DC_OPS_PATH not in sys.path:
    sys.path.insert(0, _DC_OPS_PATH)

from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment  # noqa: E402
from dc_ops_env.models import DcOpsAction  # noqa: E402
from dc_ops_env.simulation.types import CRACStatus, UPSMode  # noqa: E402


# ---------------------------------------------------------------------------
# Variant injectors — applied AFTER env.reset() to elicit rare commands
# ---------------------------------------------------------------------------
def _inject_crac_standby(env: DcOpsEnvironment, rng: random.Random) -> str:
    candidates = [crac for zone in env._thermal_sim.state.zones
                  for crac in zone.crac_units if crac.status == CRACStatus.RUNNING]
    if not candidates:
        return ""
    target = rng.choice(candidates)
    target.status = CRACStatus.STANDBY
    target.fan_speed_pct = 0.0
    env._alert = (
        f"NOTICE: {target.unit_id} is in STANDBY following scheduled maintenance. "
        "Verify status and bring it back online to restore N+1 redundancy."
    )
    env._scenario_type = "thermal"
    return target.unit_id


def _inject_crac_maintenance_due(env: DcOpsEnvironment, rng: random.Random) -> str:
    candidates = [crac for zone in env._thermal_sim.state.zones
                  for crac in zone.crac_units if crac.status == CRACStatus.RUNNING]
    if len(candidates) < 3:
        return ""
    target = rng.choice(candidates)
    env._alert = (
        f"MAINTENANCE: {target.unit_id} is scheduled for preventive servicing this window. "
        "Verify N+1 redundancy is preserved, then place the unit in standby for the technician."
    )
    env._scenario_type = "thermal"
    return target.unit_id


def _inject_gen_low_fuel(env: DcOpsEnvironment, rng: random.Random) -> str:
    if env._power_sim is None:
        return ""
    gen = env._power_sim.state.generator
    gen.fuel_level_liters = rng.uniform(80.0, 250.0)
    env._alert = (
        f"WARNING: {gen.gen_id} fuel level low ({gen.fuel_level_liters:.0f}L of "
        f"{gen.fuel_tank_liters:.0f}L). Verify and refuel before next test or transfer."
    )
    env._scenario_type = "power"
    return gen.gen_id


def _inject_ups_eco_review(env: DcOpsEnvironment, rng: random.Random) -> str:
    if env._power_sim is None or not env._power_sim.state.ups_units:
        return ""
    target = rng.choice(env._power_sim.state.ups_units)
    target.mode = UPSMode.ECO
    env._alert = (
        f"NOTICE: {target.unit_id} was switched to ECO mode for an efficiency trial. "
        "Trial is complete — verify status and restore the unit to its protected operating mode."
    )
    env._scenario_type = "power"
    return target.unit_id


VARIANT_INJECTORS = {
    "VAR_CRAC_STANDBY":  _inject_crac_standby,
    "VAR_CRAC_MAINT":    _inject_crac_maintenance_due,
    "VAR_GEN_LOWFUEL":   _inject_gen_low_fuel,
    "VAR_UPS_MODE":      _inject_ups_eco_review,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class EpisodeMeta:
    scenario_key: str
    scenario_id: str
    scenario_type: str
    difficulty: str
    step_budget: int
    target_unit: str = ""


def reset_episode(scenario_key: str, seed: int):
    """Returns (env, dashboard, action_result, steps_remaining, meta)."""
    rng = random.Random(seed)
    env = DcOpsEnvironment()

    if scenario_key in VARIANT_INJECTORS:
        if scenario_key in ("VAR_GEN_LOWFUEL", "VAR_UPS_MODE"):
            base = "B3"
        else:
            base = "A1"
        env.reset(scenario=base, seed=seed)
        target_unit = VARIANT_INJECTORS[scenario_key](env, rng)
        obs = env._make_observation(
            action_result="Environment initialized. Awaiting your command.",
        )
        meta = EpisodeMeta(
            scenario_key=scenario_key,
            scenario_id=base,
            scenario_type=env._scenario_type,
            difficulty="custom",
            step_budget=env._step_budget,
            target_unit=target_unit,
        )
    else:
        obs = env.reset(scenario=scenario_key, seed=seed)
        sc = env._scenario
        meta = EpisodeMeta(
            scenario_key=scenario_key,
            scenario_id=sc.scenario_id if sc else "",
            scenario_type=sc.scenario_type if sc else "",
            difficulty=sc.difficulty if sc else "",
            step_budget=env._step_budget,
        )

    return env, obs.dashboard, obs.action_result, obs.steps_remaining, meta


def step_episode(env: DcOpsEnvironment, command: str):
    """Returns (dashboard, action_result, reward, done, steps_remaining)."""
    obs = env.step(DcOpsAction(command=command))
    return obs.dashboard, obs.action_result, float(obs.reward or 0.0), obs.done, obs.steps_remaining


def format_user_turn(action_result: str, steps_remaining: int, dashboard: str) -> str:
    return (
        f"**Action Result:** {action_result.strip()}\n\n"
        f"**Steps Remaining:** {steps_remaining}\n\n"
        f"{dashboard}"
    )
