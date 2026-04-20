# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Shared constants: scenario IDs, scenario→alert markers, known commands."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The 6 scenarios the DC-Ops environment actually registers. The SFT dataset
# also contains 4 `VAR_*` alert families that have no matching scenario in the
# env, so we drop them (see data_utils.episode_scenario).
# ---------------------------------------------------------------------------
SCENARIOS: list[str] = ["A1", "A2", "A4", "B1", "B3", "B4"]

# Alert-line substrings that identify each registered scenario.
# All substrings in a tuple must be present in the first human turn's ALERT
# line for the episode to match that scenario.
BASE_ALERT_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("A1", ("PUE exceeds", "subop")),            # "NOTICE: PUE exceeds 1.8 … suboptimal"
    ("A2", ("CRAC-3 compressor failure",)),      # single-CRAC compressor fault
    ("A4", ("Multiple CRAC failures",)),         # cascade
    ("B1", ("UPS-1 transferred to battery",)),   # brief UPS outage
    ("B3", ("Monthly generator test",)),         # scheduled gen test
    ("B4", ("Utility power lost",)),             # full utility outage
]

# Valid operator command verbs (maps to dc_ops_env.actions.parser.AVAILABLE_ACTIONS,
# but kept here as a plain set for fast reward-fn checks without importing the env).
KNOWN_COMMANDS: set[str] = {
    "diagnose", "check_status", "adjust_setpoint", "set_fan_speed",
    "set_rack_load", "migrate_workload", "start_crac", "stop_crac",
    "start_generator", "stop_generator", "set_ups_mode", "refuel_generator",
    "acknowledge_alarm", "escalate", "wait",
}

# ---------------------------------------------------------------------------
# Scenario priors used by the command_quality reward function.
# ---------------------------------------------------------------------------

# First-turn actions that are *reasonable* per scenario (not necessarily ideal).
GOOD_FIRST_ACTIONS: dict[str, set[str]] = {
    "A1": {"check_status", "adjust_setpoint", "diagnose"},
    "A2": {"check_status", "diagnose"},
    "A4": {"check_status", "diagnose"},
    "B1": {"check_status", "diagnose"},
    "B3": {"check_status", "start_generator"},
    "B4": {"check_status", "diagnose", "start_generator"},
}

# First-turn actions that are *optimal* per scenario.
OPTIMAL_FIRST_ACTIONS: dict[str, set[str]] = {
    "A1": {"check_status"},
    "A2": {"diagnose"},
    "A4": {"diagnose"},
    "B1": {"diagnose"},
    "B3": {"check_status"},
    "B4": {"diagnose"},
}

# Actions that are almost always wrong on turn 1.
BAD_FIRST_ACTIONS: set[str] = {"escalate", "stop_generator", "stop_crac"}

# Post-diagnosis follow-up actions per scenario (used when `warmup_actions`
# is non-empty in command_quality_fn).
POST_DIAGNOSIS_ACTIONS: dict[str, set[str]] = {
    "A1": {"adjust_setpoint", "set_fan_speed", "wait"},
    "A2": {"set_fan_speed", "adjust_setpoint", "set_rack_load"},
    "A4": {"set_fan_speed", "adjust_setpoint", "set_rack_load"},
    "B1": {"acknowledge_alarm", "diagnose", "wait"},
    "B3": {"start_generator", "diagnose", "wait", "stop_generator", "acknowledge_alarm"},
    "B4": {"set_rack_load", "diagnose", "start_generator", "wait"},
}

# Warmup action sequences for GRPO mid-game prompts. The SAME list is stored
# on each prompt record so env_reward_fn can replay it before scoring.
SCENARIO_WARMUP_ACTIONS: dict[str, list[list[str]]] = {
    "A1": [["check_status"], ["check_status", "adjust_setpoint CRAC-1 22"]],
    "A2": [["check_status"], ["diagnose CRAC-3"],
           ["diagnose CRAC-3", "set_fan_speed CRAC-1 100"]],
    "A4": [["check_status"], ["diagnose CRAC-1"],
           ["diagnose CRAC-1", "diagnose CRAC-3"]],
    "B1": [["check_status"], ["diagnose UPS-1"]],
    "B3": [["check_status"], ["check_status", "start_generator"]],
    "B4": [["check_status"], ["diagnose UPS-1"],
           ["diagnose UPS-1", "start_generator"]],
}

# Expert-operator follow-ups used in multi-step eval (not training).
SCENARIO_EXPERT_FOLLOWUPS: dict[str, list[str]] = {
    "A1": ["check_status", "adjust_setpoint CRAC-1 22", "adjust_setpoint CRAC-2 22",
           "adjust_setpoint CRAC-3 22", "adjust_setpoint CRAC-4 22", "wait"],
    "A2": ["check_status", "diagnose CRAC-3", "set_fan_speed CRAC-1 100",
           "set_fan_speed CRAC-2 100", "adjust_setpoint CRAC-1 20", "wait"],
    "A4": ["check_status", "diagnose CRAC-1", "diagnose CRAC-3",
           "set_fan_speed CRAC-2 100", "set_fan_speed CRAC-4 100",
           "set_rack_load B-01 3", "wait"],
    "B1": ["check_status", "diagnose UPS-1", "diagnose GEN-1",
           "acknowledge_alarm", "wait"],
    "B3": ["check_status", "start_generator", "wait", "wait",
           "diagnose GEN-1", "stop_generator", "acknowledge_alarm"],
    "B4": ["check_status", "diagnose UPS-1", "start_generator",
           "set_rack_load A-01 3", "set_rack_load B-01 3", "wait", "wait"],
}

# Regex keywords used by env_reward_fn to classify terminal states.
RESOLVE_KEYWORDS = ("stabilized", "resolved", "optimized", "properly", "successfully")
CRASH_KEYWORDS   = ("critical", "exhausted", "emergency", "unprotected")
