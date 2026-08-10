"""Shared constants: scenario IDs, scenario→alert markers, known commands."""

from __future__ import annotations

SCENARIOS: list[str] = ["A1", "A2", "A4", "B1", "B3", "B4"]

BASE_ALERT_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("A1", ("PUE exceeds", "subop")),
    ("A2", ("CRAC-3 compressor failure",)),
    ("A4", ("Multiple CRAC failures",)),
    ("B1", ("UPS-1 transferred to battery",)),
    ("B3", ("Monthly generator test",)),
    ("B4", ("Utility power lost",)),
]

KNOWN_COMMANDS: set[str] = {
    "diagnose", "check_status", "adjust_setpoint", "set_fan_speed",
    "set_rack_load", "migrate_workload", "start_crac", "stop_crac",
    "start_generator", "stop_generator", "set_ups_mode", "refuel_generator",
    "acknowledge_alarm", "escalate", "wait",
}

# ---------------------------------------------------------------------------
# Scenario priors used by command_quality_fn.
# ---------------------------------------------------------------------------
GOOD_FIRST_ACTIONS: dict[str, set[str]] = {
    "A1": {"check_status", "adjust_setpoint", "diagnose"},
    "A2": {"check_status", "diagnose"},
    "A4": {"check_status", "diagnose"},
    "B1": {"check_status", "diagnose"},
    "B3": {"check_status", "start_generator"},
    "B4": {"check_status", "diagnose", "start_generator"},
}

OPTIMAL_FIRST_ACTIONS: dict[str, set[str]] = {
    "A1": {"check_status"},
    "A2": {"diagnose"},
    "A4": {"diagnose"},
    "B1": {"diagnose"},
    "B3": {"check_status"},
    "B4": {"diagnose"},
}

BAD_FIRST_ACTIONS: set[str] = {"escalate", "stop_generator", "stop_crac"}

POST_DIAGNOSIS_ACTIONS: dict[str, set[str]] = {
    "A1": {"adjust_setpoint", "set_fan_speed", "wait"},
    "A2": {"set_fan_speed", "adjust_setpoint", "set_rack_load"},
    "A4": {"set_fan_speed", "adjust_setpoint", "set_rack_load"},
    "B1": {"acknowledge_alarm", "diagnose", "wait"},
    "B3": {"start_generator", "diagnose", "wait", "stop_generator", "acknowledge_alarm"},
    "B4": {"set_rack_load", "start_generator", "diagnose", "wait"},
}

# Expanded warmup sequences (7–9 per scenario, up from 2–3).
# More patterns → more distinct mid-game contexts → more GRPO signal diversity.
SCENARIO_WARMUP_ACTIONS: dict[str, list[list[str]]] = {
    "A1": [
        ["check_status"],
        ["check_status", "adjust_setpoint CRAC-1 22"],
        ["check_status", "diagnose CRAC-1"],
        ["diagnose CRAC-1"],
        ["check_status", "adjust_setpoint CRAC-1 22", "adjust_setpoint CRAC-2 22"],
        ["adjust_setpoint CRAC-1 20", "adjust_setpoint CRAC-2 20"],
        ["check_status", "adjust_setpoint CRAC-1 24"],
    ],
    "A2": [
        ["check_status"],
        ["diagnose CRAC-3"],
        ["check_status", "diagnose CRAC-3"],
        ["diagnose CRAC-3", "set_fan_speed CRAC-1 100"],
        ["diagnose CRAC-3", "set_fan_speed CRAC-2 100"],
        ["diagnose CRAC-3", "set_fan_speed CRAC-1 100", "set_fan_speed CRAC-2 100"],
        ["check_status", "diagnose CRAC-3", "adjust_setpoint CRAC-1 20"],
        ["diagnose CRAC-3", "set_rack_load B-01 5"],
    ],
    "A4": [
        ["check_status"],
        ["diagnose CRAC-1"],
        ["diagnose CRAC-3"],
        ["check_status", "diagnose CRAC-1"],
        ["diagnose CRAC-1", "diagnose CRAC-3"],
        ["diagnose CRAC-1", "diagnose CRAC-3", "set_fan_speed CRAC-2 100"],
        ["diagnose CRAC-1", "diagnose CRAC-3", "set_fan_speed CRAC-2 100", "set_fan_speed CRAC-4 100"],
        ["diagnose CRAC-1", "diagnose CRAC-3", "set_rack_load B-01 4"],
        ["check_status", "diagnose CRAC-1", "diagnose CRAC-3", "set_fan_speed CRAC-2 100"],
    ],
    "B1": [
        ["check_status"],
        ["diagnose UPS-1"],
        ["check_status", "diagnose UPS-1"],
        ["diagnose UPS-1", "diagnose GEN-1"],
        ["check_status", "diagnose UPS-1", "diagnose GEN-1"],
    ],
    "B3": [
        ["check_status"],
        ["check_status", "start_generator"],
        ["start_generator"],
        ["check_status", "start_generator", "wait"],
        ["check_status", "start_generator", "diagnose GEN-1"],
        ["start_generator", "diagnose GEN-1", "wait"],
    ],
    "B4": [
        ["check_status"],
        ["diagnose UPS-1"],
        ["check_status", "diagnose UPS-1"],
        ["diagnose UPS-1", "start_generator"],
        ["diagnose UPS-1", "start_generator", "set_rack_load A-01 3"],
        ["diagnose UPS-1", "diagnose GEN-1"],
        ["check_status", "diagnose UPS-1", "start_generator"],
        ["diagnose UPS-1", "start_generator", "set_rack_load A-01 3", "set_rack_load B-01 3"],
    ],
}

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

# NOTE: RESOLVE_KEYWORDS / CRASH_KEYWORDS were removed (review issue 1.3).
# Episode outcome must be read from the simulator's own signal —
# DcOpsObservation.resolved (and .steps_remaining to tell a real terminal
# failure from a plain timeout) — never inferred by string-matching the alert.
# "CRITICAL" is the literal first word of A4/B4's *opening* alert, so keyword
# matching mislabelled a live-alarm termination as a crash and could not
# distinguish a timeout from a genuine failure.