"""GRPO reward functions for DC-Ops (v4)."""

from __future__ import annotations

import re
from typing import Any

from .constants import (
    BAD_FIRST_ACTIONS,
    GOOD_FIRST_ACTIONS,
    KNOWN_COMMANDS,
    OPTIMAL_FIRST_ACTIONS,
    POST_DIAGNOSIS_ACTIONS,
)


def _resolved(obs) -> bool:
    """True iff the simulator itself reports the incident resolved."""
    return bool(getattr(obs, "resolved", False))


def _crashed(obs) -> bool:
    """True iff the episode terminated in a genuine failure, as opposed to a
    plain timeout. A crash ends the episode (``done``) without resolution and
    with budget still on the clock; a timeout ends with ``steps_remaining == 0``.
    Reads the simulator's own signals — no alert-text keyword matching.
    """
    return bool(
        getattr(obs, "done", False)
        and not getattr(obs, "resolved", False)
        and getattr(obs, "steps_remaining", 0) > 0
    )

_CMD_RE       = re.compile(r"<command>\s*(.+?)\s*</command>",       re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>\s*(.+?)\s*</reasoning>",   re.DOTALL)


def extract_command(text: str) -> str | None:
    m = _CMD_RE.search(text)
    return m.group(1).strip() if m else None


def extract_reasoning(text: str) -> str | None:
    m = _REASONING_RE.search(text)
    return m.group(1).strip() if m else None


def _completion_text(completion) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _pick(value, i: int, default=None):
    if isinstance(value, list):
        return value[i] if i < len(value) else default
    return value if value is not None else default


# ---------------------------------------------------------------------------
# Proxy health score — scenario-agnostic [0, 1] signal from live sim state.
# Replaces the unreachable `scenario.resolved` gate at RL time.
# ---------------------------------------------------------------------------
def _proxy_health(env) -> float:
    """Compute [0, 1] health score from env._thermal_sim and env._power_sim.

    Thermal half: fraction of zones within ASHRAE recommended range,
      smoothly degrading through allowable and beyond.
    Power half: avg UPS battery SOC + generator-readiness-during-outage.
    """
    try:
        from dc_ops_env.config import ASHRAE_CLASSES
    except Exception:
        ASHRAE_CLASSES = {}

    score = 0.0
    weight = 0.0

    thermal = getattr(env, "_thermal_sim", None)
    if thermal is not None:
        try:
            zone_scores: list[float] = []
            for zone in thermal.state.zones:
                ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
                if not ashrae:
                    continue
                t = zone.max_inlet_temp_c
                rec = ashrae.recommended_max_c
                allow = ashrae.allowable_max_c
                span = max(1e-6, allow - rec)
                if t <= rec:
                    s = 1.0
                elif t <= allow:
                    s = 1.0 - 0.5 * (t - rec) / span
                else:
                    s = max(0.0, 0.5 - 0.1 * (t - allow))
                zone_scores.append(s)
            if zone_scores:
                score += 0.5 * (sum(zone_scores) / len(zone_scores))
                weight += 0.5
        except Exception:
            pass

    power = getattr(env, "_power_sim", None)
    if power is not None:
        try:
            sub = 0.0
            socs = [u.battery_soc for u in power.state.ups_units]
            if socs:
                sub += 0.5 * (sum(socs) / len(socs))

            any_on_battery = False
            for u in power.state.ups_units:
                mode_v = getattr(u.mode, "value", str(u.mode))
                if mode_v == "on_battery":
                    any_on_battery = True
                    break

            gen = power.state.generator
            gen_v = getattr(gen.state, "value", str(gen.state))
            if any_on_battery:
                if gen_v == "loaded":
                    sub += 0.5
                elif gen_v in ("ready", "warming"):
                    sub += 0.35
                elif gen_v in ("cranking", "start_delay"):
                    sub += 0.2
            else:
                sub += 0.5

            score += 0.5 * sub
            weight += 0.5
        except Exception:
            pass

    return score / weight if weight > 0 else 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 1) format_reward_fn — negative-dominant, does NOT saturate
# ═══════════════════════════════════════════════════════════════════════════
def format_reward_fn(completions, **kwargs):
    """Asymmetric structural check. Clean outputs land in [-0.2, +0.3];
    malformed outputs land in [-1.5, -0.4]. The deliberate lack of a flat
    +1.0 ceiling prevents the group-mean advantage from collapsing when the
    SFT checkpoint already emits clean format (your observed failure mode).
    """
    rewards = []
    for completion in completions:
        text = _completion_text(completion)

        has_reasoning = bool(_REASONING_RE.search(text))
        has_command   = bool(_CMD_RE.search(text))
        has_think     = "<think>" in text

        if not has_command and not has_reasoning:
            rewards.append(-1.5)
            continue
        if not has_command:
            rewards.append(-1.0)
            continue
        if not has_reasoning:
            rewards.append(-0.6)
            continue

        cmd = extract_command(text) or ""
        reasoning = extract_reasoning(text) or ""
        cmd_head = cmd.strip().split()[0].lower() if cmd.strip() else ""

        r = 0.0
        if has_think:
            r -= 0.4
        if cmd_head and cmd_head not in KNOWN_COMMANDS:
            r -= 0.5
        if "</command>" in text:
            tail = text.split("</command>")[-1].strip()
            if len(tail) > 30:
                r -= 0.2

        # Reasoning-length sweet spot — creates within-group variance
        n_words = len(reasoning.split())
        if n_words < 8:
            r -= 0.2
        elif n_words > 260:
            r -= 0.2
        elif 25 <= n_words <= 180:
            r += 0.15

        rewards.append(max(-1.5, min(0.3, r)))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════
# 2) env_reward_fn — physics reward with proxy-health delta
# ═══════════════════════════════════════════════════════════════════════════
def env_reward_fn(
    completions,
    prompts=None,
    scenario_id=None,
    seed=None,
    warmup_actions=None,
    **kwargs,
):
    """Physics + proxy-resolution. Replaces the unreachable `resolved` gate
    with a continuous health-delta signal that is earnable in 1–5 steps.

    Composition:
      r_now * 3.0                  (env's own clamped reward for this action)
      + (proxy_after - before) * 2.5   (immediate effect of the action)
      + (best_proxy  - before) * 1.0   (stability over 4-wait probe)
      + 3.0 on resolve (if it happens inside the probe)
      - 3.0 on crash
    Clamped to [-4.0, +5.0]. This is ~2× the range / 3× the variance of v3.
    """
    from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
    from dc_ops_env.models import DcOpsAction

    rewards = []
    for i, completion in enumerate(completions):
        text      = _completion_text(completion)
        cmd       = extract_command(text)
        reasoning = extract_reasoning(text) or ""

        sid    = _pick(scenario_id, i, None)
        s      = _pick(seed, i, 42) or 42
        warmup = _pick(warmup_actions, i, []) or []

        if cmd is None:
            rewards.append(-3.0)
            continue
        if sid is None:
            rewards.append(-1.0)
            continue

        cmd_head = cmd.strip().split()[0].lower() if cmd.strip() else ""
        if cmd_head == "escalate":
            rewards.append(-3.5)
            continue

        try:
            env = DcOpsEnvironment()
            env.reset(scenario=sid, seed=s)

            aborted = False
            for w in warmup:
                _o = env.step(DcOpsAction(command=w))
                if _o.done:
                    aborted = True
                    break
            if aborted:
                rewards.append(-0.5)
                continue

            proxy_before = _proxy_health(env)

            obs = env.step(DcOpsAction(command=cmd, reasoning=reasoning))
            r_now = float(obs.reward)
            proxy_after = _proxy_health(env)

            resolved = _resolved(obs)
            crashed  = _crashed(obs)

            best_proxy = proxy_after
            for _ in range(4):
                if obs.done:
                    break
                obs = env.step(DcOpsAction(command="wait"))
                p = _proxy_health(env)
                if p > best_proxy:
                    best_proxy = p
                if obs.done:
                    if _resolved(obs):
                        resolved = True
                    elif _crashed(obs):
                        crashed = True

            delta_immediate = proxy_after - proxy_before
            delta_best      = best_proxy  - proxy_before

            combined = (
                r_now * 3.0
                + delta_immediate * 2.5
                + delta_best * 1.0
            )
            if resolved:
                combined += 3.0
            if crashed:
                combined -= 3.0

            rewards.append(max(-4.0, min(5.0, combined)))
        except Exception:
            rewards.append(-2.0)

    return rewards


# ═══════════════════════════════════════════════════════════════════════════
# 3) command_quality_fn — per-verb target alignment + param quality
# ═══════════════════════════════════════════════════════════════════════════
def command_quality_fn(completions, scenario_id=None, warmup_actions=None, **kwargs):
    """Scenario-aware heuristic priors, split by verb so that operating the
    correct target is rewarded and operating the failed unit is penalized.
    Also penalizes overcooling (setpoint <17°C) and under-shedding.
    Range [-1.5, +1.2].
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        cmd  = extract_command(text)
        if cmd is None:
            rewards.append(-1.0)
            continue

        sid       = _pick(scenario_id, i, None)
        warmup    = _pick(warmup_actions, i, []) or []
        reasoning = extract_reasoning(text) or ""
        r         = 0.0

        cmd_lower = cmd.strip().lower()
        cmd_upper = cmd.upper()
        parts     = cmd_lower.split()
        cmd_head  = parts[0] if parts else ""
        cmd_value: float | None = None
        if len(parts) >= 3:
            try:
                cmd_value = float(parts[2])
            except ValueError:
                cmd_value = None

        is_midgame     = bool(warmup)
        warmup_lower   = [w.strip().lower() for w in warmup]
        warmup_heads   = [w.split()[0] for w in warmup_lower if w]
        has_diagnosed  = any(h == "diagnose" for h in warmup_heads)
        has_start_gen  = any("start_generator" in w for w in warmup_lower)

        # --- Turn-phase priors -------------------------------------------------
        if sid:
            if not is_midgame:
                if cmd_head in OPTIMAL_FIRST_ACTIONS.get(sid, set()):
                    r += 0.5
                elif cmd_head in GOOD_FIRST_ACTIONS.get(sid, set()):
                    r += 0.2
                elif cmd_head in BAD_FIRST_ACTIONS:
                    r -= 0.6
                elif cmd_head == "wait":
                    r -= 0.4
            else:
                post = POST_DIAGNOSIS_ACTIONS.get(sid, set())
                if cmd_head in post:
                    r += 0.4
                elif cmd_head in BAD_FIRST_ACTIONS:
                    r -= 0.6
                elif cmd_head == "diagnose":
                    same_target = any(
                        w.split()[:2] == cmd_lower.split()[:2] for w in warmup_lower
                    )
                    r += -0.25 if same_target else (0.15 if not has_diagnosed else -0.1)
                elif cmd_head == "check_status":
                    r += 0.05

        # --- Per-scenario target/value alignment (the big fix) ----------------
        if sid == "A4":
            # Failed: CRAC-1, CRAC-3. Survivors: CRAC-2, CRAC-4.
            if cmd_head == "diagnose":
                if "CRAC-1" in cmd_upper or "CRAC-3" in cmd_upper:
                    r += 0.3
                else:
                    r -= 0.1
            elif cmd_head == "set_fan_speed":
                if "CRAC-2" in cmd_upper or "CRAC-4" in cmd_upper:
                    r += 0.3
                    if cmd_value is not None and cmd_value >= 90:
                        r += 0.2
                elif "CRAC-1" in cmd_upper or "CRAC-3" in cmd_upper:
                    r -= 0.4
            elif cmd_head == "adjust_setpoint":
                if "CRAC-2" in cmd_upper or "CRAC-4" in cmd_upper:
                    r += 0.1
                elif "CRAC-1" in cmd_upper or "CRAC-3" in cmd_upper:
                    r -= 0.35
                if cmd_value is not None:
                    if cmd_value < 17:
                        r -= 0.3        # overcool — penalise 16.0 setpoints
                    elif 18 <= cmd_value <= 22:
                        r += 0.15
            elif cmd_head == "set_rack_load":
                if cmd_value is not None:
                    if cmd_value <= 5:
                        r += 0.3
                    elif cmd_value <= 7:
                        r += 0.1
                    else:
                        r -= 0.15

        elif sid == "A2":
            # Failed: CRAC-3.
            if cmd_head == "diagnose":
                if "CRAC-3" in cmd_upper:
                    r += 0.35
                else:
                    r -= 0.1
            elif cmd_head in ("set_fan_speed", "adjust_setpoint"):
                if "CRAC-3" in cmd_upper:
                    r -= 0.3
                elif any(f"CRAC-{n}" in cmd_upper for n in (1, 2, 4)):
                    r += 0.2
                if cmd_head == "set_fan_speed" and cmd_value is not None and cmd_value >= 90:
                    r += 0.15
                if cmd_head == "adjust_setpoint" and cmd_value is not None:
                    if cmd_value < 17:
                        r -= 0.25
                    elif 18 <= cmd_value <= 22:
                        r += 0.1

        elif sid == "A1":
            # PUE optimisation — raise setpoints toward 20-24.
            if cmd_head == "adjust_setpoint" and cmd_value is not None:
                if 20 <= cmd_value <= 25:
                    r += 0.3
                elif cmd_value < 18:
                    r -= 0.3
            if cmd_head == "check_status" and not is_midgame:
                r += 0.15

        elif sid == "B1":
            if cmd_head == "diagnose" and "UPS" in cmd_upper:
                r += 0.4
            elif cmd_head == "acknowledge_alarm":
                r += 0.3 if has_diagnosed else -0.1

        elif sid == "B3":
            if cmd_head == "check_status" and not is_midgame:
                r += 0.3
            elif cmd_head == "start_generator":
                r += -0.3 if has_start_gen else 0.3
            elif cmd_head == "diagnose" and "GEN" in cmd_upper:
                r += 0.2
            elif cmd_head == "stop_generator":
                r += 0.3 if has_start_gen else -0.3

        elif sid == "B4":
            if cmd_head == "diagnose":
                if "UPS" in cmd_upper:
                    r += 0.4
                elif "GEN" in cmd_upper:
                    r += 0.1
            elif cmd_head == "start_generator":
                r += -0.2 if has_start_gen else 0.5
            elif cmd_head == "set_rack_load" and cmd_value is not None:
                if cmd_value <= 4:
                    r += 0.35
                elif cmd_value <= 6:
                    r += 0.15
                else:
                    r -= 0.1
            elif cmd_head == "wait" and not has_start_gen:
                r -= 0.2    # waiting while utility lost and gen not started is bad

        # --- Domain vocab bonus (moderate) -----------------------------------
        domain = {
            "temperature", "thermal", "crac", "cooling", "compressor", "fan",
            "setpoint", "ups", "battery", "generator", "power", "load",
            "diagnose", "fault", "alarm", "shed", "pue", "ashrae", "inlet",
        }
        n = sum(1 for t in domain if t in reasoning.lower())
        r += min(0.15, n * 0.02)

        rewards.append(max(-1.5, min(1.2, r)))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════
# 4) no_repeat_fn — penalises repeat of ANY warmup action, not just the last
# ═══════════════════════════════════════════════════════════════════════════
def no_repeat_fn(completions, warmup_actions=None, **kwargs):
    """Catches:
      - exact duplicate of any warmup action (hard penalty)
      - same verb+target as any warmup action but different value (soft penalty)
      - waiting when warmup already contains a wait (loitering)
    `wait` and `check_status` without warmup-repeats remain neutral.
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        cmd  = extract_command(text)
        warmup = _pick(warmup_actions, i, []) or []

        if cmd is None:
            rewards.append(-0.3)
            continue
        if not warmup:
            rewards.append(0.0)
            continue

        cmd_norm = cmd.strip().lower()
        parts    = cmd_norm.split()
        head     = parts[0] if parts else ""

        warmup_norms = [w.strip().lower() for w in warmup]

        if head in {"wait", "check_status"}:
            # Mild penalty for redundant waits — only if multiple waits already queued
            if head == "wait" and warmup_norms.count("wait") >= 1:
                rewards.append(-0.2)
            else:
                rewards.append(0.0)
            continue

        if cmd_norm in warmup_norms:
            rewards.append(-1.0)
            continue

        if len(parts) >= 2:
            verb_target = (parts[0], parts[1])
            hit = False
            for w in warmup_norms:
                wp = w.split()
                if len(wp) >= 2 and (wp[0], wp[1]) == verb_target:
                    hit = True
                    break
            rewards.append(-0.4 if hit else 0.0)
        else:
            rewards.append(0.0)
    return rewards


ALL_REWARD_FNS = [format_reward_fn, env_reward_fn, command_quality_fn, no_repeat_fn]