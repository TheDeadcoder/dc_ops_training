# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
GRPO reward functions for DC-Ops (v3 — redesigned from the notebook).

Four signals combine to drive learning:

    1. format_reward_fn    — [-1.5, +1.0]  structural tags + known verb +
                                           trailing-text check. Does NOT
                                           saturate for SFT-perfect outputs.
    2. env_reward_fn       — [-3.0, +3.0]  runs the model's command in the
                                           real physics simulator (after
                                           replaying any warmup actions),
                                           with resolution / crash bonuses
                                           and a downstream-stability probe.
    3. command_quality_fn  — [-1.0, +1.0]  scenario-aware heuristic priors
                                           on first-turn vs mid-game actions.
    4. no_repeat_fn        — [-0.5,  0.0]  small penalty for mirroring the
                                           most recent warmup action.

GRPO does group-relative baselining internally, so we hand it the sum of
these four raw signals with no further baseline subtraction.
"""

from __future__ import annotations

import re
from typing import Any

from .constants import (
    BAD_FIRST_ACTIONS,
    CRASH_KEYWORDS,
    GOOD_FIRST_ACTIONS,
    KNOWN_COMMANDS,
    OPTIMAL_FIRST_ACTIONS,
    POST_DIAGNOSIS_ACTIONS,
    RESOLVE_KEYWORDS,
)

# ---------------------------------------------------------------------------
# Regexes / extractors
# ---------------------------------------------------------------------------
_CMD_RE       = re.compile(r"<command>\s*(.+?)\s*</command>",       re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>\s*(.+?)\s*</reasoning>",   re.DOTALL)


def extract_command(text: str) -> str | None:
    m = _CMD_RE.search(text)
    return m.group(1).strip() if m else None


def extract_reasoning(text: str) -> str | None:
    m = _REASONING_RE.search(text)
    return m.group(1).strip() if m else None


def _completion_text(completion) -> str:
    """Normalise the many shapes TRL hands a reward function.

    TRL may pass a plain string, a {'content': ...} dict, or a
    list-of-dicts (multi-turn). We peek at the first content block.
    """
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _pick(value, i: int, default=None):
    """Index into a list-or-scalar kwarg.

    TRL hands scalars to the reward when the dataset row has a single value
    and lists otherwise. This helper collapses both shapes.
    """
    if isinstance(value, list):
        return value[i] if i < len(value) else default
    return value if value is not None else default


# ═══════════════════════════════════════════════════════════════════════════
# 1) format_reward_fn
# ═══════════════════════════════════════════════════════════════════════════
def format_reward_fn(completions, **kwargs):
    """Structural check with real spread (doesn't saturate for SFT-clean outputs).

    Score recipe:
      + 0.6  both <reasoning> and <command> tags closed
      + 0.2  first token of command is a known verb
      + 0.2  no trailing junk (< 30 chars) after </command>
      − 0.5  command-only (no reasoning)
      − 0.8  reasoning-only (no command)
      − 1.0  neither tag
      − 0.5  a stray <think> block leaked through (student is non-reasoning)
      − 0.4  command verb is unknown
      − 0.2  trailing junk after </command>
    Clipped to [-1.5, +1.0].
    """
    rewards = []
    for completion in completions:
        text = _completion_text(completion)

        has_reasoning = bool(_REASONING_RE.search(text))
        has_command   = bool(_CMD_RE.search(text))
        has_think     = "<think>" in text

        cmd = extract_command(text) or ""
        cmd_head = cmd.strip().split()[0].lower() if cmd.strip() else ""

        reward = 0.0
        if has_reasoning and has_command:
            reward += 0.6
        elif has_command:
            reward -= 0.5
        elif has_reasoning:
            reward -= 0.8
        else:
            reward -= 1.0

        if has_think:
            reward -= 0.5

        if cmd_head in KNOWN_COMMANDS:
            reward += 0.2
        elif cmd_head:
            reward -= 0.4

        if "</command>" in text:
            after = text.split("</command>")[-1].strip()
            reward += 0.2 if len(after) < 30 else -0.2

        rewards.append(max(-1.5, min(1.0, reward)))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════
# 2) env_reward_fn
# ═══════════════════════════════════════════════════════════════════════════
def env_reward_fn(
    completions,
    prompts=None,
    scenario_id=None,
    seed=None,
    warmup_actions=None,
    **kwargs,
):
    """Physics reward: replay warmup → step the model's command → probe
    downstream stability → combine with resolution/crash bonuses.

    Raw env step-reward is in [-1, 1]. We scale that by 3, then:
      + 2.5 on scenario resolution (seen in alert text of a done episode)
      − 3.0 on catastrophic termination
      − 3.0 on `escalate` (hard-coded — handing off is the worst move)
      + mean(next 2 wait-step rewards)  as a stability probe (weight 1)
    Final clipped to [-3.0, +3.0].

    Imports DcOpsEnvironment lazily so this module can be imported in
    contexts without the env (e.g. format-only sanity tests).
    """
    # Local import — avoids import-order headaches
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
            rewards.append(-2.5)
            continue
        if sid is None:
            rewards.append(-1.0)
            continue

        cmd_head = cmd.strip().split()[0].lower() if cmd.strip() else ""
        if cmd_head == "escalate":
            rewards.append(-3.0)
            continue

        try:
            env = DcOpsEnvironment()
            env.reset(scenario=sid, seed=s)

            # Replay warmup so the state matches what the model was shown
            aborted = False
            for w in warmup:
                _o = env.step(DcOpsAction(command=w))
                if _o.done:
                    aborted = True
                    break
            if aborted:
                rewards.append(0.0)
                continue

            # Apply the model's action
            obs = env.step(DcOpsAction(command=cmd, reasoning=reasoning))
            r_now = float(obs.reward)

            alert_l = (obs.alert or "").lower()
            resolved_now = obs.done and any(k in alert_l for k in RESOLVE_KEYWORDS)
            crashed_now  = obs.done and any(k in alert_l for k in CRASH_KEYWORDS)

            # Roll 2 "wait" steps forward — probes downstream stability
            r_future, n_future = 0.0, 0
            resolved_future = crashed_future = False
            for _ in range(2):
                if obs.done:
                    break
                obs = env.step(DcOpsAction(command="wait"))
                r_future += float(obs.reward)
                n_future += 1
                alert_l = (obs.alert or "").lower()
                if obs.done:
                    if any(k in alert_l for k in RESOLVE_KEYWORDS):
                        resolved_future = True
                    elif any(k in alert_l for k in CRASH_KEYWORDS):
                        crashed_future = True

            combined = r_now * 3.0
            if n_future > 0:
                combined += (r_future / n_future) * 1.0
            if resolved_now or resolved_future:
                combined += 2.5
            if crashed_now or crashed_future:
                combined -= 3.0

            rewards.append(max(-3.0, min(3.0, combined)))
        except Exception:
            rewards.append(-1.5)

    return rewards


# ═══════════════════════════════════════════════════════════════════════════
# 3) command_quality_fn
# ═══════════════════════════════════════════════════════════════════════════
def command_quality_fn(completions, scenario_id=None, warmup_actions=None, **kwargs):
    """Scenario-aware heuristic: first-turn priors, mid-game priors,
    target-alignment bonus, domain-vocabulary bonus on reasoning.
    Range [-1.0, +1.0].
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        cmd  = extract_command(text)
        if cmd is None:
            rewards.append(-0.8)
            continue

        sid       = _pick(scenario_id, i, None)
        warmup    = _pick(warmup_actions, i, []) or []
        reasoning = extract_reasoning(text) or ""
        reward    = 0.0

        cmd_head   = cmd.strip().split()[0].lower()
        cmd_upper  = cmd.upper()
        is_midgame = bool(warmup)

        if sid:
            if not is_midgame:
                if cmd_head in OPTIMAL_FIRST_ACTIONS.get(sid, set()):
                    reward += 0.6
                elif cmd_head in GOOD_FIRST_ACTIONS.get(sid, set()):
                    reward += 0.25
                elif cmd_head in BAD_FIRST_ACTIONS:
                    reward -= 0.6
                elif cmd_head == "wait":
                    reward -= 0.3  # waiting on turn 1 gathers no info
            else:
                post = POST_DIAGNOSIS_ACTIONS.get(sid, set())
                if cmd_head in post:
                    reward += 0.5
                elif cmd_head in BAD_FIRST_ACTIONS:
                    reward -= 0.6
                elif cmd_head == "diagnose":
                    already_diagnosed = any(
                        w.lower().startswith("diagnose")
                        and w.strip().split()[1:2] == cmd.strip().split()[1:2]
                        for w in warmup
                    )
                    reward += -0.1 if already_diagnosed else 0.25
                elif cmd_head == "check_status":
                    reward += 0.1
                elif cmd_head == "wait":
                    reward += 0.05

        # Target-alignment — picking the right unit for the scenario
        if sid == "A2" and "CRAC-3" in cmd_upper:
            reward += 0.2
        elif sid == "A4" and ("CRAC-1" in cmd_upper or "CRAC-3" in cmd_upper):
            reward += 0.2
        elif sid == "B1" and "UPS" in cmd_upper:
            reward += 0.2
        elif sid == "B3" and "GEN" in cmd_upper:
            reward += 0.2
        elif sid == "B4" and ("UPS" in cmd_upper or "GEN" in cmd_upper):
            reward += 0.2

        # Small domain-vocab bonus on reasoning
        domain_terms = {
            "temperature", "thermal", "crac", "cooling",
            "compressor", "fan", "setpoint",
            "ups", "battery", "generator", "power",
            "load", "diagnose", "fault", "alarm",
        }
        term_count = sum(1 for t in domain_terms if t in reasoning.lower())
        reward += min(0.2, term_count * 0.03)

        rewards.append(max(-1.0, min(1.0, reward)))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════
# 4) no_repeat_fn
# ═══════════════════════════════════════════════════════════════════════════
def no_repeat_fn(completions, warmup_actions=None, **kwargs):
    """Small penalty for mirroring the most recent warmup action.

    `wait` and `check_status` are exempt — they're legitimately repeatable.
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        cmd  = extract_command(text)
        warmup = _pick(warmup_actions, i, []) or []

        if cmd is None or not warmup:
            rewards.append(0.0)
            continue

        last = warmup[-1].strip().lower()
        this = cmd.strip().lower()
        head = this.split()[0] if this else ""
        if head in {"wait", "check_status"}:
            rewards.append(0.0)
            continue
        rewards.append(-0.5 if this == last else 0.0)
    return rewards


# Public bundle
ALL_REWARD_FNS = [format_reward_fn, env_reward_fn, command_quality_fn, no_repeat_fn]
