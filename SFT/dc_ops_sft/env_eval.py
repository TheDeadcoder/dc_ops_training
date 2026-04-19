# Copyright (c) 2026. Licensed under BSD-3-Clause.
"""
In-process rollouts against the DC-Ops environment for before/after SFT eval.

Flow per episode:
  1. Instantiate DcOpsEnvironment (no HTTP server needed).
  2. env.reset(scenario="A2") -> DcOpsObservation
  3. Loop:
       - Build user turn from (action_result, steps_remaining, dashboard)
       - tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       enable_thinking=True)
       - model.generate(...)
       - Parse <command>, <reasoning> from the generation
       - env.step(DcOpsAction(command=..., reasoning=...))
       - Record reward, append turn to messages
  4. Stop on env.done or max_steps.

Resolution detection: the env sets obs.done when (a) budget exhausted,
(b) critical failure, or (c) scenario resolved. We distinguish (c) by the
alert text (scenarios set a resolution_message like "Thermal event
stabilized. All zones within recommended range.") and by the positive
speed bonus added to reward on resolution. This is a heuristic — the
authoritative signal is on the server side, but this is close enough
for SFT vs base comparison.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


# -----------------------------------------------------------------------------
# Extraction regexes
# -----------------------------------------------------------------------------
_RE_COMMAND = re.compile(r"<command>\s*(.+?)\s*</command>", re.DOTALL | re.IGNORECASE)
_RE_REASONING = re.compile(r"<reasoning>\s*(.+?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
_RE_THINK = re.compile(r"<think>\s*(.+?)\s*</think>", re.DOTALL | re.IGNORECASE)

_RESOLUTION_KEYWORDS = (
    "resolved", "stabilized", "optimized", "investigated",
    "completed", "acknowledged",
)


# -----------------------------------------------------------------------------
# System prompt (matches the one in Melikshah/dc-ops-sft-data). Kept verbatim
# so the model sees the same instructions at inference as during SFT.
# -----------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """You are DC-Ops Agent, an expert datacenter operations engineer. You manage a physics-based datacenter simulation. You observe a monitoring dashboard and issue exactly one operator command per turn to maintain thermal safety, power reliability, and energy efficiency.

AVAILABLE COMMANDS:
- check_status — Request full status report
- diagnose <unit_id> — Inspect a CRAC/UPS/PDU/GEN for faults (e.g., diagnose CRAC-3, diagnose UPS-1, diagnose GEN-1, diagnose PDU-A-01)
- adjust_setpoint <crac_id> <temp_c> — Change CRAC supply air setpoint (10–35°C). Lower setpoint = more cooling, higher PUE.
- set_fan_speed <crac_id> <pct> — Set CRAC fan speed (0–100%). More airflow lifts capacity but raises fan power cubically.
- set_rack_load <rack_id> <kw> — Migrate workload off a rack (0–30 kW). Use to shed heat from hot racks.
- start_crac <crac_id> — Start a standby CRAC unit
- stop_crac <crac_id> — Put a CRAC into standby
- start_generator — Manually start the diesel generator
- stop_generator — Initiate generator cooldown
- set_ups_mode <ups_id> <mode> — Set UPS mode: eco | double_conversion | bypass | line_interactive
- refuel_generator [liters] — Refuel generator (default: full tank)
- acknowledge_alarm — Acknowledge current alert
- escalate — Escalate to senior engineer (LAST RESORT — heavily penalized)
- wait — Take no action this step (only when waiting for a process, e.g. generator warmup)

OPERATIONAL PROCEDURES:
1. ALWAYS check_status or diagnose BEFORE making active changes.
2. ALWAYS diagnose the faulty unit BEFORE compensating with other units.
3. Pattern: assess → diagnose → compensate → verify → resolve.
4. ASHRAE limits — A2: recommended max 27°C, allowable max 35°C. H1 (HPC/AI): recommended max 22°C, allowable max 25°C.
5. Power: monitor UPS battery SOC, generator state, ATS position.
6. Use load shedding (set_rack_load) when cooling capacity is severely reduced.
7. Generator test order: check_status → start_generator → wait for warmup → diagnose GEN-1 → stop_generator → acknowledge_alarm.
8. NEVER repeat the exact same command twice in a row except `wait` and `check_status`.

RESPONSE FORMAT:
Produce three blocks in order: <think>, <reasoning>, <command>. Each block must appear exactly once.

1. <think>...</think>
   Think through the situation freely. Explore alternatives, do sanity checks, change your mind if needed. There is no length limit and self-correction is welcome here. This is your private scratchpad.

2. <reasoning>...</reasoning>
   After thinking, write a concise FINAL summary of your decision. This will be recorded as the official operations-log entry, so it must be clean and structured. STRICT REQUIREMENTS:
   • Maximum 200 words.
   • NO self-correction, NO "wait, actually", NO "let me reconsider". Only your final committed position.
   • Use exactly four numbered points:
     1. Situation — what the dashboard shows that matters (one sentence).
     2. Constraint — the relevant ASHRAE limit, procedure rule, or system state (one sentence).
     3. Step — which phase of assess→diagnose→compensate→verify→resolve you are on (one sentence).
     4. Action — the single command you are issuing and why it is the right next step (one sentence).
   Do not repeat verbatim what you wrote in <think>. This is the distilled conclusion, not a transcript.

3. <command>...</command>
   Exactly one command line from the list above. Nothing else inside the tag. Do not output multiple commands. Do not escalate."""


def parse_model_output(text: str) -> Dict[str, str]:
    """Extract <think>, <reasoning>, <command> blocks from a generation."""
    cmd = _RE_COMMAND.search(text)
    rsn = _RE_REASONING.search(text)
    thk = _RE_THINK.search(text)
    return {
        "command": cmd.group(1).strip() if cmd else "",
        "reasoning": rsn.group(1).strip() if rsn else "",
        "think": thk.group(1).strip() if thk else "",
        "raw": text,
    }


def format_user_turn(obs) -> str:
    """Match the training-data user-turn format exactly."""
    parts: List[str] = []
    if getattr(obs, "action_result", ""):
        parts.append(f"**Action Result:** {obs.action_result}")
        parts.append("")
    parts.append(f"**Steps Remaining:** {obs.steps_remaining}")
    parts.append("")
    parts.append(obs.dashboard)
    return "\n".join(parts)


def detect_resolved(obs, final_reward: float) -> bool:
    """Heuristic resolution detection from the last obs + its reward."""
    if not getattr(obs, "done", False):
        return False
    alert = (getattr(obs, "alert", "") or "").lower()
    if any(kw in alert for kw in _RESOLUTION_KEYWORDS):
        return True
    # Resolution gives a positive speed bonus (up to +1.0 depending on steps
    # remaining). Failure modes give negative or near-zero reward.
    if final_reward is not None and final_reward > 0.3:
        return True
    return False


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
@dataclass
class EpisodeMetrics:
    scenario_id: str
    episode_id: str
    steps: int = 0
    cumulative_reward: float = 0.0
    resolved: bool = False
    command_parse_failures: int = 0
    unknown_commands: int = 0
    action_history: List[str] = field(default_factory=list)
    reward_per_step: List[float] = field(default_factory=list)
    wall_s: float = 0.0
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Rollout
# -----------------------------------------------------------------------------
def _trim_history(messages: List[Dict[str, str]], window: int) -> List[Dict[str, str]]:
    """Keep system + last `window` turns (user+assistant pairs)."""
    if len(messages) <= window + 1:
        return messages
    return [messages[0]] + messages[-window:]


def rollout_episode(
    model,
    tokenizer,
    env,
    scenario_id: str,
    *,
    max_steps: Optional[int] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    history_window: int = 16,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    verbose: bool = False,
) -> EpisodeMetrics:
    """Run one rollout on the given scenario.

    Args:
      model, tokenizer: standard HF objects (Unsloth FastLanguageModel OK).
      env: a DcOpsEnvironment instance (in-process).
      scenario_id: "A1", "A2", "A4", "B1", "B3", "B4" (or variant).
      max_steps: override the scenario's step budget. None = use scenario's.
    """
    from dc_ops_env.models import DcOpsAction

    tpl_kwargs = chat_template_kwargs or {"enable_thinking": True}

    t0 = time.time()
    obs = env.reset(scenario=scenario_id)
    episode_id = env.state.episode_id

    budget = max_steps if max_steps is not None else obs.steps_remaining
    if budget <= 0:
        budget = 20

    metrics = EpisodeMetrics(
        scenario_id=scenario_id,
        episode_id=episode_id,
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_user_turn(obs)},
    ]

    last_reward: Optional[float] = None

    for step in range(budget):
        trimmed = _trim_history(messages, history_window)
        try:
            prompt_text = tokenizer.apply_chat_template(
                trimmed,
                tokenize=False,
                add_generation_prompt=True,
                **tpl_kwargs,
            )
            inputs = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(device)
        except Exception as e:
            metrics.error = f"tokenization failed at step {step}: {e}"
            break

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                use_cache=True,
            )

        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        parsed = parse_model_output(gen_text)
        command = parsed["command"]
        if not command:
            metrics.command_parse_failures += 1
            # Fall back to a safe observation action
            command = "check_status"

        metrics.action_history.append(command)

        # Step env
        try:
            action = DcOpsAction(
                command=command,
                reasoning=parsed["reasoning"] or parsed["think"][:500],
            )
            obs = env.step(action)
        except Exception as e:
            metrics.error = f"env.step failed at step {step}: {e}"
            break

        step_reward = getattr(obs, "reward", None)
        if step_reward is None:
            step_reward = 0.0
        metrics.cumulative_reward += float(step_reward)
        metrics.reward_per_step.append(float(step_reward))
        metrics.steps = step + 1
        last_reward = float(step_reward)

        # Track unknown-command feedback from env
        action_result = (getattr(obs, "action_result", "") or "").lower()
        if "unknown command" in action_result:
            metrics.unknown_commands += 1

        if verbose:
            print(f"  [{scenario_id}] step={step+1:02d}  cmd={command:<35s}"
                  f"  r={step_reward:+.3f}  cum={metrics.cumulative_reward:+.3f}")

        # Append the assistant turn and the next user turn to history
        messages.append({"role": "assistant", "content": gen_text})
        if not getattr(obs, "done", False):
            messages.append({"role": "user", "content": format_user_turn(obs)})
        else:
            break

    metrics.resolved = detect_resolved(obs, last_reward or 0.0)
    metrics.wall_s = time.time() - t0
    return metrics


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate_metrics(episodes: List[EpisodeMetrics]) -> Dict[str, Any]:
    """Compute summary statistics over a list of EpisodeMetrics."""
    if not episodes:
        return {}
    rewards = [e.cumulative_reward for e in episodes]
    steps = [e.steps for e in episodes]
    resolved = [e.resolved for e in episodes]
    parse_failures = sum(e.command_parse_failures for e in episodes)
    unknown = sum(e.unknown_commands for e in episodes)

    n = len(episodes)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    stdev = lambda xs: (
        (sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0
    )

    return {
        "n_episodes": n,
        "mean_cum_reward": mean(rewards),
        "std_cum_reward": stdev(rewards),
        "min_cum_reward": min(rewards) if rewards else 0.0,
        "max_cum_reward": max(rewards) if rewards else 0.0,
        "mean_steps": mean(steps),
        "resolved_rate": sum(resolved) / n,
        "total_parse_failures": parse_failures,
        "total_unknown_commands": unknown,
        "mean_wall_s": mean([e.wall_s for e in episodes]),
    }
