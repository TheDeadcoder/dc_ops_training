"""
Configuration for DC-Ops SFT data generation.

Format choice — three blocks per agent turn:

  <think>
  [R1's natural messy chain of thought — exploration, self-correction, all of
  it. The model thinks freely here in its native pretraining format.]
  </think>
  <reasoning>
  [Distilled gist of the decision — ≤200 words, no self-correction, just the
  final structured conclusion. This is the canonical training signal.]
  </reasoning>
  <command>
  diagnose CRAC-3
  </command>

Why three blocks:
  - <think> lets the teacher (R1-Distill) and the student (Qwen3-8B with
    enable_thinking=True) operate in their native format — no fighting
    pretraining.
  - <reasoning> is a clean, predictable summary that the student learns to
    produce after thinking. It's what shows up in the operations log.
  - <command> is what the env actually parses for reward.

If GRPO rollouts feel slow at training time, you can later switch the
student to enable_thinking=False at GRPO time — it will still produce
<reasoning>+<command> from SFT memory, but skip <think>. ~5x faster
rollouts.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# vLLM teacher (served via vLLM's OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
TEACHER_BASE_URL = "http://localhost:8000/v1"
TEACHER_API_KEY  = "EMPTY"           # vLLM accepts any string
TEACHER_MODEL    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

TEACHER_MAX_TOKENS  = 2048           # think (~1200) + reasoning (~400) + cmd + slack
# NOTE: vLLM must be launched with --max-model-len >= 24576 for safe operation.
# The growing conversation history (system + N dashboards + N stripped assistant
# turns) crosses smaller context limits at the tail of long episodes:
#   max-model-len 8192  -> errors at turn 2
#   max-model-len 16384 -> errors at turn ~9
#   max-model-len 20000 -> borderline at turn 12
#   max-model-len 24576 -> safe through turn 12 with slack
TEACHER_TEMPERATURE = 0.6
TEACHER_TOP_P       = 0.95
TEACHER_TIMEOUT_S   = 240.0
TEACHER_MAX_CONCURRENT = 28          # in-flight requests against vLLM
                                     # KV cache headroom on MI300X 192GB allows
                                     # ~28 comfortably at max-model-len 24576.
                                     # Going higher risks tail-event preemption
                                     # if a cluster of episodes all hit turn 12.
TEACHER_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Generation plan — sums to 1520 attempts, target ~1200 final after filtering
# ---------------------------------------------------------------------------
SCENARIO_PLAN: dict[str, int] = {
    "A1": 200, "A2": 280, "A4": 280,
    "B1": 160, "B3": 220, "B4": 180,
    "VAR_CRAC_STANDBY": 60,
    "VAR_CRAC_MAINT":   40,
    "VAR_GEN_LOWFUEL":  40,
    "VAR_UPS_MODE":     60,
}

# ---------------------------------------------------------------------------
# Reasoning length budget
#
# Target ≤200 words ≈ 1000–1200 chars at 5–6 chars/word.
# Hard cap leaves slack for the model to overshoot slightly while still
# being safely shorter than its <think> trace.
# ---------------------------------------------------------------------------
REASONING_HARD_CAP_CHARS = 1500       # ~250 words; truncated at sentence boundary
REASONING_TARGET_MAX_WORDS = 200      # what we tell the teacher
THINK_HARD_CAP_CHARS = 4000           # safety net only; model picks natural depth

# ---------------------------------------------------------------------------
# Filter thresholds applied AFTER generation
#
# KEEP_MIN_AGENT_TURNS = 1 — preserve fast correct trajectories (B1, VAR_*).
# <think> is OPTIONAL at filter time (chat-template glitches happen);
# <reasoning> is REQUIRED.
# ---------------------------------------------------------------------------
KEEP_MIN_AGENT_TURNS    = 1
KEEP_MIN_AVG_REWARD     = -0.20
KEEP_NO_INVALID_CMDS    = True
KEEP_NO_ESCALATION      = True
KEEP_MIN_REASONING_CHARS = 30         # drop turn if <reasoning> is junk-short
TARGET_FINAL_COUNT      = 1200

# ---------------------------------------------------------------------------
# Episode budget
# ---------------------------------------------------------------------------
MAX_AGENT_STEPS_PER_EPISODE = 12

# ---------------------------------------------------------------------------
# System prompt — used identically for teacher AND eventual student.
#
# Carefully structured to:
#   1. Encourage long, free-form thinking in <think>
#   2. Force a CONCISE, SELF-CORRECTION-FREE summary in <reasoning>
#   3. Single command in <command>
#
# The numbered-step template inside <reasoning> gives the model a structure
# to fill rather than rambling — this is the key trick to stop R1 from
# leaking "wait, actually..." into the summary block.
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = """You are DC-Ops Agent, an expert datacenter operations engineer. You manage a physics-based datacenter simulation. You observe a monitoring dashboard and issue exactly one operator command per turn to maintain thermal safety, power reliability, and energy efficiency.

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
