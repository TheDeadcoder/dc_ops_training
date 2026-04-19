"""
Main generation driver.

Spins up an asyncio worker pool. Each worker:
  1. Pops a (scenario_key, seed) job from the planned queue.
  2. Resets a fresh in-process DcOpsEnvironment.
  3. Loops up to MAX_AGENT_STEPS_PER_EPISODE turns:
       - Format user/dashboard turn
       - Send conversation to vLLM teacher
       - Parse <think> + <reasoning> + <command>
       - Step the env with the parsed command
       - Append (user, assistant) pair to the conversation
  4. Records per-episode metrics.
  5. Writes the episode to a per-worker JSONL shard.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from config import (
    AGENT_SYSTEM_PROMPT,
    MAX_AGENT_STEPS_PER_EPISODE,
    SCENARIO_PLAN,
    TEACHER_MAX_CONCURRENT,
)
from env_runner import format_user_turn, reset_episode, step_episode
from teacher_client import TeacherClient


OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "data_out"))
RAW_DIR = OUTPUT_DIR / "raw_episodes"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# Strip <think>...</think> from an assistant turn before putting it into the
# teacher's conversation history. Rationale: the teacher doesn't need to re-
# read its own prior chain-of-thought to pick the next action — the
# <reasoning>+<command> summary is sufficient context. This keeps each
# teacher request's input budget well under vLLM's model-context limit even
# on long multi-turn episodes.
#
# The saved training data (`conversations`) keeps the full <think> so the
# student still learns to think.
_THINK_STRIP_RE = re.compile(r"<think\s*>.*?</think\s*>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think_for_history(assistant_text: str) -> str:
    return _THINK_STRIP_RE.sub("", assistant_text).strip()


def _format_assistant_turn(think: str, reasoning: str, command: str) -> str:
    """Three-block agent reply saved into training conversations."""
    return (
        f"<think>\n{think}\n</think>\n"
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<command>\n{command}\n</command>"
    )


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------
async def rollout_one_episode(
    teacher: TeacherClient,
    scenario_key: str,
    seed: int,
    episode_idx: int,
) -> dict:
    """Run a single episode with the teacher in the loop. Returns a record."""
    try:
        env, dashboard, action_result, steps_remaining, meta = reset_episode(scenario_key, seed)
    except Exception as e:
        return {
            "episode_idx": episode_idx,
            "scenario_key": scenario_key,
            "seed": seed,
            "error": f"reset_failed: {type(e).__name__}: {e}",
        }

    conversations: list[dict] = [{"from": "system", "value": AGENT_SYSTEM_PROMPT}]
    teacher_history: list[dict] = []

    cmd_counts: Counter = Counter()
    rewards: list[float] = []
    parse_failures = 0
    invalid_commands = 0
    self_correction_leaks = 0
    escalated = False
    resolved = False

    max_steps = min(MAX_AGENT_STEPS_PER_EPISODE, meta.step_budget)

    for step in range(max_steps):
        # 1) Build user turn
        user_text = format_user_turn(action_result, steps_remaining, dashboard)
        conversations.append({"from": "human", "value": user_text})

        # 2) Ask teacher
        request_id = f"ep{episode_idx}_step{step}"
        result = await teacher.turn(
            agent_system_prompt=AGENT_SYSTEM_PROMPT,
            history=teacher_history,
            user_turn=user_text,
            request_id=request_id,
        )
        if not result.success:
            parse_failures += 1
            conversations.pop()
            break

        if result.leaked_self_correction:
            self_correction_leaks += 1

        # 3) Step env
        try:
            new_dashboard, new_action_result, reward, done, new_steps_remaining = step_episode(
                env, result.command,
            )
        except Exception as e:
            conversations.pop()
            return {
                "episode_idx": episode_idx,
                "scenario_key": scenario_key,
                "seed": seed,
                "error": f"step_failed: {type(e).__name__}: {e}",
            }

        ar_lower = new_action_result.lower()
        if any(s in ar_lower for s in ("unknown command", "invalid", "out of range", "not found")):
            invalid_commands += 1
        if result.command.strip().lower().startswith("escalate"):
            escalated = True

        # 4) Build assistant turn — three-block format
        assistant_text = _format_assistant_turn(
            think=result.think,
            reasoning=result.reasoning,
            command=result.command,
        )
        # Saved training data: keep FULL three-block text (student learns to think)
        conversations.append({"from": "gpt", "value": assistant_text})
        # Teacher's working context: strip <think> to stay under input budget
        # across long multi-turn episodes. The <reasoning>+<command> is enough
        # for the teacher to track what it has already done.
        assistant_text_for_history = _strip_think_for_history(assistant_text)
        teacher_history.append({"role": "user", "content": user_text})
        teacher_history.append({"role": "assistant", "content": assistant_text_for_history})

        cmd_name = result.command.split()[0].lower() if result.command else "unknown"
        cmd_counts[cmd_name] += 1
        rewards.append(reward)

        # 5) Advance state
        dashboard = new_dashboard
        action_result = new_action_result
        steps_remaining = new_steps_remaining

        if done:
            final_alert = (env._alert or "").lower()
            resolved = any(kw in final_alert for kw in (
                "stabilized", "completed", "optimized", "resolved", "investigated",
            ))
            break

    n_agent_turns = sum(1 for c in conversations if c["from"] == "gpt")
    cum_reward = sum(rewards)
    avg_reward = (cum_reward / n_agent_turns) if n_agent_turns else 0.0

    return {
        "conversations": conversations,
        "metadata": {
            "episode_idx": episode_idx,
            "scenario_key": scenario_key,
            "scenario_id": meta.scenario_id,
            "scenario_type": meta.scenario_type,
            "difficulty": meta.difficulty,
            "step_budget": meta.step_budget,
            "target_unit": meta.target_unit,
            "seed": seed,
            "n_agent_turns": n_agent_turns,
            "cumulative_reward": cum_reward,
            "avg_reward": avg_reward,
            "rewards_per_step": rewards,
            "command_counts": dict(cmd_counts),
            "parse_failures": parse_failures,
            "invalid_commands": invalid_commands,
            "self_correction_leaks": self_correction_leaks,
            "escalated": escalated,
            "resolved": resolved,
        },
    }


# ---------------------------------------------------------------------------
# Worker pool driver
# ---------------------------------------------------------------------------
async def worker_loop(
    worker_id: int,
    teacher: TeacherClient,
    job_queue: asyncio.Queue,
    out_path: Path,
    progress_state: dict,
    shutdown_event: asyncio.Event | None = None,
):
    with out_path.open("a", encoding="utf-8") as f:
        while True:
            # Honor graceful shutdown before pulling another job. The episode
            # already in flight will finish (it's blocking on the await
            # inside rollout_one_episode), but no new ones start.
            if shutdown_event is not None and shutdown_event.is_set():
                return
            job = await job_queue.get()
            if job is None:
                job_queue.task_done()
                return
            scenario_key, seed, idx = job
            try:
                rec = await rollout_one_episode(teacher, scenario_key, seed, idx)
            except Exception as e:
                rec = {
                    "episode_idx": idx,
                    "scenario_key": scenario_key,
                    "seed": seed,
                    "error": f"unhandled: {type(e).__name__}: {e}",
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())   # guarantee on-disk durability for status.sh
            progress_state["done"] += 1
            done = progress_state["done"]
            total = progress_state["total"]
            if done % 10 == 0:
                elapsed = time.time() - progress_state["t0"]
                rate = done / elapsed if elapsed > 0 else 0.0
                eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
                print(
                    f"[progress] {done}/{total}  "
                    f"({rate:.2f} eps/s, ETA {eta_min:.1f} min)",
                    flush=True,
                )
            job_queue.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main_async(num_workers: int):
    teacher = TeacherClient()
    job_queue: asyncio.Queue = asyncio.Queue()
    shutdown_event = asyncio.Event()

    # Register signal handlers so stop.sh (SIGTERM) and Ctrl-C (SIGINT)
    # both trigger a graceful drain: workers stop pulling new jobs and
    # finish whatever they're currently rolling out, then we flush + exit.
    loop = asyncio.get_running_loop()

    def _on_signal(signame: str):
        print(f"\n[shutdown] received {signame}, draining ...", flush=True)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal, sig.name)
        except (NotImplementedError, RuntimeError):
            # add_signal_handler is Unix-only; ignore on platforms without it
            pass

    rng = random.Random(20260418)
    next_idx = 0
    all_jobs: list[tuple[str, int, int]] = []
    for scenario_key, count in SCENARIO_PLAN.items():
        for _ in range(count):
            seed = rng.randint(1, 2**31 - 1)
            all_jobs.append((scenario_key, seed, next_idx))
            next_idx += 1
    rng.shuffle(all_jobs)

    total = len(all_jobs)
    print(f"[plan] total jobs: {total}", flush=True)
    for k, v in SCENARIO_PLAN.items():
        print(f"[plan]  {k:20s} {v}", flush=True)

    for j in all_jobs:
        job_queue.put_nowait(j)
    for _ in range(num_workers):
        job_queue.put_nowait(None)

    progress_state = {"done": 0, "total": total, "t0": time.time()}
    workers = [
        asyncio.create_task(worker_loop(
            worker_id=i, teacher=teacher, job_queue=job_queue,
            out_path=RAW_DIR / f"shard_{i:02d}.jsonl",
            progress_state=progress_state,
            shutdown_event=shutdown_event,
        ))
        for i in range(num_workers)
    ]

    await asyncio.gather(*workers)
    await teacher.aclose()

    elapsed = time.time() - progress_state["t0"]
    print(
        f"[done] {progress_state['done']} episodes in {elapsed/60:.1f} min "
        f"({progress_state['done']/elapsed:.2f} eps/s)",
        flush=True,
    )
    print(f"[done] shards written to {RAW_DIR}/", flush=True)


def main():
    nw = int(os.environ.get("NUM_WORKERS", str(TEACHER_MAX_CONCURRENT)))
    asyncio.run(main_async(num_workers=nw))


if __name__ == "__main__":
    main()
