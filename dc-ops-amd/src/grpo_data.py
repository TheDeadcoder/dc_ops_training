"""GRPO prompt-dataset builder."""

from __future__ import annotations

import random
from typing import Any

from datasets import Dataset

from .constants import SCENARIO_WARMUP_ACTIONS, SCENARIOS
from .prompts import messages_to_prompt, user_content_from_obs


def build_grpo_prompts(
    tokenizer,
    system_prompt: str,
    *,
    num_initial: int = 30,
    num_midgame: int = 55,
    seed: int = 42,
) -> Dataset:
    """Build the GRPO prompt dataset.

    Produces ~(num_initial + num_midgame) * 6 prompts.

    Each prompt is generated with a distinct seed. The environment now seeds
    initial conditions in reset() (env issue 1.1), so distinct seeds produce
    genuinely distinct starting states — jittered zone/outside temperatures,
    UPS charge, per-rack IT load, and fault timing — rather than the identical
    screens the old seedless env produced. Initial-state prompts therefore vary
    by initial-condition jitter; mid-game prompts additionally vary by which
    warmup-action sequence was pre-rolled, landing the agent in different
    slices of state space.

    Seeds here (initial: 1000 + i*7 -> 1000..1203; mid-game: 5000 + j*13 ->
    5000..5702) are kept disjoint from the evaluation seed range (evaluate.py
    defaults to --seed-base 1_000_000) so that GRPO never trains on an episode
    the evaluation later scores.
    """
    from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
    from dc_ops_env.models import DcOpsAction

    rng = random.Random(seed)
    prompts: list[dict[str, Any]] = []
    env = DcOpsEnvironment()

    for scenario_id in SCENARIOS:
        # ---- (a) Initial-state prompts -------------------------------------
        for i in range(num_initial):
            s = 1000 + i * 7
            try:
                obs = env.reset(scenario=scenario_id, seed=s)
                prompts.append({
                    "prompt": messages_to_prompt(
                        tokenizer, system_prompt, user_content_from_obs(obs)
                    ),
                    "scenario_id":    scenario_id,
                    "seed":           s,
                    "warmup_actions": [],
                })
            except Exception as e:
                print(f"  [warn] initial prompt failed for {scenario_id} seed={s}: {e}")

        # ---- (b) Mid-game prompts (warmup pre-rolled) ----------------------
        warmups = SCENARIO_WARMUP_ACTIONS.get(scenario_id, [[]])
        for j in range(num_midgame):
            # Cycle through warmups AND vary the seed so each (warmup, seed)
            # combination lands in a distinct slice of state space.
            s = 5000 + j * 13
            warmup = warmups[j % len(warmups)]
            try:
                obs = env.reset(scenario=scenario_id, seed=s)
                aborted = False
                for cmd in warmup:
                    obs = env.step(DcOpsAction(command=cmd))
                    if obs.done:
                        aborted = True
                        break
                if aborted:
                    continue

                prompts.append({
                    "prompt": messages_to_prompt(
                        tokenizer, system_prompt, user_content_from_obs(obs)
                    ),
                    "scenario_id":    scenario_id,
                    "seed":           s,
                    "warmup_actions": list(warmup),
                })
            except Exception as e:
                print(f"  [warn] midgame prompt failed for {scenario_id} seed={s}: {e}")

    rng.shuffle(prompts)
    return Dataset.from_list(prompts)