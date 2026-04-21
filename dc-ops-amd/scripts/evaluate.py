"""
Evaluation: Base model vs GRPO-trained model on hard DC-Ops scenarios.

Runs full multi-step episodes on:
  - A4 (CRAC Failure Cascade — Hard thermal)
  - B4 (Power Failure Cascade — Hard power)

These are the hardest scenarios because they require sequential multi-step
reasoning: diagnose, identify compound failures, redistribute load/cooling,
and avoid catastrophic outcomes under time pressure.

For each model × scenario × seed, the script:
  1. Resets the environment.
  2. Runs the model's policy for up to `step_budget` steps (model generates
     <reasoning> + <command> at each step from the live dashboard).
  3. Records per-step env rewards (physics simulator), total episode reward,
     resolution flag, and steps-to-resolution.

Usage:
    python scripts/evaluate.py \
        --grpo-model ./outputs/dc_ops_grpo_final \
        --base-model unsloth/Qwen2.5-7B-Instruct \
        --scenarios A4 B4 \
        --seeds 100 200 300 400 500 \
        --temperature 0.0

    # Use --no-base to skip base model (saves time if you just re-ran GRPO):
    python scripts/evaluate.py --grpo-model ./outputs/dc_ops_grpo_final --no-base
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import asdict, dataclass, field
from statistics import mean, stdev
from typing import Optional

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src import rocm_env  # noqa: F401


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    model_name: str
    scenario_id: str
    seed: int
    resolved: bool
    steps_taken: int
    step_budget: int
    total_reward: float
    per_step_rewards: list[float]
    actions: list[str]
    crashed: bool = False

    @property
    def steps_to_resolution(self) -> Optional[int]:
        return self.steps_taken if self.resolved else None


@dataclass
class AggregateStats:
    model_name: str
    scenario_id: str
    n_episodes: int
    resolution_rate: float
    mean_total_reward: float
    std_total_reward: float
    mean_steps_to_resolution: Optional[float]
    mean_per_step_reward: float


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------
class ModelRunner:
    """Wraps Unsloth model loading and greedy/stochastic generation."""

    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 3072,
        load_in_4bit: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ):
        from unsloth import FastLanguageModel

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        print(f"[eval] loading model from {model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=False,  # no vLLM needed for serial eval
        )
        FastLanguageModel.for_inference(self.model)
        print(f"[eval] model loaded")

    def generate(self, system_prompt: str, user_content: str) -> str:
        """Generate a single response given system + user content."""
        from src.prompts import messages_to_prompt

        prompt = messages_to_prompt(self.tokenizer, system_prompt, user_content)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        do_sample = self.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.95

        with __import__("torch").no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def close(self):
        """Free GPU memory after evaluation."""
        import torch, gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    runner: ModelRunner,
    system_prompt: str,
    scenario_id: str,
    seed: int,
    step_budget: int = 10,
) -> EpisodeResult:
    """Run one full episode and return results."""
    from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
    from dc_ops_env.models import DcOpsAction
    from src.rewards import extract_command, extract_reasoning
    from src.prompts import user_content_from_obs
    from src.constants import CRASH_KEYWORDS, RESOLVE_KEYWORDS

    env = DcOpsEnvironment()
    obs = env.reset(scenario=scenario_id, seed=seed)

    per_step_rewards: list[float] = []
    actions: list[str] = []
    resolved = False
    crashed = False

    for step in range(step_budget):
        user_content = user_content_from_obs(obs)
        response = runner.generate(system_prompt, user_content)

        cmd = extract_command(response)
        reasoning = extract_reasoning(response) or ""

        if cmd is None:
            # Format failure — counts as a wasted step
            cmd = "wait"

        obs = env.step(DcOpsAction(command=cmd, reasoning=reasoning))
        per_step_rewards.append(float(obs.reward))
        actions.append(cmd)

        if obs.done:
            alert_l = (obs.alert or "").lower()
            resolved = any(k in alert_l for k in RESOLVE_KEYWORDS)
            crashed  = any(k in alert_l for k in CRASH_KEYWORDS)
            break

    return EpisodeResult(
        model_name=runner.model.__class__.__name__,  # overwritten by caller
        scenario_id=scenario_id,
        seed=seed,
        resolved=resolved,
        crashed=crashed,
        steps_taken=len(per_step_rewards),
        step_budget=step_budget,
        total_reward=sum(per_step_rewards),
        per_step_rewards=per_step_rewards,
        actions=actions,
    )


def aggregate(results: list[EpisodeResult], model_name: str, scenario_id: str) -> AggregateStats:
    subset = [r for r in results if r.model_name == model_name and r.scenario_id == scenario_id]
    if not subset:
        return AggregateStats(model_name, scenario_id, 0, 0.0, 0.0, 0.0, None, 0.0)

    n = len(subset)
    total_rewards = [r.total_reward for r in subset]
    resolved_results = [r for r in subset if r.resolved]
    resolution_rate = len(resolved_results) / n
    mean_steps_res = (
        mean(r.steps_taken for r in resolved_results) if resolved_results else None
    )
    all_step_rewards = [r for ep in subset for r in ep.per_step_rewards]

    return AggregateStats(
        model_name=model_name,
        scenario_id=scenario_id,
        n_episodes=n,
        resolution_rate=resolution_rate,
        mean_total_reward=mean(total_rewards),
        std_total_reward=stdev(total_rewards) if n > 1 else 0.0,
        mean_steps_to_resolution=mean_steps_res,
        mean_per_step_reward=mean(all_step_rewards) if all_step_rewards else 0.0,
    )


# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------
def load_system_prompt(model_path: str, data_cfg_hf: str) -> str:
    sp_path = os.path.join(model_path, "system_prompt.txt")
    if os.path.exists(sp_path):
        with open(sp_path) as f:
            return f.read()
    # Fall back: derive from dataset
    from datasets import load_dataset
    from src.prompts import rewrite_system_prompt
    ds = load_dataset(
        "json",
        data_files={"train": f"hf://datasets/{data_cfg_hf}/train.jsonl"},
        split="train",
    )
    return rewrite_system_prompt(ds[0]["conversations"][0]["value"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DC-Ops model evaluation")
    p.add_argument("--grpo-model", default="./outputs/dc_ops_grpo_final",
                   help="Path to GRPO-trained model dir (LoRA or merged).")
    p.add_argument("--base-model", default="unsloth/Qwen2.5-7B-Instruct",
                   help="Base model id or path (no LoRA).")
    p.add_argument("--no-base", action="store_true",
                   help="Skip base-model evaluation.")
    p.add_argument("--scenarios", nargs="+", default=["A4", "B4"],
                   help="Scenario IDs to evaluate on (default: A4 B4).")
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[100, 200, 300, 400, 500],
                   help="Random seeds (one episode per seed per scenario).")
    p.add_argument("--step-budget", type=int, default=10,
                   help="Max steps per episode.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Generation temperature (0=greedy).")
    p.add_argument("--max-seq-length", type=int, default=3072)
    p.add_argument("--hf-source", default="Melikshah/dc-ops-sft-data",
                   help="HF dataset for system prompt fallback.")
    p.add_argument("--output", default="./outputs/eval_results.json",
                   help="Where to write JSON results.")
    return p.parse_args()


def print_table(stats_list: list[AggregateStats]):
    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'Scenario':<10} {'N':>4}  {'ResRate':>8}  "
          f"{'MeanRew':>9}  {'StdRew':>8}  {'MeanStep':>9}  {'PerStep':>8}")
    print("-" * 80)
    for s in stats_list:
        steps_str = f"{s.mean_steps_to_resolution:.1f}" if s.mean_steps_to_resolution else "  —"
        print(f"{s.model_name:<30} {s.scenario_id:<10} {s.n_episodes:>4}  "
              f"{s.resolution_rate:>8.1%}  "
              f"{s.mean_total_reward:>9.3f}  "
              f"{s.std_total_reward:>8.3f}  "
              f"{steps_str:>9}  "
              f"{s.mean_per_step_reward:>8.3f}")
    print("=" * 80)


def main():
    args = parse_args()

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Determine which models to evaluate
    models_to_eval: list[tuple[str, str]] = []  # (name, path)
    if not args.no_base:
        models_to_eval.append(("base", args.base_model))
    models_to_eval.append(("grpo", args.grpo_model))

    all_results: list[EpisodeResult] = []

    for model_name, model_path in models_to_eval:
        print(f"\n{'='*60}")
        print(f"[eval] Evaluating: {model_name}  ({model_path})")
        print(f"{'='*60}")

        system_prompt = load_system_prompt(model_path, args.hf_source)
        runner = ModelRunner(
            model_path=model_path,
            max_seq_length=args.max_seq_length,
            temperature=args.temperature,
        )

        for scenario_id in args.scenarios:
            print(f"\n[eval] Scenario {scenario_id}:")
            for seed in args.seeds:
                print(f"  seed={seed} ...", end=" ", flush=True)
                result = run_episode(
                    runner=runner,
                    system_prompt=system_prompt,
                    scenario_id=scenario_id,
                    seed=seed,
                    step_budget=args.step_budget,
                )
                result.model_name = model_name
                all_results.append(result)

                status = "✓ RESOLVED" if result.resolved else ("✗ CRASHED" if result.crashed else "— timeout")
                print(f"{status}  steps={result.steps_taken}  "
                      f"total_reward={result.total_reward:.3f}  "
                      f"actions={result.actions}")

        runner.close()

    # Aggregate stats
    all_stats: list[AggregateStats] = []
    for (model_name, _) in models_to_eval:
        for scenario_id in args.scenarios:
            stats = aggregate(all_results, model_name, scenario_id)
            all_stats.append(stats)

    print_table(all_stats)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "config": vars(args),
        "aggregate": [asdict(s) for s in all_stats],
        "episodes": [asdict(r) for r in all_results],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[eval] results saved → {args.output}")


if __name__ == "__main__":
    main()