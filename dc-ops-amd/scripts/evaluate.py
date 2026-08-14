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

The episode outcome (resolved / crashed / timed out) is read from the
simulator's own signal — DcOpsObservation.resolved plus steps_remaining — not
by string-matching the alert text. Alongside the environment reward (the trained
objective), it reports an independent physics-outcome scorecard the model was
never rewarded for (src/scorecard.py), with bootstrap 95% CIs and a paired
base-vs-GRPO test on matched seeds.

Usage:
    python scripts/evaluate.py \
        --grpo-model ./outputs/dc_ops_grpo_final \
        --base-model unsloth/Qwen2.5-7B-Instruct \
        --scenarios A4 B4 \
        --n-seeds 20 \
        --temperature 0.0
    # Episodes run at each scenario's own declared horizon (A4/B4 = 20 steps).
    # Eval seeds default to a range disjoint from the GRPO training seeds.

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
    timed_out: bool = False
    # Physics-outcome scorecard (src/scorecard.py) — metrics the model was
    # never rewarded for. Independent of total_reward on purpose (issue 1.4).
    scorecard: dict = field(default_factory=dict)

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
    # Bootstrap 95% CIs (issue 1.6). None when n < 2.
    reward_ci_low: Optional[float] = None
    reward_ci_high: Optional[float] = None
    resolution_rate_ci_low: Optional[float] = None
    resolution_rate_ci_high: Optional[float] = None
    # Per-episode-averaged physics scorecard.
    mean_scorecard: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Statistics (issue 1.6): bootstrap CIs + a paired test on matched seeds.
# Hand-rolled so the script has no scipy/numpy dependency. Deterministic:
# the bootstrap RNG is seeded, so reported CIs are reproducible.
# ---------------------------------------------------------------------------
import random as _random  # noqa: E402


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng_seed: int = 12345,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for the mean. Returns (None, None) if n < 2."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return (None, None)
    rng = _random.Random(rng_seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_diff_test(
    results: list["EpisodeResult"],
    model_a: str,
    model_b: str,
    scenario_id: str,
    *,
    metric: str = "total_reward",
    rng_seed: int = 999,
) -> Optional[dict]:
    """Paired comparison of ``model_b`` vs ``model_a`` on matched seeds.

    Pairs episodes by seed (only seeds both models ran), computes the per-seed
    difference ``b - a``, and reports its mean with a bootstrap 95% CI. The
    difference is 'significant at 95%' when that CI excludes zero. Pairing on
    seed removes the shared per-episode difficulty variance, which is the whole
    point of running both models on the *same* seeds.
    """
    a = {r.seed: getattr(r, metric) for r in results
         if r.model_name == model_a and r.scenario_id == scenario_id}
    b = {r.seed: getattr(r, metric) for r in results
         if r.model_name == model_b and r.scenario_id == scenario_id}
    seeds = sorted(set(a) & set(b))
    diffs = [float(b[s]) - float(a[s]) for s in seeds]
    if not diffs:
        return None
    mean_diff = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs, rng_seed=rng_seed)
    significant = lo is not None and hi is not None and (lo > 0 or hi < 0)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "scenario_id": scenario_id,
        "metric": metric,
        "n_pairs": len(diffs),
        "mean_diff": mean_diff,
        "ci_low": lo,
        "ci_high": hi,
        "significant_95": significant,
    }


def _mean_scorecard(scorecards: list[dict]) -> dict:
    """Average each numeric scorecard field across episodes, skipping None."""
    if not scorecards:
        return {}
    keys = set().union(*(sc.keys() for sc in scorecards))
    out: dict = {}
    for k in sorted(keys):
        vals = [sc[k] for sc in scorecards
                if isinstance(sc.get(k), (int, float))]
        out[k] = (sum(vals) / len(vals)) if vals else None
    return out


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
    step_budget: Optional[int] = None,
) -> EpisodeResult:
    """Run one full episode and return results.

    `step_budget=None` (the default) lets the environment use the scenario's
    own declared horizon — A4/B4 declare 20 (issue 1.2). Passing an int
    overrides it (used by the horizon sweep). The loop length is taken from the
    environment's `steps_remaining` after reset, so the loop and the dashboard
    can never disagree the way the old hardcoded `range(10)` did.
    """
    from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
    from dc_ops_env.models import DcOpsAction
    from src.rewards import extract_command, extract_reasoning
    from src.prompts import user_content_from_obs
    from src.scorecard import PhysicsScorecard, zone_thresholds_from_env

    env = DcOpsEnvironment()
    reset_kwargs: dict = {"scenario": scenario_id, "seed": seed}
    if step_budget is not None:
        reset_kwargs["step_budget"] = step_budget
    obs = env.reset(**reset_kwargs)

    # The environment is authoritative on the horizon. Right after reset,
    # steps_remaining == the full budget for this episode.
    budget = obs.steps_remaining if obs.steps_remaining and obs.steps_remaining > 0 \
        else (step_budget or 20)

    scorecard = PhysicsScorecard(zone_thresholds=zone_thresholds_from_env(env))
    scorecard.observe_reset(obs)

    per_step_rewards: list[float] = []
    actions: list[str] = []
    resolved = False
    crashed = False
    timed_out = False

    for step in range(budget):
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
        scorecard.observe_step(cmd, obs)

        if obs.done:
            # Outcome comes from the simulator's own signal (issue 1.3), not
            # from string-matching the alert. A crash ends the episode with
            # budget still remaining; a timeout ends with steps_remaining == 0.
            resolved = bool(obs.resolved)
            timed_out = (not resolved) and obs.steps_remaining <= 0
            crashed = (not resolved) and not timed_out
            break
    else:
        # Loop ran the full budget without the env ever setting done.
        timed_out = not resolved

    return EpisodeResult(
        model_name=runner.model.__class__.__name__,  # overwritten by caller
        scenario_id=scenario_id,
        seed=seed,
        resolved=resolved,
        crashed=crashed,
        timed_out=timed_out,
        steps_taken=len(per_step_rewards),
        step_budget=budget,
        total_reward=sum(per_step_rewards),
        per_step_rewards=per_step_rewards,
        actions=actions,
        scorecard=scorecard.summary(),
    )


def aggregate(results: list[EpisodeResult], model_name: str, scenario_id: str) -> AggregateStats:
    subset = [r for r in results if r.model_name == model_name and r.scenario_id == scenario_id]
    if not subset:
        return AggregateStats(model_name, scenario_id, 0, 0.0, 0.0, 0.0, None, 0.0)

    n = len(subset)
    total_rewards = [r.total_reward for r in subset]
    resolved_flags = [1.0 if r.resolved else 0.0 for r in subset]
    resolved_results = [r for r in subset if r.resolved]
    resolution_rate = len(resolved_results) / n
    mean_steps_res = (
        mean(r.steps_taken for r in resolved_results) if resolved_results else None
    )
    all_step_rewards = [r for ep in subset for r in ep.per_step_rewards]

    reward_lo, reward_hi = bootstrap_ci(total_rewards, rng_seed=hash(
        (model_name, scenario_id, "reward")) & 0xFFFFFFFF)
    res_lo, res_hi = bootstrap_ci(resolved_flags, rng_seed=hash(
        (model_name, scenario_id, "resolved")) & 0xFFFFFFFF)

    return AggregateStats(
        model_name=model_name,
        scenario_id=scenario_id,
        n_episodes=n,
        resolution_rate=resolution_rate,
        mean_total_reward=mean(total_rewards),
        std_total_reward=stdev(total_rewards) if n > 1 else 0.0,
        mean_steps_to_resolution=mean_steps_res,
        mean_per_step_reward=mean(all_step_rewards) if all_step_rewards else 0.0,
        reward_ci_low=reward_lo,
        reward_ci_high=reward_hi,
        resolution_rate_ci_low=res_lo,
        resolution_rate_ci_high=res_hi,
        mean_scorecard=_mean_scorecard([r.scorecard for r in subset]),
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
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Explicit seeds (one episode per seed per scenario). "
                        "Overrides --n-seeds.")
    p.add_argument("--n-seeds", type=int, default=20,
                   help="Number of seeds per scenario when --seeds is not given "
                        "(default 20; issue 1.6 needs n>=20 for usable CIs).")
    p.add_argument("--seed-base", type=int, default=1_000_000,
                   help="First seed; seeds are seed-base + i*step. Default is far "
                        "from the GRPO training seed ranges (1000-1203, 5000-5702 "
                        "in src/grpo_data.py) so evaluation is a genuine held-out "
                        "test set, not the training states (Part 5 #3).")
    p.add_argument("--seed-step", type=int, default=1000,
                   help="Spacing between generated seeds.")
    p.add_argument("--step-budget", type=int, default=None,
                   help="Max steps per episode. Default: use each scenario's own "
                        "declared budget (A4/B4 = 20; issue 1.2). Set an int to "
                        "override, e.g. for a horizon sweep.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Generation temperature (0=greedy).")
    p.add_argument("--max-seq-length", type=int, default=3072)
    p.add_argument("--hf-source", default="Melikshah/dc-ops-sft-data",
                   help="HF dataset for system prompt fallback.")
    p.add_argument("--output", default="./outputs/eval_results.json",
                   help="Where to write JSON results.")
    return p.parse_args()


def _ci_str(lo: Optional[float], hi: Optional[float], fmt: str = ".3f") -> str:
    if lo is None or hi is None:
        return "     —      "
    return f"[{lo:{fmt}}, {hi:{fmt}}]"


def print_table(stats_list: list[AggregateStats]):
    """Reward / resolution table WITH bootstrap 95% CIs (issue 1.6).

    NOTE: mean reward here is the *training* signal. It answers "did the number
    we optimised move?" — not "did the model get better". Read it alongside the
    physics scorecard below, which the model was never rewarded for (issue 1.4).
    """
    print("\n" + "=" * 92)
    print("REWARD / RESOLUTION  (reward == the trained objective; interpret with care)")
    print("-" * 92)
    print(f"{'Model':<14} {'Scen':<6} {'N':>3}  {'ResRate':>8} "
          f"{'Res 95% CI':>16}  {'MeanRew':>8} {'Reward 95% CI':>18}  {'Steps':>6}")
    print("-" * 92)
    for s in stats_list:
        steps_str = f"{s.mean_steps_to_resolution:.1f}" if s.mean_steps_to_resolution else "—"
        res_ci = _ci_str(s.resolution_rate_ci_low, s.resolution_rate_ci_high, ".2f")
        rew_ci = _ci_str(s.reward_ci_low, s.reward_ci_high)
        print(f"{s.model_name:<14} {s.scenario_id:<6} {s.n_episodes:>3}  "
              f"{s.resolution_rate:>8.1%} {res_ci:>16}  "
              f"{s.mean_total_reward:>8.3f} {rew_ci:>18}  {steps_str:>6}")
    print("=" * 92)


def print_physics_table(stats_list: list[AggregateStats]):
    """Physics-outcome scorecard (issue 1.4) — metrics NOT in the reward."""
    cols = [
        ("peak_inlet_c", "PeakInlet", ".1f"),
        ("degree_min_over_allowable", "DegMin>Allow", ".1f"),
        ("min_ups_soc", "MinUPS_SoC", ".2f"),
        ("generator_online_latency_s", "GenLat_s", ".0f"),
        ("mean_pue", "PUE", ".3f"),
        ("total_energy_kwh", "kWh", ".1f"),
        ("invalid_command_rate", "InvCmd", ".2f"),
        ("malformed_target_rate", "BadTgt", ".2f"),
    ]
    print("\n" + "=" * 100)
    print("PHYSICS-OUTCOME SCORECARD  (independent of reward — the real evidence)")
    print("-" * 100)
    header = f"{'Model':<14} {'Scen':<6}" + "".join(f"{c[1]:>13}" for c in cols)
    print(header)
    print("-" * 100)
    for s in stats_list:
        row = f"{s.model_name:<14} {s.scenario_id:<6}"
        for key, _, fmt in cols:
            v = s.mean_scorecard.get(key)
            row += f"{('—' if v is None else format(v, fmt)):>13}"
        print(row)
    print("=" * 100)


def print_paired_tests(tests: list[dict]):
    if not tests:
        return
    print("\n" + "=" * 92)
    print("PAIRED TEST  (grpo − base on matched seeds; bootstrap 95% CI of mean diff)")
    print("-" * 92)
    print(f"{'Scenario':<10} {'Metric':<16} {'Pairs':>6} {'MeanDiff':>10} "
          f"{'95% CI':>22} {'Sig@95%':>8}")
    print("-" * 92)
    for t in tests:
        ci = _ci_str(t["ci_low"], t["ci_high"])
        sig = "yes" if t["significant_95"] else "no"
        print(f"{t['scenario_id']:<10} {t['metric']:<16} {t['n_pairs']:>6} "
              f"{t['mean_diff']:>10.3f} {ci:>22} {sig:>8}")
    print("=" * 92)


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

    # Resolve seeds. Both models run the SAME seeds so the paired test can
    # match episodes seed-for-seed (issue 1.6). With the environment now
    # seeding initial conditions (env issue 1.1), distinct seeds are distinct
    # episodes rather than identical copies.
    if args.seeds:
        seeds = list(args.seeds)
    else:
        seeds = [args.seed_base + i * args.seed_step for i in range(args.n_seeds)]
    print(f"[eval] {len(seeds)} seeds/scenario: {seeds[0]}..{seeds[-1]}  "
          f"step_budget={'scenario-default' if args.step_budget is None else args.step_budget}")

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
            for seed in seeds:
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
                print(f"{status}  steps={result.steps_taken}/{result.step_budget}  "
                      f"total_reward={result.total_reward:.3f}  "
                      f"actions={result.actions}")

        runner.close()

    # Aggregate stats
    all_stats: list[AggregateStats] = []
    for (model_name, _) in models_to_eval:
        for scenario_id in args.scenarios:
            stats = aggregate(all_results, model_name, scenario_id)
            all_stats.append(stats)

    # Paired tests: grpo vs base on matched seeds (only if base was run).
    paired_tests: list[dict] = []
    model_names = [m for (m, _) in models_to_eval]
    if "base" in model_names and "grpo" in model_names:
        for scenario_id in args.scenarios:
            for metric in ("total_reward", "resolved"):
                t = paired_diff_test(all_results, "base", "grpo", scenario_id, metric=metric)
                if t:
                    paired_tests.append(t)

    print_table(all_stats)
    print_physics_table(all_stats)
    print_paired_tests(paired_tests)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "config": vars(args),
        "seeds": seeds,
        "aggregate": [asdict(s) for s in all_stats],
        "paired_tests": paired_tests,
        "episodes": [asdict(r) for r in all_results],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[eval] results saved → {args.output}")


if __name__ == "__main__":
    main()