#!/usr/bin/env python3
# Copyright (c) 2026. Licensed under BSD-3-Clause.
"""
Compare base Qwen3-8B and the SFT-fine-tuned checkpoint on DC-Ops scenarios.

Runs N episodes per scenario with each model, then prints a side-by-side
reward table and dumps the full episode-level data to JSON.

Usage:
  python scripts/eval_compare.py --config configs/sft.yaml
  python scripts/eval_compare.py --config configs/sft.yaml \\
      --base-model unsloth/Qwen3-8B \\
      --sft-model  ./outputs/qwen3-8b-dcops-sft-v1/final_merged_16bit \\
      --output     ./eval_results.json
"""

# Unsloth first (see train_sft.py comment)
from unsloth import FastLanguageModel  # noqa: I001
import torch  # noqa: E402

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from transformers import set_seed  # noqa: E402

from dc_ops_sft.env_eval import (  # noqa: E402
    EpisodeMetrics,
    aggregate_metrics,
    rollout_episode,
)


console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override cfg.model.name",
    )
    p.add_argument(
        "--sft-model",
        type=str,
        default=None,
        help="Override default (cfg.run.output_dir/final_merged_16bit)",
    )
    p.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="(Alt) Path to a LoRA adapter to load on top of base model. "
             "Use this if you only have LoRA (no merged save).",
    )
    p.add_argument("--output", type=str, default="./eval_comparison.json")
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episodes_per_scenario from config",
    )
    p.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Override scenario list from config (e.g. A2 B3)",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Only evaluate the SFT model",
    )
    return p.parse_args()


def load_model_for_inference(
    model_path: str,
    *,
    max_seq_length: int,
    dtype: str = "bfloat16",
    lora_adapter: Optional[str] = None,
):
    """Load a model in bf16 for inference; optionally attach a LoRA adapter."""
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    console.print(f"[dim]loading {model_path} (bf16)...[/]")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if lora_adapter:
        console.print(f"[dim]attaching LoRA adapter: {lora_adapter}[/]")
        model.load_adapter(lora_adapter)
    FastLanguageModel.for_inference(model)
    console.print(f"[dim]loaded in {time.time() - t0:.1f}s[/]")
    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    scenarios: List[str],
    episodes_per_scenario: int,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    history_window: int,
    chat_template_kwargs: Dict[str, Any],
    seed: int,
    verbose: bool,
    label: str,
) -> Dict[str, Any]:
    """Run env rollouts for every (scenario, episode_idx) pair and aggregate."""
    from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment

    env = DcOpsEnvironment()
    per_scenario: Dict[str, Dict[str, Any]] = {}
    all_episodes: List[EpisodeMetrics] = []

    for scen in scenarios:
        console.print(f"\n[bold]{label}[/] — scenario [cyan]{scen}[/] "
                      f"({episodes_per_scenario} episodes)")
        scen_episodes: List[EpisodeMetrics] = []

        for ep_idx in range(episodes_per_scenario):
            # Deterministic-ish per (scenario, episode) seed
            ep_seed = (seed + 1000 * ep_idx + sum(ord(c) for c in scen)) % (2**31 - 1)
            set_seed(ep_seed)

            t0 = time.time()
            m = rollout_episode(
                model,
                tokenizer,
                env,
                scenario_id=scen,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                history_window=history_window,
                chat_template_kwargs=chat_template_kwargs,
                verbose=verbose,
            )
            dt = time.time() - t0

            console.print(
                f"  ep {ep_idx+1}/{episodes_per_scenario}: "
                f"reward={m.cumulative_reward:+.2f}  "
                f"steps={m.steps:2d}  "
                f"resolved={'✓' if m.resolved else '✗'}  "
                f"parse_fail={m.command_parse_failures}  "
                f"unk={m.unknown_commands}  "
                f"wall={dt:.1f}s"
            )
            scen_episodes.append(m)
            all_episodes.append(m)

        per_scenario[scen] = aggregate_metrics(scen_episodes)

    return {
        "per_scenario": per_scenario,
        "overall": aggregate_metrics(all_episodes),
        "episodes": [asdict(e) for e in all_episodes],
    }


def print_comparison(
    base: Optional[Dict[str, Any]],
    sft: Dict[str, Any],
    scenarios: List[str],
) -> None:
    tbl = Table(title="Base vs SFT — DC-Ops Reward Comparison", show_lines=True)
    tbl.add_column("Scenario", style="cyan", no_wrap=True)
    if base:
        tbl.add_column("Base reward", justify="right")
        tbl.add_column("SFT reward", justify="right")
        tbl.add_column("Δ reward", justify="right", style="bold")
        tbl.add_column("Base solved", justify="right")
        tbl.add_column("SFT solved", justify="right")
    else:
        tbl.add_column("SFT reward", justify="right")
        tbl.add_column("SFT solved", justify="right")
        tbl.add_column("SFT steps", justify="right")

    for s in scenarios:
        sft_row = sft["per_scenario"].get(s, {})
        if base:
            b = base["per_scenario"].get(s, {})
            db = sft_row.get("mean_cum_reward", 0.0) - b.get("mean_cum_reward", 0.0)
            delta_str = (
                f"[green]+{db:.2f}[/]" if db > 0
                else (f"[red]{db:.2f}[/]" if db < 0 else "0.00")
            )
            tbl.add_row(
                s,
                f"{b.get('mean_cum_reward', 0):+.2f}±{b.get('std_cum_reward', 0):.2f}",
                f"{sft_row.get('mean_cum_reward', 0):+.2f}±{sft_row.get('std_cum_reward', 0):.2f}",
                delta_str,
                f"{b.get('resolved_rate', 0):.0%}",
                f"{sft_row.get('resolved_rate', 0):.0%}",
            )
        else:
            tbl.add_row(
                s,
                f"{sft_row.get('mean_cum_reward', 0):+.2f}±{sft_row.get('std_cum_reward', 0):.2f}",
                f"{sft_row.get('resolved_rate', 0):.0%}",
                f"{sft_row.get('mean_steps', 0):.1f}",
            )

    console.print(tbl)

    if base:
        b_overall = base["overall"]
        s_overall = sft["overall"]
        console.print(
            f"\n[bold]Overall:[/]  "
            f"base reward={b_overall['mean_cum_reward']:+.2f} "
            f"(solved {b_overall['resolved_rate']:.0%})  |  "
            f"SFT reward={s_overall['mean_cum_reward']:+.2f} "
            f"(solved {s_overall['resolved_rate']:.0%})  |  "
            f"Δ = [bold]{s_overall['mean_cum_reward'] - b_overall['mean_cum_reward']:+.2f}[/]"
        )


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    base_model = args.base_model or cfg.model.name
    sft_model = args.sft_model or str(
        Path(cfg.run.output_dir) / "final_merged_16bit"
    )

    scenarios: List[str] = args.scenarios or list(cfg.eval.scenarios)
    episodes = args.episodes or int(cfg.eval.episodes_per_scenario)

    tpl_kwargs = {"enable_thinking": bool(cfg.model.thinking_mode)}

    base_results: Optional[Dict[str, Any]] = None

    # ---- base ----
    if not args.skip_base:
        console.rule(f"[bold cyan]Evaluating BASE: {base_model}")
        model, tok = load_model_for_inference(
            base_model,
            max_seq_length=cfg.model.max_seq_length,
            dtype=cfg.model.dtype,
        )
        base_results = evaluate_model(
            model, tok, scenarios, episodes,
            max_new_tokens=cfg.eval.max_new_tokens,
            temperature=cfg.eval.temperature,
            top_p=cfg.eval.top_p,
            top_k=cfg.eval.top_k,
            history_window=cfg.eval.history_window,
            chat_template_kwargs=tpl_kwargs,
            seed=cfg.run.seed,
            verbose=args.verbose,
            label="BASE",
        )
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ---- sft ----
    console.rule(f"[bold cyan]Evaluating SFT: {sft_model}")
    model, tok = load_model_for_inference(
        sft_model if not args.lora_adapter else base_model,
        max_seq_length=cfg.model.max_seq_length,
        dtype=cfg.model.dtype,
        lora_adapter=args.lora_adapter,
    )
    sft_results = evaluate_model(
        model, tok, scenarios, episodes,
        max_new_tokens=cfg.eval.max_new_tokens,
        temperature=cfg.eval.temperature,
        top_p=cfg.eval.top_p,
        top_k=cfg.eval.top_k,
        history_window=cfg.eval.history_window,
        chat_template_kwargs=tpl_kwargs,
        seed=cfg.run.seed,
        verbose=args.verbose,
        label="SFT",
    )
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- print + save ----
    print_comparison(base_results, sft_results, scenarios)

    out = {
        "config": {
            "base_model": base_model,
            "sft_model": sft_model,
            "lora_adapter": args.lora_adapter,
            "scenarios": scenarios,
            "episodes_per_scenario": episodes,
            "temperature": cfg.eval.temperature,
            "top_p": cfg.eval.top_p,
            "top_k": cfg.eval.top_k,
            "max_new_tokens": cfg.eval.max_new_tokens,
            "history_window": cfg.eval.history_window,
        },
        "base": base_results,
        "sft": sft_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    console.print(f"\n[green]saved full results to {output_path}[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
