#!/usr/bin/env python
# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
EDA on the DC-Ops SFT dataset. Prints:
  - episode / window counts per scenario (after VAR_* filtering)
  - command frequency across GPT turns
  - length statistics for <think>, <reasoning>, <command>, and full GPT turns
  - rough token counts for windowed examples

This replicates and extends the EDA in the notebook's pre-training cells.
Doesn't require a GPU or any training libraries — just `datasets` + stdlib.

Usage:
    python scripts/eda.py --local data/train.jsonl           # offline
    python scripts/eda.py                                    # pulls from HF
    python scripts/eda.py --out eda.json                     # also writes JSON
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from collections import Counter

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.constants import BASE_ALERT_MARKERS, KNOWN_COMMANDS

_THINK_RE     = re.compile(r"<think>.*?</think>\s*",             re.DOTALL)
_THINK_GET    = re.compile(r"<think>(.*?)</think>",              re.DOTALL)
_REASONING_GET= re.compile(r"<reasoning>(.*?)</reasoning>",      re.DOTALL)
_COMMAND_GET  = re.compile(r"<command>\s*(.+?)\s*</command>",    re.DOTALL)
_ALERT_RE     = re.compile(r"ALERT:\s*([^║]{0,200})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA on DC-Ops SFT dataset")
    p.add_argument("--local", default=None,
                   help="Path to a local train.jsonl. If omitted, pulls from the HF dataset.")
    p.add_argument("--hf-source", default="Melikshah/dc-ops-sft-data",
                   help="HF dataset id (used when --local is not given)")
    p.add_argument("--out", default=None,
                   help="Optional path to write the same stats as JSON.")
    return p.parse_args()


def classify(ex):
    first_human = next(
        (t["value"] for t in ex["conversations"] if t["from"] == "human"),
        "",
    )
    m = _ALERT_RE.search(first_human)
    alert = m.group(1).strip() if m else ""
    for sid, markers in BASE_ALERT_MARKERS:
        if all(m in alert for m in markers):
            return sid
    return None


def _describe(name, x):
    if not x:
        return {"name": name, "count": 0}
    xs = sorted(x)
    return {
        "name":   name,
        "count":  len(x),
        "mean":   round(statistics.mean(x), 1),
        "median": xs[len(xs)//2],
        "p95":    xs[int(0.95*len(xs))],
        "p99":    xs[int(0.99*len(xs))],
        "max":    max(x),
    }


def main() -> None:
    args = parse_args()

    if args.local:
        with open(args.local) as f:
            data = [json.loads(line) for line in f]
    else:
        from datasets import load_dataset
        ds = load_dataset(
            "json",
            data_files={"train": f"hf://datasets/{args.hf_source}/train.jsonl"},
            split="train",
        )
        data = list(ds)

    # -------- episode / scenario counts --------
    kept = Counter()
    dropped = 0
    for ex in data:
        sid = classify(ex)
        if sid is None:
            dropped += 1
        else:
            kept[sid] += 1

    # -------- turn / command stats --------
    total_turns = 0
    cmd_head_counter = Counter()
    full_gpt_lens, think_lens, reasoning_lens, cmd_lens = [], [], [], []
    for ex in data:
        sid = classify(ex)
        if sid is None:
            continue
        total_turns += len(ex["conversations"])
        for turn in ex["conversations"]:
            if turn["from"] != "gpt":
                continue
            text = turn["value"]
            full_gpt_lens.append(len(text))
            if (m := _THINK_GET.search(text)):      think_lens.append(len(m.group(1)))
            if (m := _REASONING_GET.search(text)):  reasoning_lens.append(len(m.group(1)))
            stripped = _THINK_RE.sub("", text)
            if (m := _COMMAND_GET.search(stripped)):
                cmd = m.group(1).strip()
                cmd_lens.append(len(cmd))
                head = cmd.split()[0].lower() if cmd.split() else ""
                cmd_head_counter[head] += 1

    # -------- print --------
    print(f"\n═══════ DC-Ops SFT dataset EDA ═══════")
    print(f"source: {'local ' + args.local if args.local else 'HF ' + args.hf_source}")
    print(f"raw episodes:          {len(data):,}")
    print(f"dropped VAR_* rows:    {dropped:,}  ({dropped/len(data)*100:.1f}%)")
    print(f"kept episodes:         {sum(kept.values()):,}")
    for sid in ["A1", "A2", "A4", "B1", "B3", "B4"]:
        frac = kept[sid] / sum(kept.values()) if kept else 0
        print(f"    {sid}: {kept[sid]:4d}  ({frac*100:.1f}%)")

    print(f"\ncommand frequency (GPT turns, kept episodes):")
    total_cmds = sum(cmd_head_counter.values())
    for cmd, n in cmd_head_counter.most_common():
        flag = "  " if cmd in KNOWN_COMMANDS else " ⚠"
        print(f"   {flag}{n:5d}  {cmd}  ({n/total_cmds*100:.1f}%)")
    unknown = [c for c in cmd_head_counter if c not in KNOWN_COMMANDS]
    if unknown:
        print(f"   ⚠ unknown command verbs (not in env.parser.AVAILABLE_ACTIONS): {unknown}")

    print(f"\nlength stats (chars — rough token ≈ chars/4):")
    for s in (_describe("full GPT turn",     full_gpt_lens),
              _describe("<think> block",     think_lens),
              _describe("<reasoning> block", reasoning_lens),
              _describe("<command> block",   cmd_lens)):
        if s["count"] == 0:
            print(f"   {s['name']:20s} n=0")
        else:
            print(f"   {s['name']:20s} n={s['count']:5d}  "
                  f"mean={s['mean']:>6}  median={s['median']:>5}  "
                  f"p95={s['p95']:>5}  p99={s['p99']:>5}  max={s['max']}")

    # sizing recommendation — account for the fact that <think> is stripped
    # in windowing (the student is non-reasoning). Post-strip assistant turn
    # length ≈ reasoning + command + framing tags.
    #
    # Important: with `packing=true` in configs/sft.yaml, multiple windows
    # get packed into one block of size max_seq_length. INDIVIDUAL windows
    # that exceed max_seq_length get truncated (losing the trailing assistant
    # turn = the supervised target), which is catastrophic for SFT.
    #
    # The estimator below is the per-component p99 sum (intentionally
    # conservative). Real tokenizer-measured p99 on full windows including
    # the system prompt is ~2700 tokens, max ~2900 tokens.
    CHARS_PER_TOKEN = 3.85   # Qwen2.5 BPE on mixed English+structured text
    print(f"\n→ max_seq_length recommendation (windowed prompts):")
    if reasoning_lens and cmd_lens:
        stripped_assist_p99 = sorted(
            [r + c + 60 for r, c in zip(reasoning_lens, cmd_lens)]  # +60 for tag framing
        )[int(0.99 * len(reasoning_lens))]
        rough_window_p99_chars = (
            3500                          # rewritten system prompt (each window has it)
            + 2500 + stripped_assist_p99  # prior user + prior stripped assistant
            + 2500                        # current user dashboard
            + stripped_assist_p99         # current stripped assistant
            + 200                         # chat-template framing
        )
        rough_tokens = rough_window_p99_chars / CHARS_PER_TOKEN
        print(f"   window p99 (upper-bound estimate) ≈ {rough_tokens:.0f} tokens")
        print(f"   empirical p99 (tokenizer-measured): ~2700 tokens")
        print(f"   empirical max (tokenizer-measured): ~2900 tokens")
        print(f"   configs/sft.yaml sets max_seq_length=4096 → safe (covers max + packing headroom)")
        print(f"   ⚠  NEVER drop max_seq_length below 3072 — would truncate >0% of windows")
        if rough_tokens > 3500:
            print(f"   note: per-component estimate is high but tokenizer-measured is lower")

    # -------- optional JSON dump --------
    if args.out:
        out = {
            "raw_episodes": len(data),
            "dropped_variants": dropped,
            "kept_episodes": sum(kept.values()),
            "per_scenario": dict(kept),
            "command_frequency": dict(cmd_head_counter),
            "length_stats": {
                "gpt_turn":   _describe("gpt",       full_gpt_lens),
                "think":      _describe("think",     think_lens),
                "reasoning":  _describe("reasoning", reasoning_lens),
                "command":    _describe("command",   cmd_lens),
            },
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote JSON summary → {args.out}")


if __name__ == "__main__":
    main()
