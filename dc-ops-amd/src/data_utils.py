# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
SFT data pipeline: load `Melikshah/dc-ops-sft-data` (or a local jsonl),
filter to registered scenarios, strip <think>, and emit windowed
(system → user → assistant → …) conversations ready for chat-template
encoding.

Replicates the logic of notebook Cell 16.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from datasets import Dataset, load_dataset

from .constants import BASE_ALERT_MARKERS
from .prompts import rewrite_system_prompt, strip_think


MAX_PRIOR_TURNS = 1  # number of prior (human, gpt) pairs to include per window


# ---------------------------------------------------------------------------
# Episode → scenario classification
# ---------------------------------------------------------------------------
_ALERT_RE = re.compile(r"ALERT:\s*([^║]{0,200})")


def _alert_line(human_text: str) -> str:
    m = _ALERT_RE.search(human_text)
    return m.group(1).strip() if m else ""


def episode_scenario(example: dict) -> str | None:
    """Return 'A1'..'B4' if the episode is a registered scenario, else None.

    The registered scenarios are identified by substrings in the *first*
    human turn's ALERT line. Anything else (VAR_* variants) returns None
    and the caller is expected to drop it.
    """
    first_human = next(
        (t["value"] for t in example["conversations"] if t["from"] == "human"),
        "",
    )
    alert = _alert_line(first_human)
    for sid, markers in BASE_ALERT_MARKERS:
        if all(m in alert for m in markers):
            return sid
    return None


# ---------------------------------------------------------------------------
# Episode → list of (messages) windows
# ---------------------------------------------------------------------------
def episode_to_windowed_turns(example: dict, default_system: str) -> list[dict]:
    """Convert a full ShareGPT-style episode into a list of SFT windows.

    Each window is a (system, [optional prior user/assistant…], user, assistant)
    message list. <think> blocks are stripped from all assistant turns.
    """
    convs = example["conversations"]
    system_raw = next(
        (t["value"] for t in convs if t["from"] == "system"),
        default_system,
    )
    system_msg = rewrite_system_prompt(system_raw)
    human_turns = [t["value"] for t in convs if t["from"] == "human"]
    gpt_turns   = [t["value"] for t in convs if t["from"] == "gpt"]

    windows: list[dict] = []
    for t, (human_t, gpt_t) in enumerate(zip(human_turns, gpt_turns)):
        gpt_content = strip_think(gpt_t)
        # Drop turns where the teacher produced no committed command
        if not gpt_content or "<command>" not in gpt_content:
            continue

        messages: list[dict] = [{"role": "system", "content": system_msg}]

        # Include up to MAX_PRIOR_TURNS prior (user, assistant) pairs so the
        # model sees short-horizon context. 87% of windows end up with prior
        # context per our EDA.
        prior_start = max(0, t - MAX_PRIOR_TURNS)
        for prior_t in range(prior_start, t):
            prior_gpt = strip_think(gpt_turns[prior_t])
            if not prior_gpt:
                continue
            messages.append({"role": "user",      "content": human_turns[prior_t]})
            messages.append({"role": "assistant", "content": prior_gpt})

        messages.append({"role": "user",      "content": human_t})
        messages.append({"role": "assistant", "content": gpt_content})
        windows.append({"messages": messages})

    return windows


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------
@dataclass
class SftLoadResult:
    train: Dataset
    eval: Dataset
    system_prompt: str               # final rewritten system prompt (for use in GRPO)
    scenario_counts: dict[str, int]  # kept-episode counts per scenario
    dropped_variants: int            # VAR_* rows that were filtered out
    num_windows: int


def load_sft_dataset(
    tokenizer,
    *,
    source: str = "Melikshah/dc-ops-sft-data",
    local_jsonl: str | None = None,
    eval_split: float = 0.05,
    seed: int = 42,
) -> SftLoadResult:
    """Load + filter + window + split the SFT dataset.

    Args:
        tokenizer: A HuggingFace tokenizer (used for chat-template formatting).
        source: HF dataset id. Used if `local_jsonl` is None.
        local_jsonl: Optional local path to a train.jsonl with the same schema
                     as the HF dataset. Useful for reproducible offline runs.
        eval_split: Fraction of windows to reserve for evaluation.
        seed: RNG seed for the train/eval split.
    """
    # 1. Load raw
    if local_jsonl:
        hf_dataset = load_dataset("json", data_files={"train": local_jsonl}, split="train")
    else:
        hf_dataset = load_dataset(
            "json",
            data_files={"train": f"hf://datasets/{source}/train.jsonl"},
            split="train",
        )

    # 2. Lift + rewrite system prompt from the first row (all rows share it)
    raw_system = hf_dataset[0]["conversations"][0]["value"]
    system_prompt = rewrite_system_prompt(raw_system)
    assert "<think>" not in system_prompt, "system prompt rewrite failed to strip <think>"

    # 3. Filter + window
    kept: Counter = Counter()
    dropped = 0
    all_windows: list[dict] = []
    for ex in hf_dataset:
        sid = episode_scenario(ex)
        if sid is None:
            dropped += 1
            continue
        kept[sid] += 1
        all_windows.extend(episode_to_windowed_turns(ex, default_system=system_prompt))

    # 4. Chat-template encode
    ds = Dataset.from_list(all_windows)

    def apply_template(examples):
        return {
            "text": [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                for msgs in examples["messages"]
            ]
        }

    ds = ds.map(apply_template, batched=True, remove_columns=["messages"])

    # 5. Split
    split = ds.train_test_split(test_size=eval_split, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    # 6. Sanity: no <think> leaked
    sample = train_ds[0]["text"]
    assert "<think>" not in sample,      "<think> leaked into a training sample"
    assert "<reasoning>" in sample,      "training sample missing <reasoning> tag"
    assert "<command>" in sample,        "training sample missing <command> tag"

    return SftLoadResult(
        train=train_ds,
        eval=eval_ds,
        system_prompt=system_prompt,
        scenario_counts=dict(kept),
        dropped_variants=dropped,
        num_windows=len(all_windows),
    )
