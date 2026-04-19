# Copyright (c) 2026. Licensed under BSD-3-Clause.
"""
Dataset preparation for DC-Ops SFT.

The HF dataset `Melikshah/dc-ops-sft-data` has one row per episode with a
`conversations` list in ShareGPT format:

    [
        {"from": "system", "value": "You are DC-Ops Agent..."},
        {"from": "human",  "value": "<dashboard>"},
        {"from": "gpt",    "value": "<think>...</think>\n<reasoning>...</reasoning>\n<command>...</command>"},
        {"from": "human",  "value": "<next dashboard>"},
        {"from": "gpt",    "value": "..."},
        ...
    ]

This module:

  1. Converts ShareGPT keys -> OpenAI keys (role/content)
  2. Fans out each episode into N prompt-completion pairs, one per agent turn.
     For turn t, the prompt is all messages before turn t; the completion is
     the gpt message at turn t. Qwen3's chat template strips `<think>` from
     any *non-final* assistant message automatically, so the prompt's history
     has no think blocks (matches inference distribution), while the
     completion keeps its think block (the training target).
  3. Filters out examples longer than max_seq_length after tokenization, and
     examples with suspiciously short completions.

This format is directly consumable by TRL's SFTTrainer with
completion_only_loss=True and packing=True.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

# ShareGPT -> OpenAI role mapping
ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def _conversations_to_messages(
    conversations: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Convert ShareGPT [{from, value}] to OpenAI [{role, content}]."""
    messages: List[Dict[str, str]] = []
    for c in conversations:
        role = ROLE_MAP.get(c.get("from", ""))
        if role is None:
            continue
        messages.append({"role": role, "content": c.get("value", "")})
    return messages


def _fan_out_messages(
    messages: List[Dict[str, str]],
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    """Return one (prompt, completion) pair per assistant turn.

    For assistant turn i:
      prompt = messages[:i]     (ends in a user turn)
      completion = [messages[i]]
    """
    pairs: List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        if i == 0:
            # Degenerate: assistant as first message -> no context to learn from.
            continue
        pairs.append((messages[:i], [msg]))
    return pairs


def fan_out_dataset(ds: Dataset, num_proc: int = 8) -> Dataset:
    """Expand each episode row into N prompt-completion rows (one per agent turn)."""

    def _batched(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        out_prompts: List[List[Dict[str, str]]] = []
        out_completions: List[List[Dict[str, str]]] = []
        for conv in batch["conversations"]:
            messages = _conversations_to_messages(conv)
            for prompt, completion in _fan_out_messages(messages):
                out_prompts.append(prompt)
                out_completions.append(completion)
        return {"prompt": out_prompts, "completion": out_completions}

    return ds.map(
        _batched,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Fanning out episodes",
    )


def _to_messages_format(ds: Dataset, num_proc: int = 8) -> Dataset:
    """Non-fan-out path: one row per episode in TRL's conversational 'messages' format."""

    def _batched(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        out: List[List[Dict[str, str]]] = []
        for conv in batch["conversations"]:
            out.append(_conversations_to_messages(conv))
        return {"messages": out}

    return ds.map(
        _batched,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="Converting episodes to messages format",
    )


def _render_length(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: Dict[str, Any] | None,
) -> int:
    """Apply chat template and return token length. Returns -1 on failure."""
    tpl_kwargs = chat_template_kwargs or {}
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            **tpl_kwargs,
        )
    except Exception:
        return -1
    return len(ids)


def filter_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    min_completion_chars: int = 50,
    num_proc: int = 8,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> Dataset:
    """Drop examples that won't fit or look corrupt."""

    is_prompt_completion = "prompt" in ds.column_names and "completion" in ds.column_names

    def _keep(example: Dict[str, Any]) -> bool:
        if is_prompt_completion:
            completion = example["completion"]
            if not completion:
                return False
            content = completion[0].get("content", "") if isinstance(completion, list) else ""
            if len(content) < min_completion_chars:
                return False
            full_messages = example["prompt"] + completion
        else:
            full_messages = example["messages"]
            # at least one assistant with non-trivial content
            has_good_assistant = any(
                m.get("role") == "assistant"
                and len(m.get("content", "")) >= min_completion_chars
                for m in full_messages
            )
            if not has_good_assistant:
                return False

        length = _render_length(full_messages, tokenizer, chat_template_kwargs)
        return 0 < length <= max_seq_length

    return ds.filter(_keep, num_proc=num_proc, desc="Filtering by length")


def prepare_dataset(
    hf_dataset: str | None,
    local_jsonl: str | None,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    *,
    fan_out: bool = True,
    eval_size: int = 100,
    shuffle_seed: int = 3407,
    num_proc: int = 8,
    min_completion_chars: int = 50,
    chat_template_kwargs: Dict[str, Any] | None = None,
    hf_dataset_split: str = "train",
) -> Tuple[Dataset, Dataset | None]:
    """Top-level dataset preparation.

    Returns (train_ds, eval_ds) where eval_ds may be None if eval_size<=0.
    """
    # 1. Load raw
    if local_jsonl:
        raw = load_dataset("json", data_files=local_jsonl, split="train")
    else:
        assert hf_dataset is not None, "Must provide hf_dataset or local_jsonl"
        raw = load_dataset(hf_dataset, split=hf_dataset_split)

    print(f"[data] loaded {len(raw):,} raw episodes from "
          f"{hf_dataset or local_jsonl}")

    # 2. Reshape
    if fan_out:
        ds = fan_out_dataset(raw, num_proc=num_proc)
        print(f"[data] after fan-out: {len(ds):,} examples")
    else:
        ds = _to_messages_format(raw, num_proc=num_proc)
        print(f"[data] non-fan-out: {len(ds):,} multi-turn episodes")

    # 3. Filter
    before = len(ds)
    ds = filter_dataset(
        ds,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        min_completion_chars=min_completion_chars,
        num_proc=num_proc,
        chat_template_kwargs=chat_template_kwargs,
    )
    dropped = before - len(ds)
    pct = 100.0 * dropped / max(before, 1)
    print(f"[data] after filter (<= {max_seq_length:,} tok): "
          f"{len(ds):,} / {before:,} (dropped {dropped:,}, {pct:.1f}%)")

    # 4. Shuffle + split
    ds = ds.shuffle(seed=shuffle_seed)
    if eval_size > 0 and len(ds) > eval_size * 4:
        split = ds.train_test_split(test_size=eval_size, seed=shuffle_seed)
        print(f"[data] train={len(split['train']):,}, eval={len(split['test']):,}")
        return split["train"], split["test"]
    return ds, None


# -----------------------------------------------------------------------------
# Quick sanity checks (run: python -m dc_ops_sft.data)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    model_name = sys.argv[1] if len(sys.argv) > 1 else "unsloth/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(model_name)
    train_ds, eval_ds = prepare_dataset(
        hf_dataset="Melikshah/dc-ops-sft-data",
        local_jsonl=None,
        tokenizer=tok,
        max_seq_length=8192,
        fan_out=True,
        eval_size=50,
        chat_template_kwargs={"enable_thinking": True},
    )
    print("\nExample 0 prompt (last msg):")
    print(train_ds[0]["prompt"][-1]["content"][:500], "...")
    print("\nExample 0 completion:")
    print(train_ds[0]["completion"][0]["content"][:500], "...")