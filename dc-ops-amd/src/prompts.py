# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
Prompt utilities shared by SFT (for windowing the teacher's conversations)
and GRPO (for constructing rollout prompts from live env observations).

The teacher model (DeepSeek-R1-Distill-Qwen-32B) emitted a <think>/<reasoning>/
<command> triple. The student (Qwen2.5-7B-Instruct) is a non-reasoning model,
so we strip <think> everywhere — both from the teacher outputs AND from the
instructions in the system prompt — before training.
"""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove the teacher's private <think>…</think> scratchpad."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def rewrite_system_prompt(text: str) -> str:
    """Collapse the 3-block (think/reasoning/command) instruction into a
    2-block (reasoning/command) instruction for the non-reasoning student.

    The rewrite touches four places in the original prompt:
      1. The "Produce three blocks…" header becomes "Produce two blocks…".
      2. The numbered description of `<think>` is deleted.
      3. The subsequent block numbers (2→1, 3→2) are renumbered.
      4. The reasoning-block copy is edited so it no longer references
         <think> at all.

    Any stray <think> / </think> tokens that survive are nuked, and excess
    blank lines are collapsed.
    """
    # 1) Header
    text = text.replace(
        "Produce three blocks in order: <think>, <reasoning>, <command>. "
        "Each block must appear exactly once.",
        "Produce two blocks in order: <reasoning>, <command>. "
        "Each block must appear exactly once.",
    )

    # 2) Delete the "1. <think>...</think>" block (everything up to "2. <reasoning>")
    text = re.sub(
        r"1\. <think>\.\.\.</think>.*?(?=2\. <reasoning>)",
        "",
        text,
        flags=re.DOTALL,
    )

    # 3) Renumber 2→1 and 3→2 (once each — only for the top-level block headers)
    text = text.replace("2. <reasoning>", "1. <reasoning>", 1)
    text = text.replace("3. <command>",   "2. <command>",   1)

    # 4) Remove the reasoning-block instruction that referenced <think>
    text = text.replace(
        "Do not repeat verbatim what you wrote in <think>. "
        "This is the distilled conclusion, not a transcript.",
        "Be concise — this is the distilled committed conclusion.",
    )

    # 5) Nuke any surviving bare <think>/</think> tokens
    text = text.replace("<think>", "").replace("</think>", "")

    # 6) Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def user_content_from_obs(obs) -> str:
    """Format a DcOpsObservation into the exact user-turn string the teacher
    was given. Must stay byte-identical between SFT and GRPO, or the model
    sees a distribution shift at RL time."""
    return (
        f"**Action Result:** {obs.action_result}\n\n"
        f"**Steps Remaining:** {obs.steps_remaining}\n\n"
        f"{obs.dashboard}"
    )


def messages_to_prompt(tokenizer, system_prompt: str, user_content: str) -> str:
    """Render a (system, user) pair through the tokenizer's chat template
    with generation prompt appended."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
