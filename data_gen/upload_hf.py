"""
Upload the final dataset to HuggingFace Hub.

Usage:
  pip install huggingface_hub
  huggingface-cli login   (or set HF_TOKEN env var)
  python upload_hf.py YourUserName/dc-ops-sft
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


OUTPUT_DIR = Path("data_out")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
TRAIN_META_PATH = OUTPUT_DIR / "train_with_meta.jsonl"
STATS_JSON = OUTPUT_DIR / "stats.json"
STATS_MD = OUTPUT_DIR / "stats.md"
README_PATH = OUTPUT_DIR / "README.md"


CARD_TEMPLATE = """---
license: bsd
language:
- en
tags:
- conversational
- reinforcement-learning
- datacenter
- llm-agent
- openenv
task_categories:
- conversational
size_categories:
- 1K<n<10K
---

# DC-Ops SFT Dataset

Supervised fine-tuning conversations for the **DC-Ops** OpenEnv environment —
a physics-based datacenter operations RL environment built on Meta's
[OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

Generated using **DeepSeek-R1-Distill-Qwen-32B** as a teacher model, rolled
out against the live `DcOpsEnvironment` so every dashboard the student sees
is produced by the actual thermal+power simulation.

## Format

Each agent turn contains **three blocks**:

```
<think>
[R1's natural messy chain of thought — exploration, self-correction, all of it]
</think>
<reasoning>
1. Situation: [what the dashboard shows that matters].
2. Constraint: [the relevant ASHRAE limit, procedure rule, or system state].
3. Step: [which phase of assess→diagnose→compensate→verify→resolve].
4. Action: [the chosen command and why].
</reasoning>
<command>
diagnose CRAC-3
</command>
```

Why three blocks:
- **`<think>`** lets the model think freely in its native format. No length cap.
- **`<reasoning>`** is the canonical training signal — concise, structured, ≤200 words, no self-correction. This is what shows up in the operations log.
- **`<command>`** is the single action sent to the env.

The student model (Qwen3-8B with `enable_thinking=True`) emits `<think>`
natively. SFT teaches it to also produce the structured `<reasoning>` block
followed by `<command>`. At GRPO time you can either keep `<think>` enabled
or disable it (`enable_thinking=False`) for ~5x faster rollouts — the model
will still produce `<reasoning>+<command>` from SFT memory.

JSONL schema: each line is `{{"conversations": [...]}}` with `{from, value}` turns:
- `from: "system"` — the agent system prompt
- `from: "human"`  — environment observation: `**Action Result:** ... **Steps Remaining:** N <dashboard>`
- `from: "gpt"`    — agent reply (three blocks above)

## Headline numbers

{HEADLINE}

## Scenario coverage

{SCENARIO_TABLE}

## Command coverage

{COMMAND_TABLE}

## Generation pipeline

1. **Environment**: in-process `DcOpsEnvironment` instances (one per worker)
2. **Teacher**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` via vLLM, called
   with the official `openai` async SDK pointed at vLLM's OpenAI-compatible
   endpoint
3. **Concurrent rollout**: asyncio worker pool, semaphore-throttled to
   vLLM's batch capacity (default 24 in-flight)
4. **Filtering**: drop episodes with parse failures, invalid commands,
   escalations, missing `<reasoning>` blocks, reasoning <30 chars, or
   avg reward < −0.20
5. **Balanced cap**: round-robin across scenario keys to preserve
   rare-command coverage

## Citation

- DC-Ops environment: <link to your repo>
- OpenEnv framework: https://github.com/meta-pytorch/OpenEnv
- Teacher model: DeepSeek-R1-Distill-Qwen-32B (DeepSeek)
"""


def render_card() -> str:
    stats = json.loads(STATS_JSON.read_text())
    headline = (
        f"- Episodes: **{stats['num_episodes']}**\n"
        f"- Agent turns (SFT targets): **{stats['num_agent_turns_training_targets']}**\n"
        f"- Median `<reasoning>` length: **{stats['reasoning_words_per_turn']['median']} words "
        f"({stats['reasoning_chars_per_turn']['median']} chars)**\n"
        f"- Median `<think>` length: **{stats['think_chars_per_turn'].get('median', 'N/A')} chars** "
        f"(present in {stats['think_present_pct']}% of turns)\n"
        f"- Median agent turns/episode: **{stats['agent_turns_per_episode']['median']}**\n"
        f"- Resolved episodes: **{stats['resolved_episodes']} ({stats['resolved_pct']}%)**"
    )
    sc_lines = ["| Scenario | Count | % |", "|---|---|---|"]
    for k, (c, p) in sorted(stats["scenario_key_distribution"].items()):
        sc_lines.append(f"| {k} | {c} | {p}% |")
    cmd_lines = ["| Command | Count | % of agent turns |", "|---|---|---|"]
    items = sorted(stats["command_distribution"].items(), key=lambda kv: kv[1][0], reverse=True)
    for k, (c, p) in items:
        cmd_lines.append(f"| `{k}` | {c} | {p}% |")
    return CARD_TEMPLATE.format(
        HEADLINE=headline,
        SCENARIO_TABLE="\n".join(sc_lines),
        COMMAND_TABLE="\n".join(cmd_lines),
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_hf.py <hf_username>/<dataset_name>")
        sys.exit(1)
    repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN")

    if not all(p.exists() for p in [TRAIN_PATH, TRAIN_META_PATH, STATS_JSON, STATS_MD]):
        print(f"Missing one of the expected files in {OUTPUT_DIR}/. "
              "Run validate_and_stats.py first.")
        sys.exit(1)

    README_PATH.write_text(render_card())
    print(f"Wrote dataset card -> {README_PATH}")

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    print(f"Created/verified repo: {repo_id}")

    for fp in [TRAIN_PATH, TRAIN_META_PATH, STATS_JSON, STATS_MD, README_PATH]:
        print(f"Uploading {fp} ...")
        api.upload_file(
            path_or_fileobj=str(fp),
            path_in_repo=fp.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {fp.name}",
        )
    print(f"\nDone. Dataset live at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
