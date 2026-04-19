#!/usr/bin/env python3
# Copyright (c) 2026. Licensed under BSD-3-Clause.
"""
Push the SFT-trained model to the HuggingFace Hub.

Pushes two repos by default (both controllable via config / flags):
  • {cfg.hub.repo_id}        — the merged 16-bit safetensors + tokenizer
  • {cfg.hub.repo_id}-lora   — the LoRA adapter only (small, fast download)

Usage:
  # Make sure you're logged in:  huggingface-cli login
  python scripts/push_to_hub.py --config configs/sft.yaml
  python scripts/push_to_hub.py --config configs/sft.yaml \\
      --merged-dir ./outputs/.../final_merged_16bit --skip-lora
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, login
from omegaconf import OmegaConf
from rich.console import Console

console = Console()


def _write_model_card(
    target_dir: Path,
    cfg,
    kind: str,
    base_model: str,
) -> None:
    """Write a minimal README.md describing the checkpoint."""
    lora_or_merged = "LoRA adapter" if kind == "lora" else "merged 16-bit"
    head = f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - unsloth
  - trl
  - sft
  - lora
  - dc-ops
  - qwen3
  - reasoning
datasets:
  - Melikshah/dc-ops-sft-data
language:
  - en
pipeline_tag: text-generation
---

# {cfg.run.name} — DC-Ops agent ({lora_or_merged})

SFT checkpoint for the [DC-Ops OpenEnv environment](https://huggingface.co/spaces/Melikshah/dc_ops_env)
(a physics-based datacenter-operations RL environment built on Meta's
[OpenEnv](https://github.com/meta-pytorch/OpenEnv)).

## Pipeline

```
Qwen3-8B (base)
   │  SFT on Melikshah/dc-ops-sft-data  (this checkpoint)
   ▼
dc-ops-sft model  ─────►  GRPO on live DC-Ops environment
```

The model observes a text-based monitoring dashboard and emits three
blocks per turn:

```xml
<think> ... freeform reasoning ... </think>
<reasoning>
1. Situation: ...
2. Constraint: ...
3. Step: ...
4. Action: ...
</reasoning>
<command>diagnose CRAC-3</command>
```

## Training setup

| Setting | Value |
|---|---|
| Base model | `{base_model}` |
| Adapter | bf16 LoRA (r={cfg.lora.r}, α={cfg.lora.alpha}) |
| Target modules | {', '.join(list(cfg.lora.target_modules))} |
| Max seq len | {cfg.model.max_seq_length} |
| Effective batch | {cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps} |
| Optimizer | {cfg.training.optim} |
| LR / schedule | {cfg.training.learning_rate} / {cfg.training.lr_scheduler_type} |
| Epochs | {cfg.training.num_train_epochs} |
| Packing | {cfg.training.packing} |
| Loss | completion-only (fanned out per-turn) |
| Hardware | 1× AMD MI300X (192 GB HBM3, ROCm 7.2) |

## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("{cfg.hub.repo_id}")
model = AutoModelForCausalLM.from_pretrained("{cfg.hub.repo_id}", torch_dtype="bfloat16", device_map="auto")

messages = [
    {{"role": "system", "content": "You are DC-Ops Agent..."}},
    {{"role": "user",   "content": "**Action Result:** ...\\n\\n**Steps Remaining:** 15\\n\\n<dashboard>"}},
]
text = tok.apply_chat_template(messages, tokenize=False,
                               add_generation_prompt=True,
                               enable_thinking=True)
out = model.generate(**tok(text, return_tensors="pt").to(model.device),
                     max_new_tokens=1024, temperature=0.6, top_p=0.95)
print(tok.decode(out[0]))
```

## License

Apache 2.0 (inherits from base Qwen3-8B license).
"""
    target_dir.mkdir(parents=True, exist_ok=True)
    readme = target_dir / "README.md"
    # Only overwrite if missing or if it's a placeholder
    if not readme.exists() or readme.stat().st_size < 500:
        readme.write_text(head)
        console.print(f"[dim]wrote model card -> {readme}[/]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--token", type=str, default=None,
                   help="HF token (or use HF_TOKEN env var / `huggingface-cli login`)")
    p.add_argument("--merged-dir", type=str, default=None)
    p.add_argument("--lora-dir", type=str, default=None)
    p.add_argument("--repo-id", type=str, default=None,
                   help="Override cfg.hub.repo_id")
    p.add_argument("--private", action="store_true", default=None)
    p.add_argument("--skip-merged", action="store_true")
    p.add_argument("--skip-lora", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    if args.token:
        login(args.token)
    elif os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])

    output_dir = Path(cfg.run.output_dir)
    merged_dir = Path(args.merged_dir) if args.merged_dir else output_dir / "final_merged_16bit"
    lora_dir = Path(args.lora_dir) if args.lora_dir else output_dir / "final_lora"

    repo_id = args.repo_id or cfg.hub.repo_id
    private = args.private if args.private is not None else bool(cfg.hub.private)

    api = HfApi()

    # ---- merged 16-bit ----
    if not args.skip_merged and bool(cfg.hub.push_merged):
        if not merged_dir.exists():
            console.print(f"[yellow]merged dir missing: {merged_dir} — skipping[/]")
        else:
            _write_model_card(merged_dir, cfg, "merged", cfg.model.name)
            console.print(f"[bold cyan]pushing merged to {repo_id}[/]")
            api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=str(merged_dir),
                repo_id=repo_id,
                commit_message=cfg.hub.commit_message,
                repo_type="model",
            )
            console.print(f"[green]done → https://huggingface.co/{repo_id}[/]")

    # ---- LoRA adapter ----
    if not args.skip_lora and bool(cfg.hub.push_lora):
        if not lora_dir.exists():
            console.print(f"[yellow]lora dir missing: {lora_dir} — skipping[/]")
        else:
            lora_repo = f"{repo_id}-lora"
            _write_model_card(lora_dir, cfg, "lora", cfg.model.name)
            console.print(f"[bold cyan]pushing LoRA to {lora_repo}[/]")
            api.create_repo(lora_repo, private=private, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=str(lora_dir),
                repo_id=lora_repo,
                commit_message=cfg.hub.commit_message + " (LoRA adapter)",
                repo_type="model",
            )
            console.print(f"[green]done → https://huggingface.co/{lora_repo}[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
