#!/usr/bin/env python
# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
SFT training for DC-Ops on AMD Instinct MI300X (ROCm 7.2).

Usage:
    python scripts/run_sft.py --config configs/sft.yaml
    # or in background:
    ./launch/sft.sh

Outputs:
    outputs/sft/                 — TRL intermediate checkpoints
    outputs/dc_ops_sft_lora/     — final LoRA adapter + tokenizer
    logs/sft.log                 — stdout (when launched via ./launch/sft.sh)
    wandb/                       — wandb run artefacts (also mirrored to cloud)
"""

from __future__ import annotations

# ------------------------------------------------------------------
# !! THIS IMPORT MUST BE FIRST — it sets ROCm/vLLM env vars before
# any heavy library touches the GPU.
# ------------------------------------------------------------------
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src import rocm_env  # noqa: F401  (applies env vars on import)

# ------------------------------------------------------------------
import argparse
import json
import os
from dataclasses import asdict
from pprint import pformat

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DC-Ops SFT trainer (ROCm 7.2 / MI300X)")
    p.add_argument("--config", type=str, default="configs/sft.yaml",
                   help="Path to YAML config.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load everything, print config, but don't call trainer.train().")
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    print("[sft] config:\n" + pformat(cfg))

    # -------- wandb ------------------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled"):
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "wandb.enabled=true but WANDB_API_KEY is not in the environment. "
                "Set it in .env or run: export WANDB_API_KEY=<key>"
            )
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "dc-ops-amd")
        if wandb_cfg.get("run_name"):
            os.environ["WANDB_NAME"] = wandb_cfg["run_name"]
        if wandb_cfg.get("tags"):
            os.environ["WANDB_TAGS"] = ",".join(wandb_cfg["tags"])
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    # -------- GPU check (ROCm) ------------------------------------------
    assert torch.cuda.is_available(), "torch.cuda (ROCm) is not available"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[sft] device: {gpu_name}  |  VRAM: {vram_gb:.1f} GB  |  bf16: {torch.cuda.is_bf16_supported()}")

    # -------- Unsloth model load (must import AFTER rocm_env) ------------
    from unsloth import FastLanguageModel

    mcfg = cfg["model"]
    lcfg = cfg["lora"]
    scfg = cfg["sft"]

    print(f"[sft] loading base model: {mcfg['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["name"],
        max_seq_length=mcfg["max_seq_length"],
        load_in_4bit=mcfg["load_in_4bit"],
        # attn_implementation is set internally by Unsloth; we pass dtype
        # via load_in_4bit=True which forces bf16 compute dtype.
    )

    # -------- LoRA adapter -----------------------------------------------
    gckpt = lcfg["use_gradient_checkpointing"]
    # Unsloth accepts True / False / "unsloth" — we explicitly allow False
    # because the user has 192 GB VRAM and asked us not to use checkpointing.
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"],
        target_modules=lcfg["target_modules"],
        use_gradient_checkpointing=gckpt,  # False on MI300X, per config
        random_state=lcfg["random_state"],
    )
    print(f"[sft] LoRA trainable params:")
    model.print_trainable_parameters()
    print(f"[sft] VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # -------- Data --------------------------------------------------------
    from src.data_utils import load_sft_dataset

    dcfg = cfg["data"]
    data = load_sft_dataset(
        tokenizer,
        source=dcfg["hf_source"],
        local_jsonl=dcfg["local_jsonl"],
        eval_split=dcfg["eval_split"],
        seed=dcfg["seed"],
    )
    print(f"[sft] dataset: train={len(data.train):,}  eval={len(data.eval):,}  "
          f"total windows={data.num_windows:,}  dropped VAR_* rows={data.dropped_variants}")
    for sid in sorted(data.scenario_counts):
        print(f"[sft]   scenario {sid}: {data.scenario_counts[sid]:4d} episodes")

    # Persist the rewritten system prompt so GRPO can pick the same string up.
    os.makedirs(cfg["save"]["local_dir"], exist_ok=True)
    sys_prompt_path = os.path.join(cfg["save"]["local_dir"], "system_prompt.txt")
    with open(sys_prompt_path, "w") as f:
        f.write(data.system_prompt)
    print(f"[sft] wrote system prompt → {sys_prompt_path}")

    # -------- Trainer -----------------------------------------------------
    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=scfg["output_dir"],
        per_device_train_batch_size=scfg["per_device_train_batch_size"],
        gradient_accumulation_steps=scfg["gradient_accumulation_steps"],
        num_train_epochs=scfg["num_train_epochs"],
        learning_rate=scfg["learning_rate"],
        lr_scheduler_type=scfg["lr_scheduler_type"],
        warmup_steps=scfg["warmup_steps"],
        weight_decay=scfg["weight_decay"],
        max_grad_norm=scfg["max_grad_norm"],
        bf16=scfg["bf16"],
        fp16=scfg["fp16"],
        logging_steps=scfg["logging_steps"],
        eval_strategy=scfg["eval_strategy"],
        eval_steps=scfg["eval_steps"],
        save_strategy=scfg["save_strategy"],
        save_steps=scfg["save_steps"],
        save_total_limit=scfg["save_total_limit"],
        max_seq_length=mcfg["max_seq_length"],
        dataset_text_field=scfg["dataset_text_field"],
        packing=scfg["packing"],
        dataloader_num_workers=scfg["dataloader_num_workers"],
        dataloader_pin_memory=scfg["dataloader_pin_memory"],
        report_to=report_to,
        seed=scfg["seed"],
        run_name=wandb_cfg.get("run_name"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data.train,
        eval_dataset=data.eval,
        args=sft_config,
    )

    eff_batch = (
        scfg["per_device_train_batch_size"] * scfg["gradient_accumulation_steps"]
    )
    print(f"[sft] effective batch = {eff_batch}  "
          f"(per_device={scfg['per_device_train_batch_size']}, "
          f"grad_accum={scfg['gradient_accumulation_steps']})")

    if args.dry_run:
        print("[sft] --dry-run: skipping trainer.train()")
        return

    # -------- Train -------------------------------------------------------
    print("[sft] starting training …")
    stats = trainer.train()
    print(f"[sft] DONE. final training loss = {stats.training_loss:.4f}  "
          f"runtime = {stats.metrics['train_runtime']:.0f} s")

    # -------- Save local LoRA --------------------------------------------
    save_dir = cfg["save"]["local_dir"]
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    # Also save the rewritten system prompt (again — survives if the dir was wiped)
    with open(os.path.join(save_dir, "system_prompt.txt"), "w") as f:
        f.write(data.system_prompt)
    # And a small metadata file so downstream scripts can recover config
    with open(os.path.join(save_dir, "sft_run.json"), "w") as f:
        json.dump({
            "base_model": mcfg["name"],
            "lora_rank": lcfg["r"],
            "max_seq_length": mcfg["max_seq_length"],
            "scenarios_kept": data.scenario_counts,
            "num_windows": data.num_windows,
            "dropped_variants": data.dropped_variants,
        }, f, indent=2)

    print(f"[sft] LoRA + tokenizer saved → {save_dir}")
    print(f"[sft] next step:  python scripts/push_to_hf.py --config configs/sft.yaml")


if __name__ == "__main__":
    main()
