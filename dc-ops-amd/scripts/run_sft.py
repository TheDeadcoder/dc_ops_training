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
    outputs/sft/                     — TRL intermediate checkpoints
    outputs/dc_ops_sft_lora/         — final LoRA adapter + tokenizer + system_prompt.txt
    logs/sft-<timestamp>.log         — stdout (when launched via ./launch/sft.sh)
    wandb/                           — wandb run artefacts (also mirrored to cloud)

Import order matters:
    1. src.rocm_env          (sets HIP env vars before any GPU touch)
    2. unsloth               (must be FIRST among ML libs — patches TRL/PEFT/Transformers)
    3. torch / transformers / trl / peft / etc.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1) ROCm env vars — MUST be set before any library touches the GPU.
# ---------------------------------------------------------------------------
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src import rocm_env  # noqa: F401  (applies env vars on import)

# ---------------------------------------------------------------------------
# 2) Unsloth — MUST be imported before trl/transformers/peft so its kernel
#    patches land. Importing it here, not inside main(), silences the warning:
#       "Unsloth should be imported before [trl, transformers, peft]"
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel  # noqa: E402

# ---------------------------------------------------------------------------
# 3) Standard library + everything else
# ---------------------------------------------------------------------------
import argparse                            # noqa: E402
import gc                                  # noqa: E402
import json                                # noqa: E402
import os                                  # noqa: E402
import re                                  # noqa: E402
from pprint import pformat                 # noqa: E402

import torch                               # noqa: E402
import yaml                                # noqa: E402


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def verify_sft_format(model, tokenizer, system_prompt: str, vcfg: dict) -> bool:
    """Port of notebook cell 33 — generate one response on a fresh env reset
    and check it has both <reasoning> and <command> tags. Catches the case
    where SFT trained-but-format-collapsed (which means GRPO will never
    learn anything because every group will have identical zero-format reward).

    Returns True if format is intact, False otherwise. Doesn't raise — just
    logs a loud warning, because the model is still saved either way.
    """
    print("[sft-verify] running post-train format check …")
    from dc_ops_env.server.dc_ops_env_environment import DcOpsEnvironment
    from dc_ops_env.models import DcOpsAction  # noqa: F401  (kept for parity with notebook imports)
    from src.prompts import user_content_from_obs

    FastLanguageModel.for_inference(model)

    test_env = DcOpsEnvironment()
    test_obs = test_env.reset(scenario=vcfg["scenario"], seed=vcfg["seed"])
    test_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content_from_obs(test_obs)},
    ]

    prompt = tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=vcfg["max_new_tokens"],
            temperature=vcfg["temperature"],
            top_p=vcfg["top_p"],
            do_sample=True,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )

    has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", response, re.DOTALL))
    has_command   = bool(re.search(r"<command>.*?</command>",     response, re.DOTALL))

    print("=" * 60)
    print("[sft-verify] SFT MODEL FORMAT VERIFICATION")
    print("=" * 60)
    print(f"[sft-verify] Response (first 500 chars):\n{response[:500]}")
    print(f"[sft-verify] Has <reasoning>: {has_reasoning}")
    print(f"[sft-verify] Has <command>:   {has_command}")

    if has_reasoning and has_command:
        print("[sft-verify] OK — format intact, ready for GRPO")
        return True
    print("[sft-verify] !!  WARNING: SFT model does NOT produce correct format")
    print("[sft-verify] !!  GRPO will likely fail. Consider more epochs or check")
    print("[sft-verify] !!  the data pipeline (scripts/eda.py).")
    return False


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    print("[sft] config:\n" + pformat(cfg))

    # -------- wandb -----------------------------------------------------
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
        report_to = "none"

    # -------- GPU sanity -----------------------------------------------
    assert torch.cuda.is_available(), "torch.cuda (ROCm) is not available"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[sft] device: {gpu_name}  |  VRAM: {vram_gb:.1f} GB  |  bf16: {torch.cuda.is_bf16_supported()}")

    # -------- Model load (Unsloth) --------------------------------------
    mcfg = cfg["model"]
    lcfg = cfg["lora"]
    scfg = cfg["sft"]

    print(f"[sft] loading base model: {mcfg['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["name"],
        max_seq_length=mcfg["max_seq_length"],
        load_in_4bit=mcfg["load_in_4bit"],
    )

    # -------- LoRA adapter ----------------------------------------------
    print(f"[sft] attaching LoRA r={lcfg['r']}, alpha={lcfg['alpha']}, "
          f"gradient_checkpointing={lcfg['use_gradient_checkpointing']}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"],
        target_modules=lcfg["target_modules"],
        use_gradient_checkpointing=lcfg["use_gradient_checkpointing"],
        random_state=lcfg["random_state"],
    )
    model.print_trainable_parameters()
    print(f"[sft] VRAM after model+LoRA load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # -------- Data ------------------------------------------------------
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

    # -------- Trainer ----------------------------------------------------
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
    n_steps_per_epoch = max(1, len(data.train) // eff_batch)
    print(f"[sft] approx steps/epoch = {n_steps_per_epoch}  "
          f"(total ≈ {n_steps_per_epoch * scfg['num_train_epochs']} steps)")

    if args.dry_run:
        print("[sft] --dry-run: skipping trainer.train()")
        return

    # -------- Train ------------------------------------------------------
    print("[sft] starting training …")
    stats = trainer.train()
    print(f"[sft] DONE. final training loss = {stats.training_loss:.4f}  "
          f"runtime = {stats.metrics['train_runtime']:.0f} s")

    # -------- Save local LoRA -------------------------------------------
    save_dir = cfg["save"]["local_dir"]
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "system_prompt.txt"), "w") as f:
        f.write(data.system_prompt)
    with open(os.path.join(save_dir, "sft_run.json"), "w") as f:
        json.dump({
            "base_model":      mcfg["name"],
            "lora_rank":       lcfg["r"],
            "max_seq_length":  mcfg["max_seq_length"],
            "scenarios_kept":  data.scenario_counts,
            "num_windows":     data.num_windows,
            "dropped_variants": data.dropped_variants,
            "final_loss":      float(stats.training_loss),
            "train_runtime_s": float(stats.metrics["train_runtime"]),
        }, f, indent=2)
    print(f"[sft] LoRA + tokenizer + system_prompt + run-meta saved → {save_dir}")

    # -------- Post-SFT format verification (notebook cell 33) ----------
    vcfg = cfg.get("verify_after_train", {})
    if vcfg.get("enabled"):
        try:
            verify_sft_format(model, tokenizer, data.system_prompt, vcfg)
        except Exception as e:
            print(f"[sft-verify] verification step failed (non-fatal): {e}")

    # Free up VRAM in case the user runs GRPO from the same shell
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("[sft] next step:  python scripts/push_to_hf.py --repo-id <user>/<repo>")
    print("[sft]      then:  ./launch/grpo.sh")


if __name__ == "__main__":
    main()
