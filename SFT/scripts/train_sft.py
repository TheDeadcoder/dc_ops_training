#!/usr/bin/env python3
# Copyright (c) 2026. Licensed under BSD-3-Clause.
"""
SFT training for DC-Ops Qwen3-8B agent.

Usage:
  python scripts/train_sft.py --config configs/sft.yaml
  # With CLI overrides:
  python scripts/train_sft.py --config configs/sft.yaml \\
      --set training.num_train_epochs=3 run.name=qwen3-8b-v2
"""

# NOTE: `unsloth` MUST be imported before `transformers` / `torch` — it
# monkey-patches hooks for flash attention, rope, etc.
from unsloth import FastLanguageModel  # noqa: I001 — import order is intentional
import torch  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Optional  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402
from transformers import set_seed  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from dc_ops_sft.data import prepare_dataset  # noqa: E402
from dc_ops_sft.logging_utils import (  # noqa: E402
    GpuMemoryCallback,
    JsonlLogger,
    TokensPerSecCallback,
    configure_wandb_env,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DC-Ops SFT trainer")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument(
        "--set",
        dest="overrides",
        type=str,
        nargs="*",
        default=[],
        help="CLI overrides: key=value (OmegaConf dotlist)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint dir (or 'latest')",
    )
    return p.parse_args()


def load_config(path: str, overrides: List[str]):
    base = OmegaConf.load(path)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, cli)
    return base


def _dtype_from_str(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }[name.lower()]


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    print(f"[cfg] resolved config saved to {output_dir / 'resolved_config.yaml'}")

    # ---- seed ----
    set_seed(cfg.run.seed)

    # ---- wandb env (picked up by HF Trainer's built-in integration) ----
    if cfg.run.report_to == "wandb":
        configure_wandb_env(
            run_name=cfg.run.name,
            project=cfg.run.wandb_project,
            entity=cfg.run.wandb_entity,
        )

    # ---- model + tokenizer (Unsloth) ----
    dtype = _dtype_from_str(cfg.model.dtype)
    print(f"[model] loading {cfg.model.name} in {cfg.model.dtype} "
          f"(load_in_16bit={cfg.model.load_in_16bit}, "
          f"load_in_4bit={cfg.model.load_in_4bit})")
    t_load = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        dtype=dtype,
        load_in_4bit=cfg.model.load_in_4bit,
        load_in_16bit=cfg.model.load_in_16bit,
        full_finetuning=cfg.model.full_finetuning,
    )
    print(f"[model] loaded in {time.time() - t_load:.1f}s")

    # ---- LoRA ----
    print(f"[lora] r={cfg.lora.r}, alpha={cfg.lora.alpha}, "
          f"target_modules={list(cfg.lora.target_modules)}")
    use_gc = cfg.lora.gradient_checkpointing
    if isinstance(use_gc, str) and use_gc.lower() == "unsloth":
        use_gc = "unsloth"
    else:
        use_gc = bool(use_gc) 
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=list(cfg.lora.target_modules),
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing=use_gc,
        use_rslora=cfg.lora.use_rslora,
        random_state=cfg.run.seed,
    )

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable: {trainable/1e6:.2f}M / {total/1e9:.2f}B "
          f"({100*trainable/total:.2f}%)")

    # ---- data ----
    tpl_kwargs = {"enable_thinking": bool(cfg.model.thinking_mode)}
    train_ds, eval_ds = prepare_dataset(
        hf_dataset=cfg.data.hf_dataset,
        local_jsonl=cfg.data.local_jsonl,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        fan_out=cfg.data.fan_out,
        eval_size=cfg.data.eval_size,
        shuffle_seed=cfg.data.shuffle_seed,
        num_proc=cfg.data.num_proc,
        min_completion_chars=cfg.data.min_completion_chars,
        chat_template_kwargs=tpl_kwargs,
        hf_dataset_split=cfg.data.hf_dataset_split,
    )

    # ---- trainer ----
    report_to = cfg.run.report_to if cfg.run.report_to != "none" else "none"

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=cfg.run.name,
        seed=cfg.run.seed,
        data_seed=cfg.run.seed,

        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,

        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        optim=cfg.training.optim,

        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,

        # ---- TRL-specific ----
        max_length=cfg.model.max_seq_length,
        packing=cfg.training.packing,
        completion_only_loss=cfg.training.completion_only_loss,
        chat_template_kwargs=tpl_kwargs,
        dataset_num_proc=cfg.data.num_proc,

        # ---- logging / saving ----
        logging_steps=cfg.training.logging_steps,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        eval_strategy=(
            cfg.training.eval_strategy if eval_ds is not None else "no"
        ),
        eval_steps=cfg.training.eval_steps if eval_ds is not None else None,
        report_to=report_to,

        # ---- dataloader ----
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,

        # ---- misc ----
        group_by_length=cfg.training.group_by_length,
        remove_unused_columns=cfg.training.remove_unused_columns,
        gradient_checkpointing=False,  # Unsloth already handles this via get_peft_model
        label_names=["labels"],
    )

    callbacks = [
        JsonlLogger(str(output_dir)),
        GpuMemoryCallback(log_every=1),
        TokensPerSecCallback(),
    ]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
    )

    # ---- go ----
    print(f"[train] starting. total_examples={len(train_ds):,}, "
          f"eff_batch={cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps}")
    t_train = time.time()
    train_result = trainer.train(resume_from_checkpoint=args.resume)
    train_wall = time.time() - t_train
    print(f"[train] done in {train_wall/60:.1f} min")

    # ---- metrics ----
    metrics = train_result.metrics
    metrics["train_wall_min"] = round(train_wall / 60, 2)
    trainer.save_metrics("train", metrics)
    print(f"[train] metrics: {json.dumps(metrics, default=str, indent=2)}")

    # ---- save ----
    if cfg.training.save_lora_only:
        lora_dir = output_dir / "final_lora"
        print(f"[save] LoRA adapter -> {lora_dir}")
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))

    if cfg.training.save_merged_16bit:
        merged_dir = output_dir / "final_merged_16bit"
        print(f"[save] merged 16-bit -> {merged_dir}")
        try:
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
        except Exception as e:
            print(f"[save] merged save failed ({e}); "
                  f"falling back to save_pretrained on adapter + tokenizer")
            model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))

    # ---- final JSONL record ----
    try:
        with open(output_dir / "final_summary.json", "w") as f:
            json.dump(
                {
                    "name": cfg.run.name,
                    "model": cfg.model.name,
                    "train_wall_min": round(train_wall / 60, 2),
                    "train_examples": len(train_ds),
                    "eval_examples": len(eval_ds) if eval_ds else 0,
                    "effective_batch": (
                        cfg.training.per_device_train_batch_size
                        * cfg.training.gradient_accumulation_steps
                    ),
                    "lora_r": cfg.lora.r,
                    "lora_alpha": cfg.lora.alpha,
                    "epochs": cfg.training.num_train_epochs,
                    "lr": cfg.training.learning_rate,
                    "max_seq_length": cfg.model.max_seq_length,
                    "final_metrics": {k: v for k, v in metrics.items()},
                },
                f,
                indent=2,
                default=str,
            )
    except Exception as e:
        print(f"[save] final summary failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
