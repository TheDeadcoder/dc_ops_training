# Copyright (c) 2026. Licensed under BSD-3-Clause.
"""
Logging helpers for DC-Ops SFT training.

Layered so that wandb outages never kill the run:
  • JsonlLogger     — dumps every Trainer log dict to metrics.jsonl
  • GpuMemoryCallback — logs VRAM use + throughput to the Trainer's log stream
  • TokensPerSecCallback — measures effective tokens/sec

The Trainer's built-in wandb integration (via report_to="wandb") handles
remote logging. These callbacks add robustness and extra signal.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback


class JsonlLogger(TrainerCallback):
    """Append every Trainer log dict to metrics.jsonl (line-buffered)."""

    def __init__(self, output_dir: str, filename: str = "metrics.jsonl") -> None:
        self.path = Path(output_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = None
        self._start_time: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.fh = open(self.path, "a", buffering=1)
        self._start_time = time.time()
        self._write({
            "event": "train_begin",
            "timestamp": self._start_time,
            "output_dir": str(self.path.parent),
        })

    def on_log(self, args, state, control, logs: Dict[str, Any] | None = None, **kwargs):
        if logs is None or self.fh is None:
            return
        record = dict(logs)
        record["step"] = state.global_step
        record["epoch"] = state.epoch
        record["timestamp"] = time.time()
        if self._start_time:
            record["wall_s"] = time.time() - self._start_time
        self._write(record)

    def on_save(self, args, state, control, **kwargs):
        self._write({
            "event": "save",
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": time.time(),
        })

    def on_train_end(self, args, state, control, **kwargs):
        self._write({"event": "train_end", "timestamp": time.time()})
        if self.fh:
            self.fh.close()
            self.fh = None

    def _write(self, record: Dict[str, Any]) -> None:
        try:
            self.fh.write(json.dumps(record, default=str) + "\n")
        except Exception:
            # Never fail the training run because of a log write.
            pass


class GpuMemoryCallback(TrainerCallback):
    """Inject VRAM stats into the Trainer's log dict every N steps."""

    def __init__(self, log_every: int = 10) -> None:
        self.log_every = log_every

    def on_log(self, args, state, control, logs: Dict[str, Any] | None = None, **kwargs):
        if logs is None:
            return
        if not torch.cuda.is_available():
            return
        try:
            free, total = torch.cuda.mem_get_info(0)
            used_gb = (total - free) / 2**30
            peak_gb = torch.cuda.max_memory_allocated(0) / 2**30
            reserved_gb = torch.cuda.memory_reserved(0) / 2**30
            logs["gpu/used_gb"] = round(used_gb, 2)
            logs["gpu/peak_gb"] = round(peak_gb, 2)
            logs["gpu/reserved_gb"] = round(reserved_gb, 2)
            logs["gpu/total_gb"] = round(total / 2**30, 2)
        except Exception:
            pass


class TokensPerSecCallback(TrainerCallback):
    """Measure effective tokens/sec (forward + backward) using step duration."""

    def __init__(self) -> None:
        self._last_step_time: Optional[float] = None
        self._last_step_num: int = 0
        self._tokens_per_step: Optional[int] = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Estimate tokens/step from config: batch * max_length * grad_accum
        try:
            self._tokens_per_step = (
                args.per_device_train_batch_size
                * args.max_length
                * args.gradient_accumulation_steps
                * max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
            )
        except Exception:
            self._tokens_per_step = None
        self._last_step_time = time.time()
        self._last_step_num = 0

    def on_log(self, args, state, control, logs: Dict[str, Any] | None = None, **kwargs):
        if logs is None or self._tokens_per_step is None:
            return
        now = time.time()
        steps_delta = state.global_step - self._last_step_num
        if self._last_step_time is None or steps_delta <= 0:
            return
        dt = now - self._last_step_time
        if dt <= 0:
            return
        tokens_processed = steps_delta * self._tokens_per_step
        tps = tokens_processed / dt
        logs["perf/tokens_per_sec"] = round(tps, 1)
        logs["perf/sec_per_step"] = round(dt / steps_delta, 3)
        self._last_step_time = now
        self._last_step_num = state.global_step


def configure_wandb_env(run_name: str, project: str, entity: Optional[str]) -> None:
    """Set env vars consumed by HF Trainer's built-in wandb integration."""
    os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    # WANDB_NAME is the canonical env var; HF Trainer also reads it.
    os.environ["WANDB_NAME"] = run_name
    # Tell wandb to log the full config (otherwise it only logs TrainingArguments)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")
