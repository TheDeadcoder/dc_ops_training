#!/usr/bin/env python
# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
Push a trained SFT LoRA adapter to the HuggingFace Hub.

This is a separate step so you can:
  (a) train → verify → push (human-in-the-loop),
  (b) train on machine A, push from machine B without re-training,
  (c) skip pushing entirely.

Usage:
    export HUGGINGFACE_TOKEN=hf_...
    python scripts/push_to_hf.py \\
        --local-dir  outputs/dc_ops_sft_lora \\
        --repo-id    your-username/dc-ops-sft-lora \\
        [--private]  [--commit-message "..."]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src import rocm_env  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push SFT LoRA adapter to HuggingFace Hub")
    p.add_argument("--local-dir", default="outputs/dc_ops_sft_lora",
                   help="Where the LoRA adapter was saved by run_sft.py.")
    p.add_argument("--repo-id", required=True,
                   help="Target HF repo, e.g. 'your-username/dc-ops-sft-lora'.")
    p.add_argument("--private", action="store_true",
                   help="Push as a private repository.")
    p.add_argument("--commit-message", default="SFT stage completed - DC-Ops Qwen2.5-7B (MI300X)",
                   help="Commit message for the LoRA push.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN (or HF_TOKEN) not set in the environment. "
            "Get a token at https://huggingface.co/settings/tokens and "
            "export it before running this script."
        )

    local_dir = pathlib.Path(args.local_dir).resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"Local LoRA dir does not exist: {local_dir}")
    # Sanity: require adapter_config.json (PEFT) to exist
    if not (local_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"{local_dir}/adapter_config.json missing — is this really a LoRA adapter dir?"
        )

    from huggingface_hub import login, HfApi
    login(token=token)
    api = HfApi()

    # Ensure the repo exists (create if missing)
    api.create_repo(
        repo_id=args.repo_id,
        private=args.private,
        exist_ok=True,
        repo_type="model",
    )

    # --- Upload the LoRA adapter via the HfApi so we don't need to load
    #     the model back into VRAM just to push it. This is faster and
    #     works from a machine that doesn't even have a GPU.
    print(f"[push] uploading {local_dir} → https://huggingface.co/{args.repo_id}")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=["*.bin.tmp", "__pycache__", "*.pyc"],
    )

    print(f"[push] DONE. model is at https://huggingface.co/{args.repo_id}")
    print(f"[push] next step:  python scripts/run_grpo.py --config configs/grpo.yaml")


if __name__ == "__main__":
    main()
