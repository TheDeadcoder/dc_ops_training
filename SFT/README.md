# dc-ops-sft — DC-Ops agent SFT pipeline (Qwen3-8B on MI300X / ROCm 7.2)
Supervised fine-tuning of **Qwen3-8B** on [`Melikshah/dc-ops-sft-data`](https://huggingface.co/datasets/Melikshah/dc-ops-sft-data)
for the [DC-Ops OpenEnv environment](https://github.com/TheDeadcoder/dc_ops_environment).

## Layout
 
```
SFT/
├── install.sh                       # ROCm-aware bootstrap (torch, unsloth, vllm, trl)
├── pyproject.toml
├── configs/sft.yaml                 # all hyperparams
├── dc_ops_sft/
│   ├── data.py                      # HF dataset → fan-out → filter → (train, eval)
│   ├── env_eval.py                  # in-process DC-Ops rollouts, command parser
│   └── logging_utils.py             # JSONL + W&B + GPU mem / tok/s callbacks
└── scripts/
    ├── train_sft.py                 # the SFT run
    ├── eval_compare.py              # base vs SFT reward on all 6 scenarios
    └── push_to_hub.py               # merged + LoRA to HF Hub
```

### Setup and Train
```bash
chmod +x install.sh
./install.sh
source .venv/bin/activate
huggingface-cli login
wandb login 
```

### Verify the stack
 
```bash
python - <<'PY'
import torch, transformers, trl, unsloth
print(f"torch={torch.__version__}, hip={torch.version.hip}")
print(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"transformers={transformers.__version__}, trl={trl.__version__}, unsloth={unsloth.__version__}")
PY
```

### Train
 
```bash
python scripts/train_sft.py --config configs/sft.yaml
```

### Evaluate (base vs SFT)
Runs 3 episodes per scenario across all 6 scenarios, by default:
 
```bash
python scripts/eval_compare.py --config configs/sft.yaml \
    --output ./eval_compare_v1.json
```

### Push to HuggingFace
 
```bash
python scripts/push_to_hub.py --config configs/sft.yaml
```