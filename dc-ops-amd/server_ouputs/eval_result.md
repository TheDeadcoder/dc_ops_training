# Evaluation Run — 2026-04-22

## Overview

| Field | Value |
|---|---|
| Compared models | `unsloth/Qwen2.5-7B-Instruct` vs `./outputs/dc_ops_grpo_final` |
| Scenarios | `A4` and `B4` |
| Seeds | `100, 200, 300, 400, 500` |
| Temperature | `0.4` |
| Step budget | `10` |
| Episodes | `20` total (`2` models x `2` scenarios x `5` seeds) |
| Saved output | `./outputs/eval_results.json` |

This report includes both a polished interpretation and the exact raw evaluation output.

### Evaluation command

```bash
python scripts/evaluate.py \
  --grpo-model ./outputs/dc_ops_grpo_final \
  --base-model unsloth/Qwen2.5-7B-Instruct \
  --scenarios A4 B4 \
  --seeds 100 200 300 400 500 \
  --temperature 0.4
```

---

## Headline

- No episode resolved within the 10-step budget.
- GRPO is clearly stronger on `A4`, both in reward and in action selection.
- GRPO is more proactive on `B4`, but still inconsistent and noisy.
- The main logged advantage is improved task grounding and less passivity, not full recovery.

---

## How To Read The Metrics

| Metric | Meaning |
|---|---|
| `ResRate` | Fraction of episodes that fully resolved |
| `MeanRew` | Mean total episode reward across seeds |
| `StdRew` | Reward variability across seeds |
| `MeanStep` | Mean steps to resolution; blank when nothing resolves |
| `PerStep` | Mean reward per environment step |

Because every episode timed out, `ResRate` is `0.0%` everywhere and `MeanStep` is blank in all rows.

---

## Aggregate Results

| Model | Scenario | N | ResRate | MeanRew | StdRew | PerStep |
|---|---:|---:|---:|---:|---:|---:|
| base | A4 | 5 | 0.0% | 0.041 | 0.037 | 0.004 |
| base | B4 | 5 | 0.0% | -0.147 | 0.011 | -0.015 |
| grpo | A4 | 5 | 0.0% | 0.430 | 0.033 | 0.043 |
| grpo | B4 | 5 | 0.0% | -0.092 | 0.364 | -0.009 |

### Advantage over base model

| Scenario | MeanRew delta | PerStep delta | Read |
|---|---:|---:|---|
| A4 | `+0.389` | `+0.039` | Large and consistent GRPO gain |
| B4 | `+0.055` | `+0.006` | Small average gain, but highly variable |

---

## Raw Episode Results

### Base Model — Scenario A4

| Seed | Status | Steps | Total Reward | Actions |
|---:|---|---:|---:|---|
| 100 | timeout | 10 | 0.033 | `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `diagnose PDU-A-01`, `wait`, `wait` |
| 200 | timeout | 10 | 0.006 | `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `set_rack_load A-01 5.5`, `wait`, `wait`, `wait` |
| 300 | timeout | 10 | 0.018 | `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait` |
| 400 | timeout | 10 | 0.048 | `diagnose CRAC-1`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `diagnose PDU-A-01` |
| 500 | timeout | 10 | 0.102 | `diagnose CRAC-1`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `set_rack_load A-01 4.4` |

### Base Model — Scenario B4

| Seed | Status | Steps | Total Reward | Actions |
|---:|---|---:|---:|---|
| 100 | timeout | 10 | -0.128 | `wait`, `wait`, `wait`, `wait`, `wait`, `diagnose PDU-A-01`, `wait`, `wait`, `wait`, `wait` |
| 200 | timeout | 10 | -0.153 | `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait` |
| 300 | timeout | 10 | -0.153 | `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait` |
| 400 | timeout | 10 | -0.153 | `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait` |
| 500 | timeout | 10 | -0.148 | `start_generator`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait`, `wait` |

### GRPO Model — Scenario A4

| Seed | Status | Steps | Total Reward | Actions |
|---:|---|---:|---:|---|
| 100 | timeout | 10 | 0.448 | `diagnose CRAC-1`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-4 28.0`, `adjust_setpoint CRAC-2 28.0`, `adjust_setpoint CRAC-4 27.1`, `adjust_setpoint CRAC-2 27.1` |
| 200 | timeout | 10 | 0.463 | `diagnose CRAC-1`, `adjust_setpoint CRAC-1 28.0`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-2 22.1`, `adjust_setpoint CRAC-4 28.0`, `adjust_setpoint CRAC-4 28`, `adjust_setpoint CRAC-2 27.0` |
| 300 | timeout | 10 | 0.451 | `diagnose CRAC-1`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-2 20.0`, `adjust_setpoint CRAC-4 28.0`, `adjust_setpoint CRAC-4 20`, `adjust_setpoint CRAC-4 18.0`, `adjust_setpoint CRAC-2 18.0` |
| 400 | timeout | 10 | 0.395 | `diagnose CRAC-1`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-2 22`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-2 22.5` |
| 500 | timeout | 10 | 0.394 | `diagnose CRAC-1`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 22`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-4 22.0`, `adjust_setpoint CRAC-2 22.0`, `adjust_setpoint CRAC-4 20`, `adjust_setpoint CRAC-4 22.0` |

### GRPO Model — Scenario B4

| Seed | Status | Steps | Total Reward | Actions |
|---:|---|---:|---:|---|
| 100 | timeout | 10 | -0.255 | `start_generator`, `start_generator`, `set_rack_load A-01 10`, `set_rack_load A-02 10.0`, `set_rack_load A-03 11.0`, `set_rack_load A-04 11.0`, `set_rack_load A-05 11.0`, `set_rack_load A-04 11.0`, `set_rack_load A-05 11.0`, `set_rack_load A-03 10.0` |
| 200 | timeout | 10 | -0.388 | `start_generator`, `wait`, `set_rack_load A-01 11`, `set_rack_load A-02 11.0`, `set_rack_load A-03 11.0`, `set_rack_load A-4 11.0`, `set_rack_load A-4 0.0`, `set_rack_load A-4 8`, `set_rack_load A-4 8.0`, `set_rack_load A-4 11.0` |
| 300 | timeout | 10 | -0.146 | `start_generator`, `start_generator`, `set_rack_load A-01 0.0`, `set_rack_load A-02 0`, `set_rack_load A-03 00`, `set_rack_load A-04 0.0`, `set_rack_load A-05 01`, `set_rack_load A-01 1.0`, `set_rack_load A-0 1.0`, `set_rack_load A-01 01` |
| 400 | timeout | 10 | 0.540 | `diagnose GEN-1`, `start_generator`, `set_rack_load A-01 10`, `set_rack_load A-02 10.0`, `set_rack_load A-03 11.0`, `set_rack_load A-04 11.0`, `set_rack_load A-05 11.0`, `set_rack_load A-4 11.0`, `set_rack_load A-3 11.0`, `set_rack_load A-01 11.0` |
| 500 | timeout | 10 | -0.211 | `start_generator`, `start_generator`, `set_rack_load A-01 7.0`, `set_rack_load A-02 7.0`, `set_rack_load A-03 7.0`, `set_rack_load A-04 7.0`, `set_rack_load A-05 7.0`, `set_rack_load A-06 7.0`, `set_rack_load A-07 7.0`, `set_rack_load A-8 7.0` |

---

## What The Raw Results Show

### Scenario A4 — CRAC Failure Cascade

- The base model is mostly inert. Across all five seeds it spends most of the episode on `wait`, with occasional off-target commands like `diagnose PDU-A-01`.
- GRPO is consistently on-scenario. Every seed begins with `diagnose CRAC-1`, then moves into CRAC setpoint control.
- GRPO reward on `A4` is both higher and more stable: `0.394` to `0.463` versus the base model's `0.006` to `0.102`.
- This is the clearest logged advantage in the run.

### Scenario B4 — Power Failure Cascade

- The base model mostly waits and does very little useful work.
- GRPO reacts earlier by starting the generator in all five seeds.
- The policy is still unstable after that. Four of five GRPO `B4` runs remain negative, and several follow-up commands use malformed rack identifiers such as `A-4`, `A-0`, and `A-8`.
- Seed `400` is the only clearly strong `B4` rollout, reaching `total_reward=0.540`.

---

## Advantages Visible In The Logs

| Advantage | Evidence |
|---|---|
| Less passivity | The base model is dominated by `wait`; GRPO takes action immediately in most runs. |
| Better thermal grounding | On `A4`, GRPO uses CRAC diagnosis and setpoint control instead of drifting to irrelevant targets. |
| Earlier power response | On `B4`, GRPO starts the generator in every seed; the base model does so once. |
| Much stronger hard-thermal performance | `A4` mean reward improves from `0.041` to `0.430`. |
| Some upside on the hard-power case | `B4` mean reward improves from `-0.147` to `-0.092`, although variance is high and no run resolves. |

---

## Important Caveat

The correct takeaway is not that GRPO solves the hardest scenarios. It does not. Every episode in this run timed out, so the resolution rate remained `0.0%` on both `A4` and `B4`. The correct takeaway is that GRPO produces better emergency-response behavior than the base model, especially on `A4`, while still falling short of full recovery inside the current 10-step horizon.

---

## Runtime Notes

- A SciPy and NumPy compatibility warning appeared at startup, but the run completed successfully.
- ROCm architecture auto-detection warned about missing `rocminfo`, but model loading and inference still proceeded.
- Unsloth disabled 4-bit `bitsandbytes` on AMD during evaluation; this changed the loading path, not the interpretation of the results.

---

## Verbatim Raw Output

```text
(.venv) root@7:~/dc_ops_training/dc-ops-amd# python scripts/evaluate.py \
        --grpo-model ./outputs/dc_ops_grpo_final \
        --base-model unsloth/Qwen2.5-7B-Instruct \
        --scenarios A4 B4 \
        --seeds 100 200 300 400 500 \
        --temperature 0.4

============================================================
[eval] Evaluating: base  (unsloth/Qwen2.5-7B-Instruct)
============================================================
/usr/lib/python3.12/importlib/__init__.py:90: UserWarning: A NumPy version >=1.23.5 and <2.3.0 is required for this version of SciPy (detected version 2.4.4)
  return _bootstrap._gcd_import(name[level:], package, level)
[bitsandbytes.cuda_specs|ERROR]Could not detect ROCm GPU architecture: [Errno 2] No such file or directory: 'rocminfo'
[bitsandbytes.cuda_specs|WARNING]
ROCm GPU architecture detection failed despite ROCm being available.

g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
[eval] loading model from unsloth/Qwen2.5-7B-Instruct
Unsloth: AMD currently is not stable with 4bit bitsandbytes. Disabling for now.
==((====))==  Unsloth 2026.4.4: Fast Qwen2 patching. Transformers: 4.54.1. vLLM: 0.19.1+rocm721.
   \\   /|    AMD Instinct MI300X VF. Num GPUs = 1. Max memory: 191.688 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+rocm7.1. ROCm Toolkit: 7.1.25424. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.39s/it]
unsloth/Qwen2.5-7B-Instruct does not have a padding token! Will use pad_token = <|PAD_TOKEN|>.
[eval] model loaded

[eval] Scenario A4:
  seed=100 ... — timeout  steps=10  total_reward=0.033  actions=['wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'diagnose PDU-A-01', 'wait', 'wait']
  seed=200 ... — timeout  steps=10  total_reward=0.006  actions=['wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'set_rack_load A-01 5.5', 'wait', 'wait', 'wait']
  seed=300 ... — timeout  steps=10  total_reward=0.018  actions=['wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait']
  seed=400 ... — timeout  steps=10  total_reward=0.048  actions=['diagnose CRAC-1', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'diagnose PDU-A-01']
  seed=500 ... — timeout  steps=10  total_reward=0.102  actions=['diagnose CRAC-1', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'set_rack_load A-01 4.4']

[eval] Scenario B4:
  seed=100 ... — timeout  steps=10  total_reward=-0.128  actions=['wait', 'wait', 'wait', 'wait', 'wait', 'diagnose PDU-A-01', 'wait', 'wait', 'wait', 'wait']
  seed=200 ... — timeout  steps=10  total_reward=-0.153  actions=['wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait']
  seed=300 ... — timeout  steps=10  total_reward=-0.153  actions=['wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait']
  seed=400 ... — timeout  steps=10  total_reward=-0.153  actions=['wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait']
  seed=500 ... — timeout  steps=10  total_reward=-0.148  actions=['start_generator', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait', 'wait']

============================================================
[eval] Evaluating: grpo  (./outputs/dc_ops_grpo_final)
============================================================
[eval] loading model from ./outputs/dc_ops_grpo_final
Unsloth: AMD currently is not stable with 4bit bitsandbytes. Disabling for now.
==((====))==  Unsloth 2026.4.4: Fast Qwen2 patching. Transformers: 4.54.1. vLLM: 0.19.1+rocm721.
   \\   /|    AMD Instinct MI300X VF. Num GPUs = 1. Max memory: 191.688 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+rocm7.1. ROCm Toolkit: 7.1.25424. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.33s/it]
[eval] model loaded

[eval] Scenario A4:
  seed=100 ... — timeout  steps=10  total_reward=0.448  actions=['diagnose CRAC-1', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-4 28.0', 'adjust_setpoint CRAC-2 28.0', 'adjust_setpoint CRAC-4 27.1', 'adjust_setpoint CRAC-2 27.1']
  seed=200 ... — timeout  steps=10  total_reward=0.463  actions=['diagnose CRAC-1', 'adjust_setpoint CRAC-1 28.0', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-2 22.1', 'adjust_setpoint CRAC-4 28.0', 'adjust_setpoint CRAC-4 28', 'adjust_setpoint CRAC-2 27.0']
  seed=300 ... — timeout  steps=10  total_reward=0.451  actions=['diagnose CRAC-1', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-2 20.0', 'adjust_setpoint CRAC-4 28.0', 'adjust_setpoint CRAC-4 20', 'adjust_setpoint CRAC-4 18.0', 'adjust_setpoint CRAC-2 18.0']
  seed=400 ... — timeout  steps=10  total_reward=0.395  actions=['diagnose CRAC-1', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-2 22', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-2 22.5']
  seed=500 ... — timeout  steps=10  total_reward=0.394  actions=['diagnose CRAC-1', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 22', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-4 22.0', 'adjust_setpoint CRAC-2 22.0', 'adjust_setpoint CRAC-4 20', 'adjust_setpoint CRAC-4 22.0']

[eval] Scenario B4:
  seed=100 ... — timeout  steps=10  total_reward=-0.255  actions=['start_generator', 'start_generator', 'set_rack_load A-01 10', 'set_rack_load A-02 10.0', 'set_rack_load A-03 11.0', 'set_rack_load A-04 11.0', 'set_rack_load A-05 11.0', 'set_rack_load A-04 11.0', 'set_rack_load A-05 11.0', 'set_rack_load A-03 10.0']
  seed=200 ... — timeout  steps=10  total_reward=-0.388  actions=['start_generator', 'wait', 'set_rack_load A-01 11', 'set_rack_load A-02 11.0', 'set_rack_load A-03 11.0', 'set_rack_load A-4 11.0', 'set_rack_load A-4 0.0', 'set_rack_load A-4 8', 'set_rack_load A-4 8.0', 'set_rack_load A-4 11.0']
  seed=300 ... — timeout  steps=10  total_reward=-0.146  actions=['start_generator', 'start_generator', 'set_rack_load A-01 0.0', 'set_rack_load A-02 0', 'set_rack_load A-03 00', 'set_rack_load A-04 0.0', 'set_rack_load A-05 01', 'set_rack_load A-01 1.0', 'set_rack_load A-0 1.0', 'set_rack_load A-01 01']
  seed=400 ... — timeout  steps=10  total_reward=0.540  actions=['diagnose GEN-1', 'start_generator', 'set_rack_load A-01 10', 'set_rack_load A-02 10.0', 'set_rack_load A-03 11.0', 'set_rack_load A-04 11.0', 'set_rack_load A-05 11.0', 'set_rack_load A-4 11.0', 'set_rack_load A-3 11.0', 'set_rack_load A-01 11.0']
  seed=500 ... — timeout  steps=10  total_reward=-0.211  actions=['start_generator', 'start_generator', 'set_rack_load A-01 7.0', 'set_rack_load A-02 7.0', 'set_rack_load A-03 7.0', 'set_rack_load A-04 7.0', 'set_rack_load A-05 7.0', 'set_rack_load A-06 7.0', 'set_rack_load A-07 7.0', 'set_rack_load A-8 7.0']

================================================================================
Model                          Scenario      N   ResRate    MeanRew    StdRew   MeanStep   PerStep
--------------------------------------------------------------------------------
base                           A4            5      0.0%      0.041     0.037          —     0.004
base                           B4            5      0.0%     -0.147     0.011          —    -0.015
grpo                           A4            5      0.0%      0.430     0.033          —     0.043
grpo                           B4            5      0.0%     -0.092     0.364          —    -0.009
================================================================================

[eval] results saved → ./outputs/eval_results.json
```

---

## Conclusion

These logs show a real behavioral gain from GRPO, not a solved benchmark. On `A4`, the gain is strong, consistent, and easy to verify from the exact per-seed outputs. On `B4`, GRPO is more active than the base model but still unreliable. The evaluation therefore supports the claim that GRPO improves task-relevant control behavior on hard DC-Ops incidents while still failing to convert that improvement into actual resolution within 10 steps.