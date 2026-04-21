# Evaluation Run — 2026-04-22

## Overview

| Field | Value |
|---|---|
| Compared models | `unsloth/Qwen2.5-7B-Instruct` vs `./outputs/dc_ops_grpo_final` |
| Scenarios | `A4` and `B4` |
| Seeds | `100, 200, 300, 400, 500` |
| Temperature | `0.4` |
| Step budget | `10` |
| Episodes | `20` total (`2` models × `2` scenarios × `5` seeds) |
| Saved output | `./outputs/eval_results.json` |

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
- GRPO showed a clear behavioral improvement on `A4`, with much higher and far more consistent rewards.
- GRPO was more proactive than the base model on `B4`, but the follow-up actions were unstable and often malformed.
- The main advantage visible in these logs is better task grounding and less passivity, not end-to-end scenario resolution yet.

---

## How To Read The Metrics

| Metric | Meaning |
|---|---|
| `ResRate` | Fraction of episodes that fully resolved |
| `MeanRew` | Mean total episode reward across seeds |
| `StdRew` | Reward variability across seeds |
| `MeanStep` | Mean steps to resolution; blank when nothing resolves |
| `PerStep` | Mean reward per environment step |

Because every episode timed out, `ResRate` is `0.0%` everywhere and `MeanStep` is blank for all rows.

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
| B4 | `+0.055` | `+0.006` | Small average gain, but very high variance |

---

## Scenario A4 — CRAC Failure Cascade

### Base model behavior

- Mostly idle behavior dominated by repeated `wait`.
- Occasional off-target actions such as `diagnose PDU-A-01`.
- Only weak positive reward across all seeds, ranging from `0.006` to `0.102`.

### GRPO behavior

- Consistently opens with `diagnose CRAC-1`.
- Follows with repeated thermal-control actions such as `adjust_setpoint CRAC-2 ...` and `adjust_setpoint CRAC-4 ...`.
- All five seeds are clearly positive, ranging from `0.394` to `0.463`.
- Behavior is stable across seeds: low reward variance and a strong scenario-specific action pattern.

### What the logs show

`A4` is the strongest result in this evaluation. GRPO is much less passive than the base model and stays on-topic with CRAC-related actions instead of drifting into irrelevant commands or repeated waiting. The logs do not show full recovery yet, but they do show substantially better thermal incident handling.

---

## Scenario B4 — Power Failure Cascade

### Base model behavior

- Mostly repeated `wait` actions.
- Only one seed starts the generator, and even that run still stalls afterward.
- Rewards stay consistently negative, from `-0.153` to `-0.128`.

### GRPO behavior

- Starts the generator immediately or near-immediately in every seed.
- Tries load-control follow-ups much more often than the base model.
- Produces one strong positive run at seed `400` with `total_reward=0.540`.
- Remains unstable overall: four of five runs are still negative.
- Follow-up commands are noisy, with malformed rack identifiers such as `A-4`, `A-0`, and `A-8` appearing in the traces.

### What the logs show

`B4` shows a narrower advantage. GRPO recognizes the emergency faster and acts earlier, especially through `start_generator`, but the policy is not yet reliable. The reward distribution is wide, which means the model sometimes finds a productive sequence but does not do so consistently.

---

## Advantages Visible In The Logs

| Advantage | Evidence |
|---|---|
| Less passivity | The base model is dominated by long `wait` streaks; GRPO takes action quickly in both scenarios. |
| Better scenario grounding on `A4` | GRPO repeatedly uses CRAC diagnosis and setpoint control, while the base model often idles or drifts. |
| Earlier emergency response on `B4` | GRPO starts the generator in all five seeds; the base model does so only once. |
| Stronger reward on the thermal hard case | `A4` mean reward improves from `0.041` to `0.430`. |
| Some upside on the power hard case | `B4` mean reward is less negative under GRPO and includes one clearly successful-looking trajectory, even though nothing fully resolves. |

---

## Important Caveat

The correct takeaway is not that GRPO solves the hardest scenarios. It does not. Every episode in this run timed out, so the resolution rate is still `0.0%` on both `A4` and `B4`. The correct takeaway is that GRPO produces better emergency-response behavior than the base model, especially on `A4`, while still falling short of full recovery inside the current 10-step horizon.

---

## Runtime Notes

- A SciPy and NumPy compatibility warning appeared at startup, but evaluation completed successfully.
- ROCm architecture auto-detection warned about missing `rocminfo`, but model loading and inference still proceeded.
- Unsloth disabled 4-bit `bitsandbytes` on AMD during evaluation; this changed the loading path, not the interpretation of the results.

---

## Conclusion

These logs show a meaningful behavioral gain from GRPO, not a solved benchmark. On `A4`, the gain is strong, consistent, and easy to see in both the rewards and the action traces. On `B4`, the model is more proactive than the base model, but the policy is still noisy and unreliable. The current evaluation therefore supports the claim that GRPO improves task-relevant control behavior on hard DC-Ops incidents, while additional work is still needed to convert that better behavior into actual resolution.