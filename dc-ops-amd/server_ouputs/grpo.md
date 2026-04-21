# GRPO Training Report

## Run Snapshot

| Item | Value |
| --- | --- |
| W&B run | `nazmus-sakib-touhid/dc-ops-amd/runs/gy38x0m3` |
| Run name | `grpo-qwen2.5-7b-mi300x` |
| Reward stack | `format_reward_fn + env_reward_fn + command_quality_fn + no_repeat_fn` |
| Prompt source | `build_grpo_prompts()` generates initial and mid-game prompts for `A1`, `A2`, `A4`, `B1`, `B3`, and `B4` |
| Prompt metadata carried into rewards | `scenario_id`, `seed`, `warmup_actions` |
| Prompt budget from config | `30` initial + `55` mid-game per scenario, or about `510` prompts before any aborted warmups |
| GRPO rollout shape | `num_generations=8`, `per_device_train_batch_size=8`, `gradient_accumulation_steps=2` |
| Effective batch | `16 = 2 prompts x 8 completions` per optimizer step |
| Length caps | `max_prompt_length=2048`, `max_completion_length=512` |
| Sampling | `temperature=0.9`, `top_p=0.9`, vLLM enabled |
| Optimizer | `learning_rate=5e-6`, cosine schedule, `warmup_ratio=0.1`, `beta=0.05`, `max_grad_norm=0.1` |
| vLLM memory target | `gpu_memory_utilization=0.65` |

> Note: the live W&B API export for this run currently returns history through epoch `6.9216` at step `15884`. The later local tail lines you pasted around epochs `6.95` to `7.00` are not present in the API-visible history, so the tables below reflect the W&B-exported portion of the run.

## Reward Flow, End to End

1. `build_grpo_prompts()` creates prompts from the live DC-Ops simulator. Each prompt carries the current observation plus `scenario_id`, deterministic `seed`, and optional `warmup_actions` that replay a partial action history.
2. `GRPOTrainer(..., reward_funcs=ALL_REWARD_FNS, ...)` sends every sampled completion through all four reward functions. Those functions receive the completion text and the prompt metadata fields above.
3. The trainer logs each reward component separately under W&B, then logs total reward as the sum of the four components.

```text
total_reward = format_reward_fn
						 + env_reward_fn
						 + command_quality_fn
						 + no_repeat_fn

env_reward_fn = clamp[-4.0, +5.0](
			3.0 * r_now
		+ 2.5 * (proxy_after - proxy_before)
		+ 1.0 * (best_proxy - proxy_before)
		+ 3.0 * resolved
		- 3.0 * crashed
)
```

### 1. `format_reward_fn`

- Purpose: structural validity and output hygiene.
- It requires both `<reasoning>...</reasoning>` and `<command>...</command>`.
- It penalizes missing tags, unknown command verbs, stray tail text after `</command>`, `<think>` leakage, and reasoning that is too short or too long.
- Range: `[-1.5, +0.3]`.
- Practical effect in this run: fully saturated. W&B shows `format_reward_fn/mean = 0.15` and `format_reward_fn/std = 0.0` on every unique logged step, which means it acted as a constant positive offset rather than an active learning signal.

### 2. `env_reward_fn`

- Purpose: simulator-grounded reward.
- For each completion it resets the DC-Ops environment with the prompt's `scenario_id` and `seed`, replays `warmup_actions`, applies the proposed command, then probes up to four `wait` steps to see whether the state keeps improving.
- `_proxy_health()` mixes thermal state and power state into a continuous `[0, 1]` health score. That lets RL optimize a live signal even when `scenario.resolved` is not reachable at training time.
- It adds immediate environment reward, immediate proxy delta, best proxy delta over the wait probe, and then a bonus or penalty for resolve or crash outcomes.
- Range: `[-4.0, +5.0]`.
- Practical effect in this run: this is the main high-variance term. Mean component value was `0.5409`, average per-step component std was `0.4336`, and it went negative on `11/107` unique W&B logging steps.

### 3. `command_quality_fn`

- Purpose: scenario-aware action prior.
- It rewards or penalizes actions by scenario, turn phase, target, and parameter value.
- The function uses `scenario_id` and `warmup_actions` to distinguish opening moves from post-diagnosis follow-ups.
- It contains target-specific fixes such as preferring surviving CRAC units in `A4`, punishing actions on failed units, rewarding sensible setpoints, and rewarding aggressive but plausible load shedding.
- Range: `[-1.5, +1.2]`.
- Practical effect in this run: this was the most consistently positive term. Mean component value was `0.6531`, average per-step component std was `0.2300`, and it exceeded the environment reward on `84/107` unique W&B logging steps.

### 4. `no_repeat_fn`

- Purpose: punish repeated actions from the warmup history.
- It gives a hard penalty for exact repeats, a softer penalty for reusing the same verb and target with a different value, and a mild penalty for repeated `wait` once waiting has already happened.
- Neutral cases stay at `0.0`.
- Practical effect in this run: mostly dormant. Mean component value was `-0.0189`, average per-step component std was `0.0630`, and it fired negatively on `28/107` unique W&B logging steps.

## Empirical Read From W&B

- Total reward matches the sum of the four logged component means to within `3.278e-7`, so the decomposition in W&B is internally consistent.
- `format_reward_fn` is saturated and flat. The model already holds format, so this term is adding a constant bias but not much gradient signal.
- `env_reward_fn` is the dominant exploratory signal. Large swings in total reward mostly track this term.
- `command_quality_fn` is the stable policy prior. It keeps total reward positive even when the environment term briefly goes negative.
- `no_repeat_fn` is a guardrail, not a driver. It matters occasionally, but it is not shaping most steps.
- `completions/clipped_ratio` stayed at `0.0`, so the completion cap was never binding.
- Mean completion length averaged `120.05` tokens with a range of `98.56` to `147.44`, far below the `512` token completion cap.
- KL averaged `1.0612`, with only `4` unique logging steps above `2.0`. There was one large transient spike to `14.6099`.

## W&B Summary Tables

W&B raw history contained duplicate rows per logging step, so the tables below are deduplicated by `_step`.

| Metric | Value |
| --- | --- |
| Unique W&B logging steps | 107 |
| W&B epoch range | 0.0627 -> 6.9216 |
| Reward mean / min / max | 1.3251 / 0.3533 / 4.1300 |
| Env reward mean / std(mean) | 0.5409 / 0.4336 |
| Command reward mean / std(mean) | 0.6531 / 0.2300 |
| Format reward mean / std(mean) | 0.1500 / 0.0000 |
| No-repeat mean / std(mean) | -0.0189 / 0.0630 |
| KL mean / max | 1.0612 / 14.6099 |
| Loss mean / min / max | 0.0255 / -0.0292 / 0.3537 |
| Mean completion length mean / min / max | 120.05 / 98.56 / 147.44 |
| Max clipped ratio | 0.0000 |
| Rows with non-zero `frac_reward_zero_std` | 6 |
| Reward sum max abs delta | 0.0000003278 |

| Point | Step | Epoch | Reward | KL |
| --- | --- | --- | --- | --- |
| First logged point | 143 | 0.0627 | 0.3943 | 0.8789 |
| Best reward point | 5624 | 2.4510 | 4.1300 | 0.6962 |
| Worst reward point | 13391 | 5.8353 | 0.3533 | 1.4273 |
| Last W&B point | 15884 | 6.9216 | 1.1358 | 1.2693 |

## Recent W&B Logs

### Training State Tail

| Time (UTC) | Step | Epoch | Loss | Grad Norm | LR | Mean Len | KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-21 21:55:36 UTC | 13868 | 6.0431 | 0.0577 | 3.9662 | 2.817e-07 | 120.62 | 1.5630 |
| 2026-04-21 22:04:32 UTC | 14444 | 6.2941 | 0.0277 | 5.3266 | 1.551e-07 | 113.56 | 1.4562 |
| 2026-04-21 22:06:25 UTC | 14570 | 6.3490 | 0.0668 | 4.2074 | 1.322e-07 | 122.38 | 1.0023 |
| 2026-04-21 22:09:09 UTC | 14741 | 6.4235 | 0.0106 | 2.7213 | 1.040e-07 | 114.19 | 0.8037 |
| 2026-04-21 22:11:48 UTC | 14912 | 6.4980 | 0.0464 | 6.4982 | 7.918e-08 | 126.81 | 1.6216 |
| 2026-04-21 22:12:05 UTC | 14930 | 6.5059 | 0.0110 | 8.1052 | 7.675e-08 | 123.12 | 1.4274 |
| 2026-04-21 22:18:19 UTC | 15335 | 6.6824 | 0.0511 | 3.0691 | 3.209e-08 | 130.75 | 0.9172 |
| 2026-04-21 22:20:37 UTC | 15488 | 6.7490 | 0.0363 | 3.1789 | 2.018e-08 | 100.62 | 1.5647 |
| 2026-04-21 22:25:28 UTC | 15812 | 6.8902 | 0.0331 | 3.1649 | 4.022e-09 | 118.12 | 0.9574 |
| 2026-04-21 22:26:31 UTC | 15884 | 6.9216 | 0.0632 | 5.6986 | 2.109e-09 | 114.38 | 1.2693 |

### Reward Breakdown Tail

| Step | Format | Env | Command | No-repeat | Reward | Reward Std |
| --- | --- | --- | --- | --- | --- | --- |
| 13868 | 0.1500 | 0.3097 | 0.5906 | -0.0625 | 0.9878 | 0.2065 |
| 14444 | 0.1500 | -0.0100 | 0.3594 | 0.0000 | 0.4993 | 0.1858 |
| 14570 | 0.1500 | -0.0109 | 0.4456 | 0.0000 | 0.5847 | 0.1295 |
| 14741 | 0.1500 | 1.5834 | 0.8112 | -0.0250 | 2.5196 | 0.0923 |
| 14912 | 0.1500 | 0.2877 | 0.5644 | -0.0625 | 0.9395 | 0.2714 |
| 14930 | 0.1500 | 1.5499 | 0.4950 | -0.1875 | 2.0074 | 0.5306 |
| 15335 | 0.1500 | 0.4315 | 0.7700 | -0.0625 | 1.2890 | 0.4777 |
| 15488 | 0.1500 | 0.0222 | 0.5550 | 0.0000 | 0.7272 | 0.0425 |
| 15812 | 0.1500 | 0.3585 | 0.7625 | 0.0000 | 1.2710 | 0.4073 |
| 15884 | 0.1500 | 0.2977 | 0.6881 | 0.0000 | 1.1358 | 0.1855 |

## Profiling Summary

Profiling metrics are logged on a sparser stream than the reward metrics, so this table uses all non-null history rows for each profiler series.

| Profiler | Count | Mean Time | Last Value |
| --- | --- | --- | --- |
| UnslothGRPOTrainer.vLLM.generate | 2006 | 5.180668 | 4.891070 |
| UnslothGRPOTrainer._calculate_rewards | 1819 | 0.675331 | 0.731762 |
| UnslothGRPOTrainer.env_reward_fn | 1853 | 0.709761 | 0.649548 |
| UnslothGRPOTrainer.command_quality_fn | 1683 | 0.000696 | 0.000640 |
| UnslothGRPOTrainer.format_reward_fn | 2057 | 0.000427 | 0.000436 |
| UnslothGRPOTrainer.no_repeat_fn | 1666 | 0.000114 | 0.000148 |

Takeaway: generation dominates wall time, environment scoring is the only materially expensive reward function, and the three string- or heuristic-based reward functions are effectively free by comparison.

## Training Graphs

### Reward Components

<table>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_rewards_format_reward_fn_mean.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_rewards_format_reward_fn_mean.png" alt="Format Reward Mean" width="100%"></a><br>
			<strong>Format Reward Mean</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_rewards_format_reward_fn_std.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_rewards_format_reward_fn_std.png" alt="Format Reward Std" width="100%"></a><br>
			<strong>Format Reward Std</strong>
		</td>
	</tr>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_rewards_env_reward_fn_mean.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_rewards_env_reward_fn_mean.png" alt="Environment Reward Mean" width="100%"></a><br>
			<strong>Environment Reward Mean</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_rewards_env_reward_fn_std.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_rewards_env_reward_fn_std.png" alt="Environment Reward Std" width="100%"></a><br>
			<strong>Environment Reward Std</strong>
		</td>
	</tr>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_rewards_command_quality_fn_mean.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_rewards_command_quality_fn_mean.png" alt="Command Quality Mean" width="100%"></a><br>
			<strong>Command Quality Mean</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_rewards_command_quality_fn_std.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_rewards_command_quality_fn_std.png" alt="Command Quality Std" width="100%"></a><br>
			<strong>Command Quality Std</strong>
		</td>
	</tr>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_rewards_no_repeat_fn_mean.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_rewards_no_repeat_fn_mean.png" alt="No Repeat Mean" width="100%"></a><br>
			<strong>No-Repeat Mean</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_rewards_no_repeat_fn_std.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_rewards_no_repeat_fn_std.png" alt="No Repeat Std" width="100%"></a><br>
			<strong>No-Repeat Std</strong>
		</td>
	</tr>
</table>

### Aggregate Reward and KL

<table>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_reward.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_reward.png" alt="Total Reward" width="100%"></a><br>
			<strong>Total Reward</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_reward_std.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_reward_std.png" alt="Total Reward Std" width="100%"></a><br>
			<strong>Total Reward Std</strong>
		</td>
	</tr>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_kl.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_kl.png" alt="KL" width="100%"></a><br>
			<strong>KL</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_frac_reward_zero_std.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_frac_reward_zero_std.png" alt="Fraction Reward Zero Std" width="100%"></a><br>
			<strong>Fraction Reward Zero Std</strong>
		</td>
	</tr>
</table>

### Completion Length and Clipping

<table>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_completions_min_length.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_completions_min_length.png" alt="Minimum Completion Length" width="100%"></a><br>
			<strong>Minimum Completion Length</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_completions_min_terminated_length.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_completions_min_terminated_length.png" alt="Minimum Terminated Length" width="100%"></a><br>
			<strong>Minimum Terminated Length</strong>
		</td>
	</tr>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_completions_mean_length.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_completions_mean_length.png" alt="Mean Completion Length" width="100%"></a><br>
			<strong>Mean Completion Length</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/rain_completions_mean_terminated_length.png"><img src="https://ik.imagekit.io/sakib61/GRPO/rain_completions_mean_terminated_length.png" alt="Mean Terminated Length" width="100%"></a><br>
			<strong>Mean Terminated Length</strong>
		</td>
	</tr>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_completions_max_length.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_completions_max_length.png" alt="Maximum Completion Length" width="100%"></a><br>
			<strong>Maximum Completion Length</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/rain_completions_max_terminated_length.png"><img src="https://ik.imagekit.io/sakib61/GRPO/rain_completions_max_terminated_length.png" alt="Maximum Terminated Length" width="100%"></a><br>
			<strong>Maximum Terminated Length</strong>
		</td>
	</tr>
	<tr>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_completions_clipped_ratio.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_completions_clipped_ratio.png" alt="Completion Clipped Ratio" width="100%"></a><br>
			<strong>Completion Clipped Ratio</strong>
		</td>
		<td align="center" width="50%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_completion_length.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_completion_length.png" alt="Completion Length" width="100%"></a><br>
			<strong>Completion Length</strong>
		</td>
	</tr>
</table>

### Optimization Metrics

<table>
	<tr>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_loss.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_loss.png" alt="Loss" width="100%"></a><br>
			<strong>Loss</strong>
		</td>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_learning_rate.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_learning_rate.png" alt="Learning Rate" width="100%"></a><br>
			<strong>Learning Rate</strong>
		</td>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/train_grad_norm.png"><img src="https://ik.imagekit.io/sakib61/GRPO/train_grad_norm.png" alt="Gradient Norm" width="100%"></a><br>
			<strong>Gradient Norm</strong>
		</td>
	</tr>
</table>

## Profiling Graphs

<table>
	<tr>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.vLLM.generate.png"><img src="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.vLLM.generate.png" alt="vLLM Generate Time" width="100%"></a><br>
			<strong>vLLM Generate Time</strong>
		</td>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer._calculate_rewards.png"><img src="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer._calculate_rewards.png" alt="Reward Calculation Time" width="100%"></a><br>
			<strong>Reward Calculation Time</strong>
		</td>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.env_reward_fn.png"><img src="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.env_reward_fn.png" alt="Environment Reward Time" width="100%"></a><br>
			<strong>Environment Reward Time</strong>
		</td>
	</tr>
	<tr>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.command_quality_fn.png"><img src="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.command_quality_fn.png" alt="Command Quality Time" width="100%"></a><br>
			<strong>Command Quality Time</strong>
		</td>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.format_reward_fn.png"><img src="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.format_reward_fn.png" alt="Format Reward Time" width="100%"></a><br>
			<strong>Format Reward Time</strong>
		</td>
		<td align="center" width="33%">
			<a href="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.no_repeat_fn.png"><img src="https://ik.imagekit.io/sakib61/GRPO/profiling_Time%20taken:%20UnslothGRPOTrainer.no_repeat_fn.png" alt="No Repeat Time" width="100%"></a><br>
			<strong>No-Repeat Time</strong>
		</td>
	</tr>
</table>

## Utilization

<table>
	<tr>
		<td align="center">
			<a href="https://ik.imagekit.io/sakib61/GRPO/GPU%20Utilization%20(_).png"><img src="https://ik.imagekit.io/sakib61/GRPO/GPU%20Utilization%20(_).png" alt="GPU Utilization" width="100%"></a><br>
			<strong>GPU Utilization</strong>
		</td>
	</tr>
</table>
