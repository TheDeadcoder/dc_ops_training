# DC-Ops SFT Data Generation Pipeline
Hf dataset: [Melikshah/dc-ops-sft-data](https://huggingface.co/datasets/Melikshah/dc-ops-sft-data)
End-to-end pipeline that uses **DeepSeek-R1-Distill-Qwen-32B** as a teacher
model to roll out trajectories against a live `DcOpsEnvironment`, then
filters and packages them as SFT training data.
 
## Format: three blocks per agent turn
 
```
<think>
[R1's natural messy chain of thought — exploration, self-correction, all of
it. Length is uncapped. The model thinks freely here in its native format.]
</think>
<reasoning>
1. Situation: [what the dashboard shows that matters].
2. Constraint: [the relevant ASHRAE limit, procedure rule, or system state].
3. Step: [which phase of assess→diagnose→compensate→verify→resolve].
4. Action: [the chosen command and why].
</reasoning>
<command>
diagnose CRAC-3
</command>
```
 
| Block | Role | Length | Required? |
|---|---|---|---|
| `<think>` | Model's freeform CoT — kept verbatim for inspection | uncapped (4KB safety net) | optional at filter time |
| `<reasoning>` | Distilled, structured, no-self-correction summary — the canonical training signal | ≤200 words, hard cap 1500 chars | **required** |
| `<command>` | The single action sent to the env | one line | **required** |
 
**Why three blocks (and not just `<reasoning>` or just `<think>`):**
- The teacher (R1-Distill) and the eventual student (Qwen3-8B) both emit `<think>` natively via their chat templates. Suppressing it fights pretraining and burns tokens on resistance.
- A `<reasoning>` block constrained to 4 numbered steps gives the model a *structure to fill* rather than freeform — this is the key trick that stops R1 from leaking "wait, actually..." into the summary.
- During SFT the student learns to think AND to summarize. During GRPO you can keep `<think>` enabled, or switch the student to `enable_thinking=False` for ~5x faster rollouts (the `<reasoning>+<command>` structure stays from SFT memory).
## Output
 
```
data_out/
├── raw_episodes/shard_*.jsonl   # raw per-worker output
├── train.jsonl                  # SFT-ready (just `conversations`)
├── train_with_meta.jsonl        # same + per-episode metadata
├── stats.json                   # corpus stats (machine-readable)
├── stats.md                     # corpus stats (human-readable)
└── README.md                    # auto-generated HF dataset card
```

## Configuration knobs (`config.py`)
 
| Knob | Default | What it does |
|---|---|---|
| `SCENARIO_PLAN` | dict summing to 1520 | Per-scenario episode quotas |
| `MAX_AGENT_STEPS_PER_EPISODE` | 12 | Cap so episodes don't blow up if teacher loops |
| `THINK_HARD_CAP_CHARS` | 4000 | Safety net only — model picks natural depth |
| `REASONING_HARD_CAP_CHARS` | 1500 | ≈250 words; truncated at sentence boundary |
| `REASONING_TARGET_MAX_WORDS` | 200 | What we tell the teacher in the prompt |
| `KEEP_MIN_AGENT_TURNS` | 1 | Even single-turn correct trajectories are valid |
| `KEEP_MIN_AVG_REWARD` | -0.20 | Drop low-quality trajectories |
| `KEEP_MIN_REASONING_CHARS` | 30 | Drop turns where `<reasoning>` is junk-short |
| `KEEP_NO_INVALID_CMDS` | True | Drop any episode that produced an unparseable command |
| `KEEP_NO_ESCALATION` | True | Never train the model to escalate (it's penalized) |
| `TARGET_FINAL_COUNT` | 1200 | Final dataset size cap (round-robin balanced) |
| `TEACHER_MAX_CONCURRENT` | 28 | Semaphore cap for in-flight teacher requests (sized to MI300X 192GB KV-cache budget at max-model-len 24576) |
| `TEACHER_MAX_TOKENS` | 2560 | think (~1500) + reasoning (~300) + command + slack |
| `TEACHER_TEMPERATURE` | 0.6 | Per DeepSeek's recommended R1 distill setting |
 
## Why DeepSeek-R1-Distill-Qwen-32B as the teacher
 
- **Native `<think>` reasoning** — same format Qwen3-8B (your student) emits.
- **Strong structured-output behavior** — reliably follows the "fill these 4 numbered points" template inside `<reasoning>`.
- **Fits on one MI300X at bf16 with `max_model_len=24576`** while leaving headroom for ~30 concurrent rollouts.

## Setup:
- **Device**: 192GB MI300X
- Start the teacher model via vLLM
  ```bash
  docker run -it --rm \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e GLOO_SOCKET_IFNAME=lo -e NCCL_SOCKET_IFNAME=lo -e VLLM_USE_V1=0 \
  vllm/vllm-openai-rocm:v0.17.1 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 24576 --gpu-memory-utilization 0.90 \
  --port 8000
  ```
- Setup UV and install dependencies for the dc_ops_environment
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.bashrc
  uv venv
  source .venv/bin/activate
  cd dc_ops_environment/dc_ops_env
  uv sync
  ```
- Make scripts executable
  ```bash
  chmod +x run_in_background.sh status.sh stop.sh
  ```
- Export the environment path
  ```bash
  export DC_OPS_ENV_PATH = dc_ops_environment
  ```
- Launch generation in the background. Detached from terminal
  ```bash
  bash run_in_background.sh
  ```
- Check progress of Generation
  ```bash
  bash status.sh
  ```
- Stop generation gracefully
  ```bash
  bash stop.sh
  ```
- After completion, run filter + stats
  ```bash
  python validate_and_stats.py
  ```
- Upload to HF
  ```bash
  HF_TOKEN=hf_xxx python upload_hf.py YourUsername/dc-ops-sft
  ```

# DC-Ops SFT Dataset — Statistics
**Huggingface Dataset**: Melikshah/dc-ops-sft-data
**Raw episodes generated:** 1520
**Episodes after filtering:** 1083
**Final episode count (after balanced cap):** 1083

**Format:** `<think>...</think><reasoning>...</reasoning><command>...</command>`
- `<think>`: model's freeform CoT (optional, kept verbatim)
- `<reasoning>`: structured ≤200-word distilled summary (canonical training signal)
- `<command>`: the action sent to the env

## Filter drops
| Reason | Count |
|---|---|
| too_few_turns | 362 |
| invalid_command | 65 |
| escalated | 10 |
## Headline numbers
- **Total conversation turns**: 17857
- **Agent turns (SFT targets)**: 8387
- **Resolved**: 179 (16.53%)
- **Episodes with self-correction leak in <reasoning>**: 0
- **Agent turns containing a non-empty <think>**: 100.0%
## Scenario key distribution
| Scenario | Count | % |
|---|---|---|
| A1 | 145 | 13.39% |
| A2 | 225 | 20.78% |
| A4 | 181 | 16.71% |
| B1 | 120 | 11.08% |
| B3 | 160 | 14.77% |
| B4 | 103 | 9.51% |
| VAR_CRAC_MAINT | 30 | 2.77% |
| VAR_CRAC_STANDBY | 41 | 3.79% |
| VAR_GEN_LOWFUEL | 25 | 2.31% |
| VAR_UPS_MODE | 53 | 4.89% |
## Difficulty distribution
| Difficulty | Count | % |
|---|---|---|
| custom | 149 | 13.76% |
| easy | 305 | 28.16% |
| hard | 284 | 26.22% |
| medium | 345 | 31.86% |
## Command distribution (across all agent turns)
| Command | Count | % of agent turns |
|---|---|---|
| `set_rack_load` | 4332 | 51.65% |
| `diagnose` | 1286 | 15.33% |
| `check_status` | 1046 | 12.47% |
| `adjust_setpoint` | 692 | 8.25% |
| `start_generator` | 250 | 2.98% |
| `wait` | 202 | 2.41% |
| `set_fan_speed` | 184 | 2.19% |
| `acknowledge_alarm` | 167 | 1.99% |
| `start_crac` | 93 | 1.11% |
| `set_ups_mode` | 64 | 0.76% |
| `stop_crac` | 40 | 0.48% |
| `refuel_generator` | 23 | 0.27% |
| `stop_generator` | 8 | 0.1% |

## Agent turns per episode
min=1 • p25=6 • **median=8** • p75=10 • max=12 • mean=7.74

## `<reasoning>` length (chars per agent turn)
min=122 • p25=310 • **median=347** • p75=388 • max=810 • mean=353.1

## `<reasoning>` length (words per agent turn) — target ≤200
min=8 • p25=47 • **median=52** • p75=59 • max=130 • mean=53.2

## `<think>` length (chars per agent turn, when present)
min=363 • p25=1851 • **median=2435** • p75=3139 • max=4000 • mean=2512.88

## Cumulative reward per episode
min=-0.471 • p25=-0.078 • median=0.071 • p75=0.376 • max=1.403 • mean=0.19




