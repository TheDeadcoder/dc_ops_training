# DC-Ops SFT Data Generation Pipeline
 
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
- **Fits on one MI300X at bf16 with `max_model_len=8192`** while leaving headroom for ~30 concurrent rollouts.

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





