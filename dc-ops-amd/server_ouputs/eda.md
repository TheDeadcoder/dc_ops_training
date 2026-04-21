# DC-Ops SFT Dataset — EDA

**Source:** `Melikshah/dc-ops-sft-data` (HuggingFace)

---

## Episode Summary

| Metric | Count | Share |
|---|---|---|
| Raw episodes | 1,083 | — |
| Dropped (`VAR_*` rows) | 149 | 13.8% |
| Kept episodes | 934 | 86.2% |

### Episode Split by Scenario

| Scenario | Count | Share |
|---|---|---|
| A1 | 145 | 15.5% |
| A2 | 225 | 24.1% |
| A4 | 181 | 19.4% |
| B1 | 120 | 12.8% |
| B3 | 160 | 17.1% |
| B4 | 103 | 11.0% |

---

## Command Frequency

Counts across all GPT turns in kept episodes (n = 7,107 turns).

| Command | Count | Share |
|---|---|---|
| `set_rack_load` | 3,658 | 51.5% |
| `diagnose` | 1,128 | 15.9% |
| `check_status` | 853 | 12.0% |
| `adjust_setpoint` | 650 | 9.1% |
| `start_generator` | 249 | 3.5% |
| `wait` | 197 | 2.8% |
| `set_fan_speed` | 137 | 1.9% |
| `acknowledge_alarm` | 136 | 1.9% |
| `start_crac` | 52 | 0.7% |
| `set_ups_mode` | 25 | 0.4% |
| `stop_crac` | 14 | 0.2% |
| `stop_generator` | 8 | 0.1% |

---

## Length Statistics

All measurements in characters. Rough token estimate: `chars / 4`.

| Field | n | Mean | Median | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| Full GPT turn | 7,107 | 2,977.8 | 2,893 | 4,409 | 4,509 | 4,771 |
| `<think>` block | 7,107 | 2,541.3 | 2,463 | 3,965 | 3,995 | 4,002 |
| `<reasoning>` block | 7,107 | 356.6 | 351 | 467 | 546 | 812 |
| `<command>` block | 7,107 | 18.9 | 20 | 25 | 27 | 36 |

---

## `max_seq_length` Recommendation

| Estimate | Tokens |
|---|---|
| Window p99 (upper-bound) | ~2,584 |
| Empirical p99 (tokenizer-measured) | ~2,700 |
| Empirical max (tokenizer-measured) | ~2,900 |
| `configs/sft.yaml` setting | **4,096** |

`configs/sft.yaml` sets `max_seq_length=4096` — safe, covers the empirical max with packing headroom.

> **Warning:** Do not drop `max_seq_length` below **3,072**. Values below this threshold would truncate more than 0% of windows.