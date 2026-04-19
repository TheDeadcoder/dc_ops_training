"""
Validate, filter, and consolidate generated episodes.

Format expected: <think>...</think><reasoning>...</reasoning><command>...</command>
  - <think>     OPTIONAL  (kept for inspection, no length minimum)
  - <reasoning> REQUIRED  (must be ≥ KEEP_MIN_REASONING_CHARS)
  - <command>   REQUIRED

Reads all shards in data_out/raw_episodes/, applies quality filters,
emits:
  - data_out/train.jsonl              # SFT-ready (just `conversations`)
  - data_out/train_with_meta.jsonl    # same + per-episode metadata
  - data_out/stats.json               # corpus stats (machine-readable)
  - data_out/stats.md                 # corpus stats (human-readable)

After filtering we cap the kept set at TARGET_FINAL_COUNT, balancing
across scenario_key so rare-command coverage is preserved.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    KEEP_MIN_AGENT_TURNS,
    KEEP_MIN_AVG_REWARD,
    KEEP_MIN_REASONING_CHARS,
    KEEP_NO_ESCALATION,
    KEEP_NO_INVALID_CMDS,
    REASONING_HARD_CAP_CHARS,
    TARGET_FINAL_COUNT,
)


OUTPUT_DIR = Path("data_out")
RAW_DIR = OUTPUT_DIR / "raw_episodes"
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
TRAIN_META_PATH = OUTPUT_DIR / "train_with_meta.jsonl"
STATS_JSON = OUTPUT_DIR / "stats.json"
STATS_MD = OUTPUT_DIR / "stats.md"


_THINK_RE     = re.compile(r"<think\s*>(.*?)</think\s*>",         re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning\s*>(.*?)</reasoning\s*>", re.DOTALL | re.IGNORECASE)
_COMMAND_RE   = re.compile(r"<command\s*>(.*?)</command\s*>",     re.DOTALL | re.IGNORECASE)


def load_all_raw() -> list[dict]:
    records = []
    for shard in sorted(RAW_DIR.glob("shard_*.jsonl")):
        with shard.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def passes_filters(rec: dict, dropped: Counter) -> bool:
    if "error" in rec or "conversations" not in rec:
        dropped["error_or_no_convs"] += 1
        return False
    md = rec.get("metadata", {})
    if md.get("n_agent_turns", 0) < KEEP_MIN_AGENT_TURNS:
        dropped["too_few_turns"] += 1
        return False
    if KEEP_NO_INVALID_CMDS and md.get("invalid_commands", 0) > 0:
        dropped["invalid_command"] += 1
        return False
    if KEEP_NO_ESCALATION and md.get("escalated", False):
        dropped["escalated"] += 1
        return False
    if md.get("avg_reward", 0.0) < KEEP_MIN_AVG_REWARD:
        dropped["low_avg_reward"] += 1
        return False
    # Per-turn structural sanity
    for c in rec["conversations"]:
        if c["from"] != "gpt":
            continue
        v = c["value"]
        if not _COMMAND_RE.search(v):
            dropped["missing_command_tag"] += 1
            return False
        m_reasoning = _REASONING_RE.search(v)
        if not m_reasoning:
            dropped["missing_reasoning_block"] += 1
            return False
        rlen = len(m_reasoning.group(1).strip())
        if rlen < KEEP_MIN_REASONING_CHARS:
            dropped["reasoning_too_short"] += 1
            return False
        if rlen > REASONING_HARD_CAP_CHARS + 100:  # post-trim sanity (allows small drift)
            dropped["reasoning_too_long"] += 1
            return False
    return True


def cap_balanced(records: list[dict], target: int) -> list[dict]:
    if len(records) <= target:
        return records
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_key[r["metadata"]["scenario_key"]].append(r)
    for k in by_key:
        by_key[k].sort(key=lambda r: r["metadata"]["avg_reward"], reverse=True)

    keys = list(by_key.keys())
    picked: list[dict] = []
    idxs = {k: 0 for k in keys}
    while len(picked) < target:
        progressed = False
        for k in keys:
            if idxs[k] < len(by_key[k]):
                picked.append(by_key[k][idxs[k]])
                idxs[k] += 1
                progressed = True
                if len(picked) >= target:
                    break
        if not progressed:
            break
    return picked


def compute_corpus_stats(records: list[dict]) -> dict:
    n_eps = len(records)
    total_turns = sum(len(r["conversations"]) for r in records)
    n_agent_turns = sum(r["metadata"]["n_agent_turns"] for r in records)

    cmd_counts: Counter = Counter()
    scenario_counts: Counter = Counter()
    scenario_type_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()
    think_lengths: list[int] = []
    reasoning_lengths: list[int] = []
    reasoning_word_counts: list[int] = []
    turn_counts: list[int] = []
    rewards: list[float] = []
    resolved_count = 0
    self_correction_episodes = 0
    think_present_count = 0

    for r in records:
        md = r["metadata"]
        scenario_counts[md["scenario_key"]] += 1
        scenario_type_counts[md.get("scenario_type", "")] += 1
        difficulty_counts[md.get("difficulty", "")] += 1
        turn_counts.append(md["n_agent_turns"])
        rewards.append(md["cumulative_reward"])
        if md.get("resolved", False):
            resolved_count += 1
        if md.get("self_correction_leaks", 0) > 0:
            self_correction_episodes += 1
        for cmd, n in md["command_counts"].items():
            cmd_counts[cmd] += n
        for c in r["conversations"]:
            if c["from"] != "gpt":
                continue
            v = c["value"]
            m_t = _THINK_RE.search(v)
            if m_t:
                tl = len(m_t.group(1).strip())
                if tl > 0:
                    think_lengths.append(tl)
                    think_present_count += 1
            m_r = _REASONING_RE.search(v)
            if m_r:
                rstr = m_r.group(1).strip()
                reasoning_lengths.append(len(rstr))
                reasoning_word_counts.append(len(rstr.split()))

    def pct(d, total):
        return {k: (v, round(100 * v / total, 2)) for k, v in d.items()} if total else {}

    def quartiles(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        n = len(xs)
        return {
            "min": xs[0], "p25": xs[n // 4], "median": xs[n // 2],
            "p75": xs[3 * n // 4], "max": xs[-1],
            "mean": round(sum(xs) / n, 2),
        }

    return {
        "num_episodes": n_eps,
        "num_total_conversation_turns": total_turns,
        "num_agent_turns_training_targets": n_agent_turns,
        "scenario_key_distribution": pct(dict(scenario_counts), n_eps),
        "scenario_type_distribution": pct(dict(scenario_type_counts), n_eps),
        "difficulty_distribution": pct(dict(difficulty_counts), n_eps),
        "command_distribution": pct(dict(cmd_counts), n_agent_turns),
        "agent_turns_per_episode": quartiles(turn_counts),
        "think_chars_per_turn": quartiles(think_lengths),
        "think_present_pct": round(100 * think_present_count / n_agent_turns, 2) if n_agent_turns else 0,
        "reasoning_chars_per_turn": quartiles(reasoning_lengths),
        "reasoning_words_per_turn": quartiles(reasoning_word_counts),
        "cumulative_reward_per_episode": quartiles(rewards),
        "resolved_episodes": resolved_count,
        "resolved_pct": round(100 * resolved_count / n_eps, 2) if n_eps else 0,
        "episodes_with_self_correction_leak": self_correction_episodes,
    }


def render_stats_md(stats: dict, dropped: Counter, raw_n: int, kept_n: int) -> str:
    L = []
    L.append("# DC-Ops SFT Dataset — Statistics\n")
    L.append(f"**Raw episodes generated:** {raw_n}")
    L.append(f"**Episodes after filtering:** {kept_n}")
    L.append(f"**Final episode count (after balanced cap):** {stats['num_episodes']}\n")
    L.append("**Format:** `<think>...</think><reasoning>...</reasoning><command>...</command>`")
    L.append("- `<think>`: model's freeform CoT (optional, kept verbatim)")
    L.append("- `<reasoning>`: structured ≤200-word distilled summary (canonical training signal)")
    L.append("- `<command>`: the action sent to the env\n")

    L.append("## Filter drops")
    if dropped:
        L.append("| Reason | Count |")
        L.append("|---|---|")
        for k, v in dropped.most_common():
            L.append(f"| {k} | {v} |")
    else:
        L.append("(no drops)")
    L.append("")

    L.append("## Headline numbers")
    L.append(f"- **Total conversation turns**: {stats['num_total_conversation_turns']}")
    L.append(f"- **Agent turns (SFT targets)**: {stats['num_agent_turns_training_targets']}")
    L.append(f"- **Resolved**: {stats['resolved_episodes']} ({stats['resolved_pct']}%)")
    L.append(f"- **Episodes with self-correction leak in <reasoning>**: {stats['episodes_with_self_correction_leak']}")
    L.append(f"- **Agent turns containing a non-empty <think>**: {stats['think_present_pct']}%\n")

    L.append("## Scenario key distribution")
    L.append("| Scenario | Count | % |")
    L.append("|---|---|---|")
    for k, (c, p) in sorted(stats["scenario_key_distribution"].items()):
        L.append(f"| {k} | {c} | {p}% |")
    L.append("")

    L.append("## Difficulty distribution")
    L.append("| Difficulty | Count | % |")
    L.append("|---|---|---|")
    for k, (c, p) in sorted(stats["difficulty_distribution"].items()):
        L.append(f"| {k} | {c} | {p}% |")
    L.append("")

    L.append("## Command distribution (across all agent turns)")
    L.append("| Command | Count | % of agent turns |")
    L.append("|---|---|---|")
    items = sorted(stats["command_distribution"].items(), key=lambda kv: kv[1][0], reverse=True)
    for k, (c, p) in items:
        L.append(f"| `{k}` | {c} | {p}% |")
    L.append("")

    L.append("## Agent turns per episode")
    q = stats["agent_turns_per_episode"]
    L.append(f"min={q['min']} • p25={q['p25']} • **median={q['median']}** • p75={q['p75']} • max={q['max']} • mean={q['mean']}")
    L.append("")

    L.append("## `<reasoning>` length (chars per agent turn)")
    q = stats["reasoning_chars_per_turn"]
    L.append(f"min={q['min']} • p25={q['p25']} • **median={q['median']}** • p75={q['p75']} • max={q['max']} • mean={q['mean']}")
    L.append("")

    L.append("## `<reasoning>` length (words per agent turn) — target ≤200")
    q = stats["reasoning_words_per_turn"]
    L.append(f"min={q['min']} • p25={q['p25']} • **median={q['median']}** • p75={q['p75']} • max={q['max']} • mean={q['mean']}")
    L.append("")

    L.append("## `<think>` length (chars per agent turn, when present)")
    q = stats["think_chars_per_turn"]
    if q:
        L.append(f"min={q['min']} • p25={q['p25']} • **median={q['median']}** • p75={q['p75']} • max={q['max']} • mean={q['mean']}")
    else:
        L.append("(none recorded)")
    L.append("")

    L.append("## Cumulative reward per episode")
    q = stats["cumulative_reward_per_episode"]
    L.append(f"min={round(q['min'],3)} • p25={round(q['p25'],3)} • median={round(q['median'],3)} • p75={round(q['p75'],3)} • max={round(q['max'],3)} • mean={round(q['mean'],3)}")

    return "\n".join(L) + "\n"


def main():
    raw = load_all_raw()
    print(f"loaded {len(raw)} raw episodes")

    dropped = Counter()
    kept = [r for r in raw if passes_filters(r, dropped)]
    print(f"kept {len(kept)} / {len(raw)} after filtering")
    for k, v in dropped.most_common():
        print(f"  dropped: {k}: {v}")

    final = cap_balanced(kept, TARGET_FINAL_COUNT)
    print(f"final: {len(final)} after balanced cap to {TARGET_FINAL_COUNT}")

    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps({"conversations": r["conversations"]}, ensure_ascii=False) + "\n")
    with TRAIN_META_PATH.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = compute_corpus_stats(final)
    with STATS_JSON.open("w") as f:
        json.dump(stats, f, indent=2)
    with STATS_MD.open("w") as f:
        f.write(render_stats_md(stats, dropped, len(raw), len(kept)))

    print(f"\nWrote: {TRAIN_PATH}")
    print(f"Wrote: {TRAIN_META_PATH}")
    print(f"Wrote: {STATS_JSON}")
    print(f"Wrote: {STATS_MD}")
    print()
    print("=" * 60)
    print(f"FINAL DATASET: {stats['num_episodes']} episodes / "
          f"{stats['num_agent_turns_training_targets']} agent turns")
    print("=" * 60)


if __name__ == "__main__":
    main()
