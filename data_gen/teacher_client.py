"""
Async client for the vLLM-served teacher.

Uses the official `openai` package against vLLM's OpenAI-compatible
endpoint. Concurrency is bounded by an asyncio.Semaphore.

Parses three blocks per response:
  - <think>      OPTIONAL: model's freeform CoT (kept verbatim if present)
  - <reasoning>  REQUIRED: distilled ≤200-word summary
  - <command>    REQUIRED: single command line

Rationale:
  Both R1-Distill and Qwen3 emit <think> via their chat template. We let
  the model think freely there, then we read the structured <reasoning>
  block as the canonical training signal. <command> is what the env
  parses for reward.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional

from openai import (
    AsyncOpenAI, APIError, APITimeoutError, APIConnectionError,
    RateLimitError, BadRequestError,
)

from config import (
    TEACHER_BASE_URL,
    TEACHER_API_KEY,
    TEACHER_MODEL,
    TEACHER_MAX_TOKENS,
    TEACHER_TEMPERATURE,
    TEACHER_TOP_P,
    TEACHER_TIMEOUT_S,
    TEACHER_MAX_CONCURRENT,
    TEACHER_MAX_RETRIES,
    THINK_HARD_CAP_CHARS,
    REASONING_HARD_CAP_CHARS,
)


# ---------------------------------------------------------------------------
# Tag regexes (whitespace-tolerant)
# ---------------------------------------------------------------------------
_THINK_OPEN_RE     = re.compile(r"<think\s*>",      re.IGNORECASE)
_THINK_CLOSE_RE    = re.compile(r"</think\s*>",     re.IGNORECASE)
_REASONING_RE      = re.compile(r"<reasoning\s*>(.*?)</reasoning\s*>", re.DOTALL | re.IGNORECASE)
_COMMAND_RE        = re.compile(r"<command\s*>(.*?)</command\s*>",     re.DOTALL | re.IGNORECASE)

_VALID_CMDS = (
    "diagnose", "adjust_setpoint", "set_fan_speed", "set_rack_load",
    "start_crac", "stop_crac", "start_generator", "stop_generator",
    "set_ups_mode", "refuel_generator", "acknowledge_alarm",
    "check_status", "escalate", "wait", "migrate_workload",
)
_FALLBACK_CMD_LINE_RE = re.compile(
    rf"^\s*({'|'.join(_VALID_CMDS)})(?:\s+\S+){{0,3}}\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Anti-leak filter: catch self-correction phrases that should NOT appear
# in a clean <reasoning> block. Used as a soft warning at parse time.
_SELF_CORRECTION_PHRASES = re.compile(
    r"\b(wait,? actually|let me reconsider|on second thought|actually,? "
    r"i should|hmm,? let me|or rather|but wait|scratch that)\b",
    re.IGNORECASE,
)


@dataclass
class TeacherResult:
    success: bool
    think: str = ""             # raw think content (no surrounding tags)
    reasoning: str = ""         # canonical structured summary
    command: str = ""
    raw: str = ""
    error: str = ""
    leaked_self_correction: bool = False  # signal for downstream filtering


def parse_teacher_output(raw: str) -> TeacherResult:
    """Extract <think> (optional), <reasoning> (required), <command> (required)."""
    if not raw or not raw.strip():
        return TeacherResult(False, error="empty teacher response", raw=raw)

    # ---- Extract <think> content (optional, robust to template-elided opener)
    close = _THINK_CLOSE_RE.search(raw)
    if close:
        pre = raw[:close.start()]
        post = raw[close.end():]
        # Strip a leading <think> tag if the model emitted one
        pre = _THINK_OPEN_RE.sub("", pre, count=1).strip()
        think = pre
    else:
        think = ""
        post = raw  # everything is "answer" if no </think> found

    # ---- Extract <reasoning>...</reasoning> (REQUIRED)
    r_match = _REASONING_RE.search(post)
    if not r_match:
        # Try the entire raw, in case ordering is unusual
        r_match = _REASONING_RE.search(raw)
    if not r_match:
        return TeacherResult(False, error="missing <reasoning> block", raw=raw)
    reasoning = r_match.group(1).strip()

    # ---- Extract <command>...</command> (REQUIRED)
    c_match = _COMMAND_RE.search(post) or _COMMAND_RE.search(raw)
    if c_match:
        command = c_match.group(1).strip().splitlines()[0].strip()
    else:
        # Fallback: scan for a command-like line in the post-think tail
        command = ""
        for m in _FALLBACK_CMD_LINE_RE.finditer(post or raw):
            command = m.group(0).strip()
        if not command:
            return TeacherResult(False, error="no parseable command", raw=raw)

    # ---- Normalize lengths
    think = _normalize_block(think, THINK_HARD_CAP_CHARS)
    reasoning = _normalize_block(reasoning, REASONING_HARD_CAP_CHARS)

    # ---- Soft check for self-correction leak in <reasoning>
    leaked = bool(_SELF_CORRECTION_PHRASES.search(reasoning))

    return TeacherResult(
        success=True,
        think=think,
        reasoning=reasoning,
        command=command,
        raw=raw,
        leaked_self_correction=leaked,
    )


def _normalize_block(text: str, cap: int) -> str:
    """Trim, collapse blank-line runs, hard-cap at sentence boundary."""
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > cap:
        truncated = text[:cap]
        last_punct = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last_punct > cap // 2:
            text = truncated[:last_punct + 1]
        else:
            text = truncated
    return text


# ---------------------------------------------------------------------------
# Async vLLM client
# ---------------------------------------------------------------------------
class TeacherClient:
    """openai.AsyncOpenAI wrapper with concurrency throttling and retries."""

    def __init__(
        self,
        base_url: str = TEACHER_BASE_URL,
        api_key: str = TEACHER_API_KEY,
        model: str = TEACHER_MODEL,
        max_concurrent: int = TEACHER_MAX_CONCURRENT,
    ) -> None:
        self.model = model
        self._sem = asyncio.Semaphore(max_concurrent)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=TEACHER_TIMEOUT_S,
            max_retries=0,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def _chat_once(self, messages: list[dict]) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=TEACHER_MAX_TOKENS,
            temperature=TEACHER_TEMPERATURE,
            top_p=TEACHER_TOP_P,
        )
        return resp.choices[0].message.content or ""

    async def chat(self, messages: list[dict], request_id: str = "") -> str:
        last_err: Optional[Exception] = None
        async with self._sem:
            for attempt in range(TEACHER_MAX_RETRIES + 1):
                try:
                    return await self._chat_once(messages)
                except BadRequestError as e:
                    # 400 from vLLM (usually context-length overflow) —
                    # deterministic, retrying won't help. Fail fast.
                    raise RuntimeError(
                        f"vLLM 400 (rid={request_id}): {e}"
                    ) from e
                except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                    last_err = e
                    backoff = min(2 ** attempt, 8)
                    await asyncio.sleep(backoff)
                except APIError as e:
                    last_err = e
                    if attempt == 0:
                        await asyncio.sleep(2)
                        continue
                    break
        raise RuntimeError(f"vLLM request failed (rid={request_id}): {last_err}")

    async def turn(
        self,
        agent_system_prompt: str,
        history: list[dict],
        user_turn: str,
        request_id: str = "",
    ) -> TeacherResult:
        messages = [{"role": "system", "content": agent_system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_turn})
        try:
            raw = await self.chat(messages, request_id=request_id)
        except RuntimeError as e:
            return TeacherResult(False, error=str(e))
        return parse_teacher_output(raw)
