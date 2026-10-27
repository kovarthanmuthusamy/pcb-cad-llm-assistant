"""Pure text helpers for cleaning model output (no model or network access)."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRAILING_ACTION_JSON = re.compile(
    r'\{\s*"action":\s*".*?"\s*,\s*"action_input":\s*".*?"\s*\}\s*$', re.DOTALL
)


def remove_think_blocks(text: str) -> str:
    """Drop ``<think>...</think>`` reasoning traces (DeepSeek-R1 style) and trim the result."""

    return _THINK_BLOCK.sub("", text).strip()


def final_answer_text(raw: str) -> str:
    """Return what the user should see from the orchestrator's raw output.

    Keeps only the text after the reasoning trace (everything up to the first ``</think>`` is
    dropped) and cuts a trailing ``{"action": ..., "action_input": ...}`` blob the model
    sometimes appends after its answer.
    """

    text = re.split(r"</think>\s*", raw, maxsplit=1)[-1]
    match = _TRAILING_ACTION_JSON.search(text)
    if match:
        text = text[: match.start()].strip()
    return text
