"""Pick the knowledge source for a question by asking a small LLM.

The router only sees the *file names* of the two local corpora, so it is a cheap first guess,
not a relevance judgement. Anything it cannot place falls back to the web.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from pcb_assistant.config import Settings, get_settings
from pcb_assistant.text import remove_think_blocks

logger = logging.getLogger(__name__)

ROUTE_DEFAULT = 1  # default corpus (data_def)
ROUTE_PDF = 2  # PDF corpus (data_pdf)
ROUTE_WEB = 3  # live web search

ROUTER_PROMPT = """You have two sets of document names, each containing various topics. \
Your task is to determine if a given question can be answered solely based on the document \
names. Analyze only the document names and do not assume any additional content.

Here are the document sets:

Set A:"{set_a}"
Set B: "{set_b}"

Instructions for response:

If Set A likely contains the answer, return 1.
If Set B likely contains the answer, return 2.
If neither contains the answer, return 3.
If both sets contain relevant information, choose the set that is more specific and detailed.
If you are unsure whether the question can be answered using these document names, return 3
Now, analyze the following question and determine the appropriate response.

Question: "{question}"

Final Output: [1/2/3] (Respond only with the number.)"""

ChatFunction = Callable[[str, str], str]  # (model, prompt) -> reply text


def list_corpus(folder: Path | str) -> list[str] | None:
    """File names in ``folder``, or ``None`` when the folder does not exist."""

    path = Path(folder)
    if not path.is_dir():
        logger.warning("Corpus folder %s not found; this source is unavailable", path)
        return None
    return [entry.name for entry in path.iterdir()]


def build_router_prompt(question: str, set_a: list[str] | None, set_b: list[str] | None) -> str:
    return ROUTER_PROMPT.format(set_a=set_a, set_b=set_b, question=question)


def parse_route(raw: str, default: int = ROUTE_WEB) -> int:
    """Read the router's reply (1, 2 or 3). Anything else maps to ``default``.

    Reasoning traces are removed first, and light decoration such as ``[2]`` or ``**2**`` is
    accepted.
    """

    match = re.fullmatch(r"\W*([123])\W*", remove_think_blocks(raw))
    return int(match.group(1)) if match else default


def _ollama_chat(model: str, prompt: str) -> str:
    import ollama

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return str(response["message"]["content"])


def choose_source(
    question: str,
    settings: Settings | None = None,
    chat: ChatFunction | None = None,
) -> int:
    """Return ``ROUTE_DEFAULT``, ``ROUTE_PDF`` or ``ROUTE_WEB`` for ``question``."""

    settings = settings or get_settings()
    prompt = build_router_prompt(
        question,
        list_corpus(settings.data_default_dir),
        list_corpus(settings.data_pdf_dir),
    )
    reply = (chat or _ollama_chat)(settings.rag_router_model, prompt)
    return parse_route(reply)
