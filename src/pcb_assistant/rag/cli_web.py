"""Command line for the web store (used by the web UI):

``python -m pcb_assistant.rag.cli_web ACTION QUERY IS_LINK QUESTION MODEL SEARCH NUM_RESULTS``
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from typing import cast

from pcb_assistant.config import Settings, get_settings

from .web_rag import WebAction, web_rag


def run(
    action: str,
    query: str,
    is_link: bool,
    question: str,
    model: str,
    searcher: str,
    num_results: int,
    settings: Settings,
) -> str:
    started = time.perf_counter()
    result = web_rag(
        cast("WebAction", action),
        query=query,
        question=question,
        chroma_path=settings.chroma_web_dir,
        num_results=num_results,
        searcher=searcher,
        is_link=is_link,
        model=model,
        settings=settings,
    )
    return f"{result},Execution Time: {time.perf_counter() - started:.6f} seconds"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["get_result", "store_data", "delete_chroma"])
    parser.add_argument("query")
    parser.add_argument("is_link", choices=["true", "false"])
    parser.add_argument("question")
    parser.add_argument("model")
    parser.add_argument("search")
    parser.add_argument("num_results", type=int)
    args = parser.parse_args(argv)
    output = run(
        args.action,
        args.query,
        args.is_link == "true",
        args.question,
        args.model,
        args.search,
        args.num_results,
        get_settings(),
    )
    print(output)  # stdout goes to the UI
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
