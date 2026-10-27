"""Command line for the document corpus (used by the web UI):

``python -m pcb_assistant.rag.cli_pdf QUESTION ACTION MODEL [FILES...]``
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from typing import cast

from pcb_assistant.config import Settings, get_settings

from .pdf_rag import PdfAction, pdf_rag


def run(question: str, action: str, model: str, settings: Settings) -> str:
    started = time.perf_counter()
    result = pdf_rag(
        cast("PdfAction", action),
        question=question,
        model=model,
        chroma_path=settings.chroma_pdf_dir,
        data_path=settings.data_upload_dir,
        settings=settings,
    )
    return f"{result},Execution Time: {time.perf_counter() - started:.6f} seconds"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument(
        "action", choices=["get_result", "store_data", "delete_chroma", "delete_data"]
    )
    parser.add_argument("model")
    parser.add_argument("files", nargs="*", help="uploaded files (already saved by the web UI)")
    args = parser.parse_args(argv)
    print(run(args.question, args.action, args.model, get_settings()))  # stdout goes to the UI
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
