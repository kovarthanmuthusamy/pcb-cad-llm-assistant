"""PDF/office-document retrieval: index a folder, query it, or clear it."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pcb_assistant.config import Settings, get_settings

from .documents import write_chroma
from .query import answer_from_chroma
from .storage import clear_files, delete_directory

PdfAction = Literal["delete_chroma", "delete_data", "store_data", "get_result"]


def pdf_rag(
    action: PdfAction,
    *,
    question: str = "How to external program in an unlocked state?",
    model: str | None = None,
    chroma_path: Path | str | None = None,
    data_path: Path | str | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Run one action on the document store; ``get_result`` returns the answer text."""

    settings = settings or get_settings()
    chroma = Path(chroma_path) if chroma_path is not None else settings.chroma_pdf_dir
    data = Path(data_path) if data_path is not None else settings.data_upload_dir

    if action == "delete_chroma":
        delete_directory(chroma)
    elif action == "delete_data":
        clear_files(data)
    elif action == "store_data":
        write_chroma(chroma, data, settings)
    elif action == "get_result":
        return answer_from_chroma(question, chroma, model, settings)
    else:
        raise ValueError(f"Unknown action {action!r}")
    return None
