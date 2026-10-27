"""Web retrieval: search and index pages, answer from them, or clear the store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pcb_assistant.config import Settings, get_settings

from .storage import delete_directory
from .web_query import answer_from_web_store
from .web_store import store_data

WebAction = Literal["delete_chroma", "store_data", "get_result"]


def web_rag(
    action: WebAction,
    *,
    query: str = "What is eCadstar?",
    question: str = "What is eCadstar?",
    chroma_path: Path | str | None = None,
    num_results: int = 2,
    searcher: str = "duckDuckGo",
    is_link: bool = False,
    model: str | None = None,
    settings: Settings | None = None,
) -> Any:
    """Run one action on the web store. ``store_data`` returns the fetched links and
    ``get_result`` the chain output."""

    settings = settings or get_settings()
    chroma = Path(chroma_path) if chroma_path is not None else settings.chroma_web_dir

    if action == "store_data":
        return store_data(
            query,
            chroma,
            num_results=num_results,
            searcher=searcher,
            is_link=is_link,
            settings=settings,
        )
    if action == "get_result":
        return answer_from_web_store(question, chroma, model, settings)
    if action == "delete_chroma":
        delete_directory(chroma)
        return None
    raise ValueError(f"Unknown action {action!r}")
