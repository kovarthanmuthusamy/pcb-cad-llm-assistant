"""RAG entry point: route the question, then answer from the chosen source."""

from __future__ import annotations

import logging
from typing import Any

from pcb_assistant.config import Settings, get_settings

from .pdf_rag import pdf_rag
from .query import answer_from_chroma
from .router import ROUTE_DEFAULT, ROUTE_PDF, ROUTE_WEB, ChatFunction, choose_source
from .web_rag import web_rag

logger = logging.getLogger(__name__)

RagResult = str | dict[str, Any] | None


def answer(
    question: str = "What is eCadstar?",
    *,
    num_results: int | None = None,
    searcher: str | None = None,
    is_link: bool = False,
    model: str | None = None,
    settings: Settings | None = None,
    chat: ChatFunction | None = None,
) -> RagResult:
    """Answer ``question``.

    Returns the answer text for the document sources, the chain output (``{"query", "result"}``)
    for the web source.
    """

    settings = settings or get_settings()
    model = model or settings.rag_model
    route = choose_source(question, settings, chat)
    logger.info("RAG route for %r: %d", question, route)

    if route == ROUTE_WEB:
        web_options: dict[str, Any] = {
            "query": question,
            "question": question,
            "chroma_path": settings.chroma_web_dir,
            "num_results": num_results or settings.rag_num_results,
            "searcher": searcher or settings.rag_search_engine,
            "is_link": is_link,
            "model": model,
            "settings": settings,
        }
        web_rag("store_data", **web_options)
        return web_rag("get_result", **web_options)
    if route == ROUTE_PDF:
        pdf_rag(
            "store_data",
            chroma_path=settings.chroma_pdf_dir,
            data_path=settings.data_pdf_dir,
            settings=settings,
        )
        return pdf_rag(
            "get_result",
            question=question,
            model=model,
            chroma_path=settings.chroma_pdf_dir,
            settings=settings,
        )
    if route == ROUTE_DEFAULT:
        return answer_from_chroma(question, settings.chroma_default_dir, model, settings)
    return None
