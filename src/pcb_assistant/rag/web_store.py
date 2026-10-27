"""Fetch web pages for a query (or given links), clean them and index them into Chroma."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from pcb_assistant.config import Settings, get_settings

from .embeddings import gpt4all_embeddings
from .web_search import get_search

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace and drop special characters."""

    from bs4 import BeautifulSoup

    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s,.!?]", "", text)
    return text.strip()


def store_data(
    query: str,
    chroma_path: Path | str,
    *,
    num_results: int = 2,
    searcher: str = "duckDuckGo",
    is_link: bool = False,
    settings: Settings | None = None,
) -> str | list[str]:
    """Index the pages found for ``query`` (or ``query`` itself when it is a link).

    Returns the links that were fetched.
    """

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import WebBaseLoader

    settings = settings or get_settings()
    os.environ.setdefault("USER_AGENT", settings.user_agent)  # read by WebBaseLoader

    links = query if is_link else get_search(query, searcher, num_results=num_results)

    loader = WebBaseLoader(links, continue_on_failure=True)
    loader.requests_per_second = 1
    pages = loader.aload()

    cleaned = [clean_text(page.page_content) for page in pages]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.create_documents(cleaned)

    Chroma.from_documents(
        documents=chunks, embedding=gpt4all_embeddings(), persist_directory=str(chroma_path)
    )
    store = Chroma(persist_directory=str(chroma_path), embedding_function=gpt4all_embeddings())
    logger.info("Number of documents stored: %d", store._collection.count())
    return links
