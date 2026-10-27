"""Embedding back-ends. The PDF/default stores and the web store use different embedders, so
each store must always be queried with the embedder it was built with."""

from __future__ import annotations

from typing import Any

from pcb_assistant.config import Settings


def ollama_embeddings(settings: Settings) -> Any:
    """Embeddings served by Ollama (``nomic-embed-text``) for the PDF and default stores."""

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.embedding_model)


def gpt4all_embeddings() -> Any:
    """Local GPT4All embeddings for the web store."""

    from langchain_community.embeddings import GPT4AllEmbeddings

    return GPT4AllEmbeddings()
