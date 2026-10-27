"""Answer a question from the web store with a retrieval-QA chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pcb_assistant.config import Settings, get_settings

from .embeddings import gpt4all_embeddings


def answer_from_web_store(
    question: str,
    chroma_path: Path | str,
    model: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return the chain output, a dict with ``query`` and ``result``."""

    from langchain import hub
    from langchain.chains import RetrievalQA
    from langchain_chroma import Chroma
    from langchain_ollama import ChatOllama

    settings = settings or get_settings()
    vectorstore = Chroma(
        persist_directory=str(chroma_path), embedding_function=gpt4all_embeddings()
    )
    llm = ChatOllama(model=model or settings.rag_model)
    prompt = hub.pull("rlm/rag-prompt")  # needs network access to the LangChain hub
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )
    result: dict[str, Any] = chain.invoke({"query": question})
    return result
