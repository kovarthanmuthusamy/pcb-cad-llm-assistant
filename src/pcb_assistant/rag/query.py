"""Answer a question from a Chroma store built with Ollama embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pcb_assistant.config import Settings, get_settings

from .embeddings import ollama_embeddings

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

TOP_K = 5


def build_prompt(context: str, question: str) -> str:
    from langchain.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return str(template.format(context=context, question=question))


def format_response(response_text: str, sources: list[Any]) -> str:
    return f"Response: {response_text}\nSources: {sources}"


def answer_from_chroma(
    question: str,
    chroma_path: Path | str,
    model: str | None = None,
    settings: Settings | None = None,
) -> str:
    """Retrieve the top chunks for ``question`` and let ``model`` answer from them."""

    from langchain_chroma import Chroma
    from langchain_ollama import OllamaLLM

    settings = settings or get_settings()
    db = Chroma(persist_directory=str(chroma_path), embedding_function=ollama_embeddings(settings))
    results = db.similarity_search_with_score(question, k=TOP_K)

    context = "\n\n---\n\n".join(doc.page_content for doc, _score in results)
    llm = OllamaLLM(model=model or settings.rag_model)
    response_text = llm.invoke(build_prompt(context, question))

    sources = [doc.metadata.get("id") for doc, _score in results]
    return format_response(response_text, sources)
