"""Load, split and index local documents (PDF, CSV, DOCX, XLSX, PPTX) into a Chroma store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pcb_assistant.config import Settings, get_settings

from .embeddings import ollama_embeddings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _loader_for(path: Path) -> Any | None:
    """Return the LangChain loader for ``path`` or ``None`` for unsupported file types."""

    suffix = path.suffix.lower()
    file = str(path)
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(file)
    if suffix == ".csv":
        from langchain_community.document_loaders.csv_loader import CSVLoader

        return CSVLoader(file)
    if suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(file)
    if suffix == ".xlsx":
        from langchain_community.document_loaders import UnstructuredExcelLoader

        return UnstructuredExcelLoader(file_path=file, mode="elements")
    if suffix == ".pptx":
        from langchain_community.document_loaders import UnstructuredPowerPointLoader

        return UnstructuredPowerPointLoader(file)
    return None


def load_documents(data_path: Path | str) -> list[Any]:
    """Load every supported file directly inside ``data_path``."""

    documents: list[Any] = []
    for path in sorted(Path(data_path).iterdir()):
        if not path.is_file():
            continue
        loader = _loader_for(path)
        if loader is None:
            logger.debug("Skipping unsupported file %s", path)
            continue
        logger.info("Loading %s", path)
        documents.extend(loader.load())
    return documents


def split_documents(documents: list[Any]) -> list[Any]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return list(splitter.split_documents(documents))


def calculate_chunk_ids(chunks: list[Any]) -> list[Any]:
    """Give every chunk a stable ``source:page:index`` id (stored in ``metadata["id"]``)."""

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        current_page_id = f"{chunk.metadata.get('source')}:{chunk.metadata.get('page')}"
        # Same page as the previous chunk -> next index; a new page restarts at 0.
        current_chunk_index = current_chunk_index + 1 if current_page_id == last_page_id else 0
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def add_to_chroma(chunks: list[Any], chroma_path: Path | str, settings: Settings) -> int:
    """Add the chunks that are not stored yet. Returns how many were added."""

    from langchain_chroma import Chroma

    db = Chroma(persist_directory=str(chroma_path), embedding_function=ollama_embeddings(settings))
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_ids = set(db.get(include=[])["ids"])  # ids are always returned
    logger.info("Number of existing documents in DB: %d", len(existing_ids))

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    if not new_chunks:
        logger.info("No new documents to add")
        return 0

    logger.info("Adding new documents: %d", len(new_chunks))
    db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
    return len(new_chunks)


def write_chroma(
    chroma_path: Path | str, data_path: Path | str, settings: Settings | None = None
) -> int:
    """Create or update the Chroma store at ``chroma_path`` from the files in ``data_path``."""

    settings = settings or get_settings()
    chunks = split_documents(load_documents(data_path))
    return add_to_chroma(chunks, chroma_path, settings)
