from pathlib import Path
from types import SimpleNamespace

import pytest

from pcb_assistant.config import Settings
from pcb_assistant.rag import pdf_rag as pdf_module
from pcb_assistant.rag.documents import calculate_chunk_ids
from pcb_assistant.rag.pdf_rag import pdf_rag
from pcb_assistant.rag.query import format_response
from pcb_assistant.rag.storage import clear_files, delete_directory


def chunk(source: str, page: int | None) -> SimpleNamespace:
    return SimpleNamespace(metadata={"source": source, "page": page})


def test_chunk_ids_count_up_within_a_page_and_restart_on_the_next() -> None:
    chunks = [chunk("a.pdf", 0), chunk("a.pdf", 0), chunk("a.pdf", 1), chunk("b.pdf", 1)]
    ids = [c.metadata["id"] for c in calculate_chunk_ids(chunks)]
    assert ids == ["a.pdf:0:0", "a.pdf:0:1", "a.pdf:1:0", "b.pdf:1:0"]


def test_chunk_ids_for_documents_without_pages() -> None:
    ids = [
        c.metadata["id"]
        for c in calculate_chunk_ids([chunk("x.docx", None), chunk("x.docx", None)])
    ]
    assert ids == ["x.docx:None:0", "x.docx:None:1"]


def test_format_response_lists_the_sources() -> None:
    assert format_response("42", ["a:0:0"]) == "Response: 42\nSources: ['a:0:0']"


def test_clear_files_removes_files_but_keeps_folders_and_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "one.pdf").write_text("1")
    (tmp_path / "two.csv").write_text("2")
    (tmp_path / ".keep").write_text("")
    (tmp_path / "sub").mkdir()
    assert clear_files(tmp_path) == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == [".keep", "sub"]


def test_delete_directory(tmp_path: Path) -> None:
    store = tmp_path / "chroma"
    (store / "inner").mkdir(parents=True)
    assert delete_directory(store) is True
    assert not store.exists()
    assert delete_directory(store) is False


def test_pdf_rag_delete_actions_use_the_given_paths(tmp_path: Path) -> None:
    chroma, data = tmp_path / "chroma", tmp_path / "data"
    chroma.mkdir()
    data.mkdir()
    (data / "notes.pdf").write_text("x")
    settings = Settings(rag_home=tmp_path / "unused")

    assert pdf_rag("delete_data", data_path=data, settings=settings) is None
    assert list(data.iterdir()) == []
    assert pdf_rag("delete_chroma", chroma_path=chroma, settings=settings) is None
    assert not chroma.exists()


def test_pdf_rag_defaults_come_from_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(rag_home=tmp_path)
    seen: dict[str, object] = {}

    def fake_write(chroma_path: Path, data_path: Path, settings: Settings) -> int:
        seen.update(chroma=chroma_path, data=data_path)
        return 0

    monkeypatch.setattr(pdf_module, "write_chroma", fake_write)
    pdf_rag("store_data", settings=settings)
    assert seen == {"chroma": tmp_path / "chroma", "data": tmp_path / "data"}


def test_pdf_rag_rejects_unknown_actions() -> None:
    with pytest.raises(ValueError, match="Unknown action"):
        pdf_rag("explode", settings=Settings())  # type: ignore[arg-type]
