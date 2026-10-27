from pathlib import Path
from typing import Any

import pytest

from pcb_assistant.config import Settings
from pcb_assistant.rag import pipeline
from pcb_assistant.rag.router import ROUTE_DEFAULT, ROUTE_PDF, ROUTE_WEB


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, Any]]]:
    recorded: list[tuple[str, str, dict[str, Any]]] = []

    def record(source: str, result: Any) -> Any:
        def fake(action_or_question: str, *args: Any, **kwargs: Any) -> Any:
            recorded.append((source, action_or_question, kwargs))
            if args:
                recorded[-1][2]["args"] = args
            return result

        return fake

    monkeypatch.setattr(pipeline, "web_rag", record("web", {"query": "q", "result": "web answer"}))
    monkeypatch.setattr(pipeline, "pdf_rag", record("pdf", "pdf answer"))
    monkeypatch.setattr(pipeline, "answer_from_chroma", record("default", "default answer"))
    return recorded


def route_to(monkeypatch: pytest.MonkeyPatch, route: int) -> None:
    monkeypatch.setattr(pipeline, "choose_source", lambda question, settings, chat: route)


def test_web_route_indexes_fresh_results_then_answers(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str, dict[str, Any]]], tmp_path: Path
) -> None:
    route_to(monkeypatch, ROUTE_WEB)
    settings = Settings(rag_home=tmp_path, rag_num_results=4, rag_search_engine="brave")

    result = pipeline.answer("What is TDR?", settings=settings)

    assert result == {"query": "q", "result": "web answer"}
    assert [(source, action) for source, action, _ in calls] == [
        ("web", "store_data"),
        ("web", "get_result"),
    ]
    options = calls[0][2]
    assert options["chroma_path"] == tmp_path / "chroma_web"
    assert options["num_results"] == 4
    assert options["searcher"] == "brave"
    assert options["model"] == "deepseek-r1:32b"


def test_pdf_route_indexes_the_pdf_folder_then_answers(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str, dict[str, Any]]], tmp_path: Path
) -> None:
    route_to(monkeypatch, ROUTE_PDF)
    settings = Settings(rag_home=tmp_path)

    assert pipeline.answer("How do I unlock the file?", settings=settings) == "pdf answer"
    assert [(source, action) for source, action, _ in calls] == [
        ("pdf", "store_data"),
        ("pdf", "get_result"),
    ]
    assert calls[0][2]["data_path"] == tmp_path / "data_pdf"
    assert calls[1][2]["question"] == "How do I unlock the file?"


def test_default_route_queries_the_default_store_without_indexing(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str, dict[str, Any]]]
) -> None:
    route_to(monkeypatch, ROUTE_DEFAULT)
    assert pipeline.answer("What is eCadstar?", settings=Settings()) == "default answer"
    assert [source for source, _, _ in calls] == ["default"]


def test_unknown_route_returns_nothing(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str, dict[str, Any]]]
) -> None:
    route_to(monkeypatch, 99)
    assert pipeline.answer("anything", settings=Settings()) is None
    assert calls == []


def test_no_route_deletes_anything(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str, dict[str, Any]]]
) -> None:
    for route in (ROUTE_DEFAULT, ROUTE_PDF, ROUTE_WEB, 4):
        route_to(monkeypatch, route)
        pipeline.answer("anything", settings=Settings())
    assert not any(action.startswith("delete") for _, action, _ in calls)
