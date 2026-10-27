import logging
from pathlib import Path

import pytest

from pcb_assistant.config import Settings
from pcb_assistant.rag.router import (
    ROUTE_DEFAULT,
    ROUTE_PDF,
    ROUTE_WEB,
    build_router_prompt,
    choose_source,
    list_corpus,
    parse_route,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", ROUTE_DEFAULT),
        ("2", ROUTE_PDF),
        ("3", ROUTE_WEB),
        (" 2\n", ROUTE_PDF),
        ("[2]", ROUTE_PDF),
        ("**1**", ROUTE_DEFAULT),
        ("2.", ROUTE_PDF),
        ("<think>Set A looks relevant, so 1... no, 2.</think>\n2", ROUTE_PDF),
    ],
)
def test_parse_route_accepts_the_expected_replies(raw: str, expected: int) -> None:
    assert parse_route(raw) == expected


@pytest.mark.parametrize("raw", ["", "4", "12", "I think set B", "0", "<think>2</think>"])
def test_parse_route_falls_back_to_web(raw: str) -> None:
    assert parse_route(raw) == ROUTE_WEB


def test_the_router_can_never_select_a_delete_action() -> None:
    # An older version mapped reply "4" to deleting the stores; it is now just a fallback.
    assert parse_route("4") == ROUTE_WEB


def test_list_corpus_returns_names_or_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "si_basics.pdf").write_text("x")
    assert list_corpus(tmp_path) == ["si_basics.pdf"]
    with caplog.at_level(logging.WARNING):
        assert list_corpus(tmp_path / "missing") is None
    assert "not found" in caplog.text


def test_router_prompt_contains_the_corpora_and_the_question() -> None:
    prompt = build_router_prompt("What is TDR?", ["a.pdf"], None)
    assert "Set A:\"['a.pdf']\"" in prompt
    assert 'Set B: "None"' in prompt
    assert 'Question: "What is TDR?"' in prompt


def test_choose_source_asks_the_router_model(tmp_path: Path) -> None:
    settings = Settings(rag_home=tmp_path, rag_router_model="router-model")
    (tmp_path / "data_pdf").mkdir()
    (tmp_path / "data_pdf" / "manual.pdf").write_text("x")
    calls: list[tuple[str, str]] = []

    def fake_chat(model: str, prompt: str) -> str:
        calls.append((model, prompt))
        return "<think>manual.pdf fits</think>2"

    assert choose_source("How do I unlock the file?", settings, fake_chat) == ROUTE_PDF
    assert calls[0][0] == "router-model"
    assert "manual.pdf" in calls[0][1]
