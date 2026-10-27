import pytest

pytest.importorskip("bs4")

from pcb_assistant.rag.web_store import clean_text  # noqa: E402


def test_clean_text_strips_markup_whitespace_and_special_characters() -> None:
    html = "<h1>TDR</h1>\n<p>Reflections,   at <b>50 ohm</b>!  © 2024 {x}</p>"
    assert clean_text(html) == "TDR Reflections, at 50 ohm! 2024 x"
