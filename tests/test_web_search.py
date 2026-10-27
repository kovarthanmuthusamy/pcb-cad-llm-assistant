from typing import Any

import pytest

from pcb_assistant.rag import web_search


class FakeLink:
    def __init__(self, href: str = "", text: str = "") -> None:
        self.href, self.text = href, text

    def __getitem__(self, key: str) -> str:
        assert key == "href"
        return self.href

    def get_text(self) -> str:
        return self.text


class FakeSoup:
    def __init__(self, links: list[FakeLink]) -> None:
        self.links = links
        self.selectors: list[str] = []

    def select(self, selector: str) -> list[FakeLink]:
        self.selectors.append(selector)
        return self.links


def test_get_search_dispatches_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_engine(name: str) -> Any:
        def engine(query: str, num_results: int = 5) -> list[str]:
            calls.append((name, query, num_results))
            return [f"https://{name}.example"]

        return engine

    for name in web_search.ENGINES:
        monkeypatch.setitem(web_search.ENGINES, name, fake_engine(name))

    assert web_search.get_search("tdr", "Brave", num_results=3) == ["https://brave.example"]
    assert web_search.get_search("tdr", "duckDuckGo") == ["https://duckduckgo.example"]
    assert web_search.get_search("tdr", "unknown-engine") == ["https://duckduckgo.example"]
    assert calls == [("brave", "tdr", 3), ("duckduckgo", "tdr", 5), ("duckduckgo", "tdr", 5)]


def test_every_documented_engine_is_registered() -> None:
    assert set(web_search.ENGINES) == {
        "google",
        "duckduckgo",
        "brave",
        "bing",
        "yahoo",
        "startpage",
    }


def test_scraping_engines_honour_num_results_and_encode_the_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls: list[str] = []
    soup = FakeSoup([FakeLink(f"https://r{i}.example") for i in range(6)])

    def fake_soup(url: str, headers: dict[str, str] | None = None) -> FakeSoup:
        urls.append(url)
        return soup

    monkeypatch.setattr(web_search, "_soup", fake_soup)

    assert len(web_search.bing_search("a & b #1", num_results=2)) == 2
    assert urls == ["https://www.bing.com/search?q=a+%26+b+%231"]
    assert len(web_search.startpage_search("x", num_results=4)) == 4


def test_duckduckgo_results_are_turned_into_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    soup = FakeSoup([FakeLink(text=" example.com/page "), FakeLink(text="other.org")])
    monkeypatch.setattr(web_search, "_soup", lambda url, headers=None: soup)
    assert web_search.duckduckgo_search("q", 5) == ["https://example.com/page", "https://other.org"]


def test_google_search_reads_links_from_the_api_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def json(self) -> dict[str, Any]:
            return {"items": [{"link": "https://a.example"}, {"link": "https://b.example"}]}

    captured: dict[str, Any] = {}

    def fake_get(url: str, params: dict[str, Any] | None = None, headers: Any = None) -> Response:
        captured.update(url=url, params=params)
        return Response()

    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cse")
    monkeypatch.setattr(web_search, "_get", fake_get)

    assert web_search.google_search("q", 2) == ["https://a.example", "https://b.example"]
    assert captured["params"] == {"key": "key", "cx": "cse", "q": "q", "num": 2}


def test_google_search_without_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def json(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(web_search, "_get", lambda *args, **kwargs: Response())
    assert web_search.google_search("q") == []


def test_duckduckgo_parsing_with_real_html(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bs4")
    html = (
        '<div><a class="result__url" href="//x">  example.com/si-basics </a>'
        '<a class="result__url" href="//y">other.org</a><a class="other">skip</a></div>'
    )

    class Page:
        text = html

    monkeypatch.setattr(web_search, "_get", lambda url, params=None, headers=None: Page())
    assert web_search.duckduckgo_search("si basics", num_results=1) == [
        "https://example.com/si-basics"
    ]
