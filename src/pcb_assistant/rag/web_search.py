"""Search back-ends that return result links for a query.

The HTML-scraping engines depend on the search sites' current markup and may need their
selectors updated; ``google`` uses the official Custom Search API (``GOOGLE_API_KEY`` and
``GOOGLE_CSE_ID``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from urllib.parse import quote_plus

REQUEST_TIMEOUT_S = 10
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _get(
    url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> Any:
    import requests

    return requests.get(url, params=params, headers=headers or _HEADERS, timeout=REQUEST_TIMEOUT_S)


def _soup(url: str, headers: dict[str, str] | None = None) -> Any:
    from bs4 import BeautifulSoup

    return BeautifulSoup(_get(url, headers=headers).text, "html.parser")


def google_search(query: str, num_results: int = 5) -> list[str]:
    """Google Custom Search API (set ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID``)."""

    params = {
        "key": os.getenv("GOOGLE_API_KEY"),
        "cx": os.getenv("GOOGLE_CSE_ID"),
        "q": query,
        "num": num_results,
    }
    results = _get("https://www.googleapis.com/customsearch/v1", params=params).json()
    return [item["link"] for item in results.get("items", [])]


def duckduckgo_search(query: str, num_results: int = 5) -> list[str]:
    soup = _soup(f"https://duckduckgo.com/html/?q={quote_plus(query)}")
    links = soup.select("a.result__url")
    return [f"https://{link.get_text().strip()}" for link in links[:num_results]]


def brave_search(query: str, num_results: int = 5) -> list[str]:
    headers = {**_HEADERS, "Accept-Encoding": "gzip, deflate"}
    soup = _soup(f"https://search.brave.com/search?q={quote_plus(query)}", headers)
    return [link["href"] for link in soup.select("a.result-header")[:num_results]]


def bing_search(query: str, num_results: int = 5) -> list[str]:
    soup = _soup(f"https://www.bing.com/search?q={quote_plus(query)}")
    return [link["href"] for link in soup.select("li.b_algo h2 a")[:num_results]]


def yahoo_search(query: str, num_results: int = 5) -> list[str]:
    soup = _soup(f"https://search.yahoo.com/search?p={quote_plus(query)}")
    return [link["href"] for link in soup.select("h3.title a")[:num_results]]


def startpage_search(query: str, num_results: int = 5) -> list[str]:
    soup = _soup(f"https://www.startpage.com/sp/search?q={quote_plus(query)}")
    return [link["href"] for link in soup.select("a.result-link")[:num_results]]


ENGINES: dict[str, Callable[..., list[str]]] = {
    "google": google_search,
    "duckduckgo": duckduckgo_search,
    "brave": brave_search,
    "bing": bing_search,
    "yahoo": yahoo_search,
    "startpage": startpage_search,
}
DEFAULT_ENGINE = "duckduckgo"


def get_search(query: str, searcher: str = "duckDuckGo", num_results: int = 5) -> list[str]:
    """Search with the named engine (case-insensitive); unknown names use DuckDuckGo."""

    engine = ENGINES.get(searcher.lower(), ENGINES[DEFAULT_ENGINE])
    return engine(query, num_results=num_results)
