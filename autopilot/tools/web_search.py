from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from autopilot.config import Settings


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = ""


HF_DATASET_URL_RE = re.compile(r"https?://(?:www\.)?huggingface\.co/datasets/([^\s?#]+)", re.I)


def extract_hf_dataset_id(url: str) -> str | None:
    """Extract a Hugging Face dataset repo id from a URL.

    Handles normal dataset pages and viewer/file paths:
      https://huggingface.co/datasets/org/name
      https://huggingface.co/datasets/org/name/viewer/default/train
    """
    if not url:
        return None
    parsed = urlparse(url)

    # Search engines sometimes wrap the real URL in a redirect parameter.
    if "huggingface.co" not in parsed.netloc.lower():
        qs = parse_qs(parsed.query)
        for values in qs.values():
            for value in values:
                found = extract_hf_dataset_id(unquote(value))
                if found:
                    return found
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2 or parts[0] != "datasets":
        return None
    # Most HF dataset repos are owner/name. Some legacy repos are bare names.
    if len(parts) >= 3:
        return f"{parts[1]}/{parts[2]}"
    return parts[1]


def extract_hf_dataset_ids(results: Iterable[WebSearchResult]) -> list[str]:
    ids: list[str] = []
    for result in results:
        dataset_id = extract_hf_dataset_id(result.url)
        if dataset_id and dataset_id not in ids:
            ids.append(dataset_id)
    return ids


class WebSearchTool:
    """Provider-agnostic web search scaffold.

    Supported providers:
    - duckduckgo: no API key, best-effort HTML search;
    - serper: Google Serper API;
    - brave: Brave Search API;
    - tavily: Tavily Search API;
    - bocha: Bocha AI Web Search API, useful for domestic/RMB workflows;
    - none/disabled/off: skip generic web search and rely on HF Hub search.

    This tool is intentionally generic so later nodes can reuse it for paper
    search/literature reading instead of being coupled to Hugging Face.
    """

    def __init__(self, settings: Settings, timeout: float = 20.0) -> None:
        self.settings = settings
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; llm-training-autopilot/0.3; +https://example.local)"}
        )

    def search(self, query: str, limit: int = 10) -> list[WebSearchResult]:
        provider = (self.settings.web_search_provider or "duckduckgo").lower().strip()
        if provider in {"none", "disabled", "off", "hf_only", "hf-only"}:
            return []
        if provider == "duckduckgo":
            return self._duckduckgo(query, limit)
        if provider == "serper" and self.settings.serper_api_key:
            return self._serper(query, limit)
        if provider == "brave" and self.settings.brave_api_key:
            return self._brave(query, limit)
        if provider == "tavily" and self.settings.tavily_api_key:
            return self._tavily(query, limit)
        if provider == "bocha" and self.settings.bocha_api_key:
            return self._bocha(query, limit)
        return []

    def _duckduckgo(self, query: str, limit: int) -> list[WebSearchResult]:
        response = self.session.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[WebSearchResult] = []
        for result in soup.select("div.result"):
            link = result.select_one("a.result__a")
            if link is None:
                continue
            href = link.get("href") or ""
            parsed = urlparse(href)
            if "duckduckgo.com" in parsed.netloc:
                qs = parse_qs(parsed.query)
                if qs.get("uddg"):
                    href = unquote(qs["uddg"][0])
            snippet_el = result.select_one("a.result__snippet") or result.select_one("div.result__snippet")
            results.append(
                WebSearchResult(
                    title=link.get_text(" ", strip=True),
                    url=href,
                    snippet=snippet_el.get_text(" ", strip=True) if snippet_el else "",
                    source="duckduckgo",
                )
            )
            if len(results) >= limit:
                break
        return results

    def _serper(self, query: str, limit: int) -> list[WebSearchResult]:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.settings.serper_api_key or "", "Content-Type": "application/json"},
            json={"q": query, "num": limit},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in (data.get("organic") or [])[:limit]:
            results.append(WebSearchResult(title=item.get("title", ""), url=item.get("link", ""), snippet=item.get("snippet", ""), source="serper"))
        return results

    def _brave(self, query: str, limit: int) -> list[WebSearchResult]:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.settings.brave_api_key or ""},
            params={"q": query, "count": min(limit, 20)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in ((data.get("web") or {}).get("results") or [])[:limit]:
            results.append(WebSearchResult(title=item.get("title", ""), url=item.get("url", ""), snippet=item.get("description", ""), source="brave"))
        return results

    def _tavily(self, query: str, limit: int) -> list[WebSearchResult]:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": self.settings.tavily_api_key, "query": query, "max_results": limit},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in (data.get("results") or [])[:limit]:
            results.append(WebSearchResult(title=item.get("title", ""), url=item.get("url", ""), snippet=item.get("content", ""), source="tavily"))
        return results


# Keep Bocha as a method on WebSearchTool without disturbing earlier providers.
def _web_search_tool_bocha(self: WebSearchTool, query: str, limit: int) -> list[WebSearchResult]:
    response = requests.post(
        self.settings.bocha_endpoint or "https://api.bochaai.com/v1/web-search",
        headers={"Authorization": f"Bearer {self.settings.bocha_api_key or ''}", "Content-Type": "application/json"},
        json={"query": query, "freshness": "noLimit", "summary": True, "count": max(1, min(limit, 50))},
        timeout=self.timeout,
    )
    response.raise_for_status()
    data = response.json()
    values = ((data.get("webPages") or {}).get("value") or []) if isinstance(data, dict) else []
    results: list[WebSearchResult] = []
    for item in values[:limit]:
        if not isinstance(item, dict):
            continue
        results.append(
            WebSearchResult(
                title=item.get("name") or item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("snippet") or item.get("summary") or item.get("description") or "",
                source="bocha",
            )
        )
    return results


WebSearchTool._bocha = _web_search_tool_bocha  # type: ignore[attr-defined]
