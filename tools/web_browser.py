from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests
from bs4 import BeautifulSoup


@dataclass
class WebPage:
    url: str
    title: str = ""
    text: str = ""
    status_code: int | None = None
    error: str | None = None


class WebBrowserTool:
    """Tiny generic web browser for future dataset/paper reading nodes."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def fetch_text(self, url: str, max_chars: int = 12000, headers: dict[str, str] | None = None) -> WebPage:
        try:
            response = requests.get(url, timeout=self.timeout, headers=headers or {})
            status = response.status_code
            if status >= 400:
                return WebPage(url=url, status_code=status, error=response.text[:500])
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            return WebPage(url=url, title=title, text=text[:max_chars], status_code=status)
        except Exception as exc:
            return WebPage(url=url, error=f"{type(exc).__name__}: {exc}")
