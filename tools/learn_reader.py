"""Securely retrieve and extract content from Microsoft Learn pages."""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup, Tag

from services.cache import page_cache
from tools.learn_search import USER_AGENT, is_microsoft_learn_url

MAX_PAGE_CHARACTERS = 45_000


class LearnReaderError(RuntimeError):
    """Raised when an allowed Microsoft Learn page cannot be read."""


class MicrosoftLearnReaderTool:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client

    def get_page(self, url: str) -> dict:
        if not is_microsoft_learn_url(url):
            raise ValueError("Only https://learn.microsoft.com/ URLs are allowed.")
        canonical_url = url.split("#", 1)[0]
        cached = page_cache.get(canonical_url)
        if cached is not None:
            return cached

        try:
            html = self._request(canonical_url)
            page = self._extract(html, canonical_url)
        except httpx.HTTPError as exc:
            raise LearnReaderError(f"Unable to retrieve Microsoft Learn page: {exc}") from exc
        if not page["content"]:
            raise LearnReaderError("The Microsoft Learn page contained no readable documentation.")
        page_cache.set(canonical_url, page)
        return page

    def _request(self, url: str) -> str:
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html"}
        if self.client is not None:
            response = self.client.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        else:
            response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        if not is_microsoft_learn_url(str(response.url)):
            raise LearnReaderError("Microsoft Learn redirected to an external domain.")
        return response.text

    @staticmethod
    def _extract(html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        for element in soup.select(
            "script, style, nav, footer, header, form, iframe, .cookie-banner, "
            ".feedback, .metadata, [aria-label='Page actions']"
        ):
            element.decompose()

        main = soup.select_one("main") or soup.select_one("article") or soup.body
        if main is None:
            return {"title": "Microsoft Learn", "url": url, "content": "", "headings": []}

        title_node = soup.find("h1") or soup.find("title")
        title = title_node.get_text(" ", strip=True) if title_node else "Microsoft Learn"
        headings: list[str] = []
        lines: list[str] = []
        seen: set[str] = set()
        for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"]):
            if isinstance(element, Tag) and element.name == "li" and element.find_parent("li"):
                continue
            text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            if element.name in {"h1", "h2", "h3", "h4"}:
                headings.append(text)
                lines.append(f"\n## {text}")
            elif element.name == "li":
                lines.append(f"- {text}")
            elif element.name == "pre":
                lines.append(f"```\n{text}\n```")
            else:
                lines.append(text)

        content = "\n".join(lines).strip()[:MAX_PAGE_CHARACTERS]
        return {"title": title, "url": url, "content": content, "headings": headings}


def get_microsoft_learn_page(url: str) -> dict:
    return MicrosoftLearnReaderTool().get_page(url)