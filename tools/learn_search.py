"""Custom search tool backed by the public Microsoft Learn search endpoint."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from services.cache import search_cache

SEARCH_ENDPOINT = "https://learn.microsoft.com/api/search"
USER_AGENT = "MicrosoftLearnKnowledgeGenerator/1.0"


def is_microsoft_learn_url(url: str) -> bool:
    """Return whether a URL is an absolute, secure Microsoft Learn URL."""
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname == "learn.microsoft.com"
    except (TypeError, ValueError):
        return False


class MicrosoftLearnSearchTool:
    """Search Microsoft Learn behind a replaceable, mockable abstraction."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        query = query.strip()
        if not query or max_results < 1:
            return []
        limit = min(max_results, 10)
        cache_key = f"{query.casefold()}:{limit}"
        cached = search_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            results = self._request(query, limit)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

        parsed_results = self._parse_results(results, limit)
        search_cache.set(cache_key, parsed_results)
        return parsed_results

    def _request(self, query: str, limit: int) -> list[dict]:
        params = {"search": query, "locale": "en-us", "$top": limit * 2}
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.client is not None:
            response = self.client.get(SEARCH_ENDPOINT, params=params, headers=headers, timeout=10.0)
        else:
            response = httpx.get(SEARCH_ENDPOINT, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", [])

    @staticmethod
    def _parse_results(results: list[dict], limit: int) -> list[dict]:
        parsed: list[dict] = []
        seen: set[str] = set()
        for index, item in enumerate(results):
            url = str(item.get("url", "")).split("#", 1)[0]
            if not is_microsoft_learn_url(url) or url in seen:
                continue
            seen.add(url)
            description = re.sub(r"<[^>]+>", "", str(item.get("description", ""))).strip()
            parsed.append(
                {
                    "title": str(item.get("title") or url).strip(),
                    "url": url,
                    "description": description,
                    "score": round(1.0 / (index + 1), 3),
                }
            )
            if len(parsed) == limit:
                break
        return parsed


def search_microsoft_learn(query: str, max_results: int = 5) -> list[dict]:
    return MicrosoftLearnSearchTool().search(query, max_results)