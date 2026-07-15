"""
Search client — communicates with a local SearXNG instance.

Uses a single reusable httpx.AsyncClient.  Implements smart search
triggers, result sanitization, and HTML escaping.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Dict, List, Optional

import httpx

from config import Config

logger = logging.getLogger(__name__)

# Keywords that suggest a web search is useful.
_SEARCH_TRIGGERS = [
    "latest", "today", "news", "who is", "what is", "when is",
    "release date", "current version", "documentation", "doc",
    "price", "weather", "score", "rank", "top", "new",
    "recent", "update", "version", "review", "compare",
    "where is", "how much", "cost", "stock", "market",
]

# Keywords that suggest a web search is *not* needed.
_SEARCH_SKIP = [
    "write a", "write me", "create a", "generate a",
    "summarize", "translate", "rewrite", "paraphrase",
    "brainstorm", "explain", "define", "describe",
    "hello", "hi ", "hey", "thanks", "thank you",
    "good morning", "good evening", "good night",
    "what if", "imagine", "story", "poem", "essay",
    "code", "function", "class", "loop", "variable",
    "sort", "filter", "map", "reduce", "list",
]


class SearchClient:
    """
    Async client for a local SearXNG instance.

    Responsibilities:
    - Maintain one reusable httpx.AsyncClient.
    - Decide whether a query warrants a web search.
    - Fetch and sanitize search results.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.searxng_base_url,
            timeout=httpx.Timeout(config.searxng_timeout),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
        )
        logger.info("SearchClient initialized (base_url=%s).", config.searxng_base_url)

    def should_search(self, query: str) -> bool:
        """
        Heuristic: should this query trigger a web search?

        Returns ``True`` if the query contains search-triggering keywords
        and does not contain skip keywords.
        """
        q = query.lower().strip()

        # Skip very short queries.
        if len(q) < 4:
            return False

        # Skip if a "skip" keyword matches.
        for skip in _SEARCH_SKIP:
            if skip in q:
                return False

        # Trigger if a "search" keyword matches.
        for trigger in _SEARCH_TRIGGERS:
            if trigger in q:
                return True

        # If the query looks like a question, search.
        if q.endswith("?") or q.startswith("what") or q.startswith("who"):
            return True

        return False

    async def search(self, query: str) -> List[Dict[str, str]]:
        """
        Search SearXNG and return a list of sanitized result dicts.

        Each dict has keys: ``title``, ``url``, ``snippet``.

        Returns an empty list on timeout or network error.
        """
        try:
            resp = await self._client.get(
                "/search",
                params={
                    "q": query,
                    "format": "json",
                    "num_results": 5,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            sanitized = self._sanitize_results(results)
            logger.info("Search for '%s' returned %d results.", query, len(sanitized))
            return sanitized
        except httpx.TimeoutException:
            logger.warning("Search timed out for query: %s", query)
            return []
        except httpx.RequestError as exc:
            logger.warning("Search request error: %s", exc)
            return []
        except Exception as exc:
            logger.error("Unexpected search error: %s", exc)
            return []

    def _sanitize_results(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Sanitize search results: strip HTML, escape special chars,
        and limit snippet length.
        """
        sanitized: List[Dict[str, str]] = []
        for r in results:
            title = html.escape(str(r.get("title", "") or ""), quote=False)
            url = html.escape(str(r.get("url", "") or ""), quote=False)
            snippet = html.escape(str(r.get("content", "") or ""), quote=False)
            # Strip HTML tags.
            snippet = re.sub(r"<[^>]+>", "", snippet)
            # Truncate snippet to ~200 chars.
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            if title and url:
                sanitized.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })
        return sanitized

    async def health_check(self) -> bool:
        """
        Ping the SearXNG ``/search?q=test`` endpoint.

        Returns ``True`` if the server responds with a 200 OK.
        """
        try:
            resp = await self._client.get("/search", params={"q": "test", "format": "json"})
            resp.raise_for_status()
            return True
        except httpx.RequestError as exc:
            logger.debug("SearXNG health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
        logger.info("SearchClient closed.")
