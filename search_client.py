"""
SearXNG search client.

Provides web search with result sanitization and intelligent
auto-search triggering.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Dict, List, Optional

import httpx

from config import get_config

logger = logging.getLogger(__name__)

# Keywords that strongly suggest a web search is useful.
_SEARCH_TRIGGERS = [
    "latest", "today", "news", "who is", "what is", "when is",
    "release date", "current version", "documentation", "doc",
    "price", "weather", "score", "rank", "top", "review",
    "announcement", "update", "version", "changelog",
    "is it", "does", "can", "where is", "how much",
    "meaning of", "definition", "wiki", "wikipedia",
    "stock", "market", "election", "tournament",
]

# Keywords that usually mean the model can handle it without searching.
_SEARCH_SKIP = [
    "write a", "rewrite", "summarize", "translate", "brainstorm",
    "generate", "create a", "invent", "compose", "draft",
    "hello", "hi", "hey", "good morning", "good evening",
    "thank", "thanks", "welcome", "bye",
    "explain how to code", "code a", "implement", "debug",
    "refactor", "optimize", "sort", "reverse",
    "tell a story", "poem", "sonnet", "haiku",
    "joke", "riddle", "metaphor",
    "convert", "format", "parse",
]


class SearchClient:
    """
    Async client for a SearXNG instance.

    Creates a single ``httpx.AsyncClient`` and reuses it across searches.
    """

    def __init__(self) -> None:
        config = get_config()
        self._base_url = config.searxng_base_url.rstrip("/")
        self._timeout = config.searxng_timeout
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(
            "SearchClient initialized: url=%s, timeout=%.1fs",
            self._base_url,
            self._timeout,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize the underlying HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=5.0),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
            )
        return self._client

    async def search(
        self, query: str, max_results: int = 5
    ) -> List[dict]:
        """
        Perform a web search and return a list of result dicts.

        Each dict contains ``title``, ``url``, and ``snippet``.
        """
        client = await self._get_client()
        try:
            response = await client.get(
                "/search",
                params={
                    "q": query,
                    "format": "json",
                    "num_results": max_results,
                },
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            sanitized = self.sanitize_results(results)
            logger.info(
                "Search for '%s' returned %d results (sanitized to %d).",
                query,
                len(results),
                len(sanitized),
            )
            return sanitized
        except httpx.TimeoutException:
            logger.warning("Search timed out for query: %s", query)
            return []
        except httpx.HTTPStatusError as exc:
            logger.error("Search HTTP error %d: %s", exc.response.status_code, exc.response.text)
            return []
        except Exception as exc:
            logger.error("Search error for '%s': %s", query, exc)
            return []

    def sanitize_results(self, results: List[dict]) -> List[dict]:
        """
        Sanitize search result dicts.

        Strips HTML, escapes entities, and ensures each result has
        ``title``, ``url``, and ``snippet`` keys.
        """
        sanitized: List[dict] = []
        for r in results:
            title = html.unescape(str(r.get("title", "") or ""))
            title = re.sub(r"<[^>]+>", "", title)
            url = str(r.get("url", "") or "")
            snippet = html.unescape(str(r.get("snippet", "") or ""))
            snippet = re.sub(r"<[^>]+>", "", snippet)
            snippet = re.sub(r"\.{3}", "...", snippet)  # Normalize ellipses
            if title or url or snippet:
                sanitized.append({
                    "title": title.strip(),
                    "url": url.strip(),
                    "snippet": snippet.strip(),
                })
        return sanitized

    def should_search(self, query: str) -> bool:
        """
        Heuristic: should the bot perform a web search for *query*?

        Returns ``True`` if the query contains search-triggering keywords
        and does not contain skip keywords.
        """
        q = query.lower().strip()

        # Skip very short queries.
        if len(q) < 4:
            return False

        # Check for skip keywords first.
        for skip in _SEARCH_SKIP:
            if skip in q:
                logger.debug("Skipping search for '%s' (matched skip: '%s').", q, skip)
                return False

        # Check for trigger keywords.
        for trigger in _SEARCH_TRIGGERS:
            if trigger in q:
                logger.debug("Triggering search for '%s' (matched: '%s').", q, trigger)
                return True

        # If the query ends with a question mark, lean toward searching.
        if q.endswith("?"):
            return True

        return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("SearchClient closed.")