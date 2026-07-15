"""
Tool Registry Architecture

Provides:
- Tool: Abstract interface for all bot tools
- ToolMetadata: Describes a tool's identity and capabilities
- ToolResult: Standardized return type for tool execution
- ToolRegistry: Central registration and dispatch for tools
- SearchTool: SearXNG search implemented as a Tool
"""

from __future__ import annotations

import abc
import asyncio
import html
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolMetadata:
    """Immutable metadata describing a tool."""
    name: str
    description: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Standardized result returned by tool.execute()."""
    success: bool
    data: Any
    message: str = ""
    raw: Any = None

    @classmethod
    def ok(cls, data: Any, message: str = "", raw: Any = None) -> ToolResult:
        return cls(success=True, data=data, message=message, raw=raw)

    @classmethod
    def fail(cls, data: Any, message: str = "", raw: Any = None) -> ToolResult:
        return cls(success=False, data=data, message=message, raw=raw)


# ---------------------------------------------------------------------------
# Abstract Tool interface
# ---------------------------------------------------------------------------

class Tool(abc.ABC):
    """Abstract interface for all bot tools.

    Subclasses implement `execute()` and optionally `health_check()`.
    Registration is handled by `ToolRegistry`.
    """

    @property
    @abc.abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return immutable metadata for this tool."""

    @abc.abstractmethod
    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given query and optional parameters."""

    async def health_check(self) -> Dict[str, Any]:
        """Optional health check. Returns {'name': ..., 'healthy': ..., 'details': ...}."""
        return {
            "name": self.metadata.name,
            "healthy": True,
            "details": "OK",
        }

    async def close(self) -> None:
        """Close any resources held by the tool (e.g., HTTP clients)."""
        pass


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for tool registration and dispatch.

    Usage:
        registry = ToolRegistry()
        registry.register(SearchTool(config))
        result = await registry.execute("search", "latest Python release")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError if the name is already taken."""
        name = tool.metadata.name
        if name in self._tools:
            raise ValueError(
                f"Tool '{name}' already registered (existing: {type(self._tools[name]).__name__})"
            )
        self._tools[name] = tool
        logger.info("Registered tool: %s (%s)", name, tool.metadata.description)

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolMetadata]:
        """Return metadata for all registered tools."""
        return [t.metadata for t in self._tools.values()]

    def has(self, name: str) -> bool:
        """Check if a tool with the given name is registered."""
        return name in self._tools

    async def execute(self, tool_name: str, query: str, **kwargs: Any) -> ToolResult:
        """Dispatch a query to the named tool."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.fail(
                data=None,
                message=f"Unknown tool: '{tool_name}' (available: {', '.join(self._tools.keys()) or 'none'})",
            )
        try:
            result = await tool.execute(query, **kwargs)
            return result
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", tool_name)
            return ToolResult.fail(
                data=None,
                message=f"Tool '{tool_name}' error: {exc}",
            )

    async def health_checks(self) -> List[Dict[str, Any]]:
        """Run health checks on all registered tools."""
        tasks = [t.health_check() for t in self._tools.values()]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        """Close all registered tools."""
        for tool in self._tools.values():
            await tool.close()
        logger.info("ToolRegistry closed (%d tools).", len(self._tools))


# ---------------------------------------------------------------------------
# SearchTool — SearXNG search as a Tool
# ---------------------------------------------------------------------------

_SEARCH_TRIGGERS_POSITIVE = [
    "latest", "today", "news", "who is", "what is", "release date",
    "current version", "documentation", "doc", "price", "stock",
    "weather", "score", "result", "results", "review", "reviews",
    "is it", "does it", "can you find", "find", "search", "compare",
    "vs", "versus", "announcement", "announced", "just released",
    "new", "recent", "recently", "current", "live", "status",
    "where is", "when is", "how much", "how many",
]

_SEARCH_TRIGGERS_NEGATIVE = [
    "write a", "write me", "create a", "generate a", "generate me",
    "summarize", "summary", "translate", "translation", "rewrite",
    "rephrase", "paraphrase", "brainstorm", "brainstorming",
    "explain how to", "how do i", "how does", "tutorial",
    "hello", "hi ", "hey", "good morning", "good afternoon",
    "good evening", "thank", "thanks", "welcome",
    "is this bot", "what can you do", "help me with",
    "tell me a joke", "tell me a story",
    "format", "convert", "convert to",
]


def should_search(query: str) -> bool:
    """Decide whether a query warrants a web search."""
    q = query.lower().strip()
    if not q or len(q) < 3:
        return False
    if q.startswith("/"):
        return False
    if any(neg in q for neg in _SEARCH_TRIGGERS_NEGATIVE):
        return False
    return any(pos in q for pos in _SEARCH_TRIGGERS_POSITIVE)


def sanitize_html(text: str) -> str:
    """Strip HTML tags and unescape HTML entities from a string."""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = html.unescape(clean)
    return clean.strip()


class SearchTool(Tool):
    """SearXNG web search tool.

    Provides web search capabilities with automatic result sanitization.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.searxng_base_url,
            timeout=httpx.Timeout(config.searxng_timeout),
            limits=httpx.Limits(max_connections=10, max_connections_per_host=5),
        )
        logger.info("SearchTool initialized (base_url=%s).", config.searxng_base_url)

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="search",
            description="Web search via SearXNG",
            version="1.0.0",
            tags=["web", "searxng", "search"],
        )

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute a web search query."""
        if not query or not query.strip():
            return ToolResult.fail(data=[], message="Empty search query")

        try:
            results = await self._search(query)
            sanitized = self._sanitize_results(results)
            if not sanitized:
                return ToolResult.ok(data=[], message="No search results found")
            return ToolResult.ok(data=sanitized, message=f"Found {len(sanitized)} result(s)")
        except httpx.TimeoutException:
            logger.warning("Search timed out for query: %s", query)
            return ToolResult.fail(data=[], message="Search timed out")
        except httpx.NetworkError as exc:
            logger.warning("Search network error: %s", exc)
            return ToolResult.fail(data=[], message=f"Search network error: {exc}")
        except Exception as exc:
            logger.exception("Search failed for query: %s", query)
            return ToolResult.fail(data=[], message=f"Search error: {exc}")

    async def health_check(self) -> Dict[str, Any]:
        """Check if the SearXNG instance is reachable."""
        try:
            resp = await self._client.get("/health", timeout=3.0)
            healthy = resp.status_code == 200
            return {
                "name": self.metadata.name,
                "healthy": healthy,
                "details": f"HTTP {resp.status_code}" if healthy else f"HTTP {resp.status_code}",
            }
        except Exception as exc:
            return {
                "name": self.metadata.name,
                "healthy": False,
                "details": str(exc),
            }

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
        logger.info("SearchTool closed.")

    async def _search(self, query: str) -> List[Dict[str, Any]]:
        """Perform the raw SearXNG search."""
        params = {
            "q": query,
            "format": "json",
            "num_results": 5,
        }
        resp = await self._client.get("/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
            }
            for r in results
        ]

    def _sanitize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize search results (strip HTML, escape markdown)."""
        sanitized = []
        for r in results:
            sanitized.append({
                "title": sanitize_html(r.get("title", "")),
                "url": r.get("url", ""),
                "snippet": sanitize_html(r.get("snippet", "")),
            })
        return sanitized