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
            logger.error("Tool '%s' execution failed: %s", tool_name, exc)
            return ToolResult.fail(data=None, message=str(exc))

    async def health_checks(self) -> List[Dict[str, Any]]:
        """Run health checks on all registered tools."""
        return [await t.health_check() for t in self._tools.values()]

    async def close(self) -> None:
        """Close all registered tools."""
        for tool in self._tools.values():
            await tool.close()
        logger.info("ToolRegistry closed (%d tools).", len(self._tools))


# ---------------------------------------------------------------------------
# SearchTool (SearXNG)
# ---------------------------------------------------------------------------

class SearchTool(Tool):
    """SearXNG search tool.

    Communicates with a local SearXNG instance to perform web searches.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.searxng_base_url,
            timeout=config.searxng_timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        logger.info("SearchTool initialized (base_url=%s, timeout=%.1fs)",
                     config.searxng_base_url, config.searxng_timeout)

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="search",
            description="Search the web using SearXNG",
            tags=["search", "searxng", "web"],
        )

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute a web search query."""
        if not self._should_search(query):
            return ToolResult.ok(data=[], message="Query does not benefit from a web search.")

        try:
            results = await self._search(query)
            sanitized = self._sanitize_results(results)
            if not sanitized:
                return ToolResult.ok(data=[], message="No results found.")
            return ToolResult.ok(data=sanitized, message=f"Found {len(sanitized)} result(s).")
        except httpx.TimeoutException:
            return ToolResult.fail(
                data=[],
                message=f"Search timed out after {self._config.searxng_timeout}s.",
            )
        except httpx.RequestError as exc:
            return ToolResult.fail(data=[], message=f"Search request failed: {exc}")

    async def _search(self, query: str) -> List[Dict[str, str]]:
        """Fetch raw results from SearXNG."""
        resp = await self._client.get(
            "/search",
            params={
                "q": query,
                "format": "json",
                "results_per_page": 5,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("results", [])
        logger.info("SearXNG returned %d results for '%s'.", len(raw_results), query[:50])
        return raw_results

    def _sanitize_results(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Sanitize search results, extracting snippet from `content` (or `snippet`)."""
        sanitized = []
        for r in results[:5]:
            url = r.get("url", "")
            title = r.get("title", "Untitled")
            content = r.get("content", r.get("snippet", ""))
            # Strip HTML tags
            content = re.sub(r"<[^>]+>", "", content)
            # Unescape HTML entities
            content = html.unescape(content)
            # Truncate
            if len(content) > 300:
                content = content[:297] + "..."
            sanitized.append({
                "title": title,
                "url": url,
                "snippet": content,
            })
        return sanitized

    def _should_search(self, query: str) -> bool:
        """Decide if a query benefits from a web search."""
        q = query.lower().strip()

        # Should search
        search_triggers = [
            "latest", "today", "news", "current", "release date",
            "who is", "what is", "when did", "price", "weather",
            "version", "documentation", "doc", "review", "rating",
            "stock", "score", "population", "capital", "currency",
        ]
        for trigger in search_triggers:
            if trigger in q:
                return True

        # Should NOT search
        skip_triggers = [
            "write a", "write me", "draft", "create", "generate",
            "summarize", "translate", "rewrite", "paraphrase",
            "brainstorm", "explain", "code", "debug", "fix",
            "hello", "hi", "hey", "thanks", "thank you",
            "tell me a", "what do you think",
        ]
        for trigger in skip_triggers:
            if trigger in q:
                return False

        # Default: search queries longer than 5 words
        return len(q.split()) > 5

    async def health_check(self) -> Dict[str, Any]:
        """Check if the SearXNG server is responsive."""
        try:
            resp = await self._client.get("/search?q=test&format=json")
            resp.raise_for_status()
            return {
                "name": self.metadata.name,
                "healthy": True,
                "details": "OK",
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