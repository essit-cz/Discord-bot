"""
Tool Registry and Tool implementations.

Tools are independently registered capabilities. DiscordBot dispatches through
ToolRegistry and does not need to know how individual tools work.
"""

from __future__ import annotations

import abc
import html
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolMetadata:
    """Public metadata for a registered tool."""

    name: str
    description: str
    enabled: bool = True


@dataclass(frozen=True)
class ToolResult:
    """Result returned by a tool execution."""

    success: bool
    data: Any = None
    message: str = ""
    error: str = ""

    @classmethod
    def ok(cls, data: Any = None, message: str = "") -> "ToolResult":
        return cls(success=True, data=data, message=message)

    @classmethod
    def fail(cls, error: str, message: str = "") -> "ToolResult":
        return cls(success=False, error=error, message=message)


class Tool(abc.ABC):
    """Abstract interface for executable tools."""

    @property
    @abc.abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return metadata for this tool."""

    @abc.abstractmethod
    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute the tool."""

    async def health_check(self) -> dict[str, Any]:
        """Return tool health information."""
        return {"name": self.metadata.name, "healthy": True}

    async def close(self) -> None:
        """Release resources held by this tool."""


class ToolRegistry:
    """Registry responsible for tool registration and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by metadata name."""
        name = tool.metadata.name
        self._tools[name] = tool
        logger.info("Registered tool: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a registered tool."""
        self._tools.pop(name, None)

    def has(self, name: str) -> bool:
        """Return whether a tool is registered."""
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        """Return a registered tool if present."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolMetadata]:
        """Return metadata for all registered tools."""
        return [tool.metadata for tool in self._tools.values()]

    async def execute(self, tool_name: str, query: str, **kwargs: Any) -> ToolResult:
        """Dispatch execution to a registered tool."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.fail(f"Tool not registered: {tool_name}")
        return await tool.execute(query, **kwargs)

    async def health_checks(self) -> list[dict[str, Any]]:
        """Run health checks for all tools."""
        checks = []
        for tool in self._tools.values():
            try:
                checks.append(await tool.health_check())
            except Exception as exc:
                logger.exception("Tool health check failed: %s", tool.metadata.name)
                checks.append(
                    {
                        "name": tool.metadata.name,
                        "healthy": False,
                        "error": str(exc),
                    }
                )
        return checks

    async def close(self) -> None:
        """Close all registered tools."""
        for tool in self._tools.values():
            try:
                await tool.close()
            except Exception:
                logger.exception("Failed to close tool: %s", tool.metadata.name)


class SearchTool(Tool):
    """SearXNG-backed web search tool."""

    _SEARCH_KEYWORDS = {
        "latest",
        "today",
        "news",
        "who is",
        "what is",
        "release date",
        "current version",
        "documentation",
        "docs",
        "current",
        "price",
        "cost",
        "stojí",
        "proslavil",
    }
    _NO_SEARCH_KEYWORDS = {
        "coding",
        "translation",
        "translate",
        "rewrite",
        "summarize",
        "brainstorm",
        "conversation",
        "creative writing",
        "hello",
        "hi",
        "hey",
    }

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.searxng_base_url,
            timeout=httpx.Timeout(config.searxng_timeout),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
            headers={"User-Agent": "DiscordBot/1.0"},
        )

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="search",
            description="Search the web using SearXNG",
            enabled=getattr(self._config, "search_enabled", True),
        )

    def should_search(self, query: str) -> bool:
        """Return whether the user query should trigger web search."""
        if not getattr(self._config, "search_enabled", True):
            return False

        normalized = query.lower().strip()
        if not normalized:
            return False

        if any(keyword in normalized for keyword in self._NO_SEARCH_KEYWORDS):
            return False

        return any(keyword in normalized for keyword in self._SEARCH_KEYWORDS)

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute a SearXNG search and return sanitized results."""
        limit = int(kwargs.get("limit", 5))

        try:
            response = await self._client.get(
                "/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": "auto",
                    "safesearch": 1,
                },
            )
            response.raise_for_status()
            payload = response.json()
            results = self.sanitize_results(payload.get("results", []), limit=limit)
            return ToolResult.ok(
                data=results,
                message=f"Found {len(results)} search results.",
            )
        except httpx.TimeoutException:
            logger.warning("Search timed out for query: %s", query[:120])
            return ToolResult.fail("Search timed out.", "Search timed out.")
        except Exception:
            logger.exception("Search failed for query: %s", query[:120])
            return ToolResult.fail("Search failed.", "Search failed.")

    def sanitize_results(self, results: list[dict[str, Any]], limit: int = 5) -> list[dict[str, str]]:
        """Sanitize SearXNG results for prompt use."""
        sanitized: list[dict[str, str]] = []

        for result in results:
            if not isinstance(result, dict):
                continue

            title = self._sanitize_html(str(result.get("title", ""))).strip()
            url = str(result.get("url", "")).strip()
            snippet_source = result.get("content") or result.get("snippet") or ""
            snippet = self._sanitize_html(str(snippet_source)).strip()

            if not title and not snippet:
                continue

            sanitized.append(
                {
                    "title": title[:200],
                    "url": url[:500],
                    "snippet": snippet[:500],
                }
            )

            if len(sanitized) >= limit:
                break

        return sanitized

    async def health_check(self) -> dict[str, Any]:
        """Check SearXNG health."""
        try:
            response = await self._client.get(
                "/search",
                params={"q": "test", "format": "json"},
            )
            return {
                "name": self.metadata.name,
                "healthy": response.status_code < 500,
                "status_code": response.status_code,
            }
        except Exception as exc:
            return {
                "name": self.metadata.name,
                "healthy": False,
                "error": str(exc),
            }

    async def close(self) -> None:
        """Close the reusable HTTP client."""
        await self._client.aclose()

    def _sanitize_html(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text)