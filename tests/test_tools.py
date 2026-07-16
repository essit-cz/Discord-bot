"""Unit tests for ToolRegistry and SearchTool."""

from __future__ import annotations

import asyncio
from typing import Any

from config import Config
from tools import SearchTool, Tool, ToolMetadata, ToolRegistry, ToolResult


class FakeTool(Tool):
    """Minimal tool implementation for ToolRegistry tests."""

    def __init__(self, name: str = "fake") -> None:
        self._name = name
        self.executions: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name=self._name, description="Fake test tool")

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        self.executions.append((query, kwargs))
        return ToolResult.ok(data={"query": query})

    async def close(self) -> None:
        self.closed = True


def make_config(*, search_enabled: bool = True) -> Config:
    """Create the smallest valid configuration for SearchTool tests."""
    return Config(
        discord_token="test-token",
        searxng_base_url="http://127.0.0.1:8080",
        search_enabled=search_enabled,
    )


def test_registry_registers_dispatches_and_unregisters_tools() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        tool = FakeTool()
        registry.register(tool)

        assert registry.has("fake")
        assert registry.get("fake") is tool
        assert registry.list_tools() == [tool.metadata]

        result = await registry.execute("fake", "example", limit=3)
        assert result.success is True
        assert result.data == {"query": "example"}
        assert tool.executions == [("example", {"limit": 3})]

        registry.unregister("fake")
        assert registry.has("fake") is False
        missing = await registry.execute("fake", "example")
        assert missing.success is False
        assert "not registered" in missing.error.lower()

    asyncio.run(run())


def test_registry_closes_all_registered_tools() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        first = FakeTool("first")
        second = FakeTool("second")
        registry.register(first)
        registry.register(second)

        await registry.close()

        assert first.closed is True
        assert second.closed is True

    asyncio.run(run())


def test_search_tool_should_search_respects_enabled_setting() -> None:
    enabled_tool = SearchTool(make_config(search_enabled=True))
    disabled_tool = SearchTool(make_config(search_enabled=False))
    try:
        assert enabled_tool.should_search("What is the latest stable vLLM version?")
        assert enabled_tool.should_search("Kolik teď stojí RTX 5090?")
        assert enabled_tool.should_search("Who is Ada Lovelace?")
        assert not enabled_tool.should_search("Please translate this text.")
        assert not enabled_tool.should_search("Hello!")
        assert not disabled_tool.should_search("What is the current version?")
    finally:
        asyncio.run(enabled_tool.close())
        asyncio.run(disabled_tool.close())


def test_search_tool_sanitizes_content_before_snippet_and_skips_bad_results() -> None:
    tool = SearchTool(make_config())
    try:
        results = tool.sanitize_results(
            [
                {
                    "title": "<b>First &amp; Result</b>",
                    "url": "https://example.test/first",
                    "content": "<p>Preferred <em>content</em></p>",
                    "snippet": "This fallback must not be used.",
                },
                {
                    "title": "Second",
                    "url": "https://example.test/second",
                    "snippet": "<div>Fallback &amp; snippet</div>",
                },
                "not-a-result",
                {"title": "", "url": "https://example.test/empty"},
            ],
            limit=5,
        )

        assert results == [
            {
                "title": "First & Result",
                "url": "https://example.test/first",
                "snippet": "Preferred content",
            },
            {
                "title": "Second",
                "url": "https://example.test/second",
                "snippet": "Fallback & snippet",
            },
        ]
    finally:
        asyncio.run(tool.close())


def test_search_tool_truncates_prompt_bound_result_fields() -> None:
    tool = SearchTool(make_config())
    try:
        results = tool.sanitize_results(
            [
                {
                    "title": "t" * 250,
                    "url": "https://example.test/" + ("u" * 600),
                    "content": "c" * 600,
                }
            ]
        )

        assert len(results) == 1
        assert len(results[0]["title"]) == 200
        assert len(results[0]["url"]) == 500
        assert len(results[0]["snippet"]) == 500
    finally:
        asyncio.run(tool.close())