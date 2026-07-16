"""
Search query planning for Discord messages.

This module converts conversational Discord messages into concise search-engine
queries before they are sent to SearXNG. It is intentionally separate from
SearchTool: SearchTool executes searches, while SearchQueryPlanner decides what
queries should be executed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlparse

from llm_client import LLMClient

logger = logging.getLogger(__name__)

_MAX_QUERIES = 3
_MAX_QUERY_LENGTH = 160
_MAX_RECENT_MESSAGES = 4


@dataclass(frozen=True)
class SearchPlan:
    """Structured search-query plan."""

    queries: list[str]
    reason: str
    fallback_used: bool = False


class SearchQueryPlanner:
    """Plan focused web-search queries from conversational user messages."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def plan(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        current_date: date | None = None,
    ) -> SearchPlan:
        """Return one to three concise search queries for the current message."""
        started = time.perf_counter()
        today = current_date or date.today()

        try:
            messages = self._build_messages(user_message, history or [], today)
            raw = await self._llm_client.chat(
                messages=messages,
                image_url=None,
                stream=False,
                temperature=0.0,
                max_tokens=200,
            )
            plan = self.parse_plan(raw)
            fallback_used = False
        except Exception as exc:
            logger.info("Search planning failed; using deterministic fallback: %s", exc)
            plan = self.fallback_plan(user_message)
            fallback_used = True

        if not plan.queries:
            plan = SearchPlan(
                queries=[self._final_fallback_query(user_message)],
                reason="Planner produced no usable query.",
                fallback_used=True,
            )
            fallback_used = True

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = SearchPlan(
            queries=plan.queries[:_MAX_QUERIES],
            reason=plan.reason,
            fallback_used=plan.fallback_used or fallback_used,
        )
        logger.info(
            "Search query planning completed",
            extra={
                "original_query": self._safe_log_query(user_message),
                "planned_queries": result.queries,
                "planning_latency_ms": elapsed_ms,
                "fallback_used": result.fallback_used,
            },
        )
        return result

    def parse_plan(self, raw_response: str) -> SearchPlan:
        """Parse and validate strict JSON returned by the planning LLM."""
        if not raw_response or raw_response.count("\n") > 20:
            raise ValueError("Planner response is empty or newline-heavy.")

        text = raw_response.strip()
        if text.startswith("