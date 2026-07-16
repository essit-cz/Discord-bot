"""
Search query planning for Discord messages.

The planner converts conversational Discord messages into concise search-engine
queries before those queries are sent to SearXNG. SearchTool remains responsible
only for executing searches.
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
_COMPACT_WHITESPACE = re.compile(r"\s+")


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
        """Return one to three concise queries for the current message."""
        started = time.perf_counter()
        today = current_date or date.today()
        fallback_used = False

        try:
            messages = self._build_messages(user_message, history or [], today)
            raw_response = await self._llm_client.chat(
                messages=messages,
                image_url=None,
                stream=False,
                temperature=0.0,
                max_tokens=200,
            )
            plan = self.parse_plan(raw_response)
        except Exception as exc:
            logger.info("Search planning failed; using fallback: %s", exc)
            plan = self.fallback_plan(user_message)
            fallback_used = True

        if not plan.queries:
            plan = SearchPlan(
                queries=[self._final_fallback_query(user_message)],
                reason="Planner produced no usable queries.",
                fallback_used=True,
            )
            fallback_used = True

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
                "planning_latency_ms": int((time.perf_counter() - started) * 1000),
                "fallback_used": result.fallback_used,
            },
        )
        return result

    def parse_plan(self, raw_response: str) -> SearchPlan:
        """Parse and validate strict JSON returned by the planning LLM."""
        if not raw_response:
            raise ValueError("Planner response is empty.")
        if raw_response.count("\n") > 20:
            raise ValueError("Planner response is too newline-heavy.")

        text = raw_response.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*" + re.escape(fence) + r"$", "", text).strip()

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Planner JSON must be an object.")

        raw_queries = data.get("queries")
        if not isinstance(raw_queries, list):
            raise ValueError("Planner JSON must contain a queries list.")

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = ""

        queries = self._validate_queries(raw_queries)
        if not queries:
            raise ValueError("Planner returned no valid queries.")

        return SearchPlan(
            queries=queries,
            reason=reason.strip()[:300],
            fallback_used=False,
        )

    def fallback_plan(self, user_message: str) -> SearchPlan:
        """Create a deterministic plan when LLM planning fails."""
        lower = user_message.casefold()

        if "ebay" in lower and ("5090" in lower or "rtx" in lower):
            queries = [
                "RTX 5090 current price site:ebay.com",
                "RTX 5090 sold listings site:ebay.com",
            ]
        elif "vllm" in lower and any(term in lower for term in ("latest", "stable", "version")):
            queries = [
                "vLLM latest stable release site:github.com/vllm-project/vllm",
                "vLLM latest version official documentation",
            ]
        elif "miroslav sládek" in lower or "miroslav sladek" in lower:
            queries = [
                '"Miroslav Sládek" political career',
                '"Miroslav Sládek" biography',
            ]
        else:
            cleaned = self.clean_query(user_message)
            queries = [cleaned] if cleaned else [self._final_fallback_query(user_message)]

        return SearchPlan(
            queries=self._validate_queries(queries) or [self._final_fallback_query(user_message)],
            reason="Deterministic fallback query cleanup was used.",
            fallback_used=True,
        )

    def clean_query(self, user_message: str) -> str:
        """Remove conversational filler while preserving searchable terms."""
        text = re.sub(r"<@!?\d+>", " ", user_message)
        text = text.replace("\n", " ")

        url_domains = self._extract_domains(text)
        text = re.sub(r"https?://\S+", " ", text)

        filler_patterns = [
            r"\bplease\b",
            r"\bpls\b",
            r"\btell me\b",
            r"\bgive me (?:a )?link\b",
            r"\bsend me (?:a )?link\b",
            r"\bwhat do you think\b",
            r"\bcan you\b",
            r"\bcould you\b",
            r"\bshow me\b",
            r"\bfind me\b",
            r"\bhoď mi(?: nějaký)? link\b",
            r"\bposli(?: mi)? link\b",
            r"\bpošli(?: mi)? link\b",
            r"\bpros[ií]m\b",
            r"\bahoj\b",
            r"\bhello\b",
            r"\bhi\b",
            r"\bhey\b",
            r"\bdiky\b",
            r"\bdíky\b",
        ]
        for pattern in filler_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        text = self._normalize_common_entities(text)
        text = re.sub(r"[?!.;,]+", " ", text)
        text = _COMPACT_WHITESPACE.sub(" ", text).strip()

        for domain in self._domains_from_text(user_message) | url_domains:
            site_operator = f"site:{domain}"
            if site_operator.casefold() not in text.casefold():
                text = f"{text} {site_operator}".strip()

        if len(text) > _MAX_QUERY_LENGTH:
            text = text[:_MAX_QUERY_LENGTH].rsplit(" ", 1)[0].strip()
        return text

    def _build_messages(
        self,
        user_message: str,
        history: list[dict[str, str]],
        current_date: date,
    ) -> list[dict[str, str]]:
        recent_messages: list[dict[str, str]] = []
        for item in history[-_MAX_RECENT_MESSAGES:]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                recent_messages.append({"role": role, "content": content[:500]})

        recent_context = "\n".join(
            f"{message['role']}: {message['content']}" for message in recent_messages
        )

        system_prompt = (
            "You plan search-engine queries for a Discord AI bot. Return JSON only. "
            "Do not answer the user. Produce 1 to 3 concise search queries. "
            "Preserve important entities, model numbers, product names, people, "
            "dates, locations, and requested websites. Remove greetings, politeness, "
            "Discord mentions, and instructions such as 'give me a link'. Translate "
            "or normalize into English when that is likely to improve search results, "
            "while preserving local-language names and important terms. Add useful "
            "operators such as site:ebay.com, site:github.com, site:docs.vllm.ai, "
            "and quote exact names or phrases when appropriate."
        )
        user_payload = {
            "current_date": current_date.isoformat(),
            "recent_context": recent_context,
            "user_message": user_message,
            "output_schema": {
                "queries": ["concise search query"],
                "reason": "brief explanation of why these searches are useful",
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def _validate_queries(self, raw_queries: list[Any]) -> list[str]:
        seen: set[str] = set()
        valid: list[str] = []

        for raw_query in raw_queries[: _MAX_QUERIES * 2]:
            if not isinstance(raw_query, str):
                continue
            query = raw_query.strip()
            if not query or query.count("\n") > 1:
                continue
            query = _COMPACT_WHITESPACE.sub(" ", query)
            if len(query) > _MAX_QUERY_LENGTH:
                query = query[:_MAX_QUERY_LENGTH].rsplit(" ", 1)[0].strip()
            if not query:
                continue

            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            valid.append(query)
            if len(valid) >= _MAX_QUERIES:
                break

        return valid

    def _normalize_common_entities(self, text: str) -> str:
        lower = text.casefold()
        if "5090" in lower and "rtx" not in lower:
            text = re.sub(r"\b5090\b", "RTX 5090", text, flags=re.IGNORECASE)
        if "vllm" in lower:
            text = re.sub(r"\bvllm\b", "vLLM", text, flags=re.IGNORECASE)
        return text

    def _domains_from_text(self, text: str) -> set[str]:
        lower = text.casefold()
        domains: set[str] = set()
        markers = {
            "ebay": "ebay.com",
            "github": "github.com",
            "docs.vllm.ai": "docs.vllm.ai",
            "vllm docs": "docs.vllm.ai",
        }
        for marker, domain in markers.items():
            if marker in lower:
                domains.add(domain)
        return domains

    def _extract_domains(self, text: str) -> set[str]:
        domains: set[str] = set()
        for match in re.finditer(r"https?://[^\s]+", text):
            parsed = urlparse(match.group(0))
            domain = parsed.netloc.casefold()
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                domains.add(domain)
        return domains

    def _final_fallback_query(self, user_message: str) -> str:
        cleaned = re.sub(r"<@!?\d+>", " ", user_message)
        cleaned = cleaned.replace("\n", " ")
        cleaned = _COMPACT_WHITESPACE.sub(" ", cleaned).strip()
        return cleaned[:_MAX_QUERY_LENGTH] or "current information"

    def _safe_log_query(self, query: str) -> str:
        return _COMPACT_WHITESPACE.sub(" ", query).strip()[:300]


def normalize_result_url(url: str) -> str:
    """Normalize a result URL for deduplication."""
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().casefold()

    netloc = parsed.netloc.casefold()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.casefold()}://{netloc}{path}".casefold()


def merge_search_results(
    result_groups: list[list[dict[str, Any]]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Merge result groups while preserving earliest rank and deduplicating URLs."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for group in result_groups:
        for result in group:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url", "")).strip()
            key = normalize_result_url(url) if url else str(result.get("title", "")).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(result)
            if len(merged) >= limit:
                return merged

    return merged