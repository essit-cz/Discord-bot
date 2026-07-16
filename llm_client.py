"""
Async vLLM OpenAI-compatible client.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """Reusable async client for OpenAI-compatible chat completions."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.vllm_base_url,
            timeout=httpx.Timeout(config.llm_timeout),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
            headers={"Content-Type": "application/json"},
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        image_url: str | None = None,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Create a chat completion and return a plain string."""
        payload = self._build_payload(
            messages=messages,
            image_url=image_url,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            if stream:
                chunks: list[str] = []
                async for chunk in self.stream_chat(
                    messages=messages,
                    image_url=image_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    chunks.append(chunk)
                return "".join(chunks)

            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return self._extract_content(data)
        except httpx.TimeoutException:
            logger.warning("LLM request timed out")
            return "Sorry, the AI backend timed out. Please try again."
        except httpx.HTTPStatusError as exc:
            logger.error("LLM HTTP error: %s", exc.response.status_code)
            return "Sorry, the AI backend returned an error. Please try again."
        except Exception:
            logger.exception("LLM request failed")
            return "Sorry, something went wrong while contacting the AI backend."

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        image_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield streaming chat completion chunks."""
        payload = self._build_payload(
            messages=messages,
            image_url=image_url,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                        delta = event["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception:
            logger.exception("LLM streaming request failed")
            yield "Sorry, something went wrong while streaming the AI response."

    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        try:
            response = await self._client.get("/v1/models")
            return {
                "name": "llm",
                "healthy": response.status_code < 500,
                "status_code": response.status_code,
            }
        except Exception as exc:
            return {"name": "llm", "healthy": False, "error": str(exc)}

    async def close(self) -> None:
        """Close the reusable HTTP client."""
        await self._client.aclose()

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        image_url: str | None,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        model_messages = messages
        if image_url:
            model_messages = self._with_image(messages, image_url)

        return {
            "model": self._config.model_name,
            "messages": model_messages,
            "temperature": self._config.temperature if temperature is None else temperature,
            "max_tokens": self._config.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    def _with_image(self, messages: list[dict[str, Any]], image_url: str) -> list[dict[str, Any]]:
        if not messages:
            return messages

        copied = [dict(message) for message in messages]
        last = dict(copied[-1])
        content = last.get("content", "")

        last["content"] = [
            {"type": "text", "text": str(content)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        copied[-1] = last
        return copied

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
            return str(content).strip()
        except (KeyError, IndexError, TypeError):
            logger.error("Unexpected LLM response schema")
            return "Sorry, the AI backend returned an unexpected response."