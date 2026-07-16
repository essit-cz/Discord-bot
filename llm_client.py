"""
LLM client — communicates with a local vLLM OpenAI-compatible server.

Uses one reusable httpx.AsyncClient. Supports chat completions, image messages,
optional streaming, per-request generation overrides, and friendly errors.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for a vLLM OpenAI-compatible endpoint."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.llm_base_url,
            timeout=httpx.Timeout(config.llm_timeout),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )
        logger.info(
            "LLMClient initialized (base_url=%s, model=%s).",
            config.llm_base_url,
            config.llm_model,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        image_url: str | None = None,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a chat completion request and return the assistant response.

        Optional temperature and max_tokens are per-request overrides used by the
        search-query planner. Normal bot responses continue to use Config values.
        """
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

            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return self._extract_content(data)
        except httpx.TimeoutException:
            logger.error("LLM request timed out after %.1fs.", self._config.llm_timeout)
            return "The model took too long to respond. Try again shortly."
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error("LLM HTTP %d: %s", status, exc.response.text[:200])
            if status == 429:
                return "The model is busy (rate-limited). Try again shortly."
            if status == 503:
                return "The model server is temporarily unavailable."
            return f"The model server returned an unexpected error ({status})."
        except httpx.RequestError as exc:
            logger.error("LLM request error: %s", exc)
            return "Could not reach the model server. Check your network."
        except Exception:
            logger.exception("Unexpected LLM client error")
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
                "/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                        delta = event["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
        except Exception:
            logger.exception("LLM streaming request failed")
            yield "Sorry, something went wrong while streaming the AI response."

    async def health_check(self) -> bool:
        """Return True when the vLLM models endpoint responds successfully."""
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            return True
        except httpx.RequestError as exc:
            logger.debug("LLM health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
        logger.info("LLMClient closed.")

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        image_url: str | None,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        model_messages = self._attach_image(messages, image_url) if image_url else messages
        return {
            "model": self._config.llm_model,
            "messages": model_messages,
            "temperature": self._config.llm_temperature if temperature is None else temperature,
            "max_tokens": self._config.llm_max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    def _attach_image(
        self,
        messages: list[dict[str, Any]],
        image_url: str,
    ) -> list[dict[str, Any]]:
        """Attach an image URL to the last user message only."""
        new_messages = [dict(message) for message in messages]
        for index in range(len(new_messages) - 1, -1, -1):
            if new_messages[index].get("role") == "user":
                content = new_messages[index].get("content", "")
                new_messages[index] = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": str(content)},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
                break
        return new_messages

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.error("Unexpected LLM response schema")
            return "The model returned an unexpected response."

        text = str(content).strip()
        if not text:
            return "The model returned an empty response."
        return text