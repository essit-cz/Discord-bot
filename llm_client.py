"""
LLM client — communicates with a local vLLM OpenAI-compatible server.

Uses a single reusable httpx.AsyncClient with retries and exponential
backoff.  Supports chat completions, image messages, and optional
streaming.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async client for a vLLM OpenAI-compatible endpoint.

    Responsibilities:
    - Maintain one reusable httpx.AsyncClient.
    - Send chat completion requests (text + optional images).
    - Support optional streaming responses.
    - Handle retries with exponential backoff.
    - Return friendly error messages (no raw stack traces).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.llm_base_url,
            timeout=httpx.Timeout(config.llm_timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        logger.info("LLMClient initialized (base_url=%s, model=%s).",
                     config.llm_base_url, config.llm_model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        image_url: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Send a chat completion request and return the assistant's response.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            image_url: Optional image URL to attach to the last user message.
            stream: If True, use streaming (returns concatenated text).

        Returns:
            The assistant's text response.

        Raises:
            RuntimeError: On a friendly-wrapped backend error.
        """
        # Attach image to the last user message if provided.
        if image_url:
            messages = self._attach_image(messages, image_url)

        payload = {
            "model": self._config.llm_model,
            "messages": messages,
            "temperature": self._config.llm_temperature,
            "max_tokens": self._config.llm_max_tokens,
            "stream": stream,
        }

        try:
            if stream:
                return await self._chat_stream(payload)
            else:
                return await self._chat_standard(payload)
        except httpx.TimeoutException:
            logger.error("LLM request timed out after %.1fs.", self._config.llm_timeout)
            raise RuntimeError("The model took too long to respond. Try again shortly.")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text
            logger.error("LLM HTTP %d: %s", status, body[:200])
            if status == 429:
                raise RuntimeError("The model is busy (rate-limited). Try again shortly.")
            elif status == 503:
                raise RuntimeError("The model server is temporarily unavailable.")
            else:
                raise RuntimeError(f"The model server returned an unexpected error ({status}).")
        except httpx.RequestError as exc:
            logger.error("LLM request error: %s", exc)
            raise RuntimeError("Could not reach the model server. Check your network.")

    async def _chat_standard(self, payload: Dict[str, Any]) -> str:
        """Send a standard (non-streaming) chat request."""
        response = await self._client.post(
            "/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("The model returned an empty response.")
        return content

    async def _chat_stream(self, payload: Dict[str, Any]) -> str:
        """Send a streaming chat request and concatenate chunks."""
        chunks: List[str] = []
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        chunks.append(delta["content"])
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        result = "".join(chunks)
        if not result:
            raise RuntimeError("The model returned an empty streaming response.")
        return result

    def _attach_image(
        self, messages: List[Dict[str, str]], image_url: str
    ) -> List[Dict[str, Any]]:
        """
        Attach an image URL to the last user message in the message list.

        Returns a new list with the modified message.
        """
        new_messages = []
        for msg in messages:
            if msg["role"] == "user":
                # Replace the last user message with a content array.
                new_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                })
            else:
                new_messages.append(msg)
        return new_messages

    async def health_check(self) -> bool:
        """
        Ping the vLLM server's ``/v1/models`` endpoint.

        Returns ``True`` if the server responds with a 200 OK.
        """
        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
            return True
        except httpx.RequestError as exc:
            logger.debug("LLM health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
        logger.info("LLMClient closed.")
