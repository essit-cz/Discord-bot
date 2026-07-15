"""
LLM client — thin async wrapper around an OpenAI-compatible chat completions
endpoint (vLLM).
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Dict, List, Optional

import json
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config import get_config
from utils import estimate_tokens

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async client for an OpenAI-compatible LLM backend.

    Creates a single ``httpx.AsyncClient`` and reuses it across requests.
    Supports standard chat completions, image messages, and optional
    streaming.
    """

    def __init__(self) -> None:
        config = get_config()
        self._base_url = config.llm_base_url.rstrip("/")
        self._model = config.llm_model
        self._temperature = config.llm_temperature
        self._max_tokens = config.llm_max_tokens
        self._timeout = config.llm_timeout
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(
            "LLMClient initialized: url=%s, model=%s, timeout=%.1fs",
            self._base_url,
            self._model,
            self._timeout,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize the underlying HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True,
    )
    async def chat(
        self,
        messages: List[dict],
        image_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request and return the assistant's text.

        Parameters
        ----------
        messages : list[dict]
            Standard OpenAI message list (``role`` + ``content``).
        image_url : str, optional
            URL of an image to include in the last user message.
        temperature : float, optional
            Override the default temperature.
        max_tokens : int, optional
            Override the default max tokens.

        Returns
        -------
        str
            The assistant's response text.
        """
        client = await self._get_client()

        # If an image URL is provided, inject it into the last user message.
        if image_url:
            messages = self._inject_image(messages, image_url)

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        try:
            response = await client.post(
                "/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.debug(
                "LLM response: %d input tokens (est), %d output chars",
                sum(estimate_tokens(m.get("content", "")) for m in messages),
                len(content) if content else 0,
            )
            return content or ""
        except httpx.HTTPStatusError as exc:
            logger.error("LLM HTTP error %d: %s", exc.response.status_code, exc.response.text)
            return _friendly_http_error(exc.response.status_code)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.error("LLM response parse error: %s", exc)
            return "⚙️ *The model returned an unexpected response.*"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True,
    )
    async def chat_stream(
        self,
        messages: List[dict],
        image_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion, yielding text deltas.

        Parameters are the same as :meth:`chat`.
        """
        client = await self._get_client()

        if image_url:
            messages = self._inject_image(messages, image_url)

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": True,
        }

        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except httpx.HTTPStatusError as exc:
            logger.error("LLM stream HTTP error %d: %s", exc.response.status_code, exc.response.text)
            yield _friendly_http_error(exc.response.status_code)

    def _inject_image(self, messages: List[dict], image_url: str) -> List[dict]:
        """
        Add an image URL to the last user message in the message list.

        Converts text content to the multi-modal ``content`` array format.
        """
        new_messages = list(messages)
        for i in range(len(new_messages) - 1, -1, -1):
            if new_messages[i].get("role") == "user":
                old_content = new_messages[i]["content"]
                new_messages[i]["content"] = [
                    {"type": "text", "text": old_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ]
                logger.debug("Injected image URL into user message: %s", image_url)
                return new_messages

        # If no user message exists, prepend one.
        new_messages.insert(
            0,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        )
        return new_messages

    async def get_model_info(self) -> dict:
        """Query the backend for the current model name."""
        client = await self._get_client()
        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            return {"available": [m["id"] for m in models], "count": len(models)}
        except Exception as exc:
            logger.error("Error fetching model info: %s", exc)
            return {"available": [self._model], "count": 1}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("LLMClient closed.")


def _friendly_http_error(status_code: int) -> str:
    """Map an HTTP status code to a user-friendly error string."""
    messages = {
        400: "⚙️ *Bad request sent to the model.*",
        401: "⚙️ *The model server reports an authentication error.*",
        404: "⚙️ *The model endpoint was not found.*",
        408: "⏳ *The model took too long to respond.*",
        429: "🔄 *The model server is rate-limiting requests.*",
        500: "⚙️ *The model server encountered an internal error.*",
        502: "⚙️ *Bad gateway — the model server returned invalid data.*",
        503: "🔄 *The model server is temporarily unavailable.*",
        504: "⏳ *The model server timed out.*",
    }
    return messages.get(status_code, "⚙️ *The model returned an unexpected response.*")