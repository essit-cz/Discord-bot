"""
Health monitor — periodically checks the LLM and SearXNG backends.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

import httpx

from config import get_config

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Monitors the health of the LLM and SearXNG backends.

    Runs a periodic check (default: every 60 seconds) and stores the
    latest status for each backend.
    """

    def __init__(self) -> None:
        config = get_config()
        self._llm_url = config.llm_base_url.rstrip("/")
        self._searxng_url = config.searxng_base_url.rstrip("/")
        self._llm_timeout = config.llm_timeout
        self._searxng_timeout = config.searxng_timeout

        self._llm_healthy: bool = False
        self._searxng_healthy: bool = False
        self._model_name: str = config.llm_model
        self._client: Optional[httpx.AsyncClient] = None
        self._interval_seconds: int = 60
        self._task: Optional[asyncio.Task] = None

        logger.info("HealthMonitor initialized.")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        return self._client

    async def check_llm(self) -> bool:
        """Check if the LLM backend responds to a GET /models request."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self._llm_url}/models",
                timeout=self._llm_timeout,
            )
            self._llm_healthy = response.status_code == 200
            if self._llm_healthy:
                data = response.json()
                models = data.get("data", [])
                if models:
                    self._model_name = models[0].get("id", self._model_name)
                logger.debug("LLM health check: OK (model=%s)", self._model_name)
            else:
                logger.warning("LLM health check: HTTP %d", response.status_code)
        except Exception as exc:
            self._llm_healthy = False
            logger.warning("LLM health check failed: %s", exc)
        return self._llm_healthy

    async def check_searxng(self) -> bool:
        """Check if the SearXNG backend responds to a simple search."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self._searxng_url}/search",
                params={"q": "test", "format": "json"},
                timeout=self._searxng_timeout,
            )
            self._searxng_healthy = response.status_code == 200
            if self._searxng_healthy:
                logger.debug("SearXNG health check: OK")
            else:
                logger.warning("SearXNG health check: HTTP %d", response.status_code)
        except Exception as exc:
            self._searxng_healthy = False
            logger.warning("SearXNG health check failed: %s", exc)
        return self._searxng_healthy

    async def check_all(self) -> dict:
        """Run all health checks and return a summary dict."""
        await asyncio.gather(self.check_llm(), self.check_searxng())
        return self.get_status()

    def get_status(self) -> dict:
        """Return the latest cached health status."""
        return {
            "llm_healthy": self._llm_healthy,
            "searxng_healthy": self._searxng_healthy,
            "model_name": self._model_name,
        }

    def is_llm_healthy(self) -> bool:
        return self._llm_healthy

    def is_searxng_healthy(self) -> bool:
        return self._searxng_healthy

    def get_model_name(self) -> str:
        return self._model_name

    async def start(self, interval_seconds: int = 60) -> None:
        """Start the periodic health check loop."""
        self._interval_seconds = interval_seconds
        self._task = asyncio.create_task(self._periodic_check())
        logger.info("HealthMonitor started (interval=%ds).", interval_seconds)

    async def _periodic_check(self) -> None:
        """Run health checks every *interval_seconds*."""
        while True:
            try:
                await self.check_all()
            except Exception as exc:
                logger.error("Periodic health check error: %s", exc)
            await asyncio.sleep(self._interval_seconds)

    async def stop(self) -> None:
        """Stop the periodic health check loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("HealthMonitor stopped.")