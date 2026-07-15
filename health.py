"""
Health monitor — periodically checks the LLM and SearXNG backends.

Runs a background task that pings both endpoints every 60 seconds
and logs their status.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

from config import Config

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Periodically pings the LLM and SearXNG backends.

    Responsibilities:
    - Run a background health-check loop.
    - Expose current health status.
    - Start/stop the background task.
    """

    def __init__(self, config: Config, llm_client: object, search_client: object) -> None:
        """
        Args:
            config: Application configuration.
            llm_client: An LLMClient instance with a ``health_check()`` method.
            search_client: A SearchClient instance with a ``health_check()`` method.
        """
        self._config = config
        self._llm_client = llm_client
        self._search_client = search_client
        self._task: Optional[asyncio.Task] = None
        self._llm_healthy = False
        self._search_healthy = False
        self._model_name = config.llm_model
        logger.info("HealthMonitor initialized.")

    async def start(self) -> None:
        """Start the background health-check loop."""
        self._task = asyncio.create_task(self._check_loop())
        logger.info("HealthMonitor started (interval=60s).")

    async def stop(self) -> None:
        """Stop the background health-check loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("HealthMonitor stopped.")

    async def _check_loop(self) -> None:
        """Run health checks every 60 seconds."""
        while True:
            try:
                await self._run_checks()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Health check error: %s", exc)

            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break

    async def _run_checks(self) -> None:
        """Execute a single round of health checks."""
        llm_ok = await self._llm_client.health_check()
        search_ok = await self._search_client.health_check()

        llm_changed = llm_ok != self._llm_healthy
        search_changed = search_ok != self._search_healthy

        self._llm_healthy = llm_ok
        self._search_healthy = search_ok

        if llm_changed:
            logger.info("LLM health: %s", "UP" if llm_ok else "DOWN")
        if search_changed:
            logger.info("SearXNG health: %s", "UP" if search_ok else "DOWN")

    def get_status(self) -> Dict[str, object]:
        """Return the current health status of all backends."""
        return {
            "llm_healthy": self._llm_healthy,
            "searxng_healthy": self._search_healthy,
            "model_name": self._model_name,
        }
