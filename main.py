"""
Discord AI Bot — Entry Point

Constructs all shared objects and wires them together via dependency
injection.  Handles structured logging and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from config import Config
from conversation import ConversationManager
from discord_bot import DiscordBot
from formatter import MessageFormatter
from health import HealthMonitor
from llm_client import LLMClient
from prompt_builder import PromptBuilder
from search_client import SearchClient

# ── Structured Logging ──────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FMT,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def build_app() -> DiscordBot:
    """
    Construct all shared objects and wire them together.

    This is the single place where dependencies are instantiated.
    Every class receives its dependencies through its constructor.
    """
    # 1. Configuration (loaded once, immutable).
    config = Config.load()

    # 2. HTTP clients (one per backend).
    llm_client = LLMClient(config)
    search_client = SearchClient(config)

    # 3. State managers.
    conversation_manager = ConversationManager(config)

    # 4. Stateless helpers.
    prompt_builder = PromptBuilder()
    formatter = MessageFormatter()

    # 5. Health monitor (depends on both clients).
    health_monitor = HealthMonitor(config, llm_client, search_client)

    # 6. The bot itself (depends on everything).
    bot = DiscordBot(
        config=config,
        llm_client=llm_client,
        search_client=search_client,
        conversation_manager=conversation_manager,
        formatter=formatter,
        prompt_builder=prompt_builder,
        health_monitor=health_monitor,
    )

    logger.info("Application wired up successfully.")
    return bot


async def run_bot(bot: DiscordBot) -> None:
    """
    Run the bot with graceful shutdown on SIGINT / SIGTERM.
    """
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received.")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Start the bot in a background task.
    bot_task = asyncio.create_task(bot.start())

    # Wait for a shutdown signal.
    await shutdown_event.wait()

    # Graceful shutdown.
    logger.info("Initiating graceful shutdown...")
    await bot.close()
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass
    logger.info("Bot shut down complete.")


def main() -> None:
    """Entry point."""
    logger.info("Starting Discord AI Bot...")
    bot = build_app()
    try:
        asyncio.run(run_bot(bot))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught; shutting down.")


if __name__ == "__main__":
    main()
