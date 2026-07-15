"""
Discord AI Bot — Entry Point.

Usage:
    python main.py

Environment:
    Copy `.env.example` to `.env` and fill in the values.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Ensure the project root is on the import path.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from discord_bot import DiscordBot

# ── Structured Logging ──────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# Silence noisy third-party loggers.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("discord").setLevel(logging.INFO)
logging.getLogger("websockets").setLevel(logging.WARNING)

logger = logging.getLogger("bot.main")


class BotRunner:
    """Manages the bot lifecycle: startup, graceful shutdown, and cleanup."""

    def __init__(self) -> None:
        self.bot: DiscordBot | None = None
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """Start the bot and wait for a graceful shutdown signal."""
        # Load configuration.
        load_config()

        # Create the bot.
        self.bot = DiscordBot()
        logger.info("Starting Discord AI Bot...")

        # Set up signal handlers for graceful shutdown.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        # Run the bot in a task.
        bot_task = asyncio.create_task(self.bot.start())

        # Wait for a shutdown signal.
        await self._shutdown_event.wait()

        logger.info("Shutdown signal received. Stopping bot...")
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass

        await self.bot.close()
        logger.info("Bot shut down gracefully.")

    def run_sync(self) -> None:
        """Synchronous entry point."""
        asyncio.run(self.run())


def main() -> None:
    """Module-level entry point."""
    runner = BotRunner()
    runner.run_sync()


if __name__ == "__main__":
    main()