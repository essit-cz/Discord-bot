"""
Discord AI Bot — Entry Point

Constructs all shared objects and wires them together via dependency injection.
"""

import asyncio
import logging
import signal
import sys

from config import Config
from conversation import ConversationManager
from discord_bot import DiscordBot
from formatter import MessageFormatter
from llm_client import LLMClient
from prompt_builder import PromptBuilder
from search_planner import SearchQueryPlanner
from tools import SearchTool, ToolRegistry

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------

def build_app():
    """Construct all shared objects and wire them together."""
    # Configuration
    config = Config.load()

    # LLM client
    llm_client = LLMClient(config)

    # Tool registry
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool(config))

    # Conversation manager
    conversation_manager = ConversationManager(config)

    # Prompt builder
    prompt_builder = PromptBuilder()

    # Search query planner
    search_query_planner = SearchQueryPlanner(llm_client)

    # Formatter
    formatter = MessageFormatter()

    # Discord bot
    bot = DiscordBot(
        config=config,
        llm_client=llm_client,
        tool_registry=tool_registry,
        conversation_manager=conversation_manager,
        formatter=formatter,
        prompt_builder=prompt_builder,
        search_query_planner=search_query_planner,
    )

    return bot


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

async def graceful_shutdown(bot: DiscordBot, loop: asyncio.AbstractEventLoop) -> None:
    """Gracefully shut down the bot and all HTTP clients."""
    logger.info("Shutting down...")
    await bot.close()
    logger.info("Bot shut down.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Main entry point."""
    loop = asyncio.get_event_loop()

    # Register signal handlers for graceful shutdown.
    stop = loop.create_future()

    def _signal_handler() -> None:
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Build and run the bot.
    bot = build_app()

    logger.info("Starting Discord bot...")
    try:
        await bot.start(token=bot._config.discord_token)
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Bot crashed.")
    finally:
        await graceful_shutdown(bot, loop)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")