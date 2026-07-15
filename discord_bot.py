"""
DiscordBot — the main discord.py bot class.

Wires together all sub-components (LLM, Search, Conversation,
PromptBuilder, Formatter, HealthMonitor) and handles Discord events
and commands.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import discord
from discord.ext import commands

from config import get_config
from conversation import ConversationManager
from formatter import MessageFormatter
from health import HealthMonitor
from llm_client import LLMClient
from prompt_builder import PromptBuilder
from search_client import SearchClient
from utils import is_valid_image_url

logger = logging.getLogger(__name__)

# Intents the bot needs.
_INTENTS = discord.Intents.default()
_INTENTS.message_content = True
_INTENTS.members = True
_INTENTS.dm_messages = True


class DiscordBot:
    """
    Production-quality Discord AI bot.

    Supports:
    - Mentions and DMs
    - Image understanding (vision)
    - Conversation memory per channel
    - Automatic web search via SearXNG
    - Slash commands and prefix commands
    - Streaming-ready architecture
    """

    def __init__(self) -> None:
        config = get_config()

        # Sub-components.
        self.conversation_manager = ConversationManager()
        self.llm_client = LLMClient()
        self.search_client = SearchClient()
        self.prompt_builder = PromptBuilder()
        self.formatter = MessageFormatter()
        self.health_monitor = HealthMonitor()

        # Discord bot instance.
        self._prefix = config.bot_prefix
        self._intents = _INTENTS
        self._search_enabled_global = config.search_enabled

        # discord.py Bot with command prefix.
        self._bot: Optional[commands.Bot] = None

        logger.info("DiscordBot initialized.")

    def _get_bot(self) -> commands.Bot:
        """Lazily create the discord.py Bot instance."""
        if self._bot is None:
            config = get_config()
            self._bot = commands.Bot(
                command_prefix=self._prefix,
                intents=self._intents,
                help_command=None,  # Custom help command.
            )
            self._register_events()
            self._register_commands()
            self._register_slash_commands()
        return self._bot

    def _register_events(self) -> None:
        """Register discord.py event handlers."""
        bot = self._get_bot()

        @bot.event
        async def on_ready():
            logger.info("Discord bot logged in as %s (ID: %s)", bot.user, bot.user.id)
            # Start the health monitor.
            await self.health_monitor.start()

        @bot.event
        async def on_message(message: discord.Message) -> None:
            # Ignore the bot's own messages.
            if message.author == bot.user:
                return

            # Ignore messages from bots (optional).
            if message.author.bot and message.author.id != bot.user.id:
                return

            # Check for prefix commands first.
            if message.content.startswith(self._prefix):
                await bot.process_commands(message)
                return

            # Check if the bot was mentioned.
            bot_mentioned = bot.user.mention in message.content
            is_dm = isinstance(message.channel, discord.DMChannel)

            if not bot_mentioned and not is_dm:
                return

            # Strip the bot mention from the content.
            if bot_mentioned:
                content = re.sub(
                    rf"{re.escape(bot.user.mention)}\s*",
                    "",
                    message.content,
                    count=1,
                ).strip()
            else:
                content = message.content

            # Handle command-like messages (e.g., "!ping" via mention).
            if content.startswith(self._prefix):
                message.content = content
                await bot.process_commands(message)
                return

            # Process as a chat message.
            await self._handle_chat(message, content)

        @bot.event
        async def on_command_error(
            ctx: commands.Context, error: commands.CommandError
        ) -> None:
            if isinstance(error, commands.MissingPermissions):
                await ctx.send("⚙️ *You lack the required permissions.*")
            elif isinstance(error, commands.NotOwner):
                await ctx.send("⚙️ *Owner-only command.*")
            elif isinstance(error, commands.CommandOnCooldown):
                await ctx.send(f"⏳ *Try again in {error.retry_after:.1f}s.*")
            else:
                logger.error("Command error in %s: %s", ctx.channel, error)
                await ctx.send("⚙️ *An unexpected error occurred.*")

    def _register_commands(self) -> None:
        """Register prefix commands (!ping, !status, etc.)."""
        bot = self._get_bot()

        @bot.command(name="ping")
        async def cmd_ping(ctx: commands.Context) -> None:
            latency = round(bot.latency * 1000)
            await ctx.send(f"🏓 *Pong! {latency}ms*")

        @bot.command(name="status")
        async def cmd_status(ctx: commands.Context) -> None:
            status = self.health_monitor.get_status()
            stats = self.conversation_manager.get_stats()
            response = self.formatter.format_status(
                llm_ok=status["llm_healthy"],
                search_ok=status["searxng_healthy"],
                model=status["model_name"],
                history_stats=stats,
            )
            await ctx.send(response)

        @bot.command(name="reset")
        async def cmd_reset(ctx: commands.Context) -> None:
            is_dm = isinstance(ctx.channel, discord.DMChannel)
            count = self.conversation_manager.clear_history(
                ctx.channel.id, is_dm=is_dm
            )
            await ctx.send(self.formatter.format_history_cleared(count))

        @bot.command(name="help")
        async def cmd_help(ctx: commands.Context) -> None:
            response = self.formatter.format_help(self._prefix)
            await ctx.send(response)

        @bot.command(name="search", aliases=["websearch"])
        async def cmd_search(
            ctx: commands.Context, action: str
        ) -> None:
            is_dm = isinstance(ctx.channel, discord.DMChannel)
            if action.lower() == "on":
                self.conversation_manager.set_search_enabled(
                    ctx.channel.id, True, is_dm=is_dm
                )
                await ctx.send(self.formatter.format_search_enabled())
            elif action.lower() == "off":
                self.conversation_manager.set_search_enabled(
                    ctx.channel.id, False, is_dm=is_dm
                )
                await ctx.send(self.formatter.format_search_disabled())
            else:
                await ctx.send(f"⚙️ *Usage: {self._prefix}search [on|off]*")

        @bot.command(name="history")
        async def cmd_history(ctx: commands.Context, action: str) -> None:
            if action.lower() == "clear":
                is_dm = isinstance(ctx.channel, discord.DMChannel)
                count = self.conversation_manager.clear_history(
                    ctx.channel.id, is_dm=is_dm
                )
                await ctx.send(self.formatter.format_history_cleared(count))
            else:
                await ctx.send(f"⚙️ *Usage: {self._prefix}history clear*")

    def _register_slash_commands(self) -> None:
        """Register slash commands (/ping, /status, etc.)."""
        bot = self._get_bot()

        @bot.tree.command(name="ping", description="Check if the bot is alive")
        async def slash_ping(interaction: discord.Interaction) -> None:
            latency = round(bot.latency * 1000)
            await interaction.response.send_message(f"🏓 *Pong! {latency}ms*")

        @bot.tree.command(name="status", description="Show backend status")
        async def slash_status(interaction: discord.Interaction) -> None:
            status = self.health_monitor.get_status()
            stats = self.conversation_manager.get_stats()
            response = self.formatter.format_status(
                llm_ok=status["llm_healthy"],
                search_ok=status["searxng_healthy"],
                model=status["model_name"],
                history_stats=stats,
            )
            await interaction.response.send_message(response)

        @bot.tree.command(name="reset", description="Clear conversation memory")
        async def slash_reset(interaction: discord.Interaction) -> None:
            is_dm = isinstance(interaction.channel, discord.DMChannel)
            count = self.conversation_manager.clear_history(
                interaction.channel.id, is_dm=is_dm
            )
            await interaction.response.send_message(
                self.formatter.format_history_cleared(count)
            )

        @bot.tree.command(name="help", description="Show available commands")
        async def slash_help(interaction: discord.Interaction) -> None:
            response = self.formatter.format_help(self._prefix)
            await interaction.response.send_message(response)

        @bot.tree.command(name="search", description="Toggle web search")
        @discord.app_commands.describe(action="on or off")
        async def slash_search(
            interaction: discord.Interaction, action: str
        ) -> None:
            is_dm = isinstance(interaction.channel, discord.DMChannel)
            if action.lower() == "on":
                self.conversation_manager.set_search_enabled(
                    interaction.channel.id, True, is_dm=is_dm
                )
                await interaction.response.send_message(
                    self.formatter.format_search_enabled()
                )
            elif action.lower() == "off":
                self.conversation_manager.set_search_enabled(
                    interaction.channel.id, False, is_dm=is_dm
                )
                await interaction.response.send_message(
                    self.formatter.format_search_disabled()
                )
            else:
                await interaction.response.send_message(
                    "⚙️ *Usage: /search [on|off]*"
                )

        @bot.tree.command(name="history", description="Manage conversation history")
        @discord.app_commands.describe(action="clear")
        async def slash_history(
            interaction: discord.Interaction, action: str
        ) -> None:
            if action.lower() == "clear":
                is_dm = isinstance(interaction.channel, discord.DMChannel)
                count = self.conversation_manager.clear_history(
                    interaction.channel.id, is_dm=is_dm
                )
                await interaction.response.send_message(
                    self.formatter.format_history_cleared(count)
                )
            else:
                await interaction.response.send_message(
                    "⚙️ *Usage: /history clear*"
                )

    async def _handle_chat(
        self, message: discord.Message, content: str
    ) -> None:
        """
        Process a chat message: check for images, run optional search,
        build the prompt, call the LLM, and send the response.
        """
        channel_id = message.channel.id
        is_dm = isinstance(message.channel, discord.DMChannel)

        # Check if search is enabled for this channel.
        search_allowed = self.conversation_manager.is_search_enabled(
            channel_id, is_dm
        )

        # Extract image URL from attachments.
        image_url = self._extract_image_url(message)

        # Determine if we should search.
        search_results = None
        if search_allowed and self.search_client.should_search(content):
            # Show typing indicator while searching.
            async with message.typing():
                search_results = await self.search_client.search(content)

        # Get conversation history.
        history = self.conversation_manager.get_messages(channel_id, is_dm)

        # Build the prompt.
        prompt_messages = self.prompt_builder.build(
            user_message=content,
            history=history,
            search_results=search_results,
        )

        # Store the user message in history.
        self.conversation_manager.add_message(
            channel_id, "user", content, is_dm
        )

        # Show typing indicator while the LLM processes.
        try:
            async with message.typing():
                response = await self.llm_client.chat(
                    messages=prompt_messages,
                    image_url=image_url,
                )
        except Exception as exc:
            logger.error("Error processing message in channel %d: %s", channel_id, exc)
            response = "⚙️ *The model returned an unexpected response.*"

        # Store the assistant response in history.
        self.conversation_manager.add_message(
            channel_id, "assistant", response, is_dm
        )

        # Split and send the response.
        chunks = self.formatter.format_response(response)
        reply_target = message

        for i, chunk in enumerate(chunks):
            if i == 0:
                await reply_target.reply(chunk, mention=False)
            else:
                await message.channel.send(chunk)

    def _extract_image_url(self, message: discord.Message) -> Optional[str]:
        """
        Extract a valid image URL from message attachments.

        Returns the URL of the first valid image attachment, or ``None``.
        """
        for attachment in message.attachments:
            if is_valid_image_url(attachment.url):
                logger.debug("Found image attachment: %s", attachment.url)
                return attachment.url
        return None

    async def start(self) -> None:
        """Start the Discord bot."""
        config = get_config()
        bot = self._get_bot()
        await bot.tree.sync()
        logger.info("Slash commands synced.")
        await bot.start(config.discord_token)

    async def close(self) -> None:
        """Gracefully shut down the bot and all sub-components."""
        logger.info("Shutting down DiscordBot...")
        await self.health_monitor.stop()
        await self.llm_client.close()
        await self.search_client.close()
        if self._bot:
            await self._bot.close()
        logger.info("DiscordBot shut down complete.")