"""
DiscordBot — Main bot class.

Wires together all injected dependencies and handles Discord events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import discord
from discord.ext import commands

from config import Config
from conversation import ConversationManager
from formatter import MessageFormatter
from llm_client import LLMClient
from prompt_builder import PromptBuilder
from tools import ToolRegistry

logger = logging.getLogger(__name__ + ".discord_bot")

# Discord message length limits
_MESSAGE_LIMIT = 2000


def _get_prefix(bot: commands.Bot, message: discord.Message) -> str:
    """Dynamic prefix: '!' in guilds, '!' in DMs."""
    return "!"


class DiscordBot(commands.Bot):
    """Main Discord bot class.

    All dependencies are injected through the constructor.
    """
    _MESSAGE_LIMIT = 2000

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        conversation_manager: ConversationManager,
        formatter: MessageFormatter,
        prompt_builder: PromptBuilder,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.dm_messages = True

        super().__init__(
            command_prefix=_get_prefix,
            intents=intents,
            help_command=None,
        )

        self._config = config
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._conversation_manager = conversation_manager
        self._formatter = formatter
        self._prompt_builder = prompt_builder

        self._search_enabled: Dict[int, bool] = {}

        self._register_commands()

    # ------------------------------------------------------------------
    # Command Registration
    # ------------------------------------------------------------------

    def _register_commands(self) -> None:
        """Register all slash and prefix commands."""
        self._register_slash_commands()
        self._register_prefix_commands()

    def _register_slash_commands(self) -> None:
        """Register all slash commands."""
        @self.tree.command(name="ping", description="Check if the bot is alive")
        async def ping(interaction: discord.Interaction) -> None:
            latency = round(self.latency * 1000)
            await interaction.response.send_message(f"🏓 Pong! {latency}ms", ephemeral=True)

        @self.tree.command(name="status", description="Show bot status")
        async def status(interaction: discord.Interaction) -> None:
            status_text = self._get_status_text()
            chunks = self._formatter.format_response(status_text)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await interaction.response.send_message(chunk)
                else:
                    await interaction.followup.send(chunk)

        @self.tree.command(name="reset", description="Reset conversation history for this channel")
        async def reset(interaction: discord.Interaction) -> None:
            channel_id = interaction.channel_id
            self._conversation_manager.clear_history(channel_id)
            await interaction.response.send_message("🔄 Conversation history reset.", ephemeral=True)

        @self.tree.command(name="help", description="Show available commands")
        async def help_cmd(interaction: discord.Interaction) -> None:
            help_text = self._formatter.format_help("!")
            chunks = self._formatter.format_response(help_text)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await interaction.response.send_message(chunk)
                else:
                    await interaction.followup.send(chunk)

        @self.tree.command(name="search", description="Toggle search for this channel")
        @discord.app_commands.describe(action="on or off")
        async def search_toggle(interaction: discord.Interaction, action: str) -> None:
            channel_id = interaction.channel_id
            action_lower = action.lower()
            if action_lower == "on":
                self._search_enabled[channel_id] = True
                await interaction.response.send_message("🔍 Search enabled for this channel.", ephemeral=True)
            elif action_lower == "off":
                self._search_enabled[channel_id] = False
                await interaction.response.send_message("🔎 Search disabled for this channel.", ephemeral=True)
            else:
                await interaction.response.send_message(f"Unknown action: '{action}'. Use 'on' or 'off'.", ephemeral=True)

        @self.tree.command(name="history", description="Manage conversation history")
        @discord.app_commands.describe(action="clear")
        async def history_cmd(interaction: discord.Interaction, action: str) -> None:
            if action.lower() == "clear":
                channel_id = interaction.channel_id
                self._conversation_manager.clear_history(channel_id)
                await interaction.response.send_message("📜 History cleared.", ephemeral=True)
            else:
                await interaction.response.send_message(f"Unknown history action: '{action}'. Use 'clear'.", ephemeral=True)

    def _register_prefix_commands(self) -> None:
        """Register all prefix commands."""
        @self.command(name="ping")
        async def ping_prefix(ctx: commands.Context) -> None:
            latency = round(self.latency * 1000)
            await ctx.send(f"🏓 Pong! {latency}ms")

        @self.command(name="status")
        async def status_prefix(ctx: commands.Context) -> None:
            status_text = self._get_status_text()
            chunks = self._formatter.format_response(status_text)
            for chunk in chunks:
                await ctx.send(chunk)

        @self.command(name="reset")
        async def reset_prefix(ctx: commands.Context) -> None:
            self._conversation_manager.clear_history(ctx.channel.id)
            await ctx.send("🔄 Conversation history reset.")

        @self.command(name="help")
        async def help_prefix(ctx: commands.Context) -> None:
            help_text = self._formatter.format_help("!")
            chunks = self._formatter.format_response(help_text)
            for chunk in chunks:
                await ctx.send(chunk)

        @self.command(name="search")
        async def search_prefix(ctx: commands.Context, action: Optional[str] = None) -> None:
            if action is None:
                enabled = self._search_enabled.get(ctx.channel.id, self._config.search_enabled)
                await ctx.send(f"Search is currently {'enabled' if enabled else 'disabled'} for this channel.")
                return
            action_lower = action.lower()
            if action_lower == "on":
                self._search_enabled[ctx.channel.id] = True
                await ctx.send("🔍 Search enabled for this channel.")
            elif action_lower == "off":
                self._search_enabled[ctx.channel.id] = False
                await ctx.send("🔎 Search disabled for this channel.")
            else:
                await ctx.send(f"Unknown action: '{action}'. Use 'on' or 'off'.")

        @self.command(name="history")
        async def history_prefix(ctx: commands.Context, action: Optional[str] = None) -> None:
            if action and action.lower() == "clear":
                self._conversation_manager.clear_history(ctx.channel.id)
                await ctx.send("📜 History cleared.")
            else:
                await ctx.send("Usage: !history clear")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def setup_hook(self) -> None:
        """Called when the bot is ready (after login)."""
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)
        logger.info("Syncing commands...")
        await self.tree.sync()
        logger.info("Commands synced. Bot is ready.")

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages."""
        if message.author.bot:
            return

        if message.author.id == self.user.id:
            return

        # Let discord.py handle registered prefix commands.
        await self.process_commands(message)

        # Skip slash commands (handled by interaction handler).
        if message.content.startswith("/"):
            return

        # Check if the bot was mentioned.
        if not self._is_mentioned(message):
            return

        # Strip the mention from the content.
        content = message.content
        if self.user.mention in content:
            content = content.replace(self.user.mention, "", 1).strip()

        if not content:
            return

        await self.handle_message(message, content)

    def _is_mentioned(self, message: discord.Message) -> bool:
        """Check if the bot is mentioned in the message."""
        if isinstance(message.channel, discord.DMChannel):
            return True
        return self.user.mention in message.content

    async def handle_message(self, message: discord.Message, content: str) -> None:
        """Handle a user message (mention or DM)."""
        channel_id = message.channel.id
        is_dm = isinstance(message.channel, discord.DMChannel)

        # Check if search is enabled for this channel.
        search_enabled = self._search_enabled.get(channel_id, self._config.search_enabled)
        if not self._conversation_manager.is_search_enabled(channel_id):
            search_enabled = False

        # Show typing indicator.
        async with message.channel.typing():
            try:
                # Get conversation history.
                history = self._conversation_manager.get_messages(channel_id, is_dm=is_dm)

                # Determine if we should search.
                search_results = None
                if search_enabled:
                    search_results = await self._perform_search(content)

                # Build the prompt.
                prompt = self._prompt_builder.build(content, history, search_results)

                # Check for images in the message.
                image_url = self._extract_image_url(message)

                # Get the LLM response.
                response = await self._llm_client.chat(prompt, image_url=image_url)

                # Store the exchange in conversation history.
                self._conversation_manager.add_message(channel_id, "user", content, is_dm=is_dm)
                self._conversation_manager.add_message(channel_id, "assistant", response, is_dm=is_dm)

                # Format and send the response.
                chunks = self._formatter.format_response(response)
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        await message.reply(chunk)
                    else:
                        await message.channel.send(chunk)

            except Exception as exc:
                logger.exception("Error handling message in channel %d", channel_id)
                friendly_msg = self._formatter.format_error(str(exc))
                await message.reply(friendly_msg)

    async def _perform_search(self, query: str) -> Optional[List[dict]]:
        """Perform a web search if the tool is available."""
        try:
            result = await self._tool_registry.execute("search", query)
            if result.success and result.data:
                return result.data
        except Exception as exc:
            logger.warning("Search failed: %s", exc)
        return None

    def _extract_image_url(self, message: discord.Message) -> Optional[str]:
        """Extract the first valid image URL from a Discord message."""
        from utils import is_valid_image_url

        # Check attachments first.
        for attachment in message.attachments:
            if is_valid_image_url(attachment.url):
                return attachment.url

        # Check embeds.
        for embed in message.embeds:
            if embed.image and is_valid_image_url(embed.image.url):
                return embed.image.url

        # Check content for image URLs.
        import re
        urls = re.findall(r"https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp)\b", message.content)
        for url in urls:
            if is_valid_image_url(url):
                return url

        return None

    def _get_status_text(self) -> str:
        """Build a status text showing backend health."""
        history_stats = self._conversation_manager.get_stats()
        return self._formatter.format_status(
            llm_ok=True,
            search_ok=self._tool_registry.has("search"),
            model=self._config.llm_model,
            history_stats=history_stats,
        )

    async def close(self) -> None:
        """Close the bot and all injected HTTP clients."""
        logger.info("Closing bot and HTTP clients...")
        await self._llm_client.close()
        await self._tool_registry.close()
        await super().close()
        logger.info("Bot and HTTP clients closed.")
