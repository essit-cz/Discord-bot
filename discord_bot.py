"""
DiscordBot — Main bot class.

Wires together all injected dependencies and handles Discord events.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

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
            chunks = self._formatter.chunk_message(status_text, _MESSAGE_LIMIT)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await interaction.response.send_message(chunk)
                else:
                    await interaction.followup.send(chunk)

        @self.tree.command(name="reset", description="Reset conversation history for this channel")
        async def reset(interaction: discord.Interaction) -> None:
            channel_id = interaction.channel_id
            self._conversation_manager.clear(channel_id)
            await interaction.response.send_message("🔄 Conversation history reset.", ephemeral=True)

        @self.tree.command(name="help", description="Show available commands")
        async def help_cmd(interaction: discord.Interaction) -> None:
            help_text = self._formatter.format_help()
            chunks = self._formatter.chunk_message(help_text, _MESSAGE_LIMIT)
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
                self._conversation_manager.clear(channel_id)
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
            chunks = self._formatter.chunk_message(status_text, _MESSAGE_LIMIT)
            for chunk in chunks:
                await ctx.send(chunk)

        @self.command(name="reset")
        async def reset_prefix(ctx: commands.Context) -> None:
            self._conversation_manager.clear(ctx.channel.id)
            await ctx.send("🔄 Conversation history reset.")

        @self.command(name="help")
        async def help_prefix(ctx: commands.Context) -> None:
            help_text = self._formatter.format_help()
            chunks = self._formatter.chunk_message(help_text, _MESSAGE_LIMIT)
            for chunk in chunks:
                await ctx.send(chunk)

        @self.command(name="search")
        @commands.argument("action", nargs="?", default=None)
        async def search_prefix(ctx: commands.Context, action: Optional[str]) -> None:
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
        @commands.argument("action", nargs="?", default=None)
        async def history_prefix(ctx: commands.Context, action: Optional[str]) -> None:
            if action and action.lower() == "clear":
                self._conversation_manager.clear(ctx.channel.id)
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

        if message.content.startswith("/"):
            return

        channel_id = message.channel.id

        if self._is_command(message.content):
            await self.handle_command(message)
            return

        await self.handle_message(message)

    def _is_command(self, content: str) -> bool:
        """Check if a message is a prefix command."""
        return content.startswith("!")

    async def handle_command(self, message: discord.Message) -> None:
        """Handle a prefix command message."""
        content = message.content
        command = content.split()[0].lower()

        if command == "!ping":
            latency = round(self.latency * 1000)
            await message.reply(f"🏓 Pong! {latency}ms")
        elif command == "!status":
            status_text = self._get_status_text()
            chunks = self._formatter.chunk_message(status_text, _MESSAGE_LIMIT)
            for chunk in chunks:
                await message.reply(chunk)
        elif command == "!reset":
            self._conversation_manager.clear(message.channel.id)
            await message.reply("🔄 Conversation history reset.")
        elif command == "!help":
            help_text = self._formatter.format_help()
            chunks = self._formatter.chunk_message(help_text, _MESSAGE_LIMIT)
            for chunk in chunks:
                await message.reply(chunk)
        elif command == "!search":
            parts = content.split()
            if len(parts) < 2:
                enabled = self._search_enabled.get(message.channel.id, self._config.search_enabled)
                await message.reply(f"Search is currently {'enabled' if enabled else 'disabled'} for this channel.")
                return
            action = parts[1].lower()
            if action == "on":
                self._search_enabled[message.channel.id] = True
                await message.reply("🔍 Search enabled for this channel.")
            elif action == "off":
                self._search_enabled[message.channel.id] = False
                await message.reply("🔎 Search disabled for this channel.")
            else:
                await message.reply(f"Unknown action: '{action}'. Use 'on' or 'off'.")
        elif command == "!history":
            parts = content.split()
            if len(parts) >= 2 and parts[1].lower() == "clear":
                self._conversation_manager.clear(message.channel.id)
                await message.reply("📜 History cleared.")
            else:
                await message.reply("Usage: !history clear")
        else:
            await message.reply(f"Unknown command: '{command}'. Type `!help` for available commands.")

    async def handle_message(self, message: discord.Message) -> None:
        """Handle a regular message (mention or DM)."""
        channel_id = message.channel.id

        is_mention = self.user and self.user.mentions in [m.id for m in message.mentions]
        is_dm = isinstance(message.channel, discord.DMChannel)

        if not is_mention and not is_dm:
            return

        content = message.content
        if is_mention:
            content = re.sub(rf"<@!?{self.user.id}>", "", content).strip()

        if not content:
            return

        await message.typing()

        search_enabled = self._search_enabled.get(channel_id, self._config.search_enabled)

        try:
            response = await self._process_message(content, channel_id, search_enabled, message)
            if response:
                chunks = self._formatter.chunk_message(response, _MESSAGE_LIMIT)
                for chunk in chunks:
                    await message.reply(chunk)
        except Exception as exc:
            logger.exception("Error processing message in channel %s", channel_id)
            await message.reply("⚙️ Processing your message...")

    async def _process_message(
        self,
        content: str,
        channel_id: int,
        search_enabled: bool,
        message: discord.Message,
    ) -> Optional[str]:
        """Process a message and return the LLM response."""
        history = self._conversation_manager.get_history(channel_id)

        search_context = ""
        if search_enabled:
            search_result = await self._tool_registry.execute("search", content)
            if search_result.success and search_result.data:
                search_context = self._format_search_context(search_result.data)

        prompt = self._prompt_builder.build_prompt(
            content=content,
            history=history,
            search_context=search_context,
        )

        images = self._extract_images(message)

        response = await self._llm_client.chat(
            messages=prompt,
            images=images,
            stream=False,
        )

        if response.success:
            self._conversation_manager.add_message(channel_id, "user", content)
            self._conversation_manager.add_message(channel_id, "assistant", response.data)
            return response.data
        else:
            return f"⚙️ {response.message}"

    def _format_search_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a context string."""
        if not results:
            return ""

        lines = ["\n--- Search Results ---\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            lines.append(f"**{i}. {title}**\n{url}\n{snippet}\n")

        lines.append("--- End Search Results ---\n")
        return "\n".join(lines)

    def _extract_images(self, message: discord.Message) -> List[str]:
        """Extract image URLs from a message."""
        images = []
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                images.append(attachment.url)
        for embed in message.embeds:
            if embed.image:
                images.append(embed.image.url)
        return images

    def _get_status_text(self) -> str:
        """Get the current bot status."""
        tools_list = self._tool_registry.list_tools()
        tools_str = ", ".join(t.name for t in tools_list) if tools_list else "none"

        return (
            f"**Bot Status**\n"
            f"🟢 **Uptime:** Running\n"
            f"🤖 **LLM:** Connected\n"
            f"🛠️ **Tools:** {tools_str}\n"
            f"💬 **Channels with history:** {self._conversation_manager.get_active_channel_count()}"
        )