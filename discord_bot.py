"""
DiscordBot — Main bot class.

Wires together injected dependencies and handles Discord events, commands,
conversation history, image extraction, and the web-search pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from config import Config
from conversation import ConversationManager
from formatter import MessageFormatter
from llm_client import LLMClient
from prompt_builder import PromptBuilder
from search_planner import SearchQueryPlanner, merge_search_results
from tools import SearchTool, ToolRegistry
from utils import is_valid_image_url

logger = logging.getLogger(__name__ + ".discord_bot")


class DiscordBot(commands.Bot):
    """Main Discord bot class with constructor-injected dependencies."""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        conversation_manager: ConversationManager,
        formatter: MessageFormatter,
        prompt_builder: PromptBuilder,
        search_query_planner: SearchQueryPlanner,
    ) -> None:
        self._config = config
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._conversation_manager = conversation_manager
        self._formatter = formatter
        self._prompt_builder = prompt_builder
        self._search_query_planner = search_query_planner

        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.dm_messages = True

        super().__init__(
            command_prefix=config.bot_prefix,
            intents=intents,
            help_command=None,
        )

        self._register_prefix_commands()

    async def setup_hook(self) -> None:
        """Register slash commands and sync them with Discord."""

        @self.tree.command(name="ping", description="Check if the bot is alive.")
        async def slash_ping(interaction: discord.Interaction) -> None:
            await interaction.response.send_message("🏓 Pong!", ephemeral=True)

        @self.tree.command(name="status", description="Show bot status.")
        async def slash_status(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(self._get_status_text(), ephemeral=True)

        @self.tree.command(name="reset", description="Reset conversation history for this channel.")
        async def slash_reset(interaction: discord.Interaction) -> None:
            self._clear_history(self._interaction_channel_key(interaction))
            await interaction.response.send_message("🔄 Conversation history reset.", ephemeral=True)

        @self.tree.command(name="help", description="Show available commands.")
        async def slash_help(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(
                self._formatter.format_help(self._config.bot_prefix),
                ephemeral=True,
            )

        @self.tree.command(name="search", description="Toggle web search for this channel.")
        @app_commands.describe(action="on or off")
        async def slash_search(interaction: discord.Interaction, action: str) -> None:
            normalized = action.lower().strip()
            if normalized not in {"on", "off"}:
                await interaction.response.send_message(
                    "Unknown action. Use `on` or `off`.",
                    ephemeral=True,
                )
                return
            self._set_search_enabled(self._interaction_channel_key(interaction), normalized == "on")
            await interaction.response.send_message(
                f"🔍 Search {'enabled' if normalized == 'on' else 'disabled'} for this channel.",
                ephemeral=True,
            )

        @self.tree.command(name="history", description="Manage conversation history.")
        @app_commands.describe(action="clear")
        async def slash_history(interaction: discord.Interaction, action: str) -> None:
            if action.lower().strip() != "clear":
                await interaction.response.send_message("Usage: `/history clear`", ephemeral=True)
                return
            self._clear_history(self._interaction_channel_key(interaction))
            await interaction.response.send_message("📜 History cleared.", ephemeral=True)

        try:
            synced = await self.tree.sync()
            logger.info("Synced %s slash commands.", len(synced))
        except Exception:
            logger.exception("Failed to sync slash commands.")

    async def on_ready(self) -> None:
        """Log readiness."""
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")

    async def on_message(self, message: discord.Message) -> None:
        """Handle DMs and mentions while letting discord.py handle prefix commands."""
        if message.author.bot:
            return

        if self.user and message.author.id == self.user.id:
            return

        if message.content.startswith(self._config.bot_prefix):
            await self.process_commands(message)
            return

        if isinstance(message.channel, discord.DMChannel):
            await self.handle_message(message, message.content.strip())
            return

        if not self._is_mentioned(message):
            return

        content = self._remove_bot_mentions(message.content)
        if content:
            await self.handle_message(message, content)

    async def handle_message(self, message: discord.Message, content: str) -> None:
        """Handle a user message from a DM or bot mention."""
        channel_id = self._channel_key(message)
        is_dm = isinstance(message.channel, discord.DMChannel)

        async with message.channel.typing():
            try:
                history = self._get_history(channel_id, is_dm=is_dm)
                search_results = await self._maybe_search(
                    user_message=content,
                    history=history,
                    channel_id=channel_id,
                    is_dm=is_dm,
                )
                prompt = self._prompt_builder.build(
                    user_message=content,
                    history=history,
                    search_results=search_results,
                )
                response = await self._llm_client.chat(
                    messages=prompt,
                    image_url=self._extract_image_url(message),
                    stream=False,
                )

                self._add_history(channel_id, "user", content, is_dm=is_dm)
                self._add_history(channel_id, "assistant", response, is_dm=is_dm)

            except Exception:
                logger.exception("Error handling message in channel %s", channel_id)
                response = self._format_error("Sorry, something went wrong while handling your message.")

        for chunk in self._formatter.format_response(response):
            await message.reply(
                chunk,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    async def _maybe_search(
        self,
        user_message: str,
        history: list[dict[str, str]],
        channel_id: str,
        is_dm: bool,
    ) -> list[dict[str, Any]]:
        """Plan and execute web searches when the search tool says search is useful."""
        if not user_message or not self._is_search_enabled(channel_id, is_dm=is_dm):
            return []

        tool = self._tool_registry.get("search")
        if not isinstance(tool, SearchTool):
            return []

        if not tool.should_search(user_message):
            return []

        plan = await self._search_query_planner.plan(
            user_message=user_message,
            history=history,
            current_date=date.today(),
        )

        queries = plan.queries[:3]
        if not queries:
            return []

        tasks = [self._tool_registry.execute("search", query, limit=5) for query in queries]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        grouped_results: list[list[dict[str, Any]]] = []
        result_counts: dict[str, int] = {}

        for query, result in zip(queries, raw_results, strict=False):
            if isinstance(result, Exception):
                logger.exception("Search failed for planned query: %s", query[:120])
                grouped_results.append([])
                result_counts[query] = 0
                continue

            data = result.data if result.success and isinstance(result.data, list) else []
            grouped_results.append(data)
            result_counts[query] = len(data)

        merged_results = merge_search_results(grouped_results, limit=8)
        logger.info(
            "Search pipeline completed",
            extra={
                "original_query": user_message[:300],
                "planned_queries": queries,
                "result_counts": result_counts,
                "fallback_used": plan.fallback_used,
                "combined_result_count": len(merged_results),
            },
        )
        return merged_results

    def _register_prefix_commands(self) -> None:
        """Register prefix commands using discord.py's normal command system."""

        @self.command(name="ping")
        async def ping_prefix(ctx: commands.Context) -> None:
            await ctx.reply("🏓 Pong!", mention_author=False)

        @self.command(name="status")
        async def status_prefix(ctx: commands.Context) -> None:
            for chunk in self._formatter.format_response(self._get_status_text()):
                await ctx.reply(
                    chunk,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )

        @self.command(name="reset")
        async def reset_prefix(ctx: commands.Context) -> None:
            self._clear_history(self._channel_key(ctx.message))
            await ctx.reply("🔄 Conversation history reset.", mention_author=False)

        @self.command(name="help")
        async def help_prefix(ctx: commands.Context) -> None:
            help_text = self._formatter.format_help(self._config.bot_prefix)
            for chunk in self._formatter.format_response(help_text):
                await ctx.reply(
                    chunk,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )

        @self.group(name="search", invoke_without_command=True)
        async def search_prefix(ctx: commands.Context) -> None:
            enabled = self._is_search_enabled(
                self._channel_key(ctx.message),
                is_dm=isinstance(ctx.channel, discord.DMChannel),
            )
            await ctx.reply(
                f"Search is currently {'enabled' if enabled else 'disabled'} for this channel. "
                f"Use `{self._config.bot_prefix}search on` or `{self._config.bot_prefix}search off`.",
                mention_author=False,
            )

        @search_prefix.command(name="on")
        async def search_on_prefix(ctx: commands.Context) -> None:
            self._set_search_enabled(self._channel_key(ctx.message), True)
            await ctx.reply("🔍 Search enabled for this channel.", mention_author=False)

        @search_prefix.command(name="off")
        async def search_off_prefix(ctx: commands.Context) -> None:
            self._set_search_enabled(self._channel_key(ctx.message), False)
            await ctx.reply("🔎 Search disabled for this channel.", mention_author=False)

        @self.group(name="history", invoke_without_command=True)
        async def history_prefix(ctx: commands.Context) -> None:
            await ctx.reply(f"Usage: `{self._config.bot_prefix}history clear`", mention_author=False)

        @history_prefix.command(name="clear")
        async def history_clear_prefix(ctx: commands.Context) -> None:
            self._clear_history(self._channel_key(ctx.message))
            await ctx.reply("📜 History cleared.", mention_author=False)

    def _is_mentioned(self, message: discord.Message) -> bool:
        """Return whether this bot was mentioned."""
        return bool(self.user and self.user in message.mentions)

    def _remove_bot_mentions(self, content: str) -> str:
        """Remove Discord mention tokens for this bot from message content."""
        if not self.user:
            return content.strip()
        return (
            content.replace(f"<@{self.user.id}>", " ")
            .replace(f"<@!{self.user.id}>", " ")
            .strip()
        )

    def _channel_key(self, message: discord.Message) -> str:
        """Return the conversation key for a message channel."""
        if isinstance(message.channel, discord.DMChannel):
            return f"dm:{message.author.id}"
        guild_id = message.guild.id if message.guild else "unknown"
        return f"guild:{guild_id}:channel:{message.channel.id}"

    def _interaction_channel_key(self, interaction: discord.Interaction) -> str:
        """Return the conversation key for a slash-command interaction."""
        if interaction.guild_id is None:
            return f"dm:{interaction.user.id}"
        return f"guild:{interaction.guild_id}:channel:{interaction.channel_id}"

    def _get_history(self, channel_id: str, is_dm: bool) -> list[dict[str, str]]:
        try:
            return self._conversation_manager.get_messages(channel_id)
        except TypeError:
            return self._conversation_manager.get_messages(channel_id, is_dm=is_dm)

    def _add_history(self, channel_id: str, role: str, content: str, is_dm: bool) -> None:
        try:
            self._conversation_manager.add_message(channel_id, role, content)
        except TypeError:
            self._conversation_manager.add_message(channel_id, role, content, is_dm=is_dm)

    def _clear_history(self, channel_id: str) -> None:
        self._conversation_manager.clear_history(channel_id)

    def _is_search_enabled(self, channel_id: str, is_dm: bool) -> bool:
        if not self._config.search_enabled:
            return False

        if hasattr(self._conversation_manager, "is_search_enabled"):
            try:
                return bool(self._conversation_manager.is_search_enabled(channel_id))
            except TypeError:
                return bool(self._conversation_manager.is_search_enabled(channel_id, is_dm=is_dm))

        return True

    def _set_search_enabled(self, channel_id: str, enabled: bool) -> None:
        if hasattr(self._conversation_manager, "set_search_enabled"):
            self._conversation_manager.set_search_enabled(channel_id, enabled)

    def _extract_image_url(self, message: discord.Message) -> str | None:
        """Extract the first valid image URL from attachments, embeds, or text."""
        for attachment in message.attachments:
            if is_valid_image_url(attachment.url):
                return attachment.url

        for embed in message.embeds:
            if embed.image and embed.image.url and is_valid_image_url(embed.image.url):
                return embed.image.url

        import re

        urls = re.findall(r"https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp)\b", message.content)
        for url in urls:
            if is_valid_image_url(url):
                return url

        return None

    def _get_status_text(self) -> str:
        """Build a status text showing backend/tool availability."""
        history_stats = self._conversation_manager.get_stats()
        return self._formatter.format_status(
            llm_ok=True,
            search_ok=self._tool_registry.has("search"),
            model=self._config.llm_model,
            history_stats=history_stats,
        )

    def _format_error(self, message: str) -> str:
        if hasattr(self._formatter, "format_error"):
            return self._formatter.format_error(message)
        return message

    async def close(self) -> None:
        """Close Discord and all injected HTTP clients."""
        logger.info("Closing bot and HTTP clients...")
        await self._llm_client.close()
        await self._tool_registry.close()
        await super().close()
        logger.info("Bot and HTTP clients closed.")