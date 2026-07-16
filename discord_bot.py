"""
Discord bot orchestration.

This module contains Discord-specific event handling and command wiring. Shared
dependencies are injected by main.py.
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

logger = logging.getLogger(__name__)


class DiscordBot(commands.Bot):
    """Discord AI bot with injected dependencies."""

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
            command_prefix=config.command_prefix,
            intents=intents,
            help_command=None,
        )

        self._register_prefix_commands()

    async def setup_hook(self) -> None:
        """Register slash commands."""
        self.tree.add_command(self._slash_ping)
        self.tree.add_command(self._slash_status)
        self.tree.add_command(self._slash_reset)
        self.tree.add_command(self._slash_help)
        self.tree.add_command(self._slash_search)
        self.tree.add_command(self._slash_history)

        try:
            synced = await self.tree.sync()
            logger.info("Synced %s slash commands", len(synced))
        except Exception:
            logger.exception("Failed to sync slash commands")

    async def close(self) -> None:
        """Close Discord and all shared HTTP clients."""
        await self._tool_registry.close()
        await self._llm_client.close()
        await super().close()

    async def on_ready(self) -> None:
        """Log readiness."""
        logger.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "unknown")

    async def on_message(self, message: discord.Message) -> None:
        """Handle DMs and guild mentions while preserving prefix commands."""
        if message.author.bot:
            return

        if message.content.startswith(str(self._config.command_prefix)):
            await self.process_commands(message)
            return

        if isinstance(message.channel, discord.DMChannel):
            await self._handle_user_message(message, message.content)
            return

        if self.user and self.user in message.mentions:
            content = self._remove_bot_mentions(message.content)
            await self._handle_user_message(message, content)

    async def _handle_user_message(self, message: discord.Message, user_content: str) -> None:
        channel_id = self._channel_key(message)
        content = user_content.strip()
        if not content and not message.attachments:
            return

        image_url = self._extract_image_url(message)
        history = self._conversation_manager.get_messages(channel_id)

        async with message.channel.typing():
            search_results = await self._maybe_search(
                user_message=content,
                history=history,
                channel_id=channel_id,
            )

            prompt_messages = self._prompt_builder.build(
                user_message=content,
                history=history,
                search_results=search_results,
            )
            response = await self._llm_client.chat(
                messages=prompt_messages,
                image_url=image_url,
                stream=False,
            )

        self._conversation_manager.add_message(channel_id, "user", content)
        self._conversation_manager.add_message(channel_id, "assistant", response)

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
    ) -> list[dict[str, Any]]:
        if not user_message:
            return []

        if hasattr(self._conversation_manager, "is_search_enabled"):
            if not self._conversation_manager.is_search_enabled(channel_id):
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
        tasks = [self._tool_registry.execute("search", query, limit=5) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        groups: list[list[dict[str, Any]]] = []
        result_counts: dict[str, int] = {}

        for query, result in zip(queries, results, strict=False):
            if isinstance(result, Exception):
                logger.exception("Planned search failed for query: %s", query[:120])
                result_counts[query] = 0
                groups.append([])
                continue

            data = result.data if result.success and isinstance(result.data, list) else []
            result_counts[query] = len(data)
            groups.append(data)

        merged = merge_search_results(groups, limit=8)
        logger.info(
            "Search pipeline completed",
            extra={
                "original_query": user_message[:300],
                "planned_queries": queries,
                "result_counts": result_counts,
                "fallback_used": plan.fallback_used,
                "combined_result_count": len(merged),
            },
        )
        return merged

    def _register_prefix_commands(self) -> None:
        @self.command(name="ping")
        async def ping_command(ctx: commands.Context) -> None:
            await ctx.reply("Pong!", mention_author=False)

        @self.command(name="status")
        async def status_command(ctx: commands.Context) -> None:
            await ctx.reply(
                self._format_status(),
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

        @self.command(name="reset")
        async def reset_command(ctx: commands.Context) -> None:
            self._conversation_manager.clear_history(self._channel_key(ctx.message))
            await ctx.reply("Conversation history cleared.", mention_author=False)

        @self.command(name="help")
        async def help_command(ctx: commands.Context) -> None:
            await ctx.reply(
                self._formatter.format_help(str(self._config.command_prefix)),
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

        @self.group(name="search", invoke_without_command=True)
        async def search_group(ctx: commands.Context) -> None:
            await ctx.reply("Usage: !search on or !search off", mention_author=False)

        @search_group.command(name="on")
        async def search_on(ctx: commands.Context) -> None:
            self._set_search_enabled(self._channel_key(ctx.message), True)
            await ctx.reply("Web search enabled for this conversation.", mention_author=False)

        @search_group.command(name="off")
        async def search_off(ctx: commands.Context) -> None:
            self._set_search_enabled(self._channel_key(ctx.message), False)
            await ctx.reply("Web search disabled for this conversation.", mention_author=False)

        @self.group(name="history", invoke_without_command=True)
        async def history_group(ctx: commands.Context) -> None:
            await ctx.reply("Usage: !history clear", mention_author=False)

        @history_group.command(name="clear")
        async def history_clear(ctx: commands.Context) -> None:
            self._conversation_manager.clear_history(self._channel_key(ctx.message))
            await ctx.reply("Conversation history cleared.", mention_author=False)

    @app_commands.command(name="ping", description="Check whether the bot is alive.")
    async def _slash_ping(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message("Pong!", ephemeral=True)

    @app_commands.command(name="status", description="Show bot status.")
    async def _slash_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(self._format_status(), ephemeral=True)

    @app_commands.command(name="reset", description="Clear this conversation history.")
    async def _slash_reset(self, interaction: discord.Interaction) -> None:
        channel_id = self._interaction_channel_key(interaction)
        self._conversation_manager.clear_history(channel_id)
        await interaction.response.send_message("Conversation history cleared.", ephemeral=True)

    @app_commands.command(name="help", description="Show bot help.")
    async def _slash_help(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            self._formatter.format_help(str(self._config.command_prefix)),
            ephemeral=True,
        )

    @app_commands.command(name="search", description="Enable or disable web search.")
    @app_commands.describe(action="on or off")
    async def _slash_search(self, interaction: discord.Interaction, action: str) -> None:
        normalized = action.lower().strip()
        if normalized not in {"on", "off"}:
            await interaction.response.send_message("Use /search on or /search off.", ephemeral=True)
            return

        self._set_search_enabled(self._interaction_channel_key(interaction), normalized == "on")
        await interaction.response.send_message(
            f"Web search {'enabled' if normalized == 'on' else 'disabled'} for this conversation.",
            ephemeral=True,
        )

    @app_commands.command(name="history", description="Manage conversation history.")
    @app_commands.describe(action="clear")
    async def _slash_history(self, interaction: discord.Interaction, action: str) -> None:
        if action.lower().strip() != "clear":
            await interaction.response.send_message("Use /history clear.", ephemeral=True)
            return

        self._conversation_manager.clear_history(self._interaction_channel_key(interaction))
        await interaction.response.send_message("Conversation history cleared.", ephemeral=True)

    def _format_status(self) -> str:
        stats = self._conversation_manager.get_stats()
        tools = ", ".join(tool.name for tool in self._tool_registry.list_tools()) or "none"
        return (
            "**Status**\n"
            f"- Model: `{self._config.model_name}`\n"
            f"- Tools: `{tools}`\n"
            f"- Conversations: `{stats}`"
        )

    def _channel_key(self, message: discord.Message) -> str:
        if isinstance(message.channel, discord.DMChannel):
            return f"dm:{message.author.id}"
        return f"guild:{message.guild.id if message.guild else 'unknown'}:channel:{message.channel.id}"

    def _interaction_channel_key(self, interaction: discord.Interaction) -> str:
        if interaction.guild_id is None:
            return f"dm:{interaction.user.id}"
        return f"guild:{interaction.guild_id}:channel:{interaction.channel_id}"

    def _remove_bot_mentions(self, content: str) -> str:
        if not self.user:
            return content
        return (
            content.replace(f"<@{self.user.id}>", "")
            .replace(f"<@!{self.user.id}>", "")
            .strip()
        )

    def _extract_image_url(self, message: discord.Message) -> str | None:
        for attachment in message.attachments:
            url = attachment.url
            content_type = attachment.content_type or ""
            if content_type.startswith("image/") and is_valid_image_url(url):
                return url
            if is_valid_image_url(url):
                return url
        return None

    def _set_search_enabled(self, channel_id: str, enabled: bool) -> None:
        if hasattr(self._conversation_manager, "set_search_enabled"):
            self._conversation_manager.set_search_enabled(channel_id, enabled)