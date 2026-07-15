"""
Message formatter — handles Discord-specific text formatting,
chunking, and embed construction.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from utils import split_discord_message, escape_markdown

logger = logging.getLogger(__name__)

_DISCORD_MAX_LENGTH = 2000
_DISCORD_EMBED_DESC_LENGTH = 2000
_DISCORD_EMBED_FIELD_VALUE_LENGTH = 1024


class MessageFormatter:
    """
    Formats bot responses for Discord, handling length limits,
    embeds, and markdown escaping.
    """

    def __init__(self) -> None:
        logger.info("MessageFormatter initialized.")

    def format_response(self, text: str) -> List[str]:
        """
        Split a response into chunks that fit within Discord's message
        length limit.
        """
        return split_discord_message(text, _DISCORD_MAX_LENGTH)

    def format_help(self, prefix: str) -> str:
        """Build a help message showing available commands."""
        lines = [
            "🤖 **Bot Commands**",
            "",
            f"**{prefix}ping** — Check if the bot is alive",
            f"**{prefix}status** — Show backend status",
            f"**{prefix}reset** — Clear conversation memory for this channel",
            f"**{prefix}help** — Show this message",
            f"**{prefix}search on** — Enable automatic web search",
            f"**{prefix}search off** — Disable automatic web search",
            f"**{prefix}history clear** — Same as reset",
            "",
            "💡 *Mention the bot or send a DM to start chatting!*",
            "🖼️ *Attach an image for vision understanding.*",
        ]
        return "\n".join(lines)

    def format_status(self, llm_ok: bool, search_ok: bool, model: str, history_stats: dict) -> str:
        """Build a status message summarizing backend health."""
        llm_icon = "✅" if llm_ok else "⚙️"
        search_icon = "✅" if search_ok else "⚙️"

        lines = [
            "📊 **Bot Status**",
            "",
            f"{llm_icon} **LLM Backend**: {'Connected' if llm_ok else 'Checking...'}",
            f"🧠 **Model**: {model}",
            f"{search_icon} **SearXNG**: {'Connected' if search_ok else 'Checking...'}",
            "",
            f"📝 **Active Conversations**: {history_stats.get('channels_tracked', 0)}",
            f"💬 **Total Messages**: {history_stats.get('total_entries', 0)}",
        ]
        return "\n".join(lines)

    def format_search_disabled(self) -> str:
        return "🔍 *Web search has been disabled for this channel.*"

    def format_search_enabled(self) -> str:
        return "🔍 *Web search has been enabled for this channel.*"

    def format_history_cleared(self, count: int) -> str:
        return f"🧹 *Cleared {count} message(s) from conversation memory.*"

    def format_error(self, friendly_message: str) -> str:
        return f"⚙️ {friendly_message}"