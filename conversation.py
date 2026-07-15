"""
Conversation memory manager.

Stores per-channel message history with automatic TTL expiry and
token-budget trimming.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from config import Config
from utils import estimate_tokens, now_timestamp

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """A single user↔assistant exchange."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=now_timestamp)


@dataclass
class ChannelHistory:
    """History for one Discord channel (or DM)."""
    entries: List[HistoryEntry] = field(default_factory=list)
    last_accessed: float = field(default_factory=now_timestamp)
    search_enabled: bool = True

    def add(self, role: str, content: str) -> None:
        """Append a message entry."""
        self.entries.append(HistoryEntry(role=role, content=content))
        self.last_accessed = now_timestamp()

    def clear(self) -> int:
        """Remove all entries and return the count removed."""
        count = len(self.entries)
        self.entries.clear()
        self.last_accessed = now_timestamp()
        return count

    def get_messages(self, token_budget: int) -> List[dict]:
        """
        Return history as a list of ``{"role": ..., "content": ...}`` dicts,
        trimmed to the given token budget.
        """
        messages = [
            {"role": e.role, "content": e.content}
            for e in self.entries
        ]

        # Trim to token budget.
        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        if total_tokens > token_budget:
            # Keep most recent messages.
            trimmed: List[dict] = []
            running = 0
            for m in reversed(messages):
                t = estimate_tokens(m["content"])
                if running + t > token_budget and trimmed:
                    break
                running += t
                trimmed.insert(0, m)
            messages = trimmed

        return messages

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if this channel's history has exceeded the TTL."""
        return (now_timestamp() - self.last_accessed) > ttl_seconds


class ConversationManager:
    """
    Manages conversation history keyed by Discord channel ID.

    Each guild channel and each DM gets its own independent history.
    Histories expire automatically after the configured TTL.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._histories: Dict[str, ChannelHistory] = {}
        logger.info("ConversationManager initialized (ttl=%ds, budget=%d tokens).",
                     config.history_ttl_seconds, config.history_token_budget)

    def _get_key(self, channel_id: int, is_dm: bool) -> str:
        """Build a unique history key for a channel."""
        prefix = "dm" if is_dm else "guild"
        return f"{prefix}:{channel_id}"

    def _get_or_create(self, channel_id: int, is_dm: bool) -> ChannelHistory:
        """Return (or create) the history for a channel."""
        key = self._get_key(channel_id, is_dm)
        if key not in self._histories:
            self._histories[key] = ChannelHistory()
        return self._histories[key]

    def add_message(self, channel_id: int, role: str, content: str, is_dm: bool = False) -> None:
        """Store a message in the channel's history."""
        history = self._get_or_create(channel_id, is_dm)
        history.add(role, content)

    def get_messages(self, channel_id: int, is_dm: bool = False) -> List[dict]:
        """
        Retrieve conversation messages for a channel, trimmed to the
        token budget.
        """
        history = self._get_or_create(channel_id, is_dm)
        return history.get_messages(self._config.history_token_budget)

    def clear_history(self, channel_id: int, is_dm: bool = False) -> int:
        """Clear the history for a channel. Returns the number of entries removed."""
        history = self._get_or_create(channel_id, is_dm)
        count = history.clear()
        logger.info("Cleared %d history entries for channel %d (dm=%s).", count, channel_id, is_dm)
        return count

    def set_search_enabled(self, channel_id: int, enabled: bool, is_dm: bool = False) -> None:
        """Toggle web search for a channel."""
        history = self._get_or_create(channel_id, is_dm)
        history.search_enabled = enabled
        logger.info("Search %s for channel %d (dm=%s).", "enabled" if enabled else "disabled", channel_id, is_dm)

    def is_search_enabled(self, channel_id: int, is_dm: bool = False) -> bool:
        """Check if web search is enabled for a channel."""
        history = self._get_or_create(channel_id, is_dm)
        return history.search_enabled

    def cleanup_expired(self) -> int:
        """
        Remove expired channel histories.

        Returns the number of channels cleaned up.
        """
        expired_keys = [
            key for key, h in self._histories.items()
            if h.is_expired(self._config.history_ttl_seconds)
        ]
        for key in expired_keys:
            del self._histories[key]
        if expired_keys:
            logger.info("Cleaned up %d expired channel histories.", len(expired_keys))
        return len(expired_keys)

    def get_stats(self) -> dict:
        """Return summary statistics about stored histories."""
        total_entries = sum(len(h.entries) for h in self._histories.values())
        return {
            "channels_tracked": len(self._histories),
            "total_entries": total_entries,
        }
