"""Tests for public ConversationManager behavior."""

from __future__ import annotations

from config import Config
from conversation import ConversationManager


def make_config() -> Config:
    return Config(
        discord_token="test-token",
        history_ttl_seconds=3600,
        history_token_budget=32,
    )


def test_histories_are_isolated_by_channel() -> None:
    manager = ConversationManager(make_config())

    manager.add_message("guild:1:channel:10", "user", "first channel")
    manager.add_message("guild:1:channel:20", "user", "second channel")

    assert manager.get_messages("guild:1:channel:10") == [
        {"role": "user", "content": "first channel"}
    ]
    assert manager.get_messages("guild:1:channel:20") == [
        {"role": "user", "content": "second channel"}
    ]


def test_clear_history_removes_only_the_requested_channel() -> None:
    manager = ConversationManager(make_config())
    first_channel = "guild:1:channel:10"
    second_channel = "dm:42"

    manager.add_message(first_channel, "user", "remove me")
    manager.add_message(second_channel, "user", "keep me")

    manager.clear_history(first_channel)

    assert manager.get_messages(first_channel) == []
    assert manager.get_messages(second_channel) == [
        {"role": "user", "content": "keep me"}
    ]


def test_history_is_trimmed_to_the_configured_token_budget() -> None:
    manager = ConversationManager(
        Config(
            discord_token="test-token",
            history_ttl_seconds=3600,
            history_token_budget=5,
        )
    )
    channel_id = "guild:1:channel:10"

    manager.add_message(channel_id, "user", "one two three four")
    manager.add_message(channel_id, "assistant", "five six seven eight")

    messages = manager.get_messages(channel_id)

    assert messages
    assert messages[-1] == {"role": "assistant", "content": "five six seven eight"}