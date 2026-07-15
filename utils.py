"""
Utility functions shared across modules.
"""

from __future__ import annotations

import html
import logging
import re
import time
from typing import List

logger = logging.getLogger(__name__)

# Approximate characters-per-token ratio for common LLM tokenizers.
_CHARS_PER_TOKEN = 4.0

_DISCORD_MAX_LENGTH = 2000
_DISCORD_EMBED_DESC_LENGTH = 2000
_DISCORD_EMBED_FIELD_VALUE_LENGTH = 1024


def estimate_tokens(text: str) -> int:
    """Roughly estimate the number of tokens in *text*."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def truncate_to_token_budget(
    messages: list[dict], budget: int
) -> list[dict]:
    """
    Trim a list of ``{"role": ..., "content": ...}`` messages so their
    combined token count stays within *budget*.

    Keeps the most recent messages and drops the oldest.
    """
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))

    if total <= budget:
        return messages

    # Trim from the front until we fit.
    trimmed = list(messages)
    while total > budget and len(trimmed) > 1:
        removed = trimmed.pop(0)
        total -= estimate_tokens(removed.get("content", ""))

    return trimmed


def split_discord_message(text: str, max_length: int = _DISCORD_MAX_LENGTH) -> List[str]:
    """
    Split *text* into chunks no longer than *max_length*, trying to
    break on word boundaries.
    """
    if len(text) <= max_length:
        return [text]

    chunks: List[str] = []
    while len(text) > max_length:
        # Find the last space within the limit.
        split_at = text.rfind(" ", 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def sanitize_html(text: str) -> str:
    """Strip basic HTML tags and unescape entities."""
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


def escape_markdown(text: str) -> str:
    """
    Escape common markdown characters so they render literally in Discord.
    """
    for char in ("_", "*", "`", "~", "|"):
        text = text.replace(char, f"\\{char}")
    return text


def is_valid_image_url(url: str) -> bool:
    """
    Basic validation for an image URL.

    Checks scheme and common image extensions.
    """
    if not url:
        return False
    allowed_schemes = ("http://", "https://", "data:")
    if not any(url.startswith(s) for s in allowed_schemes):
        return False
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    url_lower = url.lower().split("?")[0]
    return any(url_lower.endswith(ext) for ext in image_extensions)


def now_timestamp() -> float:
    """Return the current Unix timestamp."""
    return time.time()