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


def estimate_tokens(text: str) -> int:
    """
    Roughly estimate the number of tokens in a string.

    Uses a simple heuristic: ~4 characters per token for English text.
    Good enough for trimming conversation history.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def now_timestamp() -> float:
    """Return the current Unix timestamp."""
    return time.time()


def split_discord_message(text: str, max_length: int = 2000) -> List[str]:
    """
    Split a long string into chunks that fit within Discord's message
    length limit, trying to break on word boundaries.
    """
    if len(text) <= max_length:
        return [text]

    chunks: List[str] = []
    while len(text) > max_length:
        # Find the last space within the limit.
        split_point = text.rfind(" ", 0, max_length)
        if split_point == -1:
            # No space found; hard split.
            split_point = max_length
        chunks.append(text[:split_point])
        text = text[split_point:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def escape_markdown(text: str) -> str:
    """
    Escape common markdown characters in a string so they render
    literally in Discord.
    """
    # Escape backslash first, then other characters.
    text = text.replace("\\", "\\\\")
    text = text.replace("*", "\\*")
    text = text.replace("_", "\\_")
    text = text.replace("`", "\\`")
    text = text.replace("~", "\\~")
    return text


def sanitize_html(text: str) -> str:
    """
    Strip HTML tags and unescape HTML entities from a string.
    Useful for cleaning up search result snippets.
    """
    # Strip HTML tags.
    clean = re.sub(r"<[^>]+>", "", text)
    # Unescape HTML entities.
    clean = html.unescape(clean)
    return clean.strip()


def is_valid_image_url(url: str) -> bool:
    """
    Validate that a URL looks like a plausible image URL.
    """
    if not url:
        return False
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    lower = url.lower()
    return any(lower.endswith(ext) for ext in image_extensions)