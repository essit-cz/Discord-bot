"""
Configuration module.

Loads and validates all settings from a .env file using python-dotenv.
Provides typed accessors so the rest of the bot never reads raw strings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Immutable application configuration."""

    # Discord
    discord_token: str
    bot_prefix: str = "!"
    bot_status_name: str = "AI Bot"
    bot_status_type: int = 0  # 0 = Custom, 1 = Playing, 2 = Streaming, 3 = Listening

    # vLLM
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "Qwen/Qwen3-8B"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_timeout: float = 120.0

    # SearXNG
    searxng_base_url: str = "http://127.0.0.1:8080"
    searxng_timeout: float = 10.0

    # Conversation memory
    history_ttl_seconds: int = 3600
    history_token_budget: int = 4096

    # Search
    search_enabled: bool = True

    @classmethod
    def load(cls, env_path: Optional[Path] = None) -> Config:
        """
        Load configuration from a .env file.

        Falls back to ``.env`` in the project root if no path is given.
        """
        env_file = env_path or Path(__file__).resolve().parent / ".env"
        found = load_dotenv(env_file)
        if found:
            logger.info("Loaded .env file: %s", env_file)
        else:
            logger.info("No .env file found at %s; falling back to environment variables.", env_file)

        import os

        discord_token = os.getenv("DISCORD_TOKEN")
        if not discord_token:
            raise ValueError(
                "DISCORD_TOKEN is required but not set in the environment or .env file."
            )

        search_enabled_raw = os.getenv("SEARCH_ENABLED", "true").lower()
        search_enabled = search_enabled_raw not in {"false", "0", "off"}

        return cls(
            discord_token=discord_token,
            bot_prefix=os.getenv("BOT_PREFIX", "!"),
            bot_status_name=os.getenv("BOT_STATUS_NAME", "AI Bot"),
            bot_status_type=int(os.getenv("BOT_STATUS_TYPE", "0")),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "120.0")),
            searxng_base_url=os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080"),
            searxng_timeout=float(os.getenv("SEARXNG_TIMEOUT", "10.0")),
            history_ttl_seconds=int(os.getenv("HISTORY_TTL_SECONDS", "3600")),
            history_token_budget=int(os.getenv("HISTORY_TOKEN_BUDGET", "4096")),
            search_enabled=search_enabled,
        )


# Module-level singleton — initialized once at startup.
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the current global Config instance."""
    global _config
    if _config is None:
        raise RuntimeError("Config has not been loaded yet. Call `load_config()` first.")
    return _config


def load_config(env_path: Optional[Path] = None) -> Config:
    """Load and store the global Config instance."""
    global _config
    _config = Config.load(env_path)
    logger.info(
        "Configuration loaded: model=%s, search=%s, history_budget=%d",
        _config.llm_model,
        _config.search_enabled,
        _config.history_token_budget,
    )
    return _config