"""Local startup health check for CI and container diagnostics.

This check intentionally performs no network requests and never connects to
Discord. It only verifies that the application composition root can construct
all configured shared dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from main import build_app


async def _run() -> None:
    bot = build_app()
    try:
        print("Application construction check passed.")
    finally:
        await bot.close()


def main() -> int:
    """Run the non-network application-construction health check."""
    logging.disable(logging.CRITICAL)
    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"Application construction check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())