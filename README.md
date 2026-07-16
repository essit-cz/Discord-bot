# Discord AI Bot

A Python 3.12 Discord bot that uses a local vLLM OpenAI-compatible endpoint
and an optional local SearXNG instance. It supports DMs, mentions, images,
per-channel conversation memory, Discord slash/prefix commands, and planned
web searches.

## Features

- Discord DMs and mention-based conversations
- Prefix commands using `!` by default and slash commands
- Per-channel/DM history with a TTL and approximate token budget
- Image URL forwarding to compatible vLLM models
- Tool registry with SearXNG web search
- Search-query planning: conversational requests are converted into concise
  search queries before SearXNG is called
- Search results are treated as untrusted prompt context
- Reusable async HTTP clients with graceful shutdown

## Requirements

- Python 3.12
- A Discord bot application and token
- A local vLLM server exposing an OpenAI-compatible API
- Optional: a local SearXNG server with JSON output enabled

In the Discord Developer Portal, enable the **Message Content Intent** for the
bot. Invite the bot with permissions to view channels, send messages, read
message history, and use application commands.

## Configuration

Copy the example configuration and set the required value: