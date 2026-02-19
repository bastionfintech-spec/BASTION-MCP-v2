"""
BASTION MCP Server — Authentication & Rate Limiting
"""
import time
import logging
from collections import defaultdict
from typing import Optional

from . import config

logger = logging.getLogger("bastion.mcp.auth")

# ── In-Memory Rate Limiter ──────────────────────────────────────
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate an MCP API key.
    Returns True if the key is valid, False otherwise.
    """
    if not api_key:
        return False

    # Strip "Bearer " prefix if present
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]

    # Check master key (no rate limit)
    if config.MCP_MASTER_KEY and api_key == config.MCP_MASTER_KEY:
        return True

    # Check against comma-separated key list
    valid_keys = [k.strip() for k in config.MCP_API_KEYS.split(",") if k.strip()]
    if not valid_keys:
        # If no keys configured, allow all (development mode)
        logger.warning("[MCP Auth] No MCP_API_KEYS configured — allowing all requests (dev mode)")
        return True

    return api_key in valid_keys


def is_master_key(api_key: Optional[str]) -> bool:
    """Check if this is the master key (exempt from rate limits)."""
    if not api_key or not config.MCP_MASTER_KEY:
        return False
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    return api_key == config.MCP_MASTER_KEY


def check_rate_limit(api_key: str) -> bool:
    """
    Check if the API key has exceeded its rate limit.
    Returns True if within limits, False if exceeded.
    """
    if is_master_key(api_key):
        return True

    now = time.time()
    window = config.MCP_RATE_WINDOW
    limit = config.MCP_RATE_LIMIT

    # Clean old entries
    key = api_key[-8:] if len(api_key) > 8 else api_key  # Use suffix as bucket key
    _rate_buckets[key] = [t for t in _rate_buckets[key] if now - t < window]

    if len(_rate_buckets[key]) >= limit:
        return False

    _rate_buckets[key].append(now)
    return True


def get_rate_limit_info(api_key: str) -> dict:
    """Get current rate limit status for a key."""
    now = time.time()
    window = config.MCP_RATE_WINDOW
    limit = config.MCP_RATE_LIMIT
    key = api_key[-8:] if len(api_key) > 8 else api_key

    recent = [t for t in _rate_buckets.get(key, []) if now - t < window]
    return {
        "limit": limit,
        "remaining": max(0, limit - len(recent)),
        "window_seconds": window,
        "reset_at": int(now + window)
    }
