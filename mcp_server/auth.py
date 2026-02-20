"""
BASTION MCP Server — Authentication & Rate Limiting
====================================================
Validates user-generated API keys (bst_...) against Supabase.
Keys are SHA-256 hashed in the DB; plaintext never stored.
"""

import time
import hashlib
import logging
from collections import defaultdict
from typing import Optional, Dict, Any

from . import config

logger = logging.getLogger("bastion.mcp.auth")

# ── Supabase Client (injected from terminal_api.py at startup) ────
_supabase_client = None          # supabase.Client
_api_keys_table = "bastion_api_keys"


def set_supabase_client(client) -> None:
    """
    Called once from terminal_api.py startup to share the Supabase client
    with the MCP auth layer (same process, no network hop).
    """
    global _supabase_client
    _supabase_client = client
    if client:
        logger.info("[MCP Auth] Supabase client wired for API key validation")
    else:
        logger.warning("[MCP Auth] Supabase client is None — key validation will fail")


# ── Key Validation Cache ──────────────────────────────────────────
# {key_hash: {"user_id": str, "scopes": list, "cached_at": float}}
_key_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


def _hash_key(raw_key: str) -> str:
    """SHA-256 hash a raw API key (must match what we store in Supabase)."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _cache_get(key_hash: str) -> Optional[Dict[str, Any]]:
    """Return cached key info if still fresh, else None."""
    entry = _key_cache.get(key_hash)
    if entry and (time.time() - entry["cached_at"]) < _CACHE_TTL:
        return entry
    # Expired — evict
    _key_cache.pop(key_hash, None)
    return None


def _cache_set(key_hash: str, user_id: str, scopes: list) -> Dict[str, Any]:
    """Cache a validated key."""
    entry = {
        "user_id": user_id,
        "scopes": scopes,
        "cached_at": time.time(),
    }
    _key_cache[key_hash] = entry
    return entry


def invalidate_cache(key_hash: str = None) -> None:
    """
    Invalidate one key or the entire cache.
    Called when a key is revoked.
    """
    if key_hash:
        _key_cache.pop(key_hash, None)
    else:
        _key_cache.clear()


# ── Key Validation ────────────────────────────────────────────────

async def validate_bst_key(api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Validate a BASTION API key (bst_...) against Supabase.

    Returns:
        {"user_id": str, "scopes": ["read", ...]}  on success
        None  on failure (invalid, revoked, expired, no DB)
    """
    if not api_key:
        return None

    # Strip "Bearer " prefix if present
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]

    # Must start with bst_ prefix
    if not api_key.startswith("bst_"):
        # Legacy: check env-var keys for backward compat during migration
        return _check_legacy_key(api_key)

    key_hash = _hash_key(api_key)

    # 1. Check cache first
    cached = _cache_get(key_hash)
    if cached:
        return {"user_id": cached["user_id"], "scopes": cached["scopes"]}

    # 2. Query Supabase
    if not _supabase_client:
        logger.error("[MCP Auth] No Supabase client — cannot validate bst_ key")
        return None

    try:
        result = (
            _supabase_client.table(_api_keys_table)
            .select("user_id, scopes, expires_at, revoked")
            .eq("key_hash", key_hash)
            .eq("revoked", False)
            .execute()
        )

        if not result.data:
            logger.warning(f"[MCP Auth] Key not found: bst_{api_key[4:8]}...")
            return None

        row = result.data[0]

        # Check expiration
        if row.get("expires_at"):
            from datetime import datetime, timezone
            expires = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > expires:
                logger.warning(f"[MCP Auth] Key expired: bst_{api_key[4:8]}...")
                return None

        # Valid — cache and return
        scopes = row.get("scopes", ["read"])
        info = _cache_set(key_hash, row["user_id"], scopes)

        # Update last_used_at (fire-and-forget, don't block response)
        try:
            _supabase_client.table(_api_keys_table).update(
                {"last_used_at": datetime.now(timezone.utc).isoformat()}
            ).eq("key_hash", key_hash).execute()
        except Exception:
            pass  # Non-critical

        logger.info(f"[MCP Auth] Key validated: bst_{api_key[4:8]}... → user={row['user_id'][:8]}... scopes={scopes}")
        return {"user_id": row["user_id"], "scopes": scopes}

    except Exception as e:
        logger.error(f"[MCP Auth] Supabase query error: {e}")
        return None


def _check_legacy_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Backward-compat: check against MCP_API_KEYS / MCP_MASTER_KEY env vars.
    Returns guest-level access (read-only, no user_id).
    """
    # Master key
    if config.MCP_MASTER_KEY and api_key == config.MCP_MASTER_KEY:
        return {"user_id": None, "scopes": ["read", "trade", "engine"]}

    # Env-var key list
    valid_keys = [k.strip() for k in config.MCP_API_KEYS.split(",") if k.strip()]
    if not valid_keys:
        # No keys configured = dev mode, allow unauthenticated
        return None

    if api_key in valid_keys:
        return {"user_id": None, "scopes": ["read"]}

    return None


# ── Scope Checking ────────────────────────────────────────────────

def check_scope(key_info: Optional[Dict[str, Any]], required_scope: str) -> bool:
    """
    Check if a validated key has the required scope.

    Scope hierarchy:
        engine > trade > read
    So a key with 'engine' scope also has 'trade' and 'read'.
    """
    if not key_info:
        return False

    scopes = key_info.get("scopes", [])

    # Hierarchy: engine implies trade implies read
    if "engine" in scopes:
        return True  # engine can do everything
    if "trade" in scopes and required_scope in ("trade", "read"):
        return True
    if "read" in scopes and required_scope == "read":
        return True

    return False


# ── Rate Limiting (unchanged from original) ───────────────────────
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(api_key: str) -> bool:
    """
    Check if the API key has exceeded its rate limit.
    Returns True if within limits, False if exceeded.
    """
    # Master key exempt
    if config.MCP_MASTER_KEY and api_key == config.MCP_MASTER_KEY:
        return True

    now = time.time()
    window = config.MCP_RATE_WINDOW
    limit = config.MCP_RATE_LIMIT

    # Use key suffix as bucket identifier
    key = api_key[-8:] if len(api_key) > 8 else api_key
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
        "reset_at": int(now + window),
    }
