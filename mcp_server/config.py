"""
BASTION MCP Server — Configuration
"""
import os
import secrets

# ── API Connection ──────────────────────────────────────────────
# The MCP server calls the BASTION API internally
# When running on the same process, we call localhost
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", os.getenv("PORT", "3001")))
API_BASE_URL = os.getenv("MCP_API_BASE_URL", f"http://127.0.0.1:{API_PORT}")

# ── MCP Authentication ─────────────────────────────────────────
# Simple API key auth for MVP — keys validated against this env var
# Multiple keys can be comma-separated: "key1,key2,key3"
MCP_API_KEYS = os.getenv("MCP_API_KEYS", "")
MCP_MASTER_KEY = os.getenv("MCP_MASTER_KEY", "")  # Admin key with no rate limit

# ── Rate Limiting ───────────────────────────────────────────────
MCP_RATE_LIMIT = int(os.getenv("MCP_RATE_LIMIT", "100"))  # requests per minute
MCP_RATE_WINDOW = int(os.getenv("MCP_RATE_WINDOW", "60"))  # seconds

# ── Internal Auth (MCP ↔ Backend) ─────────────────────────────
# Shared secret for MCP server to pass authenticated user_id to backend
# Since MCP runs in the same process, this just prevents external spoofing
MCP_INTERNAL_SECRET = os.getenv("MCP_INTERNAL_SECRET", secrets.token_urlsafe(32))

# ── Server Info ─────────────────────────────────────────────────
MCP_SERVER_NAME = "bastion-risk-intelligence"
MCP_SERVER_VERSION = "1.0.0"
MCP_SERVER_DESCRIPTION = (
    "BASTION Risk Intelligence MCP Server — "
    "Autonomous crypto risk analysis powered by a fine-tuned 72 billion parameter AI model. "
    "Evaluate positions, analyze market structure, track whales, and generate institutional research."
)

# ── Model Info ──────────────────────────────────────────────────
MODEL_VERSION = "v6"
MODEL_BASE = "BASTION 72B"
MODEL_ACCURACY = "75.4%"
MODEL_TRAINING_EXAMPLES = 328
MODEL_GPU = "4x NVIDIA H200 (564GB VRAM)"

# ── Supported Symbols ───────────────────────────────────────────
SUPPORTED_SYMBOLS = [
    "BTC", "ETH", "SOL", "AVAX", "XRP", "DOGE", "LINK", "ADA",
    "MATIC", "DOT", "ATOM", "UNI", "AAVE", "ARB", "OP", "SUI",
    "APT", "NEAR", "FTM", "INJ", "TIA", "SEI", "JUP", "WIF",
    "PEPE", "BONK", "RENDER", "FET", "TAO", "ONDO"
]
