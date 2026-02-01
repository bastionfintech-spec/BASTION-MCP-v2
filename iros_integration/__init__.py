"""
BASTION AI Integration (Powered by IROS Infrastructure)
=========================================================

Institutional-grade crypto analysis with:
- 33 real-time quant endpoints (Helsinki VM - FREE)
- 32B parameter LLM (Vast.ai GPU cluster)
- Coinglass Premium API ($299/mo - INCLUDED)
- Whale Alert Premium API ($29.95/mo - INCLUDED)
- Dynamic context extraction
- Institutional response format

Quick Start:
    from iros_integration import BastionAI, ask_bastion
    
    # Full control
    bastion = BastionAI()
    result = await bastion.process_query("Should I long BTC with $50K?")
    
    # Quick query
    response = await ask_bastion("What's the ETH volatility regime?")
    
    # Premium APIs
    from iros_integration import CoinglassClient, WhaleAlertClient
    
    coinglass = CoinglassClient()  # API key pre-configured!
    liquidations = await coinglass.get_liquidation_history("BTC")
    
    whale = WhaleAlertClient()  # API key pre-configured!
    txs = await whale.get_transactions(min_value=10000000)
"""

from .services.bastion_ai import BastionAI, ask_bastion
from .services.helsinki import HelsinkiClient, SYMBOL_ENDPOINTS, STATIC_ENDPOINTS
from .services.query_processor import QueryProcessor, QueryContext
from .services.coinglass import CoinglassClient, CoinglassResponse
from .services.whale_alert import WhaleAlertClient, WhaleAlertResponse, WhaleTransaction
from .config.settings import settings, BastionConfig

__version__ = "1.0.0"
__all__ = [
    # Core AI
    "BastionAI",
    "ask_bastion",
    # Helsinki (FREE)
    "HelsinkiClient",
    "SYMBOL_ENDPOINTS",
    "STATIC_ENDPOINTS",
    # Query Processing
    "QueryProcessor",
    "QueryContext",
    # Coinglass Premium (API key included!)
    "CoinglassClient",
    "CoinglassResponse",
    # Whale Alert Premium (API key included!)
    "WhaleAlertClient",
    "WhaleAlertResponse",
    "WhaleTransaction",
    # Config
    "settings",
    "BastionConfig",
]

