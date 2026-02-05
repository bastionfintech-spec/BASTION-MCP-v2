"""
BASTION AI - Configuration Settings
====================================
Centralized configuration for all IROS/Bastion infrastructure
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class HelsinkiConfig:
    """Helsinki VM (Free Quant Data Network) Configuration"""
    base_url: str = os.getenv("HELSINKI_VM_URL", "http://77.42.29.188:5002")
    timeout: int = int(os.getenv("HELSINKI_TIMEOUT", "5000"))


@dataclass
class ModelConfig:
    """GPU Cluster (Vast.ai vLLM) Configuration"""
    # NOTE: Update base_url when Cloudflare tunnel URL changes
    base_url: str = os.getenv("BASTION_MODEL_URL", "")
    # Pre-configured API key for vLLM authentication
    api_key: str = os.getenv("BASTION_MODEL_API_KEY", "5c37b5e8e6c2480813aa0cfd4de5c903544b7a000bff729e1c99d9b4538eb34d")
    timeout: int = int(os.getenv("BASTION_MODEL_TIMEOUT", "120000"))
    model_name: str = os.getenv("BASTION_MODEL_NAME", "iros")
    max_tokens: int = 1200
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass 
class CoinglassConfig:
    """Coinglass Premium API Configuration - $299/mo"""
    # Pre-configured with your paid API key
    api_key: str = os.getenv("COINGLASS_API_KEY", "03e5a43afaa4489384cb935b9b2ea16b")
    base_url: str = "https://open-api-v3.coinglass.com/api"
    timeout: int = 10000


@dataclass
class WhaleAlertConfig:
    """Whale Alert Premium API Configuration - $29.95/mo"""
    # Pre-configured with your paid API key
    api_key: str = os.getenv("WHALE_ALERT_API_KEY", "OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ")
    rest_url: str = "https://api.whale-alert.io/v1"
    ws_url: str = "wss://ws.whale-alert.io"
    min_value: int = 1000000  # $1M minimum transaction


@dataclass
class BastionConfig:
    """Main Bastion AI Configuration"""
    helsinki: HelsinkiConfig
    model: ModelConfig
    coinglass: CoinglassConfig
    whale_alert: WhaleAlertConfig
    
    # Server settings
    port: int = int(os.getenv("PORT", "3001"))
    env: str = os.getenv("NODE_ENV", "development")
    test_mode: bool = os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes")


# Global settings instance
settings = BastionConfig(
    helsinki=HelsinkiConfig(),
    model=ModelConfig(),
    coinglass=CoinglassConfig(),
    whale_alert=WhaleAlertConfig(),
)


# Validation warnings
if not settings.helsinki.base_url:
    print("[WARNING] HELSINKI_VM_URL not set. Data network will be unavailable.")

if not settings.model.base_url:
    print("[WARNING] BASTION_MODEL_URL not set. AI model will be unavailable.")

if not settings.model.api_key:
    print("[WARNING] BASTION_MODEL_API_KEY not set. Model requests may fail.")

if settings.test_mode:
    print("[TEST] TEST MODE ENABLED")

