"""
BASTION AI - Configuration Settings
====================================
Centralized configuration for all IROS/Bastion infrastructure

IMPORTANT: All API keys must be set via environment variables (.env file).
           Never hardcode API keys in source code.
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
    timeout: int = int(os.getenv("HELSINKI_TIMEOUT", "15000"))


@dataclass
class ModelConfig:
    """GPU Cluster (Vast.ai vLLM) Configuration"""
    base_url: str = os.getenv("BASTION_MODEL_URL", "")
    api_key: str = os.getenv("BASTION_MODEL_API_KEY", "")
    timeout: int = int(os.getenv("BASTION_MODEL_TIMEOUT", "120000"))
    model_name: str = os.getenv("BASTION_MODEL_NAME", "bastion-32b")
    max_tokens: int = 1200
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class CoinglassConfig:
    """Coinglass Premium API Configuration"""
    api_key: str = os.getenv("COINGLASS_API_KEY", "")
    base_url: str = "https://open-api-v3.coinglass.com/api"
    timeout: int = 10000


@dataclass
class WhaleAlertConfig:
    """Whale Alert Premium API Configuration"""
    api_key: str = os.getenv("WHALE_ALERT_API_KEY", "")
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

if not settings.coinglass.api_key:
    print("[WARNING] COINGLASS_API_KEY not set. Market data will be limited.")

if not settings.whale_alert.api_key:
    print("[WARNING] WHALE_ALERT_API_KEY not set. Whale alerts will be unavailable.")

if settings.test_mode:
    print("[TEST] TEST MODE ENABLED")
