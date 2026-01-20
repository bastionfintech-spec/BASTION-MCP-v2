"""
BASTION API Models
==================

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class CalculateRiskRequest(BaseModel):
    """Request model for /calculate endpoint."""
    
    symbol: str = Field(
        default="BTCUSDT",
        description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)"
    )
    entry_price: float = Field(
        ...,
        gt=0,
        description="Entry price for the trade"
    )
    direction: str = Field(
        ...,
        pattern="^(long|short)$",
        description="Trade direction: 'long' or 'short'"
    )
    timeframe: str = Field(
        default="4h",
        description="Candle timeframe: 1m, 5m, 15m, 1h, 4h, 1d"
    )
    account_balance: float = Field(
        default=100000,
        gt=0,
        description="Account balance in USD"
    )
    risk_per_trade_pct: float = Field(
        default=1.0,
        gt=0,
        le=10,
        description="Risk percentage per trade (0.1 - 10%)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "BTCUSDT",
                "entry_price": 95000,
                "direction": "long",
                "timeframe": "4h",
                "account_balance": 100000,
                "risk_per_trade_pct": 1.0
            }
        }
    }


class StopLevelResponse(BaseModel):
    """Stop-loss level in response."""
    
    type: str = Field(description="Stop type: structural, atr, secondary, safety_net")
    price: float = Field(description="Stop price level")
    distance_pct: float = Field(description="Distance from entry as percentage")
    reason: str = Field(description="Reason for this stop level")
    confidence: float = Field(description="Confidence score (0-1)")


class TargetLevelResponse(BaseModel):
    """Take-profit target in response."""
    
    price: float = Field(description="Target price level")
    type: str = Field(default="structural", description="Target type: structural, vpvr, extension")
    exit_percentage: float = Field(description="Percentage of position to exit")
    distance_pct: float = Field(description="Distance from entry as percentage")
    reason: str = Field(description="Reason for this target")
    confidence: float = Field(description="Confidence score (0-1)")


class MarketContextResponse(BaseModel):
    """Market context (informational, not judgmental)."""
    
    structure_quality: float = Field(
        default=0.0,
        description="Structure quality score (0-10) from trendline analysis"
    )
    volume_profile_score: float = Field(
        default=0.0,
        description="Volume profile score (0-10) from VPVR analysis"
    )
    orderflow_bias: str = Field(
        default="neutral",
        description="Order flow bias: bullish, bearish, or neutral"
    )
    mtf_alignment: float = Field(
        default=0.0,
        description="Multi-timeframe alignment (0-1)"
    )


class RiskLevelsResponse(BaseModel):
    """Complete risk calculation response."""
    
    symbol: str
    entry_price: float
    direction: str
    current_price: float
    
    stops: List[StopLevelResponse]
    targets: List[TargetLevelResponse]
    
    position_size: float = Field(description="Position size in base currency")
    position_size_pct: float = Field(description="Position as % of account")
    risk_amount: float = Field(description="Dollar risk amount")
    
    risk_reward_ratio: float = Field(description="R:R to first target")
    max_risk_reward_ratio: float = Field(description="R:R to final target")
    
    guarding_line: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Guarding line parameters for swing trades"
    )
    
    market_context: Optional[MarketContextResponse] = Field(
        default=None,
        description="Market context scores (informational only)"
    )
    
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "BTCUSDT",
                "entry_price": 95000,
                "direction": "long",
                "current_price": 94328,
                "stops": [
                    {
                        "type": "structural",
                        "price": 92500,
                        "distance_pct": 2.6,
                        "reason": "Below structural support at 92800",
                        "confidence": 0.8
                    }
                ],
                "targets": [
                    {
                        "price": 98000,
                        "type": "vpvr",
                        "exit_percentage": 33,
                        "distance_pct": 3.2,
                        "reason": "HVN mountain (vol z=2.1)",
                        "confidence": 0.75
                    }
                ],
                "position_size": 0.421,
                "position_size_pct": 33.3,
                "risk_amount": 1000,
                "risk_reward_ratio": 2.5,
                "max_risk_reward_ratio": 6.5,
                "guarding_line": {
                    "slope": 50.0,
                    "intercept": 92000,
                    "activation_bar": 10
                },
                "market_context": {
                    "structure_quality": 7.5,
                    "volume_profile_score": 8.2,
                    "orderflow_bias": "bullish",
                    "mtf_alignment": 0.78
                }
            }
        }
    }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="ok")
    service: str = Field(default="BASTION")
    version: str = Field(default="2.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
