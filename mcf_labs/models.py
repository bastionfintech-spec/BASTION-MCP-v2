"""
MCF Labs Report Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class ReportType(Enum):
    MARKET_STRUCTURE = "market_structure"
    WHALE_INTELLIGENCE = "whale_intelligence"
    OPTIONS_FLOW = "options_flow"
    CYCLE_POSITION = "cycle_position"
    FUNDING_ARBITRAGE = "funding_arbitrage"
    LIQUIDATION_CASCADE = "liquidation_cascade"


class Bias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Report:
    """MCF Labs Report"""
    id: str
    type: ReportType
    title: str
    generated_at: datetime
    bias: Bias
    confidence: Confidence
    summary: str
    sections: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    views: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "bias": self.bias.value,
            "confidence": self.confidence.value,
            "summary": self.summary,
            "sections": self.sections,
            "tags": self.tags,
            "data_sources": self.data_sources,
            "views": self.views
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Report":
        return cls(
            id=data["id"],
            type=ReportType(data["type"]),
            title=data["title"],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            bias=Bias(data["bias"]),
            confidence=Confidence(data["confidence"]),
            summary=data["summary"],
            sections=data["sections"],
            tags=data.get("tags", []),
            data_sources=data.get("data_sources", []),
            views=data.get("views", 0)
        )


@dataclass 
class TradeScenario:
    """Trade scenario from market structure report"""
    bias: str
    entry_zone: List[float]
    stop_loss: float
    targets: List[float]
    invalidation: str
    risk_reward: float = 0.0
    
    def __post_init__(self):
        if self.targets and self.entry_zone:
            entry = sum(self.entry_zone) / 2
            target = self.targets[0]
            risk = abs(entry - self.stop_loss)
            reward = abs(target - entry)
            self.risk_reward = round(reward / risk, 2) if risk > 0 else 0


@dataclass
class LiquidationZone:
    """Liquidation zone data"""
    price: float
    usd_at_risk: float
    leverage: str
    distance_percent: float
    side: str  # "LONG" or "SHORT"


@dataclass
class WhalePosition:
    """Hyperliquid whale position"""
    rank: int
    side: str
    size_usd: float
    entry_price: float
    current_price: float
    leverage: int
    pnl_usd: float
    pnl_percent: float
    liquidation_price: float


