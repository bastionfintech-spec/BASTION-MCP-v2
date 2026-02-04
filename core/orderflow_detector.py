"""
Order Flow Detector - Liquidity Zone Analysis
==============================================

REAL EDGE: Institutional order flow leaves footprints in the order book.
Detect accumulation/distribution zones, large bid/ask imbalances, and
smart money positioning.

This module implements:
1. Order book imbalance detection (bid vs ask pressure)
2. Large trade identification (block trades, whale activity)
3. Liquidity zone mapping (areas of high bid/ask concentration)
4. CVD (Cumulative Volume Delta) for directional flow
5. Smart money proxy (institutional accumulation/distribution)

Data Sources:
- Helsinki VM Quant API (http://77.42.29.188:5002)
- Real-time order book snapshots
- Trade flow data

Author: MCF Labs / BASTION
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


class FlowDirection(str, Enum):
    """Dominant order flow direction."""
    STRONG_BULLISH = "strong_bullish"   # Heavy buying
    BULLISH = "bullish"                  # Moderate buying
    NEUTRAL = "neutral"                  # Balanced
    BEARISH = "bearish"                  # Moderate selling
    STRONG_BEARISH = "strong_bearish"    # Heavy selling


class LiquidityType(str, Enum):
    """Type of liquidity zone."""
    BID_WALL = "bid_wall"        # Strong support (buy orders)
    ASK_WALL = "ask_wall"        # Strong resistance (sell orders)
    THIN = "thin"                # Weak liquidity (price can move fast)
    BALANCED = "balanced"        # Neutral zone


@dataclass
class LiquidityZone:
    """A zone of concentrated liquidity in the order book."""
    
    price: float
    liquidity_type: LiquidityType
    
    # Volume metrics
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    total_volume: float = 0.0
    
    # Imbalance
    imbalance_ratio: float = 0.0  # >1 = more bids, <1 = more asks
    
    # Price range
    price_low: float = 0.0
    price_high: float = 0.0
    
    # Strength
    strength: float = 0.0  # 0-1
    
    def to_dict(self) -> Dict:
        return {
            'price': self.price,
            'type': self.liquidity_type.value,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'imbalance_ratio': self.imbalance_ratio,
            'strength': self.strength,
        }


@dataclass
class OrderFlowAnalysis:
    """Complete order flow analysis."""
    
    # Flow direction
    flow_direction: FlowDirection = FlowDirection.NEUTRAL
    flow_strength: float = 0.5  # 0-1
    
    # Cumulative Volume Delta
    cvd: float = 0.0
    cvd_trend: str = "neutral"  # 'accumulation', 'distribution', 'neutral'
    
    # Order book imbalance
    bid_ask_imbalance: float = 0.0  # Positive = bid pressure, Negative = ask pressure
    
    # Liquidity zones
    liquidity_zones: List[LiquidityZone] = field(default_factory=list)
    support_zones: List[LiquidityZone] = field(default_factory=list)
    resistance_zones: List[LiquidityZone] = field(default_factory=list)
    
    # Smart money indicators
    large_buy_volume: float = 0.0
    large_sell_volume: float = 0.0
    smart_money_direction: str = "neutral"  # 'accumulating', 'distributing', 'neutral'
    
    # Order flow score (0-10)
    orderflow_score: float = 5.0
    
    # Context
    current_price: float = 0.0
    timestamp: Optional[str] = None
    
    def get_summary(self) -> Dict:
        return {
            'flow_direction': self.flow_direction.value,
            'flow_strength': self.flow_strength,
            'cvd': self.cvd,
            'bid_ask_imbalance': self.bid_ask_imbalance,
            'smart_money_direction': self.smart_money_direction,
            'orderflow_score': self.orderflow_score,
            'support_zones': len(self.support_zones),
            'resistance_zones': len(self.resistance_zones),
        }


class OrderFlowDetector:
    """
    Order Flow Detector - Analyzes institutional footprints and liquidity.
    
    Connects to Helsinki VM Quant API for real-time order flow data.
    
    Usage:
        detector = OrderFlowDetector()
        analysis = await detector.analyze(symbol='BTCUSDT')
        
        if analysis.flow_direction == FlowDirection.STRONG_BULLISH:
            # Strong buying pressure
            consider_long()
    """
    
    # Helsinki VM Quant API
    HELSINKI_QUANT = "http://77.42.29.188:5002"
    
    def __init__(
        self,
        # Liquidity detection
        liquidity_threshold_pct: float = 0.20,  # 20% above mean = wall
        thin_threshold_pct: float = 0.50,       # 50% below mean = thin
        
        # Imbalance thresholds
        strong_imbalance: float = 2.0,          # 2:1 ratio = strong
        moderate_imbalance: float = 1.5,        # 1.5:1 ratio = moderate
        
        # Large trade detection
        large_trade_multiplier: float = 3.0,    # 3x mean volume = large
        
        # CVD settings
        cvd_lookback: int = 100,                # Bars for CVD calculation
        
        # Timeout
        timeout_seconds: int = 5,
    ):
        self.liquidity_threshold_pct = liquidity_threshold_pct
        self.thin_threshold_pct = thin_threshold_pct
        self.strong_imbalance = strong_imbalance
        self.moderate_imbalance = moderate_imbalance
        self.large_trade_multiplier = large_trade_multiplier
        self.cvd_lookback = cvd_lookback
        self.timeout_seconds = timeout_seconds
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def analyze(
        self,
        symbol: str = "BTCUSDT",
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> OrderFlowAnalysis:
        """
        Perform complete order flow analysis.
        
        Args:
            symbol: Trading pair symbol
            ohlcv: Optional OHLCV DataFrame for CVD calculation
            
        Returns:
            OrderFlowAnalysis with all order flow data
        """
        analysis = OrderFlowAnalysis()
        
        try:
            # Fetch order flow data from Helsinki VM
            orderbook_data = await self._fetch_orderbook_imbalance(symbol)
            large_trades = await self._fetch_large_trades(symbol)
            cvd_data = await self._fetch_cvd(symbol)
            
            # Analyze order book imbalance
            if orderbook_data:
                analysis.bid_ask_imbalance = orderbook_data.get('imbalance', 0.0)
                analysis.liquidity_zones = self._detect_liquidity_zones(orderbook_data)
                analysis.support_zones = [z for z in analysis.liquidity_zones if z.liquidity_type == LiquidityType.BID_WALL]
                analysis.resistance_zones = [z for z in analysis.liquidity_zones if z.liquidity_type == LiquidityType.ASK_WALL]
            
            # Analyze large trades
            if large_trades:
                analysis.large_buy_volume = large_trades.get('large_buy_volume', 0.0)
                analysis.large_sell_volume = large_trades.get('large_sell_volume', 0.0)
                analysis.smart_money_direction = self._determine_smart_money_direction(
                    analysis.large_buy_volume, analysis.large_sell_volume
                )
            
            # Analyze CVD
            if cvd_data:
                analysis.cvd = cvd_data.get('cvd', 0.0)
                analysis.cvd_trend = cvd_data.get('trend', 'neutral')
            elif ohlcv is not None:
                # Fallback: Calculate CVD from OHLCV
                analysis.cvd = self._calculate_cvd_from_ohlcv(ohlcv)
                analysis.cvd_trend = 'accumulation' if analysis.cvd > 0 else 'distribution' if analysis.cvd < 0 else 'neutral'
            
            # Determine flow direction
            analysis.flow_direction, analysis.flow_strength = self._determine_flow_direction(
                analysis.bid_ask_imbalance,
                analysis.cvd,
                analysis.smart_money_direction
            )
            
            # Calculate order flow score
            analysis.orderflow_score = self._calculate_orderflow_score(analysis)
            
        except Exception as e:
            logger.error(f"Order flow analysis failed: {e}")
            # Return neutral analysis on error
            pass
        
        return analysis
    
    async def _fetch_orderbook_imbalance(self, symbol: str) -> Optional[Dict]:
        """Fetch order book imbalance from Helsinki VM."""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            url = f"{self.HELSINKI_QUANT}/orderbook/{symbol}"
            async with self._session.get(url, timeout=self.timeout_seconds) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch orderbook imbalance: {e}")
        
        return None
    
    async def _fetch_large_trades(self, symbol: str) -> Optional[Dict]:
        """Fetch large trade data from Helsinki VM."""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            url = f"{self.HELSINKI_QUANT}/large_trades/{symbol}"
            async with self._session.get(url, timeout=self.timeout_seconds) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch large trades: {e}")
        
        return None
    
    async def _fetch_cvd(self, symbol: str) -> Optional[Dict]:
        """Fetch CVD (Cumulative Volume Delta) from Helsinki VM."""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            url = f"{self.HELSINKI_QUANT}/cvd/{symbol}"
            async with self._session.get(url, timeout=self.timeout_seconds) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch CVD: {e}")
        
        return None
    
    def _detect_liquidity_zones(self, orderbook_data: Dict) -> List[LiquidityZone]:
        """Detect liquidity zones from order book data."""
        zones = []
        
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        
        # Calculate mean volume
        all_volumes = [float(bid[1]) for bid in bids] + [float(ask[1]) for ask in asks]
        if not all_volumes:
            return zones
        
        mean_volume = np.mean(all_volumes)
        
        # Detect bid walls (support)
        for price, volume in bids:
            price = float(price)
            volume = float(volume)
            
            if volume > mean_volume * (1 + self.liquidity_threshold_pct):
                strength = min(1.0, volume / (mean_volume * 3))
                zones.append(LiquidityZone(
                    price=price,
                    liquidity_type=LiquidityType.BID_WALL,
                    bid_volume=volume,
                    total_volume=volume,
                    imbalance_ratio=999.0,  # Extreme bid pressure
                    strength=strength,
                ))
        
        # Detect ask walls (resistance)
        for price, volume in asks:
            price = float(price)
            volume = float(volume)
            
            if volume > mean_volume * (1 + self.liquidity_threshold_pct):
                strength = min(1.0, volume / (mean_volume * 3))
                zones.append(LiquidityZone(
                    price=price,
                    liquidity_type=LiquidityType.ASK_WALL,
                    ask_volume=volume,
                    total_volume=volume,
                    imbalance_ratio=0.001,  # Extreme ask pressure
                    strength=strength,
                ))
        
        # Detect thin zones (fast moves possible)
        for price, volume in bids + asks:
            price = float(price)
            volume = float(volume)
            
            if volume < mean_volume * self.thin_threshold_pct:
                zones.append(LiquidityZone(
                    price=price,
                    liquidity_type=LiquidityType.THIN,
                    total_volume=volume,
                    strength=0.3,
                ))
        
        return zones
    
    def _determine_smart_money_direction(
        self,
        large_buy_volume: float,
        large_sell_volume: float,
    ) -> str:
        """Determine smart money direction from large trades."""
        if large_buy_volume + large_sell_volume == 0:
            return "neutral"
        
        buy_ratio = large_buy_volume / (large_buy_volume + large_sell_volume)
        
        if buy_ratio > 0.65:
            return "accumulating"
        elif buy_ratio < 0.35:
            return "distributing"
        else:
            return "neutral"
    
    def _calculate_cvd_from_ohlcv(self, ohlcv: pd.DataFrame) -> float:
        """
        Calculate CVD from OHLCV data (fallback method).
        
        CVD = Sum of (Volume * sign(Close - Open))
        """
        if len(ohlcv) < self.cvd_lookback:
            data = ohlcv
        else:
            data = ohlcv.tail(self.cvd_lookback)
        
        # Calculate delta for each bar
        delta = np.sign(data['close'] - data['open']) * data['volume']
        cvd = np.sum(delta)
        
        return float(cvd)
    
    def _determine_flow_direction(
        self,
        bid_ask_imbalance: float,
        cvd: float,
        smart_money_direction: str,
    ) -> Tuple[FlowDirection, float]:
        """Determine overall flow direction and strength."""
        # Combine signals
        signals = []
        
        # Bid/ask imbalance signal
        if bid_ask_imbalance > self.strong_imbalance:
            signals.append(('bullish', 0.9))
        elif bid_ask_imbalance > self.moderate_imbalance:
            signals.append(('bullish', 0.6))
        elif bid_ask_imbalance < -self.strong_imbalance:
            signals.append(('bearish', 0.9))
        elif bid_ask_imbalance < -self.moderate_imbalance:
            signals.append(('bearish', 0.6))
        else:
            signals.append(('neutral', 0.5))
        
        # CVD signal
        if cvd > 0:
            cvd_strength = min(1.0, abs(cvd) / 1000000)  # Normalize
            signals.append(('bullish', cvd_strength))
        elif cvd < 0:
            cvd_strength = min(1.0, abs(cvd) / 1000000)
            signals.append(('bearish', cvd_strength))
        else:
            signals.append(('neutral', 0.5))
        
        # Smart money signal
        if smart_money_direction == 'accumulating':
            signals.append(('bullish', 0.8))
        elif smart_money_direction == 'distributing':
            signals.append(('bearish', 0.8))
        else:
            signals.append(('neutral', 0.5))
        
        # Aggregate signals
        bullish_score = sum(s for d, s in signals if d == 'bullish')
        bearish_score = sum(s for d, s in signals if d == 'bearish')
        total_signals = len(signals)
        
        bullish_pct = bullish_score / total_signals
        bearish_pct = bearish_score / total_signals
        
        # Determine direction
        if bullish_pct > 0.7:
            return FlowDirection.STRONG_BULLISH, bullish_pct
        elif bullish_pct > 0.55:
            return FlowDirection.BULLISH, bullish_pct
        elif bearish_pct > 0.7:
            return FlowDirection.STRONG_BEARISH, bearish_pct
        elif bearish_pct > 0.55:
            return FlowDirection.BEARISH, bearish_pct
        else:
            return FlowDirection.NEUTRAL, 0.5
    
    def _calculate_orderflow_score(self, analysis: OrderFlowAnalysis) -> float:
        """Calculate order flow score (0-10)."""
        score = 5.0  # Neutral baseline
        
        # Flow direction contribution
        if analysis.flow_direction == FlowDirection.STRONG_BULLISH:
            score += 3.0
        elif analysis.flow_direction == FlowDirection.BULLISH:
            score += 1.5
        elif analysis.flow_direction == FlowDirection.STRONG_BEARISH:
            score -= 3.0
        elif analysis.flow_direction == FlowDirection.BEARISH:
            score -= 1.5
        
        # CVD trend contribution
        if analysis.cvd_trend == 'accumulation':
            score += 1.0
        elif analysis.cvd_trend == 'distribution':
            score -= 1.0
        
        # Smart money contribution
        if analysis.smart_money_direction == 'accumulating':
            score += 1.0
        elif analysis.smart_money_direction == 'distributing':
            score -= 1.0
        
        # Liquidity contribution
        if len(analysis.support_zones) > 2:
            score += 0.5  # Strong support
        if len(analysis.resistance_zones) > 2:
            score -= 0.5  # Strong resistance
        
        return max(0.0, min(10.0, score))
    
    async def close(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

