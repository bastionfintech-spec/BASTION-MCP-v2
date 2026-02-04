"""
BASTION Core Risk Engine - Pure Risk Management
================================================

BASTION is strategy-agnostic risk management.

What it DOES:
✅ Calculate optimal stops (structural, ATR-based, multi-tier)
✅ Calculate optimal targets (structural, volume-informed)
✅ Position sizing (volatility-adjusted)
✅ Trade management (trailing stops, partial exits, guarding lines)
✅ Provide market context (structure quality, volume profile, order flow)

What it DOES NOT do:
❌ Judge if your trade is "good" or "bad" (that's IROS)
❌ Give entry signals (that's your strategy)
❌ Score trade quality (that's IROS with MCF)

BASTION provides the infrastructure. You provide the strategy.

Author: MCF Labs / BASTION
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import asyncio

from .vpvr_analyzer import VPVRAnalyzer, VPVRAnalysis
from .structure_detector import StructureDetector, StructureAnalysis
from .mtf_structure import MTFStructureAnalyzer, MTFAlignment
from .orderflow_detector import OrderFlowDetector, OrderFlowAnalysis

logger = logging.getLogger(__name__)


@dataclass
class RiskLevels:
    """Risk levels output - pure risk management, no trade scoring."""
    
    # Stop levels
    stops: List[Dict[str, Any]] = field(default_factory=list)
    
    # Target levels
    targets: List[Dict[str, Any]] = field(default_factory=list)
    
    # Position sizing
    position_size: float = 0.0
    position_size_pct: float = 0.0
    risk_amount: float = 0.0
    
    # Risk metrics
    risk_reward_ratio: float = 0.0
    max_risk_reward_ratio: float = 0.0
    
    # Entry context
    entry_price: float = 0.0
    direction: str = "long"
    timeframe: str = "4h"
    
    # Market Context (for your strategy to use)
    structure_quality: float = 0.0      # 0-10 from StructureDetector
    volume_profile_score: float = 0.0   # 0-10 from VPVRAnalyzer
    orderflow_bias: str = "neutral"     # bullish/bearish/neutral
    mtf_alignment: float = 0.0          # 0-1 alignment score
    
    # Guarding line (swing trades)
    guarding_line: Optional[Dict[str, Any]] = None
    
    # Detailed analyses (for advanced users)
    structure_analysis: Optional[StructureAnalysis] = None
    vpvr_analysis: Optional[VPVRAnalysis] = None
    orderflow_analysis: Optional[OrderFlowAnalysis] = None
    mtf_analysis: Optional[MTFAlignment] = None
    
    # Timestamp
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'entry_price': self.entry_price,
            'direction': self.direction,
            'stops': self.stops,
            'targets': self.targets,
            'position_size': self.position_size,
            'position_size_pct': self.position_size_pct,
            'risk_amount': self.risk_amount,
            'risk_reward_ratio': self.risk_reward_ratio,
            'max_risk_reward_ratio': self.max_risk_reward_ratio,
            'market_context': {
                'structure_quality': self.structure_quality,
                'volume_profile_score': self.volume_profile_score,
                'orderflow_bias': self.orderflow_bias,
                'mtf_alignment': self.mtf_alignment,
            },
            'calculated_at': self.calculated_at.isoformat(),
        }


@dataclass
class RiskEngineConfig:
    """Configuration for Risk Engine."""
    
    # Detection Systems (all optional)
    enable_structure_detection: bool = True
    enable_vpvr_analysis: bool = True
    enable_orderflow_detection: bool = True
    enable_mtf_analysis: bool = True
    
    # Stop-loss settings
    use_structural_stops: bool = True       # Use detected S/R levels
    atr_stop_multiplier: float = 2.0        # Fallback ATR multiplier
    max_stop_pct: float = 5.0               # Maximum stop distance
    enable_multi_tier_stops: bool = True    # Primary/Secondary/Safety-net
    
    # Take-profit settings
    use_structural_targets: bool = True     # Use detected R/S levels
    min_rr_ratio: float = 2.0               # Minimum risk:reward
    enable_partial_exits: bool = True       # Scale out at targets
    partial_exit_ratios: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
    # Position sizing
    default_risk_pct: float = 1.0           # Default risk per trade
    volatility_adjusted_sizing: bool = True  # Adjust size based on ATR
    
    # Guarding line (swing trading)
    enable_guarding_line: bool = True
    guarding_activation_bars: int = 10
    guarding_buffer_pct: float = 0.3


class RiskEngine:
    """
    BASTION Risk Engine - Strategy-Agnostic Risk Management.
    
    YOU provide:
    - Entry price
    - Direction (long/short)
    - Account balance
    - Risk tolerance
    
    BASTION provides:
    - Optimal stops (structural + ATR-based)
    - Optimal targets (structural + volume-informed)
    - Position sizing (volatility-adjusted)
    - Market context (structure quality, volume profile, order flow, MTF)
    
    BASTION does NOT tell you if your trade is "good" or "bad".
    That's your strategy's job (or IROS's job).
    
    Usage:
        engine = RiskEngine()
        
        # Calculate risk levels for YOUR trade idea
        levels = await engine.calculate_risk_levels(
            symbol='BTCUSDT',
            entry_price=94500,
            direction='long',
            timeframe='4h',
            account_balance=100000,
            ohlcv_data={'4h': df_4h, '1d': df_daily}
        )
        
        # Use the stops and targets BASTION calculated
        place_order(
            entry=levels.entry_price,
            stop=levels.stops[0]['price'],
            targets=[t['price'] for t in levels.targets],
            size=levels.position_size
        )
        
        # Optionally: Check market context
        if levels.structure_quality < 5.0:
            print("Warning: Weak structure detected")
        
        if levels.orderflow_bias == 'bearish' and direction == 'long':
            print("Warning: Order flow is bearish")
    """
    
    def __init__(self, config: Optional[RiskEngineConfig] = None):
        self.config = config or RiskEngineConfig()
        
        # Initialize detection systems (optional)
        self.structure_detector = StructureDetector() if self.config.enable_structure_detection else None
        self.vpvr_analyzer = VPVRAnalyzer() if self.config.enable_vpvr_analysis else None
        self.orderflow_detector = OrderFlowDetector() if self.config.enable_orderflow_detection else None
        self.mtf_analyzer = MTFStructureAnalyzer() if self.config.enable_mtf_analysis else None
    
    async def calculate_risk_levels(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        timeframe: str,
        account_balance: float,
        ohlcv_data: Dict[str, pd.DataFrame],  # {'4h': df, '1d': df, ...}
        risk_per_trade_pct: float = 1.0,
    ) -> RiskLevels:
        """
        Calculate risk levels for YOUR trade setup.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            entry_price: YOUR entry price
            direction: YOUR direction ('long' or 'short')
            timeframe: Primary timeframe ('4h', '1d', etc.)
            account_balance: Account size in USD
            ohlcv_data: Dict of timeframe -> OHLCV DataFrame
            risk_per_trade_pct: Risk per trade (default 1%)
            
        Returns:
            RiskLevels with stops, targets, and market context
        """
        levels = RiskLevels(
            entry_price=entry_price,
            direction=direction,
            timeframe=timeframe,
        )
        
        # Get primary timeframe data
        primary_df = ohlcv_data.get(timeframe)
        if primary_df is None or len(primary_df) < 50:
            logger.error(f"Insufficient data for timeframe {timeframe}")
            return levels
        
        # Calculate ATR for volatility context
        atr = self._calculate_atr(primary_df)
        atr_pct = (atr / entry_price) * 100
        
        # Step 1: Structure Analysis (optional)
        if self.structure_detector:
            levels.structure_analysis = self.structure_detector.analyze(primary_df)
            levels.structure_quality = levels.structure_analysis.structure_score
        
        # Step 2: VPVR Analysis (optional)
        if self.vpvr_analyzer:
            levels.vpvr_analysis = self.vpvr_analyzer.analyze(
                primary_df, direction=direction
            )
            levels.volume_profile_score = levels.vpvr_analysis.volume_score
        
        # Step 3: Order Flow Analysis (optional)
        if self.orderflow_detector:
            levels.orderflow_analysis = await self.orderflow_detector.analyze(
                symbol=symbol, ohlcv=primary_df
            )
            levels.orderflow_bias = self._determine_orderflow_bias(
                levels.orderflow_analysis
            )
        
        # Step 4: Multi-Timeframe Analysis (optional)
        if self.mtf_analyzer and len(ohlcv_data) > 1:
            levels.mtf_analysis = self.mtf_analyzer.analyze(
                ohlcv_data, proposed_direction=direction
            )
            levels.mtf_alignment = levels.mtf_analysis.alignment_score
        
        # Step 5: Calculate Stops (using structural levels if available)
        levels.stops = self._calculate_stops(
            levels, primary_df, atr
        )
        
        # Step 6: Calculate Targets (using structural levels + VPVR if available)
        levels.targets = self._calculate_targets(
            levels, primary_df, atr
        )
        
        # Step 7: Position Sizing
        primary_stop_price = levels.stops[0]['price'] if levels.stops else entry_price - (atr * 2)
        risk_distance = abs(entry_price - primary_stop_price)
        
        levels.position_size, levels.position_size_pct, levels.risk_amount = self._calculate_position_size(
            account_balance, risk_per_trade_pct, entry_price, risk_distance, atr_pct
        )
        
        # Step 8: Risk Metrics
        if levels.stops and levels.targets:
            levels.risk_reward_ratio = self._calculate_rr_ratio(
                entry_price, levels.stops[0]['price'], levels.targets[0]['price']
            )
            levels.max_risk_reward_ratio = self._calculate_rr_ratio(
                entry_price, levels.stops[0]['price'], levels.targets[-1]['price']
            )
        
        return levels
    
    def _determine_orderflow_bias(self, orderflow: OrderFlowAnalysis) -> str:
        """Determine order flow bias (bullish/bearish/neutral)."""
        from .orderflow_detector import FlowDirection
        
        if orderflow.flow_direction in [FlowDirection.STRONG_BULLISH, FlowDirection.BULLISH]:
            return "bullish"
        elif orderflow.flow_direction in [FlowDirection.STRONG_BEARISH, FlowDirection.BEARISH]:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_stops(
        self,
        levels: RiskLevels,
        ohlcv: pd.DataFrame,
        atr: float,
    ) -> List[Dict[str, Any]]:
        """Calculate stops using structural levels."""
        stops = []
        entry = levels.entry_price
        direction = levels.direction
        
        if direction == "long":
            # Use structure analysis for support levels
            if levels.structure_analysis and levels.structure_analysis.best_support:
                support_price = levels.structure_analysis.best_support.price
                if support_price < entry:
                    stop_price = support_price - (atr * 0.2)  # Small buffer
                    distance_pct = ((entry - stop_price) / entry) * 100
                    
                    if distance_pct <= self.config.max_stop_pct:
                        stops.append({
                            'price': stop_price,
                            'type': 'structural',
                            'reason': f"Below structural support at {support_price:.2f}",
                            'confidence': levels.structure_analysis.best_support.confluence_score / 10,
                            'distance_pct': distance_pct,
                        })
            
            # Fallback: ATR-based stop
            if not stops:
                stop_price = entry - (atr * self.config.atr_stop_multiplier)
                distance_pct = ((entry - stop_price) / entry) * 100
                stops.append({
                    'price': stop_price,
                    'type': 'atr',
                    'reason': f"{self.config.atr_stop_multiplier}x ATR stop",
                    'confidence': 0.6,
                    'distance_pct': distance_pct,
                })
            
            # Multi-tier stops (optional)
            if self.config.enable_multi_tier_stops and len(stops) == 1:
                primary = stops[0]
                
                # Secondary stop (1.5x wider)
                stops.append({
                    'price': entry - (atr * self.config.atr_stop_multiplier * 1.5),
                    'type': 'secondary',
                    'reason': "Secondary stop (wider protection)",
                    'confidence': 0.5,
                    'distance_pct': ((entry - (entry - atr * self.config.atr_stop_multiplier * 1.5)) / entry) * 100,
                })
                
                # Safety net (max loss)
                stops.append({
                    'price': entry * (1 - self.config.max_stop_pct / 100),
                    'type': 'safety_net',
                    'reason': f"Maximum {self.config.max_stop_pct}% loss protection",
                    'confidence': 1.0,
                    'distance_pct': self.config.max_stop_pct,
                })
        
        else:  # short
            # Use structure analysis for resistance levels
            if levels.structure_analysis and levels.structure_analysis.best_resistance:
                resistance_price = levels.structure_analysis.best_resistance.price
                if resistance_price > entry:
                    stop_price = resistance_price + (atr * 0.2)
                    distance_pct = ((stop_price - entry) / entry) * 100
                    
                    if distance_pct <= self.config.max_stop_pct:
                        stops.append({
                            'price': stop_price,
                            'type': 'structural',
                            'reason': f"Above structural resistance at {resistance_price:.2f}",
                            'confidence': levels.structure_analysis.best_resistance.confluence_score / 10,
                            'distance_pct': distance_pct,
                        })
            
            if not stops:
                stop_price = entry + (atr * self.config.atr_stop_multiplier)
                distance_pct = ((stop_price - entry) / entry) * 100
                stops.append({
                    'price': stop_price,
                    'type': 'atr',
                    'reason': f"{self.config.atr_stop_multiplier}x ATR stop",
                    'confidence': 0.6,
                    'distance_pct': distance_pct,
                })
            
            if self.config.enable_multi_tier_stops and len(stops) == 1:
                stops.append({
                    'price': entry + (atr * self.config.atr_stop_multiplier * 1.5),
                    'type': 'secondary',
                    'reason': "Secondary stop (wider protection)",
                    'confidence': 0.5,
                    'distance_pct': ((entry + atr * self.config.atr_stop_multiplier * 1.5 - entry) / entry) * 100,
                })
                
                stops.append({
                    'price': entry * (1 + self.config.max_stop_pct / 100),
                    'type': 'safety_net',
                    'reason': f"Maximum {self.config.max_stop_pct}% loss protection",
                    'confidence': 1.0,
                    'distance_pct': self.config.max_stop_pct,
                })
        
        return stops
    
    def _calculate_targets(
        self,
        levels: RiskLevels,
        ohlcv: pd.DataFrame,
        atr: float,
    ) -> List[Dict[str, Any]]:
        """Calculate targets using structural levels + VPVR."""
        targets = []
        entry = levels.entry_price
        direction = levels.direction
        exit_ratios = self.config.partial_exit_ratios
        
        # Get targets from VPVR (HVN mountains)
        vpvr_targets = []
        if levels.vpvr_analysis:
            vpvr_targets = self.vpvr_analyzer.get_targets(
                levels.vpvr_analysis, direction, entry
            )
        
        # Get targets from structure
        structural_targets = []
        if levels.structure_analysis:
            if direction == "long" and levels.structure_analysis.best_resistance:
                resist_price = levels.structure_analysis.best_resistance.price
                if resist_price > entry:
                    structural_targets.append((resist_price, "Structural resistance"))
            elif direction == "short" and levels.structure_analysis.best_support:
                support_price = levels.structure_analysis.best_support.price
                if support_price < entry:
                    structural_targets.append((support_price, "Structural support"))
        
        # Combine targets
        all_targets = vpvr_targets + structural_targets
        
        # Sort by distance from entry
        all_targets.sort(key=lambda t: abs(t[0] - entry))
        
        # Create target levels with partial exits
        for i, (target_price, reason) in enumerate(all_targets[:3]):
            exit_pct = exit_ratios[i] if i < len(exit_ratios) else exit_ratios[-1]
            distance_pct = abs((target_price - entry) / entry) * 100
            
            targets.append({
                'price': target_price,
                'type': 'structural',
                'reason': reason,
                'exit_percentage': exit_pct * 100,
                'distance_pct': distance_pct,
                'confidence': 0.75,
            })
        
        # If no targets, use R multiples
        if not targets:
            stop_distance = atr * self.config.atr_stop_multiplier
            for i, multiple in enumerate([2.0, 3.0, 5.0]):
                if direction == "long":
                    target_price = entry + (stop_distance * multiple)
                else:
                    target_price = entry - (stop_distance * multiple)
                
                exit_pct = exit_ratios[i] if i < len(exit_ratios) else exit_ratios[-1]
                targets.append({
                    'price': target_price,
                    'type': 'r_multiple',
                    'reason': f"{multiple}R target",
                    'exit_percentage': exit_pct * 100,
                    'distance_pct': abs((target_price - entry) / entry) * 100,
                    'confidence': 0.5,
                })
        
        return targets
    
    def _calculate_position_size(
        self,
        account_balance: float,
        risk_pct: float,
        entry_price: float,
        risk_distance: float,
        atr_pct: float,
    ) -> Tuple[float, float, float]:
        """Calculate position size with volatility adjustment."""
        # Volatility adjustment
        if self.config.volatility_adjusted_sizing:
            vol_factor = 2.0 / max(atr_pct, 0.5)  # Normalize to ~2% ATR
            vol_factor = max(0.5, min(2.0, vol_factor))  # Clamp
            risk_pct *= vol_factor
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_pct / 100)
        
        # Position size = Risk Amount / Risk Distance
        if risk_distance > 0:
            position_size = risk_amount / risk_distance
        else:
            position_size = 0
        
        # Position as percentage of account
        position_value = position_size * entry_price
        position_pct = (position_value / account_balance) * 100
        
        return position_size, position_pct, risk_amount
    
    def _calculate_rr_ratio(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk:reward ratio."""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        return reward / risk if risk > 0 else 0.0
    
    def _calculate_atr(self, ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
        return float(atr)
    
    async def close(self):
        """Close async resources."""
        if self.orderflow_detector:
            await self.orderflow_detector.close()

