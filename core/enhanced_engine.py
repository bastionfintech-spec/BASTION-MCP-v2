"""
BASTION Core Risk Engine with MCF Integration
==============================================

Combines:
1. VPVR Analysis (Volume Profile)
2. Structure Detection (Trendlines, Swing Points, Pressure Points)
3. Multi-Timeframe Analysis (MTF Alignment)
4. Order Flow Detection (Liquidity Zones, CVD, Smart Money)
5. Dynamic Stop-Loss & Take-Profit Management

This is the main orchestrator that brings together all MCF components
for institutional-grade risk management.

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
from .structure_detector import StructureDetector, StructureAnalysis, StructureGrade
from .mtf_structure import MTFStructureAnalyzer, MTFAlignment
from .orderflow_detector import OrderFlowDetector, OrderFlowAnalysis

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRiskLevels:
    """Enhanced risk levels with MCF integration."""
    
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
    win_probability: float = 0.45
    expected_value: float = 0.0
    
    # Entry context
    entry_price: float = 0.0
    direction: str = "long"
    timeframe: str = "4h"
    
    # MCF Scores
    structure_score: float = 0.0        # 0-10 from StructureDetector
    volume_score: float = 0.0           # 0-10 from VPVRAnalyzer
    orderflow_score: float = 0.0        # 0-10 from OrderFlowDetector
    mtf_score: float = 0.0              # 0-10 from MTFAlignment
    
    # Overall MCF Score
    mcf_score: float = 0.0              # Weighted composite
    mcf_grade: str = "F"                # A+, A, B+, B, C+, C, F
    
    # Guarding line (swing trades)
    guarding_line: Optional[Dict[str, Any]] = None
    
    # Detailed analyses
    structure_analysis: Optional[StructureAnalysis] = None
    vpvr_analysis: Optional[VPVRAnalysis] = None
    orderflow_analysis: Optional[OrderFlowAnalysis] = None
    mtf_alignment: Optional[MTFAlignment] = None
    
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
            'win_probability': self.win_probability,
            'expected_value': self.expected_value,
            'mcf_score': self.mcf_score,
            'mcf_grade': self.mcf_grade,
            'structure_score': self.structure_score,
            'volume_score': self.volume_score,
            'orderflow_score': self.orderflow_score,
            'mtf_score': self.mtf_score,
            'calculated_at': self.calculated_at.isoformat(),
        }


@dataclass
class EnhancedRiskEngineConfig:
    """Configuration for Enhanced Risk Engine."""
    
    # MCF Integration
    enable_structure_detection: bool = True
    enable_vpvr_analysis: bool = True
    enable_orderflow_detection: bool = True
    enable_mtf_analysis: bool = True
    
    # MCF Weights (for composite score)
    structure_weight: float = 0.35
    volume_weight: float = 0.25
    orderflow_weight: float = 0.15
    mtf_weight: float = 0.25
    
    # Stop-loss settings
    use_structural_stops: bool = True       # Use detected S/R levels
    atr_stop_multiplier: float = 2.0        # Fallback ATR multiplier
    max_stop_pct: float = 5.0               # Maximum stop distance
    
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


class EnhancedRiskEngine:
    """
    BASTION Enhanced Risk Engine with MCF Integration.
    
    This engine combines:
    - Volume Profile (VPVR) for targets and danger zones
    - Structure Detection for trendlines and pressure points
    - Multi-Timeframe Analysis for directional bias
    - Order Flow Detection for institutional footprints
    
    Usage:
        engine = EnhancedRiskEngine()
        
        # Calculate risk levels
        levels = await engine.calculate_risk_levels(
            symbol='BTCUSDT',
            entry_price=94500,
            direction='long',
            timeframe='4h',
            account_balance=100000,
            ohlcv_data={'4h': df_4h, '1d': df_daily}
        )
        
        print(f"MCF Score: {levels.mcf_score:.1f}/10 (Grade: {levels.mcf_grade})")
        print(f"Structure: {levels.structure_score:.1f}/10")
        print(f"Volume: {levels.volume_score:.1f}/10")
        print(f"Order Flow: {levels.orderflow_score:.1f}/10")
    """
    
    def __init__(self, config: Optional[EnhancedRiskEngineConfig] = None):
        self.config = config or EnhancedRiskEngineConfig()
        
        # Initialize MCF components
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
    ) -> EnhancedRiskLevels:
        """
        Calculate complete risk levels with MCF integration.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            entry_price: Proposed entry price
            direction: 'long' or 'short'
            timeframe: Primary timeframe ('4h', '1d', etc.)
            account_balance: Account size in USD
            ohlcv_data: Dict of timeframe -> OHLCV DataFrame
            risk_per_trade_pct: Risk per trade (default 1%)
            
        Returns:
            EnhancedRiskLevels with stops, targets, and MCF scores
        """
        levels = EnhancedRiskLevels(
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
        
        # Step 1: Structure Analysis
        if self.structure_detector:
            levels.structure_analysis = self.structure_detector.analyze(primary_df)
            levels.structure_score = levels.structure_analysis.structure_score
        
        # Step 2: VPVR Analysis
        if self.vpvr_analyzer:
            levels.vpvr_analysis = self.vpvr_analyzer.analyze(
                primary_df, direction=direction
            )
            levels.volume_score = levels.vpvr_analysis.volume_score
        
        # Step 3: Order Flow Analysis
        if self.orderflow_detector:
            levels.orderflow_analysis = await self.orderflow_detector.analyze(
                symbol=symbol, ohlcv=primary_df
            )
            levels.orderflow_score = levels.orderflow_analysis.orderflow_score
        
        # Step 4: Multi-Timeframe Analysis
        if self.mtf_analyzer and len(ohlcv_data) > 1:
            levels.mtf_alignment = self.mtf_analyzer.analyze(
                ohlcv_data, proposed_direction=direction
            )
            levels.mtf_score = levels.mtf_alignment.mtf_structure_score
        
        # Calculate composite MCF Score
        levels.mcf_score = self._calculate_mcf_score(levels)
        levels.mcf_grade = self._mcf_grade(levels.mcf_score)
        
        # Step 5: Calculate Stops (using structural levels)
        levels.stops = self._calculate_enhanced_stops(
            levels, primary_df, atr
        )
        
        # Step 6: Calculate Targets (using structural levels + VPVR)
        levels.targets = self._calculate_enhanced_targets(
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
        
        levels.win_probability = self._estimate_win_probability(levels)
        levels.expected_value = (levels.win_probability * levels.risk_reward_ratio) - ((1 - levels.win_probability) * 1.0)
        
        return levels
    
    def _calculate_mcf_score(self, levels: EnhancedRiskLevels) -> float:
        """Calculate weighted composite MCF score."""
        score = (
            levels.structure_score * self.config.structure_weight +
            levels.volume_score * self.config.volume_weight +
            levels.orderflow_score * self.config.orderflow_weight +
            levels.mtf_score * self.config.mtf_weight
        )
        return max(0.0, min(10.0, score))
    
    def _mcf_grade(self, score: float) -> str:
        """Convert MCF score to letter grade."""
        if score >= 9.0:
            return "A+"
        elif score >= 8.0:
            return "A"
        elif score >= 7.0:
            return "B+"
        elif score >= 6.0:
            return "B"
        elif score >= 5.0:
            return "C+"
        elif score >= 4.0:
            return "C"
        else:
            return "F"
    
    def _calculate_enhanced_stops(
        self,
        levels: EnhancedRiskLevels,
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
                            'type': 'structural_support',
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
                    'type': 'atr_based',
                    'reason': f"{self.config.atr_stop_multiplier}x ATR stop",
                    'confidence': 0.6,
                    'distance_pct': distance_pct,
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
                            'type': 'structural_resistance',
                            'reason': f"Above structural resistance at {resistance_price:.2f}",
                            'confidence': levels.structure_analysis.best_resistance.confluence_score / 10,
                            'distance_pct': distance_pct,
                        })
            
            if not stops:
                stop_price = entry + (atr * self.config.atr_stop_multiplier)
                distance_pct = ((stop_price - entry) / entry) * 100
                stops.append({
                    'price': stop_price,
                    'type': 'atr_based',
                    'reason': f"{self.config.atr_stop_multiplier}x ATR stop",
                    'confidence': 0.6,
                    'distance_pct': distance_pct,
                })
        
        return stops
    
    def _calculate_enhanced_targets(
        self,
        levels: EnhancedRiskLevels,
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
                'type': 'structural_vpvr',
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
    
    def _estimate_win_probability(self, levels: EnhancedRiskLevels) -> float:
        """Estimate win probability based on MCF scores."""
        base_prob = 0.45
        
        # Adjust based on MCF score
        if levels.mcf_score >= 8.0:
            return 0.65  # A+ / A grade
        elif levels.mcf_score >= 6.0:
            return 0.55  # B+ / B grade
        elif levels.mcf_score >= 4.0:
            return 0.45  # C+ / C grade
        else:
            return 0.35  # F grade
    
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

