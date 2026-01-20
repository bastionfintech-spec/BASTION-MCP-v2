"""
BASTION Core Risk Engine - Pure Risk Management
================================================

BASTION is strategy-agnostic risk management.

What it DOES:
✅ Calculate optimal stops (structural, ATR-based, multi-tier)
✅ Calculate optimal targets (structural, volume-informed, dynamic)
✅ Position sizing (volatility-adjusted)
✅ Trade management (trailing stops, partial exits, guarding lines)
✅ Provide market context (structure quality, volume profile, order flow)
✅ Dynamic position updates (living TP, guarding line trailing)

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
from enum import Enum
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


# =============================================================================
# ENUMS
# =============================================================================

class StopType(str, Enum):
    """Types of stop-loss levels."""
    PRIMARY = "primary"          # Structural stop - tightest
    SECONDARY = "secondary"      # Backup stop - gives room to breathe  
    SAFETY_NET = "safety_net"    # Emergency stop - max loss protection
    GUARDING = "guarding"        # Trailing structural stop


class TargetType(str, Enum):
    """Types of take-profit targets."""
    STRUCTURAL = "structural"    # Based on S/R levels
    VPVR = "vpvr"               # Volume profile HVN
    EXTENSION = "extension"      # R-multiple extensions
    DYNAMIC = "dynamic"          # Added after entry as price extends


class StructureHealth(str, Enum):
    """Health status of supporting structure."""
    STRONG = "strong"       # Structure intact, no concerns
    WEAKENING = "weakening" # Structure showing stress
    BROKEN = "broken"       # Structure has failed


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TradeSetup:
    """Input to the RiskEngine representing a potential trade."""
    symbol: str
    entry_price: float
    direction: str              # "long" or "short"
    timeframe: str              # "1m", "5m", "15m", "1h", "4h", "1d"
    account_balance: float = 10000.0
    risk_per_trade_pct: float = 1.0


@dataclass
class PositionUpdate:
    """Real-time position update for dynamic stop/target adjustment."""
    current_price: float
    bars_since_entry: int
    highest_since_entry: float
    lowest_since_entry: float
    unrealized_pnl_pct: float
    recent_lows: Optional[List[float]] = None
    recent_highs: Optional[List[float]] = None


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
    symbol: str = ""
    current_price: float = 0.0
    
    # Market Context (for your strategy to use - informational only)
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
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'direction': self.direction,
            'current_price': self.current_price,
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
            'guarding_line': self.guarding_line,
            'calculated_at': self.calculated_at.isoformat(),
        }
    
    def get_primary_stop(self) -> Optional[Dict]:
        """Get the tightest (primary) stop level."""
        for stop in self.stops:
            if stop.get('type') == 'primary' or stop.get('type') == 'structural':
                return stop
        return self.stops[0] if self.stops else None


@dataclass
class RiskUpdate:
    """Output from dynamic risk updates (position management)."""
    updated_stops: List[Dict[str, Any]] = field(default_factory=list)
    updated_targets: List[Dict[str, Any]] = field(default_factory=list)
    
    # Signals
    exit_signal: bool = False
    exit_reason: Optional[str] = None
    exit_percentage: float = 0.0
    
    # Trailing adjustments
    stop_moved: bool = False
    new_stop_price: Optional[float] = None
    
    # Guarding line status
    guarding_active: bool = False
    guarding_broken: bool = False
    guarding_level: Optional[float] = None
    
    # Structure health
    structure_health: StructureHealth = StructureHealth.STRONG


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RiskEngineConfig:
    """Configuration for Risk Engine."""
    
    # Detection Systems (all optional)
    enable_structure_detection: bool = True
    enable_vpvr_analysis: bool = True
    enable_orderflow_detection: bool = True
    enable_mtf_analysis: bool = True
    
    # Stop-loss settings
    use_structural_stops: bool = True
    atr_stop_multiplier: float = 2.0
    max_stop_pct: float = 5.0
    enable_multi_tier_stops: bool = True
    
    # Take-profit settings
    use_structural_targets: bool = True
    min_rr_ratio: float = 2.0
    enable_partial_exits: bool = True
    partial_exit_ratios: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
    # Position sizing
    default_risk_pct: float = 1.0
    volatility_adjusted_sizing: bool = True
    
    # Guarding line (swing trading)
    enable_guarding_line: bool = True
    guarding_activation_bars: int = 10
    guarding_buffer_pct: float = 0.3
    
    # Living Take-Profit
    enable_dynamic_targets: bool = True
    dynamic_target_threshold: float = 1.5  # Add new target at 1.5R


# =============================================================================
# GUARDING LINE (Integrated)
# =============================================================================

class GuardingLineManager:
    """Trailing structural stop for swing trades."""
    
    def __init__(self, activation_bars: int = 10, buffer_pct: float = 0.3):
        self.activation_bars = activation_bars
        self.buffer_pct = buffer_pct
    
    def calculate_initial_line(
        self,
        entry_price: float,
        direction: str,
        price_data: List[float],
        lookback: int = 20
    ) -> Dict[str, float]:
        """Calculate initial guarding line parameters."""
        if len(price_data) < 5:
            return {
                "slope": 0,
                "intercept": entry_price * (0.97 if direction == "long" else 1.03),
                "activation_bar": self.activation_bars,
                "buffer_pct": self.buffer_pct
            }
        
        recent = price_data[:min(lookback, len(price_data))]
        swing_points = self._find_swing_points(recent, direction)
        
        if len(swing_points) < 2:
            x = np.arange(len(recent))
            y = np.array(recent)
            slope, intercept = np.polyfit(x, y, 1)
        else:
            x = np.array([p[0] for p in swing_points])
            y = np.array([p[1] for p in swing_points])
            slope, intercept = np.polyfit(x, y, 1)
        
        if direction == "long":
            slope = max(0, slope)
            intercept = intercept * (1 - self.buffer_pct / 100)
        else:
            slope = min(0, slope)
            intercept = intercept * (1 + self.buffer_pct / 100)
        
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "activation_bar": self.activation_bars,
            "buffer_pct": self.buffer_pct
        }
    
    def get_current_level(self, line_params: Dict[str, float], bars_since_entry: int) -> float:
        """Get current guarding line level."""
        slope = line_params["slope"]
        intercept = line_params["intercept"]
        activation = line_params.get("activation_bar", self.activation_bars)
        
        if bars_since_entry < activation:
            return intercept * 0.9 if slope >= 0 else intercept * 1.1
        
        bars_active = bars_since_entry - activation
        return intercept + (slope * bars_active)
    
    def check_break(self, current_price: float, guarding_level: float, direction: str) -> Tuple[bool, str]:
        """Check if guarding line is broken."""
        if direction == "long" and current_price < guarding_level:
            return True, f"Price {current_price:.2f} broke below guarding at {guarding_level:.2f}"
        elif direction == "short" and current_price > guarding_level:
            return True, f"Price {current_price:.2f} broke above guarding at {guarding_level:.2f}"
        return False, ""
    
    def _find_swing_points(self, prices: List[float], direction: str) -> List[Tuple[int, float]]:
        """Find swing lows (long) or swing highs (short)."""
        swing_points = []
        for i in range(2, len(prices) - 2):
            if direction == "long":
                if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
                   prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                    swing_points.append((i, prices[i]))
            else:
                if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
                   prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                    swing_points.append((i, prices[i]))
        return swing_points


# =============================================================================
# MAIN RISK ENGINE
# =============================================================================

class RiskEngine:
    """
    BASTION Risk Engine - Strategy-Agnostic Risk Management.
    
    YOU provide:
    - Entry price, direction, account balance, risk tolerance
    
    BASTION provides:
    - Optimal stops (structural + ATR-based)
    - Optimal targets (structural + volume-informed)
    - Position sizing (volatility-adjusted)
    - Market context (structure quality, volume profile, order flow, MTF)
    - Dynamic position updates (guarding line, living TP)
    
    Usage:
        engine = RiskEngine()
        
        levels = await engine.calculate_risk_levels(
            symbol='BTCUSDT',
            entry_price=94500,
            direction='long',
            timeframe='4h',
            account_balance=100000,
            ohlcv_data={'4h': df_4h, '1d': df_daily}
        )
        
        # Later, update position
        update = engine.update_position(levels, position_update)
    """
    
    def __init__(self, config: Optional[RiskEngineConfig] = None):
        self.config = config or RiskEngineConfig()
        
        # Initialize detection systems
        self.structure_detector = StructureDetector() if self.config.enable_structure_detection else None
        self.vpvr_analyzer = VPVRAnalyzer() if self.config.enable_vpvr_analysis else None
        self.orderflow_detector = OrderFlowDetector() if self.config.enable_orderflow_detection else None
        self.mtf_analyzer = MTFStructureAnalyzer() if self.config.enable_mtf_analysis else None
        
        # Guarding line manager
        self.guarding_manager = GuardingLineManager(
            activation_bars=self.config.guarding_activation_bars,
            buffer_pct=self.config.guarding_buffer_pct
        )
    
    async def calculate_risk_levels(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        timeframe: str,
        account_balance: float,
        ohlcv_data: Dict[str, pd.DataFrame],
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
            symbol=symbol,
        )
        
        primary_df = ohlcv_data.get(timeframe)
        if primary_df is None or len(primary_df) < 50:
            logger.error(f"Insufficient data for timeframe {timeframe}")
            return levels
        
        # Current price
        levels.current_price = float(primary_df['close'].iloc[-1])
        
        # Calculate ATR
        atr = self._calculate_atr(primary_df)
        atr_pct = (atr / entry_price) * 100
        
        # Step 1: Structure Analysis
        if self.structure_detector:
            levels.structure_analysis = self.structure_detector.analyze(primary_df)
            levels.structure_quality = levels.structure_analysis.structure_score
        
        # Step 2: VPVR Analysis
        if self.vpvr_analyzer:
            levels.vpvr_analysis = self.vpvr_analyzer.analyze(primary_df, direction=direction)
            levels.volume_profile_score = levels.vpvr_analysis.volume_score
        
        # Step 3: Order Flow Analysis
        if self.orderflow_detector:
            levels.orderflow_analysis = await self.orderflow_detector.analyze(symbol=symbol, ohlcv=primary_df)
            levels.orderflow_bias = self._determine_orderflow_bias(levels.orderflow_analysis)
        
        # Step 4: MTF Analysis
        if self.mtf_analyzer and len(ohlcv_data) > 1:
            levels.mtf_analysis = self.mtf_analyzer.analyze(ohlcv_data, proposed_direction=direction)
            levels.mtf_alignment = levels.mtf_analysis.alignment_score
        
        # Step 5: Calculate Stops
        levels.stops = self._calculate_stops(levels, primary_df, atr)
        
        # Step 6: Calculate Targets
        levels.targets = self._calculate_targets(levels, primary_df, atr)
        
        # Step 7: Guarding Line (for swing timeframes)
        if self.config.enable_guarding_line and self._is_swing_timeframe(timeframe):
            price_data = primary_df['low'].tolist() if direction == "long" else primary_df['high'].tolist()
            levels.guarding_line = self.guarding_manager.calculate_initial_line(
                entry_price, direction, price_data
            )
        
        # Step 8: Position Sizing
        primary_stop_price = levels.stops[0]['price'] if levels.stops else entry_price - (atr * 2)
        risk_distance = abs(entry_price - primary_stop_price)
        
        levels.position_size, levels.position_size_pct, levels.risk_amount = self._calculate_position_size(
            account_balance, risk_per_trade_pct, entry_price, risk_distance, atr_pct
        )
        
        # Step 9: Risk Metrics
        if levels.stops and levels.targets:
            levels.risk_reward_ratio = self._calculate_rr_ratio(
                entry_price, levels.stops[0]['price'], levels.targets[0]['price']
            )
            levels.max_risk_reward_ratio = self._calculate_rr_ratio(
                entry_price, levels.stops[0]['price'], levels.targets[-1]['price']
            )
        
        return levels
    
    def update_position(self, levels: RiskLevels, update: PositionUpdate) -> RiskUpdate:
        """
        Update risk levels based on current price action.
        
        Call this on each bar to get dynamic stop/target adjustments.
        """
        result = RiskUpdate(
            updated_stops=list(levels.stops),
            updated_targets=list(levels.targets),
        )
        
        direction = levels.direction
        entry = levels.entry_price
        current = update.current_price
        
        # Check targets hit
        for target in levels.targets:
            if direction == "long" and current >= target['price']:
                result.exit_signal = True
                result.exit_reason = f"Target hit: {target['reason']}"
                result.exit_percentage = target.get('exit_percentage', 100)
                break
            elif direction == "short" and current <= target['price']:
                result.exit_signal = True
                result.exit_reason = f"Target hit: {target['reason']}"
                result.exit_percentage = target.get('exit_percentage', 100)
                break
        
        # Check guarding line
        if levels.guarding_line and update.bars_since_entry >= self.config.guarding_activation_bars:
            result.guarding_active = True
            result.guarding_level = self.guarding_manager.get_current_level(
                levels.guarding_line, update.bars_since_entry
            )
            
            is_broken, reason = self.guarding_manager.check_break(current, result.guarding_level, direction)
            if is_broken:
                result.guarding_broken = True
                result.exit_signal = True
                result.exit_reason = reason
                result.exit_percentage = 100.0
        
        # Trail stop if in profit
        if update.unrealized_pnl_pct > 0:
            new_stop = self._trail_stop(direction, entry, current, update, levels.stops)
            if new_stop:
                result.stop_moved = True
                result.new_stop_price = new_stop
        
        # Check structure health
        if update.recent_lows and update.recent_highs:
            result.structure_health = self._check_structure_health(
                direction, current, update.recent_lows, update.recent_highs
            )
            if result.structure_health == StructureHealth.BROKEN and not result.exit_signal:
                result.exit_signal = True
                result.exit_reason = "Supporting structure broken"
                result.exit_percentage = 100.0
        
        return result
    
    def _determine_orderflow_bias(self, orderflow: OrderFlowAnalysis) -> str:
        """Determine order flow bias."""
        from .orderflow_detector import FlowDirection
        
        if orderflow.flow_direction in [FlowDirection.STRONG_BULLISH, FlowDirection.BULLISH]:
            return "bullish"
        elif orderflow.flow_direction in [FlowDirection.STRONG_BEARISH, FlowDirection.BEARISH]:
            return "bearish"
        return "neutral"
    
    def _calculate_stops(self, levels: RiskLevels, ohlcv: pd.DataFrame, atr: float) -> List[Dict[str, Any]]:
        """Calculate stops using structural levels."""
        stops = []
        entry = levels.entry_price
        direction = levels.direction
        
        if direction == "long":
            # Use structure for support
            if levels.structure_analysis and levels.structure_analysis.best_support:
                support_price = levels.structure_analysis.best_support.price
                if support_price < entry:
                    stop_price = support_price - (atr * 0.2)
                    distance_pct = ((entry - stop_price) / entry) * 100
                    
                    if distance_pct <= self.config.max_stop_pct:
                        stops.append({
                            'price': stop_price,
                            'type': 'structural',
                            'reason': f"Below structural support at {support_price:.2f}",
                            'confidence': levels.structure_analysis.best_support.confluence_score / 10,
                            'distance_pct': distance_pct,
                        })
            
            # Fallback: ATR-based
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
            
            # Multi-tier stops
            if self.config.enable_multi_tier_stops:
                stops.append({
                    'price': entry - (atr * self.config.atr_stop_multiplier * 1.5),
                    'type': 'secondary',
                    'reason': "Secondary stop (wider protection)",
                    'confidence': 0.5,
                    'distance_pct': (atr * self.config.atr_stop_multiplier * 1.5 / entry) * 100,
                })
                stops.append({
                    'price': entry * (1 - self.config.max_stop_pct / 100),
                    'type': 'safety_net',
                    'reason': f"Maximum {self.config.max_stop_pct}% loss protection",
                    'confidence': 1.0,
                    'distance_pct': self.config.max_stop_pct,
                })
        
        else:  # short
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
            
            if self.config.enable_multi_tier_stops:
                stops.append({
                    'price': entry + (atr * self.config.atr_stop_multiplier * 1.5),
                    'type': 'secondary',
                    'reason': "Secondary stop (wider protection)",
                    'confidence': 0.5,
                    'distance_pct': (atr * self.config.atr_stop_multiplier * 1.5 / entry) * 100,
                })
                stops.append({
                    'price': entry * (1 + self.config.max_stop_pct / 100),
                    'type': 'safety_net',
                    'reason': f"Maximum {self.config.max_stop_pct}% loss protection",
                    'confidence': 1.0,
                    'distance_pct': self.config.max_stop_pct,
                })
        
        return stops
    
    def _calculate_targets(self, levels: RiskLevels, ohlcv: pd.DataFrame, atr: float) -> List[Dict[str, Any]]:
        """Calculate targets using structural levels + VPVR."""
        targets = []
        entry = levels.entry_price
        direction = levels.direction
        exit_ratios = self.config.partial_exit_ratios
        
        # VPVR targets (HVN mountains)
        vpvr_targets = []
        if levels.vpvr_analysis and self.vpvr_analyzer:
            vpvr_targets = self.vpvr_analyzer.get_targets(levels.vpvr_analysis, direction, entry)
        
        # Structural targets
        structural_targets = []
        if levels.structure_analysis:
            if direction == "long" and levels.structure_analysis.best_resistance:
                resist_price = levels.structure_analysis.best_resistance.price
                if resist_price > entry:
                    structural_targets.append((resist_price, "Structural resistance", "structural"))
            elif direction == "short" and levels.structure_analysis.best_support:
                support_price = levels.structure_analysis.best_support.price
                if support_price < entry:
                    structural_targets.append((support_price, "Structural support", "structural"))
        
        # Combine and sort
        all_targets = [(p, r, "vpvr") for p, r in vpvr_targets] + structural_targets
        all_targets.sort(key=lambda t: abs(t[0] - entry))
        
        # Create target levels
        for i, (target_price, reason, ttype) in enumerate(all_targets[:3]):
            exit_pct = exit_ratios[i] if i < len(exit_ratios) else exit_ratios[-1]
            distance_pct = abs((target_price - entry) / entry) * 100
            
            targets.append({
                'price': target_price,
                'type': ttype,
                'reason': reason,
                'exit_percentage': exit_pct * 100,
                'distance_pct': distance_pct,
                'confidence': 0.75,
            })
        
        # Fallback: R multiples
        if not targets:
            stop_distance = atr * self.config.atr_stop_multiplier
            for i, multiple in enumerate([2.0, 3.0, 5.0]):
                target_price = entry + (stop_distance * multiple) if direction == "long" else entry - (stop_distance * multiple)
                exit_pct = exit_ratios[i] if i < len(exit_ratios) else exit_ratios[-1]
                
                targets.append({
                    'price': target_price,
                    'type': 'extension',
                    'reason': f"{multiple}R target",
                    'exit_percentage': exit_pct * 100,
                    'distance_pct': abs((target_price - entry) / entry) * 100,
                    'confidence': 0.5,
                })
        
        return targets
    
    def _calculate_position_size(
        self, account_balance: float, risk_pct: float, entry_price: float,
        risk_distance: float, atr_pct: float
    ) -> Tuple[float, float, float]:
        """Calculate position size with volatility adjustment."""
        if self.config.volatility_adjusted_sizing:
            vol_factor = 2.0 / max(atr_pct, 0.5)
            vol_factor = max(0.5, min(2.0, vol_factor))
            risk_pct *= vol_factor
        
        risk_amount = account_balance * (risk_pct / 100)
        position_size = risk_amount / risk_distance if risk_distance > 0 else 0
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
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        
        return float(np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr))
    
    def _is_swing_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is suitable for swing trading."""
        swing_timeframes = {"4h", "1d", "1w", "4H", "1D", "1W", "daily", "weekly"}
        return timeframe.lower() in {tf.lower() for tf in swing_timeframes}
    
    def _trail_stop(
        self, direction: str, entry: float, current: float,
        update: PositionUpdate, stops: List[Dict]
    ) -> Optional[float]:
        """Calculate trailed stop price."""
        if not stops:
            return None
        
        primary = stops[0]
        primary_distance = primary.get('distance_pct', 2.0)
        
        if direction == "long":
            profit_pct = (current - entry) / entry * 100
            if profit_pct >= primary_distance:
                new_stop = entry * 1.001
                if new_stop > primary['price']:
                    return new_stop
        else:
            profit_pct = (entry - current) / entry * 100
            if profit_pct >= primary_distance:
                new_stop = entry * 0.999
                if new_stop < primary['price']:
                    return new_stop
        
        return None
    
    def _check_structure_health(
        self, direction: str, current: float,
        recent_lows: List[float], recent_highs: List[float]
    ) -> StructureHealth:
        """Check if supporting structure is still intact."""
        if len(recent_lows) < 3 or len(recent_highs) < 3:
            return StructureHealth.STRONG
        
        if direction == "long":
            # Lower lows = weakening
            if recent_lows[0] < recent_lows[1] < recent_lows[2]:
                return StructureHealth.WEAKENING
            # Significant drop
            if (max(recent_highs[-10:]) - current) / current > 0.03:
                return StructureHealth.BROKEN
        else:
            # Higher highs = weakening
            if recent_highs[0] > recent_highs[1] > recent_highs[2]:
                return StructureHealth.WEAKENING
            # Significant rise
            if (current - min(recent_lows[-10:])) / current > 0.03:
                return StructureHealth.BROKEN
        
        return StructureHealth.STRONG
    
    async def close(self):
        """Close async resources."""
        if self.orderflow_detector:
            await self.orderflow_detector.close()
