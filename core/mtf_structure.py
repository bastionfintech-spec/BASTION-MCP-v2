"""
Multi-Timeframe Structure Analysis
===================================

REAL EDGE: Higher timeframe structure DOMINATES lower timeframe.
Trade direction must align across all timeframes.

Timeframe Hierarchy:
1. WEEKLY/DAILY: Macro S/R (Grade 4 levels) - defines bias
2. 4H/1H: Trendlines and wedge patterns - defines structure
3. 15m: Entry execution on retest rejection - defines timing

Rules:
- Never trade against Weekly/Daily bias
- Enter only when 4H structure supports
- Execute on 15m rejection pattern
- Hold for 4H/Daily target duration (swing trading)

Author: MCF Labs
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

from .structure_detector import StructureDetector, StructureAnalysis, StructureGrade
from .vpvr_analyzer import VPVRAnalyzer, VPVRAnalysis

logger = logging.getLogger(__name__)


class TimeframeBias(str, Enum):
    """Bias direction on a timeframe."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class TimeframeRole(str, Enum):
    """Role of each timeframe."""
    MACRO = "macro"       # Weekly/Daily - defines bias
    STRUCTURE = "structure"  # 4H/1H - defines structure
    EXECUTION = "execution"  # 15m/5m - defines entry


@dataclass
class TimeframeContext:
    """Analysis context for a single timeframe."""
    
    timeframe: str        # '1d', '4h', '1h', '15m', '5m'
    role: TimeframeRole
    
    # Bias
    bias: TimeframeBias = TimeframeBias.NEUTRAL
    bias_strength: float = 0.5  # 0-1
    
    # Structure analysis
    structure: Optional[StructureAnalysis] = None
    
    # Volume profile
    vpvr: Optional[VPVRAnalysis] = None
    
    # Key levels from this timeframe
    major_support: float = 0.0
    major_resistance: float = 0.0
    
    # Trade validity
    allows_long: bool = True
    allows_short: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'timeframe': self.timeframe,
            'role': self.role.value,
            'bias': self.bias.value,
            'bias_strength': self.bias_strength,
            'allows_long': self.allows_long,
            'allows_short': self.allows_short,
        }


@dataclass
class MTFAlignment:
    """Multi-timeframe alignment analysis."""
    
    # Contexts by timeframe
    contexts: Dict[str, TimeframeContext] = field(default_factory=dict)
    
    # Overall alignment
    alignment_score: float = 0.5    # 0-1, how aligned are TFs
    suggested_direction: str = "none"  # 'long', 'short', 'none'
    
    # Trade validity
    can_trade_long: bool = False
    can_trade_short: bool = False
    
    # Conflicts
    conflicts: List[str] = field(default_factory=list)
    
    # Best levels (combined across TFs)
    best_support: float = 0.0
    best_resistance: float = 0.0
    
    # Structure score (0-10)
    mtf_structure_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'alignment_score': self.alignment_score,
            'suggested_direction': self.suggested_direction,
            'can_trade_long': self.can_trade_long,
            'can_trade_short': self.can_trade_short,
            'conflicts': self.conflicts,
            'mtf_structure_score': self.mtf_structure_score,
            'contexts': {tf: ctx.to_dict() for tf, ctx in self.contexts.items()},
        }


class MTFStructureAnalyzer:
    """
    Multi-Timeframe Structure Analyzer.
    
    Analyzes structure across multiple timeframes and determines
    alignment for trade direction.
    
    Usage:
        analyzer = MTFStructureAnalyzer()
        
        alignment = analyzer.analyze({
            '1d': daily_df,
            '4h': h4_df,
            '15m': m15_df,
        })
        
        if alignment.can_trade_long and alignment.alignment_score >= 0.7:
            # Aligned for longs - look for entry
            look_for_long_entry()
    """
    
    # Timeframe roles
    TIMEFRAME_ROLES = {
        '1w': TimeframeRole.MACRO,
        '1d': TimeframeRole.MACRO,
        '4h': TimeframeRole.STRUCTURE,
        '1h': TimeframeRole.STRUCTURE,
        '15m': TimeframeRole.EXECUTION,
        '5m': TimeframeRole.EXECUTION,
    }
    
    # Timeframe weights for alignment
    TIMEFRAME_WEIGHTS = {
        '1w': 0.15,
        '1d': 0.30,
        '4h': 0.30,
        '1h': 0.15,
        '15m': 0.07,
        '5m': 0.03,
    }
    
    def __init__(
        self,
        # Structure detection
        structure_detector: Optional[StructureDetector] = None,
        vpvr_analyzer: Optional[VPVRAnalyzer] = None,
        
        # Alignment thresholds
        min_alignment: float = 0.6,     # Minimum alignment to trade
        
        # Bias detection
        trend_lookback: int = 50,       # Bars to determine bias
        strong_trend_threshold: float = 0.04,  # 4% = strong trend
    ):
        self.structure_detector = structure_detector or StructureDetector()
        self.vpvr_analyzer = vpvr_analyzer or VPVRAnalyzer()
        self.min_alignment = min_alignment
        self.trend_lookback = trend_lookback
        self.strong_trend_threshold = strong_trend_threshold
    
    def analyze(
        self,
        ohlcv_by_tf: Dict[str, pd.DataFrame],
        proposed_direction: Optional[str] = None,
    ) -> MTFAlignment:
        """
        Analyze multi-timeframe structure.
        
        Args:
            ohlcv_by_tf: Dict of timeframe -> OHLCV DataFrame
            proposed_direction: Optional direction to check alignment for
            
        Returns:
            MTFAlignment with all analysis
        """
        alignment = MTFAlignment()
        
        # Analyze each timeframe
        for tf, ohlcv in ohlcv_by_tf.items():
            role = self.TIMEFRAME_ROLES.get(tf, TimeframeRole.EXECUTION)
            ctx = self._analyze_timeframe(tf, role, ohlcv)
            alignment.contexts[tf] = ctx
        
        # Calculate overall alignment
        self._calculate_alignment(alignment, proposed_direction)
        
        # Find best levels
        self._find_best_levels(alignment)
        
        # Calculate MTF structure score
        alignment.mtf_structure_score = self._calculate_mtf_score(alignment)
        
        return alignment
    
    def _analyze_timeframe(
        self,
        tf: str,
        role: TimeframeRole,
        ohlcv: pd.DataFrame,
    ) -> TimeframeContext:
        """Analyze a single timeframe."""
        ctx = TimeframeContext(timeframe=tf, role=role)
        
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        
        # Determine bias
        ctx.bias, ctx.bias_strength = self._determine_bias(close, high, low)
        
        # Set trade permissions based on bias
        if ctx.bias in [TimeframeBias.STRONG_BULLISH, TimeframeBias.BULLISH]:
            ctx.allows_long = True
            ctx.allows_short = role != TimeframeRole.MACRO  # Macro blocks shorts
        elif ctx.bias in [TimeframeBias.STRONG_BEARISH, TimeframeBias.BEARISH]:
            ctx.allows_short = True
            ctx.allows_long = role != TimeframeRole.MACRO  # Macro blocks longs
        else:
            ctx.allows_long = True
            ctx.allows_short = True
        
        # Structure analysis (for STRUCTURE and EXECUTION roles)
        if role in [TimeframeRole.STRUCTURE, TimeframeRole.EXECUTION]:
            ctx.structure = self.structure_detector.analyze(ohlcv)
        
        # VPVR analysis
        ctx.vpvr = self.vpvr_analyzer.analyze(ohlcv)
        
        # Extract key levels
        if ctx.structure:
            if ctx.structure.best_support:
                ctx.major_support = ctx.structure.best_support.price
            if ctx.structure.best_resistance:
                ctx.major_resistance = ctx.structure.best_resistance.price
        
        return ctx
    
    def _determine_bias(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Tuple[TimeframeBias, float]:
        """Determine bias based on price action."""
        if len(close) < self.trend_lookback:
            return TimeframeBias.NEUTRAL, 0.5
        
        # Calculate trend
        start_price = close[-self.trend_lookback]
        end_price = close[-1]
        change_pct = (end_price - start_price) / start_price
        
        # Higher highs and higher lows
        highs = high[-self.trend_lookback:]
        lows = low[-self.trend_lookback:]
        
        # Count HH/HL and LH/LL
        hh_count = 0
        hl_count = 0
        lh_count = 0
        ll_count = 0
        
        for i in range(5, len(highs), 5):
            if highs[i] > highs[i-5]:
                hh_count += 1
            else:
                lh_count += 1
            
            if lows[i] > lows[i-5]:
                hl_count += 1
            else:
                ll_count += 1
        
        # Determine bias
        if change_pct > self.strong_trend_threshold and hh_count > lh_count and hl_count > ll_count:
            return TimeframeBias.STRONG_BULLISH, 0.9
        elif change_pct > self.strong_trend_threshold / 2 and hh_count >= lh_count:
            return TimeframeBias.BULLISH, 0.7
        elif change_pct < -self.strong_trend_threshold and lh_count > hh_count and ll_count > hl_count:
            return TimeframeBias.STRONG_BEARISH, 0.9
        elif change_pct < -self.strong_trend_threshold / 2 and lh_count >= hh_count:
            return TimeframeBias.BEARISH, 0.7
        else:
            return TimeframeBias.NEUTRAL, 0.5
    
    def _calculate_alignment(
        self,
        alignment: MTFAlignment,
        proposed_direction: Optional[str],
    ) -> None:
        """Calculate overall alignment across timeframes."""
        if not alignment.contexts:
            return
        
        # Calculate weighted bias
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        
        for tf, ctx in alignment.contexts.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.1)
            total_weight += weight
            
            if ctx.bias in [TimeframeBias.STRONG_BULLISH, TimeframeBias.BULLISH]:
                bullish_score += weight * ctx.bias_strength
            elif ctx.bias in [TimeframeBias.STRONG_BEARISH, TimeframeBias.BEARISH]:
                bearish_score += weight * ctx.bias_strength
        
        if total_weight > 0:
            bullish_score /= total_weight
            bearish_score /= total_weight
        
        # Determine suggested direction
        if bullish_score > bearish_score + 0.1:
            alignment.suggested_direction = 'long'
            alignment.alignment_score = bullish_score
        elif bearish_score > bullish_score + 0.1:
            alignment.suggested_direction = 'short'
            alignment.alignment_score = bearish_score
        else:
            alignment.suggested_direction = 'none'
            alignment.alignment_score = 0.5
        
        # Check trade validity
        alignment.can_trade_long = all(
            ctx.allows_long for ctx in alignment.contexts.values()
            if ctx.role == TimeframeRole.MACRO
        )
        alignment.can_trade_short = all(
            ctx.allows_short for ctx in alignment.contexts.values()
            if ctx.role == TimeframeRole.MACRO
        )
        
        # Detect conflicts
        macro_biases = [
            ctx.bias for ctx in alignment.contexts.values()
            if ctx.role == TimeframeRole.MACRO
        ]
        structure_biases = [
            ctx.bias for ctx in alignment.contexts.values()
            if ctx.role == TimeframeRole.STRUCTURE
        ]
        
        if macro_biases and structure_biases:
            # Check for conflicting biases
            macro_bullish = any(b in [TimeframeBias.BULLISH, TimeframeBias.STRONG_BULLISH] for b in macro_biases)
            macro_bearish = any(b in [TimeframeBias.BEARISH, TimeframeBias.STRONG_BEARISH] for b in macro_biases)
            structure_bullish = any(b in [TimeframeBias.BULLISH, TimeframeBias.STRONG_BULLISH] for b in structure_biases)
            structure_bearish = any(b in [TimeframeBias.BEARISH, TimeframeBias.STRONG_BEARISH] for b in structure_biases)
            
            if macro_bullish and structure_bearish:
                alignment.conflicts.append("Macro bullish but structure bearish")
            if macro_bearish and structure_bullish:
                alignment.conflicts.append("Macro bearish but structure bullish")
    
    def _find_best_levels(self, alignment: MTFAlignment) -> None:
        """Find best support and resistance across timeframes."""
        supports = []
        resistances = []
        
        for tf, ctx in alignment.contexts.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.1)
            
            if ctx.major_support > 0:
                supports.append((ctx.major_support, weight))
            if ctx.major_resistance > 0:
                resistances.append((ctx.major_resistance, weight))
        
        # Use weighted average (prefer higher TF levels)
        if supports:
            total_weight = sum(w for _, w in supports)
            alignment.best_support = sum(p * w for p, w in supports) / total_weight
        
        if resistances:
            total_weight = sum(w for _, w in resistances)
            alignment.best_resistance = sum(p * w for p, w in resistances) / total_weight
    
    def _calculate_mtf_score(self, alignment: MTFAlignment) -> float:
        """Calculate overall MTF structure score (0-10)."""
        score = 5.0  # Baseline
        
        # Alignment bonus (0-2)
        score += (alignment.alignment_score - 0.5) * 4
        
        # No conflicts bonus (0-1)
        if not alignment.conflicts:
            score += 1.0
        
        # Structure quality bonus (0-2)
        structure_scores = [
            ctx.structure.structure_score 
            for ctx in alignment.contexts.values() 
            if ctx.structure
        ]
        if structure_scores:
            avg_structure = np.mean(structure_scores)
            score += (avg_structure - 5) / 2.5  # -2 to +2
        
        # VPVR alignment bonus (0-1)
        vpvr_good = sum(
            1 for ctx in alignment.contexts.values()
            if ctx.vpvr and ctx.vpvr.volume_score >= 6
        )
        score += min(1.0, vpvr_good * 0.25)
        
        return max(0.0, min(10.0, score))
    
    def check_entry_window(
        self,
        alignment: MTFAlignment,
        execution_tf: str,
        direction: str,
    ) -> Tuple[bool, List[str]]:
        """
        Check if execution timeframe shows entry opportunity.
        
        Args:
            alignment: MTF alignment analysis
            execution_tf: Timeframe to check for entry (e.g., '15m')
            direction: 'long' or 'short'
            
        Returns:
            (is_valid, reasons)
        """
        reasons = []
        
        # Check alignment
        if alignment.alignment_score < self.min_alignment:
            reasons.append(f"Alignment too low: {alignment.alignment_score:.0%}")
            return False, reasons
        
        # Check direction matches
        if direction == 'long' and not alignment.can_trade_long:
            reasons.append("Long blocked by macro bias")
            return False, reasons
        if direction == 'short' and not alignment.can_trade_short:
            reasons.append("Short blocked by macro bias")
            return False, reasons
        
        # Check execution TF structure
        if execution_tf in alignment.contexts:
            ctx = alignment.contexts[execution_tf]
            
            if ctx.structure:
                # Need a pressure point near current price
                if ctx.structure.structure_score < 5.0:
                    reasons.append(f"Execution TF structure weak: {ctx.structure.structure_score:.1f}")
                    return False, reasons
            
            if ctx.vpvr:
                # Need good volume profile
                if ctx.vpvr.volume_score < 5.0:
                    reasons.append(f"VPVR unfavorable: {ctx.vpvr.volume_score:.1f}")
                    return False, reasons
        
        # Check for conflicts
        if alignment.conflicts:
            reasons.extend(alignment.conflicts)
            return False, reasons
        
        reasons.append("All MTF conditions met")
        return True, reasons
    
    def get_mtf_summary(self, alignment: MTFAlignment) -> str:
        """Get human-readable MTF summary."""
        lines = [
            f"MTF Alignment: {alignment.alignment_score:.0%}",
            f"Suggested Direction: {alignment.suggested_direction}",
            f"Can Trade Long: {alignment.can_trade_long}",
            f"Can Trade Short: {alignment.can_trade_short}",
            f"MTF Structure Score: {alignment.mtf_structure_score:.1f}/10",
            "",
            "Timeframe Biases:",
        ]
        
        for tf, ctx in sorted(alignment.contexts.items()):
            lines.append(f"  {tf}: {ctx.bias.value} ({ctx.bias_strength:.0%})")
        
        if alignment.conflicts:
            lines.append("")
            lines.append("Conflicts:")
            for conflict in alignment.conflicts:
                lines.append(f"  - {conflict}")
        
        return "\n".join(lines)








