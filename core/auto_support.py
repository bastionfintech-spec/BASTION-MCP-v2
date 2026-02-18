"""
Auto-Support Detector - Priority-Scored Support/Resistance Levels
=================================================================

Ported from MCF Pine Script "Auto-Support New with Enhanced Priority".

Algorithm:
1. Run 16 sensitivity levels (10, 20, 30... 160 bar lookback windows)
2. For each level: find highest high (resistance) and lowest low (support)
3. Score each level by:
   - Close match weight (2.0) — candle close equals the level price
   - High/low match weight (1.0) — wick touches the level price
   - Inverse distance weight (10.0 / |current - level|) — proximity to current price
4. Sort by priority score descending
5. Cluster nearby levels (within 0.3%) and merge their scores

This gives MCF-quality support/resistance levels ranked by structural importance.

Author: MCF Labs / BASTION
Date: February 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class AutoLevel:
    """A priority-scored support or resistance level."""

    price: float
    level_type: str               # "support" or "resistance"
    sensitivity: int              # Which lookback window found it (10, 20, ... 160)
    priority_score: float         # Composite score from Pine formula
    distance_pct: float           # Distance from current price as %

    # Scoring breakdown
    close_matches: int = 0        # How many candle closes equal this level
    wick_touches: int = 0         # How many highs/lows touch this level
    distance_score: float = 0.0   # Inverse distance contribution

    # Clustering
    merged_count: int = 1         # How many raw levels merged into this one

    def to_dict(self) -> Dict:
        return {
            'price': self.price,
            'type': self.level_type,
            'sensitivity': self.sensitivity,
            'priority_score': round(self.priority_score, 2),
            'distance_pct': round(self.distance_pct, 4),
            'close_matches': self.close_matches,
            'wick_touches': self.wick_touches,
            'merged_count': self.merged_count,
        }


@dataclass
class AutoSupportAnalysis:
    """Complete auto-support analysis result."""

    # Priority-sorted levels (highest score first)
    support_levels: List[AutoLevel] = field(default_factory=list)
    resistance_levels: List[AutoLevel] = field(default_factory=list)

    # Quick access to nearest levels
    nearest_support: Optional[AutoLevel] = None
    nearest_resistance: Optional[AutoLevel] = None

    # Top-priority levels (best structural importance regardless of distance)
    top_support: Optional[AutoLevel] = None
    top_resistance: Optional[AutoLevel] = None

    def get_summary(self) -> Dict:
        return {
            'support_count': len(self.support_levels),
            'resistance_count': len(self.resistance_levels),
            'nearest_support': self.nearest_support.to_dict() if self.nearest_support else None,
            'nearest_resistance': self.nearest_resistance.to_dict() if self.nearest_resistance else None,
            'top_support': self.top_support.to_dict() if self.top_support else None,
            'top_resistance': self.top_resistance.to_dict() if self.top_resistance else None,
        }


class AutoSupportDetector:
    """
    MCF Auto-Support Detector with Priority Scoring.

    Direct port of Pine Script "Auto-Support New with Enhanced Priority".
    Uses 16 sensitivity levels with weighted scoring to identify the most
    structurally important support and resistance levels.

    Usage:
        detector = AutoSupportDetector()
        analysis = detector.analyze(ohlcv_df, current_price=95000.0)

        # Best support level
        if analysis.top_support:
            print(f"Key support: ${analysis.top_support.price} (score: {analysis.top_support.priority_score})")
    """

    def __init__(
        self,
        # Pine Script default parameters
        base_sensitivity: int = 10,
        multiplier: int = 1,
        num_levels: int = 16,

        # Scoring weights (from Pine Script)
        close_match_weight: float = 2.0,
        high_low_match_weight: float = 1.0,
        inverse_distance_weight: float = 10.0,
        distance_exponent: float = 1.0,
        base_priority: float = 0.0,

        # Clustering
        cluster_pct: float = 0.003,   # 0.3% — merge levels within this distance

        # Price matching tolerance for crypto (Pine uses exact ==, we need tolerance)
        price_match_pct: float = 0.0005,  # 0.05% tolerance for "equals"
    ):
        self.base_sensitivity = base_sensitivity
        self.multiplier = multiplier
        self.num_levels = num_levels
        self.close_match_weight = close_match_weight
        self.high_low_match_weight = high_low_match_weight
        self.inverse_distance_weight = inverse_distance_weight
        self.distance_exponent = distance_exponent
        self.base_priority = base_priority
        self.cluster_pct = cluster_pct
        self.price_match_pct = price_match_pct

    def analyze(
        self,
        ohlcv: pd.DataFrame,
        current_price: float,
    ) -> AutoSupportAnalysis:
        """
        Perform auto-support analysis.

        Args:
            ohlcv: OHLCV DataFrame with columns: open, high, low, close, volume
            current_price: Current market price for distance scoring

        Returns:
            AutoSupportAnalysis with priority-scored levels
        """
        analysis = AutoSupportAnalysis()

        if len(ohlcv) < self.base_sensitivity:
            return analysis

        high = ohlcv['high'].values.astype(float)
        low = ohlcv['low'].values.astype(float)
        close = ohlcv['close'].values.astype(float)

        raw_supports = []
        raw_resistances = []

        # Run 16 sensitivity passes (matching Pine Script)
        for i in range(self.num_levels):
            sensitivity = self.base_sensitivity * (self.multiplier * (i + 1))

            # Skip if not enough data for this sensitivity level
            if sensitivity > len(ohlcv):
                continue

            # Find resistance = highest high in lookback window
            # Find support = lowest low in lookback window
            # Pine: ta.highest(high, sensitivity) and ta.lowest(low, sensitivity)
            # We use the LAST 'sensitivity' bars (matching Pine's current-bar behavior)
            window_high = high[-sensitivity:]
            window_low = low[-sensitivity:]
            window_close = close[-sensitivity:]

            resistance_price = float(np.max(window_high))
            support_price = float(np.min(window_low))

            # Score resistance
            res_score = self._score_level(
                resistance_price, window_high, window_low, window_close,
                current_price, sensitivity
            )

            raw_resistances.append(AutoLevel(
                price=resistance_price,
                level_type="resistance",
                sensitivity=sensitivity,
                priority_score=res_score['total'],
                distance_pct=abs(current_price - resistance_price) / current_price,
                close_matches=res_score['close_matches'],
                wick_touches=res_score['wick_touches'],
                distance_score=res_score['distance_score'],
            ))

            # Score support
            sup_score = self._score_level(
                support_price, window_high, window_low, window_close,
                current_price, sensitivity
            )

            raw_supports.append(AutoLevel(
                price=support_price,
                level_type="support",
                sensitivity=sensitivity,
                priority_score=sup_score['total'],
                distance_pct=abs(current_price - support_price) / current_price,
                close_matches=sup_score['close_matches'],
                wick_touches=sup_score['wick_touches'],
                distance_score=sup_score['distance_score'],
            ))

        # Cluster nearby levels (merge levels within cluster_pct)
        analysis.resistance_levels = self._cluster_levels(raw_resistances)
        analysis.support_levels = self._cluster_levels(raw_supports)

        # Sort by priority score (highest first)
        analysis.resistance_levels.sort(key=lambda l: l.priority_score, reverse=True)
        analysis.support_levels.sort(key=lambda l: l.priority_score, reverse=True)

        # Set top-priority levels
        if analysis.support_levels:
            analysis.top_support = analysis.support_levels[0]
        if analysis.resistance_levels:
            analysis.top_resistance = analysis.resistance_levels[0]

        # Set nearest levels (closest to current price)
        supports_below = [l for l in analysis.support_levels if l.price < current_price]
        resistances_above = [l for l in analysis.resistance_levels if l.price > current_price]

        if supports_below:
            analysis.nearest_support = min(supports_below, key=lambda l: l.distance_pct)
        if resistances_above:
            analysis.nearest_resistance = min(resistances_above, key=lambda l: l.distance_pct)

        return analysis

    def _score_level(
        self,
        level_price: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        current_price: float,
        sensitivity: int,
    ) -> Dict[str, float]:
        """
        Apply Pine Script priority scoring formula.

        Pine Script logic:
            for j = 0 to sensitivity - 1
                if close[idx] == level_price
                    priority += close_match_weight  (2.0)
                if high[idx] == level_price or low[idx] == level_price
                    priority += high_low_match_weight  (1.0)
            priority += inverse_distance_weight / |close - level_price|^exponent

        We use a small tolerance for price matching (crypto prices are float).
        """
        priority = self.base_priority

        # Tolerance for price matching (Pine uses exact ==, but crypto floats need tolerance)
        tolerance = level_price * self.price_match_pct

        # Count close matches
        close_matches = int(np.sum(np.abs(close - level_price) <= tolerance))
        priority += close_matches * self.close_match_weight

        # Count high/low touches
        high_touches = int(np.sum(np.abs(high - level_price) <= tolerance))
        low_touches = int(np.sum(np.abs(low - level_price) <= tolerance))
        wick_touches = high_touches + low_touches
        priority += wick_touches * self.high_low_match_weight

        # Inverse distance score
        price_distance = abs(current_price - level_price)
        if price_distance > 0:
            distance_score = self.inverse_distance_weight / (price_distance ** self.distance_exponent)
        else:
            distance_score = self.inverse_distance_weight * 100  # At the level = max score

        priority += distance_score

        return {
            'total': priority,
            'close_matches': close_matches,
            'wick_touches': wick_touches,
            'distance_score': distance_score,
        }

    def _cluster_levels(
        self,
        levels: List[AutoLevel],
    ) -> List[AutoLevel]:
        """
        Cluster levels within cluster_pct distance.

        When multiple sensitivity levels find similar prices, merge them
        into a single level with combined priority scores. This prevents
        16 levels at nearly the same price — instead you get one strong level.
        """
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda l: l.price)

        clusters: List[List[AutoLevel]] = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            cluster_price = np.mean([l.price for l in current_cluster])

            if abs(level.price - cluster_price) / cluster_price < self.cluster_pct:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]

        clusters.append(current_cluster)

        # Merge each cluster into a single level
        merged = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                # Weighted average price (weight by priority score)
                total_score = sum(l.priority_score for l in cluster)
                if total_score > 0:
                    avg_price = sum(l.price * l.priority_score for l in cluster) / total_score
                else:
                    avg_price = np.mean([l.price for l in cluster])

                # Sum scores (more sensitivity levels agree = stronger level)
                combined_score = sum(l.priority_score for l in cluster)

                # Use the highest individual sensitivity for reference
                best_level = max(cluster, key=lambda l: l.priority_score)

                merged.append(AutoLevel(
                    price=avg_price,
                    level_type=best_level.level_type,
                    sensitivity=best_level.sensitivity,
                    priority_score=combined_score,
                    distance_pct=best_level.distance_pct,
                    close_matches=sum(l.close_matches for l in cluster),
                    wick_touches=sum(l.wick_touches for l in cluster),
                    distance_score=sum(l.distance_score for l in cluster),
                    merged_count=len(cluster),
                ))

        return merged
