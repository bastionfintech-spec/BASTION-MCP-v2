"""
VPVR Analyzer - Volume Profile Visible Range
=============================================

REAL EDGE: Price moves FAST through Low Volume Nodes (valleys)
            and SLOW through High Volume Nodes (mountains).

This module implements:
1. Volume Profile calculation from OHLCV data
2. High Volume Node (HVN) detection = "Mountains" = targets/resistance
3. Low Volume Node (LVN) detection = "Valleys" = fast moves
4. POC (Point of Control) = highest volume level
5. Value Area (68% of volume)

Trading Logic:
- Entry: Breaking INTO a valley (LVN ahead)
- Target: Next mountain (HVN ahead)
- Avoid: Trading INTO a mountain (price will stall)

Author: MCF Labs / BASTION
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VolumeNodeType(str, Enum):
    """Type of volume node."""
    HVN = "hvn"      # High Volume Node - Mountain
    LVN = "lvn"      # Low Volume Node - Valley
    POC = "poc"      # Point of Control - Highest volume
    NEUTRAL = "neutral"


@dataclass
class VolumeNode:
    """A volume node at a price level."""
    
    price: float
    volume: float
    node_type: VolumeNodeType
    
    # Price range this node covers
    price_low: float = 0.0
    price_high: float = 0.0
    
    # Relative strength
    volume_pct: float = 0.0     # % of total volume
    z_score: float = 0.0        # How many std devs from mean
    
    def to_dict(self) -> Dict:
        return {
            'price': self.price,
            'volume': self.volume,
            'type': self.node_type.value,
            'volume_pct': self.volume_pct,
            'z_score': self.z_score,
        }


@dataclass
class ValueArea:
    """The Value Area containing 68% of volume."""
    
    vah: float  # Value Area High
    val: float  # Value Area Low
    poc: float  # Point of Control
    
    # Volume metrics
    total_volume: float = 0.0
    value_area_volume: float = 0.0
    value_area_pct: float = 0.68
    
    def price_in_value_area(self, price: float) -> bool:
        """Check if price is within value area."""
        return self.val <= price <= self.vah
    
    def to_dict(self) -> Dict:
        return {
            'vah': self.vah,
            'val': self.val,
            'poc': self.poc,
            'total_volume': self.total_volume,
        }


@dataclass
class VPVRAnalysis:
    """Complete Volume Profile analysis."""

    # Profile data
    price_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    volume_at_price: np.ndarray = field(default_factory=lambda: np.array([]))

    # Buy/sell volume split (Pine Script MCF VPVR style)
    buy_volume_at_price: np.ndarray = field(default_factory=lambda: np.array([]))
    sell_volume_at_price: np.ndarray = field(default_factory=lambda: np.array([]))

    # Detected nodes
    nodes: List[VolumeNode] = field(default_factory=list)
    hvn_nodes: List[VolumeNode] = field(default_factory=list)
    lvn_nodes: List[VolumeNode] = field(default_factory=list)
    poc: Optional[VolumeNode] = None

    # Value Area
    value_area: Optional[ValueArea] = None

    # Current context
    current_price: float = 0.0

    # Volume score (0-10)
    volume_score: float = 5.0

    # Path analysis
    lvn_ahead: bool = False       # Valley ahead (good for entry)
    hvn_ahead: bool = False       # Mountain ahead (target zone)
    distance_to_next_hvn: float = 0.0
    distance_to_next_lvn: float = 0.0

    # Buy/sell dominance context
    buy_sell_dominant: str = "balanced"  # "buy_dominant", "sell_dominant", "balanced"
    
    def get_summary(self) -> Dict:
        return {
            'volume_score': self.volume_score,
            'poc': self.poc.price if self.poc else 0,
            'hvn_count': len(self.hvn_nodes),
            'lvn_count': len(self.lvn_nodes),
            'lvn_ahead': self.lvn_ahead,
            'hvn_ahead': self.hvn_ahead,
            'value_area': self.value_area.to_dict() if self.value_area else {},
        }


class VPVRAnalyzer:
    """
    Volume Profile Visible Range Analyzer.
    
    Calculates volume distribution across price levels and identifies
    High Volume Nodes (mountains) and Low Volume Nodes (valleys).
    
    Usage:
        analyzer = VPVRAnalyzer()
        analysis = analyzer.analyze(ohlcv_df, direction='long')
        
        if analysis.lvn_ahead:
            # Good entry - price will move fast through valley
            enter_trade()
        
        if analysis.hvn_ahead:
            # Set target at the mountain
            target = analysis.hvn_nodes[0].price
    """
    
    def __init__(
        self,
        # Profile settings
        num_bins: int = 250,          # Number of price bins (Pine Script MCF VPVR uses 250)
        lookback_bars: int = 200,     # Bars for profile calculation
        
        # Node detection
        hvn_threshold: float = 1.5,   # Z-score threshold for HVN
        lvn_threshold: float = -0.8,  # Z-score threshold for LVN
        
        # Value Area
        value_area_pct: float = 0.68, # 68% for value area
        
        # Volume weighting
        recency_weight: bool = True,  # Weight recent volume more
        recency_decay: float = 0.95,  # Decay factor per bar
    ):
        self.num_bins = num_bins
        self.lookback_bars = lookback_bars
        self.hvn_threshold = hvn_threshold
        self.lvn_threshold = lvn_threshold
        self.value_area_pct = value_area_pct
        self.recency_weight = recency_weight
        self.recency_decay = recency_decay
    
    def analyze(
        self,
        ohlcv: pd.DataFrame,
        direction: Optional[str] = None,
        current_bar: Optional[int] = None,
    ) -> VPVRAnalysis:
        """
        Perform VPVR analysis.
        
        Args:
            ohlcv: OHLCV DataFrame
            direction: 'long' or 'short' for directional analysis
            current_bar: Bar index to analyze (default: last bar)
            
        Returns:
            VPVRAnalysis with all volume profile data
        """
        if current_bar is None:
            current_bar = len(ohlcv) - 1
        
        # Limit to lookback period
        start_bar = max(0, current_bar - self.lookback_bars + 1)
        data = ohlcv.iloc[start_bar:current_bar + 1]
        
        analysis = VPVRAnalysis()
        analysis.current_price = float(ohlcv['close'].iloc[current_bar])
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        # Step 1: Build volume profile (with buy/sell split)
        profile = self._build_profile(high, low, close, volume)
        analysis.price_bins = profile[0]
        analysis.volume_at_price = profile[1]
        analysis.buy_volume_at_price = profile[2]
        analysis.sell_volume_at_price = profile[3]
        
        # Step 2: Detect nodes
        analysis.nodes = self._detect_nodes(
            analysis.price_bins, analysis.volume_at_price
        )
        
        # Separate by type
        analysis.hvn_nodes = [n for n in analysis.nodes if n.node_type == VolumeNodeType.HVN]
        analysis.lvn_nodes = [n for n in analysis.nodes if n.node_type == VolumeNodeType.LVN]
        analysis.poc = next(
            (n for n in analysis.nodes if n.node_type == VolumeNodeType.POC), 
            None
        )
        
        # Step 3: Calculate Value Area
        analysis.value_area = self._calculate_value_area(
            analysis.price_bins, analysis.volume_at_price
        )
        
        # Step 4: Analyze path ahead
        self._analyze_path(analysis, direction)

        # Step 5: Determine buy/sell dominance near current price
        analysis.buy_sell_dominant = self._get_buy_sell_dominance(
            analysis.price_bins, analysis.buy_volume_at_price,
            analysis.sell_volume_at_price, analysis.current_price
        )

        # Step 6: Calculate volume score
        analysis.volume_score = self._calculate_volume_score(analysis, direction)

        return analysis
    
    def _build_profile(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build volume profile by distributing volume across price bins.

        For each candle, distribute its volume proportionally across
        the price range it covered (high to low).

        Buy/sell split uses Pine Script MCF VPVR formula:
            buy_volume = volume * (close - low) / (high - low)
            sell_volume = volume * (high - close) / (high - low)

        Returns:
            (price_bins, volume_at_price, buy_volume_at_price, sell_volume_at_price)
        """
        # Define price range
        price_min = np.min(low)
        price_max = np.max(high)
        price_range = price_max - price_min

        if price_range == 0:
            total = np.array([np.sum(volume)])
            return np.array([price_min]), total, total.copy(), np.zeros(1)

        # Create bins
        bin_size = price_range / self.num_bins
        price_bins = np.linspace(price_min, price_max, self.num_bins)
        volume_at_price = np.zeros(self.num_bins)
        buy_volume_at_price = np.zeros(self.num_bins)
        sell_volume_at_price = np.zeros(self.num_bins)

        # Apply recency weighting if enabled
        if self.recency_weight:
            weights = np.array([
                self.recency_decay ** (len(volume) - 1 - i)
                for i in range(len(volume))
            ])
            weighted_volume = volume * weights
        else:
            weighted_volume = volume

        # Distribute volume to bins
        for i in range(len(high)):
            candle_low = low[i]
            candle_high = high[i]
            candle_vol = weighted_volume[i]
            candle_range = candle_high - candle_low

            if candle_range == 0:
                # Doji - all volume at close, counted as buy
                bin_idx = int((close[i] - price_min) / bin_size)
                bin_idx = min(bin_idx, self.num_bins - 1)
                volume_at_price[bin_idx] += candle_vol
                buy_volume_at_price[bin_idx] += candle_vol
            else:
                # Pine Script buy/sell split:
                # buy_vol = volume * (close - low) / (high - low)
                # sell_vol = volume * (high - close) / (high - low)
                buy_ratio = (close[i] - candle_low) / candle_range
                sell_ratio = (candle_high - close[i]) / candle_range
                candle_buy_vol = candle_vol * buy_ratio
                candle_sell_vol = candle_vol * sell_ratio

                # Distribute volume proportionally across bins
                for bin_idx in range(self.num_bins):
                    bin_low = price_min + bin_idx * bin_size
                    bin_high = bin_low + bin_size

                    # Calculate overlap
                    overlap_low = max(candle_low, bin_low)
                    overlap_high = min(candle_high, bin_high)

                    if overlap_high > overlap_low:
                        overlap = overlap_high - overlap_low
                        pct_of_candle = overlap / candle_range
                        volume_at_price[bin_idx] += candle_vol * pct_of_candle
                        buy_volume_at_price[bin_idx] += candle_buy_vol * pct_of_candle
                        sell_volume_at_price[bin_idx] += candle_sell_vol * pct_of_candle

        return price_bins, volume_at_price, buy_volume_at_price, sell_volume_at_price

    def _get_buy_sell_dominance(
        self,
        price_bins: np.ndarray,
        buy_volume: np.ndarray,
        sell_volume: np.ndarray,
        current_price: float,
        window_pct: float = 0.02,  # Look at bins within 2% of current price
    ) -> str:
        """
        Determine buy/sell dominance near current price.

        Returns "buy_dominant", "sell_dominant", or "balanced".
        """
        if len(price_bins) == 0 or len(buy_volume) == 0:
            return "balanced"

        # Find bins near current price
        tolerance = current_price * window_pct
        mask = np.abs(price_bins - current_price) <= tolerance

        if not np.any(mask):
            return "balanced"

        total_buy = np.sum(buy_volume[mask])
        total_sell = np.sum(sell_volume[mask])
        total = total_buy + total_sell

        if total == 0:
            return "balanced"

        buy_pct = total_buy / total

        if buy_pct > 0.60:
            return "buy_dominant"
        elif buy_pct < 0.40:
            return "sell_dominant"
        else:
            return "balanced"
    
    def _detect_nodes(
        self,
        price_bins: np.ndarray,
        volume_at_price: np.ndarray,
    ) -> List[VolumeNode]:
        """
        Detect High Volume Nodes (mountains) and Low Volume Nodes (valleys).
        
        Uses z-score to identify statistically significant nodes.
        """
        if len(volume_at_price) == 0:
            return []
        
        nodes = []
        total_volume = np.sum(volume_at_price)
        mean_vol = np.mean(volume_at_price)
        std_vol = np.std(volume_at_price)
        
        if std_vol == 0:
            std_vol = 1e-10
        
        # Find POC first
        poc_idx = np.argmax(volume_at_price)
        
        bin_size = price_bins[1] - price_bins[0] if len(price_bins) > 1 else 0
        
        for i, vol in enumerate(volume_at_price):
            z_score = (vol - mean_vol) / std_vol
            
            price = price_bins[i]
            price_low = price - bin_size / 2
            price_high = price + bin_size / 2
            volume_pct = vol / total_volume if total_volume > 0 else 0
            
            # Determine node type
            if i == poc_idx:
                node_type = VolumeNodeType.POC
            elif z_score >= self.hvn_threshold:
                node_type = VolumeNodeType.HVN
            elif z_score <= self.lvn_threshold:
                node_type = VolumeNodeType.LVN
            else:
                node_type = VolumeNodeType.NEUTRAL
            
            nodes.append(VolumeNode(
                price=price,
                volume=vol,
                node_type=node_type,
                price_low=price_low,
                price_high=price_high,
                volume_pct=volume_pct,
                z_score=z_score,
            ))
        
        return nodes
    
    def _calculate_value_area(
        self,
        price_bins: np.ndarray,
        volume_at_price: np.ndarray,
    ) -> ValueArea:
        """
        Calculate Value Area (68% of volume centered on POC).
        
        Expands outward from POC until value_area_pct of volume is captured.
        """
        if len(volume_at_price) == 0:
            return ValueArea(vah=0, val=0, poc=0)
        
        total_volume = np.sum(volume_at_price)
        target_volume = total_volume * self.value_area_pct
        
        # Start at POC
        poc_idx = np.argmax(volume_at_price)
        poc_price = price_bins[poc_idx]
        
        # Expand outward
        low_idx = poc_idx
        high_idx = poc_idx
        captured_volume = volume_at_price[poc_idx]
        
        while captured_volume < target_volume:
            # Check which direction to expand
            vol_below = volume_at_price[low_idx - 1] if low_idx > 0 else 0
            vol_above = volume_at_price[high_idx + 1] if high_idx < len(volume_at_price) - 1 else 0
            
            if vol_below == 0 and vol_above == 0:
                break
            
            if vol_below >= vol_above and low_idx > 0:
                low_idx -= 1
                captured_volume += volume_at_price[low_idx]
            elif high_idx < len(volume_at_price) - 1:
                high_idx += 1
                captured_volume += volume_at_price[high_idx]
            elif low_idx > 0:
                low_idx -= 1
                captured_volume += volume_at_price[low_idx]
            else:
                break
        
        return ValueArea(
            vah=price_bins[high_idx],
            val=price_bins[low_idx],
            poc=poc_price,
            total_volume=total_volume,
            value_area_volume=captured_volume,
            value_area_pct=captured_volume / total_volume if total_volume > 0 else 0,
        )
    
    def _analyze_path(
        self,
        analysis: VPVRAnalysis,
        direction: Optional[str],
    ) -> None:
        """
        Analyze what's ahead in the direction of the trade.
        
        For longs: look ABOVE current price
        For shorts: look BELOW current price
        """
        current_price = analysis.current_price
        
        if direction == 'long':
            # Look above current price
            hvns_above = [n for n in analysis.hvn_nodes if n.price > current_price]
            lvns_above = [n for n in analysis.lvn_nodes if n.price > current_price]
            
            if hvns_above:
                nearest_hvn = min(hvns_above, key=lambda n: n.price)
                analysis.hvn_ahead = True
                analysis.distance_to_next_hvn = (nearest_hvn.price - current_price) / current_price
            
            if lvns_above:
                nearest_lvn = min(lvns_above, key=lambda n: n.price)
                analysis.lvn_ahead = True
                analysis.distance_to_next_lvn = (nearest_lvn.price - current_price) / current_price
        
        elif direction == 'short':
            # Look below current price
            hvns_below = [n for n in analysis.hvn_nodes if n.price < current_price]
            lvns_below = [n for n in analysis.lvn_nodes if n.price < current_price]
            
            if hvns_below:
                nearest_hvn = max(hvns_below, key=lambda n: n.price)
                analysis.hvn_ahead = True
                analysis.distance_to_next_hvn = (current_price - nearest_hvn.price) / current_price
            
            if lvns_below:
                nearest_lvn = max(lvns_below, key=lambda n: n.price)
                analysis.lvn_ahead = True
                analysis.distance_to_next_lvn = (current_price - nearest_lvn.price) / current_price
    
    def _calculate_volume_score(
        self,
        analysis: VPVRAnalysis,
        direction: Optional[str],
    ) -> float:
        """
        Calculate volume score (0-10).
        
        High score = LVN ahead (price will move fast)
        Low score = HVN ahead (price will stall)
        """
        score = 5.0  # Neutral baseline
        
        # LVN ahead is GOOD (price moves fast through valleys)
        if analysis.lvn_ahead:
            # Closer LVN = higher score
            if analysis.distance_to_next_lvn < 0.01:  # <1% away
                score += 3.0
            elif analysis.distance_to_next_lvn < 0.02:  # <2%
                score += 2.0
            elif analysis.distance_to_next_lvn < 0.05:  # <5%
                score += 1.0
        
        # HVN ahead is BAD (price stalls at mountains)
        if analysis.hvn_ahead:
            # Closer HVN = lower score (unless it's your target)
            if analysis.distance_to_next_hvn < 0.01:  # <1% away - about to hit wall
                score -= 3.0
            elif analysis.distance_to_next_hvn < 0.02:  # <2%
                score -= 1.5
            elif analysis.distance_to_next_hvn < 0.05:  # <5%
                score -= 0.5
        
        # Bonus: In value area = consolidation, breaking out is good
        if analysis.value_area:
            current_price = analysis.current_price
            if not analysis.value_area.price_in_value_area(current_price):
                # Outside value area = potential trend
                if direction == 'long' and current_price > analysis.value_area.vah:
                    score += 1.0  # Breakout above value area
                elif direction == 'short' and current_price < analysis.value_area.val:
                    score += 1.0  # Breakdown below value area
        
        # POC proximity (price tends to return to POC)
        if analysis.poc:
            poc_distance = abs(analysis.current_price - analysis.poc.price) / analysis.poc.price
            if poc_distance > 0.03:  # >3% from POC
                # Extended from POC - reversion possible
                score -= 0.5
        
        return max(0.0, min(10.0, score))
    
    def get_targets(
        self,
        analysis: VPVRAnalysis,
        direction: str,
        entry_price: float,
    ) -> List[Tuple[float, str]]:
        """
        Get target levels based on volume profile.
        
        Returns list of (price, reason) tuples.
        """
        targets = []
        
        if direction == 'long':
            # Targets are HVNs above entry (mountains to reach)
            hvns_above = sorted(
                [n for n in analysis.hvn_nodes if n.price > entry_price],
                key=lambda n: n.price
            )
            for hvn in hvns_above[:3]:
                targets.append((hvn.price, f"HVN mountain (vol z={hvn.z_score:.1f})"))
            
            # Also add VAH if above entry
            if analysis.value_area and analysis.value_area.vah > entry_price:
                targets.append((analysis.value_area.vah, "Value Area High"))
        
        elif direction == 'short':
            hvns_below = sorted(
                [n for n in analysis.hvn_nodes if n.price < entry_price],
                key=lambda n: n.price,
                reverse=True
            )
            for hvn in hvns_below[:3]:
                targets.append((hvn.price, f"HVN mountain (vol z={hvn.z_score:.1f})"))
            
            if analysis.value_area and analysis.value_area.val < entry_price:
                targets.append((analysis.value_area.val, "Value Area Low"))
        
        # Sort by distance from entry
        targets.sort(key=lambda t: abs(t[0] - entry_price))
        
        return targets
    
    def get_danger_zones(
        self,
        analysis: VPVRAnalysis,
        direction: str,
        entry_price: float,
    ) -> List[Tuple[float, str]]:
        """
        Get danger zones where price might stall.
        
        For longs: HVNs between entry and target
        For shorts: HVNs between entry and target
        """
        danger_zones = []
        
        if direction == 'long':
            # HVNs immediately above are danger zones
            hvns = sorted(
                [n for n in analysis.hvn_nodes if n.price > entry_price],
                key=lambda n: n.price
            )
            for hvn in hvns[:2]:
                danger_zones.append((hvn.price, f"HVN resistance (z={hvn.z_score:.1f})"))
        
        elif direction == 'short':
            hvns = sorted(
                [n for n in analysis.hvn_nodes if n.price < entry_price],
                key=lambda n: n.price,
                reverse=True
            )
            for hvn in hvns[:2]:
                danger_zones.append((hvn.price, f"HVN support (z={hvn.z_score:.1f})"))
        
        return danger_zones

