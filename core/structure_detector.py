"""
Structure Detection - MCF Ultra Rebuild
========================================

REAL EDGE: Structural exhaustion + liquidity gaps.
Identify pressure points where mature trendline meets volume node.

This module implements:
1. Fractal swing high/low detection (not N-bar extremes)
2. Trendline construction with proper validation
3. Trendline grading (Grade 1-4 based on touches, bipolar status)
4. Pressure point detection (trendline meets horizontal level)
5. Structure staleness tracking

Author: MCF Labs
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SwingType(str, Enum):
    """Type of swing point."""
    HIGH = "high"
    LOW = "low"


class TrendlineType(str, Enum):
    """Type of trendline."""
    SUPPORT = "support"        # Connects lows (ascending or horizontal)
    RESISTANCE = "resistance"  # Connects highs (descending or horizontal)
    BIPOLAR = "bipolar"        # Has acted as both S and R


class StructureGrade(int, Enum):
    """Structure grade based on validation criteria."""
    INVALID = 0       # Slices through candles or < 2 touches
    GRADE_1 = 1       # 2 touches, basic structure
    GRADE_2 = 2       # 3 touches, valid structure
    GRADE_3 = 3       # 3+ touches with clean rejections
    GRADE_4 = 4       # 4+ touches OR bipolar status


@dataclass
class SwingPoint:
    """A validated swing high or low."""
    index: int
    price: float
    swing_type: SwingType
    timestamp: Optional[pd.Timestamp] = None
    
    # Validation metrics
    left_bars: int = 0    # Bars to the left that confirm this swing
    right_bars: int = 0   # Bars to the right that confirm this swing
    strength: float = 0.0 # Relative strength (0-1)
    
    # Staleness
    bars_since_formation: int = 0
    
    def __hash__(self):
        return hash((self.index, self.price, self.swing_type))


@dataclass
class Trendline:
    """A validated trendline connecting swing points."""
    
    # Points that define the line
    anchor_point: SwingPoint      # First point
    secondary_point: SwingPoint   # Second point defining slope
    touch_points: List[SwingPoint] = field(default_factory=list)
    
    # Classification
    line_type: TrendlineType = TrendlineType.SUPPORT
    grade: StructureGrade = StructureGrade.INVALID
    
    # Line equation: price = slope * bar_index + intercept
    slope: float = 0.0
    intercept: float = 0.0
    
    # Validation
    is_valid: bool = False
    slices_candles: bool = False  # If True, line is invalid
    touch_count: int = 0
    
    # Bipolar tracking
    support_touches: int = 0      # Times acted as support
    resistance_touches: int = 0   # Times acted as resistance
    
    # Current status
    is_broken: bool = False
    break_bar: Optional[int] = None
    break_price: Optional[float] = None
    
    # Staleness
    bars_since_last_touch: int = 0
    
    def get_price_at_bar(self, bar_index: int) -> float:
        """Calculate price on trendline at given bar."""
        return self.slope * bar_index + self.intercept
    
    def distance_to_price(self, bar_index: int, price: float) -> float:
        """Get distance from trendline to price (positive = above line)."""
        line_price = self.get_price_at_bar(bar_index)
        return price - line_price
    
    def is_bipolar(self) -> bool:
        """Check if line has acted as both support and resistance."""
        return self.support_touches >= 1 and self.resistance_touches >= 1
    
    def to_dict(self) -> Dict:
        return {
            'type': self.line_type.value,
            'grade': self.grade.value,
            'slope': self.slope,
            'intercept': self.intercept,
            'touch_count': self.touch_count,
            'is_bipolar': self.is_bipolar(),
            'is_broken': self.is_broken,
        }


@dataclass
class HorizontalLevel:
    """A horizontal support/resistance level."""
    
    price: float
    level_type: TrendlineType
    grade: StructureGrade = StructureGrade.GRADE_1
    
    # Touch history
    touches: List[Tuple[int, float]] = field(default_factory=list)  # (bar, price)
    touch_count: int = 0
    
    # Bipolar tracking
    support_touches: int = 0
    resistance_touches: int = 0
    
    # Price tolerance (how close is "at the level")
    tolerance_pct: float = 0.005  # 0.5%
    
    # Staleness
    bars_since_last_touch: int = 0
    
    def is_at_level(self, price: float) -> bool:
        """Check if price is at this level."""
        return abs(price - self.price) / self.price < self.tolerance_pct
    
    def is_bipolar(self) -> bool:
        return self.support_touches >= 1 and self.resistance_touches >= 1


@dataclass
class PressurePoint:
    """A pressure point where trendline meets horizontal level."""
    
    bar_index: int
    price: float
    
    # Components
    trendline: Trendline
    horizontal_level: Optional[HorizontalLevel] = None
    
    # Scoring
    confluence_score: float = 0.0  # 0-10
    
    # Direction
    is_support: bool = True
    is_resistance: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'bar': self.bar_index,
            'price': self.price,
            'confluence_score': self.confluence_score,
            'trendline_grade': self.trendline.grade.value,
            'is_support': self.is_support,
        }


@dataclass
class StructureAnalysis:
    """Complete structure analysis result."""
    
    # Detected elements
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    trendlines: List[Trendline] = field(default_factory=list)
    horizontal_levels: List[HorizontalLevel] = field(default_factory=list)
    pressure_points: List[PressurePoint] = field(default_factory=list)
    
    # Current structure score
    structure_score: float = 0.0  # 0-10
    
    # Best setups
    best_support: Optional[PressurePoint] = None
    best_resistance: Optional[PressurePoint] = None
    
    # Context
    current_bar: int = 0
    current_price: float = 0.0
    
    def get_summary(self) -> Dict:
        return {
            'structure_score': self.structure_score,
            'swing_highs': len(self.swing_highs),
            'swing_lows': len(self.swing_lows),
            'trendlines': len(self.trendlines),
            'valid_trendlines': sum(1 for t in self.trendlines if t.is_valid),
            'horizontal_levels': len(self.horizontal_levels),
            'pressure_points': len(self.pressure_points),
        }


class StructureDetector:
    """
    Detects market structure using fractal swing points and trendlines.
    
    This is the CORE of MCF - identifying pressure points where:
    1. A mature trendline (3+ touches) meets
    2. A horizontal level (multiple touches) at
    3. Current price (within tolerance)
    
    Usage:
        detector = StructureDetector()
        analysis = detector.analyze(ohlcv_df)
        
        if analysis.structure_score >= 7.0:
            # High probability structure setup
            trade()
    """
    
    def __init__(
        self,
        # Swing detection
        swing_lookback: int = 5,       # Bars each side for fractal
        min_swing_strength: float = 0.3,
        
        # Trendline validation
        min_touches: int = 2,          # Minimum for valid line
        touch_tolerance_pct: float = 0.003,  # 0.3% = "touch"
        slice_tolerance_pct: float = 0.001,  # 0.1% = "slice through"
        
        # Horizontal levels
        level_cluster_pct: float = 0.005,  # 0.5% to cluster touches
        min_level_touches: int = 2,
        
        # Staleness
        max_bars_fresh: int = 50,      # Bars before structure gets stale
    ):
        self.swing_lookback = swing_lookback
        self.min_swing_strength = min_swing_strength
        self.min_touches = min_touches
        self.touch_tolerance_pct = touch_tolerance_pct
        self.slice_tolerance_pct = slice_tolerance_pct
        self.level_cluster_pct = level_cluster_pct
        self.min_level_touches = min_level_touches
        self.max_bars_fresh = max_bars_fresh
    
    def analyze(
        self,
        ohlcv: pd.DataFrame,
        current_bar: Optional[int] = None,
    ) -> StructureAnalysis:
        """
        Perform complete structure analysis.
        
        Args:
            ohlcv: OHLCV DataFrame with columns: open, high, low, close, volume
            current_bar: Bar index to analyze (default: last bar)
            
        Returns:
            StructureAnalysis with all detected structure
        """
        if current_bar is None:
            current_bar = len(ohlcv) - 1
        
        analysis = StructureAnalysis(current_bar=current_bar)
        analysis.current_price = float(ohlcv['close'].iloc[current_bar])
        
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values
        open_p = ohlcv['open'].values
        
        # Step 1: Detect swing highs and lows
        analysis.swing_highs = self._detect_swing_points(
            high, low, close, SwingType.HIGH, current_bar
        )
        analysis.swing_lows = self._detect_swing_points(
            high, low, close, SwingType.LOW, current_bar
        )
        
        # Step 2: Construct and validate trendlines
        resistance_lines = self._construct_trendlines(
            analysis.swing_highs, TrendlineType.RESISTANCE,
            high, low, open_p, close, current_bar
        )
        support_lines = self._construct_trendlines(
            analysis.swing_lows, TrendlineType.SUPPORT,
            high, low, open_p, close, current_bar
        )
        analysis.trendlines = resistance_lines + support_lines
        
        # Step 3: Detect horizontal levels
        analysis.horizontal_levels = self._detect_horizontal_levels(
            analysis.swing_highs + analysis.swing_lows,
            high, low, close, current_bar
        )
        
        # Step 4: Find pressure points
        analysis.pressure_points = self._find_pressure_points(
            analysis.trendlines,
            analysis.horizontal_levels,
            current_bar,
            analysis.current_price,
        )
        
        # Step 5: Calculate structure score
        analysis.structure_score = self._calculate_structure_score(
            analysis, current_bar, analysis.current_price
        )
        
        # Step 6: Find best setups
        analysis.best_support, analysis.best_resistance = self._find_best_setups(
            analysis.pressure_points, analysis.current_price
        )
        
        return analysis
    
    def _detect_swing_points(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        swing_type: SwingType,
        current_bar: int,
    ) -> List[SwingPoint]:
        """
        Detect fractal swing points.
        
        A swing high is confirmed when:
        - It's higher than N bars to the left AND
        - It's higher than N bars to the right
        
        This is different from "N-bar high" which just looks left.
        """
        swings = []
        lookback = self.swing_lookback
        
        # Need enough bars on both sides
        for i in range(lookback, current_bar - lookback + 1):
            if swing_type == SwingType.HIGH:
                price = high[i]
                
                # Check left side
                left_valid = all(high[i] > high[i-j] for j in range(1, lookback + 1))
                
                # Check right side
                right_valid = all(high[i] >= high[i+j] for j in range(1, lookback + 1))
                
                if left_valid and right_valid:
                    # Calculate strength based on how much higher than surrounding
                    surrounding = np.concatenate([
                        high[i-lookback:i],
                        high[i+1:i+lookback+1]
                    ])
                    strength = (price - np.mean(surrounding)) / np.std(surrounding + 1e-10)
                    strength = min(1.0, max(0.0, strength / 3))  # Normalize to 0-1
                    
                    if strength >= self.min_swing_strength:
                        swings.append(SwingPoint(
                            index=i,
                            price=price,
                            swing_type=swing_type,
                            left_bars=lookback,
                            right_bars=lookback,
                            strength=strength,
                            bars_since_formation=current_bar - i,
                        ))
            
            else:  # SwingType.LOW
                price = low[i]
                
                left_valid = all(low[i] < low[i-j] for j in range(1, lookback + 1))
                right_valid = all(low[i] <= low[i+j] for j in range(1, lookback + 1))
                
                if left_valid and right_valid:
                    surrounding = np.concatenate([
                        low[i-lookback:i],
                        low[i+1:i+lookback+1]
                    ])
                    strength = (np.mean(surrounding) - price) / np.std(surrounding + 1e-10)
                    strength = min(1.0, max(0.0, strength / 3))
                    
                    if strength >= self.min_swing_strength:
                        swings.append(SwingPoint(
                            index=i,
                            price=price,
                            swing_type=swing_type,
                            left_bars=lookback,
                            right_bars=lookback,
                            strength=strength,
                            bars_since_formation=current_bar - i,
                        ))
        
        return swings
    
    def _construct_trendlines(
        self,
        swings: List[SwingPoint],
        line_type: TrendlineType,
        high: np.ndarray,
        low: np.ndarray,
        open_p: np.ndarray,
        close: np.ndarray,
        current_bar: int,
    ) -> List[Trendline]:
        """
        Construct trendlines from swing points.
        
        For each pair of swing points:
        1. Draw a line between them
        2. Check how many other swings touch the line
        3. Validate the line doesn't slice through candles
        4. Grade the line based on touches and validation
        """
        if len(swings) < 2:
            return []
        
        trendlines = []
        
        # Sort swings by index
        swings = sorted(swings, key=lambda s: s.index)
        
        # Try all pairs (older point as anchor)
        for i, anchor in enumerate(swings[:-1]):
            for secondary in swings[i+1:]:
                # Calculate line parameters
                dx = secondary.index - anchor.index
                if dx == 0:
                    continue
                    
                slope = (secondary.price - anchor.price) / dx
                intercept = anchor.price - slope * anchor.index
                
                trendline = Trendline(
                    anchor_point=anchor,
                    secondary_point=secondary,
                    line_type=line_type,
                    slope=slope,
                    intercept=intercept,
                    touch_count=2,  # Anchor and secondary count
                )
                trendline.touch_points = [anchor, secondary]
                
                # Find additional touches
                for swing in swings:
                    if swing in [anchor, secondary]:
                        continue
                    
                    line_price = trendline.get_price_at_bar(swing.index)
                    distance_pct = abs(swing.price - line_price) / line_price
                    
                    if distance_pct < self.touch_tolerance_pct:
                        trendline.touch_points.append(swing)
                        trendline.touch_count += 1
                
                # Validate: check if line slices through candles
                trendline.slices_candles = self._check_line_slices_candles(
                    trendline, high, low, open_p, close,
                    anchor.index, min(current_bar, secondary.index + 50)
                )
                
                # Determine validity
                if trendline.slices_candles:
                    trendline.is_valid = False
                    trendline.grade = StructureGrade.INVALID
                elif trendline.touch_count < self.min_touches:
                    trendline.is_valid = False
                    trendline.grade = StructureGrade.INVALID
                else:
                    trendline.is_valid = True
                    trendline.grade = self._grade_trendline(trendline)
                
                # Track bipolar touches
                self._track_bipolar_touches(
                    trendline, high, low, close,
                    anchor.index, current_bar
                )
                
                # Calculate staleness
                if trendline.touch_points:
                    last_touch_bar = max(tp.index for tp in trendline.touch_points)
                    trendline.bars_since_last_touch = current_bar - last_touch_bar
                
                # Check if broken
                trendline.is_broken = self._check_line_broken(
                    trendline, close, anchor.index, current_bar
                )
                
                if trendline.is_valid:
                    trendlines.append(trendline)
        
        # Remove duplicates (lines that are nearly identical)
        trendlines = self._deduplicate_trendlines(trendlines)
        
        return trendlines
    
    def _check_line_slices_candles(
        self,
        trendline: Trendline,
        high: np.ndarray,
        low: np.ndarray,
        open_p: np.ndarray,
        close: np.ndarray,
        start_bar: int,
        end_bar: int,
    ) -> bool:
        """
        Check if trendline slices through candle bodies.
        
        Valid trendline: touches wicks or candle edges
        Invalid trendline: cuts through the middle of candle bodies
        """
        slices_count = 0
        total_bars = 0
        
        for bar in range(start_bar, min(end_bar + 1, len(high))):
            line_price = trendline.get_price_at_bar(bar)
            
            # Get candle body bounds
            body_high = max(open_p[bar], close[bar])
            body_low = min(open_p[bar], close[bar])
            body_mid = (body_high + body_low) / 2
            
            # Check if line is in the middle of the body
            if body_low < line_price < body_high:
                # Is it in the MIDDLE of the body? (not just touching)
                body_range = body_high - body_low
                if body_range > 0:
                    relative_pos = (line_price - body_low) / body_range
                    if 0.2 < relative_pos < 0.8:  # Middle 60% of body
                        slices_count += 1
            
            total_bars += 1
        
        # Line is invalid if it slices through more than 10% of candles
        if total_bars > 0:
            slice_ratio = slices_count / total_bars
            return slice_ratio > 0.10
        
        return False
    
    def _track_bipolar_touches(
        self,
        trendline: Trendline,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        start_bar: int,
        end_bar: int,
    ) -> None:
        """Track whether line acted as support, resistance, or both."""
        for bar in range(start_bar, min(end_bar + 1, len(high))):
            line_price = trendline.get_price_at_bar(bar)
            tolerance = line_price * self.touch_tolerance_pct
            
            # Check support touch (price approached from above, held)
            if abs(low[bar] - line_price) < tolerance:
                if close[bar] > line_price:
                    trendline.support_touches += 1
            
            # Check resistance touch (price approached from below, rejected)
            if abs(high[bar] - line_price) < tolerance:
                if close[bar] < line_price:
                    trendline.resistance_touches += 1
        
        # Update line type if bipolar
        if trendline.is_bipolar():
            trendline.line_type = TrendlineType.BIPOLAR
    
    def _check_line_broken(
        self,
        trendline: Trendline,
        close: np.ndarray,
        start_bar: int,
        current_bar: int,
    ) -> bool:
        """Check if line has been broken by a candle CLOSE."""
        for bar in range(start_bar, min(current_bar + 1, len(close))):
            line_price = trendline.get_price_at_bar(bar)
            tolerance = line_price * self.slice_tolerance_pct
            
            if trendline.line_type == TrendlineType.SUPPORT:
                # Support broken when close is decisively below
                if close[bar] < line_price - tolerance:
                    trendline.break_bar = bar
                    trendline.break_price = close[bar]
                    return True
            else:
                # Resistance broken when close is decisively above
                if close[bar] > line_price + tolerance:
                    trendline.break_bar = bar
                    trendline.break_price = close[bar]
                    return True
        
        return False
    
    def _grade_trendline(self, trendline: Trendline) -> StructureGrade:
        """
        Grade trendline based on quality criteria.
        
        Grade 4: 4+ touches OR bipolar status
        Grade 3: 3+ touches with clean rejections
        Grade 2: 3 touches
        Grade 1: 2 touches
        """
        if trendline.touch_count >= 4 or trendline.is_bipolar():
            return StructureGrade.GRADE_4
        elif trendline.touch_count >= 3:
            # Check for clean rejections
            clean_rejections = sum(
                1 for tp in trendline.touch_points 
                if tp.strength >= 0.5
            )
            if clean_rejections >= 2:
                return StructureGrade.GRADE_3
            return StructureGrade.GRADE_2
        elif trendline.touch_count >= 2:
            return StructureGrade.GRADE_1
        
        return StructureGrade.INVALID
    
    def _deduplicate_trendlines(
        self,
        trendlines: List[Trendline],
    ) -> List[Trendline]:
        """Remove trendlines that are nearly identical."""
        if not trendlines:
            return []
        
        unique = []
        
        for line in trendlines:
            is_duplicate = False
            
            for existing in unique:
                # Check if slopes are similar
                slope_diff = abs(line.slope - existing.slope) / (abs(existing.slope) + 1e-10)
                intercept_diff = abs(line.intercept - existing.intercept) / (abs(existing.intercept) + 1e-10)
                
                if slope_diff < 0.1 and intercept_diff < 0.01:
                    # Keep the higher graded one
                    if line.grade.value > existing.grade.value:
                        unique.remove(existing)
                        unique.append(line)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(line)
        
        return unique
    
    def _detect_horizontal_levels(
        self,
        swings: List[SwingPoint],
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        current_bar: int,
    ) -> List[HorizontalLevel]:
        """
        Detect horizontal support/resistance levels.
        
        Cluster swing points at similar prices to identify levels.
        """
        if not swings:
            return []
        
        levels = []
        prices = sorted([s.price for s in swings])
        
        # Cluster prices
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            if abs(price - current_cluster[-1]) / current_cluster[-1] < self.level_cluster_pct:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= self.min_level_touches:
                    clusters.append(current_cluster)
                current_cluster = [price]
        
        if len(current_cluster) >= self.min_level_touches:
            clusters.append(current_cluster)
        
        # Create levels from clusters
        for cluster in clusters:
            avg_price = np.mean(cluster)
            
            # Determine type based on recent price action
            current_price = close[current_bar]
            if current_price > avg_price:
                level_type = TrendlineType.SUPPORT
            else:
                level_type = TrendlineType.RESISTANCE
            
            level = HorizontalLevel(
                price=avg_price,
                level_type=level_type,
                touch_count=len(cluster),
            )
            
            # Track bipolar touches
            for bar in range(max(0, current_bar - 200), current_bar + 1):
                tolerance = avg_price * self.touch_tolerance_pct
                
                if abs(low[bar] - avg_price) < tolerance and close[bar] > avg_price:
                    level.support_touches += 1
                if abs(high[bar] - avg_price) < tolerance and close[bar] < avg_price:
                    level.resistance_touches += 1
            
            # Grade the level
            if level.is_bipolar() or level.touch_count >= 4:
                level.grade = StructureGrade.GRADE_4
            elif level.touch_count >= 3:
                level.grade = StructureGrade.GRADE_3
            else:
                level.grade = StructureGrade.GRADE_2
            
            levels.append(level)
        
        return levels
    
    def _find_pressure_points(
        self,
        trendlines: List[Trendline],
        horizontal_levels: List[HorizontalLevel],
        current_bar: int,
        current_price: float,
    ) -> List[PressurePoint]:
        """
        Find pressure points where structure elements converge.
        
        A pressure point is where:
        1. A trendline projects to current bar AND
        2. (Optional) A horizontal level is nearby
        """
        pressure_points = []
        
        for trendline in trendlines:
            if not trendline.is_valid or trendline.is_broken:
                continue
            
            # Get trendline price at current bar
            line_price = trendline.get_price_at_bar(current_bar)
            
            # Skip if too far from current price
            distance_pct = abs(current_price - line_price) / current_price
            if distance_pct > 0.05:  # More than 5% away
                continue
            
            # Check for horizontal level confluence
            confluent_level = None
            for level in horizontal_levels:
                if abs(level.price - line_price) / line_price < self.level_cluster_pct:
                    confluent_level = level
                    break
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(
                trendline, confluent_level, distance_pct
            )
            
            pressure_points.append(PressurePoint(
                bar_index=current_bar,
                price=line_price,
                trendline=trendline,
                horizontal_level=confluent_level,
                confluence_score=confluence_score,
                is_support=trendline.line_type in [TrendlineType.SUPPORT, TrendlineType.BIPOLAR],
                is_resistance=trendline.line_type in [TrendlineType.RESISTANCE, TrendlineType.BIPOLAR],
            ))
        
        # Sort by confluence score
        pressure_points.sort(key=lambda p: p.confluence_score, reverse=True)
        
        return pressure_points
    
    def _calculate_confluence_score(
        self,
        trendline: Trendline,
        horizontal_level: Optional[HorizontalLevel],
        distance_pct: float,
    ) -> float:
        """Calculate confluence score for a pressure point."""
        score = 0.0
        
        # Trendline grade contribution (0-4 points)
        score += trendline.grade.value
        
        # Bipolar bonus (1 point)
        if trendline.is_bipolar():
            score += 1.0
        
        # Touch count bonus (0-2 points)
        score += min(2.0, (trendline.touch_count - 2) * 0.5)
        
        # Horizontal level bonus (0-3 points)
        if horizontal_level:
            score += horizontal_level.grade.value * 0.5
            if horizontal_level.is_bipolar():
                score += 0.5
        
        # Freshness bonus (0-1 point)
        if trendline.bars_since_last_touch < 20:
            score += 1.0 - (trendline.bars_since_last_touch / 20)
        
        # Proximity bonus (0-1 point) - closer is better
        score += 1.0 - min(1.0, distance_pct * 20)
        
        # Normalize to 0-10
        return min(10.0, score)
    
    def _calculate_structure_score(
        self,
        analysis: StructureAnalysis,
        current_bar: int,
        current_price: float,
    ) -> float:
        """Calculate overall structure score for current price."""
        if not analysis.pressure_points:
            return 0.0
        
        # Base score from best pressure point
        best_score = max(pp.confluence_score for pp in analysis.pressure_points)
        
        # Bonus for multiple valid trendlines
        valid_lines = sum(1 for t in analysis.trendlines if t.is_valid and not t.is_broken)
        multi_line_bonus = min(1.0, valid_lines * 0.2)
        
        # Bonus for grade 4 structures nearby
        grade4_nearby = sum(
            1 for pp in analysis.pressure_points 
            if pp.trendline.grade == StructureGrade.GRADE_4
            and abs(pp.price - current_price) / current_price < 0.02
        )
        grade4_bonus = min(1.0, grade4_nearby * 0.5)
        
        return min(10.0, best_score + multi_line_bonus + grade4_bonus)
    
    def _find_best_setups(
        self,
        pressure_points: List[PressurePoint],
        current_price: float,
    ) -> Tuple[Optional[PressurePoint], Optional[PressurePoint]]:
        """Find best support and resistance pressure points near current price."""
        best_support = None
        best_resistance = None
        
        for pp in pressure_points:
            # Support: below current price
            if pp.is_support and pp.price < current_price:
                if best_support is None or pp.confluence_score > best_support.confluence_score:
                    best_support = pp
            
            # Resistance: above current price
            if pp.is_resistance and pp.price > current_price:
                if best_resistance is None or pp.confluence_score > best_resistance.confluence_score:
                    best_resistance = pp
        
        return best_support, best_resistance
    
    def get_trendline_at_price(
        self,
        analysis: StructureAnalysis,
        target_price: float,
        tolerance_pct: float = 0.01,
    ) -> Optional[Trendline]:
        """Find a trendline at or near the target price."""
        for trendline in analysis.trendlines:
            if not trendline.is_valid:
                continue
            
            line_price = trendline.get_price_at_bar(analysis.current_bar)
            if abs(line_price - target_price) / target_price < tolerance_pct:
                return trendline
        
        return None








