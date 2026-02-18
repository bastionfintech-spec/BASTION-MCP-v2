"""
Structure-Aware Take Profit Sizing Engine

Calculates optimal exit percentages at each TP level based on:
1. Resistance grade at target (Grade 1-4)
2. VPVR zone at target (HVN/LVN/POC)
3. Momentum strength (RSI, CVD trend, OI change)
4. Volume profile buy/sell dominance
5. Distance between TP levels

The model decides WHAT action to take (TP_PARTIAL).
This module decides HOW MUCH to exit — deterministic math, not LLM inference.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TPSizerConfig:
    """Tunable weights for TP sizing calculation."""

    # Base exit percentage at TP1 (before adjustments)
    base_exit_pct: float = 0.30  # 30%

    # Minimum and maximum exit at any single TP level
    min_exit_pct: float = 0.15   # Never less than 15%
    max_exit_pct: float = 0.65   # Never more than 65% at once

    # ── Resistance Grade Adjustments ──
    # Higher grade = harder to break = take more profit
    grade_adjustments: Dict[int, float] = field(default_factory=lambda: {
        4: 0.15,    # Grade 4 (institutional): +15%
        3: 0.08,    # Grade 3 (strong): +8%
        2: 0.00,    # Grade 2 (moderate): neutral
        1: -0.08,   # Grade 1 (weak): -8% (let it run)
        0: -0.05,   # No grade info: slight negative
    })

    # ── VPVR Zone Adjustments ──
    # HVN = price stalls (take more), LVN = price slices through (take less)
    vpvr_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "hvn": 0.10,           # High Volume Node: +10%
        "near_poc": 0.08,      # Near Point of Control: +8%
        "value_area": 0.03,    # Inside value area: +3%
        "outside_va": -0.05,   # Outside value area (trending): -5%
        "lvn": -0.10,          # Low Volume Node: -10% (let it run)
        "unknown": 0.00,       # No data: neutral
    })

    # ── Momentum Adjustments ──
    # Strong momentum = let it run, fading = take more
    momentum_rsi_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "overbought": 0.10,    # RSI > 70: +10% (exhaustion likely)
        "strong": -0.08,       # RSI 55-70: -8% (let it run)
        "neutral": 0.00,       # RSI 40-55: neutral
        "weak": 0.05,          # RSI 30-40: +5% (fading)
        "oversold": 0.08,      # RSI < 30: +8% (bounce taking, take profit)
    })

    # ── Volume Dominance Adjustments ──
    # For LONG positions approaching resistance:
    #   Buy dominant = breakout pressure = take less
    #   Sell dominant = rejection pressure = take more
    dominance_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "buy_dominant": -0.05,   # Pressure to break through: -5%
        "sell_dominant": 0.05,   # Rejection pressure: +5%
        "balanced": 0.00,        # Neutral
    })

    # ── TP Spacing Adjustment ──
    # If TP levels are close together, take more at TP1
    # Measured as % distance between consecutive TPs relative to entry-TP1 distance
    close_tp_threshold: float = 0.50     # TPs within 50% of first TP distance
    close_tp_adjustment: float = 0.08    # +8% if TPs are bunched
    wide_tp_threshold: float = 2.0       # TPs more than 2x first TP distance
    wide_tp_adjustment: float = -0.08    # -8% if TPs are spread wide

    # ── Leverage Risk Multiplier ──
    # Higher leverage = more conservative = take more profit earlier
    leverage_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "extreme": 0.12,   # 20x+: +12% (protect capital)
        "high": 0.06,      # 10x-19x: +6%
        "moderate": 0.00,  # 3x-9x: neutral
        "low": -0.05,      # 1x-2x: -5% (more room to breathe)
    })


# =============================================================================
# TP SIZING ENGINE
# =============================================================================

@dataclass
class TPSizeResult:
    """Result of TP sizing calculation."""
    exit_pct: float                     # Final exit percentage (0.15-0.65)
    base_pct: float                     # Starting base before adjustments
    adjustments: Dict[str, float]       # Each adjustment applied
    trail_stop_price: Optional[float]   # Recommended trailing stop for remainder
    reasoning: str                      # Human-readable explanation
    confidence: float                   # How confident we are in the sizing (0-1)


class StructureTPSizer:
    """
    Calculates optimal exit percentages using structural analysis.

    Usage:
        sizer = StructureTPSizer()
        result = sizer.calculate_exit_pct(
            structure_context=ctx,       # From StructureService
            momentum_data=momentum,      # RSI, CVD, OI
            position_state=pos_state,    # Current position details
            tp_index=0                   # Which TP level (0=first)
        )
        # result.exit_pct = 0.42 (42%)
    """

    def __init__(self, config: Optional[TPSizerConfig] = None):
        self.config = config or TPSizerConfig()

    def calculate_exit_pct(
        self,
        structure_context: Optional[Dict[str, Any]] = None,
        momentum_data: Optional[Dict[str, Any]] = None,
        position_state: Optional[Dict[str, Any]] = None,
        tp_index: int = 0
    ) -> TPSizeResult:
        """
        Calculate the optimal exit percentage for a TP_PARTIAL action.

        Args:
            structure_context: Output from StructureService (nearest S/R, VPVR, trendlines)
            momentum_data: Dict with rsi, cvd_trend, oi_change, etc.
            position_state: Position details (direction, leverage, entry, current, take_profits)
            tp_index: Which TP level we're at (0=first, 1=second, etc.)

        Returns:
            TPSizeResult with the calculated exit_pct and full reasoning
        """
        pos = position_state or {}
        structure = structure_context or {}
        momentum = momentum_data or {}

        direction = pos.get("direction", "LONG").upper()
        leverage = float(pos.get("leverage", 1))
        entry = float(pos.get("entry_price", 0))
        current = float(pos.get("current_price", 0))
        take_profits = pos.get("take_profits", [])

        base = self.config.base_exit_pct
        adjustments = {}

        # ── 1. Resistance Grade at Target ──
        grade_adj = self._grade_adjustment(structure, direction)
        if grade_adj != 0:
            adjustments["resistance_grade"] = grade_adj

        # ── 2. VPVR Zone at Target ──
        vpvr_adj = self._vpvr_adjustment(structure)
        if vpvr_adj != 0:
            adjustments["vpvr_zone"] = vpvr_adj

        # ── 3. Momentum ──
        momentum_adj = self._momentum_adjustment(momentum, direction)
        if momentum_adj != 0:
            adjustments["momentum"] = momentum_adj

        # ── 4. Volume Dominance ──
        dominance_adj = self._dominance_adjustment(structure, direction)
        if dominance_adj != 0:
            adjustments["volume_dominance"] = dominance_adj

        # ── 5. TP Spacing ──
        spacing_adj = self._spacing_adjustment(take_profits, entry, tp_index)
        if spacing_adj != 0:
            adjustments["tp_spacing"] = spacing_adj

        # ── 6. Leverage Risk ──
        leverage_adj = self._leverage_adjustment(leverage)
        if leverage_adj != 0:
            adjustments["leverage_risk"] = leverage_adj

        # ── Calculate Final ──
        total_adjustment = sum(adjustments.values())
        raw_exit = base + total_adjustment
        final_exit = max(self.config.min_exit_pct,
                         min(self.config.max_exit_pct, raw_exit))

        # ── Trailing Stop Recommendation ──
        trail_stop = self._calculate_trail_stop(structure, direction, current)

        # ── Build Reasoning ──
        reasoning = self._build_reasoning(base, adjustments, final_exit, structure, momentum)

        # ── Confidence ──
        # Higher when we have more data points
        data_points = sum(1 for v in adjustments.values() if v != 0)
        total_possible = 6
        confidence = min(0.95, 0.50 + (data_points / total_possible) * 0.45)

        logger.info(f"[TP_SIZER] {direction} exit={final_exit:.0%} "
                    f"(base={base:.0%} + adj={total_adjustment:+.0%}) "
                    f"| {len(adjustments)} factors | conf={confidence:.2f}")

        return TPSizeResult(
            exit_pct=round(final_exit, 3),
            base_pct=base,
            adjustments=adjustments,
            trail_stop_price=trail_stop,
            reasoning=reasoning,
            confidence=confidence
        )

    # ─── Individual Adjustment Calculators ────────────────────────────

    def _grade_adjustment(self, structure: dict, direction: str) -> float:
        """Adjust based on resistance grade at or near target."""
        if direction == "LONG":
            grade = structure.get("nearest_resistance_grade", 0)
        else:
            grade = structure.get("nearest_support_grade", 0)

        return self.config.grade_adjustments.get(grade, 0.0)

    def _vpvr_adjustment(self, structure: dict) -> float:
        """Adjust based on VPVR zone the price is moving into."""
        vpvr_zone = structure.get("vpvr_zone", "unknown")
        return self.config.vpvr_adjustments.get(vpvr_zone, 0.0)

    def _momentum_adjustment(self, momentum: dict, direction: str) -> float:
        """Adjust based on momentum indicators."""
        rsi = momentum.get("rsi", 50)

        # For LONG: overbought = take more, strong = take less
        # For SHORT: oversold = take more, weak = take less (inverted)
        if direction == "SHORT":
            rsi = 100 - rsi  # Invert for short

        if rsi > 70:
            return self.config.momentum_rsi_thresholds["overbought"]
        elif rsi > 55:
            return self.config.momentum_rsi_thresholds["strong"]
        elif rsi > 40:
            return self.config.momentum_rsi_thresholds["neutral"]
        elif rsi > 30:
            return self.config.momentum_rsi_thresholds["weak"]
        else:
            return self.config.momentum_rsi_thresholds["oversold"]

    def _dominance_adjustment(self, structure: dict, direction: str) -> float:
        """Adjust based on buy/sell volume dominance."""
        dominance = structure.get("buy_sell_dominant", "balanced")

        # For LONG approaching resistance:
        #   buy dominant = breakout pressure (take less)
        #   sell dominant = rejection (take more)
        # For SHORT approaching support: invert
        adj = self.config.dominance_adjustments.get(dominance, 0.0)

        if direction == "SHORT":
            adj = -adj  # Invert for short

        return adj

    def _spacing_adjustment(self, take_profits: list, entry: float, tp_index: int) -> float:
        """Adjust based on distance between TP levels."""
        if not take_profits or len(take_profits) < 2 or entry <= 0:
            return 0.0

        try:
            tp_prices = [float(tp) if not isinstance(tp, dict) else float(tp.get("price", 0))
                         for tp in take_profits]
            tp_prices = [p for p in tp_prices if p > 0]

            if len(tp_prices) < 2:
                return 0.0

            # Distance from entry to first TP
            first_dist = abs(tp_prices[0] - entry)
            if first_dist <= 0:
                return 0.0

            # Distance between consecutive TPs
            if tp_index < len(tp_prices) - 1:
                next_dist = abs(tp_prices[tp_index + 1] - tp_prices[tp_index])
                ratio = next_dist / first_dist

                if ratio < self.config.close_tp_threshold:
                    return self.config.close_tp_adjustment  # TPs bunched: take more
                elif ratio > self.config.wide_tp_threshold:
                    return self.config.wide_tp_adjustment   # TPs spread: take less
        except (TypeError, ValueError, IndexError):
            pass

        return 0.0

    def _leverage_adjustment(self, leverage: float) -> float:
        """Adjust based on leverage — higher leverage = more conservative."""
        if leverage >= 20:
            return self.config.leverage_thresholds["extreme"]
        elif leverage >= 10:
            return self.config.leverage_thresholds["high"]
        elif leverage >= 3:
            return self.config.leverage_thresholds["moderate"]
        else:
            return self.config.leverage_thresholds["low"]

    # ─── Trailing Stop Calculator ─────────────────────────────────────

    def _calculate_trail_stop(self, structure: dict, direction: str,
                               current_price: float) -> Optional[float]:
        """Calculate recommended trailing stop after partial TP.

        For LONG: trail just below nearest HVN support or strongest support level
        For SHORT: trail just above nearest HVN resistance or strongest resistance level
        """
        if current_price <= 0:
            return None

        if direction == "LONG":
            # Find the nearest support below current price
            support_price = structure.get("nearest_support_price")
            if support_price and 0 < support_price < current_price:
                # Place trail 0.2% below the support
                buffer = support_price * 0.002
                trail = support_price - buffer
                return round(trail, 2)

            # Fallback: POC if below current price
            poc = structure.get("poc_price")
            if poc and 0 < poc < current_price:
                buffer = poc * 0.003
                return round(poc - buffer, 2)
        else:
            # SHORT: find nearest resistance above current
            resistance_price = structure.get("nearest_resistance_price")
            if resistance_price and resistance_price > current_price:
                buffer = resistance_price * 0.002
                trail = resistance_price + buffer
                return round(trail, 2)

            poc = structure.get("poc_price")
            if poc and poc > current_price:
                buffer = poc * 0.003
                return round(poc + buffer, 2)

        return None

    # ─── Reasoning Builder ────────────────────────────────────────────

    def _build_reasoning(self, base: float, adjustments: dict,
                         final: float, structure: dict, momentum: dict) -> str:
        """Build human-readable reasoning for the TP sizing decision."""
        parts = [f"Base: {base:.0%}"]

        adj_descriptions = {
            "resistance_grade": lambda v: f"Grade {structure.get('nearest_resistance_grade', '?')} resistance: {v:+.0%}",
            "vpvr_zone": lambda v: f"VPVR {structure.get('vpvr_zone', '?')}: {v:+.0%}",
            "momentum": lambda v: f"RSI {momentum.get('rsi', '?')}: {v:+.0%}",
            "volume_dominance": lambda v: f"{structure.get('buy_sell_dominant', '?')} volume: {v:+.0%}",
            "tp_spacing": lambda v: f"TP spacing: {v:+.0%}",
            "leverage_risk": lambda v: f"Leverage risk: {v:+.0%}",
        }

        for key, val in adjustments.items():
            if key in adj_descriptions:
                parts.append(adj_descriptions[key](val))
            else:
                parts.append(f"{key}: {val:+.0%}")

        parts.append(f"→ Exit {final:.0%}")

        return " | ".join(parts)
