"""
Structure Service — MCF Structural Analysis Orchestrator
=========================================================

Ties together VPVR, Structure Detector, and Auto-Support into a single
cached service that provides token-efficient structural context for
the BASTION risk evaluation prompt.

Replaces arbitrary ATR trailing stops with structure-aware exits.

Flow:
    1. Fetch candles at 15m, 1h, 4h timeframes (parallel, cached)
    2. Run VPVR (250 bins, buy/sell split) per timeframe
    3. Run Structure Detector (asymmetric pivots, trendline grading) per timeframe
    4. Run Auto-Support (16 sensitivity levels, priority scoring) per timeframe
    5. Combine multi-TF results into a StructuralContext
    6. Format as ~350-token text for the system prompt

Caching:
    - 15m: 3 min TTL
    - 1h:  5 min TTL
    - 4h: 10 min TTL
    Structure doesn't change on every 15-second evaluation cycle.

Author: MCF Labs / BASTION
Date: February 2026
"""

import asyncio
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

from .vpvr_analyzer import VPVRAnalyzer, VPVRAnalysis, VolumeNodeType
from .structure_detector import (
    StructureDetector, StructureAnalysis, StructureGrade,
    TrendlineType, Trendline, PressurePoint
)
from .auto_support import AutoSupportDetector, AutoSupportAnalysis, AutoLevel

logger = logging.getLogger(__name__)


@dataclass
class TimeframeStructure:
    """Structure analysis for a single timeframe."""

    timeframe: str
    vpvr: Optional[VPVRAnalysis] = None
    structure: Optional[StructureAnalysis] = None
    auto_support: Optional[AutoSupportAnalysis] = None
    timestamp: float = 0.0


@dataclass
class StructuralContext:
    """Combined multi-timeframe structural context for the model prompt."""

    # Nearest levels (most important for exit decisions)
    nearest_support_price: float = 0.0
    nearest_support_grade: int = 0
    nearest_support_source: str = ""           # e.g. "pivot_trendline_1h"
    nearest_support_distance_pct: float = 0.0

    nearest_resistance_price: float = 0.0
    nearest_resistance_grade: int = 0
    nearest_resistance_source: str = ""
    nearest_resistance_distance_pct: float = 0.0

    # VPVR context
    vpvr_zone: str = "unknown"                 # "hvn", "lvn", "near_poc", "value_area", "outside_va"
    poc_price: float = 0.0
    poc_distance_pct: float = 0.0
    buy_sell_dominant: str = "balanced"         # "buy_dominant", "sell_dominant", "balanced"

    # Active trendlines (unbroken, near current price)
    active_trendlines: List[Dict] = field(default_factory=list)
    strongest_trendline_grade: int = 0

    # Pressure points (confluence of trendline + horizontal)
    pressure_points: List[Dict] = field(default_factory=list)

    # Auto-support top levels
    top_supports: List[Dict] = field(default_factory=list)    # top 3
    top_resistances: List[Dict] = field(default_factory=list)  # top 3

    # MTF bias
    mtf_bias: str = "neutral"
    mtf_alignment: float = 0.0

    # Meta
    timeframes_analyzed: List[str] = field(default_factory=list)
    analysis_time_ms: float = 0.0


class StructureService:
    """
    Central orchestrator for MCF structural analysis.

    Fetches candles, runs VPVR + pivots + auto-support per timeframe,
    caches results, and produces token-efficient prompt text.

    Usage:
        service = StructureService()
        ctx = await service.get_structural_context(
            symbol="BTC", current_price=95000, direction="long", fetcher=fetcher
        )
        prompt_text = service.format_for_prompt(ctx)
    """

    # Timeframes to analyze and their cache TTLs (seconds)
    TIMEFRAME_CONFIG = {
        '15m': {'ttl': 180, 'candles': 200},   # 3 min cache
        '1h':  {'ttl': 300, 'candles': 200},   # 5 min cache
        '4h':  {'ttl': 600, 'candles': 200},   # 10 min cache
    }

    # Timeframe priority for level selection (higher = more important)
    TIMEFRAME_PRIORITY = {
        '15m': 1,
        '1h': 3,
        '4h': 5,
    }

    def __init__(self):
        self.vpvr = VPVRAnalyzer(num_bins=250)
        self.structure = StructureDetector(
            pivot_high_left=20, pivot_high_right=15,
            pivot_low_left=15, pivot_low_right=10,
            max_pivot_history=5,
        )
        self.auto_support = AutoSupportDetector()

        # Cache: key = "{symbol}_{timeframe}" -> (timestamp, TimeframeStructure)
        self._cache: Dict[str, Tuple[float, TimeframeStructure]] = {}

    async def get_structural_context(
        self,
        symbol: str,
        current_price: float,
        direction: str,
        fetcher: Any,
    ) -> Optional[StructuralContext]:
        """
        Main entry point. Fetches candles, runs analysis, returns context.

        Args:
            symbol: Trading pair (e.g., "BTC", "BTCUSDT")
            current_price: Current market price
            direction: "long" or "short"
            fetcher: LiveDataFetcher instance for candle data

        Returns:
            StructuralContext or None if analysis fails
        """
        start_time = time.time()

        # Normalize symbol for API calls
        api_symbol = symbol.upper()
        if not api_symbol.endswith("USDT"):
            api_symbol = f"{api_symbol}USDT"

        try:
            # Check cache and fetch stale timeframes
            tf_structures = {}
            fetch_tasks = {}

            for tf, config in self.TIMEFRAME_CONFIG.items():
                cache_key = f"{api_symbol}_{tf}"
                cached = self._cache.get(cache_key)

                if cached and (time.time() - cached[0]) < config['ttl']:
                    # Cache hit — use cached result
                    tf_structures[tf] = cached[1]
                else:
                    # Cache miss — need to fetch
                    fetch_tasks[tf] = config['candles']

            # Fetch stale timeframes in parallel
            if fetch_tasks:
                candle_data = await self._fetch_candles(
                    api_symbol, fetch_tasks, fetcher
                )

                # Analyze each fetched timeframe
                for tf, ohlcv in candle_data.items():
                    if ohlcv is not None and len(ohlcv) >= 30:
                        tf_struct = self._analyze_timeframe(
                            ohlcv, current_price, tf, direction
                        )
                        tf_structures[tf] = tf_struct

                        # Update cache
                        cache_key = f"{api_symbol}_{tf}"
                        self._cache[cache_key] = (time.time(), tf_struct)

            if not tf_structures:
                logger.warning(f"No structure data available for {symbol}")
                return None

            # Build combined context
            ctx = self._build_context(tf_structures, current_price, direction)
            ctx.analysis_time_ms = (time.time() - start_time) * 1000

            return ctx

        except Exception as e:
            logger.error(f"Structure analysis failed for {symbol}: {e}")
            return None

    async def _fetch_candles(
        self,
        symbol: str,
        tf_limits: Dict[str, int],
        fetcher: Any,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch candles for multiple timeframes in parallel."""
        results = {}

        async def _fetch_one(tf: str, limit: int):
            try:
                df = await fetcher.get_ohlcv(symbol, tf, limit)
                return tf, df
            except Exception as e:
                logger.warning(f"Failed to fetch {tf} candles for {symbol}: {e}")
                return tf, None

        tasks = [_fetch_one(tf, limit) for tf, limit in tf_limits.items()]
        done = await asyncio.gather(*tasks, return_exceptions=True)

        for result in done:
            if isinstance(result, Exception):
                continue
            tf, df = result
            results[tf] = df

        return results

    def _analyze_timeframe(
        self,
        ohlcv: pd.DataFrame,
        current_price: float,
        timeframe: str,
        direction: str,
    ) -> TimeframeStructure:
        """Run all 3 analyzers on one timeframe's candle data."""
        tf_struct = TimeframeStructure(timeframe=timeframe, timestamp=time.time())

        try:
            # VPVR analysis (250 bins, buy/sell split)
            tf_struct.vpvr = self.vpvr.analyze(ohlcv, direction=direction)
        except Exception as e:
            logger.warning(f"VPVR analysis failed on {timeframe}: {e}")

        try:
            # Structure detection (asymmetric pivots, trendlines, pressure points)
            tf_struct.structure = self.structure.analyze(ohlcv)
        except Exception as e:
            logger.warning(f"Structure detection failed on {timeframe}: {e}")

        try:
            # Auto-support (16 sensitivity levels, priority scoring)
            tf_struct.auto_support = self.auto_support.analyze(ohlcv, current_price)
        except Exception as e:
            logger.warning(f"Auto-support failed on {timeframe}: {e}")

        return tf_struct

    def _build_context(
        self,
        tf_structures: Dict[str, TimeframeStructure],
        current_price: float,
        direction: str,
    ) -> StructuralContext:
        """Combine multi-TF structure results into a single context."""
        ctx = StructuralContext()
        ctx.timeframes_analyzed = list(tf_structures.keys())

        # Collect all support/resistance candidates across timeframes
        support_candidates = []   # (price, grade, source, distance_pct, priority)
        resistance_candidates = []

        for tf, tf_struct in tf_structures.items():
            tf_priority = self.TIMEFRAME_PRIORITY.get(tf, 1)

            # --- From Structure Detector ---
            if tf_struct.structure:
                struct = tf_struct.structure

                # Pressure points (trendline + horizontal confluence)
                for pp in struct.pressure_points[:3]:
                    entry = {
                        'price': round(pp.price, 2),
                        'confluence': round(pp.confluence_score, 1),
                        'grade': pp.trendline.grade.value,
                        'timeframe': tf,
                        'is_support': pp.is_support,
                    }
                    ctx.pressure_points.append(entry)

                    dist_pct = abs(current_price - pp.price) / current_price

                    if pp.is_support and pp.price < current_price:
                        support_candidates.append((
                            pp.price, pp.trendline.grade.value,
                            f"pressure_point_{tf}", dist_pct,
                            pp.confluence_score * tf_priority
                        ))
                    if pp.is_resistance and pp.price > current_price:
                        resistance_candidates.append((
                            pp.price, pp.trendline.grade.value,
                            f"pressure_point_{tf}", dist_pct,
                            pp.confluence_score * tf_priority
                        ))

                # Active trendlines (unbroken, near current price)
                for tl in struct.trendlines:
                    if not tl.is_valid or tl.is_broken:
                        continue
                    tl_price = tl.get_price_at_bar(struct.current_bar)
                    dist_pct = abs(current_price - tl_price) / current_price
                    if dist_pct > 0.05:
                        continue

                    slope_dir = "ascending" if tl.slope > 0 else ("descending" if tl.slope < 0 else "horizontal")
                    entry = {
                        'grade': tl.grade.value,
                        'type': tl.line_type.value,
                        'slope': slope_dir,
                        'price': round(tl_price, 2),
                        'timeframe': tf,
                        'touches': tl.touch_count,
                        'bipolar': tl.is_bipolar(),
                    }
                    ctx.active_trendlines.append(entry)

                    if tl.grade.value > ctx.strongest_trendline_grade:
                        ctx.strongest_trendline_grade = tl.grade.value

                    # Also add to support/resistance candidates
                    if tl.line_type in [TrendlineType.SUPPORT, TrendlineType.BIPOLAR] and tl_price < current_price:
                        support_candidates.append((
                            tl_price, tl.grade.value,
                            f"trendline_{tf}", dist_pct,
                            tl.grade.value * tf_priority
                        ))
                    if tl.line_type in [TrendlineType.RESISTANCE, TrendlineType.BIPOLAR] and tl_price > current_price:
                        resistance_candidates.append((
                            tl_price, tl.grade.value,
                            f"trendline_{tf}", dist_pct,
                            tl.grade.value * tf_priority
                        ))

            # --- From Auto-Support ---
            if tf_struct.auto_support:
                asup = tf_struct.auto_support

                # Top supports
                for level in (asup.support_levels or [])[:3]:
                    if level.price < current_price:
                        dist_pct = abs(current_price - level.price) / current_price
                        ctx.top_supports.append({
                            'price': round(level.price, 2),
                            'score': round(level.priority_score, 1),
                            'timeframe': tf,
                            'merged': level.merged_count,
                        })
                        support_candidates.append((
                            level.price, 2,  # auto-support = Grade 2 equivalent
                            f"auto_support_{tf}", dist_pct,
                            level.priority_score * tf_priority * 0.5
                        ))

                # Top resistances
                for level in (asup.resistance_levels or [])[:3]:
                    if level.price > current_price:
                        dist_pct = abs(current_price - level.price) / current_price
                        ctx.top_resistances.append({
                            'price': round(level.price, 2),
                            'score': round(level.priority_score, 1),
                            'timeframe': tf,
                            'merged': level.merged_count,
                        })
                        resistance_candidates.append((
                            level.price, 2,
                            f"auto_support_{tf}", dist_pct,
                            level.priority_score * tf_priority * 0.5
                        ))

            # --- From VPVR ---
            if tf_struct.vpvr:
                vpvr = tf_struct.vpvr

                # Use highest-priority timeframe for VPVR context
                if tf_priority >= self.TIMEFRAME_PRIORITY.get(ctx.timeframes_analyzed[0], 0):
                    # Determine VPVR zone
                    ctx.vpvr_zone = self._determine_vpvr_zone(vpvr, current_price)
                    ctx.buy_sell_dominant = vpvr.buy_sell_dominant

                    if vpvr.poc:
                        ctx.poc_price = vpvr.poc.price
                        ctx.poc_distance_pct = abs(current_price - vpvr.poc.price) / current_price

                    # HVN nodes as support/resistance
                    for hvn in vpvr.hvn_nodes:
                        dist_pct = abs(current_price - hvn.price) / current_price
                        if dist_pct > 0.05:
                            continue
                        if hvn.price < current_price:
                            support_candidates.append((
                                hvn.price, 2,
                                f"vpvr_hvn_{tf}", dist_pct,
                                hvn.z_score * tf_priority
                            ))
                        else:
                            resistance_candidates.append((
                                hvn.price, 2,
                                f"vpvr_hvn_{tf}", dist_pct,
                                hvn.z_score * tf_priority
                            ))

        # Select best nearest support/resistance
        if support_candidates:
            # Sort by weighted priority (consider both proximity and structural importance)
            support_candidates.sort(key=lambda c: c[4], reverse=True)
            best = support_candidates[0]
            ctx.nearest_support_price = best[0]
            ctx.nearest_support_grade = best[1]
            ctx.nearest_support_source = best[2]
            ctx.nearest_support_distance_pct = best[3]

        if resistance_candidates:
            resistance_candidates.sort(key=lambda c: c[4], reverse=True)
            best = resistance_candidates[0]
            ctx.nearest_resistance_price = best[0]
            ctx.nearest_resistance_grade = best[1]
            ctx.nearest_resistance_source = best[2]
            ctx.nearest_resistance_distance_pct = best[3]

        # Deduplicate and sort top levels
        ctx.top_supports = self._dedupe_levels(ctx.top_supports)[:3]
        ctx.top_resistances = self._dedupe_levels(ctx.top_resistances)[:3]

        # Sort pressure points by confluence score
        ctx.pressure_points.sort(key=lambda p: p['confluence'], reverse=True)
        ctx.pressure_points = ctx.pressure_points[:3]

        # Sort active trendlines by grade
        ctx.active_trendlines.sort(key=lambda t: t['grade'], reverse=True)
        ctx.active_trendlines = ctx.active_trendlines[:4]

        # Simple MTF bias from trendline direction
        ctx.mtf_bias, ctx.mtf_alignment = self._determine_mtf_bias(tf_structures)

        return ctx

    def _determine_vpvr_zone(
        self,
        vpvr: VPVRAnalysis,
        current_price: float,
    ) -> str:
        """Determine what VPVR zone current price is in."""
        # Check POC proximity
        if vpvr.poc and abs(current_price - vpvr.poc.price) / current_price < 0.005:
            return "near_poc"

        # Check if in HVN
        for node in vpvr.hvn_nodes:
            if node.price_low <= current_price <= node.price_high:
                return "hvn"

        # Check if in LVN
        for node in vpvr.lvn_nodes:
            if node.price_low <= current_price <= node.price_high:
                return "lvn"

        # Check value area
        if vpvr.value_area and vpvr.value_area.price_in_value_area(current_price):
            return "value_area"

        return "outside_va"

    def _determine_mtf_bias(
        self,
        tf_structures: Dict[str, TimeframeStructure],
    ) -> Tuple[str, float]:
        """Determine multi-timeframe bias from trendline directions."""
        bullish_weight = 0.0
        bearish_weight = 0.0
        total_weight = 0.0

        for tf, tf_struct in tf_structures.items():
            tf_weight = self.TIMEFRAME_PRIORITY.get(tf, 1)

            if tf_struct.structure:
                for tl in tf_struct.structure.trendlines:
                    if not tl.is_valid or tl.is_broken:
                        continue
                    if tl.line_type == TrendlineType.SUPPORT and tl.slope > 0:
                        bullish_weight += tl.grade.value * tf_weight
                    elif tl.line_type == TrendlineType.RESISTANCE and tl.slope < 0:
                        bearish_weight += tl.grade.value * tf_weight
                    total_weight += tl.grade.value * tf_weight

        if total_weight == 0:
            return "neutral", 0.5

        bullish_pct = bullish_weight / total_weight
        bearish_pct = bearish_weight / total_weight

        if bullish_pct > 0.6:
            return "bullish", bullish_pct
        elif bearish_pct > 0.6:
            return "bearish", bearish_pct
        else:
            return "neutral", max(bullish_pct, bearish_pct)

    def _dedupe_levels(self, levels: List[Dict], cluster_pct: float = 0.005) -> List[Dict]:
        """Remove duplicate levels that are within cluster_pct of each other."""
        if not levels:
            return []

        # Sort by score descending
        sorted_levels = sorted(levels, key=lambda l: l.get('score', 0), reverse=True)
        unique = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            is_dupe = False
            for existing in unique:
                if abs(level['price'] - existing['price']) / existing['price'] < cluster_pct:
                    is_dupe = True
                    break
            if not is_dupe:
                unique.append(level)

        return unique

    def format_for_prompt(self, ctx: StructuralContext) -> str:
        """
        Format structural context as concise text for the system prompt.

        Target: ~350 tokens (approximately 1500 characters).
        Must be scannable by the model in a single pass.
        """
        lines = []
        lines.append(f"STRUCTURAL ANALYSIS ({'/'.join(ctx.timeframes_analyzed)}):")

        # Nearest support
        if ctx.nearest_support_price > 0:
            src = ctx.nearest_support_source.replace('_', ' ')
            lines.append(
                f"NEAREST SUPPORT: ${ctx.nearest_support_price:,.0f} "
                f"(Grade {ctx.nearest_support_grade} {src}, "
                f"{ctx.nearest_support_distance_pct*100:.1f}% below)"
            )

        # Nearest resistance
        if ctx.nearest_resistance_price > 0:
            src = ctx.nearest_resistance_source.replace('_', ' ')
            lines.append(
                f"NEAREST RESISTANCE: ${ctx.nearest_resistance_price:,.0f} "
                f"(Grade {ctx.nearest_resistance_grade} {src}, "
                f"{ctx.nearest_resistance_distance_pct*100:.1f}% above)"
            )

        # VPVR zone
        zone_descriptions = {
            'hvn': 'In HVN (high volume node) \u2014 price will stall here.',
            'lvn': 'In LVN (low volume node) \u2014 expect fast price movement.',
            'near_poc': 'Near POC (point of control) \u2014 equilibrium zone.',
            'value_area': 'Inside Value Area \u2014 normal trading range.',
            'outside_va': 'Outside Value Area \u2014 extended from equilibrium.',
        }
        zone_desc = zone_descriptions.get(ctx.vpvr_zone, '')
        if zone_desc:
            vpvr_line = f"VPVR: {zone_desc}"
            if ctx.poc_price > 0:
                vpvr_line += f" POC ${ctx.poc_price:,.0f} ({ctx.poc_distance_pct*100:.1f}% away)."
            vpvr_line += f" {ctx.buy_sell_dominant.replace('_', ' ').title()}."
            lines.append(vpvr_line)

        # Active trendlines (top 2)
        if ctx.active_trendlines:
            tl_parts = []
            for tl in ctx.active_trendlines[:2]:
                bp = " (bipolar)" if tl.get('bipolar') else ""
                tl_parts.append(
                    f"{tl['timeframe']} {tl['slope']} {tl['type']} G{tl['grade']} "
                    f"at ${tl['price']:,.0f}{bp}"
                )
            lines.append(f"TRENDLINES: {' | '.join(tl_parts)}")

        # Top pressure point
        if ctx.pressure_points:
            pp = ctx.pressure_points[0]
            lines.append(
                f"PRESSURE POINT: ${pp['price']:,.0f} confluence {pp['confluence']}/10 "
                f"({pp['timeframe']})"
            )

        # Top support levels (from auto-support)
        if ctx.top_supports:
            parts = [f"${l['price']:,.0f} [{l['score']}]" for l in ctx.top_supports[:3]]
            lines.append(f"SUPPORTS: {' | '.join(parts)}")

        # Top resistance levels
        if ctx.top_resistances:
            parts = [f"${l['price']:,.0f} [{l['score']}]" for l in ctx.top_resistances[:3]]
            lines.append(f"RESISTANCES: {' | '.join(parts)}")

        # MTF bias
        lines.append(f"MTF BIAS: {ctx.mtf_bias.title()} ({ctx.mtf_alignment*100:.0f}% alignment)")

        return '\n'.join(lines)

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached structure data."""
        if symbol:
            api_symbol = symbol.upper()
            if not api_symbol.endswith("USDT"):
                api_symbol = f"{api_symbol}USDT"
            keys_to_remove = [k for k in self._cache if k.startswith(api_symbol)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
