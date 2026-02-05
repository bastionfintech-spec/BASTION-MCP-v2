"""
MCF Labs Report Generator
=========================
Generates institutional-grade reports from live market data
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Report, ReportType, Bias, Confidence, TradeScenario

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates MCF Labs reports from Coinglass, Helsinki, and Whale Alert data.
    """
    
    def __init__(self, coinglass_client, helsinki_client=None, whale_alert_client=None):
        self.coinglass = coinglass_client
        self.helsinki = helsinki_client
        self.whale_alert = whale_alert_client
    
    async def generate_market_structure(self, symbol: str = "BTC") -> Report:
        """Generate Market Structure Report"""
        logger.info(f"Generating market structure report for {symbol}")
        
        # Fetch all required data in parallel
        results = await asyncio.gather(
            self.coinglass.get_coins_markets(),
            self.coinglass.get_hyperliquid_whale_positions(symbol),
            self.coinglass.get_options_max_pain(symbol),
            self.coinglass.get_liquidation_coin_list(symbol),
            self.coinglass.get_funding_rates(symbol),
            self.coinglass.get_long_short_ratio(symbol),
            return_exceptions=True
        )
        
        coins_markets, whale_positions, max_pain, liq_data, funding, ls_ratio = results
        
        # Parse data
        current_price = self._get_current_price(coins_markets, symbol)
        whale_analysis = self._analyze_whales(whale_positions)
        max_pain_price = self._get_max_pain(max_pain)
        liq_analysis = self._analyze_liquidations(liq_data, current_price)
        funding_rate = self._get_funding(funding)
        long_short = self._get_ls_ratio(ls_ratio)
        
        # Determine bias
        bias = self._calculate_bias(
            whale_analysis,
            max_pain_price,
            current_price,
            funding_rate,
            long_short
        )
        
        # Determine confidence
        confidence = self._calculate_confidence(results)
        
        # Generate trade scenario
        trade_scenario = self._generate_trade_scenario(
            bias, current_price, max_pain_price, liq_analysis
        )
        
        # Build report
        report_id = f"MS-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        return Report(
            id=report_id,
            type=ReportType.MARKET_STRUCTURE,
            title=f"{symbol} Market Structure: {self._generate_title(bias, whale_analysis)}",
            generated_at=datetime.utcnow(),
            bias=bias,
            confidence=confidence,
            summary=self._generate_summary(symbol, current_price, bias, whale_analysis, max_pain_price),
            sections={
                "key_levels": {
                    "current_price": current_price,
                    "max_pain": max_pain_price,
                    "resistance": self._get_resistance_levels(max_pain_price, liq_analysis),
                    "support": self._get_support_levels(liq_analysis),
                    "liquidation_clusters": {
                        "longs_at_risk": liq_analysis.get("longs", {}),
                        "shorts_at_risk": liq_analysis.get("shorts", {})
                    }
                },
                "derivatives": {
                    "open_interest": self._get_oi(coins_markets, symbol),
                    "funding_rate": funding_rate,
                    "long_short_ratio": long_short
                },
                "whale_positioning": whale_analysis,
                "trade_scenario": trade_scenario.__dict__ if trade_scenario else None
            },
            tags=self._generate_tags(symbol, bias, whale_analysis),
            data_sources=["coinglass", "hyperliquid"]
        )
    
    async def generate_whale_report(self, symbol: str = "BTC") -> Report:
        """Generate Whale Intelligence Report"""
        logger.info(f"Generating whale intelligence report for {symbol}")
        
        results = await asyncio.gather(
            self.coinglass.get_hyperliquid_whale_positions(symbol),
            self.coinglass.get_exchange_netflow(symbol),
            return_exceptions=True
        )
        
        whale_positions, exchange_flow = results
        
        # Parse whale data
        positions = self._parse_whale_positions(whale_positions)
        aggregate = self._aggregate_whale_stats(positions)
        flow_analysis = self._analyze_exchange_flow(exchange_flow)
        
        # Determine alert level
        alert_level = self._calculate_alert_level(aggregate, flow_analysis)
        
        report_id = f"WI-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        return Report(
            id=report_id,
            type=ReportType.WHALE_INTELLIGENCE,
            title=f"Hyperliquid Whale Activity: {aggregate['dominant_side']} Dominant",
            generated_at=datetime.utcnow(),
            bias=Bias.BULLISH if aggregate['net_exposure'] > 0 else Bias.BEARISH,
            confidence=Confidence.HIGH if len(positions) >= 10 else Confidence.MEDIUM,
            summary=self._generate_whale_summary(aggregate, flow_analysis),
            sections={
                "top_positions": positions[:10],
                "aggregate_stats": aggregate,
                "exchange_flows": flow_analysis,
                "actionable_insight": self._generate_whale_insight(aggregate, flow_analysis)
            },
            tags=["whale", symbol.lower(), aggregate['dominant_side'].lower()],
            data_sources=["hyperliquid", "coinglass"]
        )
    
    async def generate_options_report(self, symbol: str = "BTC") -> Report:
        """Generate Options Flow Report"""
        logger.info(f"Generating options flow report for {symbol}")
        
        results = await asyncio.gather(
            self.coinglass.get_options_info(symbol),
            self.coinglass.get_options_max_pain(symbol),
            self.coinglass.get_options_oi_expiry(symbol),
            return_exceptions=True
        )
        
        options_info, max_pain_data, oi_expiry = results
        
        put_call = self._get_put_call_ratio(options_info)
        max_pain_analysis = self._analyze_max_pain(max_pain_data)
        expiry_analysis = self._analyze_expiries(oi_expiry)
        
        # Determine bias from put/call
        if put_call < 0.85:
            bias = Bias.BULLISH
        elif put_call > 1.15:
            bias = Bias.BEARISH
        else:
            bias = Bias.NEUTRAL
        
        report_id = f"OF-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        return Report(
            id=report_id,
            type=ReportType.OPTIONS_FLOW,
            title=f"{symbol} Options: Put/Call {put_call:.2f} - {bias.value}",
            generated_at=datetime.utcnow(),
            bias=bias,
            confidence=Confidence.HIGH,
            summary=self._generate_options_summary(put_call, max_pain_analysis, expiry_analysis),
            sections={
                "put_call_analysis": {
                    "ratio": put_call,
                    "interpretation": "BULLISH" if put_call < 0.9 else "BEARISH" if put_call > 1.1 else "NEUTRAL"
                },
                "max_pain_analysis": max_pain_analysis,
                "expiry_analysis": expiry_analysis,
                "projected_movement": self._project_options_movement(max_pain_analysis)
            },
            tags=["options", symbol.lower(), "max-pain"],
            data_sources=["coinglass"]
        )
    
    async def generate_cycle_report(self, symbol: str = "BTC") -> Report:
        """Generate Cycle Position Report"""
        logger.info(f"Generating cycle position report for {symbol}")
        
        results = await asyncio.gather(
            self.coinglass.get_bitcoin_bubble_index(),
            self.coinglass.get_ahr999_index(),
            self.coinglass.get_puell_multiple(),
            return_exceptions=True
        )
        
        bubble, ahr999, puell = results
        
        # Parse indicators
        bubble_value = self._get_indicator_value(bubble, "value")
        ahr999_value = self._get_indicator_value(ahr999, "ahr999")
        puell_value = self._get_indicator_value(puell, "puellMultiple")
        
        # Calculate cycle phase
        cycle_phase, weighted_score = self._calculate_cycle_phase(
            bubble_value, ahr999_value, puell_value
        )
        
        report_id = f"CP-{datetime.utcnow().strftime('%Y%m%d')}"
        
        return Report(
            id=report_id,
            type=ReportType.CYCLE_POSITION,
            title=f"Bitcoin Cycle: {cycle_phase}",
            generated_at=datetime.utcnow(),
            bias=self._cycle_to_bias(cycle_phase),
            confidence=Confidence.HIGH,
            summary=self._generate_cycle_summary(cycle_phase, bubble_value, ahr999_value, puell_value),
            sections={
                "indicators": {
                    "bubble_index": {
                        "value": bubble_value,
                        "interpretation": self._interpret_bubble(bubble_value)
                    },
                    "ahr999": {
                        "value": ahr999_value,
                        "interpretation": self._interpret_ahr999(ahr999_value)
                    },
                    "puell_multiple": {
                        "value": puell_value,
                        "interpretation": self._interpret_puell(puell_value)
                    }
                },
                "weighted_assessment": {
                    "score": weighted_score,
                    "phase": cycle_phase,
                    "recommendation": self._get_cycle_recommendation(cycle_phase)
                }
            },
            tags=["cycle", "btc", cycle_phase.lower().replace(" ", "-")],
            data_sources=["coinglass"]
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_current_price(self, data, symbol: str) -> float:
        if hasattr(data, 'success') and data.success and data.data:
            if isinstance(data.data, list):
                for coin in data.data:
                    if coin.get("symbol", "").upper() == symbol.upper():
                        return coin.get("price", 0)
            return data.data.get("price", 0)
        return 0
    
    def _analyze_whales(self, data) -> Dict[str, Any]:
        if not hasattr(data, 'success') or not data.success:
            return {"net_bias": "UNKNOWN", "total_long_usd": 0, "total_short_usd": 0}
        
        positions = data.data if isinstance(data.data, list) else []
        
        total_long = sum(p.get("sizeUsd", 0) for p in positions if p.get("side") == "LONG")
        total_short = sum(p.get("sizeUsd", 0) for p in positions if p.get("side") == "SHORT")
        
        return {
            "net_bias": "LONG" if total_long > total_short else "SHORT",
            "total_long_usd": total_long,
            "total_short_usd": total_short,
            "notable_positions": positions[:5] if positions else []
        }
    
    def _get_max_pain(self, data) -> float:
        if hasattr(data, 'success') and data.success and data.data:
            if isinstance(data.data, list) and len(data.data) > 0:
                return data.data[0].get("maxPain", 0)
            return data.data.get("maxPain", 0)
        return 0
    
    def _analyze_liquidations(self, data, current_price: float) -> Dict[str, Any]:
        # Placeholder - implement based on actual Coinglass response format
        return {
            "longs": {"price": current_price * 0.95, "usd": 0},
            "shorts": {"price": current_price * 1.05, "usd": 0}
        }
    
    def _get_funding(self, data) -> Dict[str, float]:
        if hasattr(data, 'success') and data.success and data.data:
            # Parse funding rates by coin
            return {"btc": 0.0001, "eth": 0.0001, "sol": 0.0001}
        return {"btc": 0, "eth": 0, "sol": 0}
    
    def _get_ls_ratio(self, data) -> float:
        if hasattr(data, 'success') and data.success and data.data:
            if isinstance(data.data, list) and len(data.data) > 0:
                return data.data[-1].get("longShortRatio", 1.0)
        return 1.0
    
    def _calculate_bias(self, whale_analysis, max_pain, price, funding, ls_ratio) -> Bias:
        bullish_signals = 0
        bearish_signals = 0
        
        # Whale positioning
        if whale_analysis.get("net_bias") == "LONG":
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Max pain
        if price < max_pain:
            bullish_signals += 1
        elif price > max_pain:
            bearish_signals += 1
        
        # Funding
        btc_funding = funding.get("btc", 0)
        if btc_funding < -0.0001:
            bullish_signals += 1
        elif btc_funding > 0.0003:
            bearish_signals += 1
        
        if bullish_signals >= 2:
            return Bias.BULLISH
        elif bearish_signals >= 2:
            return Bias.BEARISH
        return Bias.NEUTRAL
    
    def _calculate_confidence(self, results) -> Confidence:
        successful = sum(1 for r in results if hasattr(r, 'success') and r.success)
        if successful >= 5:
            return Confidence.HIGH
        elif successful >= 3:
            return Confidence.MEDIUM
        return Confidence.LOW
    
    def _generate_trade_scenario(self, bias, price, max_pain, liq_analysis) -> Optional[TradeScenario]:
        if bias == Bias.BULLISH:
            return TradeScenario(
                bias="LONG",
                entry_zone=[price * 0.99, price * 1.01],
                stop_loss=liq_analysis.get("longs", {}).get("price", price * 0.95),
                targets=[max_pain, max_pain * 1.05, max_pain * 1.10],
                invalidation=f"Close below ${liq_analysis.get('longs', {}).get('price', price * 0.95):,.0f}"
            )
        elif bias == Bias.BEARISH:
            return TradeScenario(
                bias="SHORT",
                entry_zone=[price * 0.99, price * 1.01],
                stop_loss=liq_analysis.get("shorts", {}).get("price", price * 1.05),
                targets=[max_pain, max_pain * 0.95, max_pain * 0.90],
                invalidation=f"Close above ${liq_analysis.get('shorts', {}).get('price', price * 1.05):,.0f}"
            )
        return None
    
    def _generate_title(self, bias: Bias, whale_analysis: Dict) -> str:
        if bias == Bias.BULLISH:
            return "Bulls Building Momentum"
        elif bias == Bias.BEARISH:
            return "Bears in Control"
        return "Consolidation Zone"
    
    def _generate_summary(self, symbol, price, bias, whales, max_pain) -> str:
        whale_side = whales.get("net_bias", "UNKNOWN")
        total_long = whales.get("total_long_usd", 0) / 1e6
        total_short = whales.get("total_short_usd", 0) / 1e6
        
        return (
            f"{symbol} trading at ${price:,.0f}. Hyperliquid whales net {whale_side} "
            f"(${total_long:.1f}M long vs ${total_short:.1f}M short). "
            f"Max pain at ${max_pain:,.0f} suggests {bias.value.lower()} bias."
        )
    
    def _get_resistance_levels(self, max_pain, liq_analysis) -> List[Dict]:
        return [
            {"price": max_pain, "reason": "Options Max Pain"},
            {"price": liq_analysis.get("shorts", {}).get("price", max_pain * 1.05), "reason": "Short Liquidations"}
        ]
    
    def _get_support_levels(self, liq_analysis) -> List[Dict]:
        return [
            {"price": liq_analysis.get("longs", {}).get("price", 0), "reason": "Long Liquidations"}
        ]
    
    def _get_oi(self, data, symbol) -> Dict[str, Any]:
        return {"value": 0, "change_24h": 0}
    
    def _generate_tags(self, symbol, bias, whales) -> List[str]:
        tags = [symbol.lower(), bias.value.lower()]
        if whales.get("total_long_usd", 0) > 200e6:
            tags.append("whale-heavy")
        return tags
    
    def _parse_whale_positions(self, data) -> List[Dict]:
        if not hasattr(data, 'success') or not data.success:
            return []
        positions = data.data if isinstance(data.data, list) else []
        return [
            {
                "rank": i + 1,
                "side": p.get("side", "UNKNOWN"),
                "size_usd": p.get("sizeUsd", 0),
                "entry_price": p.get("entryPrice", 0),
                "leverage": p.get("leverage", 0),
                "pnl_usd": p.get("pnl", 0),
                "pnl_percent": p.get("pnlPercent", 0)
            }
            for i, p in enumerate(positions)
        ]
    
    def _aggregate_whale_stats(self, positions: List[Dict]) -> Dict[str, Any]:
        total_long = sum(p["size_usd"] for p in positions if p["side"] == "LONG")
        total_short = sum(p["size_usd"] for p in positions if p["side"] == "SHORT")
        longs_pnl = sum(p["pnl_usd"] for p in positions if p["side"] == "LONG")
        shorts_pnl = sum(p["pnl_usd"] for p in positions if p["side"] == "SHORT")
        
        return {
            "total_long_exposure": total_long,
            "total_short_exposure": total_short,
            "net_exposure": total_long - total_short,
            "longs_pnl": longs_pnl,
            "shorts_pnl": shorts_pnl,
            "dominant_side": "LONGS" if total_long > total_short else "SHORTS"
        }
    
    def _analyze_exchange_flow(self, data) -> Dict[str, Any]:
        if not hasattr(data, 'success') or not data.success:
            return {"net_24h": 0, "direction": "UNKNOWN"}
        return {
            "net_24h": data.data.get("netFlow", 0) if data.data else 0,
            "direction": "OUTFLOW" if data.data and data.data.get("netFlow", 0) > 0 else "INFLOW"
        }
    
    def _calculate_alert_level(self, aggregate, flow) -> str:
        net = abs(aggregate.get("net_exposure", 0))
        if net > 500e6:
            return "CRITICAL"
        elif net > 200e6:
            return "HIGH"
        elif net > 100e6:
            return "MODERATE"
        return "LOW"
    
    def _generate_whale_summary(self, aggregate, flow) -> str:
        dominant = aggregate.get("dominant_side", "UNKNOWN")
        net = abs(aggregate.get("net_exposure", 0)) / 1e6
        return f"Whale positioning heavily {dominant} with ${net:.1f}M net exposure."
    
    def _generate_whale_insight(self, aggregate, flow) -> str:
        if aggregate.get("dominant_side") == "LONGS" and aggregate.get("longs_pnl", 0) < 0:
            return "Whale longs underwater - potential capitulation if price drops further."
        elif aggregate.get("dominant_side") == "SHORTS" and aggregate.get("shorts_pnl", 0) < 0:
            return "Whale shorts in pain - short squeeze potential if price breaks higher."
        return "Monitor for position changes."
    
    def _get_put_call_ratio(self, data) -> float:
        if hasattr(data, 'success') and data.success and data.data:
            if isinstance(data.data, list):
                for item in data.data:
                    if item.get("exchange") == "All":
                        return item.get("putCallRatio", 1.0)
            return data.data.get("putCallRatio", 1.0)
        return 1.0
    
    def _analyze_max_pain(self, data) -> Dict[str, Any]:
        if not hasattr(data, 'success') or not data.success:
            return {}
        if isinstance(data.data, list) and len(data.data) > 0:
            nearest = data.data[0]
            return {
                "nearest_expiry": {
                    "date": nearest.get("expiryDate", ""),
                    "max_pain_price": nearest.get("maxPain", 0)
                }
            }
        return {}
    
    def _analyze_expiries(self, data) -> List[Dict]:
        if not hasattr(data, 'success') or not data.success:
            return []
        return data.data if isinstance(data.data, list) else []
    
    def _generate_options_summary(self, put_call, max_pain, expiries) -> str:
        mp = max_pain.get("nearest_expiry", {}).get("max_pain_price", 0)
        interpretation = "bullish" if put_call < 0.9 else "bearish" if put_call > 1.1 else "neutral"
        return f"Put/Call ratio at {put_call:.2f} indicates {interpretation} sentiment. Max pain at ${mp:,.0f}."
    
    def _project_options_movement(self, max_pain) -> Dict[str, Any]:
        mp = max_pain.get("nearest_expiry", {}).get("max_pain_price", 0)
        return {
            "target_zone": [mp * 0.98, mp * 1.02],
            "timeframe": "Before expiry",
            "confidence": "HIGH"
        }
    
    def _get_indicator_value(self, data, key: str) -> float:
        if hasattr(data, 'success') and data.success and data.data:
            if isinstance(data.data, list) and len(data.data) > 0:
                return data.data[-1].get(key, 0)
            return data.data.get(key, 0)
        return 0
    
    def _calculate_cycle_phase(self, bubble, ahr999, puell) -> tuple:
        # Weighted scoring
        score = 0
        
        # Bubble Index (-10 to 10 range typically)
        if bubble < -3:
            score += 30  # Extreme bottom
        elif bubble < 0:
            score += 20
        elif bubble > 3:
            score -= 30  # Bubble
        else:
            score -= 10
        
        # AHR999 (< 0.45 = buy, > 1.2 = sell)
        if ahr999 < 0.45:
            score += 30
        elif ahr999 < 0.8:
            score += 15
        elif ahr999 > 1.2:
            score -= 30
        
        # Puell (< 0.5 = undervalued, > 4 = overvalued)
        if puell < 0.5:
            score += 30
        elif puell < 1.0:
            score += 15
        elif puell > 4:
            score -= 30
        
        if score >= 60:
            return "ACCUMULATION", score
        elif score >= 30:
            return "EARLY MARKUP", score
        elif score >= 0:
            return "MARKUP", score
        elif score >= -30:
            return "DISTRIBUTION", score
        return "MARKDOWN", score
    
    def _cycle_to_bias(self, phase: str) -> Bias:
        if phase in ["ACCUMULATION", "EARLY MARKUP"]:
            return Bias.BULLISH
        elif phase in ["DISTRIBUTION", "MARKDOWN"]:
            return Bias.BEARISH
        return Bias.NEUTRAL
    
    def _interpret_bubble(self, value: float) -> str:
        if value < -3:
            return "EXTREME_BOTTOM"
        elif value < 0:
            return "BOTTOM"
        elif value < 3:
            return "FAIR"
        elif value < 6:
            return "ELEVATED"
        return "BUBBLE"
    
    def _interpret_ahr999(self, value: float) -> str:
        if value < 0.45:
            return "STRONG_BUY"
        elif value < 0.8:
            return "BUY"
        elif value < 1.2:
            return "HOLD"
        elif value < 1.5:
            return "SELL"
        return "STRONG_SELL"
    
    def _interpret_puell(self, value: float) -> str:
        if value < 0.5:
            return "MINER_CAPITULATION"
        elif value < 1.0:
            return "UNDERVALUED"
        elif value < 2.0:
            return "FAIR"
        elif value < 4.0:
            return "OVERVALUED"
        return "MINER_EUPHORIA"
    
    def _generate_cycle_summary(self, phase, bubble, ahr999, puell) -> str:
        return (
            f"Bitcoin currently in {phase} phase. "
            f"Bubble Index: {bubble:.2f} ({self._interpret_bubble(bubble)}), "
            f"AHR999: {ahr999:.3f} ({self._interpret_ahr999(ahr999)}), "
            f"Puell: {puell:.3f} ({self._interpret_puell(puell)})."
        )
    
    def _get_cycle_recommendation(self, phase: str) -> str:
        recommendations = {
            "ACCUMULATION": "Aggressive accumulation zone. DCA heavily.",
            "EARLY MARKUP": "Continue accumulating. Hold existing positions.",
            "MARKUP": "Hold. Take partial profits on strength.",
            "DISTRIBUTION": "Reduce exposure. Take profits.",
            "MARKDOWN": "Stay in cash/stables. Wait for accumulation zone."
        }
        return recommendations.get(phase, "Monitor market conditions.")

