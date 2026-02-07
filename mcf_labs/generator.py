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
        whale_analysis = self._analyze_whales(whale_positions, symbol)
        max_pain_price = self._get_max_pain(max_pain)
        liq_analysis = self._analyze_liquidations(liq_data, current_price)
        funding_rate = self._get_funding(funding, symbol)
        long_short = self._get_ls_ratio(ls_ratio)
        oi_data = self._get_oi(coins_markets, symbol)
        
        # VALIDATION: Require valid price data
        if current_price <= 0:
            raise ValueError(f"Invalid price data for {symbol}: price={current_price}")
        
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
        report_id = f"MS-{symbol}-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
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
                    "open_interest": oi_data,
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
        
        # Parse whale data - filter by symbol
        positions = self._parse_whale_positions(whale_positions, symbol)
        aggregate = self._aggregate_whale_stats(positions)
        flow_analysis = self._analyze_exchange_flow(exchange_flow)
        
        # VALIDATION: Require minimum valid positions
        valid_positions = [p for p in positions if p.get('size_usd', 0) > 0]
        if len(valid_positions) < 3:
            raise ValueError(
                f"Insufficient whale data for {symbol}: only {len(valid_positions)} valid positions found. "
                f"Need at least 3 positions with size > 0."
            )
        
        # Determine alert level
        alert_level = self._calculate_alert_level(aggregate, flow_analysis)
        
        report_id = f"WI-{symbol}-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        return Report(
            id=report_id,
            type=ReportType.WHALE_INTELLIGENCE,
            title=f"{symbol} Whale Activity: {aggregate['dominant_side']} Dominant",
            generated_at=datetime.utcnow(),
            bias=Bias.BULLISH if aggregate['net_exposure'] > 0 else Bias.BEARISH,
            confidence=Confidence.HIGH if len(valid_positions) >= 10 else Confidence.MEDIUM,
            summary=self._generate_whale_summary(aggregate, flow_analysis, symbol),
            sections={
                "top_positions": valid_positions[:10],
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
    
    def _analyze_whales(self, data, symbol: str = None) -> Dict[str, Any]:
        """
        Analyze Hyperliquid whale positions.
        
        Note: Hyperliquid API returns ALL positions. We filter by symbol.
        Fields: positionSize (positive=LONG, negative=SHORT), positionValueUsd, unrealizedPnL
        """
        if not hasattr(data, 'success') or not data.success:
            return {"net_bias": "UNKNOWN", "total_long_usd": 0, "total_short_usd": 0, "position_count": 0}
        
        positions = data.data if isinstance(data.data, list) else []
        
        # Filter by symbol if provided
        if symbol:
            positions = [p for p in positions if p.get("symbol", "").upper() == symbol.upper()]
        
        if not positions:
            return {"net_bias": "UNKNOWN", "total_long_usd": 0, "total_short_usd": 0, "position_count": 0}
        
        # Hyperliquid uses positionSize: positive = LONG, negative = SHORT
        total_long = sum(
            abs(p.get("positionValueUsd", 0)) 
            for p in positions 
            if p.get("positionSize", 0) > 0
        )
        total_short = sum(
            abs(p.get("positionValueUsd", 0)) 
            for p in positions 
            if p.get("positionSize", 0) < 0
        )
        
        # Get top positions by size
        sorted_positions = sorted(positions, key=lambda x: abs(x.get("positionValueUsd", 0)), reverse=True)
        notable = sorted_positions[:5] if sorted_positions else []
        
        return {
            "net_bias": "LONG" if total_long > total_short else "SHORT",
            "total_long_usd": total_long,
            "total_short_usd": total_short,
            "position_count": len(positions),
            "notable_positions": notable
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
    
    def _get_funding(self, data, symbol: str = "BTC") -> Dict[str, Any]:
        """Parse funding rates from Coinglass response"""
        result = {"rate": 0, "exchange_rates": [], "avg_rate": 0}
        
        if not hasattr(data, 'success') or not data.success or not data.data:
            return result
        
        rates_list = []
        
        # Handle nested structure with usdtOrUsdMarginList/tokenMarginList
        if isinstance(data.data, dict):
            for margin_list in ["usdtOrUsdMarginList", "tokenMarginList"]:
                items = data.data.get(margin_list, [])
                if isinstance(items, list):
                    for item in items:
                        exchange = item.get("exchangeName", "Unknown")
                        rate = item.get("rate", 0) or item.get("fundingRate", 0)
                        if rate:
                            rates_list.append({"exchange": exchange, "rate": rate})
        
        elif isinstance(data.data, list):
            for item in data.data:
                exchange = item.get("exchangeName", item.get("exchange", "Unknown"))
                rate = item.get("rate", 0) or item.get("fundingRate", 0)
                if rate:
                    rates_list.append({"exchange": exchange, "rate": rate})
        
        if rates_list:
            avg_rate = sum(r["rate"] for r in rates_list) / len(rates_list)
            result = {
                "rate": avg_rate,
                "exchange_rates": rates_list[:5],  # Top 5 exchanges
                "avg_rate": avg_rate,
                "btc": avg_rate  # For backward compatibility
            }
        
        return result
    
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
    
    def _get_oi(self, data, symbol: str) -> Dict[str, Any]:
        """Extract Open Interest from coins_markets response"""
        result = {"value": 0, "change_24h": 0, "change_percent_24h": 0}
        
        if not hasattr(data, 'success') or not data.success or not data.data:
            return result
        
        coins = data.data if isinstance(data.data, list) else []
        
        for coin in coins:
            if coin.get("symbol", "").upper() == symbol.upper():
                oi = coin.get("openInterest", 0)
                oi_change = coin.get("oiChg24h", 0) or coin.get("openInterestChange24h", 0)
                oi_change_pct = coin.get("oiChgPercent24h", 0) or coin.get("oiChg24hPercent", 0)
                
                return {
                    "value": oi,
                    "change_24h": oi_change,
                    "change_percent_24h": oi_change_pct
                }
        
        return result
    
    def _generate_tags(self, symbol, bias, whales) -> List[str]:
        tags = [symbol.lower(), bias.value.lower()]
        if whales.get("total_long_usd", 0) > 200e6:
            tags.append("whale-heavy")
        return tags
    
    def _parse_whale_positions(self, data, symbol: str = None) -> List[Dict]:
        """
        Parse Hyperliquid whale positions with correct field mapping.
        
        Hyperliquid fields:
        - positionSize: positive = LONG, negative = SHORT
        - positionValueUsd: position value in USD
        - unrealizedPnL: PnL in USD
        - entryPrice, markPrice, leverage
        """
        if not hasattr(data, 'success') or not data.success:
            return []
        
        positions = data.data if isinstance(data.data, list) else []
        
        # Filter by symbol if provided
        if symbol:
            positions = [p for p in positions if p.get("symbol", "").upper() == symbol.upper()]
        
        if not positions:
            return []
        
        # Sort by position size (largest first)
        sorted_positions = sorted(positions, key=lambda x: abs(x.get("positionValueUsd", 0)), reverse=True)
        
        result = []
        for i, p in enumerate(sorted_positions):
            pos_size = p.get("positionSize", 0)
            pos_value = abs(p.get("positionValueUsd", 0))
            entry_price = p.get("entryPrice", 0)
            pnl = p.get("unrealizedPnL", 0)
            
            # Skip positions with invalid data
            if pos_value == 0 or entry_price == 0:
                continue
            
            # Calculate PnL percent
            margin = p.get("marginBalance", pos_value)
            pnl_percent = (pnl / margin * 100) if margin > 0 else 0
            
            result.append({
                "rank": i + 1,
                "side": "LONG" if pos_size > 0 else "SHORT",
                "size_usd": pos_value,
                "entry_price": entry_price,
                "mark_price": p.get("markPrice", 0),
                "leverage": p.get("leverage", 1),
                "pnl_usd": pnl,
                "pnl_percent": round(pnl_percent, 2),
                "liquidation_price": p.get("liqPrice", 0),
                "symbol": p.get("symbol", "UNKNOWN")
            })
        
        return result
    
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
    
    def _generate_whale_summary(self, aggregate, flow, symbol: str = "BTC") -> str:
        dominant = aggregate.get("dominant_side", "UNKNOWN")
        net = abs(aggregate.get("net_exposure", 0)) / 1e6
        total_long = aggregate.get("total_long_exposure", 0) / 1e6
        total_short = aggregate.get("total_short_exposure", 0) / 1e6
        
        return (
            f"{symbol} Hyperliquid whales {dominant} dominant with ${net:.1f}M net exposure. "
            f"Longs: ${total_long:.1f}M | Shorts: ${total_short:.1f}M. "
            f"Flow: {flow.get('direction', 'UNKNOWN')}."
        )
    
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



