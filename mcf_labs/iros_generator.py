"""
MCF Labs IROS-Powered Report Generator
======================================
Generates intelligent reports using IROS (BastionAI) for analysis.
This extends the base generator with LLM-powered insights.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Report, ReportType, Bias, Confidence, TradeScenario
from .generator import ReportGenerator

logger = logging.getLogger(__name__)

# Multi-coin support
SUPPORTED_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "AVAX", "LINK", "ARB", "OP"]


class IROSReportGenerator(ReportGenerator):
    """
    IROS-powered report generator.
    
    Extends the base ReportGenerator with intelligent LLM analysis.
    Falls back to rule-based analysis if IROS unavailable.
    
    Usage:
        generator = IROSReportGenerator(coinglass, helsinki, bastion_ai)
        report = await generator.generate_market_structure("BTC")
    """
    
    def __init__(
        self,
        coinglass_client,
        helsinki_client=None,
        whale_alert_client=None,
        bastion_ai=None
    ):
        # Initialize parent with data clients
        super().__init__(coinglass_client, helsinki_client, whale_alert_client)
        
        # IROS brain for intelligent analysis
        self.iros = bastion_ai
        self._iros_available = bastion_ai is not None
        
        if self._iros_available:
            logger.info("[MCF] IROS-powered report generator initialized")
        else:
            logger.warning("[MCF] IROS not available, using rule-based analysis")
    
    # =========================================================================
    # MULTI-COIN BATCH REPORTS
    # =========================================================================
    
    async def generate_all_coins_report(self, report_type: str) -> List[Report]:
        """Generate reports for all supported coins in parallel"""
        tasks = []
        
        for symbol in SUPPORTED_COINS:
            if report_type == "market_structure":
                tasks.append(self.generate_market_structure(symbol))
            elif report_type == "whale_intelligence":
                tasks.append(self.generate_whale_report(symbol))
            elif report_type == "options_flow":
                # Options only for BTC and ETH
                if symbol in ["BTC", "ETH"]:
                    tasks.append(self.generate_options_report(symbol))
            elif report_type == "cycle_position":
                # Cycle indicators only for BTC
                if symbol == "BTC":
                    tasks.append(self.generate_cycle_report(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        reports = []
        for r in results:
            if isinstance(r, Report):
                reports.append(r)
            elif isinstance(r, Exception):
                logger.error(f"Report generation failed: {r}")
        
        return reports
    
    # =========================================================================
    # IROS-ENHANCED MARKET STRUCTURE REPORT
    # =========================================================================
    
    async def generate_market_structure(self, symbol: str = "BTC") -> Report:
        """
        Generate Market Structure Report with IROS intelligence.
        
        If IROS is available, uses LLM for deep analysis.
        Otherwise falls back to rule-based analysis.
        """
        logger.info(f"[MCF] Generating IROS market structure for {symbol}")
        
        # 1. Fetch all data in parallel (same as parent)
        results = await asyncio.gather(
            self.coinglass.get_coins_markets(),
            self.coinglass.get_hyperliquid_whale_positions(symbol),
            self.coinglass.get_options_max_pain(symbol),
            self.coinglass.get_liquidation_coin_list(symbol),
            self.coinglass.get_funding_rates(symbol),
            self.coinglass.get_long_short_ratio(symbol),
            self.coinglass.get_taker_buy_sell(symbol),
            return_exceptions=True
        )
        
        coins_markets, whale_positions, max_pain, liq_data, funding, ls_ratio, taker = results
        
        # 2. Parse data
        current_price = self._get_current_price(coins_markets, symbol)
        whale_analysis = self._analyze_whales(whale_positions)
        max_pain_price = self._get_max_pain(max_pain)
        liq_analysis = self._analyze_liquidations(liq_data, current_price)
        funding_rate = self._get_funding(funding)
        long_short = self._get_ls_ratio(ls_ratio)
        taker_flow = self._parse_taker_flow(taker)
        
        # 3. Build IROS prompt with structured data
        iros_analysis = None
        if self._iros_available:
            try:
                iros_prompt = self._build_market_structure_prompt(
                    symbol, current_price, whale_analysis, max_pain_price,
                    funding_rate, long_short, taker_flow, liq_analysis
                )
                
                iros_result = await self.iros.process_query(
                    query=iros_prompt,
                    comprehensive=False,
                    user_context={"symbol": symbol}
                )
                
                if iros_result.success:
                    iros_analysis = iros_result.response
                    logger.info(f"[MCF] IROS analysis completed for {symbol}")
                else:
                    logger.warning(f"[MCF] IROS analysis failed: {iros_result.error}")
                    
            except Exception as e:
                logger.error(f"[MCF] IROS error: {e}")
        
        # 4. Determine bias (use IROS insight if available, else rule-based)
        bias = self._calculate_bias(
            whale_analysis, max_pain_price, current_price, funding_rate, long_short
        )
        
        # 5. Calculate confidence
        confidence = self._calculate_confidence(results)
        
        # 6. Generate trade scenario
        trade_scenario = self._generate_trade_scenario(
            bias, current_price, max_pain_price, liq_analysis
        )
        
        # 7. Build report
        report_id = f"MS-{symbol}-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        # Generate summary (IROS-enhanced or rule-based)
        if iros_analysis:
            summary = iros_analysis[:500] if len(iros_analysis) > 500 else iros_analysis
        else:
            summary = self._generate_summary(symbol, current_price, bias, whale_analysis, max_pain_price)
        
        return Report(
            id=report_id,
            type=ReportType.MARKET_STRUCTURE,
            title=f"{symbol} Market Structure: {self._generate_title(bias, whale_analysis)}",
            generated_at=datetime.utcnow(),
            bias=bias,
            confidence=confidence,
            summary=summary,
            sections={
                "iros_analysis": iros_analysis,
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
                    "long_short_ratio": long_short,
                    "taker_flow": taker_flow
                },
                "whale_positioning": whale_analysis,
                "trade_scenario": trade_scenario.__dict__ if trade_scenario else None
            },
            tags=self._generate_tags(symbol, bias, whale_analysis),
            data_sources=["coinglass", "hyperliquid", "iros"] if iros_analysis else ["coinglass", "hyperliquid"]
        )
    
    def _build_market_structure_prompt(
        self,
        symbol: str,
        price: float,
        whales: Dict,
        max_pain: float,
        funding: Dict,
        ls_ratio: float,
        taker: Dict,
        liq: Dict
    ) -> str:
        """Build IROS prompt for market structure analysis"""
        
        # Format whale data
        whale_text = f"""WHALE POSITIONS (Hyperliquid):
- Net Bias: {whales.get('net_bias', 'UNKNOWN')}
- Total Long Exposure: ${whales.get('total_long_usd', 0)/1e6:.1f}M
- Total Short Exposure: ${whales.get('total_short_usd', 0)/1e6:.1f}M"""
        
        # Format funding
        btc_funding = funding.get("btc", 0)
        funding_text = f"FUNDING: BTC {btc_funding*100:.4f}%"
        
        # Format taker flow
        taker_text = ""
        if taker:
            taker_text = f"TAKER FLOW: Buy ${taker.get('buy', 0)/1e6:.1f}M / Sell ${taker.get('sell', 0)/1e6:.1f}M"
        
        return f"""Generate a MARKET STRUCTURE ANALYSIS for {symbol}.

LIVE DATA:
- Current Price: ${price:,.0f}
- Max Pain (Options): ${max_pain:,.0f}
- Long/Short Ratio: {ls_ratio:.2f}

{whale_text}

{funding_text}
{taker_text}

LIQUIDATION ZONES:
- Longs at risk below: ${liq.get('longs', {}).get('price', price*0.95):,.0f}
- Shorts at risk above: ${liq.get('shorts', {}).get('price', price*1.05):,.0f}

INSTRUCTIONS:
1. Analyze the confluence of signals
2. Identify if whales are positioned for a specific move
3. Determine the "pain trade" direction
4. Generate a trade scenario with entry/stop/targets
5. Assign BULLISH/BEARISH/NEUTRAL bias with confidence

Be specific with price levels and percentages."""
    
    def _parse_taker_flow(self, data) -> Dict[str, float]:
        """Parse taker buy/sell volume"""
        if not hasattr(data, 'success') or not data.success:
            return {"buy": 0, "sell": 0}
        
        if isinstance(data.data, list):
            total_buy = sum(t.get("buyVolUsd", 0) for t in data.data)
            total_sell = sum(t.get("sellVolUsd", 0) for t in data.data)
            return {"buy": total_buy, "sell": total_sell}
        
        return {"buy": 0, "sell": 0}
    
    # =========================================================================
    # IROS-ENHANCED WHALE REPORT
    # =========================================================================
    
    async def generate_whale_report(self, symbol: str = "BTC") -> Report:
        """Generate Whale Intelligence Report with IROS analysis"""
        logger.info(f"[MCF] Generating IROS whale report for {symbol}")
        
        results = await asyncio.gather(
            self.coinglass.get_hyperliquid_whale_positions(symbol),
            self.coinglass.get_exchange_netflow(symbol),
            return_exceptions=True
        )
        
        whale_positions, exchange_flow = results
        
        positions = self._parse_whale_positions(whale_positions)
        aggregate = self._aggregate_whale_stats(positions)
        flow_analysis = self._analyze_exchange_flow(exchange_flow)
        
        # IROS analysis
        iros_analysis = None
        if self._iros_available and positions:
            try:
                iros_prompt = self._build_whale_prompt(symbol, positions, aggregate, flow_analysis)
                iros_result = await self.iros.process_query(
                    query=iros_prompt,
                    comprehensive=False
                )
                if iros_result.success:
                    iros_analysis = iros_result.response
            except Exception as e:
                logger.error(f"[MCF] IROS whale analysis error: {e}")
        
        alert_level = self._calculate_alert_level(aggregate, flow_analysis)
        report_id = f"WI-{symbol}-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        summary = iros_analysis[:500] if iros_analysis else self._generate_whale_summary(aggregate, flow_analysis)
        
        return Report(
            id=report_id,
            type=ReportType.WHALE_INTELLIGENCE,
            title=f"{symbol} Whale Activity: {aggregate['dominant_side']} Dominant",
            generated_at=datetime.utcnow(),
            bias=Bias.BULLISH if aggregate['net_exposure'] > 0 else Bias.BEARISH,
            confidence=Confidence.HIGH if len(positions) >= 10 else Confidence.MEDIUM,
            summary=summary,
            sections={
                "iros_analysis": iros_analysis,
                "top_positions": positions[:10],
                "aggregate_stats": aggregate,
                "exchange_flows": flow_analysis,
                "actionable_insight": self._generate_whale_insight(aggregate, flow_analysis)
            },
            tags=["whale", symbol.lower(), aggregate['dominant_side'].lower()],
            data_sources=["hyperliquid", "coinglass", "iros"] if iros_analysis else ["hyperliquid", "coinglass"]
        )
    
    def _build_whale_prompt(self, symbol: str, positions: List, aggregate: Dict, flow: Dict) -> str:
        """Build IROS prompt for whale analysis"""
        
        # Format top positions
        pos_text = "\n".join([
            f"{p['rank']}. {p['side']} ${p['size_usd']/1e6:.1f}M @ ${p['entry_price']:,.0f} ({p['leverage']}x) PnL: ${p['pnl_usd']/1e6:.2f}M"
            for p in positions[:5]
        ])
        
        return f"""Analyze WHALE POSITIONING for {symbol}.

TOP 5 POSITIONS:
{pos_text}

AGGREGATE:
- Long Exposure: ${aggregate['total_long_exposure']/1e6:.1f}M
- Short Exposure: ${aggregate['total_short_exposure']/1e6:.1f}M
- Net: ${aggregate['net_exposure']/1e6:.1f}M
- Longs PnL: ${aggregate['longs_pnl']/1e6:.2f}M
- Shorts PnL: ${aggregate['shorts_pnl']/1e6:.2f}M
- Dominant: {aggregate['dominant_side']}

EXCHANGE FLOW:
- Direction: {flow.get('direction', 'UNKNOWN')}
- Net 24h: ${flow.get('net_24h', 0)/1e6:.1f}M

ANALYZE:
1. Who is winning - longs or shorts?
2. Are underwater positions at risk of liquidation?
3. What does the exchange flow tell us about accumulation/distribution?
4. What's the likely next move based on whale positioning?

Provide actionable trading insight."""
    
    # =========================================================================
    # IROS-ENHANCED CYCLE REPORT
    # =========================================================================
    
    async def generate_cycle_report(self, symbol: str = "BTC") -> Report:
        """Generate Cycle Position Report with IROS interpretation"""
        logger.info("[MCF] Generating IROS cycle report")
        
        results = await asyncio.gather(
            self.coinglass.get_bitcoin_bubble_index(),
            self.coinglass.get_ahr999_index(),
            self.coinglass.get_puell_multiple(),
            return_exceptions=True
        )
        
        bubble, ahr999, puell = results
        
        bubble_value = self._get_indicator_value(bubble, "value")
        ahr999_value = self._get_indicator_value(ahr999, "ahr999")
        puell_value = self._get_indicator_value(puell, "puellMultiple")
        
        cycle_phase, weighted_score = self._calculate_cycle_phase(
            bubble_value, ahr999_value, puell_value
        )
        
        # IROS analysis
        iros_analysis = None
        if self._iros_available:
            try:
                iros_prompt = f"""Analyze BITCOIN CYCLE POSITION.

CYCLE INDICATORS:
1. Bubble Index: {bubble_value:.2f}
   - Below 0 = Undervalued, Above 3 = Overvalued, Above 6 = Bubble

2. AHR999: {ahr999_value:.3f}
   - Below 0.45 = Strong Buy, 0.45-0.8 = Buy, 0.8-1.2 = Hold, Above 1.2 = Sell

3. Puell Multiple: {puell_value:.3f}
   - Below 0.5 = Miner capitulation, 0.5-1.0 = Undervalued, 1.0-2.0 = Fair, Above 4 = Top

WEIGHTED SCORE: {weighted_score}
CALCULATED PHASE: {cycle_phase}

INSTRUCTIONS:
1. Interpret all 3 indicators together
2. Determine cycle phase (Accumulation, Markup, Distribution, Markdown)
3. Provide DCA strategy recommendation
4. Estimate probability of being at a cycle extreme"""
                
                iros_result = await self.iros.process_query(iros_prompt, comprehensive=False)
                if iros_result.success:
                    iros_analysis = iros_result.response
            except Exception as e:
                logger.error(f"[MCF] IROS cycle analysis error: {e}")
        
        report_id = f"CP-{datetime.utcnow().strftime('%Y%m%d')}"
        
        summary = iros_analysis[:500] if iros_analysis else self._generate_cycle_summary(
            cycle_phase, bubble_value, ahr999_value, puell_value
        )
        
        return Report(
            id=report_id,
            type=ReportType.CYCLE_POSITION,
            title=f"Bitcoin Cycle: {cycle_phase}",
            generated_at=datetime.utcnow(),
            bias=self._cycle_to_bias(cycle_phase),
            confidence=Confidence.HIGH,
            summary=summary,
            sections={
                "iros_analysis": iros_analysis,
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
            data_sources=["coinglass", "iros"] if iros_analysis else ["coinglass"]
        )


# Factory function for creating IROS generator
def create_iros_generator(
    coinglass_client,
    helsinki_client=None,
    whale_alert_client=None,
    model_url: str = None,
    model_api_key: str = None
) -> IROSReportGenerator:
    """
    Create an IROS-powered report generator.
    
    Args:
        coinglass_client: Initialized CoinglassClient
        helsinki_client: Optional HelsinkiClient
        whale_alert_client: Optional WhaleAlertClient
        model_url: Vast.ai model URL
        model_api_key: Optional API key
    
    Returns:
        IROSReportGenerator instance
    """
    bastion_ai = None
    
    # Try to initialize BastionAI
    if model_url:
        try:
            from iros_integration.services.bastion_ai import BastionAI
            bastion_ai = BastionAI(
                helsinki_url=helsinki_client.base_url if helsinki_client else None,
                model_url=model_url,
                model_api_key=model_api_key
            )
            logger.info(f"[MCF] BastionAI initialized with model: {model_url}")
        except Exception as e:
            logger.warning(f"[MCF] Could not initialize BastionAI: {e}")
    
    return IROSReportGenerator(
        coinglass_client=coinglass_client,
        helsinki_client=helsinki_client,
        whale_alert_client=whale_alert_client,
        bastion_ai=bastion_ai
    )

