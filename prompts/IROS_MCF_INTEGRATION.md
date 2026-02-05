# IROS → MCF Labs Integration Guide

> **✅ STATUS: IMPLEMENTED** (Feb 2026)
> 
> This integration is now complete. See:
> - `mcf_labs/iros_generator.py` - IROS-powered report generator
> - `mcf_labs/scheduler.py` - Updated with IROS support
> - `api/terminal_api.py` - MCF API endpoints added

## Overview

This guide shows how to wire IROS (the intelligent analysis brain) into the MCF Labs report generation system for the BASTION terminal.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MCF REPORT PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   SCHEDULER          DATA FETCH           IROS            OUTPUT    │
│   ─────────          ──────────           ────            ──────    │
│                                                                     │
│   ┌─────────┐       ┌──────────┐       ┌─────────┐    ┌──────────┐ │
│   │ Cron    │──────▶│ Coinglass│──────▶│ BastionAI│───▶│ Report   │ │
│   │ 2h/4h/  │       │ 13 APIs  │       │ (32B LLM)│    │ (JSON)   │ │
│   │ 6h/24h  │       └──────────┘       └─────────┘    └──────────┘ │
│   └─────────┘              │                  │              │      │
│        │                   │                  │              │      │
│        │            ┌──────────┐              │              ▼      │
│        │            │ Helsinki │              │       ┌──────────┐  │
│        └───────────▶│ 33 APIs  │──────────────┘       │ Research │  │
│                     └──────────┘                      │ Terminal │  │
│                                                       └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
BASTION/
├── iros_integration/
│   ├── services/
│   │   ├── bastion_ai.py      # IROS brain - BastionAI class
│   │   ├── coinglass.py       # 13 working endpoints
│   │   ├── helsinki.py        # 33 quant endpoints
│   │   ├── whale_alert.py     # Transaction monitoring
│   │   └── query_processor.py # Context extraction
│   └── config/
│       └── settings.py        # API keys & URLs
│
├── mcf_labs/
│   ├── generator.py           # Report generation (needs IROS integration)
│   ├── scheduler.py           # Cron scheduling
│   ├── storage.py             # Report storage & queries
│   └── models.py              # Report data models
│
└── api/
    └── terminal_api.py        # Main API serving everything
```

---

## Step 1: Wire IROS into Report Generator

Update `mcf_labs/generator.py` to use BastionAI for intelligent analysis:

```python
"""
MCF Labs Report Generator - IROS Integrated
============================================
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from iros_integration.services.bastion_ai import BastionAI
from iros_integration.services.coinglass import CoinglassClient
from iros_integration.services.helsinki import HelsinkiClient
from .models import Report, ReportType, Bias, Confidence

logger = logging.getLogger(__name__)

# Supported coins for multi-asset reports
SUPPORTED_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "AVAX", "LINK", "ARB", "OP"]


class IROSReportGenerator:
    """
    Generates MCF Labs reports using IROS for intelligent analysis.
    
    IROS does the hard work:
    - Interprets raw data into actionable insights
    - Identifies conflicting signals
    - Creates trade setups with entry/stop/targets
    - Assigns confidence/bias scores
    """
    
    def __init__(
        self,
        coinglass: CoinglassClient,
        helsinki: HelsinkiClient = None,
        model_url: str = None,
        model_api_key: str = None
    ):
        self.coinglass = coinglass
        self.helsinki = helsinki
        
        # Initialize IROS (BastionAI)
        self.iros = BastionAI(
            helsinki_url=helsinki.base_url if helsinki else None,
            model_url=model_url,
            model_api_key=model_api_key
        )
    
    # =========================================================================
    # MULTI-COIN SUPPORT
    # =========================================================================
    
    async def generate_all_coins_report(self, report_type: str) -> List[Report]:
        """Generate reports for all supported coins"""
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
                    tasks.append(self.generate_cycle_report())
        
        reports = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        return [r for r in reports if isinstance(r, Report)]
    
    # =========================================================================
    # MARKET STRUCTURE REPORT
    # =========================================================================
    
    async def generate_market_structure(self, symbol: str = "BTC") -> Report:
        """
        Generate Market Structure Report using IROS.
        
        Data Sources:
        - Coinglass: OI, Funding, L/S ratio, Liquidations, Whale positions
        - Helsinki: CVD, Volatility, Options IV
        
        IROS Analysis:
        - Interprets confluence of signals
        - Generates trade scenarios
        - Calculates risk/reward
        """
        logger.info(f"[MCF] Generating market structure for {symbol}")
        
        # 1. Fetch all data in parallel
        data = await self._fetch_market_structure_data(symbol)
        
        # 2. Build IROS prompt with structured data
        iros_prompt = self._build_market_structure_prompt(symbol, data)
        
        # 3. Get IROS analysis
        iros_result = await self.iros.process_query(
            query=iros_prompt,
            comprehensive=True,
            user_context={"symbol": symbol}
        )
        
        # 4. Parse IROS response into structured report
        report = self._parse_market_structure_response(
            symbol=symbol,
            data=data,
            iros_response=iros_result.response if iros_result.success else None
        )
        
        return report
    
    async def _fetch_market_structure_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch all data needed for market structure report"""
        results = await asyncio.gather(
            # Coinglass Premium
            self.coinglass.get_coins_markets(),
            self.coinglass.get_hyperliquid_whale_positions(symbol),
            self.coinglass.get_options_max_pain(symbol),
            self.coinglass.get_liquidation_coin_list(symbol),
            self.coinglass.get_funding_rates(symbol),
            self.coinglass.get_long_short_ratio(symbol),
            self.coinglass.get_top_trader_sentiment(symbol),
            self.coinglass.get_taker_buy_sell(symbol),
            # Helsinki (if available)
            self._safe_helsinki_call("fetch_cvd_data", symbol),
            self._safe_helsinki_call("fetch_volatility_data", symbol),
            self._safe_helsinki_call("fetch_liquidation_data", symbol),
            return_exceptions=True
        )
        
        return {
            "coins_markets": self._safe_get(results[0]),
            "whale_positions": self._safe_get(results[1]),
            "max_pain": self._safe_get(results[2]),
            "liquidations": self._safe_get(results[3]),
            "funding": self._safe_get(results[4]),
            "ls_ratio": self._safe_get(results[5]),
            "top_traders": self._safe_get(results[6]),
            "taker_flow": self._safe_get(results[7]),
            "cvd": results[8] if not isinstance(results[8], Exception) else None,
            "volatility": results[9] if not isinstance(results[9], Exception) else None,
            "helsinki_liq": results[10] if not isinstance(results[10], Exception) else None,
        }
    
    async def _safe_helsinki_call(self, method: str, symbol: str):
        """Safely call Helsinki method"""
        if not self.helsinki:
            return None
        try:
            func = getattr(self.helsinki, method)
            return await func(symbol)
        except:
            return None
    
    def _safe_get(self, result):
        """Safely extract data from CoinglassResponse"""
        if hasattr(result, 'success') and result.success:
            return result.data
        return None
    
    def _build_market_structure_prompt(self, symbol: str, data: Dict) -> str:
        """Build the prompt for IROS to analyze market structure"""
        
        # Format whale positions
        whale_text = "No whale data"
        if data.get("whale_positions"):
            whales = data["whale_positions"]
            if isinstance(whales, list):
                whale_lines = []
                total_long = sum(w.get("sizeUsd", 0) for w in whales if w.get("side") == "LONG")
                total_short = sum(w.get("sizeUsd", 0) for w in whales if w.get("side") == "SHORT")
                
                whale_lines.append(f"Total Long Exposure: ${total_long/1e6:.1f}M")
                whale_lines.append(f"Total Short Exposure: ${total_short/1e6:.1f}M")
                whale_lines.append(f"Net Bias: {'LONG' if total_long > total_short else 'SHORT'}")
                whale_lines.append("")
                whale_lines.append("Top 5 Positions:")
                
                for i, w in enumerate(whales[:5]):
                    whale_lines.append(
                        f"  {i+1}. {w.get('side')} ${w.get('sizeUsd', 0)/1e6:.1f}M "
                        f"@ ${w.get('entryPrice', 0):,.0f} ({w.get('leverage', 0)}x) "
                        f"PnL: ${w.get('pnl', 0)/1e6:.2f}M"
                    )
                
                whale_text = "\n".join(whale_lines)
        
        # Format max pain
        max_pain_text = "No max pain data"
        if data.get("max_pain"):
            mp = data["max_pain"]
            if isinstance(mp, list) and len(mp) > 0:
                nearest = mp[0]
                max_pain_text = f"Max Pain: ${nearest.get('maxPain', 0):,.0f} (Expiry: {nearest.get('expiryDate', 'N/A')})"
        
        # Format funding
        funding_text = "No funding data"
        if data.get("funding"):
            funding = data["funding"]
            if isinstance(funding, list):
                funding_lines = ["Funding Rates by Exchange:"]
                for ex in funding[:5]:
                    rate = ex.get("fundingRate", 0)
                    funding_lines.append(f"  {ex.get('exchange', 'N/A')}: {rate*100:.4f}%")
                funding_text = "\n".join(funding_lines)
        
        # Format L/S ratio
        ls_text = "No L/S data"
        if data.get("ls_ratio"):
            ls = data["ls_ratio"]
            if isinstance(ls, list) and len(ls) > 0:
                latest = ls[-1]
                ls_text = f"Long/Short Ratio: {latest.get('longShortRatio', 1):.2f}"
        
        # Format taker flow
        taker_text = "No taker data"
        if data.get("taker_flow"):
            taker = data["taker_flow"]
            if isinstance(taker, list):
                total_buy = sum(t.get("buyVolUsd", 0) for t in taker)
                total_sell = sum(t.get("sellVolUsd", 0) for t in taker)
                taker_text = f"Taker Flow: Buy ${total_buy/1e6:.1f}M / Sell ${total_sell/1e6:.1f}M"
        
        return f"""Generate a MARKET STRUCTURE REPORT for {symbol}.

LIVE DATA FROM COINGLASS + HELSINKI:

## WHALE POSITIONS (Hyperliquid)
{whale_text}

## OPTIONS
{max_pain_text}

## FUNDING RATES
{funding_text}

## LONG/SHORT RATIO
{ls_text}

## TAKER FLOW
{taker_text}

---

INSTRUCTIONS:
1. Analyze the confluence of signals above
2. Identify if whales are positioned for a move
3. Calculate where the "pain trade" is (liquidation hunt target)
4. Determine if funding suggests crowded positioning
5. Generate a BULLISH and BEARISH trade scenario with:
   - Entry zone
   - Stop loss (with reasoning)
   - 3 targets (with % gain)
   - Risk/Reward ratio
6. Assign overall BIAS (BULLISH/BEARISH/NEUTRAL) and CONFIDENCE (HIGH/MEDIUM/LOW)

OUTPUT FORMAT:
Return a structured analysis with clear trade setups. Be specific with prices."""
    
    def _parse_market_structure_response(
        self,
        symbol: str,
        data: Dict,
        iros_response: Optional[str]
    ) -> Report:
        """Parse IROS response into a Report object"""
        
        # Extract key data points
        whale_data = data.get("whale_positions", [])
        total_long = sum(w.get("sizeUsd", 0) for w in (whale_data or []) if w.get("side") == "LONG")
        total_short = sum(w.get("sizeUsd", 0) for w in (whale_data or []) if w.get("side") == "SHORT")
        
        # Determine bias from data
        if total_long > total_short * 1.2:
            bias = Bias.BULLISH
        elif total_short > total_long * 1.2:
            bias = Bias.BEARISH
        else:
            bias = Bias.NEUTRAL
        
        # Get max pain
        max_pain = 0
        if data.get("max_pain") and isinstance(data["max_pain"], list):
            max_pain = data["max_pain"][0].get("maxPain", 0)
        
        report_id = f"MS-{symbol}-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        return Report(
            id=report_id,
            type=ReportType.MARKET_STRUCTURE,
            title=f"{symbol} Market Structure Analysis",
            generated_at=datetime.utcnow(),
            bias=bias,
            confidence=Confidence.HIGH if iros_response else Confidence.MEDIUM,
            summary=iros_response[:500] if iros_response else f"{symbol} whale positioning {'long' if bias == Bias.BULLISH else 'short'} heavy.",
            sections={
                "iros_analysis": iros_response,
                "whale_positioning": {
                    "total_long_usd": total_long,
                    "total_short_usd": total_short,
                    "net_bias": "LONG" if total_long > total_short else "SHORT",
                    "positions": whale_data[:10] if whale_data else []
                },
                "max_pain": max_pain,
                "funding": data.get("funding"),
                "ls_ratio": data.get("ls_ratio"),
                "taker_flow": data.get("taker_flow")
            },
            tags=[symbol.lower(), bias.value.lower(), "market-structure"],
            data_sources=["coinglass", "hyperliquid", "helsinki"]
        )
    
    # =========================================================================
    # WHALE INTELLIGENCE REPORT
    # =========================================================================
    
    async def generate_whale_report(self, symbol: str = "BTC") -> Report:
        """Generate Whale Intelligence Report using IROS"""
        logger.info(f"[MCF] Generating whale report for {symbol}")
        
        # Fetch whale data
        results = await asyncio.gather(
            self.coinglass.get_hyperliquid_whale_positions(symbol),
            self.coinglass.get_exchange_netflow(symbol),
            return_exceptions=True
        )
        
        whale_positions = self._safe_get(results[0])
        exchange_flow = self._safe_get(results[1])
        
        # Build IROS prompt
        iros_prompt = f"""Analyze WHALE ACTIVITY for {symbol}.

WHALE POSITIONS (Hyperliquid Top 20):
{self._format_whale_positions(whale_positions)}

EXCHANGE FLOWS:
{self._format_exchange_flow(exchange_flow)}

INSTRUCTIONS:
1. Identify the dominant whale positioning (long vs short)
2. Analyze which side is in profit vs underwater
3. Calculate liquidation risk for dominant positions
4. Determine if smart money is accumulating or distributing
5. Provide actionable insight for retail traders

Be specific with numbers and provide a clear trading implication."""
        
        # Get IROS analysis
        iros_result = await self.iros.process_query(iros_prompt, comprehensive=False)
        
        # Build report
        positions = whale_positions or []
        total_long = sum(w.get("sizeUsd", 0) for w in positions if w.get("side") == "LONG")
        total_short = sum(w.get("sizeUsd", 0) for w in positions if w.get("side") == "SHORT")
        
        report_id = f"WI-{symbol}-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        return Report(
            id=report_id,
            type=ReportType.WHALE_INTELLIGENCE,
            title=f"{symbol} Whale Activity: {'Longs' if total_long > total_short else 'Shorts'} Dominant",
            generated_at=datetime.utcnow(),
            bias=Bias.BULLISH if total_long > total_short else Bias.BEARISH,
            confidence=Confidence.HIGH,
            summary=iros_result.response[:500] if iros_result.success else "Whale data analyzed.",
            sections={
                "iros_analysis": iros_result.response if iros_result.success else None,
                "positions": positions[:10],
                "aggregate": {
                    "total_long_usd": total_long,
                    "total_short_usd": total_short,
                    "dominant_side": "LONGS" if total_long > total_short else "SHORTS"
                },
                "exchange_flow": exchange_flow
            },
            tags=[symbol.lower(), "whale", "hyperliquid"],
            data_sources=["hyperliquid", "coinglass"]
        )
    
    def _format_whale_positions(self, positions) -> str:
        if not positions:
            return "No whale position data available"
        
        lines = []
        for i, p in enumerate(positions[:10]):
            lines.append(
                f"{i+1}. {p.get('side', 'N/A')} ${p.get('sizeUsd', 0)/1e6:.1f}M "
                f"@ ${p.get('entryPrice', 0):,.0f} "
                f"({p.get('leverage', 0)}x) "
                f"PnL: {'+' if p.get('pnl', 0) > 0 else ''}${p.get('pnl', 0)/1e6:.2f}M"
            )
        return "\n".join(lines)
    
    def _format_exchange_flow(self, flow) -> str:
        if not flow:
            return "No exchange flow data"
        
        return f"""Inflows: ${flow.get('inflow', 0)/1e6:.1f}M
Outflows: ${flow.get('outflow', 0)/1e6:.1f}M
Net: ${(flow.get('outflow', 0) - flow.get('inflow', 0))/1e6:.1f}M {'(Bullish - leaving exchanges)' if flow.get('outflow', 0) > flow.get('inflow', 0) else '(Bearish - entering exchanges)'}"""
    
    # =========================================================================
    # OPTIONS FLOW REPORT
    # =========================================================================
    
    async def generate_options_report(self, symbol: str = "BTC") -> Report:
        """Generate Options Flow Report using IROS"""
        logger.info(f"[MCF] Generating options report for {symbol}")
        
        results = await asyncio.gather(
            self.coinglass.get_options_info(symbol),
            self.coinglass.get_options_max_pain(symbol),
            self.coinglass.get_options_oi_expiry(symbol),
            return_exceptions=True
        )
        
        options_info = self._safe_get(results[0])
        max_pain = self._safe_get(results[1])
        oi_expiry = self._safe_get(results[2])
        
        # Build IROS prompt
        iros_prompt = f"""Analyze OPTIONS FLOW for {symbol}.

OPTIONS DATA:
{self._format_options_data(options_info, max_pain, oi_expiry)}

INSTRUCTIONS:
1. Interpret the Put/Call ratio (< 0.9 = bullish, > 1.1 = bearish)
2. Explain max pain mechanics and expected price movement
3. Identify major OI concentrations by expiry
4. Determine the options market's directional bias
5. Provide specific price targets based on max pain

Be precise with numbers and expiry dates."""
        
        iros_result = await self.iros.process_query(iros_prompt, comprehensive=False)
        
        # Extract put/call ratio
        put_call = 1.0
        if options_info and isinstance(options_info, list):
            for item in options_info:
                if item.get("exchange") == "All":
                    put_call = item.get("putCallRatio", 1.0)
                    break
        
        # Extract max pain price
        max_pain_price = 0
        if max_pain and isinstance(max_pain, list) and len(max_pain) > 0:
            max_pain_price = max_pain[0].get("maxPain", 0)
        
        report_id = f"OF-{symbol}-{datetime.utcnow().strftime('%Y%m%d-%H')}"
        
        return Report(
            id=report_id,
            type=ReportType.OPTIONS_FLOW,
            title=f"{symbol} Options: P/C {put_call:.2f} - {'Bullish' if put_call < 0.9 else 'Bearish' if put_call > 1.1 else 'Neutral'}",
            generated_at=datetime.utcnow(),
            bias=Bias.BULLISH if put_call < 0.9 else Bias.BEARISH if put_call > 1.1 else Bias.NEUTRAL,
            confidence=Confidence.HIGH,
            summary=iros_result.response[:500] if iros_result.success else f"Put/Call at {put_call:.2f}. Max pain at ${max_pain_price:,.0f}.",
            sections={
                "iros_analysis": iros_result.response if iros_result.success else None,
                "put_call_ratio": put_call,
                "max_pain": max_pain_price,
                "max_pain_data": max_pain,
                "oi_by_expiry": oi_expiry
            },
            tags=[symbol.lower(), "options", "max-pain"],
            data_sources=["coinglass"]
        )
    
    def _format_options_data(self, info, max_pain, expiry) -> str:
        lines = []
        
        if info and isinstance(info, list):
            for item in info:
                if item.get("exchange") == "All":
                    lines.append(f"Put/Call Ratio: {item.get('putCallRatio', 0):.2f}")
                    lines.append(f"Total Open Interest: ${item.get('openInterest', 0)/1e9:.2f}B")
                    break
        
        if max_pain and isinstance(max_pain, list):
            lines.append("\nMax Pain by Expiry:")
            for mp in max_pain[:5]:
                lines.append(f"  {mp.get('expiryDate', 'N/A')}: ${mp.get('maxPain', 0):,.0f}")
        
        if expiry and isinstance(expiry, list):
            lines.append("\nOI by Expiry:")
            for exp in expiry[:5]:
                lines.append(f"  {exp.get('expiryDate', 'N/A')}: ${exp.get('openInterest', 0)/1e9:.2f}B")
        
        return "\n".join(lines) if lines else "No options data available"
    
    # =========================================================================
    # CYCLE POSITION REPORT (BTC only)
    # =========================================================================
    
    async def generate_cycle_report(self) -> Report:
        """Generate Cycle Position Report using IROS (BTC only)"""
        logger.info("[MCF] Generating cycle report for BTC")
        
        results = await asyncio.gather(
            self.coinglass.get_bitcoin_bubble_index(),
            self.coinglass.get_ahr999_index(),
            self.coinglass.get_puell_multiple(),
            return_exceptions=True
        )
        
        bubble = self._safe_get(results[0])
        ahr999 = self._safe_get(results[1])
        puell = self._safe_get(results[2])
        
        # Extract values
        bubble_val = self._get_latest_value(bubble, "value")
        ahr999_val = self._get_latest_value(ahr999, "ahr999")
        puell_val = self._get_latest_value(puell, "puellMultiple")
        
        # Build IROS prompt
        iros_prompt = f"""Analyze BITCOIN CYCLE POSITION.

CYCLE INDICATORS (Coinglass):

1. BUBBLE INDEX: {bubble_val}
   - Below 0 = Undervalued
   - Above 3 = Overvalued
   - Above 6 = Bubble territory

2. AHR999 INDEX: {ahr999_val}
   - Below 0.45 = Strong Buy (DCA heavily)
   - 0.45-0.8 = Buy zone
   - 0.8-1.2 = Hold
   - Above 1.2 = Take profits

3. PUELL MULTIPLE: {puell_val}
   - Below 0.5 = Miner capitulation (extreme bottom)
   - 0.5-1.0 = Undervalued
   - 1.0-2.0 = Fair value
   - Above 4.0 = Miner euphoria (top)

INSTRUCTIONS:
1. Weight all 3 indicators to determine cycle phase
2. Identify if we're in ACCUMULATION, MARKUP, DISTRIBUTION, or MARKDOWN
3. Compare to historical cycle patterns
4. Provide specific DCA or exit strategy based on phase
5. Estimate probability of being at a cycle extreme

Be specific with your cycle phase determination."""
        
        iros_result = await self.iros.process_query(iros_prompt, comprehensive=False)
        
        # Determine cycle phase
        score = 0
        if bubble_val < 0: score += 30
        elif bubble_val < 3: score += 10
        else: score -= 20
        
        if ahr999_val < 0.45: score += 30
        elif ahr999_val < 0.8: score += 15
        elif ahr999_val > 1.2: score -= 20
        
        if puell_val < 0.5: score += 30
        elif puell_val < 1.0: score += 15
        elif puell_val > 4.0: score -= 30
        
        if score >= 50:
            phase = "ACCUMULATION"
            bias = Bias.BULLISH
        elif score >= 20:
            phase = "EARLY MARKUP"
            bias = Bias.BULLISH
        elif score >= 0:
            phase = "MARKUP"
            bias = Bias.NEUTRAL
        else:
            phase = "DISTRIBUTION/MARKDOWN"
            bias = Bias.BEARISH
        
        report_id = f"CP-{datetime.utcnow().strftime('%Y%m%d')}"
        
        return Report(
            id=report_id,
            type=ReportType.CYCLE_POSITION,
            title=f"Bitcoin Cycle: {phase}",
            generated_at=datetime.utcnow(),
            bias=bias,
            confidence=Confidence.HIGH,
            summary=iros_result.response[:500] if iros_result.success else f"Cycle phase: {phase}. Score: {score}.",
            sections={
                "iros_analysis": iros_result.response if iros_result.success else None,
                "indicators": {
                    "bubble_index": bubble_val,
                    "ahr999": ahr999_val,
                    "puell_multiple": puell_val
                },
                "phase": phase,
                "score": score
            },
            tags=["btc", "cycle", phase.lower().replace(" ", "-")],
            data_sources=["coinglass"]
        )
    
    def _get_latest_value(self, data, key: str, default: float = 0) -> float:
        if not data:
            return default
        if isinstance(data, list) and len(data) > 0:
            return data[-1].get(key, default)
        if isinstance(data, dict):
            return data.get(key, default)
        return default
```

---

## Step 2: Update Scheduler for Multi-Coin

Update `mcf_labs/scheduler.py`:

```python
from .generator import IROSReportGenerator, SUPPORTED_COINS

class ReportScheduler:
    def __init__(self, generator: IROSReportGenerator, storage_path: str = "data/reports"):
        self.generator = generator
        # ... existing code ...
    
    async def run_all_market_structure(self):
        """Generate market structure for all coins"""
        reports = await self.generator.generate_all_coins_report("market_structure")
        for report in reports:
            await self.save_report(report)
        logger.info(f"Generated {len(reports)} market structure reports")
    
    async def run_all_whale_reports(self):
        """Generate whale reports for all coins"""
        reports = await self.generator.generate_all_coins_report("whale_intelligence")
        for report in reports:
            await self.save_report(report)
        logger.info(f"Generated {len(reports)} whale reports")
```

---

## Step 3: API Endpoints for Research Terminal

Add these to `api/terminal_api.py`:

```python
from mcf_labs.storage import get_storage
from mcf_labs.generator import IROSReportGenerator

# Initialize on startup
mcf_generator: IROSReportGenerator = None

def init_mcf():
    global mcf_generator
    if mcf_generator is None:
        mcf_generator = IROSReportGenerator(
            coinglass=coinglass,
            helsinki=helsinki,
            model_url=os.getenv("BASTION_MODEL_URL"),
            model_api_key=os.getenv("BASTION_MODEL_API_KEY")
        )

@app.get("/api/mcf/reports")
async def get_mcf_reports(
    type: str = None,
    symbol: str = None,
    limit: int = 20,
    offset: int = 0
):
    """Get MCF Labs reports for Research Terminal"""
    storage = get_storage()
    reports = storage.get_reports_for_research_terminal(limit=limit)
    
    # Filter by type
    if type:
        reports = [r for r in reports if r["type"] == type]
    
    # Filter by symbol
    if symbol:
        reports = [r for r in reports if symbol.lower() in r.get("tags", [])]
    
    return {"success": True, "reports": reports[offset:offset+limit]}

@app.get("/api/mcf/reports/{report_id}")
async def get_mcf_report_detail(report_id: str):
    """Get full report by ID"""
    storage = get_storage()
    report = storage.get_report(report_id)
    
    if report:
        return {"success": True, "report": report.to_dict()}
    return {"success": False, "error": "Report not found"}

@app.get("/api/mcf/reports/latest")
async def get_latest_mcf_reports():
    """Get most recent report of each type"""
    storage = get_storage()
    latest = storage.get_latest_by_type()
    
    return {
        "success": True,
        "reports": {k: v.to_dict() if v else None for k, v in latest.items()}
    }

@app.post("/api/mcf/generate/{report_type}/{symbol}")
async def generate_mcf_report(report_type: str, symbol: str = "BTC"):
    """Manually trigger report generation"""
    init_mcf()
    
    try:
        if report_type == "market_structure":
            report = await mcf_generator.generate_market_structure(symbol)
        elif report_type == "whale":
            report = await mcf_generator.generate_whale_report(symbol)
        elif report_type == "options":
            report = await mcf_generator.generate_options_report(symbol)
        elif report_type == "cycle":
            report = await mcf_generator.generate_cycle_report()
        else:
            return {"success": False, "error": f"Unknown report type: {report_type}"}
        
        # Save report
        storage = get_storage()
        # ... save logic ...
        
        return {"success": True, "report": report.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## Step 4: Environment Variables

Required in Railway/Vercel:

```env
# Coinglass Premium ($400/mo)
COINGLASS_API_KEY=your_key_here

# Helsinki VM (free)
HELSINKI_API_URL=http://77.42.29.188:5002

# IROS Model (Vast.ai)
BASTION_MODEL_URL=https://your-vast-instance.trycloudflare.com
BASTION_MODEL_API_KEY=optional_if_needed

# Whale Alert
WHALE_ALERT_API_KEY=your_key_here
```

---

## The IROS Intelligence Loop

```
USER REQUESTS REPORT
        │
        ▼
┌───────────────────┐
│   DATA FETCH      │  ← Coinglass + Helsinki
│   (13 endpoints)  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   BUILD PROMPT    │  ← Structure raw data
│   with context    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   IROS ANALYSIS   │  ← 32B LLM interprets
│   (BastionAI)     │     - Identifies signals
│                   │     - Spots conflicts
│                   │     - Creates trade setups
│                   │     - Assigns confidence
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   PARSE RESPONSE  │  ← Extract structured data
│   into Report     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   SAVE + SERVE    │  → Research Terminal
│   to storage      │  → MCF Labs section
└───────────────────┘
```

---

## Key Points for Your Agent

1. **IROS is the brain** - It turns raw numbers into intelligent analysis
2. **Multi-coin support** - Loop through SUPPORTED_COINS for batch reports
3. **Data sources** - Coinglass (13 endpoints) + Helsinki (33 endpoints)
4. **Report storage** - JSON files in `data/reports/{type}/{year}/{month}/`
5. **API endpoints** - `/api/mcf/reports`, `/api/mcf/reports/{id}`, `/api/mcf/generate/{type}/{symbol}`

The generator already has the scaffolding - your agent just needs to:
1. Wire up the IROS calls properly
2. Add the API endpoints to terminal_api.py
3. Ensure scheduler starts on app startup

