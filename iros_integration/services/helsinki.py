"""
Helsinki VM Data Aggregator
============================
Fetches real-time quant data from all 33 Helsinki endpoints

Base URL: http://77.42.29.188:5002
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import httpx

from ..config.settings import settings


# =============================================================================
# ALL 33 HELSINKI ENDPOINTS
# =============================================================================

# Endpoints that need {symbol} in path (dynamic)
SYMBOL_ENDPOINTS = {
    # ORDER FLOW
    "cvd": "/quant/cvd/{symbol}",
    "orderbook": "/quant/orderbook/{symbol}",
    "large_trades": "/quant/large-trades/{symbol}",
    "smart_money": "/quant/smart-money/{symbol}",
    "whale_flow": "/quant/whale-flow/{symbol}",
    
    # DERIVATIVES
    "basis": "/quant/basis/{symbol}",
    "open_interest": "/quant/open-interest/{symbol}",
    "greeks": "/quant/greeks/{symbol}",
    "liquidation_map": "/quant/liquidation-map/{symbol}",
    "liquidation_estimate": "/quant/liquidation-estimate/{symbol}",
    "options_iv": "/quant/options-iv/{symbol}",
    
    # VOLATILITY
    "iv_rv_spread": "/quant/iv-rv-spread/{symbol}",
    "volatility": "/quant/volatility/{symbol}",
    
    # TECHNICAL
    "vwap": "/quant/vwap/{symbol}",
    "momentum": "/quant/momentum/{symbol}",
    "mean_reversion": "/quant/mean-reversion/{symbol}",
    "drawdown": "/quant/drawdown/{symbol}",
    
    # FULL CONTEXT
    "full": "/quant/full/{symbol}",
}

# Static endpoints (no symbol needed)
STATIC_ENDPOINTS = {
    # DERIVATIVES
    "funding": "/derivatives/funding",
    "derivatives_oi": "/derivatives/oi",
    "long_short": "/derivatives/long-short",
    "derivatives_basis": "/derivatives/basis",
    "funding_arb": "/quant/funding-arb",
    
    # MACRO
    "dominance": "/quant/dominance",
    "defi_tvl": "/quant/defi-tvl",
    "gas": "/quant/gas",
    "macro": "/quant/macro",
    "stablecoin_supply": "/quant/stablecoin-supply",
    
    # SENTIMENT
    "fear_greed": "/sentiment/fear-greed",
    "stablecoin_dominance": "/sentiment/stablecoin-dominance",
    
    # OPTIONS
    "options_skew": "/options/skew",
    
    # FULL CONTEXT
    "context_full": "/context/full",
}


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class EndpointResult:
    """Result from a single endpoint fetch"""
    endpoint: str
    data: Optional[Any]
    success: bool
    latency_ms: int
    error: Optional[str] = None


@dataclass
class AggregatedMarketData:
    """Comprehensive market data from Helsinki"""
    symbol: str
    timestamp: int
    order_flow: Dict[str, Any] = field(default_factory=dict)
    derivatives: Dict[str, Any] = field(default_factory=dict)
    volatility: Dict[str, Any] = field(default_factory=dict)
    technical: Dict[str, Any] = field(default_factory=dict)
    macro: Dict[str, Any] = field(default_factory=dict)
    sentiment: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HELSINKI CLIENT
# =============================================================================

class HelsinkiClient:
    """
    Client for fetching data from Helsinki VM quant endpoints.
    
    Usage:
        client = HelsinkiClient()
        data = await client.fetch_full_data("BTC")
        formatted = client.format_for_prompt(data)
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        self.base_url = base_url or settings.helsinki.base_url
        self.timeout = timeout or settings.helsinki.timeout
    
    async def fetch_endpoint(
        self, 
        endpoint: str, 
        symbol: Optional[str] = None
    ) -> EndpointResult:
        """Fetch a single Helsinki endpoint"""
        start_time = time.time()
        
        # Replace {symbol} placeholder
        resolved_endpoint = endpoint.replace("{symbol}", symbol) if symbol else endpoint
        url = f"{self.base_url}{resolved_endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout / 1000) as client:
                response = await client.get(url)
                latency = int((time.time() - start_time) * 1000)
                
                if response.status_code != 200:
                    return EndpointResult(
                        endpoint=resolved_endpoint,
                        data=None,
                        success=False,
                        latency_ms=latency,
                        error=f"HTTP {response.status_code}"
                    )
                
                return EndpointResult(
                    endpoint=resolved_endpoint,
                    data=response.json(),
                    success=True,
                    latency_ms=latency
                )
                
        except Exception as e:
            return EndpointResult(
                endpoint=resolved_endpoint,
                data=None,
                success=False,
                latency_ms=int((time.time() - start_time) * 1000),
                error=str(e)
            )
    
    async def fetch_endpoints_parallel(
        self,
        endpoints: List[str],
        symbol: Optional[str] = None
    ) -> List[EndpointResult]:
        """Fetch multiple endpoints in parallel"""
        tasks = [self.fetch_endpoint(ep, symbol) for ep in endpoints]
        return await asyncio.gather(*tasks)
    
    async def fetch_full_data(self, symbol: str = "BTC") -> AggregatedMarketData:
        """
        Fetch comprehensive market data for a symbol.
        Uses /quant/full/{symbol} for efficiency plus key static endpoints.
        """
        start_time = time.time()
        
        # Priority endpoints - full data in one call + key statics
        symbol_endpoints = [
            SYMBOL_ENDPOINTS["full"],
            SYMBOL_ENDPOINTS["options_iv"],
        ]
        static_endpoints = [
            STATIC_ENDPOINTS["fear_greed"],
            STATIC_ENDPOINTS["dominance"],
            STATIC_ENDPOINTS["gas"],
            STATIC_ENDPOINTS["defi_tvl"],
        ]
        
        # Fetch in parallel
        symbol_results, static_results = await asyncio.gather(
            self.fetch_endpoints_parallel(symbol_endpoints, symbol),
            self.fetch_endpoints_parallel(static_endpoints),
        )
        
        all_results = symbol_results + static_results
        successful = [r for r in all_results if r.success]
        total_latency = int((time.time() - start_time) * 1000)
        
        # Find data helper
        def find_data(pattern: str) -> Optional[Any]:
            for r in all_results:
                if pattern in r.endpoint and r.success:
                    return r.data
            return None
        
        # Get full quant data
        full_data = find_data("full")
        options_iv = find_data("options-iv")
        
        return AggregatedMarketData(
            symbol=symbol,
            timestamp=int(time.time() * 1000),
            order_flow={
                "smart_money": full_data.get("smart_money") if full_data else None,
            },
            derivatives={
                "funding_arb": full_data.get("funding_arb") if full_data else None,
                "liquidation": full_data.get("liquidation") if full_data else None,
            },
            volatility={
                "volatility": full_data.get("volatility") if full_data else None,
                "options_iv": options_iv or (full_data.get("options_iv") if full_data else None),
            },
            technical={},
            macro={
                "dominance": find_data("dominance"),
                "gas": find_data("gas"),
                "defi_tvl": find_data("defi-tvl"),
            },
            sentiment={
                "fear_greed": find_data("fear-greed"),
            },
            signals={},
            meta={
                "endpoints_queried": len(all_results),
                "endpoints_successful": len(successful),
                "total_latency_ms": total_latency,
                "sources": [r.endpoint for r in successful],
                "full_data_available": full_data is not None,
                "current_price": (
                    options_iv.get("underlying_price") if options_iv else 
                    (full_data.get("options_iv", {}).get("underlying_price") if full_data else None)
                ),
            },
        )
    
    async def fetch_comprehensive_data(self, symbol: str = "BTC") -> AggregatedMarketData:
        """
        Fetch ALL endpoints for comprehensive analysis.
        Slower but more complete.
        """
        start_time = time.time()
        
        symbol_endpoints = list(SYMBOL_ENDPOINTS.values())
        static_endpoints = list(STATIC_ENDPOINTS.values())
        
        symbol_results, static_results = await asyncio.gather(
            self.fetch_endpoints_parallel(symbol_endpoints, symbol),
            self.fetch_endpoints_parallel(static_endpoints),
        )
        
        all_results = symbol_results + static_results
        successful = [r for r in all_results if r.success]
        total_latency = int((time.time() - start_time) * 1000)
        
        def find_data(pattern: str) -> Optional[Any]:
            for r in all_results:
                if pattern in r.endpoint and r.success:
                    return r.data
            return None
        
        return AggregatedMarketData(
            symbol=symbol,
            timestamp=int(time.time() * 1000),
            order_flow={
                "cvd": find_data("cvd"),
                "orderbook": find_data("orderbook"),
                "large_trades": find_data("large-trades"),
                "smart_money": find_data("smart-money"),
                "whale_flow": find_data("whale-flow"),
            },
            derivatives={
                "funding": find_data("funding"),
                "basis": find_data("basis"),
                "open_interest": find_data("open-interest"),
                "greeks": find_data("greeks"),
                "liquidation_map": find_data("liquidation-map"),
            },
            volatility={
                "iv_rv_spread": find_data("iv-rv-spread"),
                "volatility": find_data("volatility"),
                "options_iv": find_data("options-iv"),
            },
            technical={
                "vwap": find_data("vwap"),
                "momentum": find_data("momentum"),
                "mean_reversion": find_data("mean-reversion"),
                "drawdown": find_data("drawdown"),
            },
            macro={
                "dominance": find_data("dominance"),
                "defi_tvl": find_data("defi-tvl"),
                "gas": find_data("gas"),
                "macro": find_data("macro"),
                "stablecoin_supply": find_data("stablecoin-supply"),
            },
            sentiment={
                "fear_greed": find_data("fear-greed"),
                "stablecoin_dominance": find_data("stablecoin-dominance"),
            },
            signals={
                "long_short": find_data("long-short"),
            },
            meta={
                "endpoints_queried": len(all_results),
                "endpoints_successful": len(successful),
                "total_latency_ms": total_latency,
                "sources": [r.endpoint for r in successful],
            },
        )
    
    @staticmethod
    def format_for_prompt(data: AggregatedMarketData) -> str:
        """
        Format market data for LLM prompt injection.
        Compact format optimized for 4K context window.
        """
        lines = []
        
        # Extract current price
        current_price = data.meta.get("current_price")
        price_str = f"${current_price:,.2f}" if current_price else "N/A"
        
        lines.append(f"## {data.symbol} LIVE DATA ({data.meta.get('endpoints_successful', 0)} sources)")
        lines.append(f"**PRICE: {price_str}** - USE THIS FOR ALL CALCULATIONS\n")
        
        # Smart Money
        sm = data.order_flow.get("smart_money")
        if sm:
            lines.append(
                f"**SMART MONEY:** {sm.get('smart_money_bias', 'N/A')} | "
                f"Divergence: {sm.get('divergence', 'N/A'):.2f} | "
                f"Trend: {sm.get('trend', 'N/A')}"
            )
        
        # Funding
        funding = data.derivatives.get("funding_arb")
        if funding and funding.get("all_rates", {}).get("binance"):
            btc_funding = funding["all_rates"]["binance"].get("BTCUSDT", 0)
            lines.append(f"**FUNDING:** BTC {btc_funding * 100:.4f}% (Binance)")
        
        # Liquidation
        liq = data.derivatives.get("liquidation")
        if liq and liq.get("current_price", 0) > 0:
            oi_b = liq.get("open_interest_usd", 0) / 1e9
            lines.append(
                f"**LIQUIDATION:** OI ${oi_b:.1f}B | "
                f"L/S: {liq.get('long_short_ratio', 'N/A'):.2f} | "
                f"Cascade: {liq.get('cascade_bias', 'N/A')}"
            )
            
            down_zones = liq.get("downside_liquidation_zones", [])
            up_zones = liq.get("upside_liquidation_zones", [])
            if down_zones:
                z = down_zones[0]
                lines.append(
                    f"  ↓ ${z.get('price', 0):.0f} "
                    f"(-{abs(z.get('distance_pct', 0))}%) = "
                    f"${z.get('estimated_usd_at_risk', 0) / 1e6:.0f}M at risk"
                )
            if up_zones:
                z = up_zones[0]
                lines.append(
                    f"  ↑ ${z.get('price', 0):.0f} "
                    f"(+{z.get('distance_pct', 0)}%) = "
                    f"${z.get('estimated_usd_at_risk', 0) / 1e6:.0f}M at risk"
                )
        
        # Volatility
        vol = data.volatility.get("volatility")
        opt_iv = data.volatility.get("options_iv")
        if vol:
            lines.append(
                f"**VOLATILITY:** Regime: {vol.get('current_regime', 'N/A')} "
                f"({vol.get('confidence', 0)}% conf) | "
                f"7d: {vol.get('volatility_7d_pct', 0):.1f}%"
            )
        if opt_iv:
            lines.append(
                f"**OPTIONS IV:** ATM {opt_iv.get('atm_implied_volatility_pct', 0):.1f}% | "
                f"Skew: {opt_iv.get('skew_interpretation', 'NEUTRAL')}"
            )
        
        # Macro
        dom = data.macro.get("dominance")
        if dom:
            lines.append(
                f"**MACRO:** BTC Dom {dom.get('btc_dominance_pct', 0):.1f}% | "
                f"Alt Season: {dom.get('alt_season_score', 0)}/100 ({dom.get('season', 'N/A')})"
            )
        
        # Sentiment
        fg = data.sentiment.get("fear_greed")
        if fg:
            lines.append(
                f"**SENTIMENT:** Fear/Greed {fg.get('value', 0)}/100 "
                f"({fg.get('label', 'N/A').replace('_', ' ').upper()})"
            )
        
        return "\n".join(lines)
    
    # Specific query type fetchers
    async def fetch_cvd_data(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Fetch CVD-specific data for focused CVD analysis"""
        results = await self.fetch_endpoints_parallel(
            [SYMBOL_ENDPOINTS["cvd"], SYMBOL_ENDPOINTS["options_iv"]],
            symbol
        )
        
        cvd = next((r.data for r in results if "cvd" in r.endpoint and r.success), None)
        price_data = next((r.data for r in results if "options-iv" in r.endpoint and r.success), None)
        price = price_data.get("underlying_price") if price_data else None
        
        formatted = f"## CVD ANALYSIS - {symbol}\n"
        formatted += f"**Price:** ${price:,.2f}\n\n" if price else "**Price:** N/A\n\n"
        
        if cvd:
            formatted += f"**CVD 1H:** {cvd.get('cvd_1h', 0):.3f} BTC\n"
            formatted += f"**CVD 4H:** {cvd.get('cvd_4h', 0):.3f} BTC\n"
            formatted += f"**Divergence:** {cvd.get('divergence', 'N/A')}\n"
            formatted += f"**Signal:** {cvd.get('signal', 'N/A')}\n"
        
        return {"cvd": cvd, "price": price, "formatted": formatted}
    
    async def fetch_volatility_data(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Fetch volatility-specific data"""
        results = await self.fetch_endpoints_parallel(
            [
                SYMBOL_ENDPOINTS["volatility"],
                SYMBOL_ENDPOINTS["iv_rv_spread"],
                SYMBOL_ENDPOINTS["options_iv"],
            ],
            symbol
        )
        
        vol = next((r.data for r in results if "/volatility" in r.endpoint and r.success), None)
        iv_rv = next((r.data for r in results if "iv-rv" in r.endpoint and r.success), None)
        price_data = next((r.data for r in results if "options-iv" in r.endpoint and r.success), None)
        
        return {
            "volatility": vol,
            "iv_rv": iv_rv,
            "price": price_data.get("underlying_price") if price_data else None,
        }
    
    async def fetch_liquidation_data(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Fetch liquidation-specific data"""
        results = await self.fetch_endpoints_parallel(
            [SYMBOL_ENDPOINTS["liquidation_estimate"], SYMBOL_ENDPOINTS["options_iv"]],
            symbol
        )
        
        liq = next((r.data for r in results if "liquidation" in r.endpoint and r.success), None)
        price_data = next((r.data for r in results if "options-iv" in r.endpoint and r.success), None)
        
        return {
            "liquidation": liq,
            "price": price_data.get("underlying_price") if price_data else None,
        }










