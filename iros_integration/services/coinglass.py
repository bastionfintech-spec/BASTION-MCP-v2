"""
Coinglass Premium API Client
=============================
Real liquidation data, open interest, funding rates, L/S ratios

API Key: Pre-configured ($299/mo plan)
Docs: https://coinglass.com/api
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import httpx

from ..config.settings import settings


@dataclass
class CoinglassResponse:
    """Response from Coinglass API"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: int = 0


class CoinglassClient:
    """
    Client for Coinglass Premium API.
    
    Features:
    - Real liquidation data (not estimates)
    - Open interest by exchange
    - Funding rates across exchanges
    - Long/short ratios
    
    Usage:
        client = CoinglassClient()
        liquidations = await client.get_liquidation_history("BTC")
        oi = await client.get_open_interest("ETH")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.coinglass.api_key
        self.base_url = settings.coinglass.base_url
        self.timeout = settings.coinglass.timeout / 1000
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> CoinglassResponse:
        """Make authenticated request to Coinglass API"""
        start_time = time.time()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers={"CG-API-KEY": self.api_key}
                )
                latency = int((time.time() - start_time) * 1000)
                
                if response.status_code != 200:
                    return CoinglassResponse(
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text}",
                        latency_ms=latency
                    )
                
                data = response.json()
                
                # Coinglass wraps data in a "data" field
                if isinstance(data, dict) and "data" in data:
                    return CoinglassResponse(
                        success=True,
                        data=data["data"],
                        latency_ms=latency
                    )
                
                return CoinglassResponse(
                    success=True,
                    data=data,
                    latency_ms=latency
                )
                
        except Exception as e:
            return CoinglassResponse(
                success=False,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000)
            )
    
    # =========================================================================
    # LIQUIDATION ENDPOINTS
    # =========================================================================
    
    async def get_liquidation_history(
        self,
        symbol: str = "BTC",
        interval: str = "h1",
        limit: int = 100
    ) -> CoinglassResponse:
        """
        Get aggregated liquidation history.
        
        Args:
            symbol: BTC, ETH, SOL, etc.
            interval: h1, h4, h12, h24
            limit: Number of records (max 500)
        """
        return await self._request(
            "/futures/liquidation/aggregated-history",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
    
    async def get_liquidation_map(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get liquidation heatmap data"""
        return await self._request(
            "/futures/liquidation/heatmap",
            {"symbol": symbol}
        )
    
    # =========================================================================
    # OPEN INTEREST ENDPOINTS
    # =========================================================================
    
    async def get_open_interest(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get current open interest across all exchanges"""
        return await self._request(
            "/futures/openInterest/exchange-list",
            {"symbol": symbol}
        )
    
    async def get_open_interest_history(
        self,
        symbol: str = "BTC",
        interval: str = "h1",
        limit: int = 100
    ) -> CoinglassResponse:
        """Get open interest history"""
        return await self._request(
            "/futures/openInterest/ohlc-aggregated-history",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
    
    # =========================================================================
    # FUNDING RATE ENDPOINTS
    # =========================================================================
    
    async def get_funding_rates(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get current funding rates across exchanges"""
        return await self._request(
            "/futures/fundingRate/exchange-list",
            {"symbol": symbol}
        )
    
    async def get_funding_history(
        self,
        symbol: str = "BTC",
        limit: int = 100
    ) -> CoinglassResponse:
        """Get funding rate history"""
        return await self._request(
            "/futures/fundingRate/history",
            {"symbol": symbol, "limit": limit}
        )
    
    # =========================================================================
    # LONG/SHORT RATIO ENDPOINTS
    # =========================================================================
    
    async def get_long_short_ratio(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get global long/short account ratio"""
        return await self._request(
            "/futures/globalLongShortAccountRatio/history",
            {"symbol": symbol}
        )
    
    async def get_top_trader_sentiment(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get top trader long/short ratio"""
        return await self._request(
            "/futures/topLongShortAccountRatio/history",
            {"symbol": symbol}
        )
    
    # =========================================================================
    # OPTIONS ENDPOINTS
    # =========================================================================
    
    async def get_options_info(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get options market data - open interest, volume, put/call ratio"""
        return await self._request(
            "/option/info",
            {"symbol": symbol}
        )
    
    async def get_options_max_pain(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get options max pain price"""
        return await self._request(
            "/option/max-pain",
            {"symbol": symbol}
        )
    
    async def get_options_oi_expiry(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get options OI by expiry date"""
        return await self._request(
            "/option/oi-expiry",
            {"symbol": symbol}
        )
    
    # =========================================================================
    # TAKER BUY/SELL ENDPOINTS
    # =========================================================================
    
    async def get_taker_buy_sell(self, symbol: str = "BTC", interval: str = "h1") -> CoinglassResponse:
        """Get taker buy/sell volume ratio"""
        return await self._request(
            "/futures/takerbuy-sell-vol/exchange-list",
            {"symbol": symbol, "interval": interval}
        )
    
    async def get_taker_buy_sell_history(
        self,
        symbol: str = "BTC",
        interval: str = "h1",
        limit: int = 100
    ) -> CoinglassResponse:
        """Get historical taker buy/sell volume"""
        return await self._request(
            "/futures/takerbuy-sell-vol/aggregated-history",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
    
    # =========================================================================
    # EXCHANGE FLOW ENDPOINTS
    # =========================================================================
    
    async def get_exchange_netflow(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get exchange inflow/outflow (on-chain data)"""
        return await self._request(
            "/futures/exchange-flow/exchange-list",
            {"symbol": symbol}
        )
    
    async def get_exchange_balance(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get exchange balances"""
        return await self._request(
            "/futures/exchange-balance/exchange-list",
            {"symbol": symbol}
        )
    
    # =========================================================================
    # PRICE DATA
    # =========================================================================
    
    async def get_price(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get current price from Coinglass"""
        return await self._request(
            "/futures/price",
            {"symbol": symbol}
        )
    
    async def get_ohlc(self, symbol: str = "BTC", interval: str = "1h", limit: int = 100) -> CoinglassResponse:
        """Get OHLC candlestick data"""
        return await self._request(
            "/futures/ohlc-aggregated-history",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
    
    # =========================================================================
    # ETF DATA ENDPOINTS
    # =========================================================================
    
    async def get_bitcoin_etf(self) -> CoinglassResponse:
        """Get Bitcoin ETF flow data - IBIT, FBTC, GBTC, etc."""
        return await self._request("/bitcoin-etf/flows")
    
    async def get_grayscale_holdings(self) -> CoinglassResponse:
        """Get Grayscale fund holdings"""
        return await self._request("/grayscale/holdings")
    
    # =========================================================================
    # LIQUIDATION BY EXCHANGE
    # =========================================================================
    
    async def get_liquidation_by_exchange(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get liquidations broken down by exchange"""
        return await self._request(
            "/futures/liquidation/exchange-list",
            {"symbol": symbol}
        )
    
    async def get_liquidation_coin_list(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get liquidation coin list - more reliable heatmap endpoint"""
        return await self._request(
            "/futures/liquidation/coin-list",
            {"symbol": symbol}
        )
    
    # =========================================================================
    # COINS MARKETS - MEGA ENDPOINT
    # =========================================================================
    
    async def get_coins_markets(self) -> CoinglassResponse:
        """
        MEGA endpoint - Comprehensive market data for all coins.
        Returns: price, OI, funding, L/S ratio, liquidations at 1h/4h/12h/24h
        """
        return await self._request("/futures/coins-markets")
    
    # =========================================================================
    # HYPERLIQUID WHALE POSITIONS
    # =========================================================================
    
    async def get_hyperliquid_whale_positions(self, symbol: str = None) -> CoinglassResponse:
        """Get top 20 whale positions on Hyperliquid with entry, leverage, PnL"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("/hyperliquid/whale-position", params if params else None)
    
    # =========================================================================
    # MARKET INDICATORS
    # =========================================================================
    
    async def get_bitcoin_bubble_index(self) -> CoinglassResponse:
        """Get Bitcoin Bubble Index - BTC valuation indicator"""
        return await self._request("/index/bitcoin-bubble-index")
    
    async def get_ahr999_index(self) -> CoinglassResponse:
        """Get AHR999 Index - DCA timing indicator"""
        return await self._request("/index/ahr999")
    
    async def get_puell_multiple(self) -> CoinglassResponse:
        """Get Puell Multiple - Mining cycle indicator"""
        return await self._request("/index/puell-multiple")
    
    async def get_fear_greed_index(self) -> CoinglassResponse:
        """Get Fear & Greed Index"""
        return await self._request("/index/fear-greed")
    
    # =========================================================================
    # ADVANCED FUNDING RATE ENDPOINTS
    # =========================================================================
    
    async def get_oi_weighted_funding_rate(self, symbol: str = "BTC", interval: str = "h1", limit: int = 100) -> CoinglassResponse:
        """Get OI-weighted funding rate history"""
        return await self._request(
            "/futures/fundingRate/oi-weight-ohlc-history",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
    
    async def get_vol_weighted_funding_rate(self, symbol: str = "BTC", interval: str = "h1", limit: int = 100) -> CoinglassResponse:
        """Get volume-weighted funding rate history"""
        return await self._request(
            "/futures/fundingRate/vol-weight-ohlc-history",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
    
    # =========================================================================
    # AGGREGATED DATA
    # =========================================================================
    
    async def get_market_overview(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get comprehensive market data for a symbol.
        Fetches multiple endpoints in parallel.
        """
        import asyncio
        
        results = await asyncio.gather(
            self.get_open_interest(symbol),
            self.get_funding_rates(symbol),
            self.get_long_short_ratio(symbol),
            self.get_liquidation_history(symbol, limit=24),
            return_exceptions=True
        )
        
        return {
            "symbol": symbol,
            "open_interest": results[0].data if isinstance(results[0], CoinglassResponse) and results[0].success else None,
            "funding_rates": results[1].data if isinstance(results[1], CoinglassResponse) and results[1].success else None,
            "long_short_ratio": results[2].data if isinstance(results[2], CoinglassResponse) and results[2].success else None,
            "liquidations_24h": results[3].data if isinstance(results[3], CoinglassResponse) and results[3].success else None,
        }
    
    async def get_full_market_data(self) -> Dict[str, Any]:
        """
        Fetch all 13 working endpoints in parallel for comprehensive market intelligence.
        """
        import asyncio
        
        results = await asyncio.gather(
            self.get_coins_markets(),
            self.get_bitcoin_etf(),
            self.get_hyperliquid_whale_positions(),
            self.get_bitcoin_bubble_index(),
            self.get_ahr999_index(),
            self.get_puell_multiple(),
            self.get_fear_greed_index(),
            self.get_options_info("BTC"),
            self.get_options_max_pain("BTC"),
            return_exceptions=True
        )
        
        return {
            "coins_markets": results[0].data if isinstance(results[0], CoinglassResponse) and results[0].success else None,
            "etf_flows": results[1].data if isinstance(results[1], CoinglassResponse) and results[1].success else None,
            "hyperliquid_whales": results[2].data if isinstance(results[2], CoinglassResponse) and results[2].success else None,
            "bubble_index": results[3].data if isinstance(results[3], CoinglassResponse) and results[3].success else None,
            "ahr999": results[4].data if isinstance(results[4], CoinglassResponse) and results[4].success else None,
            "puell_multiple": results[5].data if isinstance(results[5], CoinglassResponse) and results[5].success else None,
            "fear_greed": results[6].data if isinstance(results[6], CoinglassResponse) and results[6].success else None,
            "options_info": results[7].data if isinstance(results[7], CoinglassResponse) and results[7].success else None,
            "options_max_pain": results[8].data if isinstance(results[8], CoinglassResponse) and results[8].success else None,
        }











