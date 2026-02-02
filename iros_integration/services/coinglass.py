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
    # OPTIONS DATA
    # =========================================================================
    
    async def get_options_info(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get options overview - put/call ratio, max pain"""
        return await self._request(
            "/options/info",
            {"symbol": symbol}
        )
    
    # =========================================================================
    # ETF DATA
    # =========================================================================
    
    async def get_bitcoin_etf(self) -> CoinglassResponse:
        """Get Bitcoin ETF flows data"""
        return await self._request("/index/bitcoin-etf")
    
    async def get_gbtc(self) -> CoinglassResponse:
        """Get GBTC holdings"""
        return await self._request("/index/gbtc")
    
    # =========================================================================
    # TAKER BUY/SELL
    # =========================================================================
    
    async def get_taker_buy_sell(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get taker buy/sell ratio - real-time order flow"""
        return await self._request(
            "/futures/taker-buy-sell-ratio",
            {"symbol": symbol}
        )
    
    # =========================================================================
    # LIQUIDATION BY EXCHANGE
    # =========================================================================
    
    async def get_liquidation_by_exchange(self, symbol: str = "BTC") -> CoinglassResponse:
        """Get liquidations broken down by exchange"""
        return await self._request(
            "/futures/liquidation/exchange-list",
            {"symbol": symbol}
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







