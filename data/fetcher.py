"""
Live Data Fetcher
=================

Fetches real-time OHLCV data from Binance for MCF analysis.
Uses the Helsinki VM proxy for reliability.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# Helsinki VM endpoints
HELSINKI_SPOT = "http://77.42.29.188:5000"
HELSINKI_QUANT = "http://77.42.29.188:5002"

# Binance direct (fallback)
BINANCE_API = "https://api.binance.com"


class LiveDataFetcher:
    """
    Fetches live market data for MCF analysis.
    
    Primary: Helsinki VM proxy
    Fallback: Direct Binance API
    """
    
    def __init__(self, timeout: int = 15):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_ohlcv(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for MCF analysis.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1000)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
        """
        session = await self._get_session()
        
        # Try Helsinki VM first
        try:
            url = f"{HELSINKI_SPOT}/api/klines/{symbol}"
            params = {"interval": interval, "limit": limit}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_klines(data)
        except Exception as e:
            logger.warning(f"Helsinki VM failed, falling back to Binance: {e}")
        
        # Fallback to Binance
        try:
            url = f"{BINANCE_API}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_klines(data)
        except Exception as e:
            logger.error(f"Binance also failed: {e}")
            raise
        
        raise RuntimeError(f"Could not fetch OHLCV for {symbol}")
    
    async def get_multi_timeframe(
        self,
        symbol: str = "BTCUSDT",
        timeframes: List[str] = ["15m", "1h", "4h", "1d"],
        limit: int = 200,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple timeframes for MTF analysis.
        
        Returns:
            Dict mapping timeframe -> DataFrame
        """
        tasks = [
            self.get_ohlcv(symbol, tf, limit)
            for tf in timeframes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {tf}: {result}")
                continue
            data[tf] = result
        
        return data
    
    async def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Get current price for symbol."""
        session = await self._get_session()
        
        try:
            url = f"{HELSINKI_SPOT}/api/price/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("price", data.get("last", 0)))
        except Exception:
            pass
        
        # Fallback
        try:
            url = f"{BINANCE_API}/api/v3/ticker/price"
            params = {"symbol": symbol}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data["price"])
        except Exception:
            pass
        
        return 0.0
    
    async def get_orderbook(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Get orderbook for volume/imbalance analysis."""
        session = await self._get_session()
        
        try:
            url = f"{HELSINKI_SPOT}/api/depth/{symbol}"
            params = {"limit": limit}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        
        try:
            url = f"{BINANCE_API}/api/v3/depth"
            params = {"symbol": symbol, "limit": limit}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        
        return {"bids": [], "asks": []}
    
    async def get_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        """Get current funding rate for derivatives."""
        session = await self._get_session()
        
        try:
            url = f"{HELSINKI_QUANT}/quant/basis/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("funding_rate_pct", 0)) / 100
        except Exception:
            pass
        
        return 0.0
    
    def _parse_klines(self, data: List) -> pd.DataFrame:
        """Parse Binance klines format to DataFrame."""
        if not data:
            return pd.DataFrame()
        
        # Binance klines: [timestamp, o, h, l, c, vol, close_time, ...]
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        
        return df


# Sync wrapper for non-async contexts
def fetch_ohlcv_sync(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """Synchronous wrapper for get_ohlcv."""
    async def _fetch():
        fetcher = LiveDataFetcher()
        try:
            return await fetcher.get_ohlcv(symbol, interval, limit)
        finally:
            await fetcher.close()
    
    return asyncio.run(_fetch())


def fetch_multi_tf_sync(
    symbol: str = "BTCUSDT",
    timeframes: List[str] = ["15m", "1h", "4h", "1d"],
    limit: int = 200,
) -> Dict[str, pd.DataFrame]:
    """Synchronous wrapper for get_multi_timeframe."""
    async def _fetch():
        fetcher = LiveDataFetcher()
        try:
            return await fetcher.get_multi_timeframe(symbol, timeframes, limit)
        finally:
            await fetcher.close()
    
    return asyncio.run(_fetch())











