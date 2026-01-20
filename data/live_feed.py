"""
BASTION Live Data Feed
======================

Real-time market data using Helsinki VM as primary source.

Helsinki VM Endpoints (Primary):
- /api/price/{symbol}         - Current price
- /api/klines/{symbol}        - OHLCV candles  
- /api/depth/{symbol}         - Order book
- /api/trades/{symbol}        - Recent trades
- /api/ticker/{symbol}        - 24h stats
- /api/ticker24h/{symbol}     - Extended 24h stats

Quant Endpoints (Port 5002):
- /quant/basis/{symbol}       - Funding rate, basis
- /quant/oi/{symbol}          - Open interest
- /quant/liquidations/{symbol} - Liquidation data
- /quant/cvd/{symbol}         - Cumulative volume delta
- /quant/orderflow/{symbol}   - Order flow analysis

Features:
- Auto-polling for price updates
- Session auto-update on new bars
- Multi-symbol support
- Fallback to Binance
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# Helsinki VM endpoints
HELSINKI_SPOT = "http://77.42.29.188:5000"
HELSINKI_QUANT = "http://77.42.29.188:5002"
BINANCE_API = "https://api.binance.com"


class FeedStatus(str, Enum):
    """Status of the live feed."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class PriceUpdate:
    """Real-time price update."""
    symbol: str
    price: float
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    volume_24h: float = 0.0
    change_24h_pct: float = 0.0


@dataclass
class BarUpdate:
    """New bar closed update."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_number: int = 0


@dataclass
class OrderFlowUpdate:
    """Order flow update from Helsinki Quant."""
    symbol: str
    timestamp: datetime
    cvd: float = 0.0                      # Cumulative Volume Delta
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    large_buys: int = 0
    large_sells: int = 0
    imbalance: float = 0.0                # -1 to 1
    funding_rate: float = 0.0
    open_interest: float = 0.0
    oi_change_pct: float = 0.0


@dataclass
class Subscription:
    """Active subscription to a symbol."""
    symbol: str
    timeframes: Set[str] = field(default_factory=set)
    last_price: float = 0.0
    last_bar: Dict[str, datetime] = field(default_factory=dict)
    callbacks: List[Callable] = field(default_factory=list)


class LiveFeed:
    """
    Real-time market data feed using Helsinki VM.
    
    Usage:
        feed = LiveFeed()
        
        # Subscribe to symbol
        await feed.subscribe('BTCUSDT', timeframes=['4h', '1d'])
        
        # Add callback for price updates
        feed.on_price_update(my_callback)
        
        # Add callback for new bars
        feed.on_bar_close(my_bar_callback)
        
        # Start the feed
        await feed.start()
        
        # Get current price
        price = await feed.get_price('BTCUSDT')
        
        # Get latest bars
        bars = await feed.get_bars('BTCUSDT', '4h', limit=100)
        
        # Stop
        await feed.stop()
    """
    
    def __init__(
        self,
        poll_interval: float = 1.0,      # Price polling interval (seconds)
        bar_check_interval: float = 5.0,  # Bar close check interval
        timeout: int = 10,
    ):
        self.poll_interval = poll_interval
        self.bar_check_interval = bar_check_interval
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._subscriptions: Dict[str, Subscription] = {}
        self._running = False
        self._status = FeedStatus.DISCONNECTED
        
        # Callbacks
        self._price_callbacks: List[Callable[[PriceUpdate], Any]] = []
        self._bar_callbacks: List[Callable[[BarUpdate], Any]] = []
        self._orderflow_callbacks: List[Callable[[OrderFlowUpdate], Any]] = []
        
        # Tasks
        self._price_task: Optional[asyncio.Task] = None
        self._bar_task: Optional[asyncio.Task] = None
        
        # Cache
        self._price_cache: Dict[str, PriceUpdate] = {}
        self._bar_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    @property
    def status(self) -> FeedStatus:
        return self._status
    
    @property
    def subscribed_symbols(self) -> List[str]:
        return list(self._subscriptions.keys())
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    # =========================================================================
    # SUBSCRIPTION MANAGEMENT
    # =========================================================================
    
    async def subscribe(
        self,
        symbol: str,
        timeframes: List[str] = None,
        callback: Callable = None,
    ):
        """Subscribe to a symbol for updates."""
        if symbol not in self._subscriptions:
            self._subscriptions[symbol] = Subscription(symbol=symbol)
        
        sub = self._subscriptions[symbol]
        
        if timeframes:
            sub.timeframes.update(timeframes)
        
        if callback:
            sub.callbacks.append(callback)
        
        # Initialize cache
        if symbol not in self._bar_cache:
            self._bar_cache[symbol] = {}
        
        logger.info(f"Subscribed to {symbol} (timeframes: {sub.timeframes})")
    
    async def unsubscribe(self, symbol: str):
        """Unsubscribe from a symbol."""
        if symbol in self._subscriptions:
            del self._subscriptions[symbol]
            logger.info(f"Unsubscribed from {symbol}")
    
    def on_price_update(self, callback: Callable[[PriceUpdate], Any]):
        """Register callback for price updates."""
        self._price_callbacks.append(callback)
    
    def on_bar_close(self, callback: Callable[[BarUpdate], Any]):
        """Register callback for bar close events."""
        self._bar_callbacks.append(callback)
    
    def on_orderflow_update(self, callback: Callable[[OrderFlowUpdate], Any]):
        """Register callback for order flow updates."""
        self._orderflow_callbacks.append(callback)
    
    # =========================================================================
    # FEED CONTROL
    # =========================================================================
    
    async def start(self):
        """Start the live feed."""
        if self._running:
            return
        
        self._running = True
        self._status = FeedStatus.CONNECTING
        
        # Test connection
        try:
            session = await self._get_session()
            async with session.get(f"{HELSINKI_SPOT}/api/ping") as resp:
                if resp.status == 200:
                    self._status = FeedStatus.CONNECTED
                    logger.info("Connected to Helsinki VM")
        except Exception as e:
            logger.warning(f"Helsinki VM ping failed, will use fallback: {e}")
            self._status = FeedStatus.CONNECTED  # Still run with fallback
        
        # Start polling tasks
        self._price_task = asyncio.create_task(self._price_poll_loop())
        self._bar_task = asyncio.create_task(self._bar_check_loop())
        
        logger.info("Live feed started")
    
    async def stop(self):
        """Stop the live feed."""
        self._running = False
        
        if self._price_task:
            self._price_task.cancel()
        if self._bar_task:
            self._bar_task.cancel()
        
        if self._session and not self._session.closed:
            await self._session.close()
        
        self._status = FeedStatus.DISCONNECTED
        logger.info("Live feed stopped")
    
    # =========================================================================
    # PRICE DATA
    # =========================================================================
    
    async def get_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        # Check cache first
        if symbol in self._price_cache:
            cache_age = (datetime.utcnow() - self._price_cache[symbol].timestamp).total_seconds()
            if cache_age < self.poll_interval * 2:
                return self._price_cache[symbol].price
        
        # Fetch fresh
        update = await self._fetch_price(symbol)
        if update:
            return update.price
        return 0.0
    
    async def get_price_update(self, symbol: str) -> Optional[PriceUpdate]:
        """Get full price update with bid/ask/volume."""
        return await self._fetch_price(symbol)
    
    async def _fetch_price(self, symbol: str) -> Optional[PriceUpdate]:
        """Fetch price from Helsinki VM."""
        session = await self._get_session()
        
        # Try Helsinki VM
        try:
            url = f"{HELSINKI_SPOT}/api/ticker/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    update = PriceUpdate(
                        symbol=symbol,
                        price=float(data.get('lastPrice', data.get('price', 0))),
                        timestamp=datetime.utcnow(),
                        bid=float(data.get('bidPrice', 0)),
                        ask=float(data.get('askPrice', 0)),
                        volume_24h=float(data.get('volume', data.get('quoteVolume', 0))),
                        change_24h_pct=float(data.get('priceChangePercent', 0)),
                    )
                    self._price_cache[symbol] = update
                    return update
        except Exception as e:
            logger.debug(f"Helsinki ticker failed: {e}")
        
        # Try simple price endpoint
        try:
            url = f"{HELSINKI_SPOT}/api/price/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    update = PriceUpdate(
                        symbol=symbol,
                        price=float(data.get('price', data.get('last', 0))),
                        timestamp=datetime.utcnow(),
                    )
                    self._price_cache[symbol] = update
                    return update
        except Exception as e:
            logger.debug(f"Helsinki price failed: {e}")
        
        # Fallback to Binance
        try:
            url = f"{BINANCE_API}/api/v3/ticker/24hr"
            params = {"symbol": symbol}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    update = PriceUpdate(
                        symbol=symbol,
                        price=float(data['lastPrice']),
                        timestamp=datetime.utcnow(),
                        bid=float(data.get('bidPrice', 0)),
                        ask=float(data.get('askPrice', 0)),
                        volume_24h=float(data.get('quoteVolume', 0)),
                        change_24h_pct=float(data.get('priceChangePercent', 0)),
                    )
                    self._price_cache[symbol] = update
                    return update
        except Exception as e:
            logger.error(f"All price sources failed for {symbol}: {e}")
        
        return None
    
    async def _price_poll_loop(self):
        """Background task to poll prices."""
        while self._running:
            try:
                for symbol in list(self._subscriptions.keys()):
                    update = await self._fetch_price(symbol)
                    
                    if update:
                        # Update subscription
                        self._subscriptions[symbol].last_price = update.price
                        
                        # Trigger callbacks
                        for callback in self._price_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(update)
                                else:
                                    callback(update)
                            except Exception as e:
                                logger.error(f"Price callback error: {e}")
                        
                        # Symbol-specific callbacks
                        for callback in self._subscriptions[symbol].callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(update)
                                else:
                                    callback(update)
                            except Exception as e:
                                logger.error(f"Symbol callback error: {e}")
                
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Price poll error: {e}")
                await asyncio.sleep(self.poll_interval)
    
    # =========================================================================
    # BAR/CANDLE DATA
    # =========================================================================
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Get OHLCV bars for symbol/timeframe."""
        return await self._fetch_bars(symbol, timeframe, limit)
    
    async def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[BarUpdate]:
        """Get the most recent closed bar."""
        df = await self._fetch_bars(symbol, timeframe, limit=2)
        if df.empty:
            return None
        
        # Second-to-last row is the last closed bar
        if len(df) >= 2:
            row = df.iloc[-2]
        else:
            row = df.iloc[-1]
        
        return BarUpdate(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=row.name if isinstance(row.name, datetime) else datetime.utcnow(),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
        )
    
    async def _fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Helsinki VM."""
        session = await self._get_session()
        
        # Try Helsinki VM
        try:
            url = f"{HELSINKI_SPOT}/api/klines/{symbol}"
            params = {"interval": timeframe, "limit": limit}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    df = self._parse_klines(data)
                    self._bar_cache.setdefault(symbol, {})[timeframe] = df
                    return df
        except Exception as e:
            logger.debug(f"Helsinki klines failed: {e}")
        
        # Fallback to Binance
        try:
            url = f"{BINANCE_API}/api/v3/klines"
            params = {"symbol": symbol, "interval": timeframe, "limit": limit}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    df = self._parse_klines(data)
                    self._bar_cache.setdefault(symbol, {})[timeframe] = df
                    return df
        except Exception as e:
            logger.error(f"All bar sources failed: {e}")
        
        return pd.DataFrame()
    
    async def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = ["15m", "1h", "4h", "1d"],
        limit: int = 200,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple timeframes for MTF analysis."""
        tasks = [self._fetch_bars(symbol, tf, limit) for tf in timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {tf}: {result}")
                continue
            if not result.empty:
                data[tf] = result
        
        return data
    
    async def _bar_check_loop(self):
        """Background task to detect new bar closes."""
        while self._running:
            try:
                for symbol, sub in list(self._subscriptions.items()):
                    for tf in sub.timeframes:
                        # Get latest bars
                        df = await self._fetch_bars(symbol, tf, limit=2)
                        if df.empty:
                            continue
                        
                        # Get last closed bar timestamp
                        last_bar_ts = df.index[-2] if len(df) >= 2 else df.index[-1]
                        
                        # Check if new bar
                        prev_ts = sub.last_bar.get(tf)
                        if prev_ts is None or last_bar_ts > prev_ts:
                            sub.last_bar[tf] = last_bar_ts
                            
                            if prev_ts is not None:  # Not first check
                                # New bar closed!
                                row = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
                                bar_update = BarUpdate(
                                    symbol=symbol,
                                    timeframe=tf,
                                    timestamp=last_bar_ts,
                                    open=float(row['open']),
                                    high=float(row['high']),
                                    low=float(row['low']),
                                    close=float(row['close']),
                                    volume=float(row['volume']),
                                )
                                
                                logger.info(f"New bar closed: {symbol} {tf} @ {last_bar_ts}")
                                
                                # Trigger callbacks
                                for callback in self._bar_callbacks:
                                    try:
                                        if asyncio.iscoroutinefunction(callback):
                                            await callback(bar_update)
                                        else:
                                            callback(bar_update)
                                    except Exception as e:
                                        logger.error(f"Bar callback error: {e}")
                
                await asyncio.sleep(self.bar_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bar check error: {e}")
                await asyncio.sleep(self.bar_check_interval)
    
    # =========================================================================
    # ORDER FLOW DATA (Helsinki Quant)
    # =========================================================================
    
    async def get_orderflow(self, symbol: str) -> Optional[OrderFlowUpdate]:
        """Get order flow data from Helsinki Quant."""
        session = await self._get_session()
        
        update = OrderFlowUpdate(symbol=symbol, timestamp=datetime.utcnow())
        
        # CVD
        try:
            url = f"{HELSINKI_QUANT}/quant/cvd/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    update.cvd = float(data.get('cvd', 0))
                    update.buy_volume = float(data.get('buy_volume', 0))
                    update.sell_volume = float(data.get('sell_volume', 0))
        except Exception as e:
            logger.debug(f"CVD fetch failed: {e}")
        
        # Funding & OI
        try:
            url = f"{HELSINKI_QUANT}/quant/basis/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    update.funding_rate = float(data.get('funding_rate_pct', 0)) / 100
        except Exception as e:
            logger.debug(f"Basis fetch failed: {e}")
        
        try:
            url = f"{HELSINKI_QUANT}/quant/oi/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    update.open_interest = float(data.get('open_interest', 0))
                    update.oi_change_pct = float(data.get('oi_change_pct', 0))
        except Exception as e:
            logger.debug(f"OI fetch failed: {e}")
        
        # Order flow analysis
        try:
            url = f"{HELSINKI_QUANT}/quant/orderflow/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    update.large_buys = int(data.get('large_buys', 0))
                    update.large_sells = int(data.get('large_sells', 0))
                    update.imbalance = float(data.get('imbalance', 0))
        except Exception as e:
            logger.debug(f"Orderflow fetch failed: {e}")
        
        return update
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book depth."""
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
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        session = await self._get_session()
        
        try:
            url = f"{HELSINKI_SPOT}/api/trades/{symbol}"
            params = {"limit": limit}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        
        try:
            url = f"{BINANCE_API}/api/v3/trades"
            params = {"symbol": symbol, "limit": limit}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        
        return []
    
    async def get_liquidations(self, symbol: str) -> Dict[str, Any]:
        """Get liquidation data from Helsinki Quant."""
        session = await self._get_session()
        
        try:
            url = f"{HELSINKI_QUANT}/quant/liquidations/{symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        
        return {}
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _parse_klines(self, data: List) -> pd.DataFrame:
        """Parse Binance klines format to DataFrame."""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_cached_price(self, symbol: str) -> Optional[PriceUpdate]:
        """Get cached price (no network call)."""
        return self._price_cache.get(symbol)
    
    def get_cached_bars(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get cached bars (no network call)."""
        return self._bar_cache.get(symbol, {}).get(timeframe)


# =============================================================================
# SESSION AUTO-UPDATER
# =============================================================================

class SessionAutoUpdater:
    """
    Automatically updates trading sessions with live price data.
    
    Usage:
        from core.session import SessionManager
        
        feed = LiveFeed()
        session_manager = SessionManager()
        
        auto_updater = SessionAutoUpdater(feed, session_manager)
        await auto_updater.start()
    """
    
    def __init__(
        self,
        feed: LiveFeed,
        session_manager: 'SessionManager',
    ):
        self.feed = feed
        self.session_manager = session_manager
        self._bar_counts: Dict[str, int] = {}  # session_id -> bar count
    
    async def start(self):
        """Start auto-updating sessions."""
        # Register callbacks
        self.feed.on_price_update(self._on_price_update)
        self.feed.on_bar_close(self._on_bar_close)
        
        # Subscribe to symbols from active sessions
        for session in self.session_manager.get_active_sessions():
            await self.feed.subscribe(
                session.symbol,
                timeframes=[session.timeframe],
            )
        
        logger.info("Session auto-updater started")
    
    async def _on_price_update(self, update: PriceUpdate):
        """Handle price update - update session current price."""
        sessions = self.session_manager.get_active_sessions(symbol=update.symbol)
        
        for session in sessions:
            # Just update current price for P&L tracking
            # Full update happens on bar close
            session.current_price = update.price
            
            # Recalculate unrealized P&L
            if session.remaining_size > 0:
                if session.direction == "long":
                    session.unrealized_pnl = (update.price - session.average_entry) * session.remaining_size
                else:
                    session.unrealized_pnl = (session.average_entry - update.price) * session.remaining_size
                
                session.unrealized_pnl_pct = (session.unrealized_pnl / session.account_balance) * 100
    
    async def _on_bar_close(self, update: BarUpdate):
        """Handle bar close - full session update."""
        sessions = self.session_manager.get_active_sessions(symbol=update.symbol)
        
        for session in sessions:
            if session.timeframe != update.timeframe:
                continue
            
            # Increment bar count
            session_id = session.id
            if session_id not in self._bar_counts:
                self._bar_counts[session_id] = session.bars_in_trade
            self._bar_counts[session_id] += 1
            
            # Get recent swing data for guarding line
            bars = await self.feed.get_bars(update.symbol, update.timeframe, limit=20)
            recent_lows = bars['low'].tolist() if not bars.empty else None
            recent_highs = bars['high'].tolist() if not bars.empty else None
            
            # Update session
            result = self.session_manager.update_session(
                session_id=session_id,
                current_price=update.close,
                current_bar=self._bar_counts[session_id],
                recent_lows=recent_lows,
                recent_highs=recent_highs,
            )
            
            # Log alerts
            for alert in result.alerts:
                logger.info(f"Session {session_id}: {alert}")
            
            # Handle exit signals
            if result.exit_signal:
                logger.warning(
                    f"Session {session_id} EXIT SIGNAL: {result.exit_reason} "
                    f"- Exit {result.exit_percentage}% at ${update.close:,.2f}"
                )
                # Note: Actual exit execution should be handled by trading system
                # This just notifies. You can auto-execute here if desired.


# Convenience function
async def create_live_session_tracker(session_manager: 'SessionManager') -> tuple:
    """
    Create and start a live feed with session auto-updater.
    
    Returns:
        (LiveFeed, SessionAutoUpdater)
    """
    feed = LiveFeed()
    updater = SessionAutoUpdater(feed, session_manager)
    
    await feed.start()
    await updater.start()
    
    return feed, updater

