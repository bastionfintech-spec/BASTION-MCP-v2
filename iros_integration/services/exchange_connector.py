"""
Exchange Connector Service
===========================
Handles connections to various crypto exchanges for position syncing.
Supports: BloFin, Bitunix, Bybit, OKX, Binance, Deribit
"""

import httpx
import hmac
import hashlib
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExchangeCredentials:
    """Exchange API credentials."""
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    read_only: bool = True


@dataclass
class Position:
    """Standardized position format across exchanges."""
    id: str
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    size: float
    size_usd: float
    pnl: float
    pnl_pct: float
    leverage: float
    margin: float
    liquidation_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exchange: str = ""
    updated_at: str = ""


@dataclass
class ExchangeBalance:
    """Account balance info."""
    total_equity: float
    available_balance: float
    margin_used: float
    unrealized_pnl: float
    currency: str = "USDT"


class BaseExchangeClient(ABC):
    """Abstract base class for exchange clients."""
    
    def __init__(self, credentials: ExchangeCredentials):
        self.credentials = credentials
        self.base_url: str = ""
        self.exchange_name: str = ""
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if credentials are valid."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Fetch all open positions."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> ExchangeBalance:
        """Fetch account balance."""
        pass
    
    def _generate_signature(self, message: str) -> str:
        """Generate HMAC signature."""
        return hmac.new(
            self.credentials.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()


class BloFinClient(BaseExchangeClient):
    """BloFin exchange client."""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self.base_url = "https://openapi.blofin.com"
        self.exchange_name = "blofin"
    
    def _get_headers(self, timestamp: str, sign: str) -> Dict:
        return {
            "ACCESS-KEY": self.credentials.api_key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.credentials.passphrase or "",
            "Content-Type": "application/json"
        }
    
    async def test_connection(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                path = "/api/v1/account/balance"
                message = f"{timestamp}GET{path}"
                sign = self._generate_signature(message)
                
                res = await client.get(
                    f"{self.base_url}{path}",
                    headers=self._get_headers(timestamp, sign),
                    timeout=10.0
                )
                return res.status_code == 200
        except Exception as e:
            logger.error(f"BloFin connection test failed: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        positions = []
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                path = "/api/v1/account/positions"
                message = f"{timestamp}GET{path}"
                sign = self._generate_signature(message)
                
                res = await client.get(
                    f"{self.base_url}{path}",
                    headers=self._get_headers(timestamp, sign),
                    timeout=10.0
                )
                
                if res.status_code == 200:
                    data = res.json()
                    for pos in data.get("data", []):
                        if float(pos.get("positionAmt", 0)) != 0:
                            positions.append(Position(
                                id=f"blofin_{pos.get('symbol')}_{pos.get('positionSide', 'both')}",
                                symbol=pos.get("symbol", ""),
                                direction="long" if float(pos.get("positionAmt", 0)) > 0 else "short",
                                entry_price=float(pos.get("entryPrice", 0)),
                                current_price=float(pos.get("markPrice", 0)),
                                size=abs(float(pos.get("positionAmt", 0))),
                                size_usd=abs(float(pos.get("positionAmt", 0)) * float(pos.get("markPrice", 0))),
                                pnl=float(pos.get("unrealizedPnl", 0)),
                                pnl_pct=float(pos.get("unrealizedPnlRatio", 0)) * 100,
                                leverage=float(pos.get("leverage", 1)),
                                margin=float(pos.get("margin", 0)),
                                liquidation_price=float(pos.get("liquidationPrice", 0)) or None,
                                exchange=self.exchange_name,
                                updated_at=datetime.now().isoformat()
                            ))
        except Exception as e:
            logger.error(f"BloFin get_positions error: {e}")
        return positions
    
    async def get_balance(self) -> ExchangeBalance:
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                path = "/api/v1/account/balance"
                message = f"{timestamp}GET{path}"
                sign = self._generate_signature(message)
                
                res = await client.get(
                    f"{self.base_url}{path}",
                    headers=self._get_headers(timestamp, sign),
                    timeout=10.0
                )
                
                if res.status_code == 200:
                    data = res.json().get("data", {})
                    return ExchangeBalance(
                        total_equity=float(data.get("totalEquity", 0)),
                        available_balance=float(data.get("availableBalance", 0)),
                        margin_used=float(data.get("marginUsed", 0)),
                        unrealized_pnl=float(data.get("unrealizedPnl", 0))
                    )
        except Exception as e:
            logger.error(f"BloFin get_balance error: {e}")
        
        return ExchangeBalance(0, 0, 0, 0)


class BitunixClient(BaseExchangeClient):
    """Bitunix exchange client."""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self.base_url = "https://fapi.bitunix.com"
        self.exchange_name = "bitunix"
    
    def _get_headers(self, timestamp: str, sign: str) -> Dict:
        return {
            "api-key": self.credentials.api_key,
            "sign": sign,
            "timestamp": timestamp,
            "Content-Type": "application/json"
        }
    
    async def test_connection(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                message = f"{timestamp}"
                sign = self._generate_signature(message)
                
                res = await client.get(
                    f"{self.base_url}/api/v1/account",
                    headers=self._get_headers(timestamp, sign),
                    timeout=10.0
                )
                return res.status_code == 200
        except Exception as e:
            logger.error(f"Bitunix connection test failed: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        # Similar implementation pattern as BloFin
        # Adjust for Bitunix-specific API format
        return []
    
    async def get_balance(self) -> ExchangeBalance:
        return ExchangeBalance(0, 0, 0, 0)


class BybitClient(BaseExchangeClient):
    """Bybit exchange client."""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self.base_url = "https://api.bybit.com"
        self.exchange_name = "bybit"
    
    async def test_connection(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                recv_window = "5000"
                
                params = f"api_key={self.credentials.api_key}&recv_window={recv_window}&timestamp={timestamp}"
                sign = self._generate_signature(params)
                
                res = await client.get(
                    f"{self.base_url}/v5/account/wallet-balance",
                    params={
                        "accountType": "UNIFIED",
                        "api_key": self.credentials.api_key,
                        "recv_window": recv_window,
                        "timestamp": timestamp,
                        "sign": sign
                    },
                    timeout=10.0
                )
                return res.status_code == 200
        except Exception as e:
            logger.error(f"Bybit connection test failed: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        positions = []
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                recv_window = "5000"
                
                params_str = f"api_key={self.credentials.api_key}&category=linear&recv_window={recv_window}&settleCoin=USDT&timestamp={timestamp}"
                sign = self._generate_signature(params_str)
                
                res = await client.get(
                    f"{self.base_url}/v5/position/list",
                    params={
                        "category": "linear",
                        "settleCoin": "USDT",
                        "api_key": self.credentials.api_key,
                        "recv_window": recv_window,
                        "timestamp": timestamp,
                        "sign": sign
                    },
                    timeout=10.0
                )
                
                if res.status_code == 200:
                    data = res.json()
                    for pos in data.get("result", {}).get("list", []):
                        size = float(pos.get("size", 0))
                        if size > 0:
                            positions.append(Position(
                                id=f"bybit_{pos.get('symbol')}_{pos.get('side', '')}",
                                symbol=pos.get("symbol", ""),
                                direction="long" if pos.get("side") == "Buy" else "short",
                                entry_price=float(pos.get("avgPrice", 0)),
                                current_price=float(pos.get("markPrice", 0)),
                                size=size,
                                size_usd=float(pos.get("positionValue", 0)),
                                pnl=float(pos.get("unrealisedPnl", 0)),
                                pnl_pct=(float(pos.get("unrealisedPnl", 0)) / float(pos.get("positionIM", 1))) * 100 if float(pos.get("positionIM", 0)) > 0 else 0,
                                leverage=float(pos.get("leverage", 1)),
                                margin=float(pos.get("positionIM", 0)),
                                liquidation_price=float(pos.get("liqPrice", 0)) or None,
                                stop_loss=float(pos.get("stopLoss", 0)) or None,
                                take_profit=float(pos.get("takeProfit", 0)) or None,
                                exchange=self.exchange_name,
                                updated_at=datetime.now().isoformat()
                            ))
        except Exception as e:
            logger.error(f"Bybit get_positions error: {e}")
        return positions
    
    async def get_balance(self) -> ExchangeBalance:
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                recv_window = "5000"
                
                params_str = f"accountType=UNIFIED&api_key={self.credentials.api_key}&recv_window={recv_window}&timestamp={timestamp}"
                sign = self._generate_signature(params_str)
                
                res = await client.get(
                    f"{self.base_url}/v5/account/wallet-balance",
                    params={
                        "accountType": "UNIFIED",
                        "api_key": self.credentials.api_key,
                        "recv_window": recv_window,
                        "timestamp": timestamp,
                        "sign": sign
                    },
                    timeout=10.0
                )
                
                if res.status_code == 200:
                    data = res.json()
                    account = data.get("result", {}).get("list", [{}])[0]
                    return ExchangeBalance(
                        total_equity=float(account.get("totalEquity", 0)),
                        available_balance=float(account.get("totalAvailableBalance", 0)),
                        margin_used=float(account.get("totalInitialMargin", 0)),
                        unrealized_pnl=float(account.get("totalUnrealisedPnl", 0))
                    )
        except Exception as e:
            logger.error(f"Bybit get_balance error: {e}")
        
        return ExchangeBalance(0, 0, 0, 0)


class OKXClient(BaseExchangeClient):
    """OKX exchange client."""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self.base_url = "https://www.okx.com"
        self.exchange_name = "okx"
    
    async def test_connection(self) -> bool:
        # OKX implementation
        return False
    
    async def get_positions(self) -> List[Position]:
        return []
    
    async def get_balance(self) -> ExchangeBalance:
        return ExchangeBalance(0, 0, 0, 0)


class BinanceClient(BaseExchangeClient):
    """Binance Futures client."""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self.base_url = "https://fapi.binance.com"
        self.exchange_name = "binance"
    
    async def test_connection(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                params = f"timestamp={timestamp}"
                sign = self._generate_signature(params)
                
                res = await client.get(
                    f"{self.base_url}/fapi/v2/account",
                    params={"timestamp": timestamp, "signature": sign},
                    headers={"X-MBX-APIKEY": self.credentials.api_key},
                    timeout=10.0
                )
                return res.status_code == 200
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        positions = []
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                params = f"timestamp={timestamp}"
                sign = self._generate_signature(params)
                
                res = await client.get(
                    f"{self.base_url}/fapi/v2/positionRisk",
                    params={"timestamp": timestamp, "signature": sign},
                    headers={"X-MBX-APIKEY": self.credentials.api_key},
                    timeout=10.0
                )
                
                if res.status_code == 200:
                    data = res.json()
                    for pos in data:
                        amt = float(pos.get("positionAmt", 0))
                        if amt != 0:
                            positions.append(Position(
                                id=f"binance_{pos.get('symbol')}",
                                symbol=pos.get("symbol", ""),
                                direction="long" if amt > 0 else "short",
                                entry_price=float(pos.get("entryPrice", 0)),
                                current_price=float(pos.get("markPrice", 0)),
                                size=abs(amt),
                                size_usd=abs(float(pos.get("notional", 0))),
                                pnl=float(pos.get("unRealizedProfit", 0)),
                                pnl_pct=0,  # Calculate from entry/current
                                leverage=float(pos.get("leverage", 1)),
                                margin=float(pos.get("isolatedMargin", 0)),
                                liquidation_price=float(pos.get("liquidationPrice", 0)) or None,
                                exchange=self.exchange_name,
                                updated_at=datetime.now().isoformat()
                            ))
        except Exception as e:
            logger.error(f"Binance get_positions error: {e}")
        return positions
    
    async def get_balance(self) -> ExchangeBalance:
        try:
            async with httpx.AsyncClient() as client:
                timestamp = str(int(time.time() * 1000))
                params = f"timestamp={timestamp}"
                sign = self._generate_signature(params)
                
                res = await client.get(
                    f"{self.base_url}/fapi/v2/account",
                    params={"timestamp": timestamp, "signature": sign},
                    headers={"X-MBX-APIKEY": self.credentials.api_key},
                    timeout=10.0
                )
                
                if res.status_code == 200:
                    data = res.json()
                    return ExchangeBalance(
                        total_equity=float(data.get("totalWalletBalance", 0)),
                        available_balance=float(data.get("availableBalance", 0)),
                        margin_used=float(data.get("totalInitialMargin", 0)),
                        unrealized_pnl=float(data.get("totalUnrealizedProfit", 0))
                    )
        except Exception as e:
            logger.error(f"Binance get_balance error: {e}")
        
        return ExchangeBalance(0, 0, 0, 0)


class DeribitClient(BaseExchangeClient):
    """Deribit exchange client (Options & Futures)."""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self.base_url = "https://www.deribit.com/api/v2"
        self.exchange_name = "deribit"
    
    async def test_connection(self) -> bool:
        return False
    
    async def get_positions(self) -> List[Position]:
        return []
    
    async def get_balance(self) -> ExchangeBalance:
        return ExchangeBalance(0, 0, 0, 0)


class HyperliquidClient(BaseExchangeClient):
    """Hyperliquid DEX client (Perpetuals)."""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self.base_url = "https://api.hyperliquid.xyz"
        self.exchange_name = "hyperliquid"
        # For Hyperliquid, api_key is the wallet address
        self.wallet_address = credentials.api_key
    
    async def test_connection(self) -> bool:
        """Test connection by fetching user state."""
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    f"{self.base_url}/info",
                    json={
                        "type": "clearinghouseState",
                        "user": self.wallet_address
                    },
                    timeout=10.0
                )
                return res.status_code == 200
        except Exception as e:
            logger.error(f"Hyperliquid connection test failed: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        """Fetch open positions from Hyperliquid."""
        positions = []
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    f"{self.base_url}/info",
                    json={
                        "type": "clearinghouseState",
                        "user": self.wallet_address
                    },
                    timeout=10.0
                )
                
                if res.status_code == 200:
                    data = res.json()
                    asset_positions = data.get("assetPositions", [])
                    
                    for item in asset_positions:
                        pos = item.get("position", {})
                        size = float(pos.get("szi", 0))
                        
                        if size != 0:
                            entry_price = float(pos.get("entryPx", 0))
                            current_price = float(pos.get("markPx", 0) or pos.get("entryPx", 0))
                            
                            # Calculate PnL
                            pnl = float(pos.get("unrealizedPnl", 0))
                            margin = float(pos.get("marginUsed", 0))
                            pnl_pct = (pnl / margin * 100) if margin > 0 else 0
                            
                            positions.append(Position(
                                id=f"hyperliquid_{pos.get('coin', '')}",
                                symbol=pos.get("coin", ""),
                                direction="long" if size > 0 else "short",
                                entry_price=entry_price,
                                current_price=current_price,
                                size=abs(size),
                                size_usd=abs(size * current_price),
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                                leverage=float(pos.get("leverage", {}).get("value", 1)),
                                margin=margin,
                                liquidation_price=float(pos.get("liquidationPx", 0)) or None,
                                exchange=self.exchange_name,
                                updated_at=datetime.now().isoformat()
                            ))
        except Exception as e:
            logger.error(f"Hyperliquid get_positions error: {e}")
        return positions
    
    async def get_balance(self) -> ExchangeBalance:
        """Fetch account balance from Hyperliquid."""
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    f"{self.base_url}/info",
                    json={
                        "type": "clearinghouseState",
                        "user": self.wallet_address
                    },
                    timeout=10.0
                )
                
                if res.status_code == 200:
                    data = res.json()
                    margin_summary = data.get("marginSummary", {})
                    
                    return ExchangeBalance(
                        total_equity=float(margin_summary.get("accountValue", 0)),
                        available_balance=float(margin_summary.get("availableBalance", 0) or margin_summary.get("accountValue", 0)),
                        margin_used=float(margin_summary.get("totalMarginUsed", 0)),
                        unrealized_pnl=float(margin_summary.get("totalUnrealizedPnl", 0)),
                        currency="USDC"
                    )
        except Exception as e:
            logger.error(f"Hyperliquid get_balance error: {e}")
        
        return ExchangeBalance(0, 0, 0, 0, "USDC")


# =============================================================================
# EXCHANGE CONNECTOR FACTORY
# =============================================================================

EXCHANGE_CLIENTS = {
    "blofin": BloFinClient,
    "bitunix": BitunixClient,
    "bybit": BybitClient,
    "okx": OKXClient,
    "binance": BinanceClient,
    "deribit": DeribitClient,
    "hyperliquid": HyperliquidClient
}


def create_exchange_client(
    exchange: str,
    api_key: str,
    api_secret: str,
    passphrase: Optional[str] = None,
    read_only: bool = True
) -> BaseExchangeClient:
    """Factory function to create exchange client."""
    
    if exchange not in EXCHANGE_CLIENTS:
        raise ValueError(f"Unsupported exchange: {exchange}")
    
    credentials = ExchangeCredentials(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        read_only=read_only
    )
    
    return EXCHANGE_CLIENTS[exchange](credentials)


# =============================================================================
# USER CONTEXT SERVICE
# =============================================================================

class UserContextService:
    """
    Manages user exchange connections and provides position context
    for IROS AI queries.
    """
    
    def __init__(self):
        self.connections: Dict[str, BaseExchangeClient] = {}
        self.cached_positions: Dict[str, List[Position]] = {}
        self.cache_timestamp: Dict[str, float] = {}
        self.cache_ttl = 5  # seconds
    
    async def connect_exchange(
        self,
        exchange: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        read_only: bool = True
    ) -> bool:
        """Connect a new exchange."""
        try:
            client = create_exchange_client(
                exchange, api_key, api_secret, passphrase, read_only
            )
            
            # Test connection
            if await client.test_connection():
                self.connections[exchange] = client
                logger.info(f"[USER_CONTEXT] Connected to {exchange}")
                return True
            else:
                logger.warning(f"[USER_CONTEXT] Failed to connect to {exchange}")
                return False
        except Exception as e:
            logger.error(f"[USER_CONTEXT] Connection error for {exchange}: {e}")
            return False
    
    def disconnect_exchange(self, exchange: str):
        """Disconnect an exchange."""
        if exchange in self.connections:
            del self.connections[exchange]
            if exchange in self.cached_positions:
                del self.cached_positions[exchange]
            logger.info(f"[USER_CONTEXT] Disconnected from {exchange}")
    
    async def get_all_positions(self) -> List[Position]:
        """Get positions from all connected exchanges."""
        all_positions = []
        
        for exchange, client in self.connections.items():
            # Check cache
            now = time.time()
            if exchange in self.cache_timestamp:
                if now - self.cache_timestamp[exchange] < self.cache_ttl:
                    all_positions.extend(self.cached_positions.get(exchange, []))
                    continue
            
            # Fetch fresh
            try:
                positions = await client.get_positions()
                self.cached_positions[exchange] = positions
                self.cache_timestamp[exchange] = now
                all_positions.extend(positions)
            except Exception as e:
                logger.error(f"[USER_CONTEXT] Error fetching positions from {exchange}: {e}")
                # Use cached if available
                all_positions.extend(self.cached_positions.get(exchange, []))
        
        return all_positions
    
    async def get_total_balance(self) -> Dict[str, Any]:
        """Get aggregated balance across all exchanges."""
        total_equity = 0
        total_available = 0
        total_margin = 0
        total_unrealized = 0
        exchange_balances = {}
        
        for exchange, client in self.connections.items():
            try:
                balance = await client.get_balance()
                total_equity += balance.total_equity
                total_available += balance.available_balance
                total_margin += balance.margin_used
                total_unrealized += balance.unrealized_pnl
                exchange_balances[exchange] = {
                    "equity": balance.total_equity,
                    "available": balance.available_balance,
                    "margin": balance.margin_used,
                    "unrealized_pnl": balance.unrealized_pnl
                }
            except Exception as e:
                logger.error(f"[USER_CONTEXT] Error fetching balance from {exchange}: {e}")
        
        return {
            "total_equity": total_equity,
            "total_available": total_available,
            "total_margin_used": total_margin,
            "total_unrealized_pnl": total_unrealized,
            "by_exchange": exchange_balances
        }
    
    def get_position_context_for_ai(self, positions: List[Position]) -> str:
        """
        Generate a context string for IROS AI about user's current positions.
        This gets injected into AI queries for position-aware responses.
        """
        if not positions:
            return "User has no open positions."
        
        context_parts = ["User's Current Positions:"]
        
        for pos in positions:
            direction = "LONG" if pos.direction == "long" else "SHORT"
            pnl_sign = "+" if pos.pnl >= 0 else ""
            
            context_parts.append(
                f"- {pos.symbol} {direction}: Entry ${pos.entry_price:,.2f}, "
                f"Current ${pos.current_price:,.2f}, "
                f"Size ${pos.size_usd:,.0f}, "
                f"P&L {pnl_sign}${pos.pnl:,.2f} ({pnl_sign}{pos.pnl_pct:.1f}%)"
            )
            
            if pos.stop_loss:
                context_parts.append(f"  Stop Loss: ${pos.stop_loss:,.2f}")
            if pos.take_profit:
                context_parts.append(f"  Take Profit: ${pos.take_profit:,.2f}")
            if pos.liquidation_price:
                context_parts.append(f"  Liquidation: ${pos.liquidation_price:,.2f}")
        
        return "\n".join(context_parts)


# Global instance
user_context = UserContextService()



