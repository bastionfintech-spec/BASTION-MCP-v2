"""
Whale Alert Premium API Client
===============================
Real-time whale transaction monitoring

API Key: Pre-configured ($29.95/mo plan)
Docs: https://whale-alert.io/documentation
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import httpx

from ..config.settings import settings


@dataclass
class WhaleTransaction:
    """A whale transaction"""
    id: str
    blockchain: str
    symbol: str
    amount: float
    amount_usd: float
    from_address: str
    from_owner: Optional[str]
    from_owner_type: Optional[str]
    to_address: str
    to_owner: Optional[str]
    to_owner_type: Optional[str]
    timestamp: datetime
    transaction_type: str  # transfer, mint, burn
    hash: str


@dataclass
class WhaleAlertResponse:
    """Response from Whale Alert API"""
    success: bool
    transactions: List[WhaleTransaction] = field(default_factory=list)
    error: Optional[str] = None
    cursor: Optional[str] = None


class WhaleAlertClient:
    """
    Client for Whale Alert Premium API.
    
    Features:
    - Real-time whale transaction alerts
    - Exchange attribution (Binance, Kraken, etc.)
    - 11 blockchains (BTC, ETH, SOL, etc.)
    - Stablecoin mints/burns
    
    Usage:
        client = WhaleAlertClient()
        
        # REST API - get recent transactions
        txs = await client.get_transactions(min_value=10000000)
        
        # WebSocket - real-time alerts
        await client.stream_transactions(callback=my_handler)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.whale_alert.api_key
        self.rest_url = settings.whale_alert.rest_url
        self.ws_url = settings.whale_alert.ws_url
        self.min_value = settings.whale_alert.min_value
    
    def _parse_transaction(self, tx: Dict) -> WhaleTransaction:
        """Parse API response into WhaleTransaction"""
        return WhaleTransaction(
            id=tx.get("id", ""),
            blockchain=tx.get("blockchain", ""),
            symbol=tx.get("symbol", ""),
            amount=tx.get("amount", 0),
            amount_usd=tx.get("amount_usd", 0),
            from_address=tx.get("from", {}).get("address", ""),
            from_owner=tx.get("from", {}).get("owner"),
            from_owner_type=tx.get("from", {}).get("owner_type"),
            to_address=tx.get("to", {}).get("address", ""),
            to_owner=tx.get("to", {}).get("owner"),
            to_owner_type=tx.get("to", {}).get("owner_type"),
            timestamp=datetime.fromtimestamp(tx.get("timestamp", 0)),
            transaction_type=tx.get("transaction_type", "transfer"),
            hash=tx.get("hash", ""),
        )
    
    # =========================================================================
    # REST API
    # =========================================================================
    
    async def get_status(self) -> Dict[str, Any]:
        """Check API status and rate limits"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.rest_url}/status",
                params={"api_key": self.api_key}
            )
            return response.json()
    
    async def get_transactions(
        self,
        min_value: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> WhaleAlertResponse:
        """
        Get recent whale transactions.
        
        Args:
            min_value: Minimum USD value (default: $1M)
            start: Start timestamp (Unix)
            end: End timestamp (Unix)
            cursor: Pagination cursor
            limit: Max transactions to return
        """
        params = {
            "api_key": self.api_key,
            "min_value": min_value or self.min_value,
            "limit": limit,
        }
        
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if cursor:
            params["cursor"] = cursor
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.rest_url}/transactions",
                    params=params
                )
                
                if response.status_code != 200:
                    return WhaleAlertResponse(
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text}"
                    )
                
                data = response.json()
                
                transactions = [
                    self._parse_transaction(tx) 
                    for tx in data.get("transactions", [])
                ]
                
                return WhaleAlertResponse(
                    success=True,
                    transactions=transactions,
                    cursor=data.get("cursor")
                )
                
        except Exception as e:
            return WhaleAlertResponse(success=False, error=str(e))
    
    async def get_transaction(self, blockchain: str, hash: str) -> WhaleAlertResponse:
        """Get a specific transaction by hash"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.rest_url}/transaction/{blockchain}/{hash}",
                    params={"api_key": self.api_key}
                )
                
                if response.status_code != 200:
                    return WhaleAlertResponse(
                        success=False,
                        error=f"HTTP {response.status_code}"
                    )
                
                data = response.json()
                tx = self._parse_transaction(data.get("transaction", {}))
                
                return WhaleAlertResponse(success=True, transactions=[tx])
                
        except Exception as e:
            return WhaleAlertResponse(success=False, error=str(e))
    
    # =========================================================================
    # WEBSOCKET STREAMING
    # =========================================================================
    
    async def stream_transactions(
        self,
        callback: Callable[[WhaleTransaction], None],
        min_value: Optional[int] = None,
        symbols: Optional[List[str]] = None,
    ):
        """
        Stream real-time whale transactions via WebSocket.
        
        Args:
            callback: Function to call for each transaction
            min_value: Minimum USD value filter
            symbols: List of symbols to filter (e.g., ["btc", "eth"])
        
        Usage:
            async def handle_whale(tx: WhaleTransaction):
                print(f"🐋 {tx.amount} {tx.symbol} (${tx.amount_usd:,.0f})")
            
            await client.stream_transactions(handle_whale)
        """
        try:
            import websockets
        except ImportError:
            raise ImportError("Install websockets: pip install websockets")
        
        async with websockets.connect(self.ws_url) as ws:
            # Subscribe to alerts
            subscribe_msg = {
                "type": "subscribe",
                "api_key": self.api_key,
                "min_value": min_value or self.min_value,
            }
            
            if symbols:
                subscribe_msg["symbols"] = symbols
            
            await ws.send(json.dumps(subscribe_msg))
            
            # Listen for transactions
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "transaction":
                        tx = self._parse_transaction(data.get("transaction", {}))
                        callback(tx)
                    elif data.get("type") == "error":
                        print(f"[WhaleAlert] Error: {data.get('message')}")
                    
                except json.JSONDecodeError:
                    continue
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    async def get_exchange_flows(
        self,
        symbol: str = "btc",
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Calculate exchange inflows/outflows over a time period.
        """
        end = int(time.time())
        start = end - (hours * 3600)
        
        result = await self.get_transactions(start=start, end=end, limit=500)
        
        if not result.success:
            return {"error": result.error}
        
        inflows = 0.0
        outflows = 0.0
        exchange_breakdown = {}
        
        for tx in result.transactions:
            if tx.symbol.lower() != symbol.lower():
                continue
            
            # Inflow = transfer TO exchange
            if tx.to_owner_type == "exchange":
                inflows += tx.amount_usd
                exchange = tx.to_owner or "unknown"
                exchange_breakdown[exchange] = exchange_breakdown.get(exchange, 0) + tx.amount_usd
            
            # Outflow = transfer FROM exchange
            if tx.from_owner_type == "exchange":
                outflows += tx.amount_usd
        
        return {
            "symbol": symbol,
            "hours": hours,
            "inflows_usd": inflows,
            "outflows_usd": outflows,
            "net_flow_usd": outflows - inflows,  # Positive = bullish (leaving exchanges)
            "exchange_breakdown": exchange_breakdown,
            "transaction_count": len([t for t in result.transactions if t.symbol.lower() == symbol.lower()]),
        }
    
    def format_transaction(self, tx: WhaleTransaction) -> str:
        """Format transaction for display"""
        direction = ""
        if tx.from_owner_type == "exchange" and tx.to_owner_type != "exchange":
            direction = "← WITHDRAW"
        elif tx.from_owner_type != "exchange" and tx.to_owner_type == "exchange":
            direction = "→ DEPOSIT"
        elif tx.transaction_type == "mint":
            direction = "🆕 MINT"
        elif tx.transaction_type == "burn":
            direction = "🔥 BURN"
        
        from_label = tx.from_owner or tx.from_address[:12] + "..."
        to_label = tx.to_owner or tx.to_address[:12] + "..."
        
        return (
            f"🐋 {tx.amount:,.2f} {tx.symbol.upper()} (${tx.amount_usd:,.0f}) "
            f"{direction} | {from_label} → {to_label}"
        )












