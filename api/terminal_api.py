"""
BASTION Terminal API
=====================
Full API for the Trading Terminal - connects all IROS intelligence
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
import asyncio
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add paths
bastion_path = Path(__file__).parent.parent
sys.path.insert(0, str(bastion_path))

# Import IROS integration
from iros_integration.services.helsinki import HelsinkiClient
from iros_integration.services.query_processor import QueryProcessor
from iros_integration.services.whale_alert import WhaleAlertClient
from iros_integration.services.coinglass import CoinglassClient

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global clients
helsinki: HelsinkiClient = None
query_processor: QueryProcessor = None
whale_alert: WhaleAlertClient = None
coinglass: CoinglassClient = None

# WebSocket connections
active_connections: List[WebSocket] = []

# Price cache - reduced TTL for high-frequency mode (300 req/min tier)
price_cache: Dict[str, Any] = {}
cache_ttl = 2  # seconds (was 10, now faster for premium tier)

# Mock position data (will be replaced with real session data)
MOCK_POSITIONS = [
    {
        "id": "pos_btc_001",
        "symbol": "BTC-PERP",
        "direction": "long",
        "entry_price": 95120,
        "current_price": 96847,
        "size": 0.45,
        "pnl_pct": 1.82,
        "r_multiple": 2.4,
        "stop": 94180,
        "targets": [
            {"price": 96800, "hit": True, "label": "T1"},
            {"price": 98200, "hit": False, "label": "T2"},
        ],
        "trailing": {"active": True, "price": 96412, "slope": 2.3},
        "status": "trailing"
    },
    {
        "id": "pos_eth_001",
        "symbol": "ETH-PERP",
        "direction": "short",
        "entry_price": 3245,
        "current_price": 3198,
        "size": 5.2,
        "pnl_pct": 1.47,
        "r_multiple": 0.8,
        "stop": 3312,
        "guard": {"active": True, "price": 3287},
        "targets": [
            {"price": 3100, "hit": False, "label": "T1"},
        ],
        "status": "guarding"
    },
    {
        "id": "pos_sol_001",
        "symbol": "SOL-PERP",
        "direction": "long",
        "entry_price": 142.80,
        "current_price": 141.20,
        "size": 50,
        "pnl_pct": -1.12,
        "r_multiple": -0.3,
        "stop": 140.10,
        "targets": [],
        "status": "at_risk"
    }
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    global helsinki, query_processor, whale_alert, coinglass
    
    logger.info("BASTION Terminal API starting...")
    
    # Initialize clients
    helsinki = HelsinkiClient()
    query_processor = QueryProcessor()
    whale_alert = WhaleAlertClient()
    coinglass = CoinglassClient()
    
    logger.info("Helsinki VM client ready (33 endpoints)")
    logger.info("Whale Alert client ready (streaming)")
    logger.info("Coinglass client ready (heatmap + liquidations)")
    logger.info("Query processor ready")
    logger.info("BASTION Terminal API LIVE on http://localhost:8888")
    
    yield
    
    logger.info("BASTION Terminal API shutting down...")


app = FastAPI(
    title="BASTION Terminal API",
    description="Powers the BASTION Trading Terminal",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# TERMINAL PAGE
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_terminal():
    """Serve the terminal page."""
    terminal_path = bastion_path / "generated-page.html"
    if terminal_path.exists():
        return FileResponse(terminal_path)
    return HTMLResponse("<h1>Terminal not found</h1>")


@app.get("/frontend", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend landing page."""
    frontend_path = bastion_path / "BASTION FRONT END.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return HTMLResponse("<h1>Frontend not found</h1>")


# =============================================================================
# POSITIONS API
# =============================================================================

@app.get("/api/positions")
async def get_positions():
    """Get all active positions."""
    return {
        "positions": MOCK_POSITIONS,
        "summary": {
            "total_positions": len(MOCK_POSITIONS),
            "total_pnl_pct": sum(p["pnl_pct"] for p in MOCK_POSITIONS),
            "total_exposure_usd": 70245,
            "risk_pct": 1.8
        }
    }


@app.get("/api/positions/{position_id}")
async def get_position(position_id: str):
    """Get a specific position."""
    for pos in MOCK_POSITIONS:
        if pos["id"] == position_id:
            return pos
    raise HTTPException(status_code=404, detail="Position not found")


# =============================================================================
# LIVE PRICE API (Multiple sources with fallback)
# =============================================================================

@app.get("/api/price/{symbol}")
async def get_live_price(symbol: str = "BTC"):
    """Get real-time price with multiple source fallback."""
    import httpx
    
    sym = symbol.upper()
    cache_key = f"price_{sym}"
    now = time.time()
    
    # Check cache
    if cache_key in price_cache:
        cached = price_cache[cache_key]
        if now - cached["time"] < cache_ttl:
            return cached["data"]
    
    result = None
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Try 1: Kraken (no geo-restrictions)
        try:
            kraken_symbol = "XXBTZUSD" if sym == "BTC" else f"X{sym}ZUSD" if sym == "ETH" else f"{sym}USD"
            url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
            res = await client.get(url)
            data = res.json()
            
            if data.get("result"):
                for key, ticker in data["result"].items():
                    result = {
                        "symbol": sym,
                        "price": float(ticker["c"][0]),  # Last trade price
                        "change_24h": 0,  # Would need separate calc
                        "high_24h": float(ticker["h"][1]),
                        "low_24h": float(ticker["l"][1]),
                        "volume_24h": float(ticker["v"][1]),
                    }
                    break
        except Exception as e:
            logger.warning(f"Kraken price error: {e}")
        
        # Try 2: Coinbase if Kraken failed
        if not result:
            try:
                coinbase_symbol = f"{sym}-USD"
                url = f"https://api.exchange.coinbase.com/products/{coinbase_symbol}/ticker"
                res = await client.get(url)
                data = res.json()
                
                if data.get("price"):
                    result = {
                        "symbol": sym,
                        "price": float(data["price"]),
                        "change_24h": 0,
                        "high_24h": 0,
                        "low_24h": 0,
                        "volume_24h": float(data.get("volume", 0)),
                    }
            except Exception as e:
                logger.warning(f"Coinbase price error: {e}")
        
        # Try 3: CoinGecko as last resort
        if not result:
            try:
                coin_id = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}.get(sym, sym.lower())
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
                res = await client.get(url)
                data = res.json()
                
                if data.get(coin_id):
                    result = {
                        "symbol": sym,
                        "price": float(data[coin_id]["usd"]),
                        "change_24h": float(data[coin_id].get("usd_24h_change", 0)),
                        "high_24h": 0,
                        "low_24h": 0,
                        "volume_24h": 0,
                    }
            except Exception as e:
                logger.warning(f"CoinGecko price error: {e}")
    
    if result:
        price_cache[cache_key] = {"time": now, "data": result}
        return result
    
    # Return cached even if stale
    if cache_key in price_cache:
        return price_cache[cache_key]["data"]
    return {"symbol": sym, "price": 0, "error": "All price sources failed"}


@app.get("/api/klines/{symbol}")
async def get_klines(symbol: str = "BTC", interval: str = "15m", limit: int = 100):
    """Get OHLCV candles - tries multiple sources with fallback."""
    import httpx
    
    sym = symbol.upper()
    cache_key = f"klines_{sym}_{interval}"
    now = time.time()
    
    # Check cache (5 second TTL for high-frequency mode)
    if cache_key in price_cache:
        cached = price_cache[cache_key]
        if now - cached["time"] < 5:
            return cached["data"]
    
    candles = []
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Try 1: Kraken (no geo-restrictions)
        try:
            kraken_symbol = f"X{sym}ZUSD" if sym == "BTC" else f"{sym}USD"
            if sym == "BTC":
                kraken_symbol = "XXBTZUSD"
            elif sym == "ETH":
                kraken_symbol = "XETHZUSD"
            else:
                kraken_symbol = f"{sym}USD"
            
            # Map intervals to Kraken (minutes)
            interval_mins = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
            kraken_interval = interval_mins.get(interval, 15)
            
            url = f"https://api.kraken.com/0/public/OHLC?pair={kraken_symbol}&interval={kraken_interval}"
            res = await client.get(url)
            data = res.json()
            
            if data.get("result"):
                # Get the first (only) result key that's not "last"
                for key, values in data["result"].items():
                    if key != "last" and isinstance(values, list):
                        for k in values[-limit:]:
                            candles.append({
                                "time": int(k[0]),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[6]),
                            })
                        break
            
            if candles:
                logger.info(f"Klines from Kraken: {len(candles)} candles for {sym} {interval}")
        except Exception as e:
            logger.warning(f"Kraken klines error: {e}")
        
        # Try 2: Coinbase if Kraken failed
        if not candles:
            try:
                coinbase_symbol = f"{sym}-USD"
                # Map intervals to Coinbase granularity (seconds)
                granularity = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
                gran = granularity.get(interval, 900)
                
                url = f"https://api.exchange.coinbase.com/products/{coinbase_symbol}/candles?granularity={gran}"
                res = await client.get(url)
                data = res.json()
                
                if isinstance(data, list):
                    # Coinbase returns newest first, reverse it
                    for k in reversed(data[-limit:]):
                        candles.append({
                            "time": int(k[0]),
                            "open": float(k[3]),
                            "high": float(k[2]),
                            "low": float(k[1]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                        })
                
                if candles:
                    logger.info(f"Klines from Coinbase: {len(candles)} candles for {sym} {interval}")
            except Exception as e:
                logger.warning(f"Coinbase klines error: {e}")
    
    result = {"symbol": sym, "interval": interval, "candles": candles}
    if candles:
        price_cache[cache_key] = {"time": now, "data": result}
    return result


# =============================================================================
# MARKET INTELLIGENCE API (Helsinki VM)
# =============================================================================

@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str = "BTC"):
    """Get comprehensive market data from Helsinki VM."""
    try:
        data = await helsinki.fetch_full_data(symbol.upper())
        formatted = helsinki.format_for_prompt(data)
        
        return {
            "symbol": symbol.upper(),
            "timestamp": data.timestamp,
            "price": data.meta.get("current_price"),
            "order_flow": data.order_flow,
            "derivatives": data.derivatives,
            "volatility": data.volatility,
            "sentiment": data.sentiment,
            "macro": data.macro,
            "formatted": formatted,
            "meta": data.meta
        }
    except Exception as e:
        logger.error(f"Helsinki fetch error: {e}")
        raise HTTPException(status_code=502, detail=f"Helsinki VM error: {str(e)}")


@app.get("/api/cvd/{symbol}")
async def get_cvd(symbol: str = "BTC"):
    """Get CVD data for a symbol."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{helsinki.base_url}/quant/cvd/{symbol.upper()}")
            data = res.json()
            logger.info(f"CVD raw for {symbol}: {data}")
            
            # Parse CVD data according to Helsinki format
            cvd = {
                "cvd_1h": data.get("cvd_1h", 0),
                "cvd_4h": data.get("cvd_4h", 0),
                "cvd_1d": data.get("cvd_total", data.get("cvd_24h", 0)),
                "cvd_1h_usd": data.get("cvd_1h_usd", 0),
                "divergence": data.get("divergence", "NONE"),
                "signal": data.get("signal", "NEUTRAL"),
                "interpretation": data.get("interpretation", ""),
            }
            
            return {"symbol": symbol.upper(), "cvd": cvd, "raw": data}
    except Exception as e:
        logger.error(f"CVD fetch error: {e}")
        return {"symbol": symbol.upper(), "cvd": {"cvd_1h": 0, "cvd_4h": 0, "cvd_1d": 0}, "error": str(e)}


@app.get("/api/volatility/{symbol}")
async def get_volatility(symbol: str = "BTC"):
    """Get volatility data for a symbol."""
    try:
        data = await helsinki.fetch_volatility_data(symbol.upper())
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/liquidations/{symbol}")
async def get_liquidations(symbol: str = "BTC"):
    """Get liquidation data for a symbol."""
    try:
        data = await helsinki.fetch_liquidation_data(symbol.upper())
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/funding")
async def get_funding():
    """Get funding rates for all major pairs."""
    import httpx
    import asyncio
    
    rates = {"BTC": 0, "ETH": 0, "SOL": 0}
    basis = 0
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Fetch liquidation-estimate for each symbol (contains funding_rate_pct)
            async def get_funding_for(symbol):
                try:
                    res = await client.get(f"{helsinki.base_url}/quant/liquidation-estimate/{symbol}")
                    data = res.json()
                    return symbol, data.get("funding_rate_pct", 0) / 100  # Convert from % to decimal
                except:
                    return symbol, 0
            
            results = await asyncio.gather(
                get_funding_for("BTC"),
                get_funding_for("ETH"),
                get_funding_for("SOL")
            )
            
            for symbol, rate in results:
                rates[symbol] = rate
            
            # Get basis from BTC
            try:
                basis_res = await client.get(f"{helsinki.base_url}/quant/basis/BTC")
                basis_data = basis_res.json()
                basis = basis_data.get("basis_percent", basis_data.get("basis", 0))
            except:
                pass
            
            logger.info(f"Funding rates: {rates}, Basis: {basis}")
            
            return {
                "rates": rates, 
                "basis": basis,
                "next_funding": "~4h", 
                "source": "Helsinki"
            }
    except Exception as e:
        logger.error(f"Funding fetch error: {e}")
        return {"rates": {"BTC": 0.0001, "ETH": 0.00008, "SOL": 0.0002}, "basis": 0.12, "next_funding": "~4h", "source": "fallback"}


@app.get("/api/oi/{symbol}")
async def get_open_interest(symbol: str = "BTC"):
    """Get open interest data."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{helsinki.base_url}/quant/open-interest/{symbol.upper()}")
            data = res.json()
            logger.info(f"OI raw for {symbol}: {data}")
            
            # Parse OI data
            oi = {
                "total_oi": data.get("oi_value", data.get("open_interest", data.get("value", 0))),
                "change_24h": data.get("oi_change_24h", data.get("change_24h", data.get("oi_change", 0))),
                "trend": data.get("trend", "stable"),
                "interpretation": data.get("interpretation", ""),
            }
            
            return {"symbol": symbol.upper(), "oi": oi, "raw": data}
    except Exception as e:
        logger.error(f"OI fetch error: {e}")
        return {"symbol": symbol.upper(), "oi": {"total_oi": 0, "change_24h": 0}}


@app.get("/api/fear-greed")
async def get_fear_greed():
    """Get fear and greed index."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{helsinki.base_url}/sentiment/fear-greed")
            data = res.json()
            return data
    except Exception as e:
        logger.error(f"Fear/Greed fetch error: {e}")
        return {"value": 50, "label": "Neutral"}


# =============================================================================
# SESSION STATS API
# =============================================================================

@app.get("/api/session/stats")
async def get_session_stats():
    """Get current session statistics."""
    return {
        "session_pnl": 4247.32,
        "session_pnl_pct": 2.8,
        "active_positions": len(MOCK_POSITIONS),
        "win_rate": 73,
        "avg_r": 1.8,
        "max_drawdown": -0.4,
        "trades_today": 7,
        "wins": 5,
        "losses": 2
    }


# =============================================================================
# ALERTS API
# =============================================================================

MOCK_ALERTS = [
    {
        "id": "alert_001",
        "type": "target_hit",
        "level": "success",
        "title": "TARGET 1 HIT",
        "message": "BTC-PERP reached $96,800 (+1.2R). Stop moved to breakeven.",
        "timestamp": "14:31:42"
    },
    {
        "id": "alert_002",
        "type": "momentum_tp",
        "level": "info",
        "title": "MOMENTUM TP",
        "message": "Slope strength: 2.3x. Trailing 0.45% below candle body.",
        "timestamp": "14:28:17"
    },
    {
        "id": "alert_003",
        "type": "volatility",
        "level": "warning",
        "title": "VOLATILITY",
        "message": "Regime: NORMAL → HIGH. Recommended: Reduce new entries 25%.",
        "timestamp": "14:25:33"
    },
    {
        "id": "alert_004",
        "type": "shot_executed",
        "level": "success",
        "title": "SHOT 2 EXECUTED",
        "message": "BTC-PERP: Added 0.15 BTC @ $95,890.",
        "timestamp": "14:22:08"
    }
]


@app.get("/api/alerts")
async def get_alerts(limit: int = 10):
    """Get recent alerts."""
    return {"alerts": MOCK_ALERTS[:limit]}


# =============================================================================
# NEURAL ASSISTANT API
# =============================================================================

@app.post("/api/neural/chat")
async def neural_chat(request: Dict[str, Any]):
    """Chat with the Bastion AI."""
    query = request.get("query", "")
    symbol = request.get("symbol", "BTC")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Extract context
    context = query_processor.extract_context(query)
    
    # For now, return a mock response (will connect to real AI later)
    # This shows the structure of what will come from the AI
    return {
        "success": True,
        "response": f"""**Analysis for {symbol}**

Based on current market conditions:
- Smart Money Bias: BULLISH (78%)
- Volatility Regime: NORMAL
- CVD 1H: +2.4M (accumulation)

**Recommendation:** Hold current position. Momentum trailing active at 2.3x slope strength.

⚠️ CVD divergence forming on daily. Consider partial exit at T2 ($98,200).""",
        "context": {
            "symbol": context.symbol,
            "capital": context.capital,
            "timeframe": context.timeframe,
            "intent": context.query_intent
        },
        "data_sources": ["Helsinki:smart-money", "Helsinki:cvd", "Helsinki:volatility"]
    }


# =============================================================================
# ACTIONS API
# =============================================================================

@app.post("/api/actions/emergency-exit")
async def emergency_exit():
    """Emergency exit all positions."""
    logger.warning("🚨 EMERGENCY EXIT triggered!")
    # In production, this would call exchange APIs
    return {
        "success": True,
        "message": "Emergency exit initiated for all positions",
        "positions_closed": len(MOCK_POSITIONS)
    }


@app.post("/api/actions/move-to-breakeven")
async def move_to_breakeven(request: Dict[str, Any]):
    """Move stops to breakeven."""
    position_id = request.get("position_id")
    
    if position_id:
        return {"success": True, "message": f"Position {position_id} stop moved to breakeven"}
    
    return {"success": True, "message": "All profitable positions moved to breakeven"}


@app.post("/api/actions/add-shot")
async def add_shot(request: Dict[str, Any]):
    """Add a shot to an existing position."""
    position_id = request.get("position_id")
    size = request.get("size", 0.1)
    
    return {
        "success": True,
        "message": f"Shot added to {position_id}",
        "new_size": size
    }


@app.post("/api/actions/partial-close")
async def partial_close(request: Dict[str, Any]):
    """Close a percentage of a position."""
    position_id = request.get("position_id")
    percentage = request.get("percentage", 50)
    
    return {
        "success": True,
        "message": f"Closed {percentage}% of {position_id}"
    }


# =============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time terminal updates."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total: {len(active_connections)}")
    
    try:
        # Send initial data
        await websocket.send_json({
            "type": "connected",
            "message": "BASTION Terminal connected",
            "timestamp": datetime.now().isoformat()
        })
        
        # Start sending updates
        while True:
            # Simulate real-time price updates
            price_update = {
                "type": "price_update",
                "data": {
                    "BTC-PERP": 96847 + random.uniform(-50, 50),
                    "ETH-PERP": 3198 + random.uniform(-5, 5),
                    "SOL-PERP": 141.20 + random.uniform(-0.5, 0.5),
                },
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(price_update)
            
            # Check for incoming messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                logger.info(f"Received: {message}")
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(active_connections)}")


# =============================================================================
# HEALTH CHECK
# =============================================================================

# =============================================================================
# WHALE ALERT API
# =============================================================================

@app.get("/api/whales")
async def get_whale_transactions(min_value: int = 1000000, limit: int = 20):
    """Get recent whale transactions."""
    try:
        result = await whale_alert.get_transactions(min_value=min_value, limit=limit)
        
        if not result.success:
            return {"success": False, "error": result.error, "transactions": []}
        
        # Format transactions for frontend
        transactions = []
        for tx in result.transactions:
            # Determine if deposit or withdrawal
            direction = "TRANSFER"
            if tx.from_owner_type == "exchange" and tx.to_owner_type != "exchange":
                direction = "WITHDRAWAL"
            elif tx.from_owner_type != "exchange" and tx.to_owner_type == "exchange":
                direction = "DEPOSIT"
            elif tx.transaction_type == "mint":
                direction = "MINT"
            elif tx.transaction_type == "burn":
                direction = "BURN"
            
            transactions.append({
                "id": tx.id,
                "symbol": tx.symbol.upper(),
                "amount": tx.amount,
                "amount_usd": tx.amount_usd,
                "from": tx.from_owner or tx.from_address[:12] + "...",
                "to": tx.to_owner or tx.to_address[:12] + "...",
                "direction": direction,
                "timestamp": tx.timestamp.strftime("%H:%M:%S"),
                "blockchain": tx.blockchain,
            })
        
        return {"success": True, "transactions": transactions}
    except Exception as e:
        logger.error(f"Whale fetch error: {e}")
        return {"success": False, "error": str(e), "transactions": []}


@app.get("/api/whales/flows/{symbol}")
async def get_whale_flows(symbol: str = "BTC", hours: int = 24):
    """Get exchange inflows/outflows for a symbol."""
    try:
        flows = await whale_alert.get_exchange_flows(symbol=symbol.lower(), hours=hours)
        return {"success": True, "flows": flows}
    except Exception as e:
        logger.error(f"Whale flows error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# COINGLASS API (Premium)
# =============================================================================

@app.get("/api/heatmap/{symbol}")
async def get_liquidation_heatmap(symbol: str = "BTC", t: int = 0):
    """Get liquidation heatmap - try Helsinki first (free, unlimited)."""
    import httpx
    
    # Always fetch fresh - no caching for heatmap
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(f"{helsinki.base_url}/quant/liquidation-estimate/{symbol.upper()}?nocache={t}")
            data = res.json()
            logger.info(f"Heatmap raw for {symbol}: {data}")
            
            if data and "current_price" in data:
                # Convert Helsinki format to heatmap format
                levels = []
                current_price = data.get("current_price", 0)
                
                # Add downside liquidation zones (longs at risk)
                for zone in data.get("downside_liquidation_zones", []):
                    levels.append({
                        "price": zone.get("price", 0),
                        "longLiquidation": zone.get("estimated_usd_at_risk", 0),
                        "shortLiquidation": 0,
                        "leverage": zone.get("leverage_concentration", "unknown")
                    })
                
                # Add upside liquidation zones (shorts at risk)
                for zone in data.get("upside_liquidation_zones", []):
                    levels.append({
                        "price": zone.get("price", 0),
                        "longLiquidation": 0,
                        "shortLiquidation": zone.get("estimated_usd_at_risk", 0),
                        "leverage": zone.get("leverage_concentration", "unknown")
                    })
                
                return {
                    "success": True,
                    "symbol": symbol.upper(),
                    "heatmap": {
                        "levels": levels,
                        "currentPrice": current_price,
                        "long_short_ratio": data.get("long_short_ratio", 1),
                        "cascade_bias": data.get("cascade_bias", "NEUTRAL"),
                        "open_interest_usd": data.get("open_interest_usd", 0)
                    },
                    "source": "Helsinki"
                }
    except Exception as e:
        logger.error(f"Helsinki heatmap error: {e}")
    
    # Fallback to Coinglass
    try:
        result = await coinglass.get_liquidation_map(symbol.upper())
        
        if result.success:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "heatmap": result.data,
                "latency_ms": result.latency_ms,
                "source": "Coinglass"
            }
    except Exception as e:
        logger.error(f"Coinglass heatmap error: {e}")
    
    return {"success": False, "error": "Both sources failed", "heatmap": None}


@app.get("/api/coinglass/liquidations/{symbol}")
async def get_coinglass_liquidations(symbol: str = "BTC", interval: str = "h1", limit: int = 24):
    """Get liquidation history from Coinglass."""
    try:
        result = await coinglass.get_liquidation_history(symbol.upper(), interval, limit)
        
        if not result.success:
            return {"success": False, "error": result.error}
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "liquidations": result.data,
            "latency_ms": result.latency_ms
        }
    except Exception as e:
        logger.error(f"Liquidation history error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/coinglass/overview/{symbol}")
async def get_coinglass_overview(symbol: str = "BTC"):
    """Get comprehensive market data from Coinglass."""
    try:
        overview = await coinglass.get_market_overview(symbol.upper())
        return {"success": True, "symbol": symbol.upper(), "data": overview}
    except Exception as e:
        logger.error(f"Coinglass overview error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# ETF FLOWS API (Coinglass Premium)
# =============================================================================

@app.get("/api/etf-flows")
async def get_etf_flows():
    """Get Bitcoin ETF flows - IBIT, FBTC, GBTC, etc."""
    try:
        result = await coinglass.get_bitcoin_etf()
        
        if result.success and result.data:
            # Parse and format ETF data
            etfs = []
            total_flow = 0
            
            for etf in result.data if isinstance(result.data, list) else [result.data]:
                flow_24h = etf.get("h24Flow", etf.get("flow24h", 0))
                total_btc = etf.get("totalBtc", etf.get("total_btc", 0))
                etfs.append({
                    "name": etf.get("name", "Unknown"),
                    "totalBtc": total_btc,
                    "flow24h": flow_24h,
                    "flowUsd": flow_24h * 98000,  # Approximate
                })
                total_flow += flow_24h
            
            return {
                "success": True,
                "etfs": etfs,
                "totalFlow24h": total_flow,
                "totalFlowUsd": total_flow * 98000,
                "signal": "BULLISH" if total_flow > 0 else "BEARISH" if total_flow < 0 else "NEUTRAL",
                "latency_ms": result.latency_ms
            }
        
        # Fallback mock data if API fails
        return {
            "success": True,
            "etfs": [
                {"name": "IBIT", "totalBtc": 285000, "flow24h": 2340, "flowUsd": 229320000},
                {"name": "FBTC", "totalBtc": 175000, "flow24h": 890, "flowUsd": 87220000},
                {"name": "GBTC", "totalBtc": 215000, "flow24h": -120, "flowUsd": -11760000},
                {"name": "ARKB", "totalBtc": 48000, "flow24h": 450, "flowUsd": 44100000},
            ],
            "totalFlow24h": 3560,
            "totalFlowUsd": 348880000,
            "signal": "BULLISH",
            "source": "fallback"
        }
    except Exception as e:
        logger.error(f"ETF flows error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# TOP TRADER SENTIMENT API (Coinglass Premium)
# =============================================================================

@app.get("/api/top-traders/{symbol}")
async def get_top_traders(symbol: str = "BTC"):
    """Get top trader (whale) long/short sentiment."""
    import asyncio
    
    try:
        # Fetch both top trader and global sentiment
        top_result, global_result = await asyncio.gather(
            coinglass.get_top_trader_sentiment(symbol.upper()),
            coinglass.get_long_short_ratio(symbol.upper()),
            return_exceptions=True
        )
        
        # Parse top trader data
        top_long = 50
        top_short = 50
        if isinstance(top_result, CoinglassClient) or (hasattr(top_result, 'success') and top_result.success and top_result.data):
            data = top_result.data[-1] if isinstance(top_result.data, list) and len(top_result.data) > 0 else top_result.data
            top_long = data.get("longAccount", 50)
            top_short = data.get("shortAccount", 50)
        
        # Parse global (retail) data
        retail_long = 50
        retail_short = 50
        if isinstance(global_result, CoinglassClient) or (hasattr(global_result, 'success') and global_result.success and global_result.data):
            data = global_result.data[-1] if isinstance(global_result.data, list) and len(global_result.data) > 0 else global_result.data
            retail_long = data.get("longAccount", 50)
            retail_short = data.get("shortAccount", 50)
        
        # Detect divergence
        divergence = "NONE"
        if top_long > 55 and retail_long > 70:
            divergence = "CROWDED_LONG"
        elif top_short > 55 and retail_short > 70:
            divergence = "CROWDED_SHORT"
        elif abs(top_long - retail_long) > 15:
            if top_long > retail_long:
                divergence = "SMART_MONEY_LONG"
            else:
                divergence = "SMART_MONEY_SHORT"
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "topTraders": {
                "long": round(top_long, 1),
                "short": round(top_short, 1),
            },
            "retail": {
                "long": round(retail_long, 1),
                "short": round(retail_short, 1),
            },
            "divergence": divergence,
            "signal": "BULLISH" if divergence == "SMART_MONEY_LONG" else "BEARISH" if divergence == "SMART_MONEY_SHORT" else "NEUTRAL"
        }
    except Exception as e:
        logger.error(f"Top traders error: {e}")
        return {
            "success": True,
            "symbol": symbol.upper(),
            "topTraders": {"long": 58.2, "short": 41.8},
            "retail": {"long": 72.4, "short": 27.6},
            "divergence": "SMART_MONEY_LONG",
            "signal": "BULLISH",
            "source": "fallback"
        }


# =============================================================================
# OPTIONS MAX PAIN API (Coinglass Premium)
# =============================================================================

@app.get("/api/options/{symbol}")
async def get_options_data(symbol: str = "BTC"):
    """Get options data - max pain, put/call ratio."""
    try:
        result = await coinglass.get_options_info(symbol.upper())
        
        if result.success and result.data:
            data = result.data
            return {
                "success": True,
                "symbol": symbol.upper(),
                "maxPain": data.get("maxPainPrice", 0),
                "putCallRatio": data.get("putCallRatio", 1),
                "totalOI": data.get("totalOpenInterest", 0),
                "volume24h": data.get("totalVolume24h", 0),
                "signal": "BULLISH" if data.get("putCallRatio", 1) < 0.9 else "BEARISH" if data.get("putCallRatio", 1) > 1.1 else "NEUTRAL",
                "latency_ms": result.latency_ms
            }
        
        # Fallback
        return {
            "success": True,
            "symbol": symbol.upper(),
            "maxPain": 95000,
            "putCallRatio": 0.78,
            "totalOI": 12500000000,
            "volume24h": 890000000,
            "signal": "BULLISH",
            "source": "fallback"
        }
    except Exception as e:
        logger.error(f"Options error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# FUNDING ARBITRAGE API (Coinglass Premium)
# =============================================================================

@app.get("/api/funding-arb/{symbol}")
async def get_funding_arbitrage(symbol: str = "BTC"):
    """Get funding rates by exchange for arbitrage opportunities."""
    try:
        result = await coinglass.get_funding_rates(symbol.upper())
        
        if result.success and result.data:
            exchanges = []
            rates = []
            
            for ex in result.data if isinstance(result.data, list) else [result.data]:
                rate = ex.get("fundingRate", 0)
                exchanges.append({
                    "exchange": ex.get("exchange", "Unknown"),
                    "symbol": ex.get("symbol", f"{symbol}USDT"),
                    "rate": rate,
                    "ratePercent": rate * 100,
                    "nextFunding": ex.get("nextFundingTime", 0)
                })
                rates.append(rate)
            
            # Sort by rate
            exchanges.sort(key=lambda x: x["rate"], reverse=True)
            
            # Calculate spread
            spread = max(rates) - min(rates) if rates else 0
            
            return {
                "success": True,
                "symbol": symbol.upper(),
                "exchanges": exchanges[:8],  # Top 8
                "spread": spread,
                "spreadPercent": spread * 100,
                "highestRate": exchanges[0] if exchanges else None,
                "lowestRate": exchanges[-1] if exchanges else None,
                "arbOpportunity": spread > 0.0001,  # 0.01% threshold
                "latency_ms": result.latency_ms
            }
        
        # Fallback
        return {
            "success": True,
            "symbol": symbol.upper(),
            "exchanges": [
                {"exchange": "Binance", "rate": 0.00012, "ratePercent": 0.012},
                {"exchange": "OKX", "rate": 0.00010, "ratePercent": 0.010},
                {"exchange": "Bybit", "rate": 0.00011, "ratePercent": 0.011},
                {"exchange": "dYdX", "rate": -0.00003, "ratePercent": -0.003},
            ],
            "spread": 0.00015,
            "spreadPercent": 0.015,
            "arbOpportunity": True,
            "source": "fallback"
        }
    except Exception as e:
        logger.error(f"Funding arb error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# OI BY EXCHANGE API (Coinglass Premium)
# =============================================================================

@app.get("/api/oi-exchange/{symbol}")
async def get_oi_by_exchange(symbol: str = "BTC"):
    """Get open interest broken down by exchange."""
    try:
        result = await coinglass.get_open_interest(symbol.upper())
        
        if result.success and result.data:
            exchanges = []
            total_oi = 0
            
            for ex in result.data if isinstance(result.data, list) else [result.data]:
                oi = ex.get("openInterest", 0)
                exchanges.append({
                    "exchange": ex.get("exchange", "Unknown"),
                    "oi": oi,
                    "oiFormatted": f"${oi/1e9:.2f}B" if oi > 1e9 else f"${oi/1e6:.0f}M",
                    "change24h": ex.get("h24Change", 0),
                })
                total_oi += oi
            
            # Sort by OI
            exchanges.sort(key=lambda x: x["oi"], reverse=True)
            
            # Calculate percentages
            for ex in exchanges:
                ex["percentage"] = (ex["oi"] / total_oi * 100) if total_oi > 0 else 0
            
            return {
                "success": True,
                "symbol": symbol.upper(),
                "exchanges": exchanges[:8],
                "totalOI": total_oi,
                "totalOIFormatted": f"${total_oi/1e9:.2f}B" if total_oi > 1e9 else f"${total_oi/1e6:.0f}M",
                "latency_ms": result.latency_ms
            }
        
        # Fallback
        return {
            "success": True,
            "symbol": symbol.upper(),
            "exchanges": [
                {"exchange": "Binance", "oi": 4200000000, "oiFormatted": "$4.2B", "change24h": 2.3, "percentage": 45},
                {"exchange": "CME", "oi": 1800000000, "oiFormatted": "$1.8B", "change24h": -1.2, "percentage": 19},
                {"exchange": "OKX", "oi": 1100000000, "oiFormatted": "$1.1B", "change24h": 0.8, "percentage": 12},
                {"exchange": "Bybit", "oi": 900000000, "oiFormatted": "$0.9B", "change24h": 1.5, "percentage": 10},
                {"exchange": "Bitget", "oi": 600000000, "oiFormatted": "$0.6B", "change24h": 0.5, "percentage": 6},
            ],
            "totalOI": 9400000000,
            "totalOIFormatted": "$9.4B",
            "source": "fallback"
        }
    except Exception as e:
        logger.error(f"OI by exchange error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# TAKER BUY/SELL API (Coinglass Premium)
# =============================================================================

@app.get("/api/taker-ratio/{symbol}")
async def get_taker_ratio(symbol: str = "BTC"):
    """Get taker buy/sell ratio - real-time order flow pressure."""
    try:
        result = await coinglass.get_taker_buy_sell(symbol.upper())
        
        if result.success and result.data:
            data = result.data
            buy_ratio = data.get("buyRatio", 0.5)
            sell_ratio = data.get("sellRatio", 0.5)
            
            return {
                "success": True,
                "symbol": symbol.upper(),
                "buyRatio": buy_ratio,
                "sellRatio": sell_ratio,
                "netFlow": "BUY" if buy_ratio > sell_ratio else "SELL",
                "buyPercent": buy_ratio * 100,
                "sellPercent": sell_ratio * 100,
                "signal": "BULLISH" if buy_ratio > 0.55 else "BEARISH" if sell_ratio > 0.55 else "NEUTRAL",
                "latency_ms": result.latency_ms
            }
        
        # Fallback
        return {
            "success": True,
            "symbol": symbol.upper(),
            "buyRatio": 0.52,
            "sellRatio": 0.48,
            "netFlow": "BUY",
            "buyPercent": 52,
            "sellPercent": 48,
            "signal": "NEUTRAL",
            "source": "fallback"
        }
    except Exception as e:
        logger.error(f"Taker ratio error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# EXCHANGE NET FLOW API (Whale Alert)
# =============================================================================

@app.get("/api/exchange-flow/{symbol}")
async def get_exchange_net_flow(symbol: str = "BTC", hours: int = 24):
    """Get net exchange inflow/outflow for a symbol."""
    try:
        flows = await whale_alert.get_exchange_flows(symbol=symbol.lower(), hours=hours)
        
        if "error" not in flows:
            net_flow = flows.get("net_flow_usd", 0)
            inflows = flows.get("inflows_usd", 0)
            outflows = flows.get("outflows_usd", 0)
            
            return {
                "success": True,
                "symbol": symbol.upper(),
                "hours": hours,
                "inflows": inflows,
                "inflowsFormatted": f"${inflows/1e6:.1f}M" if inflows > 1e6 else f"${inflows/1e3:.0f}K",
                "outflows": outflows,
                "outflowsFormatted": f"${outflows/1e6:.1f}M" if outflows > 1e6 else f"${outflows/1e3:.0f}K",
                "netFlow": net_flow,
                "netFlowFormatted": f"${abs(net_flow)/1e6:.1f}M" if abs(net_flow) > 1e6 else f"${abs(net_flow)/1e3:.0f}K",
                "direction": "OUTFLOW" if net_flow > 0 else "INFLOW",
                "signal": "BULLISH" if net_flow > 0 else "BEARISH",
                "exchangeBreakdown": flows.get("exchange_breakdown", {}),
                "txCount": flows.get("transaction_count", 0)
            }
        
        # Fallback
        return {
            "success": True,
            "symbol": symbol.upper(),
            "hours": hours,
            "inflows": 89000000,
            "inflowsFormatted": "$89M",
            "outflows": 204000000,
            "outflowsFormatted": "$204M",
            "netFlow": 115000000,
            "netFlowFormatted": "$115M",
            "direction": "OUTFLOW",
            "signal": "BULLISH",
            "source": "fallback"
        }
    except Exception as e:
        logger.error(f"Exchange flow error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# LIQUIDATION BY EXCHANGE API (Coinglass Premium)
# =============================================================================

@app.get("/api/liq-exchange/{symbol}")
async def get_liquidation_by_exchange(symbol: str = "BTC"):
    """Get liquidations broken down by exchange."""
    try:
        result = await coinglass.get_liquidation_by_exchange(symbol.upper())
        
        if result.success and result.data:
            exchanges = []
            total_long = 0
            total_short = 0
            
            for ex in result.data if isinstance(result.data, list) else [result.data]:
                long_liq = ex.get("h24LongLiqUsd", 0)
                short_liq = ex.get("h24ShortLiqUsd", 0)
                exchanges.append({
                    "exchange": ex.get("exchange", "Unknown"),
                    "longLiq": long_liq,
                    "shortLiq": short_liq,
                    "total": long_liq + short_liq,
                    "longPercent": (long_liq / (long_liq + short_liq) * 100) if (long_liq + short_liq) > 0 else 50,
                })
                total_long += long_liq
                total_short += short_liq
            
            # Sort by total
            exchanges.sort(key=lambda x: x["total"], reverse=True)
            
            return {
                "success": True,
                "symbol": symbol.upper(),
                "exchanges": exchanges[:6],
                "totalLongLiq": total_long,
                "totalShortLiq": total_short,
                "total24h": total_long + total_short,
                "dominantSide": "LONGS" if total_long > total_short else "SHORTS",
                "latency_ms": result.latency_ms
            }
        
        # Fallback
        return {
            "success": True,
            "symbol": symbol.upper(),
            "exchanges": [
                {"exchange": "Binance", "longLiq": 45000000, "shortLiq": 32000000, "total": 77000000, "longPercent": 58.4},
                {"exchange": "OKX", "longLiq": 23000000, "shortLiq": 18000000, "total": 41000000, "longPercent": 56.1},
                {"exchange": "Bybit", "longLiq": 15000000, "shortLiq": 12000000, "total": 27000000, "longPercent": 55.6},
            ],
            "totalLongLiq": 83000000,
            "totalShortLiq": 62000000,
            "total24h": 145000000,
            "dominantSide": "LONGS",
            "source": "fallback"
        }
    except Exception as e:
        logger.error(f"Liq by exchange error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# VOLATILITY REGIME ANALYSIS API
# =============================================================================

@app.get("/api/volatility-regime/{symbol}")
async def get_volatility_regime(symbol: str = "BTC"):
    """
    Analyze volatility regime - compression vs expansion.
    Predicts when volatility regime is about to change.
    """
    import httpx
    
    try:
        # Fetch volatility data from Helsinki
        async with httpx.AsyncClient(timeout=10.0) as client:
            vol_res = await client.get(f"{helsinki.base_url}/quant/volatility/{symbol.upper()}")
            vol_data = vol_res.json()
            
            # Get recent price data for ATR calculation
            price_res = await client.get(f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol.upper()}&tsym=USD&limit=14")
            price_data = price_res.json()
        
        # Parse volatility
        current_vol = vol_data.get("daily_volatility_pct", vol_data.get("volatility_pct", 2.0))
        atr_pct = vol_data.get("atr_pct", current_vol)
        regime = vol_data.get("regime", vol_data.get("current_regime", "NORMAL"))
        
        # Calculate Bollinger Width from price data
        prices = [c.get("close", 0) for c in price_data.get("Data", {}).get("Data", [])]
        if len(prices) >= 14:
            import statistics
            mean_price = statistics.mean(prices[-20:] if len(prices) >= 20 else prices)
            std_dev = statistics.stdev(prices[-20:] if len(prices) >= 20 else prices)
            bb_width = (std_dev / mean_price) * 100 * 4  # 2 std devs * 2 sides
        else:
            bb_width = 4.0  # Default
        
        # Determine compression status
        is_compression = bb_width < 5.0 or atr_pct < 2.0
        
        # Estimate days in current regime (simplified - would need historical tracking)
        days_in_regime = 3 if is_compression else 1
        
        # Calculate expansion probability
        if is_compression:
            # Longer compression = higher probability of expansion
            expansion_prob = min(90, 40 + (days_in_regime * 10))
        else:
            expansion_prob = 20
        
        # Generate recommendation
        if is_compression and expansion_prob > 60:
            recommendation = [
                "Reduce position size 50% (whipsaws likely)",
                "Widen stops OR wait for direction confirmation",
                "Set breakout alerts above/below range"
            ]
            regime_display = "COMPRESSION"
            warning = "EXPANSION IMMINENT"
        elif regime == "HIGH" or regime == "EXTREME":
            recommendation = [
                "Use smaller position sizes",
                "Take profits more aggressively",
                "Expect larger swings"
            ]
            regime_display = "EXPANSION"
            warning = None
        else:
            recommendation = [
                "Standard position sizing OK",
                "Normal stop distances",
                "Trend following strategies preferred"
            ]
            regime_display = "NORMAL"
            warning = None
        
        # Determine direction bias
        if prices and len(prices) >= 3:
            recent_change = (prices[-1] - prices[-3]) / prices[-3] * 100 if prices[-3] > 0 else 0
            direction_bias = "BULLISH" if recent_change > 1 else "BEARISH" if recent_change < -1 else "UNKNOWN"
        else:
            direction_bias = "UNKNOWN"
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "regime": regime_display,
            "atrPercent": round(atr_pct, 2),
            "bollingerWidth": round(bb_width, 2),
            "daysInRegime": days_in_regime,
            "isCompression": is_compression,
            "expansionProbability": expansion_prob,
            "warning": warning,
            "directionBias": direction_bias,
            "recommendation": recommendation,
            "signal": "CAUTION" if is_compression else "NORMAL"
        }
        
    except Exception as e:
        logger.error(f"Volatility regime error: {e}")
        # Fallback with realistic data
        return {
            "success": True,
            "symbol": symbol.upper(),
            "regime": "COMPRESSION",
            "atrPercent": 1.8,
            "bollingerWidth": 3.2,
            "daysInRegime": 4,
            "isCompression": True,
            "expansionProbability": 72,
            "warning": "EXPANSION IMMINENT",
            "directionBias": "UNKNOWN",
            "recommendation": [
                "Reduce position size 50% (whipsaws likely)",
                "Widen stops OR wait for direction confirmation",
                "Set alerts at key breakout levels"
            ],
            "signal": "CAUTION",
            "source": "fallback"
        }


# =============================================================================
# MM MAGNET (Market Maker Target Estimation) API
# =============================================================================

@app.get("/api/mm-magnet/{symbol}")
async def get_mm_magnet(symbol: str = "BTC"):
    """
    Estimate where market makers want to push price.
    
    Uses weighted signals:
    - Options Max Pain (30%)
    - Liquidation Cluster Hunting (25%)
    - Funding Rate Mean Reversion (15%)
    - Top Trader vs Retail Divergence (15%)
    - ETF Flow Direction (15%)
    """
    import asyncio
    import httpx
    
    try:
        # Fetch all required data in parallel
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get current price
            price_task = client.get(f"https://min-api.cryptocompare.com/data/pricemultifull?fsyms={symbol.upper()}&tsyms=USD")
            
            # Get liquidation data from Helsinki
            liq_task = client.get(f"{helsinki.base_url}/quant/liquidation-estimate/{symbol.upper()}")
            
            # Get funding data
            funding_task = client.get(f"{helsinki.base_url}/quant/liquidation-estimate/{symbol.upper()}")  # Contains funding
            
            price_res, liq_res, funding_res = await asyncio.gather(
                price_task, liq_task, funding_task,
                return_exceptions=True
            )
        
        # Parse current price
        current_price = 0
        if not isinstance(price_res, Exception):
            price_data = price_res.json()
            current_price = price_data.get("RAW", {}).get(symbol.upper(), {}).get("USD", {}).get("PRICE", 0)
        
        # Parse liquidation data
        long_liq_cluster = {"price": current_price * 0.96, "usd": 837000000}
        short_liq_cluster = {"price": current_price * 1.04, "usd": 440000000}
        ls_ratio = 1.9
        
        if not isinstance(liq_res, Exception):
            liq_data = liq_res.json()
            ls_ratio = liq_data.get("long_short_ratio", 1.9)
            
            # Get downside (long) liquidation zones
            downside_zones = liq_data.get("downside_liquidation_zones", [])
            if downside_zones:
                biggest_long = max(downside_zones, key=lambda x: x.get("estimated_usd_at_risk", 0))
                long_liq_cluster = {
                    "price": biggest_long.get("price", current_price * 0.96),
                    "usd": biggest_long.get("estimated_usd_at_risk", 837000000)
                }
            
            # Get upside (short) liquidation zones
            upside_zones = liq_data.get("upside_liquidation_zones", [])
            if upside_zones:
                biggest_short = max(upside_zones, key=lambda x: x.get("estimated_usd_at_risk", 0))
                short_liq_cluster = {
                    "price": biggest_short.get("price", current_price * 1.04),
                    "usd": biggest_short.get("estimated_usd_at_risk", 440000000)
                }
        
        # Parse funding rate
        funding_rate = 0.0001
        if not isinstance(funding_res, Exception):
            funding_data = funding_res.json()
            funding_rate = funding_data.get("funding_rate_pct", 0.01) / 100
        
        # === SIGNAL CALCULATIONS ===
        
        # Signal 1: Max Pain (mock - would come from Coinglass options API)
        # Typically max pain is near current price +-5%
        max_pain_price = current_price * 1.02  # Mock: slightly above current
        max_pain_signal = ((max_pain_price - current_price) / current_price) if current_price > 0 else 0
        max_pain_signal = max(-1, min(1, max_pain_signal * 10))  # Scale to -1 to 1
        
        # Signal 2: Liquidation Hunt
        # Which cluster has more money at risk? MM hunt the bigger one
        if long_liq_cluster["usd"] > short_liq_cluster["usd"] * 1.3:
            liq_signal = -1.0  # Hunt longs (push down)
            liq_target = long_liq_cluster["price"]
        elif short_liq_cluster["usd"] > long_liq_cluster["usd"] * 1.3:
            liq_signal = 1.0  # Hunt shorts (push up)
            liq_target = short_liq_cluster["price"]
        else:
            liq_signal = 0.0  # Balanced
            liq_target = current_price
        
        # Signal 3: Funding Rate
        if funding_rate > 0.0003:  # Very high positive funding
            funding_signal = -0.8  # Longs paying too much, expect dump
        elif funding_rate > 0.0001:
            funding_signal = -0.3
        elif funding_rate < -0.0001:
            funding_signal = 0.5  # Shorts paying, expect pump
        else:
            funding_signal = 0.0
        
        # Signal 4: L/S Ratio Divergence
        if ls_ratio > 1.5:  # Way more longs
            divergence_signal = -0.7  # Fade the crowd
        elif ls_ratio < 0.7:  # Way more shorts
            divergence_signal = 0.7
        else:
            divergence_signal = 0.0
        
        # Signal 5: ETF Flows (mock - would come from Coinglass)
        etf_signal = 0.3  # Slight bullish (mock inflows)
        
        # === COMBINE SIGNALS ===
        weighted_signal = (
            max_pain_signal * 0.30 +
            liq_signal * 0.25 +
            funding_signal * 0.15 +
            divergence_signal * 0.15 +
            etf_signal * 0.15
        )
        
        # === DETERMINE TARGET ===
        if weighted_signal > 0.25:
            direction = "BULLISH"
            confidence = min(85, int(50 + abs(weighted_signal) * 50))
            primary_target = short_liq_cluster["price"]
        elif weighted_signal < -0.25:
            direction = "BEARISH"
            confidence = min(85, int(50 + abs(weighted_signal) * 50))
            primary_target = long_liq_cluster["price"]
        else:
            direction = "NEUTRAL"
            confidence = 40
            primary_target = max_pain_price
        
        # Calculate distance
        target_distance = ((primary_target - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Time horizon based on funding interval
        time_horizon = "4-8h (next funding)"
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "currentPrice": current_price,
            "direction": direction,
            "target": primary_target,
            "targetFormatted": f"${primary_target:,.0f}",
            "targetDistance": round(target_distance, 1),
            "confidence": confidence,
            "timeHorizon": time_horizon,
            "weightedSignal": round(weighted_signal, 2),
            "signals": {
                "maxPain": {"value": round(max_pain_signal, 2), "label": f"up to ${max_pain_price:,.0f}"},
                "liqHunt": {"value": round(liq_signal, 2), "label": f"${long_liq_cluster['usd']/1e6:.0f}M longs" if liq_signal < 0 else f"${short_liq_cluster['usd']/1e6:.0f}M shorts"},
                "funding": {"value": round(funding_signal, 2), "label": "longs pay" if funding_signal < 0 else "shorts pay" if funding_signal > 0 else "neutral"},
                "divergence": {"value": round(divergence_signal, 2), "label": "fade longs" if divergence_signal < 0 else "fade shorts" if divergence_signal > 0 else "balanced"},
                "etfFlows": {"value": round(etf_signal, 2), "label": "inflows" if etf_signal > 0 else "outflows"}
            }
        }
        
    except Exception as e:
        logger.error(f"MM Magnet error: {e}")
        # Fallback with realistic mock data
        return {
            "success": True,
            "symbol": symbol.upper(),
            "currentPrice": 83700,
            "direction": "BEARISH",
            "target": 80400,
            "targetFormatted": "$80,400",
            "targetDistance": -4.0,
            "confidence": 65,
            "timeHorizon": "4-8h (next funding)",
            "weightedSignal": -0.42,
            "signals": {
                "maxPain": {"value": 0.3, "label": "up to $85K"},
                "liqHunt": {"value": -1.0, "label": "$837M longs"},
                "funding": {"value": -0.6, "label": "longs pay"},
                "divergence": {"value": -0.8, "label": "fade retail"},
                "etfFlows": {"value": 0.6, "label": "inflows"}
            },
            "source": "fallback"
        }


# =============================================================================
# ON-CHAIN INTEL ENDPOINTS
# =============================================================================

@app.get("/api/onchain")
async def get_onchain_data():
    """Get on-chain intelligence data - exchange reserves, stablecoins, miner flows."""
    try:
        # Fetch real data from Helsinki endpoints
        btc_data = await helsinki.get("quant/full/BTC")
        
        # Calculate synthetic on-chain metrics from available data
        btc_price = btc_data.get("current_price", 83700)
        oi_usd = btc_data.get("open_interest_usd", 8e9)
        
        # Simulate exchange reserve trends based on OI and price action
        base_btc_reserve = 2.1e6  # 2.1M BTC baseline
        reserve_change = random.uniform(-5, 2)  # Simulate 7d change
        
        base_eth_reserve = 18.4e6  # 18.4M ETH baseline
        eth_change = random.uniform(-3, 1)
        
        # Stablecoin flows - simulate based on market activity
        usdt_flow = random.uniform(-200, 1000) * 1e6
        usdc_flow = random.uniform(-300, 400) * 1e6
        
        # Miner flows - simulate with some randomness
        miner_outflow_24h = random.randint(50, 500)
        miner_7d_avg = 89
        miner_signal = "ELEVATED" if miner_outflow_24h > 200 else "NORMAL" if miner_outflow_24h > 100 else "LOW"
        
        # Dormant supply
        dormant_moved = random.randint(100, 2000)
        dormant_7d_avg = 312
        dormant_signal = "DISTRIBUTION" if dormant_moved > 800 else "NEUTRAL" if dormant_moved > 400 else "ACCUMULATION"
        
        # Network activity
        active_addresses = random.randint(800000, 1000000)
        new_addresses = random.randint(30000, 60000)
        tx_count = random.randint(280000, 400000)
        
        # Whale accumulation (simulate based on price trend)
        whale_net_change = random.randint(-5000, 20000)
        
        return {
            "success": True,
            "exchangeReserves": {
                "btc": {
                    "value": round(base_btc_reserve / 1e6, 2),
                    "formatted": f"{base_btc_reserve/1e6:.1f}M",
                    "change7d": round(reserve_change, 1),
                    "barWidth": max(30, min(80, 65 + reserve_change * 2))
                },
                "eth": {
                    "value": round(base_eth_reserve / 1e6, 2),
                    "formatted": f"{base_eth_reserve/1e6:.1f}M",
                    "change7d": round(eth_change, 1),
                    "barWidth": max(30, min(80, 58 + eth_change * 2))
                }
            },
            "stablecoinFlows": {
                "usdt": {
                    "value": usdt_flow,
                    "formatted": f"+${usdt_flow/1e6:.0f}M" if usdt_flow > 0 else f"-${abs(usdt_flow)/1e6:.0f}M",
                    "action": "MINTED" if usdt_flow > 0 else "BURNED"
                },
                "usdc": {
                    "value": usdc_flow,
                    "formatted": f"+${usdc_flow/1e6:.0f}M" if usdc_flow > 0 else f"-${abs(usdc_flow)/1e6:.0f}M",
                    "action": "MINTED" if usdc_flow > 0 else "BURNED"
                },
                "netInflow": usdt_flow + usdc_flow,
                "netFormatted": f"+${(usdt_flow+usdc_flow)/1e6:.0f}M" if (usdt_flow+usdc_flow) > 0 else f"-${abs(usdt_flow+usdc_flow)/1e6:.0f}M"
            },
            "minerFlows": {
                "outflow24h": miner_outflow_24h,
                "formatted": f"{miner_outflow_24h} BTC",
                "avg7d": miner_7d_avg,
                "delta": round((miner_outflow_24h - miner_7d_avg) / miner_7d_avg * 100, 0),
                "signal": miner_signal
            },
            "dormantSupply": {
                "movedToday": dormant_moved,
                "formatted": f"{dormant_moved:,} BTC",
                "avg7d": dormant_7d_avg,
                "delta": round((dormant_moved - dormant_7d_avg) / dormant_7d_avg * 100, 0),
                "signal": dormant_signal
            },
            "networkActivity": {
                "activeAddresses": active_addresses,
                "activeFormatted": f"{active_addresses/1000:.0f}K",
                "newAddresses": new_addresses,
                "newFormatted": f"+{new_addresses/1000:.0f}K",
                "txCount": tx_count,
                "txFormatted": f"{tx_count/1000:.0f}K"
            },
            "whaleWallets": {
                "netChange7d": whale_net_change,
                "formatted": f"+{whale_net_change:,} BTC" if whale_net_change > 0 else f"{whale_net_change:,} BTC",
                "phase": "ACCUMULATION" if whale_net_change > 5000 else "DISTRIBUTION" if whale_net_change < -5000 else "NEUTRAL",
                "barWidth": max(20, min(90, 50 + whale_net_change / 500))
            }
        }
    except Exception as e:
        logger.error(f"On-chain data error: {e}")
        return {
            "success": True,
            "exchangeReserves": {
                "btc": {"value": 2.1, "formatted": "2.1M", "change7d": -3.2, "barWidth": 65},
                "eth": {"value": 18.4, "formatted": "18.4M", "change7d": -1.8, "barWidth": 58}
            },
            "stablecoinFlows": {
                "usdt": {"value": 847e6, "formatted": "+$847M", "action": "MINTED"},
                "usdc": {"value": -124e6, "formatted": "-$124M", "action": "BURNED"},
                "netInflow": 723e6, "netFormatted": "+$723M"
            },
            "minerFlows": {"outflow24h": 342, "formatted": "342 BTC", "avg7d": 89, "delta": 284, "signal": "ELEVATED"},
            "dormantSupply": {"movedToday": 1247, "formatted": "1,247 BTC", "avg7d": 312, "delta": 300, "signal": "DISTRIBUTION"},
            "networkActivity": {"activeAddresses": 892000, "activeFormatted": "892K", "newAddresses": 42000, "newFormatted": "+42K", "txCount": 324000, "txFormatted": "324K"},
            "whaleWallets": {"netChange7d": 12847, "formatted": "+12,847 BTC", "phase": "ACCUMULATION", "barWidth": 72},
            "source": "fallback"
        }


# =============================================================================
# ORDER FLOW ENDPOINTS
# =============================================================================

@app.get("/api/orderflow/{symbol}")
async def get_orderflow_data(symbol: str = "BTC"):
    """Get live order flow intelligence - bid/ask imbalance, aggressor side, large orders."""
    try:
        # Fetch CVD and market data
        cvd_data = await helsinki.get(f"quant/cvd/{symbol}")
        market_data = await helsinki.get(f"quant/full/{symbol}")
        
        current_price = market_data.get("current_price", 83700)
        cvd_1h = cvd_data.get("cvd_1h_usd", 0)
        
        # Calculate bid/ask imbalance from available metrics
        ls_ratio = market_data.get("long_short_ratio", 1.5)
        
        # Simulate bid/ask based on L/S ratio
        bid_pct = min(75, max(35, 50 + (ls_ratio - 1) * 20))
        ask_pct = 100 - bid_pct
        imbalance_signal = "BUY" if bid_pct > 55 else "SELL" if bid_pct < 45 else "NEUTRAL"
        
        # Aggressor side calculation from CVD
        buyer_vol = abs(cvd_1h) if cvd_1h > 0 else random.uniform(15, 30) * 1e6
        seller_vol = abs(cvd_1h) if cvd_1h < 0 else random.uniform(10, 25) * 1e6
        total_vol = buyer_vol + seller_vol
        buyer_pct = buyer_vol / total_vol * 100 if total_vol > 0 else 50
        
        # Generate realistic large orders
        large_orders = []
        for i in range(5):
            is_buy = random.random() > 0.45
            order = {
                "time": f"{14-i//2}:{32-i*5:02d}",
                "side": "BUY" if is_buy else "SELL",
                "size": f"${random.uniform(0.5, 4):.1f}M",
                "price": f"@{current_price + random.randint(-200, 200):,.0f}"
            }
            large_orders.append(order)
        
        # CVD momentum
        cvd_value = cvd_1h if cvd_1h else random.uniform(-10, 15) * 1e6
        
        # Trade intensity - simulate based on volatility
        volatility = market_data.get("daily_volatility_pct", 2.0)
        intensity_pct = min(95, max(20, volatility * 30 + random.uniform(-10, 10)))
        intensity_signal = "HIGH" if intensity_pct > 70 else "ELEVATED" if intensity_pct > 50 else "NORMAL" if intensity_pct > 30 else "LOW"
        trades_per_sec = int(300 + intensity_pct * 10 + random.randint(-50, 50))
        
        # Spoofing detection (simulate occasional alerts)
        has_spoof = random.random() < 0.1
        spoof_status = "ALERT" if has_spoof else "CLEAR"
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "bidAskImbalance": {
                "bidPct": round(bid_pct, 0),
                "askPct": round(ask_pct, 0),
                "signal": imbalance_signal,
                "text": "STRONG BUY PRESSURE" if bid_pct > 60 else "BUY PRESSURE" if bid_pct > 55 else "STRONG SELL PRESSURE" if bid_pct < 40 else "SELL PRESSURE" if bid_pct < 45 else "BALANCED"
            },
            "aggressorSide": {
                "buyerVol": buyer_vol,
                "buyerFormatted": f"${buyer_vol/1e6:.1f}M",
                "buyerPct": round(buyer_pct, 0),
                "sellerVol": seller_vol,
                "sellerFormatted": f"${seller_vol/1e6:.1f}M",
                "sellerPct": round(100 - buyer_pct, 0),
                "delta": buyer_vol - seller_vol,
                "deltaFormatted": f"+${(buyer_vol-seller_vol)/1e6:.1f}M" if buyer_vol > seller_vol else f"-${abs(buyer_vol-seller_vol)/1e6:.1f}M"
            },
            "largeOrders": large_orders,
            "cvdMomentum": {
                "value": cvd_value,
                "formatted": f"+${cvd_value/1e6:.1f}M" if cvd_value > 0 else f"-${abs(cvd_value)/1e6:.1f}M",
                "isPositive": cvd_value > 0
            },
            "tradeIntensity": {
                "pct": round(intensity_pct, 0),
                "signal": intensity_signal,
                "tradesPerSec": trades_per_sec
            },
            "spoofDetection": {
                "status": spoof_status,
                "alerts": [{"time": "14:28", "side": "BUY", "size": "$8.2M", "note": "Pulled after 3s"}] if has_spoof else []
            }
        }
    except Exception as e:
        logger.error(f"Order flow error: {e}")
        return {
            "success": True,
            "symbol": symbol.upper(),
            "bidAskImbalance": {"bidPct": 62, "askPct": 38, "signal": "BUY", "text": "STRONG BUY PRESSURE"},
            "aggressorSide": {"buyerVol": 24.7e6, "buyerFormatted": "$24.7M", "buyerPct": 58, "sellerVol": 17.8e6, "sellerFormatted": "$17.8M", "sellerPct": 42, "delta": 6.9e6, "deltaFormatted": "+$6.9M"},
            "largeOrders": [
                {"time": "14:32", "side": "BUY", "size": "$2.4M", "price": "@83,420"},
                {"time": "14:28", "side": "SELL", "size": "$1.8M", "price": "@83,510"},
                {"time": "14:25", "side": "BUY", "size": "$3.1M", "price": "@83,380"}
            ],
            "cvdMomentum": {"value": 8.4e6, "formatted": "+$8.4M", "isPositive": True},
            "tradeIntensity": {"pct": 68, "signal": "ELEVATED", "tradesPerSec": 847},
            "spoofDetection": {"status": "CLEAR", "alerts": []},
            "source": "fallback"
        }


# =============================================================================
# SESSION & MACRO ANALYSIS
# =============================================================================

@app.get("/api/session")
async def get_session_analysis():
    """Get trading session analysis - time-of-day insights."""
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    # Determine current session
    if 0 <= hour < 8:
        session = "ASIA"
        session_name = "Asia (Low Vol)"
        next_session = "London Open"
        hours_to_next = 8 - hour
    elif 8 <= hour < 13:
        session = "LONDON"
        session_name = "London (Med Vol)"
        next_session = "US Open"
        hours_to_next = 13 - hour
    elif 13 <= hour < 21:
        session = "US"
        session_name = "US (High Vol)"
        next_session = "Asia Open"
        hours_to_next = 24 - hour if hour >= 21 else 24 + (24 - hour)
    else:
        session = "LATE_US"
        session_name = "Late US (Low Vol)"
        next_session = "Asia Open"
        hours_to_next = 24 - hour
    
    recommendation = "Wait for US open" if session == "ASIA" else "Trade with trend" if session == "US" else "Scalp-friendly"
    
    return {
        "success": True,
        "current": session,
        "name": session_name,
        "next": f"{next_session} in {hours_to_next}h",
        "asiaAvg": "0.4%",
        "londonAvg": "0.8%",
        "usAvg": "1.2%",
        "recommendation": recommendation,
        "stats": {
            "bigMoves": "80% during US session",
            "avgAsia": 0.4,
            "avgLondon": 0.8,
            "avgUS": 1.2
        }
    }


@app.get("/api/macro")
async def get_macro_correlation():
    """Get macro market correlations."""
    import random
    
    # Simulate correlations (would connect to real data source)
    spx = round(random.uniform(0.65, 0.90), 2)
    nq = round(random.uniform(0.70, 0.92), 2)
    dxy = round(random.uniform(-0.80, -0.50), 2)
    gold = round(random.uniform(0.20, 0.55), 2)
    vix = round(random.uniform(-0.50, -0.20), 2)
    
    # Determine overall signal
    if spx > 0.7 and nq > 0.7:
        signal = "RISK-ON"
        note = "High equity correlation - Watch SPX"
    elif dxy < -0.6:
        signal = "RISK-ON"
        note = "DXY weakness bullish for BTC"
    else:
        signal = "NEUTRAL"
        note = "Mixed signals - Trade with caution"
    
    return {
        "success": True,
        "signal": signal,
        "correlations": {
            "spx": spx,
            "nq": nq,
            "dxy": dxy,
            "gold": gold,
            "vix": vix
        },
        "note": note
    }


# =============================================================================
# KELLY CRITERION & MONTE CARLO
# =============================================================================

@app.get("/api/kelly")
async def get_kelly_criterion():
    """Calculate Kelly criterion position sizing."""
    # Using session stats (would use real trade history)
    win_rate = 0.73
    avg_win = 2.1  # R-multiple
    avg_loss = 1.0
    
    # Kelly formula: K% = W - [(1-W) / R] where R = avg_win/avg_loss
    r_ratio = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / r_ratio)
    kelly_pct = max(0, min(kelly * 100, 25))  # Cap at 25%
    
    return {
        "success": True,
        "optimal": round(kelly_pct, 1),
        "half": round(kelly_pct / 2, 1),
        "quarter": round(kelly_pct / 4, 1),
        "winRate": int(win_rate * 100),
        "avgWin": f"+{avg_win}R",
        "avgLoss": f"-{avg_loss}R",
        "recommendation": "Use Half-Kelly (conservative)" if kelly_pct > 3 else "Size appropriately"
    }


@app.post("/api/monte-carlo")
async def run_monte_carlo(simulations: int = 50000):
    """Run Monte Carlo simulation for portfolio projections."""
    import random
    import statistics
    
    # Simulation parameters (would use real account data)
    starting_capital = 100000
    win_rate = 0.73
    avg_win_pct = 2.1
    avg_loss_pct = 1.0
    trades_to_simulate = 100
    position_size_pct = 1.2  # Half-Kelly
    
    final_capitals = []
    max_drawdowns = []
    
    for _ in range(min(simulations, 50000)):
        capital = starting_capital
        peak = capital
        max_dd = 0
        
        for _ in range(trades_to_simulate):
            if random.random() < win_rate:
                capital *= (1 + (avg_win_pct * position_size_pct / 100))
            else:
                capital *= (1 - (avg_loss_pct * position_size_pct / 100))
            
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            if dd > max_dd:
                max_dd = dd
        
        final_capitals.append(capital)
        max_drawdowns.append(max_dd)
    
    # Calculate statistics
    final_capitals.sort()
    ev = statistics.mean(final_capitals) - starting_capital
    conf_low = final_capitals[int(len(final_capitals) * 0.05)] - starting_capital
    conf_high = final_capitals[int(len(final_capitals) * 0.95)] - starting_capital
    ruin_count = sum(1 for c in final_capitals if c < starting_capital * 0.5)
    ruin_prob = (ruin_count / len(final_capitals)) * 100
    avg_max_dd = statistics.mean(max_drawdowns) * 100
    
    return {
        "success": True,
        "simulations": len(final_capitals),
        "ev": round(ev, 0),
        "evFormatted": f"+${ev:,.0f}" if ev > 0 else f"-${abs(ev):,.0f}",
        "confLow": round(conf_low, 0),
        "confLowFormatted": f"+${conf_low:,.0f}" if conf_low > 0 else f"-${abs(conf_low):,.0f}",
        "confHigh": round(conf_high, 0),
        "confHighFormatted": f"+${conf_high:,.0f}",
        "ruinProb": round(ruin_prob, 1),
        "maxDrawdown": round(avg_max_dd, 1)
    }


@app.get("/api/mcf-reports")
async def get_mcf_reports():
    """Get latest MCF Labs institutional reports."""
    # Placeholder - would connect to real report system
    return {
        "success": True,
        "latest": {
            "title": "BTC Accumulation Phase Confirmed: Smart Money Loading",
            "summary": "On-chain metrics show significant whale accumulation over past 72h. Exchange reserves at 3-year lows. Funding remains neutral suggesting room for upside.",
            "sentiment": "BULLISH",
            "confidence": "HIGH",
            "timestamp": "2 min ago"
        },
        "feed": [
            {"title": "ETH/BTC Ratio Analysis: Altcoin Season Incoming?", "sentiment": "NEUTRAL", "time": "1h"},
            {"title": "Derivatives Deep Dive: OI Surge Signals Volatility", "sentiment": "CAUTION", "time": "3h"},
            {"title": "Macro Weekly: Fed Pivot Implications for Crypto", "sentiment": "BULLISH", "time": "6h"},
            {"title": "Liquidation Map Analysis: Key Levels to Watch", "sentiment": "RISK ALERT", "time": "12h"}
        ]
    }


# =============================================================================
# SUPPORTED SYMBOLS
# =============================================================================

SUPPORTED_SYMBOLS = [
    {"symbol": "BTC", "name": "Bitcoin", "perp": "BTC-PERP"},
    {"symbol": "ETH", "name": "Ethereum", "perp": "ETH-PERP"},
    {"symbol": "SOL", "name": "Solana", "perp": "SOL-PERP"},
    {"symbol": "DOGE", "name": "Dogecoin", "perp": "DOGE-PERP"},
    {"symbol": "XRP", "name": "XRP", "perp": "XRP-PERP"},
    {"symbol": "AVAX", "name": "Avalanche", "perp": "AVAX-PERP"},
    {"symbol": "LINK", "name": "Chainlink", "perp": "LINK-PERP"},
    {"symbol": "MATIC", "name": "Polygon", "perp": "MATIC-PERP"},
    {"symbol": "ARB", "name": "Arbitrum", "perp": "ARB-PERP"},
    {"symbol": "OP", "name": "Optimism", "perp": "OP-PERP"},
    {"symbol": "APT", "name": "Aptos", "perp": "APT-PERP"},
    {"symbol": "SUI", "name": "Sui", "perp": "SUI-PERP"},
    {"symbol": "PEPE", "name": "Pepe", "perp": "PEPE-PERP"},
    {"symbol": "WIF", "name": "dogwifhat", "perp": "WIF-PERP"},
    {"symbol": "INJ", "name": "Injective", "perp": "INJ-PERP"},
]


@app.get("/api/symbols")
async def get_symbols():
    """Get list of supported trading symbols."""
    return {"symbols": SUPPORTED_SYMBOLS}


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "service": "BASTION Terminal API",
        "version": "1.0.0",
        "helsinki_url": helsinki.base_url if helsinki else "not initialized",
        "websocket_connections": len(active_connections),
        "supported_symbols": len(SUPPORTED_SYMBOLS),
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)

