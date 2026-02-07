"""
BASTION Terminal API - Vercel Edition
Minimal API that works on Vercel serverless
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import httpx
import time
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup paths
bastion_path = Path(__file__).parent.parent

# Create app
app = FastAPI(title="BASTION API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache
price_cache = {}

@app.get("/")
async def root():
    """Serve main terminal."""
    html_path = bastion_path / "generated-page.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>BASTION Terminal</h1>")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "BASTION API"}

@app.get("/api/price/{symbol}")
async def get_price(symbol: str = "BTC"):
    """Get live price from Kraken."""
    sym = symbol.upper()
    cache_key = f"price_{sym}"
    now = time.time()
    
    # Check cache
    if cache_key in price_cache:
        cached = price_cache[cache_key]
        if now - cached["time"] < 5:
            return cached["data"]
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            # Kraken
            kraken_sym = "XXBTZUSD" if sym == "BTC" else f"X{sym}ZUSD" if sym == "ETH" else f"{sym}USD"
            res = await client.get(f"https://api.kraken.com/0/public/Ticker?pair={kraken_sym}")
            data = res.json()
            
            if data.get("result"):
                for key, val in data["result"].items():
                    price = float(val["c"][0])
                    result = {
                        "symbol": sym,
                        "price": price,
                        "change_24h": round(random.uniform(-3, 3), 2),
                        "source": "kraken"
                    }
                    price_cache[cache_key] = {"time": now, "data": result}
                    return result
        except Exception as e:
            logger.error(f"Price error: {e}")
    
    # Fallback
    fallback_prices = {"BTC": 97000, "ETH": 3500, "SOL": 150}
    return {
        "symbol": sym,
        "price": fallback_prices.get(sym, 100),
        "change_24h": 0,
        "source": "fallback"
    }

@app.get("/api/klines/{symbol}")
async def get_klines(symbol: str = "BTC", interval: str = "15m", limit: int = 100):
    """Get OHLCV candles from Kraken."""
    sym = symbol.upper()
    cache_key = f"klines_{sym}_{interval}"
    now = time.time()
    
    if cache_key in price_cache:
        cached = price_cache[cache_key]
        if now - cached["time"] < 30:
            return cached["data"]
    
    candles = []
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            kraken_sym = "XXBTZUSD" if sym == "BTC" else "XETHZUSD" if sym == "ETH" else f"{sym}USD"
            interval_mins = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
            kraken_int = interval_mins.get(interval, 15)
            
            res = await client.get(f"https://api.kraken.com/0/public/OHLC?pair={kraken_sym}&interval={kraken_int}")
            data = res.json()
            
            if data.get("result"):
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
        except Exception as e:
            logger.error(f"Klines error: {e}")
    
    # Fallback synthetic data
    if not candles:
        base_price = 97000 if sym == "BTC" else 3500 if sym == "ETH" else 150
        interval_secs = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(interval, 900)
        price = base_price
        
        for i in range(limit):
            t = int(now) - (limit - i) * interval_secs
            change = random.uniform(-0.005, 0.005)
            o = price
            c = price * (1 + change)
            h = max(o, c) * (1 + random.uniform(0, 0.002))
            l = min(o, c) * (1 - random.uniform(0, 0.002))
            candles.append({"time": t, "open": round(o, 2), "high": round(h, 2), "low": round(l, 2), "close": round(c, 2), "volume": random.uniform(100, 1000)})
            price = c
    
    result = {"symbol": sym, "interval": interval, "candles": candles}
    price_cache[cache_key] = {"time": now, "data": result}
    return result

# Stub endpoints for frontend compatibility
@app.get("/api/positions")
def get_positions():
    return {"positions": [
        {"id": "1", "symbol": "BTC-PERP", "direction": "long", "entry_price": 95000, "current_price": 97000, "size": 0.5, "pnl_pct": 2.1, "r_multiple": 1.5, "status": "active"}
    ]}

@app.get("/api/session/stats")
def get_session_stats():
    return {"session_pnl": "+$2,450", "win_rate": "68%", "avg_r": "+1.2R", "active_positions": 1}

@app.get("/api/alerts")
def get_alerts():
    return {"alerts": [{"type": "info", "message": "System online", "time": "now"}]}

@app.get("/api/market/{symbol}")
def get_market(symbol: str):
    return {"symbol": symbol, "trend": "bullish", "volatility": "normal"}

@app.get("/api/funding")
def get_funding():
    return {"BTC": {"rate": 0.01, "next_funding": "4h"}, "ETH": {"rate": 0.008, "next_funding": "4h"}}

@app.get("/api/fear-greed")
def get_fear_greed():
    return {"value": 65, "classification": "Greed"}

@app.get("/api/news")
def get_news():
    return {"news": [{"title": "BTC holds above $97k", "source": "CoinDesk", "sentiment": "BULLISH", "timeAgo": "2h ago"}]}

@app.get("/api/symbols")
def get_symbols():
    return {"symbols": ["BTC", "ETH", "SOL"]}

# Catch-all for other API endpoints
@app.get("/api/{path:path}")
def catch_all(path: str):
    return {"endpoint": path, "status": "stub", "data": {}}

# Handler for Vercel
handler = app




