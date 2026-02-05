"""
BASTION API - Vercel Compatible
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import httpx
import time
import random

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bastion_path = Path(__file__).parent.parent
price_cache = {}

@app.get("/")
async def root():
    html_path = bastion_path / "generated-page.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>BASTION</h1>")

@app.get("/api/test")
def test():
    return {"status": "ok", "message": "BASTION API Live!"}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/price/{symbol}")
async def get_price(symbol: str = "BTC"):
    sym = symbol.upper()
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            kraken_sym = "XXBTZUSD" if sym == "BTC" else "XETHZUSD" if sym == "ETH" else f"{sym}USD"
            res = await client.get(f"https://api.kraken.com/0/public/Ticker?pair={kraken_sym}")
            data = res.json()
            if data.get("result"):
                for key, val in data["result"].items():
                    return {"symbol": sym, "price": float(val["c"][0]), "change_24h": 0, "source": "kraken"}
        except:
            pass
    return {"symbol": sym, "price": 97000 if sym == "BTC" else 3500, "change_24h": 0, "source": "fallback"}

@app.get("/api/klines/{symbol}")
async def get_klines(symbol: str = "BTC", interval: str = "15m", limit: int = 100):
    sym = symbol.upper()
    candles = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            kraken_sym = "XXBTZUSD" if sym == "BTC" else "XETHZUSD" if sym == "ETH" else f"{sym}USD"
            interval_mins = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}.get(interval, 15)
            res = await client.get(f"https://api.kraken.com/0/public/OHLC?pair={kraken_sym}&interval={interval_mins}")
            data = res.json()
            if data.get("result"):
                for key, values in data["result"].items():
                    if key != "last" and isinstance(values, list):
                        for k in values[-limit:]:
                            candles.append({"time": int(k[0]), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[6])})
                        break
        except:
            pass
    if not candles:
        base = 97000 if sym == "BTC" else 3500
        now = int(time.time())
        for i in range(limit):
            t = now - (limit-i) * 900
            candles.append({"time": t, "open": base, "high": base*1.001, "low": base*0.999, "close": base, "volume": 100})
    return {"symbol": sym, "interval": interval, "candles": candles}

@app.get("/api/positions")
def positions():
    return {"positions": [{"id": "1", "symbol": "BTC-PERP", "direction": "long", "entry_price": 95000, "current_price": 97000, "size": 0.5, "pnl_pct": 2.1}]}

@app.get("/api/session/stats")
def stats():
    return {"session_pnl": "+$2,450", "win_rate": "68%", "avg_r": "+1.2R", "active_positions": 1}

@app.get("/api/alerts")
def alerts():
    return {"alerts": []}

@app.get("/api/news")
def news():
    return {"news": [{"title": "BTC holds strong", "source": "CoinDesk", "sentiment": "BULLISH", "timeAgo": "1h"}]}

@app.get("/api/symbols")
def symbols():
    return {"symbols": ["BTC", "ETH", "SOL"]}

@app.get("/api/fear-greed")
def fg():
    return {"value": 65, "classification": "Greed"}

@app.get("/api/funding")
def funding():
    return {"BTC": {"rate": 0.01}, "ETH": {"rate": 0.008}}

@app.get("/api/{path:path}")
def catch_all(path: str):
    return {"endpoint": path, "data": {}}
