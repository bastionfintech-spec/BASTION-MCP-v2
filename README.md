<img width="2145" height="1310" alt="Mask group (14)" src="https://github.com/user-attachments/assets/4a9bca8d-2884-4e45-a407-e77ce228c7c3" />
# BASTION - Risk Management Engine

A professional-grade cryptocurrency risk management system with real-time price feeds, multi-shot entry management, and dynamic stop-loss strategies.

![BASTION](https://img.shields.io/badge/BASTION-Risk%20Engine-crimson)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)

## Features

### 🎯 Risk Engine
- **Strategy-Agnostic** - Provides market context without judging trade quality
- **Multi-Tier Stops** - Structural, Safety Net, and Trailing Guard levels
- **R-Multiple Targets** - Automatic target calculation at 2R, 3R, 5R
- **Position Sizing** - ATR-adjusted sizing based on risk budget

### 📊 Live Data Integration
- **Helsinki VM** - Primary data source for real-time prices and OHLCV
- **Binance Fallback** - Automatic failover for reliability
- **Order Flow Data** - CVD, large trades, and liquidity zones

### 💹 Trade Manager UI
- **Real-Time Charts** - TradingView Lightweight Charts integration
- **Multi-Shot Entries** - 50/30/20% risk allocation across 3 shots
- **Live P&L Tracking** - Real-time unrealized and realized P&L
- **Manual Guard Activation** - Lock profits when you decide

## 🚀 QUICK START - TERMINAL DASHBOARD

### One-Command Launch
```bash
git clone https://github.com/LUGIAAAAA/Godterminal.git
cd Godterminal
pip install -r requirements.txt
python api/terminal_api.py
```

### Open the Terminal
**http://localhost:8888** - Full institutional trading dashboard with live data

### What You Get (All Live & Updating)
| Column | Panels |
|--------|--------|
| **MARKET PULSE** | BTC Price, Fear & Greed, Funding Rates, Open Interest, Liquidations |
| **CHART** | 15m Candlesticks with Session Markers (Kraken data) |
| **INTELLIGENCE** | MM Magnet, Volatility Regime, Time-of-Day, Macro Sync |
| **ON-CHAIN** | Exchange Reserves, Stablecoin Flows, Miner Outflows, Whale Wallets |
| **ORDER FLOW** | Bid/Ask Imbalance, Aggressor Side, CVD, Large Orders, Spoof Detection |
| **RISK SIM** | Kelly Criterion, Monte Carlo Simulator, Social Sentiment |
| **ALERTS** | Whale Alerts, Liquidation Warnings, Price Alerts |
| **MCF LABS** | Institutional Reports, Neural Assistant |

### API Keys (Pre-Configured)
All API keys are hardcoded in `iros_integration/config/settings.py`:
- ✅ Helsinki VM (free quant data)
- ✅ Coinglass Premium (liquidations, OI, funding)
- ✅ Whale Alert Premium (large transactions)
- ✅ Kraken/Coinbase (chart data - no keys needed)

---

## Legacy Risk Engine

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Run the risk engine server
python run.py
```

### Access the UI

- **Risk Calculator**: http://localhost:8001/app/index.html
- **Trade Manager**: http://localhost:8001/app/trade-manager.html
- **API Docs**: http://localhost:8001/docs

## API Endpoints

### Health Check
```
GET /health
```

### Calculate Risk Levels
```
POST /calculate
{
    "symbol": "BTCUSDT",
    "direction": "long",
    "entry_price": 91000,
    "stop_price": 89000,
    "account_balance": 100000,
    "risk_per_trade_pct": 2.0
}
```

### Live Price
```
GET /price/{symbol}
```

### OHLCV Bars
```
GET /bars/{symbol}?timeframe=4h&limit=100
```

### Session Management
```
POST /session/create
POST /session/{id}/shot
GET /session/{id}
DELETE /session/{id}
```

## Architecture

```
BASTION/
├── api/
│   ├── server.py          # FastAPI application
│   ├── models.py           # Pydantic request/response models
│   └── session_routes.py   # Session management endpoints
├── core/
│   ├── risk_engine.py      # Main risk calculation engine
│   ├── session.py          # Trade session management
│   └── adaptive_budget.py  # Multi-shot allocation logic
├── data/
│   ├── fetcher.py          # Data fetching utilities
│   └── live_feed.py        # Real-time price feeds
├── web/
│   ├── index.html          # Risk calculator UI
│   ├── session.html        # Session tracker UI
│   └── trade-manager.html  # Dynamic trade manager
└── run.py                  # Application entry point
```

## Risk Management Philosophy

### Multi-Tier Stop System
1. **Primary Stop** - Structural support minus ATR buffer
2. **Safety Net** - Maximum 5% loss from entry
3. **Trailing Guard** - Manual activation to lock profits (2% buffer)

### Multi-Shot Entries
- **Shot 1 (50%)** - Initial entry on setup confirmation
- **Shot 2 (30%)** - Add on support bounce
- **Shot 3 (20%)** - Add on breakout confirmation

### R-Multiple Targets
- **T1 (2R)** - Take 33% profit
- **T2 (3R)** - Take 33% profit
- **T3 (5R)** - Exit remaining via guard

## Data Sources

### Helsinki VM (Primary)
- Endpoint: `http://77.42.29.188:5000` (Spot Data)
- Endpoint: `http://77.42.29.188:5002` (Quant Data)

### Binance (Fallback)
- Endpoint: `https://api.binance.com`

## Configuration

Default settings in the engine:
- Risk per trade: 2% of account
- ATR period: 14
- Safety net: 5% from entry
- Guard buffer: 2% from current price

## License

MIT License - See LICENSE for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**BASTION** - *Protecting your capital, one trade at a time.*
