<img width="2145" height="1310" alt="Mask group (14)" src="https://github.com/user-attachments/assets/878d4b09-22de-4b1d-b92e-e90189bbac3b" />

# 🏰 BASTION

**Proactive Risk Management Infrastructure**

BASTION provides structural stop-losses, dynamic take-profit targets, and adaptive position sizing for institutional and retail traders.

---

## 🎯 Features

- **Structural Stops** - Placed at actual support/resistance, not arbitrary percentages
- **Dynamic Targets** - Based on market structure, not fixed R multiples
- **Multi-Tier Defense** - Primary, secondary, and safety-net stops
- **Guarding Line** - Structural trailing stops for swing trades
- **Adaptive Sizing** - Volatility-adjusted position sizing
- **Multi-Shot System** - Re-entry management with capped total risk

---

## 🚀 Quick Start

### Installation

```bash
cd C:\Users\Banke\MCF-Project\bastion
pip install -r requirements.txt
```

### Run API Server

```bash
python run.py
```

Server starts on `http://localhost:8001`

### Test API

```bash
curl http://localhost:8001/health
```

### Calculate Risk Levels

```bash
curl -X POST http://localhost:8001/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "entry_price": 95000,
    "direction": "long",
    "timeframe": "4h",
    "account_balance": 100000,
    "risk_per_trade_pct": 1.0
  }'
```

---

## 📊 Web Calculator

Open `web/index.html` in your browser for the visual calculator interface.

Or visit: `http://localhost:8001` (when server is running with static file serving enabled)

---

## 📚 Documentation

- [API Documentation](docs/API.md)
- [Usage Examples](docs/EXAMPLES.md)
- [Build Instructions](../BASTION_BUILD_INSTRUCTIONS.md)

---

## 🏗️ Architecture

```
bastion/
├── api/           # FastAPI backend
├── core/          # Risk calculation engine
├── data/          # Market data fetching
├── web/           # Simple calculator UI
├── tests/         # Unit tests
└── docs/          # Documentation
```

---

## 🌐 Data Sources

- **Primary:** Helsinki VM (77.42.29.188:5000, :5002)
- **Fallback:** Binance API (direct)

---

## 🔧 Configuration

Create `.env` file:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8001

# Data Sources
HELSINKI_SPOT=http://77.42.29.188:5000
HELSINKI_QUANT=http://77.42.29.188:5002

# Fallback
BINANCE_API=https://api.binance.com
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=bastion tests/
```

---

## 📈 Example Response

```json
{
  "symbol": "BTCUSDT",
  "entry_price": 95000,
  "stops": [
    {
      "type": "primary",
      "price": 92500,
      "distance_pct": 2.6,
      "reason": "Below support at 92800"
    }
  ],
  "targets": [
    {
      "price": 98000,
      "exit_percentage": 33,
      "reason": "Resistance level (R:R 2.5)"
    }
  ],
  "position_size": 0.421,
  "risk_amount": 1000,
  "risk_reward_ratio": 2.5
}
```

---

## 🎨 Brand

- **Colors:** Deep Red (#8B0000), Silver (#C0C0C0), Black (#000000)
- **Style:** Dark, professional, institutional
- **Aesthetic:** Military precision meets modern fintech

---

## 📝 License

Proprietary - All rights reserved.

---

## 🤝 Support

For questions and support, contact: [support email]

**Built with precision. Managed with structure.**

