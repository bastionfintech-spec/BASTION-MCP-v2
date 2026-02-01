# 🧠 BASTION AI Integration (Powered by IROS Infrastructure)

**Complete plug-and-play implementation of IROS for Bastion Terminal**

---

## 🎉 WHAT'S PRE-CONFIGURED (JUST WORKS!)

| Service | Status | API Key |
|---------|--------|---------|
| Helsinki VM | ✅ Ready | No key needed (free) |
| Coinglass | ✅ Ready | `03e5a43afaa4489384cb935b9b2ea16b` ($299/mo) |
| Whale Alert | ✅ Ready | `OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ` ($29.95/mo) |
| vLLM Auth | ✅ Ready | `5c37b5e8...` (pre-set) |

**ONLY THING YOU NEED:** Set `BASTION_MODEL_URL` to your Cloudflare tunnel URL!

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BASTION TERMINAL                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   FRONTEND   │────▶│   BACKEND    │────▶│  BASTION AI  │                │
│  │  (Your UI)   │     │  (FastAPI)   │     │  (32B LLM)   │                │
│  └──────────────┘     └──────┬───────┘     └──────────────┘                │
│                              │                     │                        │
│              ┌───────────────┼───────────────┐     │                        │
│              ▼               ▼               ▼     ▼                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ HELSINKI VM  │ │  COINGLASS   │ │ WHALE ALERT  │ │  VAST.AI GPU │       │
│  │   (Free)     │ │  (Premium)   │ │  (Premium)   │ │   CLUSTER    │       │
│  │              │ │              │ │              │ │              │       │
│  │ • 33 quant   │ │ • Liquidation│ │ • Whale txs  │ │ • vLLM       │       │
│  │   endpoints  │ │ • OI         │ │ • Real-time  │ │ • Qwen 32B   │       │
│  │ • Binance    │ │ • Funding    │ │ • WebSocket  │ │ • Cloudflare │       │
│  │ • Deribit    │ │ • L/S Ratio  │ │              │ │   tunnel     │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
│       FREE           $299/mo         $29.95/mo         ~$27/mo             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
iros_integration/
├── README.md                 # This file
├── env_example.txt           # Environment variables template
├── requirements.txt          # Python dependencies
├── __init__.py               # Main exports
│
├── config/
│   ├── __init__.py
│   └── settings.py           # All API keys pre-configured!
│
├── services/
│   ├── __init__.py
│   ├── bastion_ai.py         # Main AI service (32B LLM)
│   ├── helsinki.py           # Helsinki VM data fetcher (33 endpoints - FREE)
│   ├── coinglass.py          # Coinglass client ($299/mo - KEY INCLUDED!)
│   ├── whale_alert.py        # Whale Alert client ($29.95/mo - KEY INCLUDED!)
│   └── query_processor.py    # Query context extraction
│
└── endpoints/                # 📚 FULL ENDPOINT DOCUMENTATION
    ├── ALL_HELSINKI_ENDPOINTS.md   # 33 quant endpoints
    ├── COINGLASS_ENDPOINTS.md      # Liquidation, OI, funding, L/S
    └── WHALE_ALERT_ENDPOINTS.md    # Whale transactions, streaming
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r iros_integration/requirements.txt
```

### 2. Set Your GPU Cluster URL (ONLY REQUIRED STEP!)

```bash
# Create .env file with just this:
BASTION_MODEL_URL=https://your-tunnel.trycloudflare.com
```

**That's it!** Everything else (Helsinki, Coinglass, Whale Alert, vLLM API key) is pre-configured in `settings.py`.

### Optional: Override Defaults

If you need to change any default values:

```bash
# Helsinki VM - Only if server moves
HELSINKI_VM_URL=http://77.42.29.188:5002

# Premium APIs - Already configured with your paid keys!
# COINGLASS_API_KEY=03e5a43afaa4489384cb935b9b2ea16b
# WHALE_ALERT_API_KEY=OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ
```

### 3. Use in Your Code

```python
from iros_integration.services.bastion_ai import BastionAI

# Initialize
bastion = BastionAI()

# Process a query
result = await bastion.process_query(
    query="Should I long BTC at $97,000 with $50K?",
    user_context={"capital": 50000, "timeframe": "swing"}
)

print(result["response"])
print(f"Data sources: {result['data_sources']}")
print(f"Latency: {result['latency']}ms")
```

---

## 🔌 API Endpoints

### Helsinki VM (FREE - 33 Endpoints)

**Base URL:** `http://77.42.29.188:5002`

#### Symbol-Specific Endpoints (replace `{symbol}` with BTC, ETH, SOL, etc.)

| Category | Endpoint | Data Returned |
|----------|----------|---------------|
| **Order Flow** | `/quant/cvd/{symbol}` | CVD 1h, 4h, divergence, signal |
| **Order Flow** | `/quant/orderbook/{symbol}` | Bid/ask imbalance, top levels |
| **Order Flow** | `/quant/large-trades/{symbol}` | Whale transactions, buy/sell ratio |
| **Order Flow** | `/quant/smart-money/{symbol}` | Institutional flow indicators |
| **Order Flow** | `/quant/whale-flow/{symbol}` | Large wallet movements |
| **Derivatives** | `/quant/basis/{symbol}` | Spot/futures basis, arb yield |
| **Derivatives** | `/quant/open-interest/{symbol}` | OI values, changes |
| **Derivatives** | `/quant/greeks/{symbol}` | ATM IV, put/call ratio |
| **Derivatives** | `/quant/liquidation-map/{symbol}` | Liquidation clusters |
| **Derivatives** | `/quant/liquidation-estimate/{symbol}` | Estimated cascade zones |
| **Derivatives** | `/quant/options-iv/{symbol}` | Options implied volatility |
| **Volatility** | `/quant/iv-rv-spread/{symbol}` | IV vs RV spread |
| **Volatility** | `/quant/volatility/{symbol}` | Regime, percentile |
| **Technical** | `/quant/vwap/{symbol}` | VWAP, ±1σ bands |
| **Technical** | `/quant/momentum/{symbol}` | ROC, RSI, ATR, score |
| **Technical** | `/quant/mean-reversion/{symbol}` | Z-scores, Bollinger |
| **Technical** | `/quant/drawdown/{symbol}` | Current/max DD, ATH |
| **Full Context** | `/quant/full/{symbol}` | ALL data in one call! |

#### Static Endpoints (no symbol needed)

| Category | Endpoint | Data Returned |
|----------|----------|---------------|
| **Derivatives** | `/derivatives/funding` | Funding rates all exchanges |
| **Derivatives** | `/derivatives/oi` | Open interest breakdown |
| **Derivatives** | `/derivatives/long-short` | L/S ratios by exchange |
| **Derivatives** | `/quant/funding-arb` | Cross-exchange arb opportunities |
| **Macro** | `/quant/dominance` | BTC/ETH dominance, alt season |
| **Macro** | `/quant/defi-tvl` | TVL by chain/protocol |
| **Macro** | `/quant/gas` | ETH gwei, BTC sat/vB |
| **Macro** | `/quant/stablecoin-supply` | USDT, USDC, DAI supplies |
| **Sentiment** | `/sentiment/fear-greed` | Fear & Greed Index |
| **Options** | `/options/skew` | Options skew data |

---

## 🖥️ GPU Cluster Configuration

### Current Production Setup (INSANE VALUE!)

```
Provider: Vast.ai
Instance: 4x NVIDIA RTX 5090 (32GB each = 128GB total!)
Cost: $0.038/hr = ~$27/month if running 24/7
CPU: AMD EPYC 9B14 (96-core)
RAM: 386.8 GB
Network: 4.3 Gbps download
```

### vLLM Startup Command

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 18000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --api-key YOUR_API_KEY
```

### Cloudflare Tunnel Setup

Since Vast.ai IPs change, expose via Cloudflare tunnel:

```bash
# In Vast.ai instance
cloudflared tunnel --url http://localhost:18000
# Copy the generated URL to BASTION_MODEL_URL
```

---

## 🧠 Model Specifications

| Property | Value |
|----------|-------|
| **Model ID** | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` |
| **Parameters** | 32 billion |
| **Quantization** | AWQ 4-bit |
| **Context Window** | 8,192 tokens |
| **Training** | 667 curated quant trading examples |
| **Fine-tuning** | LoRA (rank 64, alpha 128) |

---

## 📊 System Prompt Template

The AI uses a dynamic system prompt with:

1. **User Context** - Extracted from query:
   - Capital amount ($10K, $50K, etc.)
   - Timeframe (scalp, day, swing, position, longterm)
   - Trade type (spot, futures, leverage)
   - Risk tolerance (conservative, moderate, aggressive)

2. **Market Context** - Injected from Helsinki:
   - Current price
   - Smart money bias
   - Funding rates
   - Liquidation levels
   - Volatility regime
   - Fear/Greed

3. **Response Format** - Institutional structure:
   - Key Structural Levels
   - Entry Setup (Test → Break → Retest)
   - Trading Scenarios (Bull/Bear)
   - Risk Shield Position Sizing
   - Verdict with confidence %

---

## 💰 Cost Breakdown

| Service | Cost | Notes |
|---------|------|-------|
| Helsinki VM | FREE | 33 endpoints, unlimited requests |
| Vast.ai GPU | ~$27/mo | At $0.038/hr 24/7 (exceptional rate!) |
| Coinglass | $299/mo | Optional - premium liquidation data |
| Whale Alert | $29.95/mo | Optional - real-time whale txs |
| **Total (Minimum)** | **~$27/mo** | Just GPU + free Helsinki! |
| **Total (Full Suite)** | **~$356/mo** | All premium features |

---

## 🔐 API Keys Reference

| Service | Status | Default Value | Override With |
|---------|--------|---------------|---------------|
| Helsinki VM | ✅ Free | N/A | N/A |
| Vast.ai vLLM | ✅ Pre-set | `5c37b5e8e6c2480813aa0cfd4de5c903544b7a000bff729e1c99d9b4538eb34d` | `BASTION_MODEL_API_KEY` |
| Coinglass | ✅ Pre-set | `03e5a43afaa4489384cb935b9b2ea16b` | `COINGLASS_API_KEY` |
| Whale Alert | ✅ Pre-set | `OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ` | `WHALE_ALERT_API_KEY` |

**All premium API keys are already configured. They're the same keys you use in IROS - shared infrastructure!**

---

## 🧪 Testing

### Test Helsinki VM

```bash
curl "http://77.42.29.188:5002/quant/full/BTC"
```

### Test GPU Model

```bash
curl -X POST https://your-tunnel.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

### Test Python Integration

```python
from iros_integration.services.helsinki import HelsinkiClient

client = HelsinkiClient()
data = await client.fetch_full_data("BTC")
print(data)
```

---

## 🔄 Upgrading

### Model Upgrade Path

| Model | VRAM Required | Performance |
|-------|---------------|-------------|
| Current: Qwen2.5-32B-AWQ | 24GB | Good |
| Qwen2.5-72B-AWQ | 48GB | Better reasoning |
| DeepSeek-V3-AWQ | 48GB | Excellent code |
| Custom Fine-tuned 72B | 48GB+ | Best for finance |

### GPU Scaling

| Users | Recommended Setup | Cost |
|-------|-------------------|------|
| 1-5 | 1x RTX 4090 | $12/day |
| 10-20 | 2x RTX 4090 | $24/day |
| 50+ | 4x RTX 5090 | $60/day |
| Enterprise | 8x H100 | $600/day |

---

## 📞 Support

For issues with:
- **Helsinki VM**: Check if `http://77.42.29.188:5002/health` responds
- **GPU Cluster**: Verify Vast.ai instance is running, check Cloudflare tunnel URL
- **API Keys**: Ensure all env vars are set correctly

---

**Built for Bastion Terminal - Institutional-grade crypto intelligence 🏦**

