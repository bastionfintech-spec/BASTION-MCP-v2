# 🏰 BASTION COMPLETE SYSTEM CONTEXT - FOR CLAUDE ULTRA

**Document Version:** 2.0 - COMPLETE CONTEXT  
**Last Updated:** February 10, 2026  
**User:** bastionfintech@gmail.com (Banke)  
**Status:** GPU Cluster ACTIVE, vLLM Deployment NEEDED

---

# PART 1: WHAT IS BASTION?

## Overview

BASTION is an **institutional-grade crypto trading terminal** that combines:

1. **Live Market Data** from premium sources (Helsinki VM, Coinglass, Whale Alert)
2. **AI-Powered Analysis** via a fine-tuned 32B parameter LLM running on 4x RTX 5090 GPUs
3. **Autonomous Risk Intelligence** - AI that can monitor and manage live trading positions
4. **Real-Time Alerts** - Push notifications via Telegram and in-app for whale movements, liquidations, etc.
5. **Exchange Integration** - Connect Bitunix, Bybit, OKX, Binance, etc. for live position tracking

## The Problem BASTION Solves

Retail traders lack access to institutional-grade tools. BASTION provides:
- **Structural stops** placed at actual support/resistance, not arbitrary percentages
- **Dynamic targets** based on market structure, not fixed R multiples  
- **Multi-tier defense** - Primary, secondary, and safety-net stops
- **AI analysis** that combines 33+ data endpoints for comprehensive market views
- **Autonomous position management** - AI monitors positions 24/7 and executes exits

## Target Users
- Crypto futures/perps traders
- Users who want institutional-grade analysis without institutional costs
- Traders who want AI to help manage risk and execute trades

---

# PART 2: SYSTEM ARCHITECTURE

## Frontend (`generated-page.html` - 6500+ lines)

The trading terminal UI includes:

### Main Panels:
1. **Chart Panel** - LightweightCharts with live price/candle updates (500ms price, 2s candles)
2. **Market Pulse** - Live alerts and signals
3. **Order Flow** - CVD (Cumulative Volume Delta), OI Changes top 10
4. **On-Chain** - Whale transactions, USDT dominance
5. **Risk Simulation** - Monte Carlo simulations, Kelly Criterion
6. **Positions** - User's live positions with entry/SL/TP lines on chart
7. **MCF Labs** - Institutional analysis reports
8. **Pre-Trade Calculator** - 50,000 simulation pre-trade analysis

### Key Features:
- **Collapsible panels** with layout saving to Supabase
- **Live position lines** drawn on chart (entry, SL, TP)
- **24h price change** displayed with color coding
- **Live alerts feed** with animations and "NEW" badges
- **Theme:** Dark crimson tactical aesthetic with scanlines

## Backend (`api/terminal_api.py` - 6700+ lines)

FastAPI-based backend deployed on Railway.

### Core Endpoints:

**Authentication:**
- `POST /api/auth/register` - User registration (increments global user count)
- `POST /api/auth/login` - Login with optional 2FA
- `GET /api/auth/me` - Get current user
- `POST /api/auth/load-exchanges` - Load saved exchange keys from cloud

**Exchange Integration:**
- `POST /api/exchange/connect` - Connect exchange API keys (increments stats)
- `GET /api/exchange/list` - List connected exchanges
- `GET /api/positions` - Get live positions from connected exchanges

**Market Data:**
- `GET /api/live-price/{symbol}` - Live price with 24h change (0.5s cache)
- `GET /api/klines/{symbol}` - Candlestick data (1s cache for live)
- `GET /api/cvd/{symbol}` - Cumulative Volume Delta
- `GET /api/oi-changes` - Top 10 OI increases/decreases (24h)
- `GET /api/usdt-dominance` - USDT market dominance

**AI & Analysis:**
- `POST /api/bastion/chat` - Chat with BASTION AI
- `GET /api/pre-trade-calculator` - 50,000 Monte Carlo simulations
- `GET /api/kelly-criterion` - Kelly Criterion position sizing
- `POST /api/risk-intelligence/evaluate` - AI evaluates position for action

**Alerts:**
- `GET /api/live-alerts` - In-memory alerts for dashboard
- `POST /api/alerts/generate` - Generate new market alerts
- `POST /api/market-pulse` - Push alert to Telegram
- `POST /api/liquidation-alert` - Liquidation-specific alerts

**Statistics (Marketing):**
- Global stats tracked: `total_users`, `total_exchanges_connected`, `total_positions_analyzed`, `total_portfolio_managed_usd`
- **CUMULATIVE tracking**: Every exchange connection adds to `total_portfolio_managed_usd` even on reconnects

## Database (Supabase)

Tables:
- `users` - User accounts, 2FA settings
- `exchange_keys` - Encrypted API keys per user
- `bastion_stats` - Global platform statistics

---

# PART 3: DATA SOURCES (CRITICAL - AI IS TRAINED ON THESE)

## 1. Helsinki VM (FREE - 33 Endpoints)

```
Base URL: http://77.42.29.188:5002
Auth: None required
Rate Limit: Unlimited
```

### Key Endpoints:

| Endpoint | Returns | Use Case |
|----------|---------|----------|
| `/quant/full/{symbol}` | ALL data for symbol | Primary data fetch |
| `/quant/cvd/{symbol}` | Cumulative Volume Delta | Order flow analysis |
| `/quant/options-iv/{symbol}` | Implied volatility + PRICE | Best price source |
| `/quant/liquidation-estimate/{symbol}` | Liquidation clusters | Risk mapping |
| `/quant/smart-money/{symbol}` | Institutional flow | Whale tracking |
| `/quant/volatility/{symbol}` | Regime detection | Position sizing |
| `/quant/orderbook/{symbol}` | Bid/ask imbalance | Short-term direction |
| `/quant/momentum/{symbol}` | RSI, ROC, ATR | Trend strength |
| `/quant/mean-reversion/{symbol}` | Z-scores, Bollinger | Mean reversion |
| `/quant/vwap/{symbol}` | VWAP + bands | Fair value |
| `/quant/basis/{symbol}` | Spot/futures basis | Contango/backwardation |
| `/quant/open-interest/{symbol}` | OI analysis | Market participation |
| `/quant/greeks/{symbol}` | Options sentiment | P/C ratio |
| `/sentiment/fear-greed` | Fear & Greed Index | Market sentiment |
| `/quant/dominance` | BTC/ETH dominance | Alt season detection |
| `/quant/gas` | Network fees | Activity level |
| `/quant/defi-tvl` | DeFi TVL | Ecosystem health |
| `/quant/funding-arb` | Funding arbitrage | Arb opportunities |

### Sample Response (`/quant/full/BTC`):
```json
{
  "symbol": "BTC",
  "price": 94500,
  "volatility": {"regime": "NORMAL", "percentile": 58, "atr_14": 1823},
  "liquidation": {
    "open_interest_usd": 28500000000,
    "long_short_ratio": 1.23,
    "cascade_bias": "DOWNSIDE",
    "downside_liquidation_zones": [{"price": 91800, "distance_pct": -2.86, "estimated_usd_at_risk": 847000000}],
    "upside_liquidation_zones": [{"price": 97200, "distance_pct": 2.86, "estimated_usd_at_risk": 234000000}]
  },
  "smart_money": {"signal": "BULLISH", "whale_buy_ratio": 0.68},
  "options_iv": {"underlying_price": 94523.45, "atm_implied_volatility_pct": 48.5}
}
```

## 2. Coinglass Premium ($299/month)

```
Base URL: https://open-api-v3.coinglass.com/api
Auth: CG-API-KEY header
API Key: 03e5a43afaa4489384cb935b9b2ea16b
Rate Limit: 100 requests/minute
```

### Key Endpoints:

| Endpoint | Returns | Use Case |
|----------|---------|----------|
| `/futures/liquidation/heatmap` | Liquidation clusters | Find squeeze zones |
| `/futures/liquidation/aggregated-history` | Historical liqs | Pattern recognition |
| `/futures/openInterest/exchange-list` | OI by exchange | Institutional activity |
| `/futures/fundingRate/exchange-list` | Funding rates | Sentiment + arb |
| `/futures/topLongShortAccountRatio/history` | Whale L/S ratio | Smart money positioning |
| `/options/info` | Put/call ratio, max pain | Options sentiment |
| `/index/bitcoin-etf` | ETF flows | Institutional demand |

### Sample Response (`/futures/liquidation/heatmap`):
```json
{
  "code": "0",
  "data": {
    "levels": [
      {"price": 91000, "longLiquidation": 145000000, "shortLiquidation": 0},
      {"price": 98000, "longLiquidation": 0, "shortLiquidation": 89000000}
    ],
    "currentPrice": 94500
  }
}
```

## 3. Whale Alert Premium ($29.95/month)

```
Base URL: https://api.whale-alert.io/v1
Auth: api_key query parameter
API Key: OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ
WebSocket: wss://ws.whale-alert.io
Rate Limit: 10 requests/minute REST
```

### Key Endpoints:

| Endpoint | Returns | Use Case |
|----------|---------|----------|
| `/transactions` | Recent whale txs | Real-time tracking |
| `/transaction/{blockchain}/{hash}` | Specific tx | Deep dive |

### Trading Signals from Whale Data:

**Bullish:**
- Large exchange WITHDRAWALS (coins leaving exchanges)
- Stablecoin MINTS (new capital entering)
- Unknown → Unknown transfers (accumulation)

**Bearish:**
- Large exchange DEPOSITS (potential selling)
- Stablecoin BURNS (capital exiting)
- Whale → Exchange transfers (distribution)

### Sample Response:
```json
{
  "transactions": [{
    "blockchain": "bitcoin",
    "symbol": "btc", 
    "amount": 500.25,
    "amount_usd": 47273625,
    "from": {"owner": "binance", "owner_type": "exchange"},
    "to": {"owner": "unknown", "owner_type": "unknown"}
  }]
}
```

---

# PART 4: THE GPU CLUSTER (4x RTX 5090)

## Why We Need GPUs

BASTION uses a **32 billion parameter AI model** (Qwen2.5-Coder-32B) that requires massive GPU memory to run. The model is:

1. **Fine-tuned** on MCF (Multi-Confirmation Framework) trading methodology
2. **Trained** on all data source APIs (Helsinki, Coinglass, Whale Alert)
3. **Specialized** for two tasks:
   - **IROS Analyst**: Market analysis, trade ideas, answering questions
   - **Risk Intelligence**: Autonomous position management, executing exits

## Current Cluster Configuration

```
Provider: Vast.ai
Cost: ~$1.91/hour (ACTIVELY BILLING RIGHT NOW)
Hardware: 4x NVIDIA RTX 5090 (32GB VRAM each = 128GB total)
Storage: 1TB NVMe
OS: Ubuntu with CUDA
Python: /opt/sys-venv/bin/python3

SSH Access:
  Command: ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178
  IP: 74.48.140.178
  Port: 26796
```

## What Exists on the Cluster

```
/root/models/iros-merged/           # ~65GB - Merged Qwen2.5-32B with IROS LoRA
/root/checkpoints/risk-intelligence-lora/  # Trained Risk Intelligence LoRA
/root/training/                     # 797 training examples (transferred)
  ├── iros_72b_train.jsonl          # 600 core analysis examples
  ├── iros_72b_eval.jsonl           # 67 eval examples
  ├── risk_intelligence_train.jsonl # 17 risk mgmt examples
  ├── risk_intelligence_expanded.jsonl # 20 more scenarios
  ├── bastion_api_knowledge.jsonl   # 9 API endpoint training
  ├── combined_analysis.jsonl       # 6 multi-source analysis
  └── mcf_intelligence_reports.jsonl # 39 full reports
```

## How the 4 GPUs Work (Tensor Parallelism)

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM SERVER (Port 8000)                  │
│                  Tensor Parallel Mode (TP=4)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   The 32B model is SPLIT across all 4 GPUs:                 │
│                                                             │
│   GPU 0: Layers 0-15   (~16GB VRAM)                         │
│   GPU 1: Layers 16-31  (~16GB VRAM)                         │
│   GPU 2: Layers 32-47  (~16GB VRAM)                         │
│   GPU 3: Layers 48-63  (~16GB VRAM)                         │
│                                                             │
│   ALL 4 GPUs work on EVERY request together                 │
│   = Faster inference than single GPU                        │
│   = Can handle 10-20 concurrent users                       │
│                                                             │
│   API Endpoint: http://74.48.140.178:8000                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## vLLM - The Inference Server

vLLM is a high-throughput LLM serving engine that:
- Keeps model loaded in GPU memory (no reload per request)
- Handles concurrent requests efficiently
- Provides OpenAI-compatible API endpoints
- Supports LoRA adapters for specialized models

### OpenAI-Compatible Endpoints (once running):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat with BASTION AI |
| `/v1/completions` | POST | Raw text completion |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |

### How BASTION Backend Calls vLLM:

```python
import httpx

VLLM_URL = "http://74.48.140.178:8000/v1/chat/completions"

async def ask_bastion(user_message: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(VLLM_URL, json={
            "model": "/root/models/iros-merged",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        })
        return response.json()["choices"][0]["message"]["content"]
```

---

# PART 5: THE AI MODELS

## Model 1: IROS Analyst (Market Analysis)

**Purpose:** Answer trading questions, provide market analysis, interpret data

**Base:** Qwen2.5-Coder-32B-Instruct  
**Fine-tuning:** LoRA (r=128, alpha=256) trained on 688 examples  
**Accuracy:** 93.7% on eval set

**System Prompt:**
```
You are BASTION - an institutional-grade crypto trading AI with access to Helsinki VM (33 endpoints), Coinglass Premium (liquidations, funding, OI), and Whale Alert (on-chain tracking). Provide comprehensive market analysis combining multiple data sources.

CRITICAL RULES:
1. USE ONLY THE PRICES AND DATA PROVIDED. NEVER INVENT NUMBERS.
2. If data shows BTC at $62,000 - use $62,000. DO NOT GUESS.
3. No emojis. Use probabilities and confidence scores.
4. Be precise, quantified, actionable.
5. Reject bad setups with clear reasoning.

RESPONSE FORMAT:
## Key Structural Levels
- Resistance: $X (Grade 1-3, touches)
- Support: $X (Grade 1-3, touches)

## Entry Setup (Test → Break → Retest)
- Current Phase: [Awaiting break / Testing / Confirmed]
- Entry Trigger: [condition]
- Confirmation Required: [candle pattern needed]

## Trading Scenarios
BULLISH: Entry: $X, T1: $X, T2: $X, Stop: $X, R:R: X:1
BEARISH: Entry: $X, T1: $X, Stop: $X, R:R: X:1

## Risk Shield Position Sizing
- User Capital: $X
- Risk Budget: 2% per trade
- Position Size: X units ($X USD)

## VERDICT
Bias: [BULLISH/BEARISH/NEUTRAL] | Confidence: X% | Action: [Specific instruction]
```

## Model 2: Risk Intelligence (Autonomous Trade Management)

**Purpose:** Monitor live positions and make execution decisions (TP, SL, trailing stops, exits)

**Training:** 37 examples + 5 eval of position management scenarios  
**Output Format:** JSON actions

**System Prompt:**
```
You are BASTION Risk Intelligence - an autonomous trade management AI. You monitor live positions and make execution decisions. Output JSON with action, reasoning, and confidence.

PRIORITY ORDER (MCF Exit Logic):
1) Hard Stop - Maximum loss threshold, NON-NEGOTIABLE
2) Safety Net Break - Secondary structure violation  
3) Guarding Line Break - Trailing structure level
4) Take Profit Targets - T1, T2, T3 based on structure
5) Trailing Stop - ATR-based dynamic stop
6) Time Exit - Position duration limits

Core philosophy: Exit on STRUCTURE BREAKS, not arbitrary targets. Let winners run when structure holds.
```

**Example Input:**
```json
{
  "position": {
    "symbol": "BTC-PERP",
    "direction": "long",
    "entry_price": 95000,
    "current_price": 97500,
    "size": 0.5,
    "stop_loss": 93500,
    "take_profit_1": 98000,
    "take_profit_2": 100000
  },
  "market_data": {
    "atr_14": 1800,
    "support_level": 96800,
    "resistance_level": 98500,
    "funding_rate": 0.0001,
    "whale_flow": "neutral"
  }
}
```

**Example Output:**
```json
{
  "action": "MOVE_STOP_TO_BREAKEVEN",
  "new_stop": 95000,
  "reasoning": "Price has moved +2.6% from entry. Support established at 96800. Moving stop to breakeven protects capital while allowing position to run to T1.",
  "confidence": 0.87,
  "next_evaluation": "When price reaches 98000 (T1) or breaks below 96800"
}
```

**Possible Actions:**
- `HOLD` - No action needed
- `TP_PARTIAL` - Take partial profit (specify %)
- `TP_FULL` - Close entire position at profit
- `MOVE_STOP_TO_BREAKEVEN` - Move stop to entry price
- `TRAIL_STOP` - Update trailing stop level
- `EXIT_FULL` - Emergency exit entire position
- `REDUCE_SIZE` - Reduce position size by X%

---

# PART 6: MCF (MULTI-CONFIRMATION FRAMEWORK)

MCF is the proprietary trading methodology that BASTION is trained on.

## Core Principles

1. **Structure-Based Exits** - Exit on structure breaks, not arbitrary price targets
2. **Multi-Tier Defense** - Multiple stop levels for different scenarios
3. **Adaptive Position Sizing** - Volatility-adjusted sizing based on ATR

## Exit Logic Priority (Risk Intelligence Training)

### 1. Hard Stop (Priority 1)
- Maximum loss threshold
- NON-NEGOTIABLE - always honored
- Typically 1.5-2x ATR from entry

### 2. Safety Net (Priority 2)
- Secondary structure level
- Triggers if price closes below with volume
- Usually a major swing low/high

### 3. Guarding Line (Priority 3)
- Dynamic trailing structure level
- Moves up as position profits
- Based on swing points, not percentages

### 4. Take Profit Targets (Priority 4)
- T1: First resistance/support (take 30-50%)
- T2: Extended target (take 30-40%)
- T3: Runners target (let remainder ride)

### 5. Trailing Stop (Priority 5)
- ATR-based dynamic stop
- Typically 1.5x ATR behind price
- Adjusts with volatility

### 6. Time Exit (Priority 6)
- Maximum position duration
- Different for scalps vs swings
- Prevents capital lockup

## MCF Scoring Methodology

When analyzing a setup, score these factors:

1. **Structure Quality** (1-3)
   - Key level touches
   - Confluence zones
   - Historical significance

2. **Order Flow Confirmation** (1-3)
   - CVD divergence
   - Volume patterns
   - Orderbook imbalance

3. **Derivatives Alignment** (1-3)
   - Funding rate bias
   - OI direction
   - L/S ratio

4. **On-Chain Support** (1-3)
   - Whale positioning
   - Exchange flows
   - Stablecoin activity

**Total Score:** 4-12 points
- 10-12: High confidence setup
- 7-9: Medium confidence
- 4-6: Low confidence (pass)

---

# PART 7: TRAINING DATA CORPUS

## Location
```
C:\Users\Banke\IROS_72B_TRAINING_CORPUS\FINAL_TRAINING\
```

## Files Summary (797 Total Examples)

### IROS Analyst Training (755 examples):

| File | Examples | Content |
|------|----------|---------|
| `iros_72b_train.jsonl` | 600 | Core market analysis |
| `mcf_intelligence_reports.jsonl` | 39 | Full MCF reports |
| `bastion_api_knowledge.jsonl` | 9 | API endpoint training |
| `trade_rejection_examples.jsonl` | 7 | Bad trade rejection |
| `enhanced_tier1_tda.jsonl` | 6 | TDA concepts |
| `adversarial_examples.jsonl` | 6 | Edge cases |
| `combined_analysis.jsonl` | 6 | Multi-source analysis |
| `enhanced_tier1_rl.jsonl` | 5 | RL concepts |
| `helsinki_endpoints.jsonl` | 4 | Helsinki API deep dive |
| `chain_of_thought_examples.jsonl` | 3 | Reasoning chains |
| `mcf_scoring_examples.jsonl` | 3 | MCF scoring |
| `iros_72b_eval.jsonl` | 67 | Evaluation set |

### Risk Intelligence Training (42 examples):

| File | Examples | Content |
|------|----------|---------|
| `risk_intelligence_train.jsonl` | 17 | Core risk management |
| `risk_intelligence_expanded.jsonl` | 20 | Expanded scenarios |
| `risk_intelligence_eval.jsonl` | 5 | Evaluation set |

## Training Data Format (JSONL)

```json
{
  "messages": [
    {"role": "system", "content": "You are BASTION..."},
    {"role": "user", "content": "Should I long BTC at $95K with $50K capital?"},
    {"role": "assistant", "content": "## Key Structural Levels\n- Resistance: $98,000...\n\n## VERDICT\nBias: BULLISH | Confidence: 72%..."}
  ]
}
```

---

# PART 8: ALERTS SYSTEM

## Alert Types

### Position-Based Alerts (Critical)
- Stop loss proximity (<2%)
- Take profit reached
- Trailing stop triggered
- R-multiple breach
- Funding rate impact

### Market Intelligence Alerts
- Whale movements (>$10M)
- Liquidation cascades (>$50M/hour)
- Funding rate extremes (>0.1%)
- OI spikes (>5% in 4h)
- Volume anomalies (>3x avg)
- CVD divergence

### On-Chain Alerts
- Exchange inflows (sell pressure)
- Exchange outflows (accumulation)
- Stablecoin mints/burns
- Whale wallet activity

## Alert Delivery

1. **In-App** - Alert feed in dashboard, modals for critical
2. **Telegram** - Bot pushes to user's channel
3. **Browser Push** - (if enabled)

## Alert Format

```json
{
  "alert_id": "uuid",
  "type": "position|market|technical|onchain",
  "severity": "info|warning|critical",
  "asset": "BTC",
  "title": "WHALE DEPOSIT: 2,500 BTC → BINANCE",
  "message": "$243M BTC moved to Binance. 67% correlation with 2%+ downside in 24h.",
  "action_suggestions": [
    "Tighten stops on BTC longs",
    "Consider reducing exposure"
  ]
}
```

---

# PART 9: GLOBAL STATISTICS (MARKETING)

BASTION tracks cumulative stats for the landing page:

```python
bastion_stats = {
    "total_users": 0,                    # Increments on registration
    "total_exchanges_connected": 0,      # Increments on exchange connect
    "total_positions_analyzed": 0,       # Increments on position analysis
    "total_portfolio_managed_usd": 0.0   # CUMULATIVE - adds EVERY connection
}
```

**Important:** `total_portfolio_managed_usd` is CUMULATIVE:
- User connects with $7K → adds $7K
- Same user reconnects → adds ANOTHER $7K
- This tracks total volume touched by BASTION over time

---

# PART 10: FILE LOCATIONS

## Windows Machine (User's PC)

```
C:\Users\Banke\BASTION\BASTION\          # Main codebase
├── api\
│   ├── terminal_api.py                   # FastAPI backend (6700+ lines)
│   ├── user_service.py                   # User management
│   └── models.py                         # Data models
├── core\
│   ├── risk_engine.py                    # Risk calculations
│   ├── structure_detector.py             # S/R detection
│   └── orderflow_detector.py             # Order flow analysis
├── iros_integration\
│   ├── services\
│   │   ├── helsinki.py                   # Helsinki VM client
│   │   ├── coinglass.py                  # Coinglass client
│   │   ├── whale_alert.py                # Whale Alert client
│   │   ├── bastion_ai.py                 # AI query processor
│   │   └── exchange_connector.py         # Exchange integration
│   └── endpoints\
│       ├── ALL_HELSINKI_ENDPOINTS.md     # Full Helsinki docs
│       ├── COINGLASS_ENDPOINTS.md        # Full Coinglass docs
│       └── WHALE_ALERT_ENDPOINTS.md      # Full Whale Alert docs
├── generated-page.html                   # Main trading terminal UI
├── web\
│   ├── login.html                        # Login page
│   ├── account.html                      # Account settings
│   └── settings.js                       # Frontend config
├── prompts\
│   ├── IROS_ALERT_AGENT.md               # Alert generation prompt
│   └── MCF_REPORT_AGENT.md               # Report generation prompt
├── CLAUDE_COMPLETE_CONTEXT.md            # THIS FILE
├── CLAUDE_MASTER_PROMPT.md               # Quick start version
└── requirements.txt                      # Python dependencies

C:\Users\Banke\IROS_72B_TRAINING_CORPUS\  # Training data
├── FINAL_TRAINING\
│   ├── *.jsonl                           # All training files
│   └── training_manifest.json            # Metadata
└── MCF_RISK_EXIT_LOGIC_EXTRACTION.md     # MCF methodology docs

C:\Users\Banke\BASTION_BACKUPS\           # Local backups
```

## GPU Cluster

```
/root/models/iros-merged/                 # 65GB merged model
/root/checkpoints/risk-intelligence-lora/ # Risk LoRA
/root/training/                           # Training data
/root/vllm_server.log                     # vLLM logs
/opt/sys-venv/bin/python3                 # Python interpreter
```

---

# PART 11: KNOWN ISSUES & FIXES

## Issue 1: vLLM v1 Engine Crash
**Error:** `RuntimeError: Engine core initialization failed`
**Cause:** vLLM 0.15.x has buggy v1 engine
**Fix:** 
```bash
VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server --enforce-eager ...
# OR downgrade:
pip install vllm==0.6.3 --force-reinstall
```

## Issue 2: PyTorch RTX 5090 Compatibility
**Error:** `CUDA capability sm_120 not supported`
**Cause:** RTX 5090 (Blackwell) needs newer PyTorch
**Fix:**
```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

## Issue 3: HuggingFace Cache Path
**Issue:** Model downloads to wrong location
**Fix:** 
```bash
export HF_HOME=/root/.cache/huggingface
# OR in Python:
os.environ['HF_HOME'] = '/root/.cache/huggingface'
```

## Issue 4: Tokenizer Config Error
**Error:** `extra_special_tokens must be dict, not list`
**Fix:**
```python
import json
with open('/root/models/iros-merged/tokenizer_config.json', 'r') as f:
    config = json.load(f)
config.pop('extra_special_tokens', None)
with open('/root/models/iros-merged/tokenizer_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## Issue 5: Cursor Agents Crashing
**Issue:** Agents die during long SSH commands
**Cause:** Timeout on long-running operations
**Fix:** Use `nohup` for background processes:
```bash
nohup /opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server ... > /root/vllm_server.log 2>&1 &
```

---

# PART 12: IMMEDIATE TASK - DEPLOY vLLM

## Current Status (as of Feb 10, 2026)
- ✅ Cluster is ACTIVE (4x RTX 5090, all GPUs idle)
- ✅ Model exists at `/root/models/iros-merged`
- ✅ Training data transferred
- ❌ vLLM NOT RUNNING - needs to be started

## Commands to Execute (IN ORDER)

### Step 1: Kill existing processes
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "pkill -9 -f vllm; pkill -9 -f python; echo KILLED"
```

### Step 2: Check vLLM version
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "/opt/sys-venv/bin/pip show vllm | grep Version"
```

### Step 3: Downgrade if needed (v0.15.x → v0.6.3)
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "/opt/sys-venv/bin/pip install vllm==0.6.3 --force-reinstall"
```
**⚠️ TAKES 2-5 MINUTES**

### Step 4: Start vLLM server
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "VLLM_USE_V1=0 nohup /opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server --model /root/models/iros-merged --tensor-parallel-size 4 --port 8000 --host 0.0.0.0 --max-model-len 4096 --trust-remote-code --enforce-eager > /root/vllm_server.log 2>&1 &"
```

### Step 5: Wait and check logs
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "sleep 60 && tail -100 /root/vllm_server.log"
```
**Look for:** `Uvicorn running on http://0.0.0.0:8000`

### Step 6: Test health
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "curl -s http://localhost:8000/health"
```

### Step 7: Test inference
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "curl -s -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"/root/models/iros-merged\", \"messages\": [{\"role\": \"user\", \"content\": \"What is MCF scoring?\"}], \"max_tokens\": 100}'"
```

## Success Criteria
1. ✅ vLLM running at `http://74.48.140.178:8000`
2. ✅ `/health` returns OK
3. ✅ `/v1/models` shows the model
4. ✅ Test query returns MCF-aware response
5. ✅ BASTION backend can call the endpoint

---

# PART 13: TROUBLESHOOTING COMMANDS

```bash
# Check GPU status
ssh -p 26796 root@74.48.140.178 "nvidia-smi"

# Check vLLM logs
ssh -p 26796 root@74.48.140.178 "tail -100 /root/vllm_server.log"

# Check full logs
ssh -p 26796 root@74.48.140.178 "cat /root/vllm_server.log"

# Kill all processes
ssh -p 26796 root@74.48.140.178 "pkill -9 -f vllm; pkill -9 -f python"

# Check disk space
ssh -p 26796 root@74.48.140.178 "df -h"

# Check model files
ssh -p 26796 root@74.48.140.178 "ls -la /root/models/iros-merged/"

# Check Python packages
ssh -p 26796 root@74.48.140.178 "/opt/sys-venv/bin/pip list | grep -E 'vllm|torch|transformers'"

# Monitor GPU memory live
ssh -p 26796 root@74.48.140.178 "watch -n 1 nvidia-smi"
```

---

# PART 14: FOR THE AGENT

## YOUR MISSION

You are helping deploy BASTION's AI infrastructure. The user has:
- Spent all day fighting crashed Cursor agents
- A GPU cluster actively billing at ~$1.91/hour
- A fine-tuned 32B model ready to serve

**YOU MUST EXECUTE COMMANDS YOURSELF.** Use Desktop Commander to run SSH commands. Do NOT just explain - DO IT.

## CRITICAL REMINDERS

1. **The cluster is LIVE and BILLING** - every hour costs money
2. **The model is already on the cluster** - no need to download
3. **vLLM version matters** - 0.15.x has bugs, use 0.6.3 or add --enforce-eager
4. **Use nohup** - prevents process death on SSH disconnect
5. **Check logs after every command** - don't assume success

## EXECUTION FLOW

1. Kill existing processes
2. Check/downgrade vLLM if needed
3. Start vLLM with correct flags
4. Wait 60 seconds
5. Check logs for "Uvicorn running"
6. Test health endpoint
7. Test inference
8. Report endpoint URL to user: `http://74.48.140.178:8000`

**START NOW. EXECUTE STEP 1.**

---

**DOCUMENT END**

This is the complete context for BASTION. Copy this entire document to any new Claude session for full context continuity.
