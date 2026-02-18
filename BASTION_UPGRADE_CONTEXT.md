# 🏰 BASTION UPGRADE CONTEXT - COMPLETE SYSTEM FOR CLAUDE

**Purpose:** Give Claude EVERYTHING about BASTION so it can upgrade the AI to be a world-class trading intelligence system.

**Date:** February 10, 2026  
**GPU Cluster:** LIVE at http://74.48.140.178:26717 (4x RTX 5090, bastion-32b model)  
**Cost:** $1.965/hr

---

# PART 1: WHAT BASTION IS

## 1.1 Product Vision

BASTION is an **institutional-grade crypto trading terminal** that provides:

1. **AI-Powered Market Analysis** - Users ask questions, get MCF-scored analysis
2. **Autonomous Risk Intelligence** - AI monitors and manages user positions 24/7
3. **Live Market Alerts** - Real-time signals pushed to Telegram and dashboard
4. **Exchange Integration** - Connect Bitunix, Bybit, OKX, Binance for live position tracking
5. **Premium Data Synthesis** - Combines Helsinki VM (33 endpoints), Coinglass ($299/mo), Whale Alert ($30/mo)

## 1.2 The Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BASTION SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│   │   FRONTEND      │     │   BACKEND       │     │   AI CLUSTER    │   │
│   │   (Terminal)    │◄───►│   (FastAPI)     │◄───►│   (4x 5090)     │   │
│   │                 │     │                 │     │                 │   │
│   │ - Live Charts   │     │ - Railway       │     │ - vLLM          │   │
│   │ - Position View │     │ - Supabase Auth │     │ - bastion-32b   │   │
│   │ - Alerts Panel  │     │ - Exchange APIs │     │ - 8192 context  │   │
│   │ - MCF Labs      │     │ - Data Clients  │     │ - Tensor TP=4   │   │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘   │
│                                   │                                     │
│                                   ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     DATA SOURCES                                 │   │
│   │                                                                  │   │
│   │   Helsinki VM        Coinglass           Whale Alert             │   │
│   │   (FREE)             ($299/mo)           ($29.95/mo)             │   │
│   │   33 endpoints       Liquidations        On-chain                │   │
│   │   CVD, OI, Vol       Funding, L/S        Whale txs               │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.3 What the AI Endpoint Does

The vLLM server at `http://74.48.140.178:26717/v1/chat/completions` handles:

| Task | System Prompt | Output |
|------|---------------|--------|
| **User Chat** | BASTION analyst prompt | Natural language analysis with levels, R:R, verdict |
| **Risk Intelligence** | Risk management prompt | JSON: `{"action": "TP_50%", "confidence": 0.87}` |
| **Alert Generation** | Alert prompt | Short alert text for Telegram |

All tasks use the SAME model with different system prompts.

---

# PART 2: ALL DATA ENDPOINTS WE HAVE ACCESS TO

## 2.1 Helsinki VM (FREE - Unlimited)

**Base URL:** `http://77.42.29.188:5002`  
**Auth:** None required  
**Rate Limit:** Unlimited  

### Order Flow Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/quant/cvd/{symbol}` | CVD 1h/4h/total, divergence, signal | Buy/sell pressure, divergence trading |
| `/quant/orderbook/{symbol}` | Bid/ask imbalance, spread, pressure | Short-term direction |
| `/quant/large-trades/{symbol}` | Whale buy/sell volume, net flow | Institutional activity |
| `/quant/smart-money/{symbol}` | Smart money bias, trend, interpretation | Follow institutions |
| `/quant/whale-flow/{symbol}` | Inflow/outflow 24h, exchange netflow | Accumulation/distribution |

### Derivatives Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/quant/basis/{symbol}` | Spot/futures basis, contango/backwardation | Sentiment, arb |
| `/quant/open-interest/{symbol}` | OI value, 24h change, trend | Market participation |
| `/quant/greeks/{symbol}` | IV, put/call ratio, options sentiment | Options sentiment |
| `/quant/liquidation-map/{symbol}` | Upside/downside liquidation levels | Squeeze zones |
| `/quant/liquidation-estimate/{symbol}` | Cascade bias, zones with $ at risk | Critical for stops |
| `/quant/options-iv/{symbol}` | Underlying price, ATM IV, skew | Reliable price source |
| `/quant/funding-arb` | Cross-exchange funding arbitrage | Arb opportunities |

### Volatility Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/quant/volatility/{symbol}` | Regime (LOW/NORMAL/HIGH), percentile | Position sizing |
| `/quant/iv-rv-spread/{symbol}` | IV vs RV spread, strategy suggestion | Options premium |

### Technical Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/quant/vwap/{symbol}` | VWAP, upper/lower 1SD bands | Fair value |
| `/quant/momentum/{symbol}` | Score, RSI, ROC, ATR | Trend strength |
| `/quant/mean-reversion/{symbol}` | Z-scores, Bollinger bands, signal | Mean reversion |
| `/quant/drawdown/{symbol}` | Current drawdown %, max drawdown | Risk assessment |

### Macro/Sentiment Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/quant/dominance` | BTC/ETH dominance, alt season score | Market structure |
| `/quant/defi-tvl` | Total DeFi TVL, top protocols | Ecosystem health |
| `/quant/gas` | ETH/BTC gas fees, congestion | Network activity |
| `/quant/stablecoin-supply` | USDT/USDC/DAI supply | Capital flows |
| `/sentiment/fear-greed` | Fear & Greed index, trend | Market sentiment |

### Master Endpoint

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/quant/full/{symbol}` | **ALL DATA IN ONE CALL** | Complete analysis |

### Sample `/quant/full/BTC` Response:

```json
{
  "symbol": "BTC",
  "price": 97500,
  "volatility": {
    "regime": "NORMAL",
    "percentile": 58,
    "atr_14": 1823.45
  },
  "liquidation": {
    "open_interest_usd": 28500000000,
    "long_short_ratio": 1.23,
    "cascade_bias": "DOWNSIDE",
    "downside_liquidation_zones": [
      {"price": 91800, "distance_pct": -2.86, "estimated_usd_at_risk": 847000000}
    ],
    "upside_liquidation_zones": [
      {"price": 100200, "distance_pct": 2.86, "estimated_usd_at_risk": 234000000}
    ]
  },
  "smart_money": {
    "signal": "BULLISH",
    "whale_buy_ratio": 0.68
  },
  "options_iv": {
    "underlying_price": 97523.45,
    "atm_implied_volatility_pct": 48.5,
    "put_call_ratio": 0.82
  },
  "cvd": {
    "cvd_1h": 1234.567,
    "cvd_4h": 5678.901,
    "divergence": "BULLISH_DIVERGENCE"
  },
  "momentum": {
    "score": 72,
    "rsi_14": 62.5,
    "interpretation": "Strong upward momentum"
  }
}
```

---

## 2.2 Coinglass Premium ($299/month)

**Base URL:** `https://open-api-v3.coinglass.com/api`  
**Auth:** `CG-API-KEY` header  
**API Key:** `03e5a43afaa4489384cb935b9b2ea16b`  
**Rate Limit:** 100 requests/minute  

### Liquidation Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/futures/liquidation/heatmap` | Liquidation clusters by price | Find squeeze zones |
| `/futures/liquidation/aggregated-history` | Historical liquidations | Pattern recognition |
| `/futures/liquidation/exchange-list` | Liquidations by exchange | Exchange-specific data |

### Open Interest Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/futures/openInterest/exchange-list` | OI by exchange with 24h change | Institutional activity |
| `/futures/openInterest/ohlc-aggregated-history` | OI time series | OI trends |

### Funding Rate Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/futures/fundingRate/exchange-list` | Current funding all exchanges | Sentiment + arb |
| `/futures/fundingRate/history` | Historical funding | Funding patterns |
| `/futures/fundingRate/oi-weight` | OI-weighted funding | True aggregate |

### Long/Short Ratio Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/futures/globalLongShortAccountRatio/history` | Retail L/S ratio | Retail sentiment |
| `/futures/topLongShortAccountRatio/history` | Whale L/S ratio | Smart money positioning |
| `/futures/topLongShortPositionRatio/history` | Size-weighted L/S | Position concentration |

### Options Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/options/info` | Put/call ratio, max pain, total OI | Options sentiment |
| `/options/openInterest/history` | Options OI history | Options trends |

### ETF Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/index/bitcoin-etf` | All spot ETF flows | Institutional demand |
| `/index/gbtc` | GBTC holdings, premium | Grayscale flows |

### Order Flow

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/futures/taker-buy-sell-ratio` | Taker buy/sell ratio | Order flow |

---

## 2.3 Whale Alert Premium ($29.95/month)

**Base URL:** `https://api.whale-alert.io/v1`  
**WebSocket:** `wss://ws.whale-alert.io`  
**Auth:** `api_key` query parameter  
**API Key:** `OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ`  
**Rate Limit:** 10 requests/minute REST  

### Supported Blockchains

BTC, ETH, XRP, USDT, USDC, BNB, ADA, SOL, AVAX, MATIC, TRX

### Endpoints

| Endpoint | Returns | Trading Use |
|----------|---------|-------------|
| `/transactions` | Recent whale transactions | Real-time tracking |
| `/transaction/{chain}/{hash}` | Specific transaction details | Deep dive |
| `/status` | API status, rate limits | Health check |

### Transaction Types

| Type | Description | Trading Signal |
|------|-------------|----------------|
| `transfer` | Wallet-to-wallet | Direction matters |
| `mint` | New tokens created | Bullish for crypto |
| `burn` | Tokens destroyed | Bearish for stables |
| `lock` | Locked in contract | Reduced supply |
| `unlock` | Released from contract | Increased supply |

### Owner Types & Signals

| Owner Type | Description | Signal |
|------------|-------------|--------|
| `exchange` | Known exchange | Deposit=bearish, Withdraw=bullish |
| `unknown` | Unidentified | Could be OTC, cold storage |
| `custodian` | Custody service | Institutional |
| `whale` | Known large holder | Watch patterns |

### Trading Signals

**Bullish:**
- Large exchange withdrawals (coins leaving)
- Stablecoin mints (new capital)
- Unknown → Unknown (accumulation)

**Bearish:**
- Large exchange deposits (selling pressure)
- Stablecoin burns (capital exiting)
- Whale → Exchange (distribution)

---

# PART 3: WHAT THE TERMINAL DISPLAYS

## 3.1 Dashboard Panels (from generated-page.html)

| Panel | Data Source | What It Shows |
|-------|-------------|---------------|
| **Chart** | Kraken/Coinbase | Live candlesticks, price lines for positions |
| **Market Pulse** | Helsinki + AI | Live alerts, signals, momentum shifts |
| **Order Flow** | Helsinki + Coinglass | CVD, OI changes (top 10 increases/decreases) |
| **On-Chain** | Whale Alert + Coinglass | Whale transactions, USDT dominance |
| **Risk Simulation** | Backend calc | Monte Carlo (50K sims), Kelly Criterion |
| **Positions** | Exchange APIs | User positions with entry/SL/TP lines, pre-trade calculator |
| **MCF Labs** | AI-generated | Institutional analysis reports |
| **Alerts & Signals** | AI + data | Live feed with animations, color coding |

## 3.2 Alert Types We Generate

| Alert Type | Trigger | Pushed To |
|------------|---------|-----------|
| Price Move | >1% change | Dashboard + Telegram |
| Level Cross | $100K, $95K, etc. | Dashboard + Telegram |
| High Funding | >0.05% | Dashboard + Telegram |
| Negative Funding | <-0.02% | Dashboard + Telegram |
| Extreme Fear | F&G <20 | Dashboard + Telegram |
| Extreme Greed | F&G >80 | Dashboard + Telegram |
| Volatility Shift | Regime change | Dashboard + Telegram |
| Whale Alert | >$10M tx | Dashboard + Telegram |
| Liquidation Cascade | >$50M | Dashboard + Telegram |

## 3.3 Pre-Trade Calculator

The terminal includes a pre-trade calculator that:
1. Takes user's entry, stop loss, leverage
2. Runs 50,000 Monte Carlo simulations
3. Shows probability distribution of outcomes
4. Calculates Kelly Criterion optimal sizing

---

# PART 4: TRAINING CORPUS STRUCTURE

## 4.1 File Location

```
C:\Users\Banke\IROS_72B_TRAINING_CORPUS\FINAL_TRAINING\
```

## 4.2 Training Files (797 Total Examples)

### BASTION Analyst Training (755 examples)

| File | Examples | Content |
|------|----------|---------|
| `iros_72b_train.jsonl` | 600 | Core market analysis |
| `mcf_intelligence_reports.jsonl` | 39 | Full MCF institutional reports |
| `bastion_api_knowledge.jsonl` | 9 | API endpoint understanding |
| `combined_analysis.jsonl` | 6 | Multi-source synthesis |
| `trade_rejection_examples.jsonl` | 7 | Rejecting bad setups |
| `enhanced_tier1_tda.jsonl` | 6 | TDA quant concepts |
| `adversarial_examples.jsonl` | 6 | Edge cases |
| `enhanced_tier1_rl.jsonl` | 5 | RL concepts |
| `helsinki_endpoints.jsonl` | 4 | Helsinki API deep dive |
| `chain_of_thought_examples.jsonl` | 3 | Reasoning chains |
| `mcf_scoring_examples.jsonl` | 3 | MCF scoring methodology |
| `iros_72b_eval.jsonl` | 67 | Evaluation set |

### Risk Intelligence Training (42 examples)

| File | Examples | Content |
|------|----------|---------|
| `risk_intelligence_train.jsonl` | 17 | Core risk management |
| `risk_intelligence_expanded.jsonl` | 20 | Expanded scenarios |
| `risk_intelligence_eval.jsonl` | 5 | Evaluation set |

## 4.3 Training Categories

| Category | Description |
|----------|-------------|
| `tda_quant_research` | Topological Data Analysis concepts |
| `reinforcement_learning` | RL for trading applications |
| `trade_rejections` | Correctly rejecting bad setups |
| `mcf_scoring` | MCF scoring walkthroughs |
| `helsinki_endpoints` | Helsinki VM endpoint deep dives |
| `adversarial` | Handling manipulative queries |
| `chain_of_thought` | Step-by-step reasoning |
| `mcf_intelligence_reports` | Full MCF reports |
| `risk_intelligence` | Autonomous trade management |
| `api_knowledge` | Helsinki + Coinglass + Whale Alert |
| `combined_analysis` | Multi-source analysis |

---

# PART 5: MCF RISK INTELLIGENCE METHODOLOGY

## 5.1 Exit Priority Order

The AI must follow this EXACT priority when managing positions:

```
1. HARD STOP (HIGHEST PRIORITY)
   └── Maximum loss threshold, NON-NEGOTIABLE
   └── Exit 100% immediately

2. SAFETY NET BREAK
   └── Long-term structure violation
   └── Exit 100% immediately

3. GUARDING LINE BREAK
   └── Dynamic trailing structure broken
   └── Exit 75% (primary) or 50% (secondary)

4. TAKE PROFIT TARGETS
   └── T1: Exit 30-50%
   └── T2: Exit 30-40% of remaining
   └── T3: Let runners ride

5. TRAILING STOP
   └── ATR-based dynamic stop
   └── Exit remaining position

6. TIME EXIT
   └── Max holding period exceeded
   └── Exit 50% gradually
```

## 5.2 Core Philosophy

> **"Exit on STRUCTURE BREAKS, not arbitrary targets. Let winners run when structure holds."**

Key Principles:
1. Don't exit just because price hits a number
2. Exit when supporting structure fails
3. Let winners run if structure is holding
4. Scale out at confluence zones
5. Guard the trade with dynamic stops
6. Adapt to volatility

## 5.3 Multi-Tier Stop System

| Tier | Exit % | Trigger | Purpose |
|------|--------|---------|---------|
| PRIMARY | 25% | First structural break | Early warning |
| SECONDARY | 50% of remaining | Second break | Intermediate protection |
| SAFETY NET | 100% of remaining | Ultimate break | Thesis invalidation |

## 5.4 ATR Formulas

```python
# Trailing Stop (Long)
SL_trail = max(SL_prev, High_since_entry - N × ATR)

# Trailing Stop (Short)
SL_trail = min(SL_prev, Low_since_entry + N × ATR)

# Structural Stop with Buffer
SL_final = Support_level - (0.1 × ATR)  # Long
SL_final = Resistance_level + (0.1 × ATR)  # Short

# Position Size
Qty = (Account × Risk%) / |Entry - Stop|
```

## 5.5 Risk Intelligence JSON Output Format

When evaluating positions, the AI outputs:

```json
{
  "action": "TP_PARTIAL",
  "exit_percentage": 50,
  "new_stop": 95000,
  "reasoning": "Price reached T1 at $98,000. Structure holding above $96,800 support. Taking 50% profit, moving stop to breakeven.",
  "confidence": 0.87,
  "warnings": ["Funding elevated at +0.02%"],
  "next_evaluation": "When price reaches $100,000 (T2) or breaks $96,800"
}
```

**Possible Actions:**
- `HOLD` - No action needed
- `TP_PARTIAL` - Take partial profit (specify %)
- `TP_FULL` - Close entire position
- `MOVE_STOP_TO_BREAKEVEN` - Move stop to entry
- `TRAIL_STOP` - Update trailing stop
- `EXIT_FULL` - Emergency exit
- `REDUCE_SIZE` - Reduce position by X%

---

# PART 6: SYSTEM PROMPTS

## 6.1 BASTION Analyst (Chat/Analysis)

```
You are BASTION - an institutional-grade crypto trading AI with access to premium data sources including Helsinki VM (33 free endpoints), Coinglass Premium ($299/mo for liquidations, OI, funding, L/S ratios), and Whale Alert Premium ($29.95/mo for on-chain tracking). Provide comprehensive market analysis by combining multiple data sources.

CRITICAL RULES:
1. USE ONLY THE PRICES AND DATA PROVIDED. NEVER INVENT NUMBERS.
2. If data shows BTC at $97,000 - use $97,000. DO NOT GUESS.
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

## Trading Scenarios
BULLISH: Entry $X, T1 $X, T2 $X, Stop $X, R:R X:1
BEARISH: Entry $X, T1 $X, Stop $X, R:R X:1

## Risk Shield Position Sizing
- User Capital: $X
- Risk Budget: 2% per trade
- Position Size: X units ($X USD)

## VERDICT
Bias: [BULLISH/BEARISH/NEUTRAL] | Confidence: X% | Action: [Specific instruction]
```

## 6.2 Risk Intelligence (Position Management)

```
You are BASTION Risk Intelligence - an autonomous trade management AI. You monitor live positions and make execution decisions. Output JSON with action, reasoning, and confidence.

PRIORITY ORDER:
1) Hard Stop - Maximum loss, NON-NEGOTIABLE
2) Safety Net Break - Long-term structure broken
3) Guarding Line Break - Dynamic trailing stop broken
4) Take Profit Targets - Structure-based exits
5) Trailing Stop - ATR-based dynamic stop
6) Time Exit - Position duration limits

Core philosophy: Exit on STRUCTURE BREAKS, not arbitrary targets. Let winners run when structure holds.

OUTPUT FORMAT:
{
  "action": "HOLD|TP_PARTIAL|TP_FULL|MOVE_STOP_TO_BREAKEVEN|TRAIL_STOP|EXIT_FULL|REDUCE_SIZE",
  "exit_percentage": 0-100,
  "new_stop": price or null,
  "reasoning": "explanation",
  "confidence": 0.0-1.0,
  "warnings": ["array of warnings"],
  "next_evaluation": "when to check again"
}
```

---

# PART 7: GPU CLUSTER STATUS

## 7.1 Current Deployment

```
Endpoint: http://74.48.140.178:26717/v1/chat/completions
Model: bastion-32b (Qwen2.5-Coder-32B + IROS fine-tune)
vLLM: v0.15.1 with FLASH_ATTN
GPUs: 4x RTX 5090 (30GB/32GB used each, 92%)
Tensor Parallel: 4
Context: 8192 tokens
Cost: $1.965/hr
```

## 7.2 How to Call the AI

```python
import httpx

VLLM_URL = "http://74.48.140.178:26717/v1/chat/completions"

async def ask_bastion(user_message: str, system_prompt: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(VLLM_URL, json={
            "model": "bastion-32b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        })
        return response.json()["choices"][0]["message"]["content"]
```

## 7.3 SSH Access

```bash
ssh -p 26796 root@74.48.140.178
tmux attach -t vllm  # View logs
```

---

# PART 8: WHAT NEEDS UPGRADING

## 8.1 Current Limitations

1. **Risk Intelligence needs more scenarios** - Only 42 training examples
2. **Alert generation is basic** - Could be smarter about what's important
3. **No correlation analysis** - Doesn't combine multiple signals intelligently
4. **Exit decisions could be sharper** - Needs more nuanced structure analysis
5. **No learning from outcomes** - Doesn't track what worked

## 8.2 Upgrade Goals

1. **Better at exits** - More precise structure break detection
2. **Smarter alerts** - Prioritize what actually matters
3. **Multi-signal synthesis** - Combine Helsinki + Coinglass + Whale Alert intelligently
4. **More scenarios** - Expand Risk Intelligence training
5. **Confidence calibration** - Know when it doesn't know

## 8.3 Available Resources

- All Helsinki endpoints (33 FREE, unlimited)
- All Coinglass endpoints ($299/mo, 100/min)
- All Whale Alert endpoints ($30/mo, 10/min)
- Full MCF methodology documentation
- 797 existing training examples
- Live GPU cluster ready to serve

---

# PART 9: FILE STRUCTURE

## 9.1 Codebase

```
C:\Users\Banke\BASTION\BASTION\
├── api\
│   ├── terminal_api.py       # 6700+ lines - main backend
│   ├── user_service.py       # User management
│   └── models.py             # Data models
├── core\
│   ├── risk_engine.py        # Risk calculations
│   └── structure_detector.py # S/R detection
├── iros_integration\
│   ├── services\
│   │   ├── helsinki.py       # Helsinki VM client
│   │   ├── coinglass.py      # Coinglass client
│   │   ├── whale_alert.py    # Whale Alert client
│   │   └── bastion_ai.py     # AI query processor
│   └── endpoints\
│       ├── ALL_HELSINKI_ENDPOINTS.md
│       ├── COINGLASS_ENDPOINTS.md
│       └── WHALE_ALERT_ENDPOINTS.md
├── generated-page.html       # Main trading terminal UI
├── prompts\
│   ├── IROS_ALERT_AGENT.md   # Alert generation
│   └── MCF_REPORT_AGENT.md   # Report generation
└── requirements.txt
```

## 9.2 Training Corpus

```
C:\Users\Banke\IROS_72B_TRAINING_CORPUS\FINAL_TRAINING\
├── iros_72b_train.jsonl            # 600 examples
├── risk_intelligence_*.jsonl       # 42 examples
├── bastion_api_knowledge.jsonl     # 9 examples
├── mcf_intelligence_reports.jsonl  # 39 examples
├── MCF_RISK_EXIT_LOGIC_EXTRACTION.md  # Full MCF methodology
└── training_manifest.json          # Metadata
```

---

# SUMMARY

**BASTION has access to:**
- 33 free quant endpoints (Helsinki VM)
- Premium liquidation/funding/OI data (Coinglass)
- On-chain whale tracking (Whale Alert)
- 4x RTX 5090 GPU cluster running bastion-32b
- Full MCF risk methodology
- 797 training examples

**The AI needs to be upgraded to:**
- Be world-class at trade exits
- Synthesize multiple data sources intelligently
- Generate smarter, more relevant alerts
- Handle more edge cases in position management
- Know when to defer vs when to act

**Claude should:**
1. Review all available data endpoints
2. Understand the MCF methodology
3. Propose upgrades to training data
4. Suggest improvements to system prompts
5. Help build more Risk Intelligence scenarios
6. Make BASTION the best trading AI possible

---

# PART 10: TERMINAL API ENDPOINTS (terminal_api.py)

## 10.1 Core Data Endpoints

| Endpoint | Method | Purpose | Data Source |
|----------|--------|---------|-------------|
| `/api/price/{symbol}` | GET | Live price with 24h change | Coinglass |
| `/api/klines/{symbol}` | GET | Candlestick data | Exchange |
| `/api/orderflow/{symbol}` | GET | CVD, orderbook imbalance | Helsinki |
| `/api/volatility-regime/{symbol}` | GET | Volatility regime, percentile | Helsinki |
| `/api/liquidation-map/{symbol}` | GET | Liquidation clusters | Helsinki + Coinglass |
| `/api/fear-greed` | GET | Fear & Greed index | Helsinki |
| `/api/smart-money/{symbol}` | GET | Institutional flow | Helsinki |

## 10.2 Coinglass Premium Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/coinglass/funding-rates/{symbol}` | GET | Funding rates all exchanges |
| `/api/coinglass/open-interest/{symbol}` | GET | OI by exchange |
| `/api/coinglass/long-short-ratio/{symbol}` | GET | L/S ratio |
| `/api/coinglass/liquidations/{symbol}` | GET | Recent liquidations |
| `/api/coinglass/oi-changes` | GET | Top 10 OI increases/decreases |

## 10.3 Whale Alert Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/whale-transactions` | GET | Recent whale txs |
| `/api/exchange-flows/{symbol}` | GET | Exchange inflows/outflows |

## 10.4 AI Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ask-bastion` | POST | Query BASTION AI |
| `/api/risk-evaluate` | POST | Risk Intelligence evaluation |
| `/api/alerts/generate` | POST | Generate market alerts |
| `/api/market-pulse` | POST | Full market pulse report |

## 10.5 User & Position Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/positions` | GET | User positions from connected exchanges |
| `/api/auth/connect-exchange` | POST | Connect exchange with API keys |
| `/api/auth/profile` | GET/POST | User profile management |
| `/api/sync/upload` | POST | Save settings to cloud |
| `/api/panel-layout` | GET/POST | Save/load panel layout |

## 10.6 Alert & Telegram Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/live-alerts` | GET | Get recent alerts |
| `/api/telegram/push` | POST | Push alert to Telegram |
| `/api/telegram/test` | POST | Test Telegram connection |
| `/api/alerts/telegram/connect` | GET | Generate Telegram connect link |

## 10.7 Calculation Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/pre-trade-calculator` | POST | 50K Monte Carlo sim |
| `/api/usdt-dominance` | GET | USDT dominance data |

---

# PART 11: CURRENT BASTION AI INTEGRATION CODE

## 11.1 BastionAI Class (from bastion_ai.py)

The `BastionAI` class in `iros_integration/services/bastion_ai.py`:

1. **Extracts query context** - Symbol, capital, timeframe, risk tolerance
2. **Detects query type** - CVD, volatility, liquidation, general
3. **Fetches data** - Coinglass (priority) + Helsinki
4. **Builds system prompt** - Institutional format with live data
5. **Queries the model** - vLLM endpoint
6. **Returns structured result** - Response, sources, latency

Key methods:
- `process_query(query, comprehensive, user_context)` - Main entry point
- `_fetch_coinglass_data(symbol)` - Gets real prices, whale positions, funding, OI
- `_build_system_prompt(market_context, context)` - Creates institutional prompt
- `_query_model(user_query, system_prompt)` - Calls vLLM

## 11.2 How to Call the AI (Updated Code)

```python
import httpx

# CONFIRMED WORKING ENDPOINT
VLLM_URL = "http://74.48.140.178:26717/v1/chat/completions"

async def ask_bastion_chat(user_message: str, market_data: dict = None):
    """For conversational AI chat"""
    system_prompt = """You are BASTION - an institutional-grade crypto trading AI with access to Helsinki VM (33 endpoints), Coinglass Premium ($299/mo for liquidations, OI, funding, L/S ratios), and Whale Alert Premium ($29.95/mo for on-chain tracking).

CRITICAL: USE ONLY THE DATA PROVIDED. NEVER INVENT NUMBERS.
No emojis. Be precise, quantified, actionable."""

    if market_data:
        system_prompt += f"\n\nLIVE DATA:\n{market_data}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(VLLM_URL, json={
            "model": "bastion-32b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        })
        return response.json()["choices"][0]["message"]["content"]


async def ask_bastion_risk(position_data: dict, market_data: dict):
    """For Risk Intelligence position management"""
    system_prompt = """You are BASTION Risk Intelligence - an autonomous trade management AI. You monitor live positions and make execution decisions.

PRIORITY ORDER:
1) Hard Stop - Maximum loss, NON-NEGOTIABLE
2) Safety Net Break - Long-term structure broken
3) Guarding Line Break - Dynamic trailing stop broken  
4) Take Profit Targets - Structure-based exits
5) Trailing Stop - ATR-based dynamic stop
6) Time Exit - Position duration limits

OUTPUT JSON FORMAT:
{
  "action": "HOLD|TP_PARTIAL|TP_FULL|MOVE_STOP_TO_BREAKEVEN|TRAIL_STOP|EXIT_FULL|REDUCE_SIZE",
  "exit_percentage": 0-100,
  "new_stop": price or null,
  "reasoning": "explanation",
  "confidence": 0.0-1.0,
  "warnings": ["array of warnings"],
  "next_evaluation": "when to check again"
}"""

    user_message = f"""POSITION:
{json.dumps(position_data, indent=2)}

MARKET DATA:
{json.dumps(market_data, indent=2)}

Evaluate this position and provide your recommendation."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(VLLM_URL, json={
            "model": "bastion-32b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 500,
            "temperature": 0.3  # Lower for more consistent outputs
        })
        return response.json()["choices"][0]["message"]["content"]
```

---

# PART 12: WHAT CLAUDE SHOULD DO

## 12.1 Immediate Tasks

1. **Review this context** - Understand all data sources, endpoints, MCF methodology
2. **Identify gaps** - What scenarios aren't covered in training?
3. **Propose upgrades** - Better system prompts, more training data, smarter alerts

## 12.2 Potential Upgrades

| Area | Current State | Upgrade Opportunity |
|------|---------------|---------------------|
| **Exit Detection** | 42 training examples | Expand to 200+ scenarios |
| **Alert Intelligence** | Basic threshold triggers | Multi-signal synthesis |
| **Confidence Calibration** | Fixed confidence scores | Dynamic uncertainty awareness |
| **Data Synthesis** | Sequential data fetching | Parallel multi-source fusion |
| **Position Context** | Snapshot evaluation | Continuous monitoring |

## 12.3 Questions for Claude

1. What additional training scenarios would make Risk Intelligence more robust?
2. How can we better synthesize Helsinki + Coinglass + Whale Alert?
3. What edge cases in position management need more coverage?
4. How should the AI handle conflicting signals from different sources?
5. What's the optimal system prompt structure for each task?

---

**This document contains EVERYTHING about BASTION. Use it to upgrade the system.**
