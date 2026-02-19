# BASTION — System Architecture

**Last Updated: 2026-02-19**
**Status: Production (Railway Pro)**

---

## Overview

BASTION is an institutional-grade crypto trading intelligence platform. It combines a custom fine-tuned 32B parameter LLM running on a GPU cluster with real-time market data from multiple premium APIs, autonomous risk management, execution capabilities, and automated research report generation.

The stack: **FastAPI backend on Railway** + **Qwen 32B fine-tuned model on Vast.ai GPU cluster** + **Helsinki VM (33 free quant endpoints)** + **Coinglass Premium** + **Whale Alert Premium** + **Supabase (auth + storage)** + **6 exchange connectors**.

---

## High-Level Architecture

```
                           bastionfi.tech (Railway Pro)
                          ┌─────────────────────────────┐
                          │  FastAPI / Uvicorn (Python)  │
                          │     terminal_api.py          │
                          │     ~8600 lines, 120+ routes │
                          └──────┬──────┬──────┬────────┘
                                 │      │      │
              ┌──────────────────┤      │      ├──────────────────┐
              │                  │      │      │                  │
   ┌──────────▼──────────┐  ┌───▼──────▼───┐  │  ┌──────────────▼──────────────┐
   │   GPU Cluster        │  │ Data Layer   │  │  │   Frontend Pages             │
   │   Vast.ai 4xRTX5090 │  │              │  │  │                              │
   │                      │  │ Helsinki VM  │  │  │ / ............. Landing       │
   │ vLLM OpenAI-compat   │  │ Coinglass    │  │  │ /terminal ..... Pro Terminal  │
   │ bastion-32b model    │  │ Whale Alert  │  │  │ /research ..... MCF Reports   │
   │ Risk Intelligence    │  │ Binance      │  │  │ /monitor ...... PULSE         │
   │ Neural Chat          │  │ Yahoo Finance│  │  │ /lite ......... Mobile App    │
   └──────────────────────┘  │ FRED         │  │  │ /login ........ Auth          │
                             └──────────────┘  │  │ /account ...... Settings      │
                                               │  └───────────────────────────────┘
                          ┌────────────────────┤
                          │                    │
               ┌──────────▼──────┐  ┌──────────▼──────────┐
               │   Supabase       │  │  Exchange Connectors │
               │                  │  │                      │
               │ Auth (users)     │  │ BloFin    Bitunix    │
               │ MCF Reports      │  │ Bybit     OKX       │
               │ Exchange Keys    │  │ Binance   Deribit    │
               │ Settings         │  │                      │
               └──────────────────┘  └──────────────────────┘
```

---

## Deployment

### Railway (Production Backend)

| Setting | Value |
|---------|-------|
| Platform | Railway Pro |
| Runtime | Python 3.11.7 |
| Framework | FastAPI + Uvicorn |
| Entry | `api/terminal_api.py` → `app` |
| Domain | `bastionfi.tech` / `web-production-93e47.up.railway.app` |
| Deploy | Auto-deploy on push to `main` branch |

**Environment Variables (Railway):**

| Variable | Purpose |
|----------|---------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase publishable key |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase admin key |
| `BASTION_MODEL_URL` | GPU cluster vLLM endpoint (e.g. `http://70.78.148.208:56333`) |
| `COINGLASS_API_KEY` | Coinglass Premium ($299/mo) |
| `WHALE_ALERT_API_KEY` | Whale Alert Premium ($30/mo) |
| `ADMIN_KEY` | Admin authentication key |
| `ENCRYPTION_KEY` | Exchange key encryption (Fernet) |
| `ALLOWED_ORIGINS` | CORS whitelist (comma-separated) |

### GPU Cluster (Vast.ai)

| Setting | Value |
|---------|-------|
| Provider | Vast.ai |
| GPUs | 4x RTX 5090 (32GB VRAM each) |
| Model | `bastion-risk-v6-merged` (LIVE) |
| Server | vLLM OpenAI-compatible API |
| Tensor Parallelism | 4 (all GPUs on every request) |
| Python | `/opt/sys-venv/bin/python3` |
| Internal port | 8000 (vLLM) |
| Port forward | socat on port 6006 → 8000 |
| External port | Assigned by Vast.ai (changes on reboot, check dashboard) |
| Model name in API | `bastion-32b` |

**vLLM Launch Command:**
```bash
/opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /root/models/bastion-risk-v6-merged \
    --served-model-name bastion-32b \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --host 0.0.0.0 \
    --port 8000
```

**Port Forward (must restart after every vLLM restart):**
```bash
nohup socat TCP-LISTEN:6006,fork,reuseaddr TCP:localhost:8000 > /dev/null 2>&1 &
```

**Critical Notes:**
- SSH port changes on reboot — always check Vast.ai dashboard
- `--gpu-memory-utilization 0.85` not 0.90 (OOM on sampler warmup)
- `--max-num-seqs 64` required to avoid OOM
- socat dies between sessions — must restart
- Must stop vLLM before training (frees ~30GB per GPU)

---

## Backend Architecture

### Entry Point

`api/terminal_api.py` (~8600 lines) is the monolithic FastAPI application. It handles:
- All HTTP routes (120+ endpoints)
- WebSocket connections for real-time updates
- Client initialization (Helsinki, Coinglass, Whale Alert)
- Static file serving (HTML pages)
- Proxy routes for external APIs

### Core Modules

| File | Purpose |
|------|---------|
| `api/terminal_api.py` | Main API — all routes, middleware, page serving |
| `api/risk_engine.py` | Autonomous position monitoring loop with MCF Exit Hierarchy |
| `api/execution_engine.py` | Translates risk decisions into exchange orders |
| `api/models.py` | Pydantic data models |
| `api/session_routes.py` | Session management routes |
| `api/user_service.py` | User auth + Supabase integration |

### Intelligence Services

| File | Purpose |
|------|---------|
| `iros_integration/services/bastion_ai.py` | BastionAI class — query processing, prompt building, model inference |
| `iros_integration/services/helsinki.py` | Helsinki VM client — 33 quant endpoints |
| `iros_integration/services/coinglass.py` | Coinglass Premium client — liquidations, OI, funding, L/S ratios |
| `iros_integration/services/whale_alert.py` | Whale Alert client — on-chain whale tracking |
| `iros_integration/services/exchange_connector.py` | Multi-exchange connector (BloFin, Bitunix, Bybit, OKX, Binance, Deribit) |
| `iros_integration/services/query_processor.py` | NLP context extraction from user queries |
| `iros_integration/config/settings.py` | Centralized configuration (all env vars) |

### MCF Structure Analysis (core/)

| File | Purpose |
|------|---------|
| `core/structure_service.py` | Orchestrator — combines VPVR + Pivots + Auto-Support, caches per timeframe |
| `core/vpvr_analyzer.py` | Volume Profile Visible Range (250 bins, buy/sell split, HVN/LVN/POC detection) |
| `core/structure_detector.py` | Pivot detection (asymmetric 20/15), trendline grading (Grade 1-4), pressure points |
| `core/auto_support.py` | Auto-Support (16 sensitivity levels, priority scoring, ported from Pine Script) |
| `core/tp_sizer.py` | Structure-aware TP sizing — deterministic math for exit percentages |
| `core/risk_engine.py` | Risk calculation engine |
| `core/adaptive_budget.py` | Adaptive position budget calculator |
| `core/orderflow_detector.py` | Order flow pattern detection |
| `core/mtf_structure.py` | Multi-timeframe structure analysis |
| `core/session.py` | Session state management |

### MCF Labs (Report Generation)

| File | Purpose |
|------|---------|
| `mcf_labs/generator.py` | Base report generator — market structure, whale intelligence reports |
| `mcf_labs/iros_generator.py` | LLM-powered report generator (extends base with BastionAI insights) |
| `mcf_labs/institutional_generator.py` | Goldman-grade institutional research reports |
| `mcf_labs/scheduler.py` | Automated report scheduling (BTC, ETH, SOL + 6 alt coins) |
| `mcf_labs/storage.py` | Hybrid storage (filesystem + Supabase) |
| `mcf_labs/supabase_storage.py` | Supabase-backed persistent report storage |
| `mcf_labs/models.py` | Report data models (ReportType, Bias, Confidence, TradeScenario) |

### Data Layer

| File | Purpose |
|------|---------|
| `data/fetcher.py` | Live OHLCV fetcher — Helsinki VM primary, Binance fallback, Bybit second fallback |
| `data/live_feed.py` | Real-time data feed management |

---

## API Endpoints

### Page Routes (HTML Serving)
| Route | Serves | File |
|-------|--------|------|
| `GET /` | Landing page | `web/index.html` |
| `GET /terminal` | Pro Terminal | `generated-page.html` |
| `GET /research` | MCF Research | `web/research.html` |
| `GET /monitor` | PULSE (World Monitor) | `web/monitor.html` |
| `GET /lite` | Mobile web app | `bastion lite.html` |
| `GET /login` | Authentication | `web/login.html` |
| `GET /account` | Account settings | `web/account.html` |

### AI Endpoints
| Route | Purpose |
|-------|---------|
| `POST /api/neural/chat` | Chat with BastionAI (includes position context, verified market data) |
| `GET /api/neural/chat/history` | Chat history |
| `POST /api/risk/evaluate` | Risk Intelligence — autonomous position evaluation (4 parallel data sources + structure) |
| `POST /api/risk/evaluate-all` | Evaluate all positions simultaneously |
| `POST /api/signals/scan` | Signal scanner |

### Exchange & Position Management
| Route | Purpose |
|-------|---------|
| `POST /api/exchange/connect` | Connect exchange (API key encrypted + stored in Supabase) |
| `GET /api/exchange/list` | List connected exchanges |
| `GET /api/exchange/{name}/positions` | Fetch positions from exchange |
| `GET /api/exchange/{name}/balance` | Fetch balance |
| `POST /api/exchange/{name}/sync` | Force sync positions |
| `GET /api/positions/all` | All positions across all exchanges |
| `GET /api/balance/total` | Combined balance |

### Execution Engine
| Route | Purpose |
|-------|---------|
| `POST /api/engine/start` | Start autonomous risk engine |
| `POST /api/engine/stop` | Stop engine |
| `POST /api/engine/arm` | Arm execution (enable auto-trading) |
| `POST /api/engine/disarm` | Disarm execution |
| `POST /api/engine/kill-switch` | Emergency kill switch |
| `POST /api/actions/emergency-exit` | Emergency exit a position |
| `POST /api/actions/partial-close` | Partial close |
| `POST /api/actions/set-take-profit` | Set TP on exchange |
| `POST /api/actions/set-stop-loss` | Set SL on exchange |
| `POST /api/actions/move-to-breakeven` | Move stop to breakeven |
| `POST /api/actions/flatten-winners` | Close all winning positions |

### Market Data
| Route | Purpose |
|-------|---------|
| `GET /api/price/{symbol}` | Live price (cached 0.5s) |
| `GET /api/klines/{symbol}` | Candlestick data |
| `GET /api/market/{symbol}` | Full market overview |
| `GET /api/cvd/{symbol}` | Cumulative Volume Delta |
| `GET /api/volatility/{symbol}` | Volatility regime |
| `GET /api/funding` | Funding rates across exchanges |
| `GET /api/oi/{symbol}` | Open interest |
| `GET /api/fear-greed` | Fear & Greed Index |
| `GET /api/usdt-dominance` | USDT dominance |
| `GET /api/etf-flows` | BTC/ETH ETF flows |
| `GET /api/news` | Crypto news aggregation |

### Derivatives / Institutional Data (Coinglass)
| Route | Purpose |
|-------|---------|
| `GET /api/liquidations/{symbol}` | Liquidation data |
| `GET /api/coinglass/liquidations/{symbol}` | Detailed liquidation map |
| `GET /api/coinglass/overview/{symbol}` | Market overview |
| `GET /api/top-traders/{symbol}` | Top trader positions |
| `GET /api/options/{symbol}` | Options data |
| `GET /api/funding-arb/{symbol}` | Funding arbitrage |
| `GET /api/oi-exchange/{symbol}` | OI by exchange |
| `GET /api/taker-ratio/{symbol}` | Taker buy/sell ratio |
| `GET /api/exchange-flow/{symbol}` | Exchange inflow/outflow |
| `GET /api/liq-exchange/{symbol}` | Liquidations by exchange |
| `GET /api/volatility-regime/{symbol}` | Volatility regime analysis |
| `GET /api/mm-magnet/{symbol}` | Market maker magnet levels |

### On-Chain / Order Flow
| Route | Purpose |
|-------|---------|
| `GET /api/whales` | Whale transactions |
| `GET /api/whales/flows/{symbol}` | Whale flows by symbol |
| `GET /api/onchain` | On-chain analytics |
| `GET /api/orderflow/{symbol}` | Order flow analysis |
| `GET /api/heatmap/{symbol}` | Liquidation heatmap |

### MCF Reports
| Route | Purpose |
|-------|---------|
| `GET /api/mcf/reports` | List reports (paginated) |
| `GET /api/mcf/reports/latest` | Latest report |
| `GET /api/mcf/reports/{id}` | Single report |
| `POST /api/mcf/generate/{type}` | Generate report (market_structure, whale_intelligence, etc.) |
| `POST /api/mcf/generate/institutional` | Generate institutional research |
| `POST /api/mcf/generate/institutional/batch` | Batch generate for all coins |
| `GET /api/mcf/status` | Scheduler status |

### Auth & User
| Route | Purpose |
|-------|---------|
| `POST /api/auth/register` | Register new user |
| `POST /api/auth/login` | Login |
| `POST /api/auth/logout` | Logout |
| `GET /api/auth/me` | Current user |
| `POST /api/auth/2fa/setup` | Setup 2FA (TOTP) |
| `POST /api/auth/2fa/verify` | Verify 2FA |
| `POST /api/auth/exchange-keys` | Store encrypted exchange keys |

### Quantitative Tools
| Route | Purpose |
|-------|---------|
| `POST /api/pre-trade-calculator` | Pre-trade risk calculator |
| `POST /api/kelly` | Kelly Criterion calculator |
| `POST /api/monte-carlo` | Monte Carlo simulation |
| `GET /api/macro` | Macro economic indicators |
| `GET /api/session` | Session performance stats |

### Alerts / Telegram
| Route | Purpose |
|-------|---------|
| `GET /api/alerts/telegram/connect` | Generate Telegram connect link |
| `POST /api/alerts/telegram/verify` | Verify Telegram connection |
| `POST /api/alerts/telegram/webhook` | Telegram webhook handler |
| `POST /api/alerts/generate` | Generate automated alerts |
| `POST /api/telegram/push` | Push alert to Telegram |

### Proxy
| Route | Purpose |
|-------|---------|
| `GET /wm-proxy/{path}` | World Monitor data proxy (CORS bypass for PULSE page) |

### WebSocket
| Route | Purpose |
|-------|---------|
| `WS /ws` | Real-time updates (positions, prices, alerts, engine events) |

---

## Frontend Pages

### Landing Page (`web/index.html`)
- Marketing/landing page at `bastionfi.tech`
- Nav: **Bastion AI** | **Terminal** | **Research** | **Pulse** | **How It Works**
- Animated ticker marquee (SPX, Oil, BTC, VIX, DOW via Yahoo Finance)

### Pro Terminal (`generated-page.html`)
- Full trading terminal at `/terminal`
- Panels: Positions, Risk Intelligence, Neural Chat, Market Data, Charts, Order Flow
- MCF Labs Latest Report section (live from `/api/mcf/reports`)
- Splash screen on load
- Live data stream with 2-minute refresh

### PULSE (`web/monitor.html`)
- Real-time financial dashboard at `/monitor`
- 20 panels: Headlines, Live News (YouTube), Webcams, Market News, Live Markets, Crypto, Sector Heatmap, Commodities, Futures, Economic Data (FRED), ETF Flows, Stablecoins, DeFi, Fear/Greed, Radar
- Yahoo Finance API wrapper (`fetchYahoo()`) for all market data
- FRED API integration for economic indicators
- RSS news aggregation with time range filtering
- Leaflet.js map with GeoJSON markers
- Time range selector (1h/6h/24h/48h/7d/All)

### Research (`web/research.html`)
- MCF Labs research reports at `/research`
- Displays institutional-grade market analysis
- Splash screen on load

### BASTION Lite (`bastion lite.html`)
- Mobile-optimized web app at `/lite`
- Core terminal features in responsive layout

### Other Pages
| File | Route | Purpose |
|------|-------|---------|
| `web/login.html` | `/login` | Authentication page |
| `web/account.html` | `/account` | Account settings, 2FA, exchange keys |
| `web/dashboard.html` | `/visualizations` | Data visualizations |
| `web/trade-manager.html` | — | Trade management interface |
| `web/institutional.html` | — | Institutional features (removed from nav) |

### Shared Frontend Files
| File | Purpose |
|------|---------|
| `web/api-client.js` | API client library for all frontend pages |
| `web/chart.js` | Chart rendering (TradingView-style) |
| `web/volume-profile.js` | Volume profile visualization |
| `web/styles.css` | Shared styles |
| `web/settings.js` | Settings management |

---

## Data Sources

### Helsinki VM (FREE — 33 Quant Endpoints)
- **URL**: `http://77.42.29.188:5002`
- **What**: Real-time quantitative data network
- **Endpoints**: CVD, Orderbook, Large Trades, Smart Money, Whale Flow, Basis, Open Interest, Greeks, Liquidation Map, Liquidation Estimate, Options IV, IV/RV Spread, Volatility, VWAP, Momentum, Mean Reversion, Drawdown, and more
- **Used by**: Neural Chat, Risk Intelligence, MCF Structure Analysis

### Coinglass Premium ($299/mo)
- **URL**: `https://open-api-v3.coinglass.com/api`
- **What**: Institutional-grade derivatives data
- **Endpoints**: Coins Markets, Hyperliquid Whale Positions, Funding Rates, Open Interest, Long/Short Ratios, Liquidation Data, Options Max Pain, Top Traders
- **Used by**: Neural Chat, Risk Intelligence, MCF Report Generation

### Whale Alert Premium ($30/mo)
- **URL**: `https://api.whale-alert.io/v1` + WebSocket
- **What**: On-chain whale transaction tracking
- **Min value**: $1M+ transactions
- **Used by**: Risk Intelligence, Whale Reports, Alerts

### Yahoo Finance (via `/wm-proxy/`)
- **What**: Stock/index/commodity prices for PULSE dashboard
- **Response format**: `{chart:{result:[{meta:{regularMarketPrice,...}}]}}`
- **Used by**: PULSE page (Live Markets, Sector Heatmap, Commodities, Ticker)

### FRED (Federal Reserve Economic Data)
- **What**: Economic indicators (GDP, CPI, Unemployment, Fed Rate, etc.)
- **Response format**: `{observations:[{date, value}]}`
- **Used by**: PULSE page (Economic Data panel)

### Binance / Bybit (OHLCV fallback)
- **What**: Candlestick data for structure analysis
- **Priority**: Helsinki VM → Binance → Bybit
- **Used by**: Structure Service (VPVR, Pivot detection, Auto-Support)

---

## AI Model

### Architecture
- **Base**: Qwen/Qwen2.5-Coder-32B-Instruct (32 billion parameters)
- **Fine-tune method**: QLoRA (4-bit NF4 quantized base + LoRA adapters)
- **LoRA config**: rank=32, alpha=64, dropout=0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Serving**: vLLM with OpenAI-compatible API

### Model Lineage (Rebuild Chain)
```
Qwen/Qwen2.5-Coder-32B-Instruct (base — download from HuggingFace)
  └── + iros-mega-lora (IROS intelligence) = iros-merged
        └── + bastion-risk-v3-lora (230 examples, 71.4%) = v3-merged
              └── + bastion-risk-v5-lora (290 examples, 65.9%) = v5-merged
                    └── + bastion-risk-v6-lora (328 examples, 75.4%) = v6-merged  ★ LIVE
```

### Two Tasks, One Model
The same `bastion-32b` model serves two different tasks via different system prompts:

| Task | Endpoint | Temperature | Output |
|------|----------|-------------|--------|
| Risk Intelligence | `POST /api/risk/evaluate` | 0.3 | Structured JSON (action, confidence, reasoning) |
| Neural Chat | `POST /api/neural/chat` | 0.6 | Markdown analysis with trade setups |

### Risk Intelligence Actions
The model evaluates positions and returns one of:
- **HOLD** — Keep position, no changes
- **TP_PARTIAL** — Take partial profit at current TP level
- **EXIT_FULL** — Close entire position
- **EXIT_100%** — Emergency exit (stop breached, critical risk)
- **REDUCE_SIZE** — Reduce position size
- **TRAIL_STOP** — Tighten trailing stop
- **MOVE_STOP_BE** — Move stop to breakeven
- **ADJUST_STOP** — Adjust stop loss level

### Risk Evaluation Data Pipeline
When `/api/risk/evaluate` is called, 4 parallel data fetches execute:
1. **Coinglass** — Price, OI, Funding, Whale positions, Liquidations
2. **Helsinki** — CVD, Volatility, Liquidation estimates, Momentum, Smart Money, Orderbook, VWAP
3. **Whale Alert** — Recent $1M+ transactions
4. **MCF Structure** — VPVR (250 bins), Pivot points, Auto-Support levels, Trendline grades

All data is injected into the system prompt (~350 tokens for structure context alone), then sent to the model for inference.

### MCF Structure-Based Exits
Replaces arbitrary ATR trailing stops with structural intelligence:
- Grade 3+ support/resistance break detection
- VPVR-informed trailing (HVN = slow, LVN = fast)
- Pressure point confluence (trendline meets horizontal level)
- Multi-timeframe analysis (15m: 3min cache, 1h: 5min cache, 4h: 10min cache)

---

## Backtest Performance

| Version | Combined | BTC | ETH | SOL | Examples | Status |
|---------|----------|-----|-----|-----|----------|--------|
| v1 | ~54% | — | — | — | 168 | Superseded |
| v2 | 54.3% | — | — | — | 200 | Superseded |
| v3 | 71.4% | — | — | — | 230 | Backup |
| v4 | 47.8% | — | — | — | 265 | ABANDONED |
| v5 | 65.9% | — | — | — | 290 | Backup |
| **v6** | **75.4%** | **71.7%** | **72.7%** | **81.8%** | **328** | **LIVE** |
| v7 | ~66% | 76.7% | 66.0% | 56.5% | 371 | ABANDONED |
| v7.1 | ~68% | 62.5% | 76.2% | 66.7% | 358 | ABANDONED |

**v6 Extended Pairs**: AVAX 68.2%, XRP 59.5% (FAIL), DOGE 89.6%, LINK 77.8%, ADA 74.5%

**v6 Action Accuracy**: EXIT_100%: 100%, EXIT_FULL: 55-76.5%, HOLD: 66.7-80%, TP_PARTIAL: 60-75%

### Lessons Learned (v4, v7, v7.1)
- Small reinforcement batches (30-43 examples on 328 base) are destabilizing
- Fixing one action type breaks others (seesaw effect)
- v6 is the stability ceiling for this approach
- Future fine-tuning: only attempt with 100+ validated examples, even distribution, held-out test set

---

## GPU Backup & Model Rebuild

### Local Backups

All LoRA adapters are saved locally — the full model can be rebuilt from scratch:

```
C:\Users\Banke\BASTION_GPU_BACKUP\
  lora_adapters\
    bastion-risk-v6-lora\     528MB  ★ CURRENT LIVE
    bastion-risk-v5-lora\     528MB
    bastion-risk-v3-lora\     528MB
    bastion-mega-lora\        1.1GB  (original IROS)
    bastion-risk-lora\        528MB  (v1)
    h200-lora\                2.8GB  (original IROS from H200)
  training_data\              Category-specific JSONL files
  scripts\                    Fine-tune, merge, training scripts
  logs\                       Training & vLLM logs
  vllm_config\                vLLM launch configs

C:\Users\Banke\BASTION\backups\
  v6-lora\                    Copy of v6 LoRA adapter
  v5-merged\                  Partial (4/14 shards — not needed since LoRAs are saved)

C:\Users\Banke\BASTION\BASTION\training_data\
  bastion_risk_v6_combined.jsonl    328 examples (the LIVE training data)
  bastion_risk_v5_combined.jsonl    290 examples
  bastion_risk_v3_combined.jsonl    230 examples
  + all reinforcement JSONL files (v3-v7.1)
  + all generator scripts (generate_v3_examples.py through generate_v7_1_examples.py)
  + all finetune scripts (finetune_v2.py through finetune_v7_1.py)
  + all merge scripts (merge_v4.py through merge_v7_1.py)
```

### Rebuild on a New GPU Instance

**Prerequisites:**
- 4x GPUs with 32GB+ VRAM each (RTX 5090, A100, H100)
- 200GB disk minimum
- CUDA 12.x + Python 3.10+

**Step 1: Install dependencies**
```bash
pip install torch transformers peft trl bitsandbytes datasets accelerate vllm safetensors
```

**Step 2: Download base model**
```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-32B-Instruct', torch_dtype='float16', device_map='auto')
model.save_pretrained('/root/models/base-qwen32b')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-32B-Instruct')
tokenizer.save_pretrained('/root/models/base-qwen32b')
"
```

**Step 3: Upload LoRA adapters from backup**
```bash
# SCP from local machine:
scp -P <SSH_PORT> -r C:\Users\Banke\BASTION_GPU_BACKUP\lora_adapters\* root@<IP>:/workspace/
```

**Step 4: Sequential merge (apply LoRA chain)**
```bash
# Each merge step follows this pattern:
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained('/root/models/PREVIOUS', torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa')
model = PeftModel.from_pretrained(base, '/workspace/LORA_ADAPTER')
model = model.merge_and_unload()
model.save_pretrained('/root/models/NEW_MERGED', safe_serialization=True)
AutoTokenizer.from_pretrained('/workspace/LORA_ADAPTER').save_pretrained('/root/models/NEW_MERGED')
"

# Merge order:
# 1. base-qwen32b + h200-lora/iros-mega-lora-2026-02-01 → iros-merged
# 2. iros-merged + bastion-risk-v3-lora → v3-merged
# 3. v3-merged + bastion-risk-v5-lora → v5-merged
# 4. v5-merged + bastion-risk-v6-lora → v6-merged  ★ LIVE
```

**Step 5: Start vLLM + socat**
```bash
# Terminal 1: vLLM
/opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /root/models/bastion-risk-v6-merged \
    --served-model-name bastion-32b \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --host 0.0.0.0 --port 8000

# Terminal 2: Port forward
nohup socat TCP-LISTEN:6006,fork,reuseaddr TCP:localhost:8000 > /dev/null 2>&1 &
```

**Step 6: Update Railway env var**
Set `BASTION_MODEL_URL=http://<NEW_IP>:<VAST_EXTERNAL_PORT>` in Railway dashboard.

### Training Gotchas (from experience)
- Use `attn_implementation="sdpa"` not `"flash_attention_2"` (flash_attn not installed)
- SFTConfig uses `max_length` not `max_seq_length` (trl 0.28.0)
- SFTTrainer uses `processing_class` not `tokenizer` (trl 0.28.0)
- Use `dtype=torch.bfloat16` not `torch_dtype=torch.bfloat16` (transformers 4.57.6)
- Must stop vLLM before training (frees ~30GB per GPU)
- Merge must be done separately after training (OOM if done in same process)

---

## Training Data

Format: ChatML (system/user/assistant) with structured JSON assistant responses.

| File | Examples | Purpose |
|------|----------|---------|
| `bastion_risk_v6_combined.jsonl` | 328 | v6 LIVE training data (v5 base + 38 structure-aware) |
| `bastion_risk_v5_combined.jsonl` | 290 | v5 (v3 base + 60 targeted reinforcement) |
| `bastion_risk_v3_combined.jsonl` | 230 | v3 base training data |
| `bastion_risk_v6_reinforcement.jsonl` | 38 | v6 new examples (structure-aware across 10 pairs) |
| `bastion_risk_v5_reinforcement.jsonl` | 60 | v5 reinforcement (6 categories) |

**v5 Reinforcement Categories (60 examples):**
1. LONG stop direction logic (10) — symmetric with SHORT
2. SHORT stop direction logic (10) — refined from v4
3. TP1 proximity + leverage → TP_PARTIAL (8) — fix "hold through TP1"
4. No-stop leverage tiers (20) — 1x=hold, 3x=warn, 5x=reduce, 10x=reduce/exit, 20x=always exit
5. Funding alone ≠ exit signal (6) — prevent panic exits
6. Breached stop = EXIT always (6) — unconditional exit rule

**v6 Reinforcement (38 examples):**
- Structure-aware reasoning across 10 crypto pairs
- VPVR, graded S/R, POC, HVN terminology embedded in model outputs

---

## Automation Scripts

### SSH Pipeline Scripts (run from Windows)
| File | Purpose |
|------|---------|
| `ssh_run_pipeline_v7_1.js` | Full SSH pipeline: kill vLLM → fine-tune → merge → restart (template) |
| `ssh_revert_v6.js` | Quick revert to v6 on GPU cluster |
| `ssh_mass_backtest.js` | Mass backtest runner (BTC/ETH/SOL, 50 each) |
| `ssh_mass_backtest_extended.js` | Extended pair backtester (AVAX/XRP/DOGE/LINK/ADA/MATIC) |

### Backtest
| File | Purpose |
|------|---------|
| `backtest/bastion_backtest.py` | Main backtester — fetches historical data, simulates positions, scores model |
| `backtest_with_structure.py` | Structure-enhanced backtest (VPVR + graded S/R) |

---

## Database (Supabase)

### Tables
| Table | Purpose |
|-------|---------|
| `users` | User accounts (email, hashed password, profile) |
| `exchange_keys` | Encrypted exchange API keys (Fernet encryption) |
| `mcf_reports` | Generated research reports |
| `user_settings` | User preferences (risk, alerts, appearance) |

### Auth Flow
1. User registers → bcrypt password hash → stored in Supabase
2. Login returns session token → stored client-side
3. Exchange keys encrypted with `ENCRYPTION_KEY` (Fernet) before Supabase storage
4. 2FA via TOTP (pyotp) with QR code generation

---

## Security

- Exchange keys: Fernet-encrypted at rest in Supabase
- Passwords: bcrypt hashed
- 2FA: TOTP-based (Google Authenticator compatible)
- CORS: Whitelist-based (`ALLOWED_ORIGINS`)
- Admin routes: Protected by `ADMIN_KEY`
- Read-only exchange connections by default
- Execution engine has safety guards: max close %, min confidence, hourly limits, daily loss limit, kill switch

---

## File Index

```
BASTION/
├── api/
│   ├── terminal_api.py          # Main API (8600 lines, 120+ routes)
│   ├── risk_engine.py           # Autonomous risk monitoring
│   ├── execution_engine.py      # Order execution translator
│   ├── models.py                # Data models
│   ├── session_routes.py        # Session management
│   ├── user_service.py          # Auth + Supabase
│   └── index.py                 # Vercel entry point (legacy)
├── core/
│   ├── structure_service.py     # MCF structure orchestrator
│   ├── vpvr_analyzer.py         # Volume Profile (250 bins)
│   ├── structure_detector.py    # Pivots + trendline grading
│   ├── auto_support.py          # 16-level auto-support (Pine Script port)
│   ├── tp_sizer.py              # Structure-aware TP sizing
│   ├── risk_engine.py           # Risk calculation
│   ├── adaptive_budget.py       # Position budget
│   ├── orderflow_detector.py    # Order flow patterns
│   ├── mtf_structure.py         # Multi-timeframe structure
│   └── session.py               # Session state
├── data/
│   ├── fetcher.py               # OHLCV fetcher (Helsinki → Binance → Bybit)
│   ├── live_feed.py             # Real-time feed
│   └── reports/                 # Generated report storage
├── iros_integration/
│   ├── config/settings.py       # Centralized configuration
│   ├── services/
│   │   ├── bastion_ai.py        # BastionAI class (prompt building + model query)
│   │   ├── helsinki.py           # Helsinki VM client (33 endpoints)
│   │   ├── coinglass.py         # Coinglass Premium client
│   │   ├── whale_alert.py       # Whale Alert client
│   │   ├── exchange_connector.py # 6 exchange connectors
│   │   └── query_processor.py   # NLP query context extraction
│   └── endpoints/               # API endpoint documentation
├── mcf_labs/
│   ├── generator.py             # Base report generator
│   ├── iros_generator.py        # LLM-powered reports
│   ├── institutional_generator.py # Goldman-grade research
│   ├── scheduler.py             # Automated scheduling
│   ├── storage.py               # Hybrid storage
│   ├── supabase_storage.py      # Supabase persistence
│   └── models.py                # Report models
├── web/
│   ├── index.html               # Landing page
│   ├── monitor.html             # PULSE dashboard
│   ├── research.html            # MCF Research
│   ├── login.html               # Auth page
│   ├── account.html             # Account settings
│   ├── dashboard.html           # Visualizations
│   ├── trade-manager.html       # Trade management
│   ├── institutional.html       # Institutional features
│   ├── api-client.js            # Shared API client
│   ├── chart.js                 # Chart rendering
│   ├── volume-profile.js        # Volume profile viz
│   ├── styles.css               # Shared styles
│   └── settings.js              # Settings JS
├── training_data/
│   ├── bastion_risk_v6_combined.jsonl   # 328 examples (LIVE)
│   ├── bastion_risk_v5_combined.jsonl   # 290 examples
│   ├── bastion_risk_v3_combined.jsonl   # 230 examples
│   ├── *_reinforcement.jsonl            # Per-version reinforcement sets
│   ├── finetune_v*.py                   # Fine-tune scripts (v2-v7.1)
│   ├── merge_v*.py                      # LoRA merge scripts (v4-v7.1)
│   └── generate_v*_examples.py          # Training data generators
├── backtest/
│   ├── bastion_backtest.py              # Main backtester
│   └── results/                         # Backtest result JSONs
├── tests/
│   └── test_api.py                      # API tests
├── generated-page.html                  # Pro Terminal (served at /terminal)
├── bastion lite.html                    # Mobile app (served at /lite)
├── bastion login page.html              # Login page (legacy)
├── run.py                               # Local dev entry point
├── requirements.txt                     # Python dependencies
├── runtime.txt                          # Python version (3.11.7)
├── .env                                 # Environment variables (gitignored)
├── .gitignore                           # Git ignore rules
└── SYSTEM_ARCHITECTURE.md               # This file
```

---

*Built by MCF Labs, 2026.*
