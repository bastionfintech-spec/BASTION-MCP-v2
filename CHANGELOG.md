# BASTION Changelog

## 2026-02-18 — v6 Structure-Aware Intelligence + Dashboard Polish

### Model: BASTION Risk v6 (LIVE)
- **Fine-tuned Qwen2.5-32B** via QLoRA (rank=32, alpha=64) on 4x RTX 5090 GPU cluster
- **328 training examples** (290 v5 base + 38 new structure-aware examples across 10 crypto pairs)
- **75.4% backtest win rate** — BTC 71.7%, ETH 72.7%, SOL 81.8%
- **Training**: 4 epochs, 164 steps, ~30 minutes, final loss 0.0916, token accuracy 98.4%
- Structure terminology (VPVR, graded S/R, POC, HVN/LVN) embedded in model reasoning
- TP_PARTIAL accuracy fixed: 33.3% (v5) → ~70% (v6)

### New: MCF Structure Analysis Pipeline
- **`core/auto_support.py`** — Auto-support/resistance detector with priority scoring
  - 16 sensitivity levels (lookback windows 10–160 bars)
  - Scoring: close match (+2.0), wick touch (+1.0), inverse distance weighting
  - Clusters levels within 0.3% and merges scores
- **`core/structure_service.py`** — Orchestration layer for structural analysis
  - Fetches candles across 15m/1h/4h timeframes in parallel
  - Runs VPVR + pivot detection + auto-support per timeframe
  - Caching with TTLs (15m→3min, 1h→5min, 4h→10min) — 830ms first call, <1ms cached
  - Produces token-efficient prompt text (~350 tokens) for model consumption
  - Graceful degradation: evaluation works without structure data if fetch fails

### Upgraded: Core Analyzers
- **`core/vpvr_analyzer.py`** — Volume Profile upgraded
  - 50 → 250 bins for higher resolution
  - Added buy/sell volume split (bullish = close-low/high-low, bearish = high-close/high-low)
  - New `get_buy_sell_context()` returns buy_dominant/sell_dominant/balanced
  - Backward compatible — existing `volume_at_price` field unchanged
- **`core/structure_detector.py`** — Pivot detection upgraded
  - Symmetric 5-bar → asymmetric pivots matching Pine Script indicators
  - Swing highs: 20 left / 15 right lookback (was 5/5)
  - Swing lows: 15 left / 10 right lookback (was 5/5)
  - Keeps only 5 most recent pivots of each type
  - Fewer but more significant pivots → higher-grade trendlines naturally

### Upgraded: Risk Evaluation Pipeline (`api/terminal_api.py`)
- Structure analysis injected as 4th parallel task alongside Coinglass, Helsinki, Whale Alert
- New `STRUCTURAL ANALYSIS` section in model prompt with nearest S/R, VPVR zone, trendlines, pressure points
- MCF exit hierarchy updated: ATR trailing stops replaced with structure-based rules
  - Structural break exits (Grade 3+ support/resistance)
  - VPVR-informed trailing (HVN/LVN awareness)
  - Confluence scoring (auto-support priority + trendline grade)
- 13 live data sources active: Coinglass (markets, whales, funding, OI), Helsinki (CVD, volatility, liquidations, momentum, smart money, orderbook, VWAP), Whale Alert, MCF Structure

### New: Exchange Connector (`iros_integration/services/exchange_connector.py`)
- Unified exchange API abstraction for position fetching
- Supports Binance, Bybit, OKX via ccxt
- Position normalization to standard format

### Upgraded: Data Fetcher (`data/fetcher.py`)
- Multi-exchange candle fetching with fallback chain (Kraken → Coinbase → Bybit → Binance)
- Async parallel timeframe fetching
- Better error handling and timeout management

### Dashboard Polish (`web/dashboard.html`)
- Added `<!DOCTYPE html>`, meta description, theme-color, inline SVG favicon
- **Loading splash screen** — animated shield icon with progress bar, auto-dismisses on init
- **Price flash effect** — green/red background flash on WebSocket price updates
- **Keyboard shortcuts** — 1/2/3 for tabs, `/` for chat, `S` for Shield, `Esc` to close
- **Keyboard hints bar** — desktop-only shortcut reference strip
- **Model offline fallback** — amber "MODEL OFFLINE" badge when v6 unreachable (shows data source count)
- **Shield analysis polish** — color-coded confidence (green ≥80%, amber ≥60%), v6 attribution, animate-pulse for urgent actions
- **Improved empty states** — positions (Connect Exchange CTA), reports (Generate First Report), scanner (bordered icon)
- **Enhanced chat** — data source tags (VPVR, OI, Funding, On-chain, MCF), better quick chips with emoji labels, dynamic symbol
- **Pro CTA upgrade** — amber gradient banner with icon badge and UPGRADE pill
- **Mobile nav polish** — icons per item, active state, Research link, PRO badge
- **Report modal upgrade** — wider (640px), rounded-xl, icon header badge
- **v6 version badge** in header
- **Print styles** for report export
- **Chart attribution** — "Powered by TradingView"

### Generated Page Polish (`generated-page.html`)
- Added `<!DOCTYPE html>`, meta description, theme-color, inline SVG favicon

### Training Pipeline (on GPU cluster)
- **v3 → v5 → v6** progressive fine-tuning chain (v4 abandoned at 47.8%)
- All scripts preserved: `finetune_v3.py` through `finetune_v6.py`, merge scripts, data generators
- Training data: `training_data/bastion_risk_v{3,4,5,6}_{combined,reinforcement}.jsonl`
- Backtest script: `backtest/bastion_backtest.py` with stop-breach scoring fix

### Backtest History
| Version | Win Rate | Notes |
|---------|----------|-------|
| v2 | 54.3% | HOLD accuracy 23.1% (terrible) |
| v3 | 71.4% | First production-viable model |
| v4 | 47.8% | ABANDONED — asymmetric training caused regression |
| v5 | 65.9% | 6 targeted reinforcement categories, stop-breach fix |
| v6 | 75.4% | Structure-aware, best overall — LIVE |

### Infrastructure
- **GPU Cluster**: Vast.ai 4x RTX 5090 (32GB each), tensor parallel across all 4
- **vLLM**: Serves model as "bastion-32b" with `--gpu-memory-utilization 0.85 --max-num-seqs 64`
- **Port forwarding**: socat 6006→8000 (internal), external port 56333
- **Disk cleanup**: v1, v2, v4 models deleted — freed ~186GB

---

## 2026-02-17 — Risk Intelligence v2 + Execution Engine

### Risk Intelligence v2 (`api/terminal_api.py`)
- Standardized 8 action types: HOLD, EXIT_FULL, EXIT_100_PERCENT_IMMEDIATELY, TP_PARTIAL, TP_FULL, REDUCE_SIZE, ADJUST_STOP, TRAIL_STOP
- Intelligent exit sizing based on leverage and distance from stop
- 200 training examples for v2 model

### Execution Engine (`api/execution_engine.py`)
- Autonomous order execution via MCF hierarchy
- Risk-based position management (TP/SL automation)

---

## Earlier — BASTION Lite Dashboard

### v2 Overhaul
- Live TradingView Lightweight Charts integration
- Portfolio risk score aggregation
- Intel tab with funding rates, liquidations, sentiment
- Mobile ticker bar
- Click-to-chart symbol switching

### Shield System
- AI-powered position risk evaluation
- Real-time data from 13 sources
- Takeover effect UI on scan
- Per-position confidence scoring

### Design System
- Bloomberg/terminal dark theme
- Monospace numbers, muted semantic colors
- Responsive grid layout (desktop-first)
- Scanline overlay effects
