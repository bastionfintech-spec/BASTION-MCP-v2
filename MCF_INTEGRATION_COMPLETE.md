# BASTION - MCF Integration Complete

## What Was Added

### 1. **VPVR Analyzer** (`core/vpvr_analyzer.py`)
- **Volume Profile Visible Range** calculation
- **HVN (High Volume Node)** detection = "Mountains" = Target zones
- **LVN (Low Volume Node)** detection = "Valleys" = Fast move zones  
- **POC (Point of Control)** = Highest volume level
- **Value Area** (68% of volume distribution)
- **Volume Score** (0-10) = Directional path analysis

**What it fixes:**
- ✅ Basic stops → **Volume-informed structural targets**
- ✅ Fixed R multiples → **HVN mountains as natural targets**
- ✅ Ignores liquidity → **Detects where price will stall (HVNs) or fly (LVNs)**

### 2. **Structure Detector** (`core/structure_detector.py`)
- **Fractal Swing Detection** (not just N-bar highs/lows)
- **Trendline Grading** (Grade 1-4 based on touches + bipolar status)
- **Horizontal Level Detection** with clustering
- **Pressure Point Identification** (trendline meets horizontal level)
- **Bipolar Tracking** (levels that acted as both S and R = Grade 4)
- **Structure Score** (0-10) = Quality of detected structure

**What it fixes:**
- ✅ Simple swing detection → **Validated trendlines with grades**
- ✅ No touch validation → **Must NOT slice through candle bodies**
- ✅ Ignores bipolar levels → **Auto-grade 4 for S→R or R→S flips**
- ✅ No confluence → **Pressure points where trendline meets horizontal**

### 3. **MTF Structure Analyzer** (`core/mtf_structure.py`)
- **Multi-Timeframe Bias Detection** (Weekly/Daily/4H/1H/15M)
- **Timeframe Role Assignment** (Macro, Structure, Execution)
- **Alignment Scoring** (0-1) = How aligned are timeframes
- **Conflict Detection** (Macro bullish but structure bearish)
- **MTF Structure Score** (0-10) = Overall alignment quality

**What it fixes:**
- ✅ Single timeframe analysis → **Multi-timeframe confluence**
- ✅ Ignores higher TF bias → **Macro timeframes block trades against trend**
- ✅ No alignment check → **Requires 60%+ alignment to trade**

### 4. **Order Flow Detector** (`core/orderflow_detector.py`)
- **Order Book Imbalance** (bid vs ask pressure)
- **Large Trade Detection** (whale activity, block trades)
- **Liquidity Zone Mapping** (bid walls, ask walls, thin zones)
- **CVD (Cumulative Volume Delta)** = Institutional accumulation/distribution
- **Smart Money Proxy** = Direction of large player positioning
- **Order Flow Score** (0-10) = Strength of institutional flow

**What it fixes:**
- ✅ No order flow data → **Helsinki VM integration (Port 5002)**
- ✅ Ignores institutional activity → **Detects smart money accumulation/distribution**
- ✅ No liquidity awareness → **Identifies bid walls (support) and ask walls (resistance)**

### 5. **Enhanced Risk Engine** (`core/enhanced_engine.py`)
- **Orchestrates all MCF components**
- **Calculates weighted MCF Score** (Structure 35%, Volume 25%, MTF 25%, OrderFlow 15%)
- **Assigns MCF Grade** (A+, A, B+, B, C+, C, F)
- **Generates structural stops and targets** from detected levels
- **Adjusts position sizing** based on volatility
- **Estimates win probability** based on MCF grade

**Composite MCF Score Formula:**
```
MCF Score = (Structure × 0.35) + (Volume × 0.25) + (MTF × 0.25) + (OrderFlow × 0.15)
```

**MCF Grade Scale:**
- **A+ (9.0-10.0):** Institutional-grade setup, maximum conviction
- **A (8.0-8.9):** High-quality setup, standard full position
- **B+ (7.0-7.9):** Good setup, reduced position
- **B (6.0-6.9):** Tradeable setup, minimum position
- **C+ (5.0-5.9):** Marginal setup, consider skipping
- **C (4.0-4.9):** Weak setup, likely skip
- **F (<4.0):** Invalid setup, **DO NOT TRADE**

---

## How It Works

### Before (Basic BASTION):
```python
# Simple swing high/low detection
if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
    # This is a swing high
```

**Problems:**
- No validation (could slice through candles)
- No grading (all levels treated equally)
- No multi-timeframe context
- No volume profile
- No order flow

### After (MCF-Enhanced BASTION):
```python
from bastion.core.enhanced_engine import EnhancedRiskEngine

engine = EnhancedRiskEngine()

levels = await engine.calculate_risk_levels(
    symbol='BTCUSDT',
    entry_price=94500,
    direction='long',
    timeframe='4h',
    account_balance=100000,
    ohlcv_data={'4h': df_4h, '1d': df_daily, '15m': df_15m}
)

print(f"MCF Score: {levels.mcf_score:.1f}/10")
print(f"MCF Grade: {levels.mcf_grade}")
print(f"Structure: {levels.structure_score:.1f}/10")
print(f"Volume: {levels.volume_score:.1f}/10")
print(f"Order Flow: {levels.orderflow_score:.1f}/10")
print(f"MTF: {levels.mtf_score:.1f}/10")

# Stops are now based on:
# - Validated trendlines (Grade 2+)
# - Horizontal support levels (multiple touches)
# - Pressure points (trendline meets horizontal)
# - Volume profile gaps (LVN zones)

# Targets are now based on:
# - HVN mountains (high volume nodes)
# - Structural resistance levels
# - Value Area High/Low
# - Multi-timeframe confluence zones
```

---

## Example Output

```json
{
  "entry_price": 94500,
  "direction": "long",
  "mcf_score": 7.8,
  "mcf_grade": "B+",
  "structure_score": 8.2,
  "volume_score": 7.5,
  "orderflow_score": 6.9,
  "mtf_score": 8.1,
  "stops": [
    {
      "price": 93200,
      "type": "structural_support",
      "reason": "Below Grade 4 trendline at 93350",
      "confidence": 0.85,
      "distance_pct": 1.38
    }
  ],
  "targets": [
    {
      "price": 97500,
      "type": "structural_vpvr",
      "reason": "HVN mountain (vol z=2.3)",
      "exit_percentage": 33,
      "distance_pct": 3.17,
      "confidence": 0.78
    },
    {
      "price": 99800,
      "type": "structural_vpvr",
      "reason": "Value Area High",
      "exit_percentage": 33,
      "distance_pct": 5.61,
      "confidence": 0.75
    }
  ],
  "position_size": 0.0543,
  "risk_reward_ratio": 2.3,
  "win_probability": 0.55,
  "expected_value": 0.27
}
```

---

## Helsinki VM Integration

**Order Flow Detector** connects to:
```
http://77.42.29.188:5002
```

**Endpoints Used:**
- `/orderbook/{symbol}` - Real-time bid/ask imbalance
- `/large_trades/{symbol}` - Whale activity detection
- `/cvd/{symbol}` - Cumulative Volume Delta

**Existing Features (Already Live):**
- CVD (Cumulative Volume Delta)
- Orderbook Imbalance
- Large Trades Detection
- Futures Basis
- Funding Rates
- Open Interest
- Options Greeks
- VWAP
- Momentum Scoring
- Mean Reversion Signals
- Gas Fees
- DeFi TVL
- Stablecoin Supply

---

## What's Different from Old RiskShield?

### Old RiskShield (`riskshield/core/engine.py`):
- Basic swing high/low detection
- Simple clustering for levels (0.5% tolerance)
- No trendline validation
- No volume profile
- No order flow
- No multi-timeframe analysis
- Basic "if support nearby, use it" logic

### New BASTION (`bastion/core/`):
- ✅ Fractal swing detection with strength scoring
- ✅ Trendline grading (Grade 1-4)
- ✅ Bipolar level tracking
- ✅ Pressure point detection (trendline meets horizontal)
- ✅ Volume Profile (VPVR) for HVN/LVN detection
- ✅ Order Flow Detection (bid walls, ask walls, CVD, smart money)
- ✅ Multi-Timeframe Analysis (alignment scoring, conflict detection)
- ✅ Weighted composite MCF Score
- ✅ Letter grade system (A+, A, B+, B, C+, C, F)
- ✅ Helsinki VM integration for real-time institutional flow

---

## Next Steps

To use the enhanced engine:

1. **Import the new engine:**
   ```python
   from bastion.core.enhanced_engine import EnhancedRiskEngine
   ```

2. **Fetch multi-timeframe data:**
   ```python
   ohlcv_data = {
       '1d': fetch_ohlcv('BTCUSDT', '1d', 200),
       '4h': fetch_ohlcv('BTCUSDT', '4h', 200),
       '15m': fetch_ohlcv('BTCUSDT', '15m', 200),
   }
   ```

3. **Calculate enhanced risk levels:**
   ```python
   engine = EnhancedRiskEngine()
   levels = await engine.calculate_risk_levels(
       symbol='BTCUSDT',
       entry_price=94500,
       direction='long',
       timeframe='4h',
       account_balance=100000,
       ohlcv_data=ohlcv_data
   )
   ```

4. **Check MCF Grade:**
   ```python
   if levels.mcf_grade in ['A+', 'A', 'B+']:
       # High-quality setup
       execute_trade()
   ```

---

## Files Modified/Added

**Added:**
- `bastion/core/vpvr_analyzer.py` (597 lines)
- `bastion/core/structure_detector.py` (908 lines)
- `bastion/core/mtf_structure.py` (518 lines)
- `bastion/core/orderflow_detector.py` (625 lines)
- `bastion/core/enhanced_engine.py` (550 lines)

**Modified:**
- `bastion/requirements.txt` (added numpy, pandas, aiohttp, scipy)

**Total:** 3,198 lines of MCF integration code

---

## Summary

**BASTION now has institutional-grade detection:**
- 🏔️ **VPVR:** Targets HVN mountains, avoids congestion
- 📐 **Structure:** Grade 1-4 trendlines, pressure points, bipolar levels
- 🌐 **MTF:** Multi-timeframe alignment, conflict detection
- 💰 **Order Flow:** Helsinki VM integration, smart money tracking

**The result:**
- Stops are based on **validated structure** (not guesses)
- Targets are based on **volume profile + structure** (not fixed R multiples)
- Position sizing accounts for **volatility + MCF grade**
- Win probability is **data-driven** (not fixed 45%)

**MCF Score replaces guesswork with quantitative analysis.**

