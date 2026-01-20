# BASTION - Strategy-Agnostic Risk Management

## What BASTION Does

BASTION is a **pure risk management engine** that works with ANY trading strategy.

### ✅ BASTION Provides:
- **Optimal Stops:** Structural support/resistance + ATR-based fallback
- **Optimal Targets:** Volume-informed (HVN mountains) + structural levels
- **Position Sizing:** Volatility-adjusted, account-based
- **Market Context:** Structure quality, volume profile, order flow, MTF alignment
- **Trade Management:** Multi-tier stops, partial exits, guarding lines

### ❌ BASTION Does NOT:
- **Judge trade quality** (no "A+, B+, F" grades) → That's IROS
- **Give entry signals** → That's your strategy
- **Make trading decisions** → That's you or IROS

---

## The Separation

| System | Purpose | Outputs |
|--------|---------|---------|
| **BASTION** | Risk management | Stops, targets, position size, market context |
| **IROS** | Trade evaluation | MCF Score, trade grade (A+/B+/F), should you trade this? |

**Your strategy** → Generates trade idea  
**IROS (optional)** → Evaluates if trade is good (MCF scoring)  
**BASTION** → Calculates stops, targets, position size  
**You** → Execute trade  

---

## Usage

```python
from bastion.core.risk_engine import RiskEngine

engine = RiskEngine()

# YOU provide the trade idea
levels = await engine.calculate_risk_levels(
    symbol='BTCUSDT',
    entry_price=94500,      # YOUR entry
    direction='long',       # YOUR direction
    timeframe='4h',
    account_balance=100000,
    ohlcv_data={'4h': df_4h, '1d': df_daily}
)

# BASTION provides the risk management
print(f"Stop: ${levels.stops[0]['price']:,.2f}")
print(f"Target 1: ${levels.targets[0]['price']:,.2f} (exit {levels.targets[0]['exit_percentage']:.0f}%)")
print(f"Position Size: {levels.position_size:.4f} BTC")
print(f"Risk: ${levels.risk_amount:,.2f}")
print(f"R:R: {levels.risk_reward_ratio:.2f}:1")

# Optionally: Check market context
print(f"\nMarket Context:")
print(f"  Structure Quality: {levels.structure_quality:.1f}/10")
print(f"  Volume Profile: {levels.volume_profile_score:.1f}/10")
print(f"  Order Flow: {levels.orderflow_bias}")
print(f"  MTF Alignment: {levels.mtf_alignment:.0%}")
```

---

## Market Context (Not Trade Scoring)

BASTION provides **market context scores** for YOUR strategy to use:

### Structure Quality (0-10)
- **8-10:** Strong structural levels (Grade 4 trendlines, pressure points)
- **4-6:** Moderate structure
- **0-4:** Weak or no structure
- **Use:** YOUR strategy can require structure_quality >= 6.0

### Volume Profile Score (0-10)
- **8-10:** LVN (valley) ahead, price will move fast
- **4-6:** Neutral volume distribution
- **0-4:** HVN (mountain) ahead, price may stall
- **Use:** YOUR strategy can check if path is clear

### Order Flow Bias
- **"bullish":** Institutional buying, bid walls
- **"bearish":** Institutional selling, ask walls
- **"neutral":** Balanced
- **Use:** YOUR strategy can check alignment with your direction

### MTF Alignment (0-1)
- **0.7-1.0:** Aligned across timeframes
- **0.4-0.7:** Mixed alignment
- **0.0-0.4:** Conflicting timeframes
- **Use:** YOUR strategy can require >= 0.6 alignment

---

## Example: Using BASTION with Your Strategy

```python
# YOUR strategy generates trade idea
if your_strategy_signal() == 'LONG':
    
    # BASTION calculates risk management
    levels = await engine.calculate_risk_levels(
        symbol='BTCUSDT',
        entry_price=current_price,
        direction='long',
        timeframe='4h',
        account_balance=100000,
        ohlcv_data=ohlcv_data
    )
    
    # YOUR strategy checks market context (optional)
    if levels.structure_quality < 5.0:
        print("Warning: Weak structure")
        # Your decision: skip or reduce size
    
    if levels.orderflow_bias == 'bearish':
        print("Warning: Order flow is bearish")
        # Your decision: skip or reduce size
    
    if levels.mtf_alignment < 0.6:
        print("Warning: Timeframes misaligned")
        # Your decision: skip or reduce size
    
    # Execute trade with BASTION's risk management
    place_order(
        entry=levels.entry_price,
        stop=levels.stops[0]['price'],
        targets=[t['price'] for t in levels.targets],
        size=levels.position_size
    )
```

---

## Example: Using BASTION with IROS

```python
# YOUR strategy generates trade idea
if your_strategy_signal() == 'LONG':
    
    # IROS evaluates trade quality (MCF scoring)
    iros_analysis = await iros.analyze(
        symbol='BTCUSDT',
        direction='long',
        entry_price=current_price
    )
    
    # IROS tells you if it's a good trade
    if iros_analysis.mcf_grade in ['A+', 'A', 'B+']:
        print(f"IROS Grade: {iros_analysis.mcf_grade}")
        print(f"MCF Score: {iros_analysis.mcf_score:.1f}/10")
        
        # BASTION calculates risk management
        levels = await bastion_engine.calculate_risk_levels(
            symbol='BTCUSDT',
            entry_price=current_price,
            direction='long',
            timeframe='4h',
            account_balance=100000,
            ohlcv_data=ohlcv_data
        )
        
        # Execute trade with BASTION's risk management
        place_order(
            entry=levels.entry_price,
            stop=levels.stops[0]['price'],
            targets=[t['price'] for t in levels.targets],
            size=levels.position_size
        )
    else:
        print(f"IROS says skip (Grade: {iros_analysis.mcf_grade})")
```

---

## API Example

```bash
curl -X POST http://localhost:8001/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "entry_price": 94500,
    "direction": "long",
    "timeframe": "4h",
    "account_balance": 100000,
    "risk_per_trade_pct": 1.0
  }'
```

**Response:**
```json
{
  "entry_price": 94500,
  "direction": "long",
  "stops": [
    {
      "price": 93200,
      "type": "structural",
      "reason": "Below structural support at 93350",
      "confidence": 0.82
    }
  ],
  "targets": [
    {
      "price": 97500,
      "type": "structural",
      "reason": "HVN mountain (vol z=2.3)",
      "exit_percentage": 33
    }
  ],
  "position_size": 0.0543,
  "risk_reward_ratio": 2.3,
  "market_context": {
    "structure_quality": 8.2,
    "volume_profile_score": 7.5,
    "orderflow_bias": "bullish",
    "mtf_alignment": 0.78
  }
}
```

---

## Configuration

Customize BASTION for your needs:

```python
from bastion.core.risk_engine import RiskEngine, RiskEngineConfig

config = RiskEngineConfig(
    # Detection systems (all optional)
    enable_structure_detection=True,
    enable_vpvr_analysis=True,
    enable_orderflow_detection=False,  # Disable if no Helsinki VM
    enable_mtf_analysis=True,
    
    # Stop-loss
    use_structural_stops=True,
    atr_stop_multiplier=2.0,
    max_stop_pct=5.0,
    enable_multi_tier_stops=True,
    
    # Take-profit
    use_structural_targets=True,
    min_rr_ratio=2.0,
    enable_partial_exits=True,
    partial_exit_ratios=[0.33, 0.33, 0.34],
    
    # Position sizing
    default_risk_pct=1.0,
    volatility_adjusted_sizing=True,
)

engine = RiskEngine(config)
```

---

## Why This Separation Matters

### ❌ Bad: Mixing concerns
```python
# BAD: Risk engine judges trade quality
levels = engine.calculate(entry, direction)
if levels.mcf_grade == 'F':
    # Engine tells you not to trade
    return
```
**Problem:** Now EVERY strategy MUST use MCF scoring

### ✅ Good: Separation of concerns
```python
# GOOD: Risk engine just provides infrastructure
levels = bastion.calculate(entry, direction)
# Returns: stops, targets, position size, market context

# YOUR strategy decides
if YOUR_STRATEGY_LOGIC:
    execute(levels.stops, levels.targets, levels.size)
```
**Benefit:** ANY strategy can use BASTION

---

## Summary

**BASTION = Infrastructure (stops, targets, sizing)**  
**IROS = Intelligence (trade evaluation, MCF scoring)**  
**Your Strategy = Decision Maker**

BASTION doesn't care if you're:
- A trend follower
- A mean reversion trader
- A breakout trader
- An algo
- A human
- Using IROS
- Using your own scoring

**BASTION just manages risk. You make decisions.**

