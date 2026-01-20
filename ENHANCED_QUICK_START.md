# BASTION Enhanced Risk Engine - Quick Start

## Installation

```bash
cd C:\Users\Banke\MCF-Project\bastion
pip install -r requirements.txt
```

## Basic Usage

```python
import asyncio
import pandas as pd
from bastion.core.enhanced_engine import EnhancedRiskEngine

async def main():
    # Initialize engine
    engine = EnhancedRiskEngine()
    
    # Load OHLCV data (from your data source)
    ohlcv_4h = pd.read_csv('btc_4h.csv')
    ohlcv_1d = pd.read_csv('btc_1d.csv')
    
    # Calculate risk levels
    levels = await engine.calculate_risk_levels(
        symbol='BTCUSDT',
        entry_price=94500,
        direction='long',
        timeframe='4h',
        account_balance=100000,
        ohlcv_data={
            '4h': ohlcv_4h,
            '1d': ohlcv_1d,
        }
    )
    
    # Check results
    print(f"MCF Score: {levels.mcf_score:.1f}/10")
    print(f"Grade: {levels.mcf_grade}")
    print(f"\nBreakdown:")
    print(f"  Structure: {levels.structure_score:.1f}/10")
    print(f"  Volume:    {levels.volume_score:.1f}/10")
    print(f"  OrderFlow: {levels.orderflow_score:.1f}/10")
    print(f"  MTF:       {levels.mtf_score:.1f}/10")
    
    print(f"\nStops:")
    for stop in levels.stops:
        print(f"  ${stop['price']:.2f} - {stop['reason']}")
    
    print(f"\nTargets:")
    for target in levels.targets:
        print(f"  ${target['price']:.2f} ({target['exit_percentage']:.0f}%) - {target['reason']}")
    
    print(f"\nPosition:")
    print(f"  Size: {levels.position_size:.4f} BTC")
    print(f"  Value: ${levels.position_size * levels.entry_price:,.2f}")
    print(f"  Risk: ${levels.risk_amount:,.2f}")
    print(f"  R:R: {levels.risk_reward_ratio:.2f}:1")
    print(f"  Win%: {levels.win_probability:.0%}")
    print(f"  EV: {levels.expected_value:.2f}")
    
    # Clean up
    await engine.close()

if __name__ == '__main__':
    asyncio.run(main())
```

## API Server

Run the FastAPI server:

```bash
python run.py
```

Then send requests:

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

## Configuration

Customize the engine:

```python
from bastion.core.enhanced_engine import EnhancedRiskEngine, EnhancedRiskEngineConfig

config = EnhancedRiskEngineConfig(
    # MCF Integration
    enable_structure_detection=True,
    enable_vpvr_analysis=True,
    enable_orderflow_detection=True,
    enable_mtf_analysis=True,
    
    # MCF Weights
    structure_weight=0.35,  # 35% weight
    volume_weight=0.25,     # 25% weight
    orderflow_weight=0.15,  # 15% weight
    mtf_weight=0.25,        # 25% weight
    
    # Stop-loss
    use_structural_stops=True,
    atr_stop_multiplier=2.0,
    max_stop_pct=5.0,
    
    # Take-profit
    use_structural_targets=True,
    min_rr_ratio=2.0,
    enable_partial_exits=True,
    partial_exit_ratios=[0.33, 0.33, 0.34],
    
    # Position sizing
    default_risk_pct=1.0,
    volatility_adjusted_sizing=True,
)

engine = EnhancedRiskEngine(config)
```

## Understanding MCF Scores

### Structure Score (0-10)
- **8-10:** Grade 4 trendlines (bipolar or 4+ touches)
- **6-8:** Grade 3 trendlines (3+ touches, clean rejections)
- **4-6:** Grade 2 trendlines (3 touches)
- **0-4:** Weak or no structure

### Volume Score (0-10)
- **8-10:** LVN (valley) ahead, price will move fast
- **6-8:** Moderate volume profile, some resistance
- **4-6:** Neutral volume distribution
- **0-4:** HVN (mountain) ahead, price will stall

### Order Flow Score (0-10)
- **8-10:** Strong bullish flow (accumulation + bid walls)
- **6-8:** Moderate bullish flow
- **4-6:** Neutral flow
- **2-4:** Moderate bearish flow
- **0-2:** Strong bearish flow (distribution + ask walls)

### MTF Score (0-10)
- **8-10:** All timeframes aligned, no conflicts
- **6-8:** Most timeframes aligned, minor conflicts
- **4-6:** Mixed alignment
- **0-4:** Conflicting timeframes, macro blocks trade

### Composite MCF Score

```
MCF = (Structure × 0.35) + (Volume × 0.25) + (MTF × 0.25) + (OrderFlow × 0.15)
```

**Grade Scale:**
- **A+ (9.0-10.0):** Maximum conviction, full position
- **A (8.0-8.9):** High quality, standard position
- **B+ (7.0-7.9):** Good quality, reduced position
- **B (6.0-6.9):** Tradeable, minimum position
- **C+ (5.0-5.9):** Marginal, consider skipping
- **C (4.0-4.9):** Weak, likely skip
- **F (<4.0):** Invalid, **DO NOT TRADE**

## Helsinki VM Integration

Order Flow Detector automatically connects to:
```
http://77.42.29.188:5002
```

**Available Endpoints:**
- `/orderbook/{symbol}` - Bid/ask imbalance
- `/large_trades/{symbol}` - Whale activity
- `/cvd/{symbol}` - Cumulative Volume Delta

If Helsinki VM is unavailable, the engine falls back to OHLCV-based CVD calculation.

## Web Dashboard

Open `web/index.html` in a browser for a simple UI to test risk calculations.

## Examples

See `docs/EXAMPLES.md` for more detailed examples including:
- Multi-shot entry strategies
- Swing trade guarding lines
- Adaptive risk budgets
- Real-time position updates

## Troubleshooting

**"Order flow analysis failed":**
- Helsinki VM may be down
- Engine falls back to OHLCV-based analysis
- Order Flow score will be neutral (5.0)

**"Insufficient data for timeframe":**
- Need at least 50 bars per timeframe
- Provide more historical data

**"MCF score is low":**
- Structure may be weak (choppy price action)
- Volume profile may be unfavorable (HVN ahead)
- MTF may be misaligned (higher TF conflict)
- Consider skipping the trade

## Support

For questions or issues, refer to:
- `MCF_INTEGRATION_COMPLETE.md` - Full technical details
- `BASTION_BUILD_INSTRUCTIONS.md` - Original build plan
- `docs/API.md` - API reference

