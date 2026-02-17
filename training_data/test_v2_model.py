#!/usr/bin/env python3
"""Test BASTION v2 model with standardized risk evaluation prompt."""
import json
import requests

URL = "http://localhost:8000/v1/chat/completions"

SYSTEM = (
    "You are BASTION Risk Intelligence \u2014 an autonomous trade management AI. "
    "You monitor live cryptocurrency positions and make execution decisions. "
    "You output structured JSON with action, reasoning, and execution parameters. "
    "PRIORITY ORDER: 1) Hard Stop breach \u2192 EXIT_100_PERCENT_IMMEDIATELY "
    "2) Safety Net break \u2192 EXIT_FULL 3) Guarding Line break \u2192 REDUCE_SIZE or EXIT_FULL "
    "4) Take Profit targets \u2192 TP_PARTIAL or TP_FULL 5) Trailing Stop updates \u2192 TRAIL_STOP "
    "6) Time-based exits \u2192 REDUCE_SIZE. Core philosophy: Exit on STRUCTURE BREAKS, "
    "not arbitrary targets. Let winners run when structure holds. Scale out intelligently "
    "\u2014 decide HOW MUCH to exit based on structure strength, R-multiple, and market context."
)

# Test 1: TP_PARTIAL scenario
test1 = """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $94,000
- Current Price: $97,200
- P&L: +2.1R
- Stop Loss: $92,800
- Guarding Line: $94,500
- Trailing Stop: $93,800
- TP1: $97,500 | TP2: $100,000 | TP3: $104,000
- Position Size: 0.4 BTC
- Time in Trade: 18h

MARKET DATA:
- ATR(14): $1,500
- CVD: Bullish, net buying pressure increasing
- Funding Rate: +0.012%
- Open Interest: $18.2B (+2.1% 24h)
- Volume: 1.8x average
- 15m Trend: Higher lows, bullish momentum
- 1H Structure: Ascending channel holding, approaching TP1
- Orderbook: Balanced with slight bid dominance below $97K
- Liquidation Data: $300M short cluster at $98,000

DECISION REQUIRED: Position approaching TP1 at $97,500. Structure intact. $300M short squeeze potential above. How to manage?"""

# Test 2: EXIT scenario
test2 = """POSITION STATE:
- Asset: ETH/USDT
- Direction: LONG
- Entry: $3,200
- Current Price: $3,050
- P&L: -1.2R
- Stop Loss: $3,000
- Guarding Line: $3,150
- Position Size: 5 ETH
- Time in Trade: 8h

MARKET DATA:
- ATR(14): $85
- CVD: Bearish, aggressive selling
- Funding Rate: +0.035% (longs overleveraged)
- Open Interest: Rising while price drops (new shorts entering)
- Volume: 3.2x average on sell side
- 15m Trend: Lower highs, lower lows, bearish
- 1H Structure: Guarding line at $3,150 broken on close with volume
- Orderbook: Bids thinning rapidly below $3,050
- Liquidation Data: $200M long cluster at $3,000

DECISION REQUIRED: Guarding line broken. Price dropping toward hard stop with $200M liquidation cluster at $3,000. Funding extreme. How to manage?"""

# Test 3: HOLD scenario
test3 = """POSITION STATE:
- Asset: SOL/USDT
- Direction: SHORT
- Entry: $185.00
- Current Price: $180.50
- P&L: +0.8R
- Stop Loss: $190.00
- Trailing Stop: $188.00
- TP1: $175.00 | TP2: $168.00
- Position Size: 40 SOL
- Time in Trade: 6h

MARKET DATA:
- ATR(14): $5.50
- CVD: Neutral, balanced flow
- Funding Rate: +0.015% (slightly long biased)
- Open Interest: Stable
- Volume: 1.2x average
- 15m Trend: Steady grind lower, clean lower highs
- 1H Structure: Downtrend intact, no bounce signals
- Orderbook: Slight ask dominance above $181
- Smart Money: Net short positioning

DECISION REQUIRED: Short in profit, structure intact, no signals to act. Should we hold or adjust?"""

tests = [
    ("Test 1 - TP approaching", test1),
    ("Test 2 - Structure break", test2),
    ("Test 3 - Hold scenario", test3),
]

for name, user_msg in tests:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    try:
        resp = requests.post(URL, json={
            "model": "bastion-32b",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": 600,
            "temperature": 0.3
        }, timeout=120)

        result = resp.json()
        content = result["choices"][0]["message"]["content"]

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print(f"  Action: {parsed.get('action', 'UNKNOWN')}")
            print(f"  Urgency: {parsed.get('urgency', 'UNKNOWN')}")
            print(f"  Confidence: {parsed.get('confidence', 'UNKNOWN')}")
            print(f"  Reason: {parsed.get('reason', 'N/A')}")

            exec_data = parsed.get("execution", {})
            print(f"  Execution:")
            print(f"    exit_pct: {exec_data.get('exit_pct', 'MISSING')}")
            print(f"    stop_price: {exec_data.get('stop_price', 'MISSING')}")
            print(f"    order_type: {exec_data.get('order_type', 'MISSING')}")

            reasoning = parsed.get("reasoning", {})
            if isinstance(reasoning, dict):
                el = reasoning.get("exit_logic", "")
                if el:
                    print(f"  Exit Logic: {el[:150]}...")

            print(f"\n  [VALID JSON - CORRECT FORMAT]")
        except json.JSONDecodeError:
            print(f"  RAW OUTPUT (not valid JSON):")
            print(f"  {content[:500]}")
            print(f"\n  [WARNING: OUTPUT IS NOT VALID JSON]")

    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print("  ALL TESTS COMPLETE")
print(f"{'='*60}")
