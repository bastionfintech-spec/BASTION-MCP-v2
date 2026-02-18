"""
BASTION v3 Training Data Generator
===================================
Generates new training examples targeting weaknesses found in backtest:
1. HOLD with losing positions (drawdowns that recover)
2. HOLD with tight stops that DON'T get hit
3. EXIT when near stop with high volatility (DON'T HOLD these)
4. Confidence calibration (medium confidence 0.55-0.70)
5. Leverage-aware decisions (same setup, different leverage = different action)
6. No-stop positions where HOLD is actually correct (low leverage spot)
"""

import json
import os

SYSTEM_PROMPT = (
    "You are BASTION Risk Intelligence \u2014 an autonomous trade management AI. "
    "You monitor live cryptocurrency positions and make execution decisions. "
    "You output structured JSON with action, reasoning, and execution parameters. "
    "PRIORITY ORDER: 1) Hard Stop breach \u2192 EXIT_100_PERCENT_IMMEDIATELY "
    "2) Safety Net break \u2192 EXIT_FULL 3) Guarding Line break \u2192 REDUCE_SIZE or EXIT_FULL "
    "4) Take Profit targets \u2192 TP_PARTIAL or TP_FULL 5) Trailing Stop updates \u2192 TRAIL_STOP "
    "6) Time-based exits \u2192 REDUCE_SIZE. Core philosophy: Exit on STRUCTURE BREAKS, not arbitrary targets. "
    "Let winners run when structure holds. Scale out intelligently \u2014 decide HOW MUCH to exit "
    "based on structure strength, R-multiple, and market context."
)

def make_example(user_content, assistant_json):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}
        ]
    }

examples = []

# ============================================================
# CATEGORY 1: HOLD with LOSING positions (drawdown recoveries)
# These teach the model that being underwater is NOT automatic EXIT
# ============================================================

# 1. BTC LONG -0.5R but structure intact, wide stop
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $84,500
- Current Price: $83,750
- P&L: -0.5R
- Stop Loss: $82,000
- Guarding Line: $83,200
- Position Size: 0.3 BTC
- Leverage: 3x
- Time in Trade: 8h

MARKET DATA:
- ATR(14): $1,580
- CVD: Slightly bearish on 15m, neutral on 1H
- Funding Rate: -0.008%
- Open Interest: $17.5B (down 2.1% in 4h)
- Volume: 0.7x average
- 15m Trend: Lower highs but holding above $83,500 support
- 1H Structure: Higher low at $83,100 from 5h ago still intact
- Orderbook: Large bid wall at $83,200-$83,400, thin asks above $84,000

DECISION REQUIRED: Position is -0.5R with bearish CVD on short timeframe. Should we exit early or hold?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.68,
        "reason": "Drawdown is within 0.5 ATR and 1H structure (higher low at $83,100) remains intact. Stop at $82,000 gives 1.1 ATR of room.",
        "reasoning": {
            "structure_analysis": "The critical level is the 1H higher low at $83,100. Current price at $83,750 is $650 above this level. The 15m shows lower highs but has NOT broken below the $83,500 support cluster. This is a normal pullback within a range, not a structural breakdown.",
            "data_assessment": "CVD is slightly bearish on 15m but neutral on 1H -- short timeframe CVD divergences are noise during consolidation. Negative funding at -0.008% means shorts are paying longs, which is bullish fuel. OI declining 2.1% shows overleveraged positions being flushed -- this is healthy deleveraging, not panic.",
            "risk_factors": "The -0.5R drawdown equals roughly $750 on a $1,580 ATR -- less than half an ATR of adverse movement. The bid wall at $83,200-$83,400 provides structural support. Stop at $82,000 gives another $1,750 of room (1.1 ATR). Risk is well-defined and manageable at 3x leverage.",
            "exit_logic": "No exit triggers active. Guarding line at $83,200 is holding. Stop at $82,000 is untouched. Price is consolidating within a normal ATR range. Exiting here would be cutting a position based on noise, not structure."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 2. ETH SHORT -0.3R but breakdown thesis intact
examples.append(make_example(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,420
- Current Price: $3,455
- P&L: -0.3R
- Stop Loss: $3,550
- Guarding Line: $3,480
- Position Size: 5.0 ETH
- Leverage: 5x
- Time in Trade: 4h

MARKET DATA:
- ATR(14): $95
- CVD: Neutral
- Funding Rate: +0.015%
- Open Interest: $8.2B (up 1.8% in 2h)
- Volume: 0.9x average
- 15m Trend: Bouncing off $3,440 support, testing $3,460 resistance
- 1H Structure: Lower high at $3,490 confirmed, descending channel intact
- Orderbook: Ask wall at $3,475-$3,490, bids thin below $3,430

DECISION REQUIRED: Short is slightly underwater with a small bounce. Is the breakdown thesis invalidated?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.65,
        "reason": "Bounce is within normal ATR noise. Descending channel on 1H intact with lower high at $3,490 confirmed. Crowded longs at +0.015% funding support short thesis.",
        "reasoning": {
            "structure_analysis": "The 1H descending channel is intact with a confirmed lower high at $3,490. Current bounce to $3,455 is a $35 move -- only 0.37 ATR. The ask wall at $3,475-$3,490 should cap upside. The guarding line at $3,480 hasn't been violated on a closing basis.",
            "data_assessment": "Funding at +0.015% means longs are crowded and paying shorts -- this is fuel for a short squeeze in reverse (long liquidations). OI up 1.8% with price bouncing slightly means new longs are entering -- these will become trapped if the channel holds. Neutral CVD during a bounce is actually bearish -- no real buying conviction.",
            "risk_factors": "At 5x leverage, the -0.3R drawdown is a $35 move against a $95 ATR. Stop at $3,550 gives $95 of room (exactly 1 ATR). The position is sized appropriately for the volatility.",
            "exit_logic": "No structural breaks detected. Guarding line at $3,480 is holding. The bounce is a lower timeframe reaction within an intact bearish structure. Exiting on this bounce would be reacting to noise."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 3. SOL LONG -0.7R at key support, high leverage but wide stop
examples.append(make_example(
    """POSITION STATE:
- Asset: SOL/USDT
- Direction: LONG
- Entry: $178.50
- Current Price: $174.80
- P&L: -0.7R
- Stop Loss: $168.00
- Guarding Line: $173.50
- TP1: $188.00 (30%)
- Position Size: 50 SOL
- Leverage: 10x
- Time in Trade: 12h

MARKET DATA:
- ATR(14): $6.80
- CVD: Bearish on 15m, turning neutral on 1H
- Funding Rate: -0.012%
- Open Interest: $2.1B (down 4.5% in 6h)
- Volume: 1.3x average (selling volume)
- 15m Trend: Bouncing off $174.20 with hammer candle
- 1H Structure: Previous swing low at $173.80 holding, double bottom forming
- Orderbook: Massive bid cluster at $173.50-$174.00, asks thinning above $176

DECISION REQUIRED: Deep drawdown at 10x leverage. Selling volume is elevated. Should we cut losses?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.58,
        "reason": "Double bottom forming at $173.80-$174.20 with massive bid support. OI deleveraging suggests weak hands already flushed. But confidence is lower due to 10x leverage risk.",
        "reasoning": {
            "structure_analysis": "The 1H swing low at $173.80 is being retested and holding. A hammer candle on the 15m at $174.20 is a bullish reversal signal at support. The double bottom pattern at this level suggests buyers are defending it. Guarding line at $173.50 aligns with the bid cluster, creating a strong support zone.",
            "data_assessment": "OI down 4.5% in 6h indicates overleveraged positions have been liquidated -- this is the 'weak hands flush' pattern. Negative funding at -0.012% means shorts are paying longs, providing income and suggesting short crowding. The elevated selling volume with price holding support means sellers are being absorbed by the bid wall.",
            "risk_factors": "At 10x leverage, the -0.7R drawdown is significant but the stop at $168.00 gives $6.80 of remaining room (exactly 1 ATR). The high leverage makes this uncomfortable but the wide stop was sized for this exact scenario. The double bottom pattern has a measurable edge.",
            "exit_logic": "The guarding line at $173.50 is the decision point. If it breaks on a 1H close, exit immediately. But currently, the structure is holding with confirming signals (hammer candle, OI flush, negative funding). This is a valid HOLD despite the drawdown."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 4. BTC LONG -1.0R but stop is far and structure holds
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $91,200
- Current Price: $89,800
- P&L: -1.0R
- Stop Loss: $87,500
- Guarding Line: $89,200
- Position Size: 0.15 BTC
- Leverage: 2x
- Time in Trade: 18h

MARKET DATA:
- ATR(14): $1,650
- CVD: Bearish
- Funding Rate: +0.005%
- Open Interest: $19.1B (flat)
- Volume: 0.6x average
- 15m Trend: Ranging between $89,600-$90,100
- 1H Structure: Consolidating above $89,500 support, no lower lows
- Orderbook: Balanced, small bid bias at $89,500

DECISION REQUIRED: Position is at -1.0R with bearish CVD. Are we holding a loser too long?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.62,
        "reason": "At 2x leverage with stop at $87,500, the risk is well-contained. 1H consolidation above $89,500 with low volume suggests accumulation, not distribution. -1.0R is uncomfortable but within the trade plan.",
        "reasoning": {
            "structure_analysis": "No lower lows on the 1H timeframe. Price is consolidating above $89,500 which has acted as support for 6+ hours. The guarding line at $89,200 has not been tested. The range-bound 15m action with decreasing volume is typical of accumulation phases.",
            "data_assessment": "Bearish CVD is concerning but volume at 0.6x average suggests the selling is not aggressive -- it's more like a slow drift than a dump. Flat OI means no new positions being built in either direction. Neutral funding at +0.005% is unremarkable.",
            "risk_factors": "At 2x leverage, a -1.0R drawdown is a 0.77% portfolio hit -- very manageable. Stop at $87,500 gives $2,300 of remaining room (1.4 ATR). The position was sized for this exact scenario. Cutting at -1.0R on a 2x trade with intact structure would be emotional, not strategic.",
            "exit_logic": "Watch the guarding line at $89,200. If 1H closes below, reassess. But currently, no exit triggers are active. The trade thesis (long from $91,200 targeting higher) hasn't been invalidated -- it's just taking longer than expected."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 5. BTC SHORT -0.4R, tight stop but high conviction setup
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $88,300
- Current Price: $88,650
- P&L: -0.4R
- Stop Loss: $89,200
- Guarding Line: $88,800
- TP1: $85,500 (30%)
- Position Size: 0.25 BTC
- Leverage: 5x
- Time in Trade: 2h

MARKET DATA:
- ATR(14): $1,420
- CVD: Bullish on 15m, bearish on 1H
- Funding Rate: +0.022%
- Open Interest: $20.3B (up 3.2% in 1h)
- Volume: 1.5x average
- 15m Trend: Grinding up toward $88,800 resistance
- 1H Structure: Clear double top at $89,100 with rejection wicks
- Orderbook: Massive ask wall at $88,800-$89,000, bids thin

DECISION REQUIRED: Short is slightly red. 15m is pushing up. But the 1H double top looks strong. Hold or cut?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.64,
        "reason": "1H double top at $89,100 is a high-probability reversal pattern. 15m bullish push into $88,800 resistance with massive ask wall is likely a bull trap. Crowded longs at +0.022% funding adds conviction.",
        "reasoning": {
            "structure_analysis": "The 1H double top at $89,100 with rejection wicks is one of the highest-probability reversal patterns. The 15m bullish grind toward $88,800 is approaching the ask wall -- this is where the move likely stalls. The guarding line at $88,800 aligns perfectly with the resistance level.",
            "data_assessment": "Funding at +0.022% is elevated -- longs are crowded and paying significant rates. OI surging 3.2% in 1h means new positions entering on the long side -- these are potential liquidation fuel. The 1H CVD being bearish while 15m is bullish is a classic divergence signal: retail pushing up into institutional selling.",
            "risk_factors": "Stop at $89,200 is tight -- only $550 away (0.39 ATR). At 5x leverage, the risk per R is defined. If price breaks $89,200, the double top is invalidated and we should exit. But the probability of that is lower given the massive ask wall and crowded funding.",
            "exit_logic": "Guarding line at $88,800 is the key level. If price closes above $88,800 on the 1H, the trade thesis weakens. But currently, we're grinding into resistance with declining momentum -- this is the hold zone."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 6. AVAX LONG -0.6R, low leverage, healthy pullback
examples.append(make_example(
    """POSITION STATE:
- Asset: AVAX/USDT
- Direction: LONG
- Entry: $42.80
- Current Price: $41.90
- P&L: -0.6R
- Stop Loss: $40.30
- Guarding Line: $41.50
- TP1: $46.00 (25%)
- Position Size: 100 AVAX
- Leverage: 2x
- Time in Trade: 16h

MARKET DATA:
- ATR(14): $2.10
- CVD: Neutral turning slightly bullish
- Funding Rate: -0.003%
- Open Interest: $420M (down 1.5%)
- Volume: 0.8x average
- 15m Trend: Basing above $41.80, higher lows forming
- 1H Structure: Pullback to 50% fib retracement of prior swing, holding
- Orderbook: Bid accumulation at $41.50-$41.80

DECISION REQUIRED: Mid-cap alt pulling back. Low leverage but starting to feel the drawdown. Hold or reduce?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.71,
        "reason": "Textbook 50% fib retracement pullback in an uptrend. Bid accumulation at $41.50-$41.80 confirms institutional buying. Low leverage (2x) means drawdown is manageable.",
        "reasoning": {
            "structure_analysis": "A pullback to the 50% fibonacci retracement of the prior swing is one of the most common and reliable continuation patterns. The 15m is forming higher lows above $41.80, suggesting the pullback is exhausting. The guarding line at $41.50 aligns with the fib level and bid accumulation.",
            "data_assessment": "CVD turning slightly bullish during a pullback is a strong continuation signal -- buying is returning before the pullback completes. Negative funding at -0.003% means the trade isn't crowded. OI down 1.5% is healthy deleveraging.",
            "risk_factors": "At 2x leverage, -0.6R is barely a 0.7% portfolio impact. Stop at $40.30 gives $1.60 of room (0.76 ATR). The risk-reward to TP1 at $46.00 is excellent: risking $1.60 to make $4.10 (2.6:1).",
            "exit_logic": "No exit triggers. Structure intact. Fib retracement holding. This is a patience trade -- the pullback is where most traders get shaken out before the next leg up."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# ============================================================
# CATEGORY 2: EXIT when near stop with high volatility
# These teach the model WHEN to exit (don't HOLD these!)
# ============================================================

# 7. BTC LONG near stop, high vol, leverage = EXIT
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $86,500
- Current Price: $85,200
- P&L: -0.8R
- Stop Loss: $84,800
- Guarding Line: $85,000
- Position Size: 0.5 BTC
- Leverage: 15x
- Time in Trade: 6h

MARKET DATA:
- ATR(14): $2,100
- CVD: Strongly bearish
- Funding Rate: +0.025%
- Open Interest: $21.5B (up 5.8% in 2h)
- Volume: 2.3x average (panic selling)
- 15m Trend: Aggressive selling, lower lows accelerating
- 1H Structure: Support at $85,500 broken, next support at $84,000
- Orderbook: Bids being pulled below $85,000, ask walls building

DECISION REQUIRED: Position is 0.24% from stop at 15x leverage. Volume is spiking. Hold for bounce or exit now?""",
    {
        "action": "EXIT_FULL",
        "urgency": "HIGH",
        "confidence": 0.88,
        "reason": "Price is $400 from stop loss with 2.3x average volume selling and broken $85,500 support. At 15x leverage, hitting the stop would be catastrophic. Exit now before slippage worsens.",
        "reasoning": {
            "structure_analysis": "The 1H support at $85,500 has broken with conviction -- this is a structural breakdown. Lower lows are accelerating on the 15m with no sign of a base forming. The next real support is at $84,000, which is BELOW the stop. The guarding line at $85,000 is about to break.",
            "data_assessment": "CVD strongly bearish confirms genuine selling, not a temporary dip. Volume at 2.3x average with OI surging 5.8% means new shorts are piling in -- this is trend continuation, not exhaustion. Crowded long funding at +0.025% means more long liquidations are coming.",
            "risk_factors": "At 15x leverage, the remaining $400 to stop represents 6% of the position. But with 2.3x volume and accelerating selling, stop slippage could be severe. A $400 slip past the stop at 15x leverage would add another 6% loss. Total downside if stopped with slippage: up to -1.5R.",
            "exit_logic": "Broken structure + high volume + crowded funding + proximity to stop + high leverage = immediate exit. The probability of a bounce here is low, and the cost of being wrong at 15x is too high. Take the -0.8R loss now rather than risk -1.2R to -1.5R on a stop run."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# 8. ETH SHORT near stop, volatility expanding = EXIT
examples.append(make_example(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,280
- Current Price: $3,340
- P&L: -0.6R
- Stop Loss: $3,380
- Guarding Line: $3,350
- Position Size: 8.0 ETH
- Leverage: 10x
- Time in Trade: 3h

MARKET DATA:
- ATR(14): $88
- CVD: Bullish, accelerating
- Funding Rate: -0.018%
- Open Interest: $9.1B (up 4.2% in 1h)
- Volume: 1.8x average
- 15m Trend: Breakout above $3,320 resistance with follow-through
- 1H Structure: Higher highs forming, breaking descending channel
- Orderbook: Bids stacking aggressively, asks retreating

DECISION REQUIRED: Short is bleeding. Only $40 from stop. CVD is accelerating bullish. Wait for stop or exit now?""",
    {
        "action": "EXIT_FULL",
        "urgency": "HIGH",
        "confidence": 0.85,
        "reason": "Descending channel broken with bullish CVD acceleration and aggressive bid stacking. Only $40 from stop at 10x leverage. Short squeeze conditions present with -0.018% funding.",
        "reasoning": {
            "structure_analysis": "The descending channel has been broken to the upside with the 15m showing genuine follow-through above $3,320. Higher highs forming on the 1H invalidate the short thesis. The guarding line at $3,350 has been broken.",
            "data_assessment": "CVD accelerating bullish means real buying pressure, not just short covering. Negative funding at -0.018% means shorts are crowded and paying -- this is classic short squeeze setup. OI surging 4.2% in 1h with price rising means new longs entering aggressively.",
            "risk_factors": "Only $40 from stop at 10x leverage. A $40 move at 10x is a 1.2% portfolio hit on top of the existing -0.6R loss. With momentum accelerating, the stop is very likely to be hit, and slippage in a squeeze can add another $20-30.",
            "exit_logic": "Structure broken (descending channel invalidated), guarding line broken ($3,350), proximity to stop with momentum against the position, short squeeze conditions. Every signal says exit. Take -0.6R now instead of waiting for -1.0R at stop with potential slippage."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# 9. BTC LONG near stop, choppy market, high leverage = EXIT (not hold!)
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $72,800
- Current Price: $71,950
- P&L: -0.5R
- Stop Loss: $71,600
- Guarding Line: $71,800
- Position Size: 0.4 BTC
- Leverage: 20x
- Time in Trade: 5h

MARKET DATA:
- ATR(14): $1,850
- CVD: Choppy, no clear direction
- Funding Rate: +0.008%
- Open Interest: $18.8B (up 2.1%)
- Volume: 1.1x average
- 15m Trend: Whipsawing between $71,800-$72,300
- 1H Structure: Range-bound, no clear trend
- Orderbook: Thin on both sides, low depth

DECISION REQUIRED: Close to stop in a choppy, directionless market at 20x leverage. Is this a hold or a "get out before the chop kills you"?""",
    {
        "action": "EXIT_FULL",
        "urgency": "MEDIUM",
        "confidence": 0.75,
        "reason": "At 20x leverage, only $350 from stop in a choppy market with thin orderbook depth. The risk of stop-hunt whipsaw is too high. Better to re-enter with confirmation than hold through chop.",
        "reasoning": {
            "structure_analysis": "No clear trend or structure on any timeframe. The 15m is whipsawing with no higher lows or lower lows -- pure noise. At 20x leverage, this chop can trigger the stop on a random wick. The guarding line at $71,800 is being tested repeatedly.",
            "data_assessment": "CVD has no direction -- this confirms the choppy, indecisive nature of the market. Thin orderbook depth means a single large order can push price $200-300 in either direction. This is the worst environment for high-leverage positions.",
            "risk_factors": "At 20x leverage, $350 to stop is a 0.48% portfolio move per $ of adverse price action. A stop-hunt wick through $71,600 could slip to $71,400 and cost an additional 0.3R. In thin liquidity, this is highly probable. The risk-reward of holding is terrible: risking another -0.5R of slippage to gain in a directionless market.",
            "exit_logic": "High leverage + proximity to stop + thin liquidity + no directional edge = exit. This isn't about the trade being wrong -- it's about the risk-reward of holding being terrible at 20x. Flat is a position. Can re-enter at lower leverage or with better confirmation."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# 10. SOL SHORT near stop with momentum against = EXIT
examples.append(make_example(
    """POSITION STATE:
- Asset: SOL/USDT
- Direction: SHORT
- Entry: $165.00
- Current Price: $168.20
- P&L: -0.7R
- Stop Loss: $169.50
- Guarding Line: $167.50
- Position Size: 30 SOL
- Leverage: 10x
- Time in Trade: 7h

MARKET DATA:
- ATR(14): $5.40
- CVD: Bullish, sustained buying
- Funding Rate: -0.015%
- Open Interest: $1.8B (up 6.1% in 4h)
- Volume: 1.6x average
- 15m Trend: Consistent higher lows, grinding up
- 1H Structure: Broke above $167.00 resistance, now support
- Orderbook: Bids building at $167.50-$168.00, asks retreating above $169

DECISION REQUIRED: Short is -0.7R and price is grinding against us. Guarding line already broken. $1.30 from stop. Hold or accept the loss?""",
    {
        "action": "EXIT_FULL",
        "urgency": "HIGH",
        "confidence": 0.82,
        "reason": "Guarding line at $167.50 already broken. Price grinding higher with bullish CVD and negative funding (short squeeze fuel). Only $1.30 from stop at 10x. Accept the -0.7R loss.",
        "reasoning": {
            "structure_analysis": "The guarding line at $167.50 has been broken -- this alone is a sell signal. Price is now grinding up with consistent higher lows on the 15m, showing sustained buying pressure. The $167.00 resistance has flipped to support.",
            "data_assessment": "Bullish CVD with sustained buying confirms trend. Negative funding at -0.015% means shorts are crowded -- this is short squeeze fuel. OI up 6.1% in 4h with price rising means new longs entering aggressively. Everything points to further upside.",
            "risk_factors": "At 10x leverage, $1.30 from stop. The remaining R to stop is small. Holding here is risking slippage on the stop for almost no potential gain -- the short thesis is already invalidated.",
            "exit_logic": "Guarding line broken + bullish momentum + short squeeze conditions + proximity to stop = exit immediately. The structural break at $167.50 invalidated the short thesis. Take -0.7R and look for the next setup."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# ============================================================
# CATEGORY 3: No-stop positions where behavior depends on leverage
# ============================================================

# 11. BTC LONG no stop, 1x spot, small drawdown = HOLD (not panic exit)
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $87,200
- Current Price: $86,500
- P&L: -0.8% (no R-multiple, no stop)
- Stop Loss: NONE
- Position Size: 0.1 BTC
- Leverage: 1x (SPOT)
- Time in Trade: 24h

MARKET DATA:
- ATR(14): $1,500
- CVD: Slightly bearish
- Funding Rate: N/A (spot)
- Open Interest: N/A (spot)
- Volume: 0.7x average
- 15m Trend: Ranging, no strong direction
- 1H Structure: Holding above $86,000 support zone
- Orderbook: Balanced

DECISION REQUIRED: Spot position with no stop loss and small drawdown. Should we panic exit?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.70,
        "reason": "1x spot position with -0.8% drawdown requires no emergency action. No liquidation risk exists. However, a stop loss should be set at $84,500 (below $86,000 support) to define risk.",
        "reasoning": {
            "structure_analysis": "Price is holding above the $86,000 support zone on the 1H. A -0.8% drawdown on spot is not even one full ATR of movement ($700 vs $1,500 ATR). There's no structural breakdown visible.",
            "data_assessment": "Slightly bearish CVD with low volume is not concerning for a spot position. This is normal market noise. Without leverage, there's no liquidation risk and no funding cost.",
            "risk_factors": "The main risk is the absence of a defined stop loss. While a 1x spot position cannot be liquidated, unbounded risk is poor practice. Recommend setting a stop at $84,500 (3.1% below entry) to create a defined 1R.",
            "exit_logic": "No exit warranted. This is a spot position experiencing normal volatility. The drawdown is less than half an ATR. HOWEVER, strongly recommend setting a stop loss to convert this from undefined risk to defined risk."
        },
        "execution": {"exit_pct": 0, "stop_price": 84500, "order_type": "STOP_LIMIT"}
    }
))

# 12. BTC LONG no stop, 20x leverage, same drawdown = EXIT IMMEDIATELY
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $87,200
- Current Price: $86,500
- P&L: -0.8% (no R-multiple, no stop)
- Stop Loss: NONE
- Position Size: 0.5 BTC
- Leverage: 20x
- Time in Trade: 2h

MARKET DATA:
- ATR(14): $1,500
- CVD: Slightly bearish
- Funding Rate: +0.010%
- Open Interest: $19.5B
- Volume: 1.0x average
- 15m Trend: Ranging, no strong direction
- 1H Structure: Holding above $86,000 support zone
- Orderbook: Balanced

DECISION REQUIRED: No stop loss on a 20x leveraged position. Small drawdown but high exposure. What do we do?""",
    {
        "action": "EXIT_100_PERCENT_IMMEDIATELY",
        "urgency": "CRITICAL",
        "confidence": 0.95,
        "reason": "No stop loss on a 20x leveraged position is unacceptable risk. At 20x, a 5% adverse move would wipe out the entire position. Liquidation price is approximately $82,840. Exit now and re-enter with proper risk management.",
        "reasoning": {
            "structure_analysis": "Structure is irrelevant when a 20x position has no stop loss. The liquidation price at approximately $82,840 is only $3,660 below current price (4.2%). A single 1H candle could move that distance in volatile conditions.",
            "data_assessment": "Market data doesn't matter here. The fundamental issue is undefined risk at extreme leverage. Even if every indicator was bullish, a 20x position without a stop is a ticking time bomb.",
            "risk_factors": "At 20x leverage, the -0.8% unrealized loss equals -16% of margin. Liquidation is approximately 4.2% away. A flash crash, exchange outage, or sudden sell-off could liquidate the entire position in seconds. There is no stop to prevent this.",
            "exit_logic": "Mandatory exit. No stop + 20x leverage = unacceptable risk regardless of market conditions. This is not a trade management decision -- it's a risk management emergency."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# ============================================================
# CATEGORY 4: Confidence calibration examples (medium confidence)
# ============================================================

# 13. BTC LONG uncertain setup, mixed signals = low confidence HOLD
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $78,400
- Current Price: $78,100
- P&L: -0.2R
- Stop Loss: $76,800
- Guarding Line: $77,500
- TP1: $82,000 (30%)
- Position Size: 0.3 BTC
- Leverage: 3x
- Time in Trade: 10h

MARKET DATA:
- ATR(14): $1,380
- CVD: Mixed -- bullish on 15m, bearish on 1H
- Funding Rate: +0.003%
- Open Interest: $17.2B (flat)
- Volume: 0.9x average
- 15m Trend: Choppy, oscillating around $78,000-$78,400
- 1H Structure: Neither higher highs nor lower lows -- flat consolidation
- Orderbook: Balanced, no strong bias

DECISION REQUIRED: Everything is mixed. No strong signal in either direction. What do we do with this uncertainty?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.55,
        "reason": "No structural damage but no confirmation either. With stop defined at $76,800 and low leverage (3x), the trade plan is intact. Low confidence reflects genuinely mixed signals.",
        "reasoning": {
            "structure_analysis": "Flat consolidation with no directional bias. No higher highs (bullish) or lower lows (bearish). This is an information vacuum -- the market hasn't decided yet. Guarding line at $77,500 is holding but untested.",
            "data_assessment": "CVD divergence between timeframes (15m bullish, 1H bearish) means the timeframes disagree. Flat OI and average volume confirm the indecision. Neutral funding adds nothing. There's simply no edge in either direction right now.",
            "risk_factors": "At 3x leverage with stop at $76,800, the maximum downside is -1.0R (from current -0.2R to stop). Risk-reward to TP1 at $82,000 is still favorable at roughly 2.5:1. The low leverage means the consolidation isn't burning capital through funding.",
            "exit_logic": "No exit triggers active. The uncertainty is reflected in the low confidence score (0.55) rather than in the action. With defined risk and low leverage, HOLD through uncertainty is the correct play -- the stop exists precisely for scenarios where the consolidation resolves downward."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 14. ETH LONG mixed signals, could go either way = medium confidence HOLD
examples.append(make_example(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: LONG
- Entry: $3,650
- Current Price: $3,620
- P&L: -0.3R
- Stop Loss: $3,530
- Guarding Line: $3,580
- TP1: $3,850 (25%)
- Position Size: 6.0 ETH
- Leverage: 5x
- Time in Trade: 14h

MARKET DATA:
- ATR(14): $92
- CVD: Neutral
- Funding Rate: +0.008%
- Open Interest: $8.5B (down 0.8%)
- Volume: 0.8x average
- 15m Trend: Slight upward drift but weak
- 1H Structure: Consolidating at midrange, no clear pattern
- Orderbook: Slight ask pressure above $3,640

DECISION REQUIRED: ETH is drifting with no clear direction. Position is slightly underwater. Should we be patient or cut our losses?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.58,
        "reason": "No structural damage. Stop at $3,530 gives $90 of room (~1 ATR). Low confidence reflects the lack of confirming signals, but defined risk means patience is justified.",
        "reasoning": {
            "structure_analysis": "Midrange consolidation with no structural breaks. The guarding line at $3,580 is holding. The 15m shows a slight upward drift which is mildly constructive. No major support or resistance levels are being tested.",
            "data_assessment": "Neutral CVD and below-average volume suggest the market is in wait-and-see mode. OI slightly declining shows some deleveraging which removes weak hands. Slight ask pressure at $3,640 is minor.",
            "risk_factors": "At 5x leverage, the remaining risk to stop is well-defined. The trade needs to play out -- 14 hours is not long enough to declare the thesis dead when no structure has broken.",
            "exit_logic": "No exit triggers. Low confidence (0.58) reflects uncertainty, not danger. The correct response to uncertainty with defined risk is patience, not premature exit."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 15. BTC SHORT losing but thesis might be right = 0.60 confidence HOLD
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $93,500
- Current Price: $94,100
- P&L: -0.4R
- Stop Loss: $95,200
- Guarding Line: $94,500
- TP1: $90,000 (30%)
- Position Size: 0.2 BTC
- Leverage: 3x
- Time in Trade: 8h

MARKET DATA:
- ATR(14): $1,580
- CVD: Bullish on 15m, neutral on 4H
- Funding Rate: +0.028%
- Open Interest: $22.0B (up 3.5% in 4h)
- Volume: 1.2x average
- 15m Trend: Pushing up toward $94,300
- 1H Structure: Testing resistance at $94,200, double top possible
- Orderbook: Mixed, ask wall at $94,500

DECISION REQUIRED: Short is red but the crowded long thesis (0.028% funding) hasn't played out yet. Are we early or wrong?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.60,
        "reason": "Crowded long funding at +0.028% is extreme and unsustainable. The contrarian thesis hasn't been invalidated yet -- it just hasn't triggered. Stop at $95,200 contains the risk. Early is not wrong.",
        "reasoning": {
            "structure_analysis": "Price is testing resistance at $94,200 where a potential double top could form. The guarding line at $94,500 aligns with the ask wall. If $94,200 rejects again, the short thesis strengthens significantly.",
            "data_assessment": "Funding at +0.028% is in the 95th percentile historically -- longs are paying shorts nearly 3x the normal rate. This level of crowding typically resolves with a sharp move down as leveraged longs capitulate. OI up 3.5% means more longs entering the crowded trade.",
            "risk_factors": "The 15m bullish push is concerning but hasn't broken the guarding line. At 3x leverage, the -0.4R drawdown is manageable. Stop at $95,200 gives $1,100 of room (0.7 ATR). The risk is defined.",
            "exit_logic": "The short thesis is based on crowded funding resolving to the downside -- this is a timing trade. The setup hasn't been invalidated (guarding line holding, resistance intact). Low confidence (0.60) reflects timing uncertainty, but the fundamental thesis remains valid."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# ============================================================
# CATEGORY 5: HOLD winning positions (don't exit winners too early)
# ============================================================

# 16. BTC LONG +1.5R, trending, CVD bearish on 15m but 1H fine = HOLD
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $82,000
- Current Price: $84,200
- P&L: +1.5R
- Stop Loss: $80,200
- Guarding Line: $83,000
- Trailing Stop: $82,800
- TP1: $86,000 (30%) -- not yet hit
- Position Size: 0.35 BTC
- Leverage: 5x
- Time in Trade: 22h

MARKET DATA:
- ATR(14): $1,450
- CVD: Bearish on 15m, bullish on 1H and 4H
- Funding Rate: +0.012%
- Open Interest: $18.9B (up 1.2%)
- Volume: 1.1x average
- 15m Trend: Minor pullback from $84,500, forming flag pattern
- 1H Structure: Higher highs and higher lows intact, clean uptrend
- Orderbook: Bid support at $84,000, asks thinning above $84,500

DECISION REQUIRED: Winner pulling back on short timeframe. CVD on 15m is bearish. Should we take profit or let it run?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.78,
        "reason": "1H and 4H structure is clean uptrend with higher highs/lows intact. 15m pullback forming a bull flag is bullish continuation. TP1 at $86,000 is only $1,800 away. Let the winner run.",
        "reasoning": {
            "structure_analysis": "The 1H structure is a textbook uptrend -- consecutive higher highs and higher lows. The 15m 'pullback' is forming a flag pattern after the push to $84,500, which is a continuation pattern. The guarding line at $83,000 is $1,200 below current price, well-protected.",
            "data_assessment": "15m CVD being bearish during a bull flag pullback is normal -- it reflects short-term profit-taking, not trend reversal. The 1H and 4H CVD remaining bullish confirms the higher timeframe trend. Funding at +0.012% is elevated but not extreme.",
            "risk_factors": "Trailing stop at $82,800 protects +0.55R of profit in the worst case. The risk of holding is limited to giving back some profit to the trailing stop. The reward is TP1 at $86,000 (+2.75R total).",
            "exit_logic": "No exit triggers. Trailing stop handles the downside protection. The uptrend is intact. Taking profit at +1.5R when TP1 is +2.75R would be cutting a winner short based on noise. Let the trade plan execute."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 17. ETH SHORT +2.0R, slight bounce but structure bearish = HOLD
examples.append(make_example(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,800
- Current Price: $3,620
- P&L: +2.0R
- Stop Loss: $3,900
- Guarding Line: $3,700
- Trailing Stop: $3,710
- TP1: $3,550 (30%) -- approaching
- TP2: $3,400 (50%)
- Position Size: 5.0 ETH
- Leverage: 3x
- Time in Trade: 30h

MARKET DATA:
- ATR(14): $95
- CVD: Neutral on 15m (bounce), bearish on 1H
- Funding Rate: -0.020%
- Open Interest: $7.8B (down 5.2% in 12h)
- Volume: 0.9x average
- 15m Trend: Small bounce from $3,600 to $3,620
- 1H Structure: Lower highs and lower lows, clean downtrend
- Orderbook: Asks building at $3,650-$3,670, bids thin below $3,600

DECISION REQUIRED: Big winner with a small bounce. TP1 is close. Should we take profit on the bounce or wait for the target?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.75,
        "reason": "1H downtrend intact with lower highs/lows. TP1 at $3,550 is only $70 away (~0.7 ATR). Small bounce to $3,620 is noise within the bearish structure. Trailing stop at $3,710 protects +1.0R minimum.",
        "reasoning": {
            "structure_analysis": "The 1H downtrend is clean -- consecutive lower highs and lower lows. The $20 bounce from $3,600 to $3,620 is 0.21 ATR -- completely insignificant in the context of a $180 downmove. TP1 at $3,550 is highly likely to be hit given the intact structure.",
            "data_assessment": "OI down 5.2% in 12h confirms long liquidation cascade. Negative funding at -0.020% means shorts are crowded but this resolves at TP1, not here. Neutral CVD on 15m during a bounce in a downtrend means buyers aren't committing.",
            "risk_factors": "Trailing stop at $3,710 protects +1.0R profit minimum. Even if the bounce extends, the worst case is a +1.0R win. The upside is TP1 at $3,550 (+2.6R) and TP2 at $3,400 (+4.2R).",
            "exit_logic": "No exit triggers. Trailing stop handles protection. The downtrend is intact. Front-running TP1 by $70 would leave significant money on the table if the trend continues to TP2."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# ============================================================
# CATEGORY 6: TP_PARTIAL examples with lower confidence
# ============================================================

# 18. BTC LONG at TP1 with weakening momentum = TP_PARTIAL 0.65 confidence
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $79,500
- Current Price: $83,400
- P&L: +2.6R
- Stop Loss: $78,000
- Guarding Line: $81,500
- Trailing Stop: $82,200
- TP1: $83,500 (30%) -- almost hit
- TP2: $87,000 (50%)
- Position Size: 0.4 BTC
- Leverage: 5x
- Time in Trade: 36h

MARKET DATA:
- ATR(14): $1,500
- CVD: Weakening, turning from bullish to neutral
- Funding Rate: +0.018%
- Open Interest: $20.5B (up 4.8% in 8h)
- Volume: 1.4x average but declining from peak
- 15m Trend: Slowing momentum, candles getting smaller
- 1H Structure: Uptrend intact but approaching resistance zone at $83,500-$84,000
- Orderbook: Large ask wall at $83,500, bids pulling back

DECISION REQUIRED: Approaching TP1 but momentum is fading. CVD weakening. Should we take the 30% or hold for more?""",
    {
        "action": "TP_PARTIAL",
        "urgency": "MEDIUM",
        "confidence": 0.65,
        "reason": "TP1 at $83,500 is $100 away with weakening momentum and CVD turning neutral. Take the planned 30% partial. Keep 70% with trailing stop for TP2.",
        "reasoning": {
            "structure_analysis": "Price approaching the planned TP1 at $83,500 which aligns with a major resistance zone ($83,500-$84,000). The 15m candles are getting smaller (declining momentum). This resistance zone could cause a multi-hour consolidation or pullback.",
            "data_assessment": "CVD weakening from bullish to neutral at a resistance zone is a distribution signal. Elevated funding at +0.018% with OI up 4.8% means the long trade is getting crowded. Volume declining from peak is bearish divergence.",
            "risk_factors": "Taking 30% at TP1 locks in profit on the planned portion. The remaining 70% stays with a trailing stop at $82,200 to capture TP2 if the trend continues. Lower confidence (0.65) reflects uncertainty about whether momentum carries through $83,500.",
            "exit_logic": "Execute the trade plan: 30% at TP1. The weakening signals don't justify a full exit (structure intact, trailing stop protects) but they do justify taking planned profit rather than greedily holding through resistance."
        },
        "execution": {"exit_pct": 30, "stop_price": None, "order_type": "LIMIT"}
    }
))

# ============================================================
# CATEGORY 7: REDUCE_SIZE for high-leverage uncomfortable positions
# ============================================================

# 19. BTC LONG 20x in choppy market, not wrong but too much risk = REDUCE
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $85,000
- Current Price: $85,300
- P&L: +0.2R
- Stop Loss: $83,500
- Guarding Line: $84,500
- Position Size: 1.0 BTC
- Leverage: 20x
- Time in Trade: 4h

MARKET DATA:
- ATR(14): $1,900
- CVD: Neutral
- Funding Rate: +0.015%
- Open Interest: $19.8B (up 2.5%)
- Volume: 0.8x average
- 15m Trend: Flat, no momentum
- 1H Structure: Above entry but range-bound
- Orderbook: Balanced

DECISION REQUIRED: Small winner at extreme leverage. Market is flat. The 20x feels risky in this low-momentum environment. What should we do?""",
    {
        "action": "REDUCE_SIZE",
        "urgency": "MEDIUM",
        "confidence": 0.72,
        "reason": "20x leverage in a flat, low-momentum market is excessive risk for minimal edge. Reduce to 5x by selling 75% of position. Keep 25% with existing stop for the breakout if it comes.",
        "reasoning": {
            "structure_analysis": "Price is in a range with no directional momentum. At 20x leverage, the position is profitable but the risk-reward of holding the full size in a range is poor. A $1,900 ATR at 20x means liquidation is approximately $4,250 away -- only 2.2x ATR.",
            "data_assessment": "Neutral CVD and below-average volume confirm no directional edge. Funding at +0.015% means the position costs money to hold. OI rising in a flat market means leverage is building without direction -- this often resolves with a violent move.",
            "risk_factors": "The primary risk is the 20x leverage in an environment with no edge. A sudden move against could cascade through the stop with severe slippage. Reducing to 5x maintains the trade thesis while dramatically reducing liquidation risk.",
            "exit_logic": "Not a full exit -- the trade idea isn't wrong. But the position size is too large for the environment. Reduce 75% to bring effective leverage to 5x. If the breakout comes, the remaining 25% captures the move."
        },
        "execution": {"exit_pct": 75, "stop_price": None, "order_type": "MARKET"}
    }
))

# ============================================================
# CATEGORY 8: More HOLD scenarios for volatile but intact setups
# ============================================================

# 20. BTC LONG volatile day, big swings but trend intact = HOLD
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $76,800
- Current Price: $76,200
- P&L: -0.4R
- Stop Loss: $74,500
- Guarding Line: $75,800
- TP1: $80,000 (30%)
- Position Size: 0.25 BTC
- Leverage: 5x
- Time in Trade: 15h

MARKET DATA:
- ATR(14): $2,200
- CVD: Choppy but net bullish on 4H
- Funding Rate: -0.005%
- Open Interest: $16.5B (down 3.8% in 6h)
- Volume: 1.6x average (high volatility day)
- 15m Trend: Large swings, just bounced from $75,600 to $76,200
- 1H Structure: Wide range day, but higher lows holding ($75,200 -> $75,600)
- Orderbook: Dynamic, shifting rapidly

DECISION REQUIRED: Very volatile day with big swings. Position is slightly red. Tempted to exit to avoid the stress. Is volatility a reason to exit?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.63,
        "reason": "Volatility is not a sell signal when structure holds. Higher lows on 1H ($75,200 -> $75,600) confirm buyers are stepping in at successively higher prices. Stop at $74,500 gives 1.5 ATR of room. OI deleveraging means weak hands are being flushed.",
        "reasoning": {
            "structure_analysis": "Despite the large swings, the 1H higher lows are intact: $75,200 then $75,600. This pattern of higher lows during high volatility is bullish -- it means each dip is met with stronger buying. The guarding line at $75,800 was tested but held.",
            "data_assessment": "4H CVD being net bullish despite choppy price action means accumulation is happening during the dips. OI down 3.8% is aggressive deleveraging which removes weak positions from the market. Negative funding means longs aren't crowded. High volume with higher lows is bullish.",
            "risk_factors": "The high ATR ($2,200) means normal daily movement is $2,200. The current -$600 drawdown is only 0.27 ATR -- well within normal. Stop at $74,500 gives $1,700 of room. At 5x leverage, this is an acceptable risk for the setup quality.",
            "exit_logic": "Volatility creates emotional discomfort but is not a technical sell signal. The structure (higher lows), data (bullish 4H CVD, OI flush), and risk management (wide stop, manageable leverage) all support holding. Exiting because of stress would be emotional, not strategic."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 21. SOL LONG flash dip recovery, scary but structure held = HOLD
examples.append(make_example(
    """POSITION STATE:
- Asset: SOL/USDT
- Direction: LONG
- Entry: $155.00
- Current Price: $153.50
- P&L: -0.3R
- Stop Loss: $148.00
- Guarding Line: $152.00
- TP1: $165.00 (30%)
- Position Size: 40 SOL
- Leverage: 3x
- Time in Trade: 6h

MARKET DATA:
- ATR(14): $5.80
- CVD: Sharp V-recovery, now bullish
- Funding Rate: -0.010%
- Open Interest: $1.9B (down 6.2% in 1h -- flash liquidation)
- Volume: 3.2x average (spike during flash dip)
- 15m Trend: V-shaped recovery from $150.80 low, now at $153.50
- 1H Structure: Wick down to $150.80 but body closed above $152.50
- Orderbook: Massive bids rebuilding at $152.00-$153.00 post-flush

DECISION REQUIRED: Flash dip just happened -- price dropped to $150.80 then bounced hard. Still slightly red. Was that a warning sign or an opportunity?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.70,
        "reason": "Flash dip was a leverage flush, not a structural break. OI dropped 6.2% in 1h (liquidation cascade). V-recovery with CVD turning bullish and massive bids rebuilding confirms the dip was absorbed. Stop at $148 was never threatened.",
        "reasoning": {
            "structure_analysis": "The 1H candle wicked down to $150.80 but closed above $152.50 -- this is a massive rejection wick, which is a bullish signal. The body closing above the guarding line at $152.00 means the structure is technically intact. Flash dips that recover quickly are typically stop-hunts followed by continuation in the original direction.",
            "data_assessment": "OI dropping 6.2% in 1h is a textbook liquidation cascade -- overleveraged positions were flushed. This is healthy and removes potential sellers. CVD showing a sharp V-recovery means the selling was absorbed instantly by real buyers. Negative funding means longs aren't crowded post-flush. Volume at 3.2x was the flush itself, not sustained selling.",
            "risk_factors": "The flash dip low at $150.80 was $2.80 above the stop at $148.00. Even during the worst of the flush, the stop had room. At 3x leverage, the risk is manageable. The flush actually improves the setup by removing weak hands.",
            "exit_logic": "Flash liquidation cascades that recover are bullish continuation signals. The market just removed overleveraged positions and rebuilt the bid wall. The trade thesis is actually STRONGER after this event, not weaker."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 22. XRP LONG -0.8R, low leverage, trend intact but patience needed
examples.append(make_example(
    """POSITION STATE:
- Asset: XRP/USDT
- Direction: LONG
- Entry: $2.45
- Current Price: $2.38
- P&L: -0.8R
- Stop Loss: $2.28
- Guarding Line: $2.35
- TP1: $2.70 (25%)
- Position Size: 2000 XRP
- Leverage: 2x
- Time in Trade: 28h

MARKET DATA:
- ATR(14): $0.09
- CVD: Neutral
- Funding Rate: +0.003%
- Open Interest: $580M (flat)
- Volume: 0.6x average
- 15m Trend: Flat, micro-ranging at $2.37-$2.39
- 1H Structure: Consolidating above $2.35 support (tested 3 times, held)
- Orderbook: Moderate bids at $2.35-$2.36, balanced asks

DECISION REQUIRED: XRP is going nowhere at -0.8R. We've been in this trade for 28 hours. Is it time to cut the opportunity cost?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.60,
        "reason": "Structure intact -- $2.35 support tested 3 times and held. Low leverage (2x) means the opportunity cost of holding is minimal. Trade needs more time, not more action. 28h is not excessive for a swing trade.",
        "reasoning": {
            "structure_analysis": "The $2.35 support has been tested three times and held each time -- this is a strong base. Triple tests of support that hold typically precede upside breakouts. The consolidation pattern is constructive, not bearish.",
            "data_assessment": "Low volume and neutral CVD reflect the waiting period. XRP is consolidating while building a base -- this is normal for alts that move in bursts. Flat OI means no aggressive positioning in either direction.",
            "risk_factors": "At 2x leverage, the -0.8R drawdown is less than 1% of portfolio. The opportunity cost argument is weak because the position is small relative to capital and there's no better setup visible. Stop at $2.28 gives $0.10 of room (1.1 ATR).",
            "exit_logic": "No exit triggers. The boredom of a consolidation is not a sell signal. The trade needs time and patience. If $2.35 breaks on the 4th test, then exit. But 3 successful tests and holding is bullish, not bearish."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# ============================================================
# CATEGORY 9: EXIT examples the model was getting wrong (false holds)
# ============================================================

# 23. BTC LONG guarding line broken, CVD bearish, OI rising = EXIT (not hold!)
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $90,500
- Current Price: $88,800
- P&L: -1.1R
- Stop Loss: $87,900
- Guarding Line: $89,500
- Position Size: 0.3 BTC
- Leverage: 5x
- Time in Trade: 16h

MARKET DATA:
- ATR(14): $1,550
- CVD: Bearish, accelerating
- Funding Rate: +0.020%
- Open Interest: $21.2B (up 4.1% in 4h)
- Volume: 1.4x average
- 15m Trend: Lower highs and lower lows, clean downtrend
- 1H Structure: Broke below $89,500 guarding line, now resistance
- Orderbook: Asks building above $89,000, bids retreating

DECISION REQUIRED: We broke below the guarding line. But we're only -1.1R with stop at $87,900 giving more room. Hold to stop or exit on the structure break?""",
    {
        "action": "EXIT_FULL",
        "urgency": "HIGH",
        "confidence": 0.83,
        "reason": "Guarding line at $89,500 broken with bearish CVD acceleration and rising OI. Structure is invalidated. Exit at -1.1R rather than riding to stop at -1.0R total risk with likely slippage.",
        "reasoning": {
            "structure_analysis": "The guarding line at $89,500 was the key structural level for this trade. Breaking below it with follow-through invalidates the long thesis. The level has flipped from support to resistance with asks building above $89,000. The 15m is printing clean lower highs and lower lows.",
            "data_assessment": "Bearish CVD accelerating confirms genuine selling pressure. OI rising 4.1% in 4h with price falling means new shorts are entering -- this is trend continuation. Crowded funding at +0.020% means longs are paying shorts and more long liquidations are likely.",
            "risk_factors": "Stop at $87,900 is $900 away, which seems like room to recover. But the probability of recovery is low with broken structure, bearish CVD, and rising OI. More likely, price continues to stop with slippage. Taking -1.1R now is better than -1.3R or -1.5R at stop with slippage.",
            "exit_logic": "EXIT ON STRUCTURE BREAKS, not at arbitrary stops. The guarding line break IS the signal. Waiting for the hard stop when the trade thesis is already dead is just hoping, not managing."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# 24. ETH SHORT guarding line broken, trend reversed = EXIT
examples.append(make_example(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,300
- Current Price: $3,365
- P&L: -0.7R
- Stop Loss: $3,420
- Guarding Line: $3,350
- Position Size: 6.0 ETH
- Leverage: 5x
- Time in Trade: 9h

MARKET DATA:
- ATR(14): $90
- CVD: Strongly bullish
- Funding Rate: -0.012%
- Open Interest: $8.8B (up 2.8% in 2h)
- Volume: 1.3x average
- 15m Trend: Higher highs, breakout above $3,340 consolidation
- 1H Structure: Previous lower high at $3,360 now being exceeded
- Orderbook: Bids aggressive, asks retreating above $3,370

DECISION REQUIRED: Short thesis was lower highs. The latest push above $3,360 breaks that pattern. Is it invalidated?""",
    {
        "action": "EXIT_FULL",
        "urgency": "HIGH",
        "confidence": 0.80,
        "reason": "Short thesis (lower highs) invalidated by push above $3,360. Guarding line at $3,350 broken. Strongly bullish CVD with negative funding (short squeeze risk). Exit at -0.7R.",
        "reasoning": {
            "structure_analysis": "The short thesis relied on the pattern of lower highs. The push above $3,360 (previous lower high) breaks this pattern. The guarding line at $3,350 is broken. The 15m shows a breakout from the $3,340 consolidation with follow-through.",
            "data_assessment": "Strongly bullish CVD confirms real buying. Negative funding at -0.012% means shorts are paying longs -- short squeeze fuel. OI up 2.8% in 2h with price rising means new longs entering. All data confirms the bearish thesis is dead.",
            "risk_factors": "Only $55 from stop at $3,420. At 5x leverage, waiting for stop risks slippage of $10-15 in a squeeze. Taking -0.7R now saves potential -0.85R to -1.0R at stop.",
            "exit_logic": "The structural basis for the trade (lower highs pattern) is broken. Exit on structure break, don't wait for the hard stop."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# 25. BTC LONG, stop too tight for the volatility = EXIT and reassess
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $71,500
- Current Price: $71,200
- P&L: -0.2R
- Stop Loss: $70,800
- Guarding Line: $71,000
- Position Size: 0.8 BTC
- Leverage: 10x
- Time in Trade: 1h

MARKET DATA:
- ATR(14): $1,800
- CVD: Neutral
- Funding Rate: +0.010%
- Open Interest: $18.0B (up 3.5% in 30m)
- Volume: 2.0x average (earnings/news event)
- 15m Trend: Volatile, $400 swings in each direction
- 1H Structure: Cannot determine, only 1 candle since entry
- Orderbook: Extremely thin, depth 50% below normal

DECISION REQUIRED: Just entered a long during a high-volatility event. Stop is only $400 away (0.22 ATR) at 10x leverage. The stop feels too tight for this environment. What do we do?""",
    {
        "action": "EXIT_FULL",
        "urgency": "MEDIUM",
        "confidence": 0.78,
        "reason": "Stop at $70,800 is only 0.22 ATR away during a 2x volume event with thin orderbook. At 10x leverage, the stop will almost certainly be hit by noise, likely with severe slippage. Exit, reassess, and re-enter with a wider stop or lower leverage.",
        "reasoning": {
            "structure_analysis": "With only 1h of data and $400 swings on the 15m, there's no reliable structure to analyze. The stop at $70,800 is within the normal noise range of the current volatility environment.",
            "data_assessment": "Volume at 2x average with 50% thinner orderbook depth means each swing is larger and faster than normal. The stop is placed at 0.22 ATR -- in this environment, that's essentially guaranteed to be hit by a random wick. OI surging 3.5% in 30m means aggressive leveraged positions are being built, increasing volatility.",
            "risk_factors": "At 10x leverage with a $400 stop in an $1,800 ATR environment, the probability of being stopped on noise exceeds 80%. Slippage in thin books during high-volatility events can easily double the expected loss. This isn't a bad trade idea -- it's bad position sizing for the environment.",
            "exit_logic": "Exit, not because the direction is wrong, but because the stop-to-ATR ratio is too tight for the volatility. Re-enter with either a wider stop ($69,000, 1.4 ATR) at lower leverage (3x) or wait for volatility to normalize."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# ============================================================
# Additional examples to round out the set
# ============================================================

# 26. HOLD: BTC SHORT winning, small bounce, don't panic
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $92,000
- Current Price: $89,800
- P&L: +1.8R
- Stop Loss: $93,500
- Guarding Line: $91,000
- Trailing Stop: $91,200
- TP1: $88,500 (30%)
- Position Size: 0.2 BTC
- Leverage: 3x
- Time in Trade: 42h

MARKET DATA:
- ATR(14): $1,600
- CVD: Neutral (bounce in progress)
- Funding Rate: +0.025%
- Open Interest: $20.0B (down 4.0% in 12h)
- Volume: 0.7x average
- 15m Trend: Bouncing from $89,500 to $89,800
- 1H Structure: Clean downtrend, this bounce hasn't made a higher high
- Orderbook: Ask wall at $90,200, moderate bids

DECISION REQUIRED: Short is up +1.8R but bouncing $300. Should we take profit or trust the trend?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.74,
        "reason": "A $300 bounce (0.19 ATR) in a downtrend is noise. 1H structure shows no higher high has been made. Trailing stop at $91,200 protects +0.65R minimum. TP1 at $88,500 is $1,300 away. Let the winner run.",
        "reasoning": {
            "structure_analysis": "The downtrend remains intact -- the bounce from $89,500 to $89,800 has NOT made a higher high on the 1H. The ask wall at $90,200 should cap any further bounce. The guarding line at $91,000 is $1,200 above current price.",
            "data_assessment": "Neutral CVD during a bounce means no real buying conviction. Low volume at 0.7x during the bounce confirms it's a dead cat bounce, not a reversal. Extreme funding at +0.025% means longs are still crowded. OI declining 4% confirms long liquidation cascade is ongoing.",
            "risk_factors": "Trailing stop at $91,200 ensures minimum +0.65R profit. The bounce is within normal parameters. Risk of holding is giving back profit to trailing stop ($91,200). Reward is TP1 at $88,500 (another +0.8R).",
            "exit_logic": "The bounce is 0.19 ATR -- statistically insignificant. The trend is intact, the trailing stop protects profit, and the target is close. Exiting on a $300 bounce in a $2,200 downmove would be emotional."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 27. HOLD: LINK LONG, alt consolidation after breakout
examples.append(make_example(
    """POSITION STATE:
- Asset: LINK/USDT
- Direction: LONG
- Entry: $18.50
- Current Price: $18.20
- P&L: -0.4R
- Stop Loss: $17.50
- Guarding Line: $18.00
- TP1: $20.50 (30%)
- Position Size: 200 LINK
- Leverage: 3x
- Time in Trade: 20h

MARKET DATA:
- ATR(14): $0.85
- CVD: Neutral
- Funding Rate: +0.002%
- Open Interest: $310M (down 1.2%)
- Volume: 0.7x average
- 15m Trend: Flat, tight range $18.10-$18.30
- 1H Structure: Consolidating after breakout above $17.80, holding above
- Orderbook: Bids building at $18.00-$18.10

DECISION REQUIRED: LINK broke out then pulled back and is consolidating. Are we trapped in a false breakout or is this normal post-breakout behavior?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.67,
        "reason": "Post-breakout consolidation above the breakout level ($17.80) is the most bullish pattern in crypto. Price is holding above the guarding line at $18.00. Low volume consolidation after a breakout precedes the next leg up. Patience required.",
        "reasoning": {
            "structure_analysis": "Price broke above $17.80 resistance, pulled back to $18.10-$18.30, and is now consolidating ABOVE the breakout level. This is textbook bullish re-accumulation. The former resistance at $17.80 is now support, and the guarding line at $18.00 is holding.",
            "data_assessment": "Low volume during consolidation is bullish -- it means sellers are exhausted and the market is waiting for the next catalyst. Neutral CVD during consolidation is normal. Bids building at $18.00-$18.10 shows institutional accumulation during the quiet period.",
            "risk_factors": "At 3x leverage with stop at $17.50, the maximum loss is -1.0R. The -0.4R drawdown is within the normal pullback range after a breakout. LINK often consolidates for 24-48h after breakouts before the next leg.",
            "exit_logic": "No exit triggers. Guarding line holding. Breakout level holding. Volume profile is constructive. The consolidation is the trade -- this is where patience gets rewarded."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 28. EXIT: ARB SHORT, trend changed, don't hold
examples.append(make_example(
    """POSITION STATE:
- Asset: ARB/USDT
- Direction: SHORT
- Entry: $1.25
- Current Price: $1.32
- P&L: -0.9R
- Stop Loss: $1.35
- Guarding Line: $1.29
- Position Size: 3000 ARB
- Leverage: 5x
- Time in Trade: 11h

MARKET DATA:
- ATR(14): $0.065
- CVD: Bullish, sustained for 6h
- Funding Rate: -0.022%
- Open Interest: $280M (up 8.2% in 4h)
- Volume: 2.1x average
- 15m Trend: Parabolic move, consecutive higher highs
- 1H Structure: Broke above falling wedge, now in impulse wave
- Orderbook: Aggressive market buys, asks being absorbed instantly

DECISION REQUIRED: ARB is in a parabolic breakout. Short is getting destroyed. Only $0.03 from stop. Hold for a miracle or accept defeat?""",
    {
        "action": "EXIT_FULL",
        "urgency": "CRITICAL",
        "confidence": 0.92,
        "reason": "Falling wedge broken to upside with parabolic impulse. Short thesis completely invalidated. Negative funding = short squeeze in progress. Exit immediately at -0.9R before slippage at stop makes it worse.",
        "reasoning": {
            "structure_analysis": "The falling wedge has broken to the upside -- this is one of the strongest bullish reversal patterns. The 15m is in a parabolic impulse with consecutive higher highs. The guarding line at $1.29 was broken $0.03 ago. Every structural element of the short thesis is destroyed.",
            "data_assessment": "CVD bullish for 6h sustained confirms genuine buying. OI surging 8.2% in 4h with parabolic price action means massive new longs entering. Negative funding at -0.022% means shorts are being squeezed -- this is the liquidation cascade. Volume at 2.1x with asks being absorbed instantly is peak momentum.",
            "risk_factors": "Only $0.03 from stop in a parabolic move. At 5x leverage, stop slippage during a squeeze could easily add $0.02-0.04 of additional loss (0.3-0.6R extra). Market exit now saves this slippage.",
            "exit_logic": "Every signal screams exit: broken structure, bullish CVD, squeeze conditions, parabolic momentum, proximity to stop. There is zero probability of this short working. Take -0.9R now rather than -1.2R to -1.5R at stop with slippage."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# 29. HOLD: BTC LONG, briefly touched guarding line but bounced = HOLD
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $68,500
- Current Price: $67,800
- P&L: -0.5R
- Stop Loss: $66,000
- Guarding Line: $67,500
- TP1: $72,000 (25%)
- Position Size: 0.3 BTC
- Leverage: 3x
- Time in Trade: 18h

MARKET DATA:
- ATR(14): $1,400
- CVD: Neutral, recovering from bearish
- Funding Rate: -0.007%
- Open Interest: $17.0B (down 2.5% in 4h)
- Volume: 1.2x average (elevated during dip)
- 15m Trend: V-bounce from $67,400 low, now at $67,800
- 1H Structure: Wick touched $67,400 (below guarding line) but body closed above $67,600
- Orderbook: Massive bids appeared at $67,400-$67,500 during the dip

DECISION REQUIRED: Price wicked below the guarding line at $67,500 but immediately bounced. The body closed above. Is this a stop hunt or a real break?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.62,
        "reason": "Wick below guarding line that immediately bounced is a stop hunt pattern, not a structural break. 1H body closed above $67,600. Massive bid response at $67,400-$67,500 confirms institutional buying at the key level.",
        "reasoning": {
            "structure_analysis": "A wick below a key level that immediately bounces with body closing above is one of the most common stop-hunt patterns. The difference between a wick and a close is critical -- wicks represent failed tests, closes represent genuine breaks. The guarding line at $67,500 held on a closing basis.",
            "data_assessment": "CVD recovering from bearish (the dip) to neutral (the bounce) shows the selling was absorbed. Massive bids appearing at $67,400-$67,500 during the dip is institutional buying -- smart money was waiting for this exact dip. OI down 2.5% means leveraged positions were flushed (the stop-hunt worked, clearing out weak longs).",
            "risk_factors": "Lower confidence (0.62) because the wick DID touch below the guarding line -- it wasn't a clean hold. If the guarding line is tested again, the probability of breaking increases. But the response (massive bids, V-bounce) was strong enough to warrant holding for now.",
            "exit_logic": "Close below guarding line would trigger exit. But a wick below with body above is a bullish signal, especially with institutional bid response. HOLD but on elevated alert -- if $67,500 is tested again, the next test is more likely to break."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
))

# 30. EXIT: BTC LONG, second test of guarding line = EXIT (not hold again!)
examples.append(make_example(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $68,500
- Current Price: $67,400
- P&L: -0.8R
- Stop Loss: $66,000
- Guarding Line: $67,500
- TP1: $72,000 (25%)
- Position Size: 0.3 BTC
- Leverage: 3x
- Time in Trade: 22h

MARKET DATA:
- ATR(14): $1,400
- CVD: Bearish, sellers in control
- Funding Rate: -0.003%
- Open Interest: $16.8B (down 1.2% in 2h)
- Volume: 1.5x average
- 15m Trend: Second push below $67,500, this time holding below
- 1H Structure: Double test of $67,500 support, now breaking. Previous bounce high was $67,800 (lower high)
- Orderbook: Bids at $67,400-$67,500 from earlier are being eaten, new bids not appearing

DECISION REQUIRED: This is the SECOND test of the guarding line. Last time it bounced but now the bids are getting absorbed. Is this the real break?""",
    {
        "action": "EXIT_FULL",
        "urgency": "HIGH",
        "confidence": 0.82,
        "reason": "Second test of guarding line with bids being absorbed = genuine break. First test bounced (stop hunt), second test holding below with bearish CVD and disappearing bids = structural failure. Exit at -0.8R.",
        "reasoning": {
            "structure_analysis": "First test of $67,500: wicked below, bounced hard, bids appeared. Second test of $67,500: holding below, bids being eaten, no new bids appearing. The lower high at $67,800 between tests confirms a mini-downtrend has formed. Double tests that fail are one of the highest-conviction breakdown patterns.",
            "data_assessment": "CVD is now bearish (was neutral after the first bounce). Volume at 1.5x with price pushing below support means genuine selling. The key difference from the first test: bids are being absorbed instead of defended. The institutional buying that saved the first test is not present on the second.",
            "risk_factors": "Stop at $66,000 gives $1,400 of room. But with the guarding line broken and no bid support, the path to stop is unobstructed. Holding from -0.8R to -1.0R at stop (with likely slippage) gains nothing. Taking -0.8R now is the disciplined exit.",
            "exit_logic": "The distinction between the first and second test is critical: first test had a V-bounce with institutional bids -- bullish response. Second test has no recovery, bids disappearing, and bearish CVD. This is the real break. Exit on structure break, not at the hard stop."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
))

# ============================================================
# Save all examples
# ============================================================

output_path = os.path.join(os.path.dirname(__file__), "bastion_risk_v3_reinforcement.jsonl")

with open(output_path, 'w', encoding='utf-8') as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"Generated {len(examples)} examples -> {output_path}")

# Print summary
from collections import Counter
actions = Counter()
confidences = []
for ex in examples:
    asst = json.loads(ex['messages'][2]['content'])
    actions[asst['action']] += 1
    confidences.append(asst['confidence'])

print("\nAction Distribution:")
for action, count in actions.most_common():
    print(f"  {action}: {count}")

print(f"\nConfidence Range: {min(confidences):.2f} - {max(confidences):.2f}")
print(f"Average Confidence: {sum(confidences)/len(confidences):.2f}")

# Count by category
hold_count = actions.get('HOLD', 0)
exit_count = actions.get('EXIT_FULL', 0) + actions.get('EXIT_100_PERCENT_IMMEDIATELY', 0)
print(f"\nHOLD examples: {hold_count} ({hold_count/len(examples)*100:.0f}%)")
print(f"EXIT examples: {exit_count} ({exit_count/len(examples)*100:.0f}%)")
print(f"Other: {len(examples) - hold_count - exit_count}")
