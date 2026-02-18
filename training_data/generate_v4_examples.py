"""
BASTION v4 Training Data Generator
===================================
Targets 5 specific failure categories found in v3 audit:

1. SHORT stop direction logic (10 examples) - BLOCKING
2. Trail stop leverage scaling (6 examples) - BLOCKING
3. No-stop leverage tiers (8 examples) - HIGH
4. HOLD confidence calibration near stops (6 examples) - HIGH
5. "Let the stop work" / panic-near-stop fix (5 examples) - MEDIUM

Total: ~35 new examples
"""

import json

SYSTEM_PROMPT = (
    "You are BASTION Risk Intelligence \u2014 an autonomous trade management AI. "
    "You monitor live cryptocurrency positions and make execution decisions. "
    "You output structured JSON with action, reasoning, and execution parameters. "
    "PRIORITY ORDER: 1) Hard Stop breach \u2192 EXIT_100_PERCENT_IMMEDIATELY "
    "2) Safety Net break \u2192 EXIT_FULL 3) Guarding Line break \u2192 REDUCE_SIZE or EXIT_FULL "
    "4) Take Profit targets \u2192 TP_PARTIAL or TP_FULL 5) Trailing Stop updates \u2192 TRAIL_STOP "
    "6) Time-based exits \u2192 REDUCE_SIZE. "
    "Core philosophy: Exit on STRUCTURE BREAKS, not arbitrary targets. "
    "Let winners run when structure holds. Scale out intelligently \u2014 "
    "decide HOW MUCH to exit based on structure strength, R-multiple, and market context."
)

examples = []

def add(user, assistant):
    examples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant)}
        ]
    })


# =============================================================================
# CATEGORY 1: SHORT STOP DIRECTION LOGIC (10 examples)
# The model must learn: For SHORT, stop is ABOVE entry.
# Stop triggered = price rises ABOVE stop. Price BELOW stop = SAFE.
# =============================================================================

# 1.1: SHORT - price below stop = SAFE, should HOLD
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $88,500.00
- Current Price: $87,200.00
- P&L: +0.8R
- Stop Loss: $89,400.00
- TP1: $86,000.00
- Leverage: 10x
- Time in Trade: 8h

MARKET DATA:
- ATR(14): $1,200
- CVD: Negative, confirming sell pressure
- Funding Rate: +0.018% (crowded longs)
- Volume: Above average
- Trend: Bearish on 1H and 4H

DECISION REQUIRED: SHORT position at +0.8R. Stop loss at $89,400 is $2,200 ABOVE current price. Position is working. What action?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.80,
        "reason": "Short is working at +0.8R with stop loss safely $2,200 above current price. For a SHORT position, the stop is only triggered if price RISES above $89,400 \u2014 currently price is falling away from the stop. Bearish momentum confirms.",
        "reasoning": {
            "structure_analysis": "SHORT position stop check: stop at $89,400 is ABOVE current price of $87,200. For shorts, price must rise ABOVE the stop to trigger it. Price is moving AWAY from the stop (favorable). The $2,200 buffer represents 1.83 ATR of protection.",
            "data_assessment": "Bearish CVD, crowded long funding (+0.018%), and above-average volume all confirm the short thesis is intact. No reason to exit a working short.",
            "risk_factors": "10x leverage means a 10% adverse move would be catastrophic, but the stop provides 2.5% of protection which is adequate.",
            "exit_logic": "No exit triggers. Stop is not threatened (price is moving away from it). TP1 at $86,000 is $1,200 away \u2014 approximately 1 ATR. Hold for target."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

# 1.2: SHORT - price ABOVE stop = TRIGGERED, must exit
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $85,000.00
- Current Price: $86,800.00
- P&L: -1.2R
- Stop Loss: $86,500.00
- Leverage: 5x
- Time in Trade: 12h

MARKET DATA:
- ATR(14): $1,100
- CVD: Flipping positive
- Funding Rate: -0.005% (shorts crowded)
- Volume: Increasing on up candles
- Trend: Reversal forming on 1H

DECISION REQUIRED: SHORT position with price at $86,800 which is ABOVE the stop loss at $86,500. What action?""",
    {
        "action": "EXIT_FULL",
        "urgency": "CRITICAL",
        "confidence": 0.95,
        "reason": "SHORT stop loss at $86,500 has been breached \u2014 current price $86,800 is $300 ABOVE the stop. For a SHORT position, price rising above the stop means the stop is triggered. Exit 100% immediately.",
        "reasoning": {
            "structure_analysis": "SHORT position stop check: stop at $86,500, current price $86,800. For a SHORT, the stop triggers when price RISES ABOVE the stop level. Price is $300 above stop = stop is BREACHED. This is the Hard Stop rule \u2014 non-negotiable exit.",
            "data_assessment": "CVD flipping positive and increasing volume on up candles confirm the reversal. The short thesis is invalidated.",
            "risk_factors": "At 5x leverage with price above stop, every further adverse tick multiplies losses by 5x. Immediate exit is the only option.",
            "exit_logic": "Hard Stop breach is Rule #1 in the MCF exit hierarchy. No discretion allowed \u2014 exit 100% at market."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
)

# 1.3: SHORT - price well below stop, winning big, HOLD
add(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,800.00
- Current Price: $3,520.00
- P&L: +2.1R
- Stop Loss: $3,930.00
- TP1: $3,600.00 | TP2: $3,400.00
- Leverage: 3x
- Time in Trade: 22h

MARKET DATA:
- ATR(14): $85
- CVD: Strongly negative
- Volume: Declining slightly
- Trend: Bearish across all timeframes

DECISION REQUIRED: SHORT ETH at +2.1R, past TP1. Stop at $3,930 is far above. Should I take profit or let it run toward TP2?""",
    {
        "action": "TP_PARTIAL",
        "urgency": "MEDIUM",
        "confidence": 0.78,
        "reason": "Short past TP1 at +2.1R with TP2 at $3,400 still $120 away. Take 35% to lock profit while keeping 65% for continuation. Stop at $3,930 is $410 above current price (4.8 ATR) \u2014 well protected.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: stop at $3,930 is far ABOVE current price $3,520. For shorts, price must rise above $3,930 to trigger stop \u2014 that's a $410 adverse move, 4.8 ATR of buffer. Stop is nowhere near being threatened. Bearish structure intact with lower lows.",
            "data_assessment": "Strongly negative CVD confirms sellers in control. Volume declining slightly could indicate selling exhaustion approaching, warranting a partial take.",
            "risk_factors": "At 3x leverage, risk is manageable. The main risk is a bounce from TP2 area trapping late shorts.",
            "exit_logic": "Past TP1 \u2192 take partial per MCF hierarchy. 35% locks meaningful profit at +2.1R while 65% rides toward TP2. Move stop to breakeven on remaining."
        },
        "execution": {"exit_pct": 35, "stop_price": 3800.00, "order_type": "MARKET"}
    }
)

# 1.4: SHORT - price slightly below stop but NOT triggered
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $92,000.00
- Current Price: $93,100.00
- P&L: -0.6R
- Stop Loss: $93,500.00
- Leverage: 20x
- Time in Trade: 6h

MARKET DATA:
- ATR(14): $1,400
- CVD: Positive, buyers in control
- Funding Rate: +0.022% (crowded longs)
- Volume: Elevated
- Trend: Bullish 1H, neutral 4H

DECISION REQUIRED: SHORT at -0.6R with price at $93,100 approaching stop at $93,500. The stop has NOT been hit yet. What action?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.62,
        "reason": "SHORT stop at $93,500 has NOT been triggered \u2014 current price $93,100 is still $400 BELOW the stop. For a short, the stop only triggers when price rises ABOVE $93,500. Position is losing but the stop still protects. Crowded long funding at +0.022% provides potential squeeze fuel.",
        "reasoning": {
            "structure_analysis": "SHORT position stop check: stop at $93,500, price at $93,100. The stop is $400 above current price. For shorts, stop triggers when price goes ABOVE $93,500 \u2014 it has NOT done so. The $400 buffer is 0.29 ATR which is thin but the stop has not been reached.",
            "data_assessment": "CVD is positive and 1H trend is bullish, which is concerning for the short. However, crowded long funding at +0.022% historically precedes long liquidation events. Neutral 4H suggests the bullish move may be a 1H pullback in a larger range.",
            "risk_factors": "20x leverage makes this very sensitive. If stop triggers at $93,500, the loss is -0.6R * 20x = significant. But the stop exists for exactly this purpose \u2014 let it do its job.",
            "exit_logic": "Stop has NOT been breached. At -0.6R the position is losing but within the planned risk parameters. The elevated funding rate provides a potential catalyst for reversal. Hold with low confidence and let the hard stop manage the risk."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

# 1.5: SHORT - common confusion scenario, price dropped far below entry = winning
add(
    """POSITION STATE:
- Asset: SOL/USDT
- Direction: SHORT
- Entry: $180.00
- Current Price: $162.50
- P&L: +1.8R
- Stop Loss: $186.00
- TP1: $170.00 | TP2: $155.00
- Leverage: 5x
- Time in Trade: 36h

MARKET DATA:
- ATR(14): $6.50
- CVD: Negative
- Volume: Normal
- Trend: Bearish

DECISION REQUIRED: SHORT SOL at +1.8R past TP1. Stop at $186 is $23.50 above current price. How to manage?""",
    {
        "action": "TRAIL_STOP",
        "urgency": "LOW",
        "confidence": 0.78,
        "reason": "Short at +1.8R past TP1 with strong bearish trend. Trail stop down to $172 to lock in +0.8R profit minimum while letting the position run toward TP2 at $155.",
        "reasoning": {
            "structure_analysis": "SHORT position: stop at $186 is $23.50 ABOVE current price. For shorts, stop triggers when price rises above $186. Price is moving AWAY from stop (favorable). Bearish structure intact. Trail the stop lower (for a short, tighter stop means moving it DOWN closer to current price) to lock profit.",
            "data_assessment": "Negative CVD and bearish trend confirm continuation potential. Normal volume suggests the move is not yet exhausted.",
            "risk_factors": "At 5x and +1.8R, the main risk is giving back all profit on a reversal. Trailing the stop to $172 limits maximum giveback to 5.8% of entry or ~0.8R.",
            "exit_logic": "Past TP1 with bearish momentum = trail stop per MCF hierarchy. For a SHORT, tightening the stop means moving it DOWN. Trail to $172 ($9.50 above current price, ~1.5 ATR buffer). This locks in profit while allowing TP2 run."
        },
        "execution": {"exit_pct": 0, "stop_price": 172.00, "order_type": "STOP_MARKET"}
    }
)

# 1.6-1.10: More SHORT stop examples
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $78,000.00
- Current Price: $79,200.00
- P&L: -0.8R
- Stop Loss: $79,500.00
- Leverage: 10x
- Time in Trade: 4h

MARKET DATA:
- ATR(14): $1,300
- CVD: Positive
- Funding Rate: +0.015%
- Volume: Increasing
- Trend: Bullish 1H

DECISION REQUIRED: SHORT losing at -0.8R, price $79,200 approaching stop at $79,500. Only $300 buffer. What action?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.55,
        "reason": "SHORT stop at $79,500 has NOT been triggered. Price at $79,200 is $300 BELOW the stop. For a SHORT, the stop only triggers when price rises ABOVE $79,500. The buffer is thin (0.23 ATR) but the stop exists to manage this exact situation. Let the hard stop work.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: $79,500 stop, $79,200 price. Stop is $300 ABOVE price. For shorts, stop triggers on upward breach. It has NOT been breached. The $300 buffer is thin at 0.23 ATR, suggesting the stop may soon be tested.",
            "data_assessment": "Bullish CVD and increasing volume on 1H are concerning for the short. The position thesis is weakening.",
            "risk_factors": "10x leverage with only $300 buffer. If stop hits, loss is -0.8R * 10x = -8% of account. This is within planned risk.",
            "exit_logic": "The hard stop at $79,500 has NOT been breached. Low confidence HOLD \u2014 the stop will handle the exit if price continues rising. Do not pre-empt the stop."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,600.00
- Current Price: $3,720.00
- P&L: -1.5R
- Stop Loss: $3,680.00
- Leverage: 3x
- Time in Trade: 18h

MARKET DATA:
- ATR(14): $70
- CVD: Strongly positive
- Volume: High
- Trend: Bullish reversal confirmed

DECISION REQUIRED: SHORT ETH with price at $3,720 which is ABOVE stop loss at $3,680. Stop has been triggered. What to do?""",
    {
        "action": "EXIT_FULL",
        "urgency": "CRITICAL",
        "confidence": 0.95,
        "reason": "SHORT stop at $3,680 has been BREACHED. Current price $3,720 is $40 ABOVE the stop. For a SHORT position, price rising above the stop level means the hard stop is triggered. Exit 100% immediately.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: stop at $3,680, price at $3,720. For a SHORT, the stop triggers when price RISES ABOVE the stop. $3,720 > $3,680 = STOP BREACHED. This is non-negotiable under MCF Hard Stop rules.",
            "data_assessment": "Strongly positive CVD and bullish reversal confirm the short thesis is fully invalidated.",
            "risk_factors": "Already past stop. Every moment held increases loss. At 3x, the -1.5R loss is manageable but will get worse.",
            "exit_logic": "Hard Stop breach \u2192 EXIT 100% immediately. No discretion. The stop was set for risk management and it has been triggered."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $91,000.00
- Current Price: $88,500.00
- P&L: +1.4R
- Stop Loss: $92,200.00
- TP1: $89,000.00 | TP2: $86,000.00
- Leverage: 20x
- Time in Trade: 10h

MARKET DATA:
- ATR(14): $1,500
- CVD: Negative
- Volume: Normal
- Trend: Bearish

DECISION REQUIRED: SHORT BTC at +1.4R past TP1 with 20x leverage. Stop at $92,200 is far above. How to manage?""",
    {
        "action": "TP_PARTIAL",
        "urgency": "HIGH",
        "confidence": 0.82,
        "reason": "SHORT past TP1 at +1.4R with 20x leverage. Take 45% to lock significant profit. Stop at $92,200 is $3,700 ABOVE current price (2.5 ATR) \u2014 safe for shorts. Keep 55% running toward TP2 with tighter trail.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: stop at $92,200 is $3,700 ABOVE current price. For shorts, stop only triggers if price rises above $92,200. Price is $88,500 and falling AWAY from stop. Bearish structure intact.",
            "data_assessment": "Negative CVD and bearish trend support continuation. Normal volume is sustainable.",
            "risk_factors": "20x leverage means a +1.4R position is +28% account value. Giving back this profit on a reversal would be devastating. Must lock substantial portion.",
            "exit_logic": "Past TP1 at 20x \u2192 take 45% per MCF hierarchy. At 20x leverage, partial exits should be larger (40-50%) to protect against leveraged reversals. Trail stop on remaining to $90,000."
        },
        "execution": {"exit_pct": 45, "stop_price": 90000.00, "order_type": "MARKET"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $85,500.00
- Current Price: $84,900.00
- P&L: +0.4R
- Stop Loss: $86,200.00
- Leverage: 10x
- Time in Trade: 3h

MARKET DATA:
- ATR(14): $1,100
- CVD: Slightly negative
- Volume: Below average
- Trend: Neutral/choppy

DECISION REQUIRED: SHORT at +0.4R, early trade. Stop at $86,200 is above price. Low conviction environment. What action?""",
    {
        "action": "HOLD",
        "urgency": "LOW",
        "confidence": 0.68,
        "reason": "Short at +0.4R in early stages. Stop at $86,200 is $1,300 ABOVE current price \u2014 for a SHORT, this means the stop is safely far away (1.18 ATR buffer). No exit triggers active. Hold for development.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: stop at $86,200 is $1,300 ABOVE current price of $84,900. For shorts, stop triggers when price RISES above $86,200. Price is below and moving slightly favorable. 1.18 ATR buffer is adequate.",
            "data_assessment": "Slightly negative CVD supports the short weakly. Below-average volume and choppy trend suggest low conviction.",
            "risk_factors": "10x leverage but 1.18 ATR of stop buffer. Risk is defined and manageable.",
            "exit_logic": "No exit triggers active. Trade is young (3h) and slightly positive. Hold with moderate confidence in choppy conditions."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: SOL/USDT
- Direction: SHORT
- Entry: $195.00
- Current Price: $198.50
- P&L: -0.9R
- Stop Loss: $198.00
- Leverage: 5x
- Time in Trade: 14h

MARKET DATA:
- ATR(14): $5.80
- CVD: Positive
- Volume: Elevated
- Trend: Bullish 1H

DECISION REQUIRED: SHORT SOL with price at $198.50 which is ABOVE stop at $198.00. What action?""",
    {
        "action": "EXIT_FULL",
        "urgency": "CRITICAL",
        "confidence": 0.95,
        "reason": "SHORT stop at $198.00 has been BREACHED. Price at $198.50 is $0.50 ABOVE the stop. For a SHORT, price rising above the stop level = stop triggered. Exit immediately.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: stop at $198.00, current price $198.50. For a SHORT position, the stop triggers when price RISES ABOVE the stop level. $198.50 > $198.00 = BREACHED. Hard Stop rule applies.",
            "data_assessment": "Positive CVD and bullish trend confirm the short is invalidated.",
            "risk_factors": "5x leverage, already past stop. Loss is defined at -0.9R.",
            "exit_logic": "Hard Stop breach = EXIT 100%. Non-negotiable."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
)

# =============================================================================
# CATEGORY 2: TRAIL STOP LEVERAGE SCALING (6 examples)
# At high leverage with big R, trail must lock in profit aggressively
# =============================================================================

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $82,000.00
- Current Price: $85,200.00
- P&L: +2.1R
- Stop Loss: $80,500.00
- TP1: $84,000.00 | TP2: $87,000.00
- Leverage: 10x
- Time in Trade: 28h

MARKET DATA:
- ATR(14): $1,200
- CVD: Positive but weakening
- Volume: Declining
- Trend: Bullish but momentum fading

DECISION REQUIRED: LONG at +2.1R past TP1 with 10x leverage. Original stop is $4,700 below. How to trail?""",
    {
        "action": "TRAIL_STOP",
        "urgency": "HIGH",
        "confidence": 0.82,
        "reason": "At 10x leverage with +2.1R profit, MUST tighten trail aggressively. Move stop to $83,800 to lock in minimum +1.2R. At 10x, letting a +2R winner revert to a loser is unacceptable. Trail locks in 57% of unrealized profit while giving $1,400 (1.17 ATR) room for TP2 run.",
        "reasoning": {
            "structure_analysis": "Bullish structure intact but momentum fading. TP2 at $87,000 is $1,800 away. The key level to protect is the TP1 breakout at $84,000 \u2014 trailing to $83,800 holds just below this.",
            "data_assessment": "Weakening CVD and declining volume suggest the momentum is waning. TP2 may not be reached without a pullback first.",
            "risk_factors": "CRITICAL: At 10x leverage, unrealized profit is +21% of account. Original stop at $80,500 would give back all profit AND create a loss. This is the worst possible outcome \u2014 a big winner becoming a big loser. Trail MUST lock in at least 50% of R-multiple at leverage above 5x.",
            "exit_logic": "Leverage-adjusted trail: at 10x with +2.1R, trail to lock in minimum +1R (rule: lock 50%+ of R at leverage >5x). $83,800 = +1.2R locked. If price continues to TP2 at $87,000, great. If it reverses, exit at +1.2R instead of losing everything."
        },
        "execution": {"exit_pct": 0, "stop_price": 83800.00, "order_type": "STOP_MARKET"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $76,000.00
- Current Price: $79,500.00
- P&L: +2.5R
- Stop Loss: $74,600.00
- TP1: $78,000.00 | TP2: $81,000.00
- Leverage: 20x
- Time in Trade: 18h

MARKET DATA:
- ATR(14): $1,300
- CVD: Positive
- Volume: Healthy
- Trend: Strong bullish

DECISION REQUIRED: LONG at +2.5R past TP1 with 20x leverage. Massive unrealized profit. How to manage?""",
    {
        "action": "TP_PARTIAL",
        "urgency": "HIGH",
        "confidence": 0.85,
        "reason": "At 20x leverage with +2.5R, take 50% immediately and trail remaining aggressively. Unrealized profit is +50% of account value \u2014 cannot risk this on a single reversal candle. Take 50% at market, trail remaining with stop at $78,500 to lock in +1.8R minimum.",
        "reasoning": {
            "structure_analysis": "Strong bullish trend with TP2 at $81,000 still $1,500 away. Structure supports continuation but at 20x, you cannot afford to be wrong.",
            "data_assessment": "Healthy volume and positive CVD support continuation, but at 20x leverage the risk/reward of holding full size is asymmetric \u2014 a 2% pullback costs 40% of account value.",
            "risk_factors": "20x at +2.5R = +50% unrealized account profit. Original stop at $74,600 would result in -28% account loss. The swing from +50% to -28% is a 78% drawdown that would destroy a retail account. MUST protect.",
            "exit_logic": "At 20x with +2.5R: take 50% partial (MCF TP hierarchy). This locks +25% account profit in cash. Trail remaining 50% with stop at $78,500 (+1.8R locked on remaining). Worst case: exit remaining at +1.8R = total capture +2.15R average. Best case: remaining rides to TP2."
        },
        "execution": {"exit_pct": 50, "stop_price": 78500.00, "order_type": "MARKET"}
    }
)

add(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,800.00
- Current Price: $3,450.00
- P&L: +3.2R
- Stop Loss: $3,910.00
- Leverage: 10x
- Time in Trade: 32h

MARKET DATA:
- ATR(14): $90
- CVD: Negative but showing divergence
- Volume: Climax spike then declining
- Trend: Bearish but oversold

DECISION REQUIRED: SHORT ETH at +3.2R with 10x leverage. Volume climax pattern. CVD divergence forming. How to manage?""",
    {
        "action": "TP_PARTIAL",
        "urgency": "HIGH",
        "confidence": 0.85,
        "reason": "SHORT at +3.2R with 10x showing volume climax and CVD divergence \u2014 classic exhaustion signals. Take 50% to lock +16% account profit. Trail remaining with stop at $3,560 to lock in +2.2R minimum on remaining shares.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: stop at $3,910 is $460 ABOVE current price (5.1 ATR) \u2014 extremely safe. But the risk isn't the stop \u2014 it's giving back +3.2R of profit on a bounce.",
            "data_assessment": "Volume climax + CVD divergence = textbook exhaustion pattern. The selling may be near-complete. At 10x leverage, waiting for confirmation of reversal is too expensive.",
            "risk_factors": "10x at +3.2R = +32% account profit. A bounce from oversold conditions could erase this in minutes. Volume climax patterns typically precede 30-50% retracements of the move.",
            "exit_logic": "Leverage-adjusted exit: at 10x with +3.2R and exhaustion signals, take 50% immediately. Trail remaining at $3,560 (1.2 ATR above current, locks +2.2R). For SHORTS, trail stop means moving the stop DOWN closer to current price to lock profit."
        },
        "execution": {"exit_pct": 50, "stop_price": 3560.00, "order_type": "MARKET"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $88,000.00
- Current Price: $89,800.00
- P&L: +1.5R
- Stop Loss: $86,800.00
- Leverage: 1x
- Time in Trade: 48h

MARKET DATA:
- ATR(14): $1,100
- CVD: Positive
- Volume: Normal
- Trend: Bullish

DECISION REQUIRED: LONG at +1.5R with 1x leverage. How to trail?""",
    {
        "action": "TRAIL_STOP",
        "urgency": "LOW",
        "confidence": 0.75,
        "reason": "At 1x leverage with +1.5R, trail stop to $88,500 to lock in +0.4R minimum. At 1x leverage, a wider trail is acceptable since there's no liquidation risk. Give the position room to breathe with $1,300 buffer (1.18 ATR).",
        "reasoning": {
            "structure_analysis": "Bullish structure with steady uptrend. At 1x leverage, the position can tolerate wider drawdowns without liquidation risk.",
            "data_assessment": "Positive CVD and normal volume support trend continuation.",
            "risk_factors": "1x leverage means NO liquidation risk. A pullback to the original stop at $86,800 would only cost 1.4% of portfolio. The trail can be wider than at high leverage.",
            "exit_logic": "Leverage-adjusted trail: at 1x with +1.5R, trail to $88,500 (locks +0.4R, giving back ~73% of unrealized profit room). This is wider than high-leverage trails because at 1x, the cost of being stopped out early (missing continuation) exceeds the cost of giving back some profit. Compare: at 10x this trail would lock 50%+ of R-multiple."
        },
        "execution": {"exit_pct": 0, "stop_price": 88500.00, "order_type": "STOP_MARKET"}
    }
)

add(
    """POSITION STATE:
- Asset: SOL/USDT
- Direction: LONG
- Entry: $155.00
- Current Price: $168.00
- P&L: +1.9R
- Stop Loss: $148.00
- TP1: $165.00 | TP2: $180.00
- Leverage: 5x
- Time in Trade: 20h

MARKET DATA:
- ATR(14): $5.50
- CVD: Positive
- Volume: Above average
- Trend: Strong bullish, momentum accelerating

DECISION REQUIRED: LONG SOL past TP1 at +1.9R with 5x. Strong momentum. How to manage?""",
    {
        "action": "TRAIL_STOP",
        "urgency": "MEDIUM",
        "confidence": 0.80,
        "reason": "At 5x with +1.9R and accelerating momentum, trail stop to $163 to lock in +1.2R minimum. Momentum is strong so give room for TP2 run, but at 5x must protect at least 60% of R-multiple.",
        "reasoning": {
            "structure_analysis": "Strong bullish structure with momentum accelerating. TP2 at $180 is $12 away (2.2 ATR). High probability of continuation.",
            "data_assessment": "Positive CVD, above-average volume, and accelerating momentum all confirm the trend. This is a trend-following hold.",
            "risk_factors": "5x at +1.9R = +9.5% account. Original stop at $148 would give back all profit AND incur -3.5% loss. Trail protects against this.",
            "exit_logic": "Leverage-adjusted trail: at 5x with +1.9R, trail to $163 (locks +1.2R, protecting 63% of unrealized R). This gives $5 (0.9 ATR) of room for normal pullbacks while ensuring the position never goes from winner to loser."
        },
        "execution": {"exit_pct": 0, "stop_price": 163.00, "order_type": "STOP_MARKET"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $84,000.00
- Current Price: $87,500.00
- P&L: +2.8R
- Stop Loss: $82,800.00
- Leverage: 10x
- Time in Trade: 40h

MARKET DATA:
- ATR(14): $1,100
- CVD: Weakening
- Volume: Dropping fast
- Trend: Bullish but exhaustion signals

DECISION REQUIRED: LONG at +2.8R with 10x. Volume dropping and CVD weakening. How to protect?""",
    {
        "action": "TP_PARTIAL",
        "urgency": "HIGH",
        "confidence": 0.88,
        "reason": "At 10x with +2.8R and clear exhaustion signals (dropping volume, weakening CVD), take 50% immediately. Trail remaining with stop at $86,200 to lock +1.8R minimum. At 10x leverage, exhaustion signals are not optional warnings \u2014 they are exit triggers.",
        "reasoning": {
            "structure_analysis": "Bullish structure but showing fatigue. The move from $84k to $87.5k may be completing.",
            "data_assessment": "Volume dropping fast + weakening CVD = textbook exhaustion. At 1x leverage this would be a trail-and-watch. At 10x leverage, this is a take-profit signal.",
            "risk_factors": "10x at +2.8R = +28% account. If price drops 3% ($87.5k to $84.9k), the remaining position loses 30% of its value. Exhaustion signals make this retracement likely.",
            "exit_logic": "Leverage-adjusted exit: at 10x with exhaustion signals, scale is aggressive. Take 50% at market. Trail remaining at $86,200 (locks +1.8R on remaining, 64% of R protected). Rule: at leverage >5x, exhaustion signals + R>2.0 = mandatory partial take."
        },
        "execution": {"exit_pct": 50, "stop_price": 86200.00, "order_type": "MARKET"}
    }
)

# =============================================================================
# CATEGORY 3: NO-STOP LEVERAGE TIERS (8 examples)
# The rule should be leverage-dependent, not binary
# =============================================================================

# 3.1: 1x no stop, winning = HOLD (no liquidation risk)
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $90,000.00
- Current Price: $88,500.00
- P&L: +0.0R (no stop to calculate R)
- Stop Loss: NONE (no stop set -- HIGH RISK)
- Leverage: 1x
- Time in Trade: 18h

MARKET DATA:
- ATR(14): $1,300
- CVD: Negative
- Funding Rate: +0.020% (crowded longs)
- Volume: Normal
- Trend: Bearish

DECISION REQUIRED: SHORT BTC at 1x with no stop. Position is winning (+1.7% unrealized). What action?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.72,
        "reason": "At 1x leverage, no stop loss is NOT a critical risk \u2014 there is zero liquidation danger. The position is winning with bearish momentum. Recommend setting a stop at $91,500 but do not exit a winning 1x position solely because no stop is set.",
        "reasoning": {
            "structure_analysis": "Position is profitable at 1x leverage. No stop means no defined risk, but at 1x there is no liquidation mechanism. The worst case is the position goes to zero minus entry, but for a short on BTC at $90k this is unrealistic in the near term.",
            "data_assessment": "Bearish CVD, crowded long funding, and bearish trend all support the short. The trade is working.",
            "risk_factors": "At 1x leverage: no liquidation risk, no margin call risk. The only risk is opportunity cost and drawdown. This is fundamentally different from 10x+ where no stop means potential account destruction.",
            "exit_logic": "No-stop risk assessment by leverage: 1x = low risk (no liquidation), recommend setting stop but DO NOT force-exit. 3-5x = medium risk (suggest stop strongly). 10x+ = high risk (EXIT if no stop). At 1x, the correct action is HOLD the winning trade and recommend the user sets a stop at recent structure."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

# 3.2: 1x no stop, losing = still HOLD with warning
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $85,000.00
- Current Price: $83,800.00
- P&L: +0.0R (no stop)
- Stop Loss: NONE (no stop set -- HIGH RISK)
- Leverage: 1x
- Time in Trade: 30h

MARKET DATA:
- ATR(14): $1,200
- CVD: Negative
- Volume: Below average
- Trend: Bearish

DECISION REQUIRED: LONG BTC at 1x no stop. Position is losing -1.4%. Bearish conditions. What action?""",
    {
        "action": "REDUCE_SIZE",
        "urgency": "MEDIUM",
        "confidence": 0.70,
        "reason": "At 1x leverage, no stop is NOT an emergency \u2014 no liquidation risk exists. However, the position is losing -1.4% with bearish conditions. Reduce 30% to manage risk while keeping 70% for potential recovery. Strongly recommend setting a stop loss.",
        "reasoning": {
            "structure_analysis": "Position is -1.4% at 1x. Bearish conditions suggest continued downside. However, at 1x there is no forced liquidation, so the trader has time to manage.",
            "data_assessment": "Bearish CVD and trend suggest more downside, but below-average volume may mean the selling is not aggressive.",
            "risk_factors": "1x leverage = zero liquidation risk. A -1.4% loss at 1x is painful but not account-threatening. Compare to 10x where this would be -14% and approaching liquidation.",
            "exit_logic": "No-stop at 1x with losing position: REDUCE 30% to manage risk, but do NOT panic-exit. At 1x, the trader can afford to hold through drawdowns. Recommend setting stop at $82,500 (recent structure) on remaining 70%."
        },
        "execution": {"exit_pct": 30, "stop_price": None, "order_type": "MARKET"}
    }
)

# 3.3: 3x no stop, winning = HOLD with strong recommendation
add(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: LONG
- Entry: $3,400.00
- Current Price: $3,520.00
- P&L: +0.0R (no stop)
- Stop Loss: NONE (no stop set -- HIGH RISK)
- Leverage: 3x
- Time in Trade: 8h

MARKET DATA:
- ATR(14): $75
- CVD: Positive
- Volume: Above average
- Trend: Bullish

DECISION REQUIRED: LONG ETH 3x no stop. Winning +3.5%. Bullish conditions. What action?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.68,
        "reason": "At 3x leverage with a winning position and bullish momentum, do not exit solely for missing stop. At 3x, liquidation requires a ~33% move which is unlikely in the short term. HOLD but strongly recommend setting stop at $3,380 (1.87 ATR below current price).",
        "reasoning": {
            "structure_analysis": "Position is profitable at 3x. Bullish structure with higher highs. No stop is a risk management gap but not an emergency at 3x.",
            "data_assessment": "Positive CVD, above-average volume, and bullish trend all confirm continuation. Exiting a winning trade in this environment purely for no-stop would be leaving money on the table.",
            "risk_factors": "3x leverage: liquidation at ~-33% move. Current unrealized profit is +3.5%. The position has buffer. Main risk is a flash crash with no stop \u2014 unlikely but possible. STRONGLY recommend stop at $3,380.",
            "exit_logic": "No-stop at 3x winning: HOLD. The no-stop rule should be leverage-tiered: 1x = no exit needed. 3x = hold but recommend stop. 5x = hold if winning, exit if losing. 10x+ = exit immediately regardless. At 3x, force-exiting a winning trade is worse than the risk of holding without a stop."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

# 3.4: 10x no stop, WINNING = REDUCE_SIZE (not full exit)
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: SHORT
- Entry: $78,000.00
- Current Price: $76,800.00
- P&L: +0.0R (no stop)
- Stop Loss: NONE (no stop set -- HIGH RISK)
- Leverage: 10x
- Time in Trade: 6h

MARKET DATA:
- ATR(14): $1,200
- CVD: Negative
- Funding Rate: +0.015% (longs crowded)
- Volume: Elevated
- Trend: Bearish

DECISION REQUIRED: SHORT BTC 10x no stop. Winning -1.5% unrealized. Bearish momentum. What action?""",
    {
        "action": "REDUCE_SIZE",
        "urgency": "HIGH",
        "confidence": 0.80,
        "reason": "At 10x leverage with no stop, the risk is genuinely high \u2014 a 10% adverse move would liquidate. However, the position is WINNING with bearish momentum. Reduce 50% to cut exposure in half, and set a stop at $79,200 on the remaining 50%. Do not fully exit a winning trade at 10x \u2014 reduce and protect instead.",
        "reasoning": {
            "structure_analysis": "Position is profitable at 10x short. Bearish structure with crowded long funding supports continuation. No stop is a critical gap at 10x but the trade is working.",
            "data_assessment": "Negative CVD, crowded longs, elevated volume, bearish trend \u2014 all favor the short. Fully exiting would miss potential continuation.",
            "risk_factors": "10x with no stop: liquidation at ~+10% adverse move ($85,800). This IS dangerous. But the position has +1.5% unrealized profit as buffer. Reducing 50% cuts the risk to effective 5x while keeping exposure to the working thesis.",
            "exit_logic": "No-stop at 10x WINNING: REDUCE 50% (not EXIT 100%). The binary 'no stop = exit everything' rule misses nuance. At 10x winning: 1) Reduce size by 50% to cut leverage risk 2) Set stop on remaining at $79,200 (1 ATR above current) 3) Let the now-protected position continue. This balances risk management with not leaving money on the table."
        },
        "execution": {"exit_pct": 50, "stop_price": 79200.00, "order_type": "MARKET"}
    }
)

# 3.5: 10x no stop, LOSING = EXIT
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $88,000.00
- Current Price: $86,500.00
- P&L: +0.0R (no stop)
- Stop Loss: NONE (no stop set -- HIGH RISK)
- Leverage: 10x
- Time in Trade: 15h

MARKET DATA:
- ATR(14): $1,300
- CVD: Negative
- Volume: Increasing on red candles
- Trend: Bearish

DECISION REQUIRED: LONG BTC 10x no stop. LOSING -1.7%. Bearish conditions. What action?""",
    {
        "action": "EXIT_100_PERCENT_IMMEDIATELY",
        "urgency": "CRITICAL",
        "confidence": 0.95,
        "reason": "10x leverage with no stop loss AND the position is losing -1.7% with bearish momentum. At 10x, a further -8.3% move would mean liquidation. With no stop and the trade going against you, exit 100% immediately.",
        "reasoning": {
            "structure_analysis": "Position is -1.7% at 10x with bearish structure. No defined risk level means no safety net if the move accelerates.",
            "data_assessment": "Negative CVD, increasing volume on red candles, bearish trend \u2014 the conditions favor continuation lower. No reason to expect a reversal.",
            "risk_factors": "10x + no stop + losing + bearish = maximum danger. Liquidation is at approximately -10% from entry ($79,200). Currently at -1.7%, a further -8.3% move liquidates the account. There is no buffer, no safety net, and momentum is against the position.",
            "exit_logic": "No-stop at 10x LOSING in bearish conditions: EXIT 100% immediately. When the position is losing at high leverage with no stop, the binary exit rule IS appropriate. The nuance is: at high leverage, no-stop is only acceptable if the trade is winning. Once it's losing, immediate exit is mandatory."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
)

# 3.6: 20x no stop = always exit regardless
add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $91,000.00
- Current Price: $91,800.00
- P&L: +0.0R (no stop)
- Stop Loss: NONE (no stop set -- HIGH RISK)
- Leverage: 20x
- Time in Trade: 2h

MARKET DATA:
- ATR(14): $1,200
- CVD: Positive
- Volume: Normal
- Trend: Bullish

DECISION REQUIRED: LONG BTC 20x no stop. Winning +0.9%. What action?""",
    {
        "action": "EXIT_100_PERCENT_IMMEDIATELY",
        "urgency": "CRITICAL",
        "confidence": 0.92,
        "reason": "At 20x leverage with no stop, liquidation is only ~5% away. Even though the position is winning, a single adverse 1H candle could liquidate the entire position. At 20x, no stop is ALWAYS an immediate exit regardless of P&L.",
        "reasoning": {
            "structure_analysis": "Position is winning at 20x but has zero downside protection. Any structural break would be catastrophic before the trader can react.",
            "data_assessment": "Bullish conditions are irrelevant at 20x with no stop. The risk of a single adverse candle outweighs all bullish signals.",
            "risk_factors": "20x leverage + no stop: liquidation at -5% from entry ($86,450). A normal 1H BTC candle can move 1-2%. A news event can move 5-10%. There is no scenario where holding 20x without a stop is acceptable.",
            "exit_logic": "No-stop at 20x: EXIT 100% regardless of P&L or momentum. At 20x, the leverage is so high that even a winning position without a stop is existential risk. This is the one leverage tier where the binary 'no stop = exit' rule is always correct."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
)

# 3.7: 5x no stop, winning = HOLD with strong warning
add(
    """POSITION STATE:
- Asset: SOL/USDT
- Direction: LONG
- Entry: $160.00
- Current Price: $165.50
- P&L: +0.0R (no stop)
- Stop Loss: NONE (no stop set -- HIGH RISK)
- Leverage: 5x
- Time in Trade: 12h

MARKET DATA:
- ATR(14): $5.20
- CVD: Positive
- Volume: Above average
- Trend: Bullish

DECISION REQUIRED: LONG SOL 5x no stop. Winning +3.4%. Bullish. What action?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.65,
        "reason": "At 5x with no stop the risk is elevated but manageable. Position is winning +3.4% with bullish momentum. Liquidation is ~20% away. HOLD but IMMEDIATELY set a stop at $158 (structural support). Do not exit a winning 5x trade purely for missing stop \u2014 ADD the stop instead.",
        "reasoning": {
            "structure_analysis": "Winning at 5x with bullish structure. Liquidation at ~-20% ($128) is far away. The immediate risk is a pullback, not liquidation.",
            "data_assessment": "All signals bullish. Exiting would leave 5x worth of money on the table in a working trend.",
            "risk_factors": "5x no stop: moderate danger. Liquidation at -20% is unlikely in the short term but a flash crash could reach it. The solution is to SET A STOP, not to exit the trade.",
            "exit_logic": "No-stop at 5x winning: HOLD with strong recommendation to set stop at $158. The risk-adjusted response at 5x is to PROTECT the position, not abandon it. At 1-5x winning with momentum, the no-stop response should be 'set a stop' not 'exit immediately.'"
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

# 3.8: 5x no stop, losing = EXIT
add(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,500.00
- Current Price: $3,580.00
- P&L: +0.0R (no stop)
- Stop Loss: NONE (no stop set -- HIGH RISK)
- Leverage: 5x
- Time in Trade: 20h

MARKET DATA:
- ATR(14): $80
- CVD: Positive
- Volume: Increasing
- Trend: Bullish reversal

DECISION REQUIRED: SHORT ETH 5x no stop. Losing -2.3%. Bullish reversal happening. What action?""",
    {
        "action": "EXIT_FULL",
        "urgency": "CRITICAL",
        "confidence": 0.90,
        "reason": "SHORT at 5x with no stop, losing -2.3%, and bullish reversal in progress. At 5x losing with no downside protection and adverse momentum, exit fully. The threshold: at 5x, no-stop + losing + adverse momentum = EXIT.",
        "reasoning": {
            "structure_analysis": "Short is losing with bullish reversal forming. No stop means unlimited loss potential on a short squeeze.",
            "data_assessment": "Positive CVD, increasing volume, bullish reversal \u2014 all confirm the short thesis is failing.",
            "risk_factors": "5x short with no stop in a bullish reversal. Shorts have theoretically unlimited loss. At 5x, a 20% adverse move = -100% (liquidation). With reversal momentum, this is real danger.",
            "exit_logic": "No-stop at 5x LOSING in adverse momentum: EXIT FULL. The leverage-tiered no-stop rule: at 5x, exit if BOTH conditions met: (1) position is losing AND (2) momentum is adverse. If either condition is absent (winning, or neutral momentum), the response is REDUCE + SET STOP instead."
        },
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
    }
)

# =============================================================================
# CATEGORY 4: HOLD CONFIDENCE CALIBRATION NEAR STOPS (6 examples)
# When buffer to stop < 1%, confidence should be LOW (55-65%)
# =============================================================================

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $87,000.00
- Current Price: $85,300.00
- P&L: -0.9R
- Stop Loss: $85,100.00
- Leverage: 5x
- Time in Trade: 20h

MARKET DATA:
- ATR(14): $1,200
- CVD: Slightly positive
- Funding Rate: -0.005%
- Volume: Normal
- Trend: Neutral

DECISION REQUIRED: LONG at -0.9R with only $200 buffer to stop (0.23%). CVD slightly positive. What action?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.55,
        "reason": "Stop at $85,100 has NOT been triggered. $200 buffer (0.17 ATR) is razor-thin. LOW confidence HOLD \u2014 the stop exists to manage this risk. Let it do its job, but acknowledge this is essentially a coin flip.",
        "reasoning": {
            "structure_analysis": "Price is $200 above stop (0.23% buffer). This is less than a single 15-minute candle of normal BTC volatility. The stop is very likely to be tested in the next few hours.",
            "data_assessment": "Slightly positive CVD is a weak signal. It does NOT override the fact that the position is 0.23% from stop. Never trust secondary signals over price proximity to hard stop.",
            "risk_factors": "Buffer-to-stop < 0.5% = maximum uncertainty. The model should NOT express high confidence when the position is this close to its defined exit level. A 55% confidence HOLD acknowledges that the outcome is nearly random at this proximity.",
            "exit_logic": "With buffer < 0.5% to stop: HOLD at LOW confidence (55%). The hard stop exists for exactly this situation. Pre-empting the stop is second-guessing the user's risk management. But confidence MUST be low \u2014 expressing 75%+ confidence when the position is 0.23% from stop is miscalibrated."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $92,000.00
- Current Price: $91,200.00
- P&L: -0.5R
- Stop Loss: $90,800.00
- Leverage: 3x
- Time in Trade: 15h

MARKET DATA:
- ATR(14): $1,400
- CVD: Positive
- Funding Rate: +0.012%
- OI Change: +3.2%
- Volume: Elevated
- Trend: Bullish 1H

DECISION REQUIRED: LONG at -0.5R with $400 buffer to stop (0.44%). CVD positive, OI increasing. What action?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.60,
        "reason": "Stop at $90,800 not triggered. $400 buffer (0.29 ATR) is thin. Hold at 60% confidence. While CVD and OI are positive, these secondary signals do NOT change the fact that the position is 0.44% from its hard stop. Moderate confidence reflects the genuine uncertainty.",
        "reasoning": {
            "structure_analysis": "$400 buffer to stop = 0.44% of price = 0.29 ATR. This is within normal 1H candle range. The stop will likely be tested. Structure assessment is irrelevant when price is this close to the defined risk level.",
            "data_assessment": "Positive CVD and increasing OI are encouraging but they are SECONDARY signals. At 0.44% from stop, these signals provide marginal edge at best. The position outcome depends almost entirely on the next 2-3 candles.",
            "risk_factors": "The temptation is to assign high confidence based on bullish CVD/OI. This is a calibration trap. When buffer-to-stop is < 0.5%, the confidence CEILING should be ~65% regardless of secondary signals. The hard stop proximity dominates all other factors.",
            "exit_logic": "Hold at 60% confidence. Let the hard stop manage the risk. Do not pre-empt the stop AND do not express overconfidence in the hold. The correct posture is humility about the outcome."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,700.00
- Current Price: $3,790.00
- P&L: -0.8R
- Stop Loss: $3,820.00
- Leverage: 10x
- Time in Trade: 10h

MARKET DATA:
- ATR(14): $75
- CVD: Positive
- Volume: High
- Trend: Bullish

DECISION REQUIRED: SHORT at -0.8R with price at $3,790 approaching stop at $3,820. Buffer $30 (0.79%). CVD positive. What action?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.55,
        "reason": "SHORT stop at $3,820 has NOT been triggered. Price at $3,790 is $30 BELOW the stop (0.4 ATR buffer). For a SHORT, stop triggers when price RISES ABOVE $3,820. Low confidence HOLD. Positive CVD and bullish trend are concerning for the short, but the hard stop will handle the exit if needed.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: stop at $3,820 is $30 ABOVE price. For shorts, stop triggers on upward breach. Not yet breached. Buffer is 0.4 ATR \u2014 thin but the stop exists for this purpose.",
            "data_assessment": "Positive CVD and bullish trend are adverse for the short. These signals increase the probability of the stop being hit. But they don't GUARANTEE it \u2014 hence low confidence, not an exit.",
            "risk_factors": "10x with thin buffer. If stop hits: -0.8R * 10x = managed loss. The stop will protect. Pre-empting the stop at 10x can be correct but only when the buffer is essentially zero.",
            "exit_logic": "Near-stop SHORT hold at 55% confidence. The calibration rule: when buffer-to-stop is < 1% AND secondary signals are adverse, confidence should be 50-60%. Never express 75%+ confidence in this situation."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $78,500.00
- Current Price: $77,900.00
- P&L: -0.4R
- Stop Loss: $77,400.00
- Leverage: 20x
- Time in Trade: 8h

MARKET DATA:
- ATR(14): $1,100
- CVD: Slightly negative
- Volume: Below average
- Trend: Neutral

DECISION REQUIRED: LONG at -0.4R with $500 buffer to stop at 20x leverage. Neutral conditions. What action?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.58,
        "reason": "Stop at $77,400 not triggered. $500 buffer (0.45 ATR). At 20x leverage with 0.64% buffer, this is HIGH urgency but the stop exists to manage the risk. HOLD at 58% confidence. Neutral conditions make this a genuine coin flip.",
        "reasoning": {
            "structure_analysis": "$500 buffer to stop at 20x. If stop hits: loss is defined and manageable per risk plan. The buffer is thin but functional.",
            "data_assessment": "Slightly negative CVD and neutral trend provide no directional edge. Below-average volume suggests no aggressive selling.",
            "risk_factors": "20x with 0.64% buffer is uncomfortable but the stop AT $77,400 was placed intentionally. The model's job is to respect the risk management framework, not to override it with discretionary exits.",
            "exit_logic": "HOLD at 58% confidence. Key insight: the confidence should reflect the PROBABILITY of the hold being correct, not the model's emotional comfort level. With 0.64% buffer and neutral conditions, ~55-60% confidence is honest. Anything higher would be miscalibrated."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $84,000.00
- Current Price: $83,600.00
- P&L: -0.3R
- Stop Loss: $83,200.00
- Leverage: 5x
- Time in Trade: 12h

MARKET DATA:
- ATR(14): $1,100
- CVD: Positive
- OI: +5% (new money entering)
- Funding: -0.008% (shorts crowded)
- Volume: Elevated
- Trend: Bullish, higher lows holding

DECISION REQUIRED: LONG at -0.3R with $400 buffer (0.48%). ALL secondary signals are bullish (CVD positive, shorts crowded, OI increasing, higher lows). Should confidence be high?""",
    {
        "action": "HOLD",
        "urgency": "MEDIUM",
        "confidence": 0.65,
        "reason": "Stop at $83,200 not triggered. $400 buffer (0.36 ATR). Even though ALL secondary signals are bullish, the buffer-to-stop is < 0.5%. Confidence CAPPED at 65%. Secondary signals provide edge but cannot overcome the fundamental uncertainty of being 0.48% from the hard stop.",
        "reasoning": {
            "structure_analysis": "Higher lows holding and bullish structure are genuine positives. But price is $400 above stop \u2014 one normal 1H candle could sweep the stop before any bullish thesis plays out.",
            "data_assessment": "This is the calibration trap scenario: every single secondary signal (CVD, OI, funding, trend) is bullish. The natural instinct is to assign 85%+ confidence. THIS IS WRONG. When buffer-to-stop is < 0.5%, secondary signals cannot raise confidence above ~65% because the stop proximity dominates the outcome distribution.",
            "risk_factors": "The risk is not that the signals are wrong \u2014 it's that the stop might trigger BEFORE the signals play out. A wick below $83,200 would stop the position out even if price immediately bounces back to $84,500. This wick risk is the dominant factor at 0.48% buffer.",
            "exit_logic": "HOLD at 65% (capped). Rule: buffer < 0.5% = confidence ceiling 65% regardless of signal strength. Buffer < 0.3% = ceiling 58%. Buffer < 0.15% = ceiling 52%. This calibration prevents the model from being 85% confident on coin-flip outcomes."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $70,000.00
- Current Price: $69,400.00
- P&L: -0.5R
- Stop Loss: $69,200.00
- Leverage: 3x
- Time in Trade: 25h

MARKET DATA:
- ATR(14): $1,000
- CVD: Negative
- Volume: Increasing
- Trend: Bearish

DECISION REQUIRED: LONG at -0.5R with $200 buffer (0.29%). CVD negative, bearish trend, increasing volume. All signals bearish. What action?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.52,
        "reason": "Stop at $69,200 not triggered. $200 buffer (0.2 ATR). All signals are bearish which makes the stop hit likely. HOLD at 52% confidence \u2014 this is barely better than random. The hard stop will manage the exit. Pre-empting it would second-guess the risk plan.",
        "reasoning": {
            "structure_analysis": "$200 buffer is 0.29% of price \u2014 essentially at the stop. The position is extremely likely to stop out in the next 1-2 candles given bearish momentum.",
            "data_assessment": "Negative CVD, increasing volume, bearish trend \u2014 all point to the stop being hit. But 'likely to stop out' does not mean 'already stopped out.' The model must distinguish between these states.",
            "risk_factors": "At 3x with $200 buffer, the stop hit would cost a defined -0.5R at 3x = manageable loss. The risk of pre-empting the stop is that price bounces from the level and the model caused an unnecessary exit.",
            "exit_logic": "HOLD at 52% confidence. When buffer < 0.3% AND all signals are adverse: confidence should be 50-55% (essentially acknowledging the outcome is near-random). The model MUST NOT pre-empt the hard stop regardless of signal alignment. The stop exists for this purpose."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

# =============================================================================
# CATEGORY 5: "LET THE STOP WORK" - PANIC NEAR STOP (5 examples)
# =============================================================================

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $75,000.00
- Current Price: $73,650.00
- P&L: -0.9R
- Stop Loss: $73,500.00
- Leverage: 5x
- Time in Trade: 16h

MARKET DATA:
- ATR(14): $1,100
- CVD: Negative
- Volume: Below average
- Trend: Bearish

DECISION REQUIRED: LONG at -0.9R, price $150 above stop. Bearish conditions. Should I exit before stop triggers?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.53,
        "reason": "Stop at $73,500 has NOT been triggered. Price is $150 above stop (0.14 ATR). Let the hard stop work \u2014 it was set for this exact scenario. Pre-empting the stop by $150 provides zero meaningful benefit but risks missing a bounce from the level.",
        "reasoning": {
            "structure_analysis": "Price is $150 above stop. This is a hair's breadth from the stop level. The stop will almost certainly be tested in the next 1-2 candles. But 'almost certainly tested' is not the same as 'triggered.' Stop levels often act as magnets AND support \u2014 price can wick to the stop and bounce.",
            "data_assessment": "Bearish CVD and trend make the stop hit probable. But the $150 difference between current price and stop is the trader's planned risk. Exiting $150 early saves essentially nothing (0.1% at 5x = 0.5% account).",
            "risk_factors": "The cost of pre-empting the stop: if price bounces from $73,500 and rallies, the model caused an unnecessary exit. The cost of NOT pre-empting: price hits $73,500 stop and the trader loses the planned amount. The planned loss is already accepted when the stop was set.",
            "exit_logic": "DO NOT pre-empt the hard stop. The stop at $73,500 represents the trader's defined risk. Exiting at $73,650 instead of $73,500 saves $150 (0.1% of position) but risks missing a bounce from the support level that the stop was placed at. The risk/reward of pre-emption is terrible. Let the stop do its job."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: SOL/USDT
- Direction: LONG
- Entry: $175.00
- Current Price: $170.80
- P&L: -0.8R
- Stop Loss: $170.50
- Leverage: 10x
- Time in Trade: 8h

MARKET DATA:
- ATR(14): $5.00
- CVD: Mixed
- Volume: Normal
- Trend: Neutral, consolidating

DECISION REQUIRED: LONG SOL at -0.8R, price $0.30 above stop. Only 0.18% buffer. Should I exit now to avoid stop?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.52,
        "reason": "Stop at $170.50 not triggered. $0.30 buffer (0.06 ATR). DO NOT pre-empt the stop. Exiting $0.30 early saves essentially nothing at any leverage. The stop exists for this exact moment. Consolidation range suggests price could bounce from this zone.",
        "reasoning": {
            "structure_analysis": "Price $170.80 is $0.30 above stop $170.50. This is 0.18% of price \u2014 essentially at the stop. But the stop has NOT been hit.",
            "data_assessment": "Mixed CVD and consolidation pattern. Not clearly bearish. The stop level may coincide with a support zone.",
            "risk_factors": "Saving $0.30 by exiting early: at 10x = $3.00 per SOL leverage-adjusted. Risking: missing a bounce from $170.50 support that could carry back to $175+. The asymmetry favors letting the stop work.",
            "exit_logic": "Let the hard stop work. The stop was placed at $170.50 for a reason \u2014 likely a structure level. If the model exits at $170.80 to 'save' $0.30, it overrides the trader's risk management for no meaningful benefit. The ONLY time to pre-empt a stop is if there is new information that wasn't available when the stop was set (e.g., major news event, exchange hack). Bearish CVD is not new information."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $68,000.00
- Current Price: $66,800.00
- P&L: -0.85R
- Stop Loss: $66,600.00
- Leverage: 3x
- Time in Trade: 20h

MARKET DATA:
- ATR(14): $1,050
- CVD: Flattening after selling
- Volume: Declining from spike
- Trend: Bearish but wicks showing buyer interest at lows

DECISION REQUIRED: LONG at -0.85R, price $200 above stop. Selling pressure declining. What action?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.58,
        "reason": "Stop at $66,600 not triggered. $200 buffer (0.19 ATR). CVD flattening and declining volume suggest selling pressure is waning. Let the hard stop work \u2014 longer wicks at lows indicate buyer interest at this zone, which is exactly where the stop was placed.",
        "reasoning": {
            "structure_analysis": "Price $200 above stop with wicks showing buyer interest. The stop at $66,600 may have been placed AT a support level. Exiting before the level is tested would be leaving before the most important structural test occurs.",
            "data_assessment": "CVD flattening and volume declining from a spike = selling exhaustion pattern. This is EXACTLY the situation where panic-exiting before the stop misses a reversal.",
            "risk_factors": "At 3x, the planned stop loss is manageable. The real risk is pre-empting the stop and missing the bounce that the declining volume suggests.",
            "exit_logic": "HOLD and let the stop work. This is the classic panic-near-stop scenario where the model must NOT exit. The declining volume and longer wicks are early signs of a potential reversal AT the stop level. If the model exits at $66,800, it misses the test of $66,600 which could be the low."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: ETH/USDT
- Direction: SHORT
- Entry: $3,500.00
- Current Price: $3,590.00
- P&L: -0.7R
- Stop Loss: $3,610.00
- Leverage: 5x
- Time in Trade: 6h

MARKET DATA:
- ATR(14): $70
- CVD: Positive
- Volume: Increasing
- Trend: Bullish 1H

DECISION REQUIRED: SHORT ETH at -0.7R, price $3,590, stop at $3,610. Only $20 below stop for a short (0.56% buffer). Bullish momentum. Pre-empt the stop?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.53,
        "reason": "SHORT stop at $3,610 NOT triggered. Price at $3,590 is $20 BELOW the stop. For a SHORT, stop triggers when price RISES ABOVE $3,610. The $20 buffer (0.29 ATR) is thin. Let the hard stop manage the exit. Do not pre-empt.",
        "reasoning": {
            "structure_analysis": "SHORT stop check: price $3,590 is BELOW stop $3,610. Stop NOT triggered. Buffer is $20 (0.56% of price). Thin but functional.",
            "data_assessment": "Positive CVD and bullish 1H are adverse for the short, making stop hit probable. But probable is not certain.",
            "risk_factors": "Pre-empting saves $20 per ETH at 5x = $100 per ETH leverage-adjusted. This is marginal. The cost of pre-empting is missing a potential rejection from the $3,610 resistance (which is where the stop was placed, likely for structural reasons).",
            "exit_logic": "Let the stop work. The stop at $3,610 was set at a structural level. If the model exits at $3,590 to 'save' $20, it removes the chance of price testing and rejecting from $3,610 resistance. The asymmetry favors holding."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)

add(
    """POSITION STATE:
- Asset: BTC/USDT
- Direction: LONG
- Entry: $82,000.00
- Current Price: $80,500.00
- P&L: -1.0R
- Stop Loss: $80,200.00
- Leverage: 10x
- Time in Trade: 22h

MARKET DATA:
- ATR(14): $1,300
- CVD: Negative
- Volume: Elevated
- Trend: Bearish
- Key Level: $80,000-$80,200 is a major support zone (daily level)

DECISION REQUIRED: LONG at -1.0R with $300 above stop at 10x. Bearish but sitting right on major support. Pre-empt?""",
    {
        "action": "HOLD",
        "urgency": "HIGH",
        "confidence": 0.57,
        "reason": "Stop at $80,200 not triggered. $300 buffer (0.23 ATR). The stop was placed AT the $80,000-$80,200 major daily support zone. This is EXACTLY where you want the stop \u2014 if this level breaks, the trade is invalidated. Let the support level and the stop work together.",
        "reasoning": {
            "structure_analysis": "The stop at $80,200 coincides with major daily support at $80,000-$80,200. This is ideal stop placement \u2014 below structure. Pre-empting the stop would mean exiting BEFORE the most important support test.",
            "data_assessment": "Bearish CVD and elevated volume are concerning but major support zones are where reversals happen. The selling may be absorption by large buyers at this level.",
            "risk_factors": "10x with $300 buffer is uncomfortable. But the stop at $80,200 represents a structural decision: if $80k support breaks, the trade is over. If it holds, this is the optimal long entry zone. Pre-empting removes the chance of catching the bounce.",
            "exit_logic": "HOLD and let the structural test play out. The stop at $80,200 is a STRUCTURAL stop, not an arbitrary level. Exiting at $80,500 removes the opportunity to benefit from the most important test \u2014 whether $80k support holds. If it doesn't hold, the stop triggers. If it does hold, this is the reversal point. The model must trust structural stops."
        },
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "NONE"}
    }
)


# =============================================================================
# Save all examples
# =============================================================================

output_file = "bastion_risk_v4_reinforcement.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Generated {len(examples)} v4 training examples")
print(f"Saved to {output_file}")
print(f"\nBreakdown:")
print(f"  Cat 1 (SHORT stop logic): 10 examples")
print(f"  Cat 2 (Trail stop leverage): 6 examples")
print(f"  Cat 3 (No-stop leverage tiers): 8 examples")
print(f"  Cat 4 (HOLD confidence near stop): 6 examples")
print(f"  Cat 5 (Let the stop work): 5 examples")
print(f"  TOTAL: {len(examples)} examples")
