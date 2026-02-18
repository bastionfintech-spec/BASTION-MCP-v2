"""
BASTION v5 Training Data Generator
===================================
Fixes v4 regression by addressing root causes:

1. LONG stop direction logic (10 examples) — v4 only trained SHORT, causing LONG confusion
2. SHORT stop direction logic (10 examples) — keep v4's good SHORT training
3. TP1 proximity + leverage → TP_PARTIAL (8 examples) — v4 "let the stop work" overcorrected
4. No-stop leverage tiers (20 examples) — v4 had only 8, couldn't override 200+ binary examples
5. Funding alone ≠ exit signal (6 examples) — model panic-exits winners on funding
6. Stop already breached = EXIT always (6 examples) — reinforce correct post-breach behavior

Total: 60 new examples (combined with v3's 230 = 290 total)

CRITICAL: Train on v3 base model (not v4) since v4 corrupted LONG logic
"""

import json
import random

examples = []

SYSTEM_PROMPT = (
    "You are BASTION-32B, an AI risk management engine for cryptocurrency trading. "
    "Evaluate the position using MCF ULTRA exit hierarchy: "
    "Hard Stop → Safety Net Break → Guarding Line Break → Take Profit → Trailing Stop → Time Exit. "
    "Respond with a JSON object containing: action, urgency, confidence, reason, reasoning "
    "(with structure_analysis, data_assessment, risk_factors, exit_logic), and execution "
    "(exit_pct, stop_price, order_type)."
)


def make_example(user_content, assistant_content):
    """Create a training example in ChatML format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_content)}
        ]
    }


# ============================================================
# CATEGORY 1: LONG Stop Direction Logic (10 examples)
# For LONG: stop is BELOW entry. Breached when price FALLS BELOW stop.
# Price ABOVE stop = safe. Price BELOW stop = triggered.
# ============================================================

long_stop_examples = [
    # LONG, price above stop, safe → HOLD
    {
        "user": "Position: LONG BTC | Entry: $85,000 | Current: $84,200 | Stop: $83,500 | Leverage: 10x | R-Multiple: -0.53 | Duration: 2h | Market: BTC slightly pulling back, support at $83,800. CVD neutral. OI flat.",
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.72,
            "reason": "LONG stop at $83,500 has NOT been triggered. Current price $84,200 is $700 ABOVE the stop loss. For a LONG position, the stop is triggered when price FALLS BELOW the stop level. Price is still safely above.",
            "reasoning": {
                "structure_analysis": "For LONG positions: stop is BELOW entry price. Stop triggers when price drops BELOW stop level. Current $84,200 > stop $83,500 = NOT triggered.",
                "data_assessment": "Price pulling back but maintaining above key support at $83,800. Normal retracement in uptrend.",
                "risk_factors": "Buffer: $700 (0.83% above stop). At 10x leverage, stop hit would mean ~8.3% loss. Acceptable risk.",
                "exit_logic": "Hard stop at $83,500 is intact. No safety net or guarding line breaks. Let the position work."
            },
            "execution": {"exit_pct": 0, "stop_price": 83500, "order_type": "LIMIT"}
        }
    },
    # LONG, price BELOW stop, breached → EXIT
    {
        "user": "Position: LONG BTC | Entry: $78,000 | Current: $76,800 | Stop: $77,200 | Leverage: 5x | R-Multiple: -1.5 | Duration: 6h | Market: BTC broke below support. Bearish engulfing on 1h. CVD declining.",
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "LONG stop at $77,200 has been BREACHED. Current price $76,800 is $400 BELOW the stop loss. For a LONG position, when price falls below the stop level, the stop is triggered. Exit immediately.",
            "reasoning": {
                "structure_analysis": "For LONG positions: stop triggers when price FALLS BELOW stop level. Current $76,800 < stop $77,200 = STOP TRIGGERED. Position is $400 past the stop.",
                "data_assessment": "Bearish engulfing confirms selling pressure. CVD declining. No reason to expect recovery.",
                "risk_factors": "Already -1.5R. At 5x leverage, additional 1% drop = another 5% loss. Risk/reward heavily unfavorable.",
                "exit_logic": "Hard stop breached. MCF hierarchy: Hard Stop is the FIRST and most critical exit level. No discretion — exit immediately."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # LONG, price barely above stop → HOLD with caution
    {
        "user": "Position: LONG ETH | Entry: $3,200 | Current: $3,085 | Stop: $3,070 | Leverage: 10x | R-Multiple: -0.88 | Duration: 4h | Market: ETH testing support. RSI 38. Funding slightly negative.",
        "assistant": {
            "action": "HOLD",
            "urgency": "HIGH",
            "confidence": 0.62,
            "reason": "LONG stop at $3,070 has NOT been triggered. Current price $3,085 is $15 ABOVE the stop. For a LONG, the stop triggers when price FALLS BELOW $3,070. Very tight buffer — let the stop do its job but be ready.",
            "reasoning": {
                "structure_analysis": "For LONG: price must fall BELOW $3,070 to trigger stop. Current $3,085 > $3,070 = still safe but razor-thin $15 buffer (0.49%).",
                "data_assessment": "RSI 38 is oversold territory, potential bounce zone. Support being tested not broken.",
                "risk_factors": "Buffer only 0.49%. At 10x, stop hit = ~4.9% loss. But stop is doing its job — it exists to protect. Don't pre-empt it.",
                "exit_logic": "Hard stop intact. Buffer <0.5% means confidence capped at 65% regardless of other signals. Let the stop work."
            },
            "execution": {"exit_pct": 0, "stop_price": 3070, "order_type": "LIMIT"}
        }
    },
    # LONG, price clearly below stop → EXIT (larger gap)
    {
        "user": "Position: LONG SOL | Entry: $145 | Current: $136.50 | Stop: $140 | Leverage: 3x | R-Multiple: -1.7 | Duration: 12h | Market: SOL gap down on exchange outage news. High volume selling.",
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "LONG stop at $140 has been BREACHED. Current price $136.50 is $3.50 BELOW the stop. For a LONG position, the stop is triggered when price FALLS BELOW $140. Price has fallen $3.50 past the stop level — exit immediately.",
            "reasoning": {
                "structure_analysis": "For LONG: stop triggers when price drops BELOW stop level. Current $136.50 < stop $140 = BREACHED by $3.50 (2.5%). This is a clear hard stop violation.",
                "data_assessment": "Gap down on negative news with high volume. No support until $130. Recovery unlikely in near term.",
                "risk_factors": "Already past stop by 2.5%. At 3x leverage, already lost ~25.5% leveraged. Further drop compounds losses.",
                "exit_logic": "Hard stop breached. MCF hierarchy demands immediate exit. No discretion allowed when hard stop is violated."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # LONG, price well above stop → strong HOLD
    {
        "user": "Position: LONG BTC | Entry: $70,000 | Current: $72,500 | Stop: $68,500 | Leverage: 5x | R-Multiple: +1.67 | Duration: 24h | Market: BTC trending up. Higher highs, higher lows. Bullish CVD.",
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.85,
            "reason": "LONG stop at $68,500 is $4,000 BELOW current price — well protected. For a LONG, the stop only triggers if price FALLS BELOW $68,500. At +1.67R with bullish structure, let the winner run.",
            "reasoning": {
                "structure_analysis": "For LONG: stop triggers when price drops BELOW $68,500. Current $72,500 is $4,000 above stop (5.5% buffer). Enormous safety margin.",
                "data_assessment": "Higher highs and higher lows = intact uptrend. Bullish CVD confirms buying pressure. No reason to exit.",
                "risk_factors": "5.5% buffer to stop. At 5x, even a 3% pullback would still be $1,500+ above stop. Very safe.",
                "exit_logic": "No exit levels triggered. Position in profit, trend intact, momentum favorable. Trail stop or wait for TP."
            },
            "execution": {"exit_pct": 0, "stop_price": 68500, "order_type": "LIMIT"}
        }
    },
    # LONG, price exactly at stop → EXIT (at stop = triggered for LONG)
    {
        "user": "Position: LONG BTC | Entry: $88,000 | Current: $86,500 | Stop: $86,500 | Leverage: 10x | R-Multiple: -1.0 | Duration: 3h | Market: BTC grinding down. No bounce at support.",
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "LONG stop at $86,500 has been TRIGGERED. Current price $86,500 is exactly AT the stop level. For a LONG, when price reaches the stop level, the position should be exited. No buffer remaining.",
            "reasoning": {
                "structure_analysis": "For LONG: stop triggers when price is AT or BELOW stop level. Current $86,500 = stop $86,500. Zero buffer. Stop is triggered.",
                "data_assessment": "Grinding down with no bounce at support indicates continued selling pressure.",
                "risk_factors": "At 10x with zero buffer, any further drop immediately compounds leveraged losses. Already at -1R.",
                "exit_logic": "Hard stop triggered. MCF hierarchy: immediate exit. The stop exists for this exact moment."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # LONG, price slightly below stop → EXIT (clearly breached)
    {
        "user": "Position: LONG AVAX | Entry: $42 | Current: $39.80 | Stop: $40.20 | Leverage: 20x | R-Multiple: -1.22 | Duration: 5h | Market: AVAX selling off with rest of alts. BTC also dropping.",
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "LONG stop at $40.20 has been BREACHED. Current price $39.80 is $0.40 BELOW the stop. For a LONG, when price FALLS BELOW the stop level, exit immediately. At 20x leverage this is catastrophic if held.",
            "reasoning": {
                "structure_analysis": "For LONG: stop triggers when price drops BELOW stop level. Current $39.80 < stop $40.20 = BREACHED by $0.40 (1.0%).",
                "data_assessment": "Broad altcoin selloff with BTC dragging everything down. No AVAX-specific catalyst for recovery.",
                "risk_factors": "At 20x leverage, already past stop. Every additional 1% drop = 20% more leveraged loss. Liquidation risk is extreme.",
                "exit_logic": "Hard stop breached at 20x leverage. CRITICAL exit. MCF hierarchy is unambiguous here."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # LONG, price above stop, drawdown but recovering → HOLD
    {
        "user": "Position: LONG BTC | Entry: $77,000 | Current: $76,200 | Stop: $75,500 | Leverage: 5x | R-Multiple: -0.53 | Duration: 8h | Market: BTC dipped to $75,800 but bouncing. Hammer candle on 1h. Funding neutral.",
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.74,
            "reason": "LONG stop at $75,500 has NOT been triggered. Current price $76,200 is $700 ABOVE the stop. For a LONG, the stop is only triggered when price FALLS BELOW $75,500. Price dipped to $75,800 but bounced — stop held.",
            "reasoning": {
                "structure_analysis": "For LONG: stop triggers when price drops BELOW $75,500. Low was $75,800, still $300 above stop. Current $76,200 is $700 above. Stop is intact.",
                "data_assessment": "Hammer candle on 1h indicates buying pressure at the lows. Bounce underway. Funding neutral = no crowding risk.",
                "risk_factors": "Buffer: $700 (0.92%). Survived the dip without hitting stop. Bounce pattern suggests near-term upside.",
                "exit_logic": "Hard stop held through the dip. Safety net intact. Position is recovering — let it work."
            },
            "execution": {"exit_pct": 0, "stop_price": 75500, "order_type": "LIMIT"}
        }
    },
    # LONG, confusing scenario — price dropped from entry but above stop
    {
        "user": "Position: LONG BTC | Entry: $89,000 | Current: $87,500 | Stop: $87,200 | Leverage: 10x | R-Multiple: -0.83 | Duration: 2h | Market: BTC pulled back after failed breakout above $89,500. Now consolidating around $87,500.",
        "assistant": {
            "action": "HOLD",
            "urgency": "HIGH",
            "confidence": 0.65,
            "reason": "LONG stop at $87,200 has NOT been triggered. Current price $87,500 is $300 ABOVE the stop. For a LONG, the stop triggers when price FALLS BELOW $87,200. Buffer is tight at 0.34% but the stop has not been hit.",
            "reasoning": {
                "structure_analysis": "For LONG: stop triggers when price drops BELOW $87,200. Current $87,500 > stop $87,200 = NOT triggered. Buffer: $300 (0.34%).",
                "data_assessment": "Failed breakout suggests short-term bearishness, but consolidation at $87,500 shows buyers stepping in above stop.",
                "risk_factors": "0.34% buffer at 10x = 3.4% leveraged risk if stop hits. Tight but the stop exists for protection. Confidence capped near 65% due to tight buffer.",
                "exit_logic": "Hard stop intact but barely. Let the stop do its job. Don't pre-empt a stop that hasn't triggered."
            },
            "execution": {"exit_pct": 0, "stop_price": 87200, "order_type": "LIMIT"}
        }
    },
    # LONG, price way below stop — obvious exit
    {
        "user": "Position: LONG XRP | Entry: $2.50 | Current: $2.20 | Stop: $2.35 | Leverage: 5x | R-Multiple: -2.0 | Duration: 24h | Market: XRP rejected at $2.55 resistance, now in freefall. Volume spike on selling.",
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "LONG stop at $2.35 has been BREACHED. Current price $2.20 is $0.15 BELOW the stop. For a LONG, when price FALLS BELOW the stop level, exit immediately. Position is already -2R and $0.15 past the stop.",
            "reasoning": {
                "structure_analysis": "For LONG: stop triggers when price drops BELOW $2.35. Current $2.20 < stop $2.35 = BREACHED by $0.15 (6.4% past stop).",
                "data_assessment": "Rejected at resistance with volume spike on selling. Freefall with no support until $2.00. Strongly bearish.",
                "risk_factors": "6.4% past stop at 5x = ~32% additional leveraged loss beyond stop. Catastrophic if held.",
                "exit_logic": "Hard stop breached by significant margin. MCF hierarchy: immediate unconditional exit."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
]

for ex in long_stop_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))


# ============================================================
# CATEGORY 2: SHORT Stop Direction Logic (10 examples)
# For SHORT: stop is ABOVE entry. Breached when price RISES ABOVE stop.
# Price BELOW stop = safe. Price ABOVE stop = triggered.
# ============================================================

short_stop_examples = [
    # SHORT, price below stop, safe → HOLD
    {
        "user": "Position: SHORT BTC | Entry: $90,000 | Current: $89,200 | Stop: $91,000 | Leverage: 10x | R-Multiple: +0.8 | Duration: 3h | Market: BTC rejecting from $90,500. Bearish divergence on RSI. Funding positive (crowded longs).",
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.82,
            "reason": "SHORT stop at $91,000 has NOT been triggered. Current price $89,200 is $1,800 BELOW the stop. For a SHORT, the stop triggers when price RISES ABOVE $91,000. Price is safely below — position is in profit at +0.8R.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price RISES ABOVE stop level. Current $89,200 < stop $91,000 = NOT triggered. Buffer: $1,800 (2.0%).",
                "data_assessment": "Bearish divergence on RSI + positive funding (crowded longs) = favorable setup for shorts. Rejection from $90,500.",
                "risk_factors": "2.0% buffer to stop. At 10x, comfortable margin. Position is profitable.",
                "exit_logic": "No exit triggers. Position profitable, trend bearish, momentum favorable. Continue holding."
            },
            "execution": {"exit_pct": 0, "stop_price": 91000, "order_type": "LIMIT"}
        }
    },
    # SHORT, price ABOVE stop → EXIT
    {
        "user": "Position: SHORT BTC | Entry: $85,000 | Current: $87,500 | Stop: $86,500 | Leverage: 5x | R-Multiple: -1.67 | Duration: 8h | Market: BTC breakout above resistance with volume. Short squeeze in progress.",
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "SHORT stop at $86,500 has been BREACHED. Current price $87,500 is $1,000 ABOVE the stop. For a SHORT, when price RISES ABOVE the stop level, the stop is triggered. Exit immediately.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price RISES ABOVE stop level. Current $87,500 > stop $86,500 = BREACHED by $1,000 (1.16%).",
                "data_assessment": "Breakout above resistance with volume and short squeeze = extremely bearish for shorts. More upside likely.",
                "risk_factors": "Already -1.67R at 5x. Short squeeze can accelerate. Every 1% up = 5% more loss.",
                "exit_logic": "Hard stop breached. MCF hierarchy: immediate exit. Short squeezes can be violent — do not hold."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # SHORT, price below stop but close → HOLD with caution
    {
        "user": "Position: SHORT ETH | Entry: $3,400 | Current: $3,380 | Stop: $3,420 | Leverage: 10x | R-Multiple: +0.29 | Duration: 1h | Market: ETH choppy around $3,380-$3,400. No clear direction.",
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.63,
            "reason": "SHORT stop at $3,420 has NOT been triggered. Current price $3,380 is $40 BELOW the stop. For a SHORT, the stop triggers when price RISES ABOVE $3,420. Buffer is $40 (1.18%) — tight but intact.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price RISES ABOVE $3,420. Current $3,380 < stop $3,420 = NOT triggered. Buffer: $40 (1.18%).",
                "data_assessment": "Choppy price action, no clear direction. Position is slightly profitable at +0.29R.",
                "risk_factors": "1.18% buffer at 10x. If stop hits, ~11.8% leveraged loss. Manageable but confidence limited due to chop.",
                "exit_logic": "Stop intact. Confidence capped due to tight buffer and indecisive market. Let the stop work."
            },
            "execution": {"exit_pct": 0, "stop_price": 3420, "order_type": "LIMIT"}
        }
    },
    # SHORT, price well below stop, winning → strong HOLD
    {
        "user": "Position: SHORT BTC | Entry: $82,000 | Current: $78,500 | Stop: $84,000 | Leverage: 3x | R-Multiple: +1.75 | Duration: 48h | Market: BTC in downtrend. Lower lows, lower highs. Negative funding.",
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.85,
            "reason": "SHORT stop at $84,000 is $5,500 ABOVE current price — massive buffer. For a SHORT, the stop triggers when price RISES ABOVE $84,000. At +1.75R with bearish structure, let the winner run.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price RISES ABOVE $84,000. Current $78,500 is $5,500 below stop (7.0% buffer). Enormous safety margin.",
                "data_assessment": "Lower lows, lower highs = intact downtrend. Negative funding means shorts are not crowded. Momentum favors continuation.",
                "risk_factors": "7.0% buffer to stop at 3x leverage. Extremely safe. Would need massive reversal to threaten stop.",
                "exit_logic": "No exit triggers. Trend intact. Consider trailing stop to lock in profit."
            },
            "execution": {"exit_pct": 0, "stop_price": 84000, "order_type": "LIMIT"}
        }
    },
    # SHORT, price exactly at stop → EXIT
    {
        "user": "Position: SHORT BTC | Entry: $88,000 | Current: $89,500 | Stop: $89,500 | Leverage: 10x | R-Multiple: -1.0 | Duration: 4h | Market: BTC pumping on ETF news. Strong buy pressure.",
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "SHORT stop at $89,500 has been TRIGGERED. Current price $89,500 is exactly AT the stop. For a SHORT, when price reaches the stop level, the position must be exited. Zero buffer.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price is AT or ABOVE stop level. Current $89,500 = stop $89,500. Zero buffer. Stop triggered.",
                "data_assessment": "ETF news pump with strong buy pressure. Likely more upside = more pain for shorts.",
                "risk_factors": "At 10x with zero buffer, any further rise immediately compounds losses. Already -1R.",
                "exit_logic": "Hard stop triggered. MCF hierarchy: immediate exit. News-driven pumps can extend far beyond expected."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # SHORT, price NOT at stop despite being slightly negative R → HOLD
    {
        "user": "Position: SHORT BTC | Entry: $70,000 | Current: $70,800 | Stop: $72,000 | Leverage: 5x | R-Multiple: -0.4 | Duration: 2h | Market: BTC bouncing off $70,500 support. Some buying but no breakout.",
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.71,
            "reason": "SHORT stop at $72,000 has NOT been triggered. Current price $70,800 is $1,200 BELOW the stop. For a SHORT, the stop triggers when price RISES ABOVE $72,000. Despite the -0.4R drawdown, stop has ample buffer.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price RISES ABOVE $72,000. Current $70,800 < stop $72,000 = NOT triggered. Buffer: $1,200 (1.69%).",
                "data_assessment": "Bounce off support but no breakout. Position slightly negative but stop well intact.",
                "risk_factors": "1.69% buffer at 5x leverage. Stop hit would mean ~8.5% loss. Normal drawdown in range-bound market.",
                "exit_logic": "Stop intact with good buffer. Position is slightly underwater but within normal variance. Hold."
            },
            "execution": {"exit_pct": 0, "stop_price": 72000, "order_type": "LIMIT"}
        }
    },
    # SHORT, price way above stop → obvious exit
    {
        "user": "Position: SHORT SOL | Entry: $160 | Current: $175 | Stop: $167 | Leverage: 10x | R-Multiple: -2.14 | Duration: 12h | Market: SOL pumping on ecosystem news. Vertical move with no pullback.",
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "SHORT stop at $167 has been BREACHED. Current price $175 is $8 ABOVE the stop. For a SHORT, when price RISES ABOVE the stop level, exit immediately. Position is $8 past stop and accelerating against us.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price RISES ABOVE $167. Current $175 > stop $167 = BREACHED by $8 (4.8% past stop).",
                "data_assessment": "Vertical pump on news with no pullback. This type of move often extends further before any retrace.",
                "risk_factors": "4.8% past stop at 10x = ~48% additional leveraged loss. Position is in catastrophic territory.",
                "exit_logic": "Hard stop breached by significant margin. MCF hierarchy: immediate unconditional exit."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # SHORT, negative funding but stop intact → HOLD (funding alone ≠ exit)
    {
        "user": "Position: SHORT BTC | Entry: $89,000 | Current: $88,200 | Stop: $90,200 | Leverage: 3x | R-Multiple: +0.67 | Duration: 6h | Market: Funding rate is -0.015% (crowded shorts). CVD neutral. OI declining.",
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.72,
            "reason": "SHORT stop at $90,200 is $2,000 above current price — NOT triggered. Negative funding indicates crowded shorts but this alone is NOT an exit signal. The position is profitable at +0.67R with intact stop.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price RISES ABOVE $90,200. Current $88,200 is $2,000 below. Buffer: 2.25%. Safe.",
                "data_assessment": "Negative funding (-0.015%) means shorts are paying longs. This is a risk factor but NOT a standalone exit signal. OI declining = positions closing, reducing squeeze risk.",
                "risk_factors": "Crowded short funding is a warning sign for potential squeeze BUT: stop is intact, position is profitable, and declining OI reduces crowding pressure. Monitor but don't exit.",
                "exit_logic": "Funding rate is a supplementary data point, NOT a structural break. MCF exit hierarchy requires actual price structure breaks — not just funding imbalances."
            },
            "execution": {"exit_pct": 0, "stop_price": 90200, "order_type": "LIMIT"}
        }
    },
    # SHORT, price just barely below stop → careful HOLD
    {
        "user": "Position: SHORT BNB | Entry: $620 | Current: $634 | Stop: $638 | Leverage: 5x | R-Multiple: -0.78 | Duration: 3h | Market: BNB grinding up slowly. Low volume. No news catalyst.",
        "assistant": {
            "action": "HOLD",
            "urgency": "HIGH",
            "confidence": 0.60,
            "reason": "SHORT stop at $638 has NOT been triggered. Current price $634 is $4 BELOW the stop. For a SHORT, the stop triggers when price RISES ABOVE $638. Buffer is tight at 0.63% but the stop is intact.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop triggers when price RISES ABOVE $638. Current $634 < stop $638 = NOT triggered. Buffer: $4 (0.63%).",
                "data_assessment": "Slow grind up on low volume is concerning but no explosive momentum. Low volume grinds often reverse.",
                "risk_factors": "0.63% buffer at 5x. If stop hits, ~3.1% leveraged loss. Tight but manageable. Confidence capped near 60% due to thin buffer.",
                "exit_logic": "Stop intact but barely. Don't pre-empt the stop. If it triggers, accept the loss. That's what stops are for."
            },
            "execution": {"exit_pct": 0, "stop_price": 638, "order_type": "LIMIT"}
        }
    },
    # SHORT, big drop, very profitable → HOLD/trail
    {
        "user": "Position: SHORT BTC | Entry: $91,000 | Current: $83,000 | Stop: $93,000 | Leverage: 10x | R-Multiple: +4.0 | Duration: 72h | Market: BTC crash. Cascading liquidations. Panic selling. CVD heavily negative.",
        "assistant": {
            "action": "TRAIL_STOP",
            "urgency": "MEDIUM",
            "confidence": 0.82,
            "reason": "SHORT at +4.0R with stop $10,000 above current price. For a SHORT, stop at $93,000 triggers if price RISES ABOVE $93,000. Price is $83,000 — miles below stop. At 10x leverage and +4R, tighten the trailing stop to lock in profit.",
            "reasoning": {
                "structure_analysis": "For SHORT: stop at $93,000 is $10,000 above current = 12% buffer. Position deep in profit at +4R.",
                "data_assessment": "Cascading liquidations and panic selling suggest more downside, but at +4R the priority shifts to profit protection.",
                "risk_factors": "At 10x with +4R = ~40% unrealized gain. A 5% bounce would cost 50% of that gain. Trail stop to protect.",
                "exit_logic": "At +4R with 10x leverage, trail stop to $86,000 (lock in ~2.5R). Let momentum continue but protect the gain."
            },
            "execution": {"exit_pct": 0, "stop_price": 86000, "order_type": "LIMIT"}
        }
    },
]

for ex in short_stop_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))


# ============================================================
# CATEGORY 3: TP1 Proximity + Leverage → TP_PARTIAL (8 examples)
# When price is near TP1 at high leverage, take partial profit
# Don't HOLD through TP1 — that's leaving money on the table
# ============================================================

tp1_examples = [
    # LONG, near TP1, 10x → take partial
    {
        "user": "Position: LONG BTC | Entry: $70,000 | Current: $71,200 | Stop: $69,000 | TP1: $71,800 | Leverage: 10x | R-Multiple: +0.6 | Duration: 6h | Market: BTC trending up. TP1 is $600 away (0.84%). Low volatility.",
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "MEDIUM",
            "confidence": 0.78,
            "reason": "LONG at +0.6R with TP1 only $600 away (0.84%) at 10x leverage. At high leverage, secure partial profit near TP1 rather than risk a reversal back to stop. Take 30-40% off at TP1 proximity.",
            "reasoning": {
                "structure_analysis": "TP1 at $71,800 is within 1 ATR. At 10x leverage, this $600 move = 8.4% leveraged gain on the partial.",
                "data_assessment": "Trending up in low volatility. Likely to reach TP1 but not guaranteed. Secure what's available.",
                "risk_factors": "At 10x, the distance to stop ($2,200) = 22% risk vs $600 to TP1 = 8.4% gain on partial. Lock in the partial.",
                "exit_logic": "MCF hierarchy: Take Profit level is approaching. At high leverage, partial TP is prudent risk management. Don't get greedy."
            },
            "execution": {"exit_pct": 35, "stop_price": 69000, "order_type": "LIMIT"}
        }
    },
    # LONG, near TP1, 20x → definitely take partial
    {
        "user": "Position: LONG ETH | Entry: $3,100 | Current: $3,180 | Stop: $3,020 | TP1: $3,220 | Leverage: 20x | R-Multiple: +1.0 | Duration: 4h | Market: ETH bullish, TP1 $40 away. Strong momentum.",
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "HIGH",
            "confidence": 0.82,
            "reason": "LONG at +1.0R with TP1 only $40 away (1.26%) at 20x leverage. At 20x, a 1% reversal = 20% leveraged loss. Secure 40-50% at TP1 proximity to lock in meaningful profit.",
            "reasoning": {
                "structure_analysis": "TP1 $40 away. At 20x, partial take locks in ~25% leveraged gain on that portion. Remaining position can ride to TP2/TP3.",
                "data_assessment": "Strong momentum suggests TP1 will be hit, but at 20x leverage the priority is profit preservation over maximization.",
                "risk_factors": "20x leverage makes every pullback devastating. At +1R, a full reversal to stop = -20x the R-distance loss. Partial take is essential.",
                "exit_logic": "MCF Take Profit level imminent. At 20x, always partial at TP1. Never risk a 20x position on hitting higher TPs."
            },
            "execution": {"exit_pct": 45, "stop_price": 3020, "order_type": "LIMIT"}
        }
    },
    # SHORT, near TP1, 10x → take partial
    {
        "user": "Position: SHORT BTC | Entry: $88,000 | Current: $87,200 | Stop: $89,200 | TP1: $86,800 | Leverage: 10x | R-Multiple: +0.67 | Duration: 5h | Market: BTC bearish. TP1 $400 away. Selling pressure.",
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "MEDIUM",
            "confidence": 0.76,
            "reason": "SHORT at +0.67R with TP1 only $400 away (0.46%) at 10x leverage. Near TP1 at high leverage = partial take. Secure 30% to lock in profit while keeping exposure for continuation.",
            "reasoning": {
                "structure_analysis": "For SHORT, TP1 at $86,800 means price needs to drop $400 more. At 10x, partial on $400 move = ~4.6% leveraged gain.",
                "data_assessment": "Selling pressure suggests TP1 will be reached. Partial take preserves opportunity while reducing risk.",
                "risk_factors": "At 10x, a $800 bounce (0.9%) back to entry = -9% leveraged loss. Partial de-risks without abandoning the position.",
                "exit_logic": "MCF Take Profit approaching. Partial at TP1, trail remainder to TP2/TP3."
            },
            "execution": {"exit_pct": 30, "stop_price": 89200, "order_type": "LIMIT"}
        }
    },
    # LONG, at TP1, 5x → definitely take partial
    {
        "user": "Position: LONG BTC | Entry: $65,000 | Current: $66,900 | Stop: $63,800 | TP1: $67,000 | Leverage: 5x | R-Multiple: +1.58 | Duration: 18h | Market: BTC approaching TP1. Momentum slowing. RSI 72.",
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "HIGH",
            "confidence": 0.82,
            "reason": "LONG at +1.58R with TP1 only $100 away (0.15%). RSI 72 signals overbought. Take 30-35% partial now — TP1 is essentially reached. Let the rest run with trailing stop.",
            "reasoning": {
                "structure_analysis": "TP1 at $67,000 is $100 from current $66,900. Effectively at TP1. RSI overbought adds conviction to take partial.",
                "data_assessment": "Momentum slowing near TP1 with RSI overbought. Classic setup for at least a partial reversal.",
                "risk_factors": "At 5x and +1.58R, partial locks in ~7.9% leveraged gain on that portion. Remaining rides with trailed stop.",
                "exit_logic": "MCF Take Profit level reached. Partial take is standard procedure at TP1. Trail stop on remainder to breakeven."
            },
            "execution": {"exit_pct": 35, "stop_price": 65000, "order_type": "LIMIT"}
        }
    },
    # LONG, near TP1, 1x → can afford to hold more
    {
        "user": "Position: LONG BTC | Entry: $70,000 | Current: $71,500 | Stop: $68,500 | TP1: $72,000 | Leverage: 1x | R-Multiple: +1.0 | Duration: 24h | Market: BTC in uptrend. TP1 $500 away. Structure intact.",
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.80,
            "reason": "LONG at +1.0R with TP1 $500 away at 1x leverage. At 1x, there's no liquidation risk and the stop provides adequate protection. Unlike high-leverage positions, 1x can afford to wait for TP1 to actually hit.",
            "reasoning": {
                "structure_analysis": "TP1 at $72,000 is $500 away. At 1x leverage, the risk/reward of waiting for the last $500 is favorable.",
                "data_assessment": "Uptrend intact with structure holding. Strong probability of reaching TP1.",
                "risk_factors": "At 1x, even a full stop hit only loses ~2.1%. No liquidation risk. Can afford patience.",
                "exit_logic": "At 1x leverage, partial TP is less urgent. Hold for TP1 to actually trigger, then take partial. Leverage tiers matter."
            },
            "execution": {"exit_pct": 0, "stop_price": 68500, "order_type": "LIMIT"}
        }
    },
    # SHORT, near TP1, 20x → urgent partial
    {
        "user": "Position: SHORT BTC | Entry: $90,000 | Current: $88,500 | Stop: $91,200 | TP1: $88,000 | Leverage: 20x | R-Multiple: +1.25 | Duration: 3h | Market: BTC dropping. TP1 $500 away. High volatility.",
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "HIGH",
            "confidence": 0.85,
            "reason": "SHORT at +1.25R with TP1 only $500 away at 20x leverage. At 20x, ALWAYS partial at TP1 proximity. A 0.5% bounce = 10% leveraged loss. Lock in 50% now.",
            "reasoning": {
                "structure_analysis": "TP1 at $88,000 is $500 below current. At 20x, partial on existing +1.25R locks in ~25% leveraged gain on that portion.",
                "data_assessment": "High volatility means quick reversals are possible even in a downtrend. At 20x, volatility is your enemy.",
                "risk_factors": "20x leverage + high volatility = extreme reversal risk. A $900 bounce (1%) = 20% leveraged loss. Must de-risk.",
                "exit_logic": "MCF Take Profit approaching at 20x. This is the highest urgency for partial takes. Lock in profit, trail the rest."
            },
            "execution": {"exit_pct": 50, "stop_price": 91200, "order_type": "LIMIT"}
        }
    },
    # LONG, past TP1 → trail and partial
    {
        "user": "Position: LONG BTC | Entry: $68,000 | Current: $70,500 | Stop: $66,500 | TP1: $70,000 | Leverage: 10x | R-Multiple: +1.67 | Duration: 36h | Market: BTC past TP1, momentum strong. TP2 at $72,500.",
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "MEDIUM",
            "confidence": 0.80,
            "reason": "LONG past TP1 ($70,000) at +1.67R with 10x leverage. Already past the first take-profit level. Secure 30% and move stop to breakeven for the remainder. Target TP2 at $72,500.",
            "reasoning": {
                "structure_analysis": "Price $70,500 is $500 above TP1 ($70,000). At 10x, locking in partial at +1.67R = ~16.7% leveraged gain on partial.",
                "data_assessment": "Momentum strong, TP2 at $72,500 is achievable. But at 10x, never leave all chips on the table past TP1.",
                "risk_factors": "Past TP1 is the textbook moment for partial take. Remaining position rides with moved stop.",
                "exit_logic": "MCF hierarchy: TP1 hit → partial take + move stop to breakeven. Standard procedure at any leverage, critical at 10x."
            },
            "execution": {"exit_pct": 30, "stop_price": 68000, "order_type": "LIMIT"}
        }
    },
    # SHORT, near TP1 but low leverage → hold more aggressively
    {
        "user": "Position: SHORT BTC | Entry: $85,000 | Current: $83,000 | Stop: $87,000 | TP1: $82,500 | Leverage: 1x | R-Multiple: +1.0 | Duration: 48h | Market: BTC downtrend. TP1 $500 away. Bearish structure intact.",
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.80,
            "reason": "SHORT at +1.0R with TP1 $500 away at 1x leverage. At 1x, no liquidation risk — can afford to hold for TP1 to actually hit and potentially ride further. Bearish structure supports continuation.",
            "reasoning": {
                "structure_analysis": "TP1 at $82,500 is $500 below current. At 1x, the remaining $500 of downside has favorable risk/reward.",
                "data_assessment": "Downtrend with bearish structure intact. High probability of reaching TP1 and potentially TP2/TP3.",
                "risk_factors": "At 1x, even a $2,000 bounce (2.4%) is manageable. No leveraged risk multiplication.",
                "exit_logic": "At 1x leverage, TP1 proximity is not urgent. Hold for TP1 trigger, then evaluate partial. Leverage context matters."
            },
            "execution": {"exit_pct": 0, "stop_price": 87000, "order_type": "LIMIT"}
        }
    },
]

for ex in tp1_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))


# ============================================================
# CATEGORY 4: No-Stop Leverage Tiers (20 examples)
# 1x = HOLD (no liquidation risk, recommend setting stop)
# 3x = HOLD + recommend stop
# 5x winning = HOLD + recommend stop, 5x losing = REDUCE 50%
# 10x winning = REDUCE 50%, 10x losing = EXIT
# 20x = ALWAYS EXIT immediately
# ============================================================

no_stop_tiers = [
    # 1x no stop, winning → HOLD
    {"user": "Position: LONG BTC | Entry: $80,000 | Current: $82,000 | Stop: NONE | Leverage: 1x | R-Multiple: +0.0 | Duration: 12h | Market: BTC uptrend. No immediate risk.",
     "assistant": {"action": "HOLD", "urgency": "LOW", "confidence": 0.78,
        "reason": "LONG at 1x with no stop. At 1x leverage there is ZERO liquidation risk. Position is profitable. Recommend setting a stop at a structural level, but no urgency to exit.",
        "reasoning": {"structure_analysis": "No stop loss set. At 1x leverage, no liquidation risk regardless of price movement.", "data_assessment": "Uptrend, position profitable. No immediate threats.", "risk_factors": "1x leverage = no forced liquidation. Max loss is 100% of position size, but that would require BTC going to $0. Recommend stop but don't force exit.", "exit_logic": "1x no-stop: HOLD. Recommend setting stop at recent swing low for risk management, but do NOT exit solely because no stop is set."},
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "LIMIT"}}},
    # 1x no stop, losing → still HOLD at 1x
    {"user": "Position: SHORT BTC | Entry: $85,000 | Current: $86,500 | Stop: NONE | Leverage: 1x | R-Multiple: +0.0 | Duration: 24h | Market: BTC bouncing. Short slightly underwater.",
     "assistant": {"action": "HOLD", "urgency": "MEDIUM", "confidence": 0.72,
        "reason": "SHORT at 1x with no stop. At 1x leverage, no liquidation risk. Position is slightly underwater but 1x leverage means no catastrophic risk. Strongly recommend setting a stop loss, but no emergency exit needed.",
        "reasoning": {"structure_analysis": "No stop at 1x. Zero liquidation risk.", "data_assessment": "BTC bouncing but no explosive move. Short is underwater by ~1.8%.", "risk_factors": "At 1x, a 5% adverse move = 5% loss. Manageable. Not ideal without stop but not dangerous.", "exit_logic": "1x no-stop: HOLD with recommendation to set stop. The absence of a stop at 1x is poor practice but not an emergency."},
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "LIMIT"}}},
    # 1x no stop, another winning → HOLD
    {"user": "Position: LONG ETH | Entry: $3,000 | Current: $3,150 | Stop: NONE | Leverage: 1x | R-Multiple: +0.0 | Duration: 48h | Market: ETH grinding up. Healthy uptrend.",
     "assistant": {"action": "HOLD", "urgency": "LOW", "confidence": 0.80,
        "reason": "LONG at 1x with no stop. Zero liquidation risk at 1x leverage. Position is profitable (+5%). Recommend setting stop at $2,900 (recent support) for risk management. No exit needed.",
        "reasoning": {"structure_analysis": "1x leverage = no liquidation. Position profitable.", "data_assessment": "Healthy uptrend, grinding up steadily. Low risk of sudden reversal.", "risk_factors": "1x leverage eliminates forced liquidation. Even a 10% drop only means 10% loss on position. No emergency.", "exit_logic": "1x no-stop: HOLD. Always recommend stop but never force exit at 1x solely due to missing stop."},
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "LIMIT"}}},
    # 3x no stop, winning → HOLD with warning
    {"user": "Position: LONG BTC | Entry: $75,000 | Current: $77,000 | Stop: NONE | Leverage: 3x | R-Multiple: +0.0 | Duration: 8h | Market: BTC bullish. Position in profit.",
     "assistant": {"action": "HOLD", "urgency": "MEDIUM", "confidence": 0.70,
        "reason": "LONG at 3x with no stop. At 3x, liquidation occurs around 33% adverse move (~$50,000 for this entry). Not immediate risk but stop loss is strongly recommended. Position is profitable — hold but SET A STOP.",
        "reasoning": {"structure_analysis": "3x leverage, no stop. Liquidation at ~33% drawdown. Current price well above any liquidation zone.", "data_assessment": "Bullish market, position profitable at ~2.7%.", "risk_factors": "3x = moderate leverage. A 10% drop = 30% leveraged loss. Not immediately dangerous but poor risk management without stop.", "exit_logic": "3x no-stop: HOLD + strong recommendation to set stop at nearest structural support. Monitor closely."},
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "LIMIT"}}},
    # 3x no stop, losing → HOLD with strong warning
    {"user": "Position: SHORT BTC | Entry: $88,000 | Current: $89,200 | Stop: NONE | Leverage: 3x | R-Multiple: +0.0 | Duration: 6h | Market: BTC bouncing. Short losing ~1.4%.",
     "assistant": {"action": "HOLD", "urgency": "HIGH", "confidence": 0.65,
        "reason": "SHORT at 3x with no stop. Position is losing ~1.4% (4.2% leveraged). At 3x, liquidation is at ~33% adverse move. Not in immediate danger but MUST set a stop loss now. Hold for now but if no stop is set within next candle close, consider reducing.",
        "reasoning": {"structure_analysis": "3x leverage losing position with no stop. Not near liquidation but trajectory is concerning.", "data_assessment": "BTC bouncing, could continue higher. Without a stop, losses can accumulate.", "risk_factors": "3x losing + no stop = potential for significant leveraged loss. Each 1% adverse = 3% loss. Urgently needs stop.", "exit_logic": "3x no-stop losing: HOLD but URGENTLY recommend stop. Not yet at exit threshold but getting close."},
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "LIMIT"}}},
    # 5x no stop, winning → HOLD + recommend stop
    {"user": "Position: LONG BTC | Entry: $70,000 | Current: $72,000 | Stop: NONE | Leverage: 5x | R-Multiple: +0.0 | Duration: 12h | Market: BTC uptrend, position profitable at +2.86%.",
     "assistant": {"action": "HOLD", "urgency": "MEDIUM", "confidence": 0.68,
        "reason": "LONG at 5x with no stop. Position is winning (+2.86% = ~14.3% leveraged). At 5x, liquidation is at ~20% adverse move. Position is profitable — hold but IMMEDIATELY set a stop at recent swing low ($69,000) to protect gains.",
        "reasoning": {"structure_analysis": "5x leverage winning, no stop. Liquidation at ~20% drawdown.", "data_assessment": "Uptrend, profitable. But 5x without stop is risky even when winning.", "risk_factors": "5x = a 5% adverse move erases the entire gain and then some. Without stop, one bad candle could be devastating.", "exit_logic": "5x no-stop winning: HOLD + immediate stop placement. Protect the gain."},
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "LIMIT"}}},
    # 5x no stop, losing → REDUCE 50%
    {"user": "Position: SHORT BTC | Entry: $82,000 | Current: $83,500 | Stop: NONE | Leverage: 5x | R-Multiple: +0.0 | Duration: 8h | Market: BTC grinding up. Short losing ~1.8%.",
     "assistant": {"action": "REDUCE_SIZE", "urgency": "HIGH", "confidence": 0.80,
        "reason": "SHORT at 5x with no stop and losing 1.8% (9% leveraged). At 5x leverage without a stop, a losing position must be reduced. Exit 50% to cut risk, set a stop on the remainder.",
        "reasoning": {"structure_analysis": "5x leverage losing with no stop. Liquidation at ~20% adverse. Currently 1.8% adverse = 9% leveraged loss.", "data_assessment": "BTC grinding up suggests continued adverse movement. No reversal signals.", "risk_factors": "5x losing + no stop + adverse trend = compounding risk. A 4% total adverse move = 20% leveraged loss. Must de-risk.", "exit_logic": "5x no-stop losing: REDUCE 50% immediately. Set stop on remainder at $84,500 (recent high). Cannot hold full size at 5x losing without stop."},
        "execution": {"exit_pct": 50, "stop_price": None, "order_type": "MARKET"}}},
    # 5x no stop, losing more → REDUCE
    {"user": "Position: LONG ETH | Entry: $3,200 | Current: $3,100 | Stop: NONE | Leverage: 5x | R-Multiple: +0.0 | Duration: 16h | Market: ETH dropping. Support at $3,050.",
     "assistant": {"action": "REDUCE_SIZE", "urgency": "HIGH", "confidence": 0.82,
        "reason": "LONG at 5x with no stop, losing 3.1% (15.5% leveraged). At 5x losing without a stop, must reduce exposure immediately. Exit 50% and set stop at $3,050 support on remaining position.",
        "reasoning": {"structure_analysis": "5x losing, no stop. 3.1% adverse = 15.5% leveraged loss and growing.", "data_assessment": "ETH dropping toward $3,050 support. If that breaks, accelerated losses.", "risk_factors": "Already 15.5% leveraged loss. Another 2% drop = 10% more. Without stop, this can snowball.", "exit_logic": "5x no-stop losing: REDUCE 50%. Critical. Every hour without action increases risk."},
        "execution": {"exit_pct": 50, "stop_price": None, "order_type": "MARKET"}}},
    # 10x no stop, winning → REDUCE 50%
    {"user": "Position: LONG BTC | Entry: $68,000 | Current: $69,500 | Stop: NONE | Leverage: 10x | R-Multiple: +0.0 | Duration: 6h | Market: BTC moving up. Position +2.2%.",
     "assistant": {"action": "REDUCE_SIZE", "urgency": "HIGH", "confidence": 0.82,
        "reason": "LONG at 10x with no stop. Even though position is winning (+2.2% = 22% leveraged), 10x without a stop is extremely dangerous. A 3% reversal = 30% leveraged loss. Reduce 50% to lock in gains, set stop on remainder.",
        "reasoning": {"structure_analysis": "10x leverage winning, no stop. Liquidation at ~10% adverse move (~$61,200).", "data_assessment": "BTC moving up, position profitable. But 10x amplifies any reversal.", "risk_factors": "10x = every 1% move = 10% leveraged P&L. A flash crash of 5% = 50% loss. Without stop, this is gambling.", "exit_logic": "10x no-stop winning: REDUCE 50% immediately. Lock in half the gain. Set stop on remainder. 10x requires active risk management."},
        "execution": {"exit_pct": 50, "stop_price": None, "order_type": "MARKET"}}},
    # 10x no stop, winning (SHORT) → REDUCE
    {"user": "Position: SHORT BTC | Entry: $90,000 | Current: $88,000 | Stop: NONE | Leverage: 10x | R-Multiple: +0.0 | Duration: 4h | Market: BTC dropping. Short is +2.2%. High volatility.",
     "assistant": {"action": "REDUCE_SIZE", "urgency": "HIGH", "confidence": 0.80,
        "reason": "SHORT at 10x with no stop, winning +2.2% (22% leveraged). At 10x, ALWAYS reduce a no-stop position even when winning. A 2% bounce = 20% leveraged loss. Take 50% off the table now.",
        "reasoning": {"structure_analysis": "10x SHORT winning, no stop. Good gains but unprotected.", "data_assessment": "BTC dropping but high volatility means sharp reversals are common.", "risk_factors": "10x + high volatility + no stop = extreme risk even when profitable. One 3% bounce wipes out all gains.", "exit_logic": "10x no-stop: REDUCE 50% regardless of direction. This is non-negotiable at 10x. Set stop on remainder."},
        "execution": {"exit_pct": 50, "stop_price": None, "order_type": "MARKET"}}},
    # 10x no stop, losing → EXIT fully
    {"user": "Position: LONG BTC | Entry: $85,000 | Current: $83,800 | Stop: NONE | Leverage: 10x | R-Multiple: +0.0 | Duration: 3h | Market: BTC selling off. Position losing 1.4%.",
     "assistant": {"action": "EXIT_100_PERCENT_IMMEDIATELY", "urgency": "CRITICAL", "confidence": 0.92,
        "reason": "LONG at 10x with no stop and LOSING 1.4% (14% leveraged). At 10x leverage, a losing position without a stop is a critical emergency. Liquidation at ~10% adverse. Already 1.4% adverse — another 8.6% and the position is liquidated. Exit immediately.",
        "reasoning": {"structure_analysis": "10x losing, no stop. Already 1.4% adverse = 14% leveraged loss. Liquidation zone approaching.", "data_assessment": "BTC selling off with no reversal signs. Trend is against the position.", "risk_factors": "At 10x losing: 14% leveraged loss already. Liquidation at ~$76,500. Every minute without exit increases risk of total loss.", "exit_logic": "10x no-stop losing: EXIT 100% IMMEDIATELY. This is the highest priority action. Cannot hold a losing 10x position without a stop under any circumstances."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    # 10x no stop, losing (SHORT) → EXIT
    {"user": "Position: SHORT BTC | Entry: $78,000 | Current: $79,200 | Stop: NONE | Leverage: 10x | R-Multiple: +0.0 | Duration: 5h | Market: BTC bouncing strongly. Short losing 1.5%.",
     "assistant": {"action": "EXIT_100_PERCENT_IMMEDIATELY", "urgency": "CRITICAL", "confidence": 0.92,
        "reason": "SHORT at 10x with no stop and LOSING 1.5% (15% leveraged). At 10x, a losing position without a stop must be exited immediately. A 10% adverse move = 100% loss (liquidation). Already 1.5% in the wrong direction.",
        "reasoning": {"structure_analysis": "10x SHORT losing, no stop. 1.5% adverse = 15% leveraged loss.", "data_assessment": "BTC bouncing strongly. Momentum against the short position.", "risk_factors": "Liquidation at ~$85,800. Already 1.5% adverse. BTC bouncing means more upside likely.", "exit_logic": "10x no-stop losing: EXIT IMMEDIATELY. Non-negotiable. Protect remaining capital."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    # 20x no stop, winning → EXIT always
    {"user": "Position: LONG BTC | Entry: $87,000 | Current: $88,500 | Stop: NONE | Leverage: 20x | R-Multiple: +0.0 | Duration: 2h | Market: BTC pumping. Position +1.7%.",
     "assistant": {"action": "EXIT_100_PERCENT_IMMEDIATELY", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "LONG at 20x with no stop. Despite being profitable (+1.7% = 34% leveraged), 20x without a stop is ALWAYS an immediate exit. A 5% reversal = 100% loss (liquidation). Lock in the 34% gain NOW.",
        "reasoning": {"structure_analysis": "20x winning, no stop. Liquidation at ~5% adverse move (~$82,650).", "data_assessment": "BTC pumping, looks good. But at 20x, one bad candle can wipe out everything.", "risk_factors": "20x + no stop = liquidation risk is always extreme. Even winning, a 2% flash dip = 40% leveraged loss. Cannot hold.", "exit_logic": "20x no-stop: ALWAYS EXIT regardless of P&L direction. This is an absolute rule. Lock in the 34% gain before it evaporates."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    # 20x no stop, losing → EXIT immediately
    {"user": "Position: SHORT BTC | Entry: $90,000 | Current: $91,000 | Stop: NONE | Leverage: 20x | R-Multiple: +0.0 | Duration: 1h | Market: BTC grinding up. Short losing 1.1%.",
     "assistant": {"action": "EXIT_100_PERCENT_IMMEDIATELY", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "SHORT at 20x with no stop and LOSING 1.1% (22% leveraged). At 20x, liquidation is at ~5% adverse move ($94,500). Already 1.1% in the wrong direction — only 3.9% from liquidation. EXIT IMMEDIATELY.",
        "reasoning": {"structure_analysis": "20x SHORT losing, no stop. 1.1% adverse = 22% leveraged loss. Liquidation at $94,500.", "data_assessment": "BTC grinding up. At this rate, liquidation zone could be reached within hours.", "risk_factors": "20x losing + no stop = approaching liquidation. Every minute increases the probability of total loss.", "exit_logic": "20x no-stop losing: CRITICAL EXIT. This is the most dangerous possible position state. Exit NOW."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    # 20x no stop, at breakeven → EXIT
    {"user": "Position: LONG ETH | Entry: $3,100 | Current: $3,105 | Stop: NONE | Leverage: 20x | R-Multiple: +0.0 | Duration: 1h | Market: ETH flat. No clear direction.",
     "assistant": {"action": "EXIT_100_PERCENT_IMMEDIATELY", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "LONG at 20x with no stop. Position is at breakeven. At 20x leverage without a stop, EXIT IMMEDIATELY regardless of current P&L. Liquidation at ~5% drawdown ($2,945). Any adverse move is catastrophically amplified.",
        "reasoning": {"structure_analysis": "20x at breakeven, no stop. Zero protection against adverse moves.", "data_assessment": "Flat market with no direction. Could go either way — at 20x that's a coin flip with extreme stakes.", "risk_factors": "20x = every 1% move = 20% P&L. A 3% drop = 60% loss. No stop means no safety net at all.", "exit_logic": "20x no-stop: ALWAYS EXIT. Doesn't matter if winning, losing, or flat. The combination of 20x leverage and no stop is unacceptable risk."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    # 20x no stop, big winner → still EXIT
    {"user": "Position: SHORT BTC | Entry: $92,000 | Current: $87,000 | Stop: NONE | Leverage: 20x | R-Multiple: +0.0 | Duration: 12h | Market: BTC crashed hard. Short is +5.4%.",
     "assistant": {"action": "EXIT_100_PERCENT_IMMEDIATELY", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "SHORT at 20x with no stop, winning +5.4% (108% leveraged gain). Despite this incredible gain, 20x without a stop means a 5% bounce = 100% loss. Lock in the 108% gain IMMEDIATELY. Do not give it back.",
        "reasoning": {"structure_analysis": "20x SHORT winning massively, no stop. Unrealized gain is 108% leveraged.", "data_assessment": "BTC crashed but after a 5.4% crash, a relief bounce is extremely likely. At 20x, that bounce could wipe out all gains.", "risk_factors": "Sitting on 108% gain with zero protection. A 5% bounce liquidates the position. The greediest mistake is holding a 20x no-stop winner.", "exit_logic": "20x no-stop: EXIT IMMEDIATELY. Even with 108% gain. Especially with 108% gain. Lock it in. This is non-negotiable."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    # 3x no stop winning → HOLD
    {"user": "Position: LONG SOL | Entry: $140 | Current: $148 | Stop: NONE | Leverage: 3x | R-Multiple: +0.0 | Duration: 24h | Market: SOL in strong uptrend. +5.7%.",
     "assistant": {"action": "HOLD", "urgency": "LOW", "confidence": 0.72,
        "reason": "LONG at 3x with no stop, winning +5.7% (17.1% leveraged). At 3x, liquidation is at ~33% drawdown ($93.80). Far from liquidation and position is profitable. Recommend setting stop at $135 but no emergency exit.",
        "reasoning": {"structure_analysis": "3x winning, no stop. Liquidation far away at ~$93.80.", "data_assessment": "Strong uptrend, healthy gains. No immediate reversal signals.", "risk_factors": "3x = moderate leverage. A 5% pullback = 15% loss but no liquidation threat. Manageable.", "exit_logic": "3x no-stop winning: HOLD + recommend stop. Not an emergency. 3x is manageable leverage."},
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "LIMIT"}}},
    # 5x no stop, winning SHORT → HOLD
    {"user": "Position: SHORT BTC | Entry: $84,000 | Current: $80,000 | Stop: NONE | Leverage: 5x | R-Multiple: +0.0 | Duration: 36h | Market: BTC downtrend. Short winning +4.8%.",
     "assistant": {"action": "HOLD", "urgency": "MEDIUM", "confidence": 0.68,
        "reason": "SHORT at 5x with no stop, winning +4.8% (24% leveraged). At 5x winning, the position is healthy but needs a stop. Recommend setting stop at $85,500 (recent resistance). Hold the position but protect it.",
        "reasoning": {"structure_analysis": "5x SHORT winning, no stop. Liquidation at ~$100,800. Far from danger.", "data_assessment": "BTC downtrend continues. Position has strong unrealized gains.", "risk_factors": "5x winning without stop is risky but manageable. The concern is a sharp reversal wiping out the 24% gain.", "exit_logic": "5x no-stop winning: HOLD + recommend stop immediately. Unlike 10x/20x, 5x winning can hold but MUST set stop."},
        "execution": {"exit_pct": 0, "stop_price": None, "order_type": "LIMIT"}}},
    # 10x no stop, breakeven → REDUCE
    {"user": "Position: LONG BTC | Entry: $77,000 | Current: $77,100 | Stop: NONE | Leverage: 10x | R-Multiple: +0.0 | Duration: 2h | Market: BTC flat. Just entered.",
     "assistant": {"action": "REDUCE_SIZE", "urgency": "HIGH", "confidence": 0.80,
        "reason": "LONG at 10x with no stop, at breakeven. At 10x, ANY position without a stop must be reduced. Even at breakeven, a 3% drop = 30% leveraged loss with no protection. Reduce 50% and set stop on remainder.",
        "reasoning": {"structure_analysis": "10x at breakeven, no stop. Liquidation at ~$69,300.", "data_assessment": "Flat market. No edge currently. 50/50 risk.", "risk_factors": "10x at breakeven with no stop = unacceptable. The position has no protection against any adverse move.", "exit_logic": "10x no-stop: REDUCE 50%. At 10x, the only acceptable no-stop state is a rapidly shrinking position. Reduce and set stop."},
        "execution": {"exit_pct": 50, "stop_price": None, "order_type": "MARKET"}}},
    # 5x no stop losing → REDUCE (not exit)
    {"user": "Position: LONG BTC | Entry: $79,000 | Current: $78,200 | Stop: NONE | Leverage: 5x | R-Multiple: +0.0 | Duration: 4h | Market: BTC pulling back. Position -1.0%.",
     "assistant": {"action": "REDUCE_SIZE", "urgency": "HIGH", "confidence": 0.78,
        "reason": "LONG at 5x with no stop, losing 1.0% (5% leveraged). At 5x losing without a stop, reduce 50% to cut risk. Set stop on the remaining 50% at nearest support ($77,500).",
        "reasoning": {"structure_analysis": "5x losing, no stop. 1% adverse = 5% leveraged loss and growing.", "data_assessment": "BTC pulling back. Without stop, continued pullback compounds losses.", "risk_factors": "5x losing + no stop = growing risk. Not as critical as 10x/20x but must de-risk.", "exit_logic": "5x no-stop losing: REDUCE 50% + set stop. Don't exit fully — 5x is manageable if properly stopped."},
        "execution": {"exit_pct": 50, "stop_price": None, "order_type": "MARKET"}}},
]

for ex in no_stop_tiers:
    examples.append(make_example(ex["user"], ex["assistant"]))


# ============================================================
# CATEGORY 5: Funding Alone ≠ Exit Signal (6 examples)
# Negative funding (crowded shorts) should NOT cause panic exits
# Funding is supplementary data, NOT a structural break
# ============================================================

funding_examples = [
    {"user": "Position: SHORT BTC | Entry: $89,000 | Current: $88,500 | Stop: $90,500 | Leverage: 5x | R-Multiple: +0.33 | Duration: 4h | Market: Funding rate -0.012% (shorts crowded). CVD neutral. OI declining. Price in downtrend.",
     "assistant": {"action": "HOLD", "urgency": "MEDIUM", "confidence": 0.72,
        "reason": "SHORT at +0.33R with stop intact. Negative funding (-0.012%) indicates crowded shorts but this is NOT a structural break. OI is declining (positions closing, reducing squeeze risk). Price remains in downtrend. Funding alone is not an exit signal.",
        "reasoning": {"structure_analysis": "Stop at $90,500 is $2,000 above current. Safe. Downtrend intact.", "data_assessment": "Negative funding is a risk factor but NOT a trigger. OI declining actually REDUCES squeeze risk as shorts are exiting. Net effect is neutral.", "risk_factors": "Funding is supplementary data. MCF hierarchy requires PRICE STRUCTURE breaks for exits — not funding imbalances. No structure broken.", "exit_logic": "Funding is noted but does not override the MCF exit hierarchy. Stop intact, trend bearish, position profitable. HOLD."},
        "execution": {"exit_pct": 0, "stop_price": 90500, "order_type": "LIMIT"}}},
    {"user": "Position: SHORT ETH | Entry: $3,500 | Current: $3,420 | Stop: $3,600 | Leverage: 3x | R-Multiple: +0.8 | Duration: 12h | Market: Funding rate -0.018% (very crowded shorts). BTC dropping. ETH following. CVD bearish.",
     "assistant": {"action": "HOLD", "urgency": "MEDIUM", "confidence": 0.75,
        "reason": "SHORT at +0.8R with bearish structure. Funding is -0.018% (crowded shorts) but BTC is dropping, ETH following, and CVD is bearish. The bearish price action OVERRIDES the crowded funding concern. Structural evidence > funding signals.",
        "reasoning": {"structure_analysis": "Stop $180 above current (5.3% buffer). Downtrend intact with BTC leading lower.", "data_assessment": "Very crowded funding BUT bearish price action, bearish CVD, BTC correlation all support continuation. Funding is one data point of many.", "risk_factors": "Crowded shorts increase squeeze risk IF price reverses. But price is NOT reversing — it's confirming the short thesis. Act on structure, not funding.", "exit_logic": "MCF hierarchy: no structural breaks. Funding is noted but does not trigger exit. Price action is king."},
        "execution": {"exit_pct": 0, "stop_price": 3600, "order_type": "LIMIT"}}},
    {"user": "Position: SHORT BTC | Entry: $70,000 | Current: $69,200 | Stop: $71,500 | Leverage: 1x | R-Multiple: +0.53 | Duration: 8h | Market: Funding rate -0.020%. CVD positive (buyers stepping in). OI increasing. But price still below entry.",
     "assistant": {"action": "HOLD", "urgency": "MEDIUM", "confidence": 0.68,
        "reason": "SHORT at +0.53R on 1x leverage. Funding is heavily negative (-0.020%) and CVD is turning positive — these are warning signs of potential squeeze. However at 1x leverage with stop at $71,500 (3.3% buffer), the risk is contained. Monitor closely but don't exit on funding alone.",
        "reasoning": {"structure_analysis": "Stop $2,300 above current. 1x leverage = no liquidation risk. Price still below entry.", "data_assessment": "Funding + CVD divergence is a real warning. But at 1x, the cost of a squeeze is manageable (max 3.3% loss if stop hits).", "risk_factors": "Funding crowding is a WARNING, not an EXIT signal. The stop loss exists to handle the squeeze if it comes. Trust the stop.", "exit_logic": "Funding divergence noted. At 1x leverage with intact stop, the appropriate response is heightened monitoring — not panic exit."},
        "execution": {"exit_pct": 0, "stop_price": 71500, "order_type": "LIMIT"}}},
    {"user": "Position: SHORT BTC | Entry: $85,000 | Current: $83,000 | Stop: $86,500 | Leverage: 10x | R-Multiple: +1.33 | Duration: 18h | Market: Funding rate -0.025% (extremely crowded). CVD turning positive. OI spiking. But price made new low at $82,500.",
     "assistant": {"action": "TP_PARTIAL", "urgency": "MEDIUM", "confidence": 0.76,
        "reason": "SHORT at +1.33R on 10x with extremely crowded funding (-0.025%). At 10x with crowded shorts AND CVD/OI divergence, the appropriate response is PARTIAL take — not full exit. Price made new lows confirming the trend. Take 30% to de-risk while keeping exposure to the trend.",
        "reasoning": {"structure_analysis": "New low at $82,500 confirms downtrend. Stop $3,500 above current (4.2% buffer).", "data_assessment": "Extremely crowded funding + positive CVD + spiking OI = squeeze setup. But price is still making new lows. Mixed signals.", "risk_factors": "At 10x with crowded funding, a squeeze could be violent. Partial take reduces exposure while maintaining the thesis.", "exit_logic": "Crowded funding at 10x justifies PARTIAL de-risk, not full exit. The trend is still intact (new lows). Take partial, trail stop on remainder."},
        "execution": {"exit_pct": 30, "stop_price": 86500, "order_type": "LIMIT"}}},
    {"user": "Position: LONG BTC | Entry: $75,000 | Current: $76,500 | Stop: $73,800 | Leverage: 5x | R-Multiple: +1.25 | Duration: 24h | Market: Funding rate +0.025% (crowded longs). CVD declining. But price in strong uptrend, higher highs.",
     "assistant": {"action": "HOLD", "urgency": "MEDIUM", "confidence": 0.72,
        "reason": "LONG at +1.25R with crowded long funding (+0.025%). Positive funding means longs are paying shorts — potential squeeze DOWN. However, price is making higher highs in a strong uptrend. Structure > funding. Hold but monitor.",
        "reasoning": {"structure_analysis": "Strong uptrend with higher highs. Stop $2,700 below (3.5% buffer). Structure intact.", "data_assessment": "Crowded longs are a risk factor for a long squeeze, but the actual price action shows no sign of reversal. Funding is a lagging indicator.", "risk_factors": "Crowded funding increases risk of sharp pullback. But the stop at $73,800 protects against that. If squeeze happens, stop handles it.", "exit_logic": "Funding crowding is noted but price structure is intact. MCF hierarchy: no structural break = no exit. The stop is the safety net."},
        "execution": {"exit_pct": 0, "stop_price": 73800, "order_type": "LIMIT"}}},
    {"user": "Position: SHORT BTC | Entry: $88,000 | Current: $87,500 | Stop: $89,200 | Leverage: 3x | R-Multiple: +0.42 | Duration: 3h | Market: Funding -0.008% (mild negative). CVD flat. OI flat. Price consolidating.",
     "assistant": {"action": "HOLD", "urgency": "LOW", "confidence": 0.75,
        "reason": "SHORT at +0.42R with mildly negative funding (-0.008%). This is barely crowded — not a concern. Stop intact at $89,200 ($1,700 buffer). Position profitable. No structural breaks. Simple hold.",
        "reasoning": {"structure_analysis": "Stop intact with 1.9% buffer. Price consolidating, no breakout.", "data_assessment": "Mild negative funding is irrelevant at -0.008%. CVD and OI flat. No squeeze indicators.", "risk_factors": "-0.008% funding is background noise. Not enough to warrant any concern or position change.", "exit_logic": "No exit signals. Funding is within normal range. Hold."},
        "execution": {"exit_pct": 0, "stop_price": 89200, "order_type": "LIMIT"}}},
]

for ex in funding_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))


# ============================================================
# CATEGORY 6: Stop Already Breached = EXIT Always (6 examples)
# When stop is clearly breached, EXIT regardless of other signals
# Price might recover but the rule is unconditional
# ============================================================

breached_exit_examples = [
    {"user": "Position: LONG BTC | Entry: $89,000 | Current: $86,500 | Stop: $87,500 | Leverage: 5x | R-Multiple: -1.67 | Duration: 8h | Market: BTC dropped but bouncing. RSI oversold at 25. Hammer forming on 1h.",
     "assistant": {"action": "EXIT_FULL", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "LONG stop at $87,500 is BREACHED — current price $86,500 is $1,000 BELOW the stop. For a LONG, price below stop = breached. EXIT IMMEDIATELY regardless of the oversold bounce setup. The stop is the unconditional exit rule.",
        "reasoning": {"structure_analysis": "LONG stop breached: $86,500 < $87,500. Past stop by $1,000 (1.14%).", "data_assessment": "RSI oversold and hammer forming suggest potential bounce. But these are IRRELEVANT when the hard stop has been breached.", "risk_factors": "Oversold bounces can fail and lead to further downside. At 5x leverage, holding past a breached stop is gambling.", "exit_logic": "MCF hierarchy: Hard Stop is the FIRST and most important exit level. When breached, all other signals are OVERRIDDEN. RSI, candle patterns, momentum — none of these matter once the hard stop is violated. EXIT."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    {"user": "Position: SHORT BTC | Entry: $82,000 | Current: $84,500 | Stop: $83,500 | Leverage: 10x | R-Multiple: -1.67 | Duration: 6h | Market: BTC spiked on news but starting to reverse. Bearish divergence forming on 15m.",
     "assistant": {"action": "EXIT_FULL", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "SHORT stop at $83,500 is BREACHED — current price $84,500 is $1,000 ABOVE the stop. For a SHORT, price above stop = breached. EXIT IMMEDIATELY even though bearish divergence suggests reversal. The stop was hit. The trade is over.",
        "reasoning": {"structure_analysis": "SHORT stop breached: $84,500 > $83,500. Past stop by $1,000 (1.2%).", "data_assessment": "Bearish divergence on 15m is tempting — it suggests the spike may reverse. But the hard stop was violated. The divergence might work or it might not. The stop is unconditional.", "risk_factors": "At 10x, holding past a breached stop means -10% for every additional 1% adverse move. The potential reversal is NOT worth the downside risk.", "exit_logic": "Hard stop breached. MCF hierarchy is absolute at this level. Bearish divergence is a SECONDARY signal that cannot override a BREACHED HARD STOP. Exit now, potentially re-enter if divergence confirms."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    {"user": "Position: LONG ETH | Entry: $3,200 | Current: $3,050 | Stop: $3,100 | Leverage: 3x | R-Multiple: -2.0 | Duration: 12h | Market: ETH oversold. Funding flipped negative (shorts crowded). Bounce expected.",
     "assistant": {"action": "EXIT_FULL", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "LONG stop at $3,100 is BREACHED — current price $3,050 is $50 BELOW the stop. For a LONG, price below stop = breached. EXIT regardless of oversold conditions or expected bounce. The hard stop is an unconditional exit rule.",
        "reasoning": {"structure_analysis": "LONG stop breached: $3,050 < $3,100. Past stop by $50 (1.6%).", "data_assessment": "Oversold with crowded shorts = bounce likely. But the HARD STOP has been violated. Expected bounces are not guaranteed.", "risk_factors": "ETH could bounce — or it could continue to $2,800. At 3x leverage, the risk of holding past a breached stop outweighs the bounce potential.", "exit_logic": "Hard stop breached. EXIT. If you believe in the bounce, re-enter with a fresh position and fresh stop. Don't hold a breached stop hoping for a bounce."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    {"user": "Position: SHORT SOL | Entry: $155 | Current: $162 | Stop: $160 | Leverage: 5x | R-Multiple: -1.4 | Duration: 4h | Market: SOL pumped but hitting resistance at $163. Volume declining. Could reject here.",
     "assistant": {"action": "EXIT_FULL", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "SHORT stop at $160 is BREACHED — current price $162 is $2 ABOVE the stop. For a SHORT, price above stop = breached. EXIT immediately. Yes, resistance at $163 might cause rejection, but the hard stop has been violated. The trade is invalid.",
        "reasoning": {"structure_analysis": "SHORT stop breached: $162 > $160. Past stop by $2 (1.25%).", "data_assessment": "Resistance at $163 with declining volume is a potential reversal setup. But this is irrelevant — the stop is breached.", "risk_factors": "If SOL breaks $163 resistance, the next leg up could be to $170+. At 5x, that's another 25%+ leveraged loss.", "exit_logic": "Hard stop breached. MCF hierarchy: EXIT unconditionally. Resistance levels and volume patterns are secondary analysis. They cannot override a violated hard stop."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    {"user": "Position: LONG BTC | Entry: $70,000 | Current: $67,500 | Stop: $68,800 | Leverage: 20x | R-Multiple: -1.67 | Duration: 3h | Market: BTC flash crashed but already recovering. Now at $67,500 bouncing from $66,800 low. V-shape recovery forming.",
     "assistant": {"action": "EXIT_FULL", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "LONG stop at $68,800 is BREACHED — current price $67,500 is $1,300 BELOW the stop. At 20x leverage, this is a -37.1% leveraged loss ALREADY. EXIT immediately. The V-shape recovery is irrelevant — the hard stop was violated at 20x leverage. This is a survival situation.",
        "reasoning": {"structure_analysis": "LONG stop breached: $67,500 < $68,800. Past stop by $1,300 (1.89%). At 20x = -37.8% leveraged loss.", "data_assessment": "V-shape recovery from $66,800 is encouraging. But at 20x past a breached stop, the downside of the V failing is TOTAL LOSS.", "risk_factors": "At 20x, already -37.8% leveraged. If the V-shape fails and BTC retests $66,800, that's -46% leveraged. Liquidation is at ~$66,500. Recovery or not, EXIT.", "exit_logic": "Hard stop breached at 20x leverage. This is the most critical exit scenario possible. No signal — RSI, V-shape, support, resistance — can override this. EXIT IMMEDIATELY."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
    {"user": "Position: SHORT BTC | Entry: $85,000 | Current: $87,000 | Stop: $86,200 | Leverage: 3x | R-Multiple: -0.95 | Duration: 10h | Market: BTC squeezed shorts but funding now -0.03% (extremely crowded). Historically, these squeezes reverse within 24h.",
     "assistant": {"action": "EXIT_FULL", "urgency": "CRITICAL", "confidence": 0.95,
        "reason": "SHORT stop at $86,200 is BREACHED — current price $87,000 is $800 ABOVE the stop. For a SHORT, price above stop = breached. EXIT immediately. Historical squeeze reversal patterns are irrelevant — the hard stop has been violated.",
        "reasoning": {"structure_analysis": "SHORT stop breached: $87,000 > $86,200. Past stop by $800 (0.93%).", "data_assessment": "Extremely crowded funding and historical squeeze reversal pattern suggest the squeeze may reverse. These are compelling arguments for re-entry — but NOT for holding a breached stop.", "risk_factors": "Squeeze reversals are probabilistic, not guaranteed. At 3x with a breached stop, the correct action is to exit, wait for reversal confirmation, and re-enter with a new stop.", "exit_logic": "Hard stop breached. EXIT. If the squeeze reverses, re-enter fresh. Never hold a position past its hard stop based on historical pattern expectations. The stop is the unconditional exit."},
        "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}}},
]

for ex in breached_exit_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))


# ============================================================
# Write output
# ============================================================
output_file = "bastion_risk_v5_reinforcement.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Generated {len(examples)} v5 training examples")
print(f"Saved to {output_file}")
print()
print("Breakdown:")

cats = {
    "Cat 1 (LONG stop logic)": 10,
    "Cat 2 (SHORT stop logic)": 10,
    "Cat 3 (TP1 proximity + leverage)": 8,
    "Cat 4 (No-stop leverage tiers)": 20,
    "Cat 5 (Funding ≠ exit)": 6,
    "Cat 6 (Breached stop = EXIT always)": 6,
}
total = 0
for name, count in cats.items():
    print(f"  {name}: {count} examples")
    total += count
print(f"  TOTAL: {total} examples")
