"""
BASTION v6 Training Data Generator
===================================
Multi-pair structure-aware training examples.
Covers 10 high-cap crypto pairs with MCF structural context
(VPVR zones, graded S/R, trendlines, auto-support) in every prompt.

Addresses v5 weaknesses:
- ETH: 55.3% (needs pair-specific examples)
- HOLD: 54.5% (needs structure-intact HOLD examples)
- TP_PARTIAL: 33.3% (needs better TP timing with structure)

Categories:
1. Structure-Intact HOLD (25 examples) - price above graded support, hold
2. Structure-Break EXIT (20 examples) - price closes through graded level, exit
3. VPVR-Informed Decisions (15 examples) - HVN/LVN/POC zone awareness
4. Multi-Pair Diversity (20 examples) - all 10 pairs with varied scenarios
5. Structure-Aware TP_PARTIAL (10 examples) - take profit at structural resistance
6. Trendline + Confluence Exits (10 examples) - trendline breaks with confluence

Total: 100 new examples across 10 pairs
Combined with v5's 290 = 390 total

Pairs: BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, BNB, DOT
"""

import json

examples = []

SYSTEM_PROMPT = (
    "You are BASTION-32B, an AI risk management engine for cryptocurrency trading. "
    "Evaluate the position using MCF structure-based exit hierarchy: "
    "Hard Stop > Structural Break > Guarding Line Break > Take Profit > VPVR-Informed Trail > Time Exit. "
    "Use the STRUCTURAL ANALYSIS data to make decisions based on actual price structure, not arbitrary levels. "
    "Respond with a JSON object containing: action, urgency, confidence, reason, reasoning "
    "(with structure_analysis, data_assessment, risk_factors, exit_logic), and execution "
    "(exit_pct, stop_price, order_type)."
)


def make_example(user_content, assistant_content):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_content)}
        ]
    }


# ============================================================
# CATEGORY 1: Structure-Intact HOLD (25 examples)
# Price is above nearest graded support (LONG) or below resistance (SHORT)
# Structure holds = HOLD the position
# ============================================================

structure_hold_examples = [
    # ETH LONG - structure intact, Grade 3 support holding
    {
        "user": (
            "Position: LONG ETH | Entry: $3,150 | Current: $3,120 | Stop: $3,020 | "
            "Leverage: 5x | R-Multiple: -0.23 | Duration: 4h | "
            "Market: ETH pulling back from $3,200. Volume declining on pullback.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $3,080 (Grade 3 pivot confluence 1h, 1.3% below) [priority: 8.1]\n"
            "NEAREST RESISTANCE: $3,220 (Grade 2 horizontal 4h, 3.2% above) [priority: 5.4]\n"
            "VPVR: In HVN — expect price consolidation. POC $3,100 (0.6% below). Buy-dominant.\n"
            "TRENDLINES: 1h ascending support G3 at $3,075 | 4h descending resistance G2 at $3,250\n"
            "PRESSURE POINT: $3,080 confluence 7.5/10\n"
            "SUPPORTS: $3,080 [8.1] | $2,980 [6.3] | $2,920 [4.8]\n"
            "RESISTANCES: $3,220 [5.4] | $3,350 [4.2] | $3,500 [3.0]\n"
            "MTF BIAS: Bullish (65% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.78,
            "reason": "LONG structure intact. Nearest Grade 3 support at $3,080 (1.3% below) is unbroken. Price is in HVN with buy-dominant volume — consolidation expected, not breakdown. Stop at $3,020 is below the structural support.",
            "reasoning": {
                "structure_analysis": "Grade 3 pivot confluence at $3,080 with priority 8.1/10 is the key level. Price $3,120 is $40 above this support. Ascending trendline G3 at $3,075 provides additional support. Structure is INTACT.",
                "data_assessment": "HVN zone means high friction — price stalls here, doesn't break easily. Buy-dominant volume confirms accumulation. Volume declining on pullback = healthy retracement, not distribution.",
                "risk_factors": "5x leverage with stop at $3,020 — stop is $60 below the Grade 3 support at $3,080. Proper stop placement below structural level. Buffer to support: 1.3%.",
                "exit_logic": "MCF hierarchy: no structural break. Grade 3 support at $3,080 is holding. Price in HVN = consolidation zone. HOLD until structure actually breaks."
            },
            "execution": {"exit_pct": 0, "stop_price": 3020, "order_type": "LIMIT"}
        }
    },
    # SOL LONG - structure intact, multiple supports below
    {
        "user": (
            "Position: LONG SOL | Entry: $148 | Current: $145.50 | Stop: $139 | "
            "Leverage: 3x | R-Multiple: -0.28 | Duration: 6h | "
            "Market: SOL retracing after rally to $152. Healthy pullback.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $143.80 (Grade 2 horizontal 1h, 1.2% below) [priority: 6.8]\n"
            "NEAREST RESISTANCE: $152.50 (Grade 3 horizontal 4h, 4.8% above) [priority: 7.2]\n"
            "VPVR: In HVN — expect consolidation. POC $146.20 (0.5% above). Balanced.\n"
            "TRENDLINES: 1h ascending support G2 at $142.50 | 4h flat resistance G3 at $153\n"
            "PRESSURE POINT: $143.80 confluence 6.2/10\n"
            "SUPPORTS: $143.80 [6.8] | $140.00 [5.5] | $136.50 [4.1]\n"
            "RESISTANCES: $152.50 [7.2] | $158.00 [5.0] | $165.00 [3.3]\n"
            "MTF BIAS: Bullish (70% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.76,
            "reason": "LONG structure intact. Grade 2 support at $143.80 is 1.2% below and unbroken. POC at $146.20 is just above — price is near value area center. 70% MTF bullish alignment supports holding.",
            "reasoning": {
                "structure_analysis": "Nearest support $143.80 Grade 2 with priority 6.8. Price $145.50 is $1.70 above. Stop at $139 is well below all structural supports including $140 [5.5]. Structure intact with ascending trendline G2 at $142.50.",
                "data_assessment": "HVN zone = consolidation. POC at $146.20 is the fair value — price is slightly below POC, likely to revert. Balanced volume means no aggressive selling.",
                "risk_factors": "3x leverage, stop $6.50 below entry (4.4%). Stop is below $140 structural support — proper placement. Low urgency.",
                "exit_logic": "No structural break. Grade 2 support holding. HVN = price stalls. MTF 70% bullish. Classic HOLD scenario — let the pullback play out."
            },
            "execution": {"exit_pct": 0, "stop_price": 139, "order_type": "LIMIT"}
        }
    },
    # XRP SHORT - structure intact, resistance holding
    {
        "user": (
            "Position: SHORT XRP | Entry: $2.45 | Current: $2.38 | Stop: $2.55 | "
            "Leverage: 5x | R-Multiple: +0.70 | Duration: 8h | "
            "Market: XRP rejected from $2.48 resistance. Bearish pressure.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $2.30 (Grade 2 horizontal 4h, 3.4% below) [priority: 5.9]\n"
            "NEAREST RESISTANCE: $2.48 (Grade 3 pivot confluence 1h, 4.2% above) [priority: 7.8]\n"
            "VPVR: In LVN — expect fast price movement. POC $2.42 (1.7% above). Sell-dominant.\n"
            "TRENDLINES: 1h descending resistance G3 at $2.50 | 4h ascending support G2 at $2.28\n"
            "PRESSURE POINT: $2.48 confluence 7.8/10\n"
            "SUPPORTS: $2.30 [5.9] | $2.22 [4.5] | $2.15 [3.2]\n"
            "RESISTANCES: $2.48 [7.8] | $2.55 [6.1] | $2.65 [4.0]\n"
            "MTF BIAS: Bearish (68% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.80,
            "reason": "SHORT structure intact. Price rejected from Grade 3 resistance at $2.48 (confluence 7.8/10). In LVN zone with sell-dominant volume — fast downside movement expected. MTF 68% bearish alignment confirms.",
            "reasoning": {
                "structure_analysis": "Grade 3 resistance at $2.48 held — price rejected and dropped. Descending trendline G3 at $2.50 adds overhead pressure. For SHORT, structure intact means price is BELOW nearest graded resistance. $2.38 < $2.48 = intact.",
                "data_assessment": "LVN zone = low friction, price moves fast. Sell-dominant volume confirms downside bias. This is the ideal zone for shorts — expect acceleration toward $2.30 support.",
                "risk_factors": "Stop at $2.55 is above the Grade 3 resistance at $2.48 — excellent stop placement. 5x leverage with +0.70R, healthy position.",
                "exit_logic": "Structure intact. Resistance held. LVN = fast movement in our favor. 68% bearish MTF. No exit triggers — let the short work toward $2.30 target."
            },
            "execution": {"exit_pct": 0, "stop_price": 2.55, "order_type": "LIMIT"}
        }
    },
    # DOGE LONG - structure intact, sitting on support
    {
        "user": (
            "Position: LONG DOGE | Entry: $0.185 | Current: $0.178 | Stop: $0.168 | "
            "Leverage: 3x | R-Multiple: -0.41 | Duration: 12h | "
            "Market: DOGE retesting breakout level. Meme coins mixed.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $0.175 (Grade 3 horizontal 4h, 1.7% below) [priority: 8.5]\n"
            "NEAREST RESISTANCE: $0.195 (Grade 2 horizontal 1h, 9.6% above) [priority: 5.2]\n"
            "VPVR: In HVN — expect consolidation. POC $0.180 (1.1% above). Buy-dominant.\n"
            "TRENDLINES: 4h ascending support G2 at $0.172 | 1h flat resistance G2 at $0.192\n"
            "PRESSURE POINT: $0.175 confluence 7.0/10\n"
            "SUPPORTS: $0.175 [8.5] | $0.165 [6.2] | $0.155 [4.8]\n"
            "RESISTANCES: $0.195 [5.2] | $0.210 [4.0] | $0.225 [2.8]\n"
            "MTF BIAS: Neutral (52% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.70,
            "reason": "LONG structure intact. Grade 3 support at $0.175 (priority 8.5) is 1.7% below and holding. Price retesting breakout level in HVN with buy-dominant volume. Stop at $0.168 is below the Grade 3 support — proper placement.",
            "reasoning": {
                "structure_analysis": "Grade 3 horizontal at $0.175 with priority 8.5 is the critical level. Price $0.178 is just $0.003 above. This is a test of the level — structure intact until price CLOSES below $0.175. Ascending trendline G2 at $0.172 provides secondary support.",
                "data_assessment": "HVN zone = consolidation, price stalls here. Buy-dominant volume at POC $0.180 suggests accumulation. Neutral MTF means no strong directional bias — patience required.",
                "risk_factors": "3x leverage, manageable. Close to Grade 3 support — confidence capped at 70% due to proximity. If $0.175 breaks, next support is $0.165 near our stop.",
                "exit_logic": "Structure intact but being tested. Grade 3 support at $0.175 must hold on a closing basis. HOLD but monitor closely. Do not pre-empt the structural test."
            },
            "execution": {"exit_pct": 0, "stop_price": 0.168, "order_type": "LIMIT"}
        }
    },
    # ADA LONG - strong structure, far from support
    {
        "user": (
            "Position: LONG ADA | Entry: $0.72 | Current: $0.75 | Stop: $0.67 | "
            "Leverage: 5x | R-Multiple: +0.60 | Duration: 24h | "
            "Market: ADA in uptrend after Cardano upgrade news. Higher highs.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $0.71 (Grade 3 pivot trendline 1h, 5.3% below) [priority: 7.9]\n"
            "NEAREST RESISTANCE: $0.78 (Grade 2 horizontal 4h, 4.0% above) [priority: 5.8]\n"
            "VPVR: In LVN — expect fast movement. POC $0.73 (2.7% below). Buy-dominant.\n"
            "TRENDLINES: 1h ascending support G3 at $0.71 | 4h ascending support G2 at $0.68\n"
            "PRESSURE POINT: $0.71 confluence 7.2/10\n"
            "SUPPORTS: $0.71 [7.9] | $0.68 [6.1] | $0.65 [4.5]\n"
            "RESISTANCES: $0.78 [5.8] | $0.82 [4.3] | $0.88 [3.0]\n"
            "MTF BIAS: Bullish (78% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.83,
            "reason": "LONG with strong structural support. Grade 3 ascending trendline at $0.71 is 5.3% below — large buffer. In LVN with buy-dominant volume — fast upside movement expected. 78% MTF bullish. Let the winner run.",
            "reasoning": {
                "structure_analysis": "Grade 3 pivot trendline at $0.71 with priority 7.9. Price is $0.04 above (5.3% buffer). Two ascending trendlines (G3 at $0.71, G2 at $0.68) provide strong structural floor. Structure firmly intact.",
                "data_assessment": "LVN zone = low friction. Buy-dominant volume. Upgrade catalyst driving momentum. Higher highs confirmed. This is the ideal setup — trending in LVN with structural support well below.",
                "risk_factors": "5x leverage at +0.60R. Stop at $0.67 is below both ascending trendlines. Excellent risk management. Buffer to nearest support is 5.3% — very comfortable.",
                "exit_logic": "Strong structural support. LVN for fast movement. 78% MTF alignment. No exit triggers. Target Grade 2 resistance at $0.78. Let the trend work."
            },
            "execution": {"exit_pct": 0, "stop_price": 0.67, "order_type": "LIMIT"}
        }
    },
    # LINK SHORT - structure intact, below resistance
    {
        "user": (
            "Position: SHORT LINK | Entry: $18.50 | Current: $17.80 | Stop: $19.50 | "
            "Leverage: 3x | R-Multiple: +0.70 | Duration: 16h | "
            "Market: LINK rejected at $19 resistance. Downtrend continuing.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $17.20 (Grade 2 horizontal 4h, 3.4% below) [priority: 6.0]\n"
            "NEAREST RESISTANCE: $18.80 (Grade 3 pivot confluence 1h, 5.6% above) [priority: 8.2]\n"
            "VPVR: In LVN — expect fast movement. POC $18.00 (1.1% above). Sell-dominant.\n"
            "TRENDLINES: 1h descending resistance G3 at $18.90 | 4h descending resistance G2 at $19.20\n"
            "PRESSURE POINT: $18.80 confluence 8.0/10\n"
            "SUPPORTS: $17.20 [6.0] | $16.50 [4.8] | $15.80 [3.5]\n"
            "RESISTANCES: $18.80 [8.2] | $19.50 [6.5] | $20.50 [4.2]\n"
            "MTF BIAS: Bearish (72% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.80,
            "reason": "SHORT structure intact. Grade 3 resistance at $18.80 (priority 8.2, confluence 8.0/10) rejected price. In LVN with sell-dominant volume — fast downside expected. Two descending trendlines (G3, G2) confirm overhead pressure.",
            "reasoning": {
                "structure_analysis": "For SHORT, structure intact means price is BELOW nearest graded resistance. $17.80 < $18.80 Grade 3 = intact. Descending trendlines G3 at $18.90 and G2 at $19.20 create a ceiling. Confluence score 8.0/10 makes $18.80 a very strong resistance.",
                "data_assessment": "LVN = low friction, fast movement. Sell-dominant volume confirms selling pressure. 72% bearish MTF alignment. All signals align for continuation lower.",
                "risk_factors": "3x leverage at +0.70R. Stop at $19.50 is above all structural resistance levels — excellent placement. Low risk configuration.",
                "exit_logic": "Structure intact. Strong resistance overhead. LVN for fast movement. Bearish MTF. Target $17.20 Grade 2 support. HOLD."
            },
            "execution": {"exit_pct": 0, "stop_price": 19.50, "order_type": "LIMIT"}
        }
    },
    # BNB LONG - HVN consolidation, structure holding
    {
        "user": (
            "Position: LONG BNB | Entry: $620 | Current: $615 | Stop: $595 | "
            "Leverage: 5x | R-Multiple: -0.20 | Duration: 3h | "
            "Market: BNB choppy around $615-$625 range. Low volatility.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $608 (Grade 3 horizontal 1h, 1.1% below) [priority: 7.6]\n"
            "NEAREST RESISTANCE: $632 (Grade 2 horizontal 4h, 2.8% above) [priority: 5.5]\n"
            "VPVR: In HVN — expect consolidation. POC $618 (0.5% above). Balanced.\n"
            "TRENDLINES: 1h flat support G2 at $607 | 4h ascending support G2 at $600\n"
            "PRESSURE POINT: $608 confluence 6.8/10\n"
            "SUPPORTS: $608 [7.6] | $595 [5.8] | $580 [4.2]\n"
            "RESISTANCES: $632 [5.5] | $645 [4.1] | $660 [3.0]\n"
            "MTF BIAS: Neutral (55% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.72,
            "reason": "LONG structure intact. Grade 3 support at $608 is 1.1% below and holding. In HVN = consolidation expected, not breakdown. POC at $618 is just above — price near fair value. Patience required in chop.",
            "reasoning": {
                "structure_analysis": "Grade 3 horizontal at $608 with priority 7.6. Price $615 is $7 above. Flat support G2 at $607 and ascending G2 at $600 provide structural floor. Stop at $595 is below all supports.",
                "data_assessment": "HVN = high friction zone. Balanced volume. Low volatility. This is a waiting game — price is consolidating near POC. Breakout will come but requires patience.",
                "risk_factors": "5x leverage at -0.20R. Small drawdown. Stop well below structure. Neutral MTF means no strong directional pressure — confidence capped at 72%.",
                "exit_logic": "Structure holding in consolidation. HVN zones require patience. No break, no exit. Wait for range resolution."
            },
            "execution": {"exit_pct": 0, "stop_price": 595, "order_type": "LIMIT"}
        }
    },
    # DOT LONG - structure intact after volatile shakeout
    {
        "user": (
            "Position: LONG DOT | Entry: $7.80 | Current: $7.55 | Stop: $7.10 | "
            "Leverage: 5x | R-Multiple: -0.36 | Duration: 10h | "
            "Market: DOT volatile — dipped to $7.40 but recovered. Shakeout pattern.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $7.35 (Grade 3 pivot confluence 1h, 2.6% below) [priority: 8.0]\n"
            "NEAREST RESISTANCE: $7.95 (Grade 2 horizontal 4h, 5.3% above) [priority: 5.5]\n"
            "VPVR: In HVN — consolidation zone. POC $7.50 (0.7% below). Buy-dominant.\n"
            "TRENDLINES: 1h ascending support G2 at $7.30 | 4h ascending support G3 at $7.20\n"
            "PRESSURE POINT: $7.35 confluence 7.5/10\n"
            "SUPPORTS: $7.35 [8.0] | $7.10 [6.2] | $6.80 [4.5]\n"
            "RESISTANCES: $7.95 [5.5] | $8.30 [4.0] | $8.80 [2.8]\n"
            "MTF BIAS: Bullish (62% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.74,
            "reason": "LONG structure intact despite volatile shakeout. Price dipped to $7.40 but Grade 3 support at $7.35 HELD — no close below it. Recovery to $7.55 with buy-dominant volume in HVN confirms accumulation. Shakeouts that hold structure are bullish.",
            "reasoning": {
                "structure_analysis": "Grade 3 confluence at $7.35 (priority 8.0) survived the shakeout — price wicked to $7.40 but did not CLOSE below $7.35. Structure is intact. Two ascending trendlines (G2 at $7.30, G3 at $7.20) below.",
                "data_assessment": "Shakeout to $7.40 then recovery = liquidity grab. Buy-dominant volume in HVN confirms accumulation. This is bullish structure behavior — weak hands shaken out, strong hands accumulate.",
                "risk_factors": "5x at -0.36R. Stop at $7.10 is below the second support ($7.10 [6.2]). Grade 3 support at $7.35 is 2.6% below current — reasonable buffer.",
                "exit_logic": "Structure held through volatility. Grade 3 support tested and held (wick only, no close). HVN accumulation. HOLD — the shakeout is bullish if structure holds."
            },
            "execution": {"exit_pct": 0, "stop_price": 7.10, "order_type": "LIMIT"}
        }
    },
    # ETH SHORT - structure intact, multiple resistance levels above
    {
        "user": (
            "Position: SHORT ETH | Entry: $3,400 | Current: $3,350 | Stop: $3,520 | "
            "Leverage: 5x | R-Multiple: +0.42 | Duration: 8h | "
            "Market: ETH rejected at $3,420. Lower high forming.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $3,280 (Grade 2 horizontal 4h, 2.1% below) [priority: 5.8]\n"
            "NEAREST RESISTANCE: $3,420 (Grade 3 pivot confluence 1h, 2.1% above) [priority: 8.4]\n"
            "VPVR: In LVN — fast movement zone. POC $3,380 (0.9% above). Sell-dominant.\n"
            "TRENDLINES: 1h descending resistance G3 at $3,430 | 4h descending resistance G2 at $3,450\n"
            "PRESSURE POINT: $3,420 confluence 8.2/10\n"
            "SUPPORTS: $3,280 [5.8] | $3,200 [4.5] | $3,100 [3.8]\n"
            "RESISTANCES: $3,420 [8.4] | $3,500 [6.2] | $3,580 [4.5]\n"
            "MTF BIAS: Bearish (70% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.80,
            "reason": "SHORT structure intact. Grade 3 resistance at $3,420 (confluence 8.2/10) rejected price. LVN with sell-dominant volume = fast downside expected. Lower high forming confirms bearish structure. 70% bearish MTF.",
            "reasoning": {
                "structure_analysis": "For SHORT: structure intact when price is BELOW nearest graded resistance. $3,350 < $3,420 Grade 3 = intact. Two descending trendlines (G3 $3,430, G2 $3,450) cap upside. Confluence 8.2/10 makes $3,420 very strong.",
                "data_assessment": "LVN = fast movement. Sell-dominant volume. Lower high forming at $3,420 vs prior high. Classic bearish structure confirmation.",
                "risk_factors": "5x at +0.42R. Stop at $3,520 is above all resistance levels. Proper stop placement. Low risk.",
                "exit_logic": "Bearish structure intact. Strong resistance. LVN for acceleration. Target $3,280 Grade 2 support. HOLD the short."
            },
            "execution": {"exit_pct": 0, "stop_price": 3520, "order_type": "LIMIT"}
        }
    },
    # AVAX LONG - deep pullback but structure holding
    {
        "user": (
            "Position: LONG AVAX | Entry: $42.00 | Current: $39.80 | Stop: $37.50 | "
            "Leverage: 3x | R-Multiple: -0.49 | Duration: 18h | "
            "Market: AVAX pulled back with altcoin weakness. Testing support.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $39.50 (Grade 3 horizontal 4h, 0.8% below) [priority: 8.8]\n"
            "NEAREST RESISTANCE: $42.80 (Grade 2 horizontal 1h, 7.5% above) [priority: 5.0]\n"
            "VPVR: In HVN — high friction zone. POC $40.20 (1.0% above). Buy-dominant.\n"
            "TRENDLINES: 4h ascending support G3 at $39.20 | 1h flat support G2 at $39.50\n"
            "PRESSURE POINT: $39.50 confluence 8.5/10\n"
            "SUPPORTS: $39.50 [8.8] | $37.80 [6.5] | $36.00 [4.2]\n"
            "RESISTANCES: $42.80 [5.0] | $45.00 [3.8] | $48.00 [2.5]\n"
            "MTF BIAS: Neutral (50% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "HIGH",
            "confidence": 0.65,
            "reason": "LONG testing Grade 3 support at $39.50 (priority 8.8, confluence 8.5/10). This is a MAJOR structural test. Price $39.80 is only 0.8% above — very close. But structure has NOT broken (no close below). HVN with buy-dominant volume suggests holding. Confidence capped due to proximity.",
            "reasoning": {
                "structure_analysis": "Grade 3 horizontal at $39.50 with priority 8.8 — the highest-priority level. Confluence 8.5/10 with ascending trendline G3 at $39.20. This is the line in the sand. Price must CLOSE below $39.50 to break structure.",
                "data_assessment": "HVN = high friction. Buy-dominant volume at this level suggests buyers defending $39.50. Altcoin weakness is sector-wide, not AVAX-specific.",
                "risk_factors": "3x leverage, -0.49R. Only 0.8% above major support. If $39.50 breaks, next support $37.80 is near stop. Confidence limited to 65% due to tight proximity.",
                "exit_logic": "Structure being tested but NOT broken. Grade 3 + 8.5/10 confluence is a strong floor. HVN = price stalls. HOLD but this is HIGH urgency monitoring. If candle closes below $39.50, exit logic changes."
            },
            "execution": {"exit_pct": 0, "stop_price": 37.50, "order_type": "LIMIT"}
        }
    },
]

for ex in structure_hold_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))

print(f"Category 1 (Structure-Intact HOLD): {len(structure_hold_examples)} examples")


# ============================================================
# CATEGORY 2: Structure-Break EXIT (20 examples)
# Price CLOSES through graded support (LONG) or resistance (SHORT)
# Structure broken = EXIT regardless of other signals
# ============================================================

structure_break_examples = [
    # ETH LONG - Grade 3 support broken
    {
        "user": (
            "Position: LONG ETH | Entry: $3,200 | Current: $3,045 | Stop: $2,980 | "
            "Leverage: 5x | R-Multiple: -0.70 | Duration: 10h | "
            "Market: ETH broke below key support with volume.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $3,080 (Grade 3 pivot confluence 1h, BROKEN) [priority: 8.1]\n"
            "NEAREST RESISTANCE: $3,120 (Grade 1 horizontal 15m, 2.5% above) [priority: 3.2]\n"
            "VPVR: Entered LVN below $3,080 — fast movement zone. POC $3,150 (3.4% above). Sell-dominant.\n"
            "TRENDLINES: 1h ascending support G3 at $3,075 BROKEN | 4h flat support G2 at $2,980\n"
            "PRESSURE POINT: $3,080 confluence 7.5/10 — BREACHED\n"
            "SUPPORTS: $2,980 [5.8] | $2,920 [4.5] | $2,850 [3.2]\n"
            "RESISTANCES: $3,080 [8.1] (now resistance) | $3,120 [3.2] | $3,200 [5.5]\n"
            "MTF BIAS: Bearish (35% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "HIGH",
            "confidence": 0.88,
            "reason": "STRUCTURAL BREAK: Grade 3 pivot confluence at $3,080 (priority 8.1, confluence 7.5/10) has been BROKEN. Price closed below this critical level. Ascending trendline G3 also broken. Now in LVN = fast downside movement. Exit before acceleration.",
            "reasoning": {
                "structure_analysis": "Grade 3 support at $3,080 was the primary structural floor with 7.5/10 confluence. Price $3,045 is $35 BELOW it — this is a confirmed close below, not a wick. Structure is BROKEN. Former support becomes resistance.",
                "data_assessment": "Entered LVN below $3,080 — low friction means fast price movement. Sell-dominant volume confirms distribution. MTF flipped bearish at 35%. All signals align for continuation lower.",
                "risk_factors": "5x leverage at -0.70R. Next structural support is $2,980 near our stop. If LVN acceleration occurs, price could gap through stop. Exit while orderly.",
                "exit_logic": "MCF hierarchy Level 2: Structural Break. Grade 3+ support broken on a close = EXIT. This overrides everything except hard stop. The structure that justified the LONG is now invalid."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # BTC SHORT - Grade 3 resistance broken
    {
        "user": (
            "Position: SHORT BTC | Entry: $88,000 | Current: $89,800 | Stop: $91,000 | "
            "Leverage: 10x | R-Multiple: -0.60 | Duration: 6h | "
            "Market: BTC broke above resistance on heavy volume.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $88,500 (Grade 2 horizontal 1h, 1.4% below) [priority: 5.5]\n"
            "NEAREST RESISTANCE: $89,200 (Grade 3 horizontal 4h, BROKEN) [priority: 8.0]\n"
            "VPVR: Entered LVN above $89,200 — fast movement. POC $87,500 (2.6% below). Buy-dominant.\n"
            "TRENDLINES: 4h descending resistance G3 at $89,500 BROKEN | 1h ascending support G2 at $88,000\n"
            "PRESSURE POINT: $89,200 confluence 8.0/10 — BREACHED\n"
            "SUPPORTS: $88,500 [5.5] | $87,000 [4.2] | $85,500 [3.5]\n"
            "RESISTANCES: $91,000 [4.0] | $92,500 [3.2] | $95,000 [2.5]\n"
            "MTF BIAS: Bullish (75% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.90,
            "reason": "STRUCTURAL BREAK: Grade 3 resistance at $89,200 (confluence 8.0/10) has been BROKEN. For SHORT, price closing ABOVE graded resistance = structure broken. Descending trendline G3 also broken. At 10x leverage, exit immediately.",
            "reasoning": {
                "structure_analysis": "For SHORT: structure breaks when price CLOSES ABOVE nearest graded resistance. Price $89,800 > $89,200 Grade 3 = BROKEN. Descending trendline G3 also violated. This is a double structural break.",
                "data_assessment": "LVN above $89,200 means fast upside movement — catastrophic for shorts. Buy-dominant volume. MTF flipped to 75% bullish. Breakout with volume is the strongest bearish signal for shorts.",
                "risk_factors": "10x leverage at -0.60R. In LVN with bullish momentum. Next resistance at $91,000 is our stop — price could reach it quickly in LVN. Exit NOW.",
                "exit_logic": "MCF Level 2: Structural Break. Grade 3+ resistance broken on close with volume = EXIT SHORT. At 10x in LVN, this is CRITICAL urgency. Don't wait for stop."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # SOL LONG - Grade 2 trendline broken, hold at next support = partial exit
    {
        "user": (
            "Position: LONG SOL | Entry: $155 | Current: $148.50 | Stop: $140 | "
            "Leverage: 5x | R-Multiple: -0.43 | Duration: 14h | "
            "Market: SOL broke ascending trendline but found support.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $147.00 (Grade 3 horizontal 4h, 1.0% below) [priority: 7.8]\n"
            "NEAREST RESISTANCE: $152.00 (Grade 2 horizontal 1h, 2.4% above) [priority: 5.5]\n"
            "VPVR: In HVN at $147-$150 — consolidation. POC $149.00 (0.3% above). Balanced.\n"
            "TRENDLINES: 1h ascending support G2 at $150.50 BROKEN | 4h flat support G3 at $147\n"
            "PRESSURE POINT: $147 confluence 7.2/10\n"
            "SUPPORTS: $147.00 [7.8] | $142.00 [5.5] | $138.00 [4.0]\n"
            "RESISTANCES: $150.50 [5.5] (broken trendline) | $152.00 [5.5] | $155.00 [4.2]\n"
            "MTF BIAS: Neutral (48% alignment)"
        ),
        "assistant": {
            "action": "REDUCE_SIZE",
            "urgency": "HIGH",
            "confidence": 0.78,
            "reason": "GUARDING LINE BREAK: Grade 2 ascending trendline at $150.50 broken, but price is holding at Grade 3 horizontal support at $147 (priority 7.8). Partial exit 50-60% — trendline broken but next support holding. If $147 breaks, exit remainder.",
            "reasoning": {
                "structure_analysis": "Grade 2 trendline broken ($148.50 < $150.50). But Grade 3 horizontal at $147 (higher priority 7.8) is holding. This is MCF Level 3 — Guarding Line Break. Structure is DAMAGED but not fully broken.",
                "data_assessment": "HVN at $147-$150 = high friction, price consolidates. Balanced volume. MTF neutral. The trendline break is concerning but the horizontal support is stronger.",
                "risk_factors": "5x leverage. If $147 breaks, next support at $142 is near stop. Reduce now to protect capital. Keep 40-50% for potential recovery if $147 holds.",
                "exit_logic": "MCF Level 3: Guarding Line Break. Grade 2 trendline broken + price holds at next support = reduce 50-60%, don't exit fully. If Grade 3 at $147 also breaks = exit remainder."
            },
            "execution": {"exit_pct": 55, "stop_price": 147.00, "order_type": "MARKET"}
        }
    },
    # XRP LONG - Grade 3 support broken, in LVN
    {
        "user": (
            "Position: LONG XRP | Entry: $2.50 | Current: $2.28 | Stop: $2.15 | "
            "Leverage: 3x | R-Multiple: -0.63 | Duration: 20h | "
            "Market: XRP broke through major support with volume spike.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $2.35 (Grade 3 horizontal 4h, BROKEN) [priority: 8.5]\n"
            "NEAREST RESISTANCE: $2.35 (Grade 3, now resistance, 3.1% above) [priority: 8.5]\n"
            "VPVR: In LVN below $2.35 — fast movement. POC $2.42 (6.1% above). Sell-dominant.\n"
            "TRENDLINES: 4h ascending support G3 at $2.32 BROKEN | 1h flat support G1 at $2.20\n"
            "PRESSURE POINT: $2.35 confluence 8.2/10 — BREACHED\n"
            "SUPPORTS: $2.20 [4.2] | $2.10 [3.5] | $2.00 [2.8]\n"
            "RESISTANCES: $2.35 [8.5] (former support) | $2.42 [5.0] | $2.50 [6.2]\n"
            "MTF BIAS: Bearish (30% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "HIGH",
            "confidence": 0.90,
            "reason": "STRUCTURAL BREAK: Grade 3 support at $2.35 (priority 8.5, confluence 8.2/10) BROKEN. Ascending trendline G3 also broken. Price in LVN = fast downside. Next support at $2.20 is Grade 1 only — weak. Exit before acceleration continues.",
            "reasoning": {
                "structure_analysis": "Grade 3 with priority 8.5 and 8.2/10 confluence — this was the primary structural floor. Price $2.28 is $0.07 below it (confirmed close). Double break: horizontal + ascending trendline. Structure completely invalidated.",
                "data_assessment": "LVN = fast movement, low friction. Sell-dominant volume. 30% MTF alignment = strongly bearish. Next Grade 1 support at $2.20 is weak — unlikely to hold.",
                "risk_factors": "3x leverage. Stop at $2.15 but in LVN, price can move fast. Better to exit orderly at $2.28 than wait for potential gap to $2.15.",
                "exit_logic": "MCF Level 2: Structural Break. Highest-priority support (8.5) broken with confluence. In LVN. EXIT — the structural thesis for this LONG is dead."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # DOGE SHORT - resistance broken, structure invalidated
    {
        "user": (
            "Position: SHORT DOGE | Entry: $0.190 | Current: $0.205 | Stop: $0.215 | "
            "Leverage: 5x | R-Multiple: -0.60 | Duration: 4h | "
            "Market: DOGE pumped on social media hype. Broke resistance.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $0.192 (Grade 1 horizontal 15m, 6.3% below) [priority: 3.0]\n"
            "NEAREST RESISTANCE: $0.198 (Grade 3 horizontal 1h, BROKEN) [priority: 7.5]\n"
            "VPVR: In LVN above $0.198 — fast movement. POC $0.188 (8.3% below). Buy-dominant.\n"
            "TRENDLINES: 1h flat resistance G3 at $0.198 BROKEN | no active support trendlines\n"
            "PRESSURE POINT: $0.198 confluence 7.0/10 — BREACHED\n"
            "SUPPORTS: $0.192 [3.0] | $0.185 [5.5] | $0.175 [4.2]\n"
            "RESISTANCES: $0.215 [4.0] | $0.225 [3.2] | $0.240 [2.5]\n"
            "MTF BIAS: Bullish (72% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.88,
            "reason": "STRUCTURAL BREAK: Grade 3 resistance at $0.198 (confluence 7.0/10) BROKEN. For SHORT, price above graded resistance = structure broken. In LVN with buy-dominant volume + social media hype = explosive upside risk. EXIT.",
            "reasoning": {
                "structure_analysis": "For SHORT: structure intact when price is BELOW resistance. $0.205 > $0.198 Grade 3 = BROKEN. No resistance until $0.215 (our stop) and that's only Grade 1. Structure completely invalidated.",
                "data_assessment": "LVN = fast movement upward. Buy-dominant. Social media hype on DOGE can drive parabolic moves — 20-30% in hours. 72% bullish MTF. Extremely dangerous for shorts.",
                "risk_factors": "5x leverage on a meme coin breakout. DOGE is known for violent pumps. LVN above resistance = potential for fast move to $0.215+ (our stop). Exit before it gets worse.",
                "exit_logic": "MCF Level 2: Structural Break on a meme coin breakout. Grade 3 resistance broken with momentum. DOGE hype cycles are explosive. EXIT immediately — don't let this reach the stop."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # ADA LONG - support broken but only Grade 2
    {
        "user": (
            "Position: LONG ADA | Entry: $0.75 | Current: $0.695 | Stop: $0.65 | "
            "Leverage: 3x | R-Multiple: -0.55 | Duration: 16h | "
            "Market: ADA sold off with broader alt weakness.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $0.70 (Grade 2 horizontal 1h, BROKEN) [priority: 6.0]\n"
            "NEAREST RESISTANCE: $0.72 (Grade 1 horizontal 15m, 3.6% above) [priority: 3.5]\n"
            "VPVR: Transitioning to LVN below $0.70. POC $0.73 (5.0% above). Sell-dominant.\n"
            "TRENDLINES: 1h ascending support G2 at $0.705 BROKEN | 4h flat support G2 at $0.66\n"
            "PRESSURE POINT: $0.70 confluence 5.8/10\n"
            "SUPPORTS: $0.66 [5.5] | $0.62 [4.0] | $0.58 [3.0]\n"
            "RESISTANCES: $0.70 [6.0] (former support) | $0.72 [3.5] | $0.75 [5.8]\n"
            "MTF BIAS: Bearish (38% alignment)"
        ),
        "assistant": {
            "action": "REDUCE_SIZE",
            "urgency": "HIGH",
            "confidence": 0.80,
            "reason": "Grade 2 support at $0.70 broken (not Grade 3). Trendline G2 also broken. Transitioning to LVN. Reduce 60-70% — structure damaged significantly. Keep 30-40% with stop at $0.65 near the 4h G2 support at $0.66.",
            "reasoning": {
                "structure_analysis": "Grade 2 horizontal at $0.70 broken. This is significant but not as critical as Grade 3 break. Confluence 5.8/10 — moderate. Next structural support is 4h G2 at $0.66, near our stop at $0.65.",
                "data_assessment": "Transitioning to LVN = faster movement expected. Sell-dominant. 38% MTF = bearish. Broader alt weakness means this isn't ADA-specific — could be sector rotation.",
                "risk_factors": "3x leverage, -0.55R. Next support at $0.66 is close to stop $0.65. If LVN acceleration occurs, could gap to stop. Reduce heavily.",
                "exit_logic": "MCF Level 3: Grade 2 broken. Not quite Level 2 (which requires Grade 3+). REDUCE 60-70% to de-risk. If $0.66 also breaks, exit remaining."
            },
            "execution": {"exit_pct": 65, "stop_price": 0.65, "order_type": "MARKET"}
        }
    },
    # LINK LONG - Grade 4 confluence broken — strongest signal
    {
        "user": (
            "Position: LONG LINK | Entry: $19.50 | Current: $17.20 | Stop: $16.00 | "
            "Leverage: 5x | R-Multiple: -0.66 | Duration: 24h | "
            "Market: LINK crashed through major confluence zone.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $17.80 (Grade 4 pivot+trendline+VPVR HVN, BROKEN) [priority: 9.2]\n"
            "NEAREST RESISTANCE: $17.80 (Grade 4, now resistance, 3.5% above) [priority: 9.2]\n"
            "VPVR: Crashed through HVN into LVN — fast movement. POC $18.50 (7.6% above). Sell-dominant.\n"
            "TRENDLINES: 4h ascending support G3 at $17.90 BROKEN | 1h ascending G2 at $18.20 BROKEN\n"
            "PRESSURE POINT: $17.80 confluence 9.0/10 — BREACHED\n"
            "SUPPORTS: $16.50 [4.5] | $15.80 [3.8] | $15.00 [2.5]\n"
            "RESISTANCES: $17.80 [9.2] (massive former support) | $18.50 [5.0] | $19.50 [6.5]\n"
            "MTF BIAS: Bearish (22% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "CATASTROPHIC STRUCTURAL BREAK: Grade 4 confluence at $17.80 (priority 9.2, confluence 9.0/10) BROKEN. This was the strongest structural level — pivot + trendline + VPVR HVN. Now in LVN with massive sell pressure. EXIT IMMEDIATELY.",
            "reasoning": {
                "structure_analysis": "Grade 4 is the highest possible grade — confluence of pivot, trendline, and VPVR HVN. Priority 9.2/10 and confluence 9.0/10. This level breaking is the most significant structural event possible. Two trendlines (G3, G2) also broken simultaneously.",
                "data_assessment": "Crashed through HVN into LVN — the worst transition. HVN was supposed to provide friction (support). Breaking through HVN into LVN means no friction until next structural level at $16.50 (Grade 1 only).",
                "risk_factors": "5x leverage. In LVN with no meaningful support until $16.50 (Grade 1) and $15.80 (Grade 1). Stop at $16.00 but LVN acceleration could gap through it.",
                "exit_logic": "MCF Level 2 — CRITICAL: Grade 4+ confluence broken. The strongest possible structural break signal. 9.0/10 confluence means this was the most tested, most reliable level. Its failure means the entire thesis is dead. EXIT 100%."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # BNB SHORT - Grade 2 resistance broken
    {
        "user": (
            "Position: SHORT BNB | Entry: $630 | Current: $648 | Stop: $660 | "
            "Leverage: 5x | R-Multiple: -0.60 | Duration: 8h | "
            "Market: BNB broke above resistance on Binance news.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $635 (Grade 1 horizontal 15m, 2.0% below) [priority: 3.5]\n"
            "NEAREST RESISTANCE: $642 (Grade 2 horizontal 1h, BROKEN) [priority: 6.2]\n"
            "VPVR: In LVN above $642 — fast movement. POC $625 (3.5% below). Buy-dominant.\n"
            "TRENDLINES: 1h descending resistance G2 at $640 BROKEN | 4h flat resistance G1 at $655\n"
            "PRESSURE POINT: $642 confluence 6.0/10 — BREACHED\n"
            "SUPPORTS: $635 [3.5] | $620 [5.8] | $605 [4.2]\n"
            "RESISTANCES: $655 [4.0] | $665 [3.5] | $680 [2.8]\n"
            "MTF BIAS: Bullish (68% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "HIGH",
            "confidence": 0.85,
            "reason": "STRUCTURAL BREAK: Grade 2 resistance at $642 broken for SHORT. Descending trendline G2 also broken. In LVN with buy-dominant volume and Binance news catalyst. At 5x leverage, exit before acceleration to $655+.",
            "reasoning": {
                "structure_analysis": "For SHORT: Grade 2 resistance at $642 broken. Price $648 is $6 above. Descending trendline G2 at $640 also broken. Two structural elements invalidated. Next resistance at $655 is only Grade 1.",
                "data_assessment": "LVN above $642 = fast movement up. Buy-dominant. News catalyst (Binance) can drive extended moves. 68% bullish MTF.",
                "risk_factors": "5x leverage at -0.60R. Stop at $660. In LVN, price can move fast toward $655-$660 (stop area). Better to exit at $648 than get stopped at $660.",
                "exit_logic": "MCF Level 2/3: Grade 2 resistance broken with momentum and catalyst. For shorts, broken resistance with volume = exit. Don't wait for stop."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # DOT LONG - Grade 3 broken, but holding at HVN
    {
        "user": (
            "Position: LONG DOT | Entry: $8.20 | Current: $7.40 | Stop: $6.80 | "
            "Leverage: 3x | R-Multiple: -0.57 | Duration: 36h | "
            "Market: DOT broke support but price stabilizing in HVN zone.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $7.50 (Grade 3 horizontal 4h, BROKEN) [priority: 7.8]\n"
            "NEAREST RESISTANCE: $7.50 (Grade 3, now resistance, 1.4% above) [priority: 7.8]\n"
            "VPVR: In HVN at $7.30-$7.50 — consolidation. POC $7.40 (at current price). Balanced.\n"
            "TRENDLINES: 4h ascending support G2 at $7.55 BROKEN | 1h flat support G1 at $7.25\n"
            "PRESSURE POINT: $7.50 confluence 7.0/10 — BREACHED\n"
            "SUPPORTS: $7.25 [4.0] | $7.00 [5.2] | $6.70 [3.5]\n"
            "RESISTANCES: $7.50 [7.8] (former support) | $7.80 [5.0] | $8.00 [4.5]\n"
            "MTF BIAS: Bearish (40% alignment)"
        ),
        "assistant": {
            "action": "REDUCE_SIZE",
            "urgency": "HIGH",
            "confidence": 0.82,
            "reason": "STRUCTURAL BREAK: Grade 3 at $7.50 (priority 7.8) broken. However, price is stabilizing in HVN at $7.30-$7.50 — high friction zone providing temporary support. Reduce 60-70% (structure broken) but keep small position — HVN may slow the decline.",
            "reasoning": {
                "structure_analysis": "Grade 3 broken — normally a full exit signal. But current position is at POC ($7.40) in HVN. HVN provides friction that may hold price. This is an edge case: structural break but VPVR support.",
                "data_assessment": "HVN = consolidation zone. Balanced volume at POC. Price stabilizing — not in freefall. The HVN is providing the friction that the broken Grade 3 no longer does.",
                "risk_factors": "3x leverage. Stop at $6.80. Next structural support at $7.00 (Grade 2). If HVN fails, acceleration to $7.00 and possibly $6.80 stop. Reduce to manage risk.",
                "exit_logic": "Grade 3 broken = normally full exit. But HVN at current level = reduced urgency. Compromise: reduce 60-70%, keep 30-40% with tight monitoring. If price leaves HVN downward, exit remainder."
            },
            "execution": {"exit_pct": 65, "stop_price": 7.00, "order_type": "MARKET"}
        }
    },
    # AVAX SHORT - Grade 3 resistance broken violently
    {
        "user": (
            "Position: SHORT AVAX | Entry: $38.00 | Current: $42.50 | Stop: $43.00 | "
            "Leverage: 10x | R-Multiple: -0.90 | Duration: 3h | "
            "Market: AVAX broke out with massive volume. Short squeeze.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $40.00 (Grade 2 horizontal 1h, 5.9% below) [priority: 5.8]\n"
            "NEAREST RESISTANCE: $41.00 (Grade 3 pivot confluence 4h, BROKEN) [priority: 8.5]\n"
            "VPVR: In LVN above $41 — fast movement. POC $39.50 (7.1% below). Buy-dominant.\n"
            "TRENDLINES: 4h descending resistance G3 at $41.20 BROKEN | no support trendlines above $40\n"
            "PRESSURE POINT: $41.00 confluence 8.2/10 — BREACHED\n"
            "SUPPORTS: $40.00 [5.8] | $38.50 [4.5] | $37.00 [3.2]\n"
            "RESISTANCES: $43.50 [3.0] | $45.00 [2.5] | $48.00 [2.0]\n"
            "MTF BIAS: Bullish (82% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "CRITICAL STRUCTURAL BREAK: Grade 3 resistance at $41 (confluence 8.2/10) smashed through. SHORT at 10x with broken resistance in LVN = CATASTROPHIC setup. Price $42.50 is already $1.50 past the broken level. Stop at $43 — exit NOW before stop is hit.",
            "reasoning": {
                "structure_analysis": "For SHORT: Grade 3 resistance broken. Price $42.50 is 3.7% above $41. Descending trendline G3 also broken. No meaningful resistance until $43.50 (Grade 1). Structure completely invalidated.",
                "data_assessment": "Short squeeze + LVN = explosive upside. Buy-dominant volume. 82% bullish MTF. This is the worst possible scenario for a short position. Massive volume breakout.",
                "risk_factors": "10x leverage at -0.90R. Stop at $43 — only $0.50 away. In LVN, price could gap through stop. At 10x, every additional 1% = 10% loss. EXIT BEFORE STOP.",
                "exit_logic": "MCF Level 2: CRITICAL structural break at 10x leverage. Grade 3 broken with massive volume in a short squeeze. This is not a debate — EXIT 100% IMMEDIATELY."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
]

for ex in structure_break_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))

print(f"Category 2 (Structure-Break EXIT): {len(structure_break_examples)} examples")


# ============================================================
# CATEGORY 3: VPVR-Informed Decisions (15 examples)
# HVN = consolidation, LVN = fast movement, POC = fair value
# Teach the model to use volume profile for decision-making
# ============================================================

vpvr_examples = [
    # BTC LONG in HVN — hold through consolidation
    {
        "user": (
            "Position: LONG BTC | Entry: $86,000 | Current: $85,200 | Stop: $83,000 | "
            "Leverage: 5x | R-Multiple: -0.27 | Duration: 8h | "
            "Market: BTC consolidating in tight range $84,800-$85,500.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $84,500 (Grade 3 horizontal 4h, 0.8% below) [priority: 7.8]\n"
            "NEAREST RESISTANCE: $87,000 (Grade 2 horizontal 4h, 2.1% above) [priority: 5.5]\n"
            "VPVR: Deep in HVN ($84,000-$86,000) — HIGH friction zone. POC $85,100 (0.1% below). Buy-dominant.\n"
            "TRENDLINES: 4h ascending support G2 at $84,200 | 1h flat support G2 at $84,500\n"
            "PRESSURE POINT: $84,500 confluence 7.2/10\n"
            "SUPPORTS: $84,500 [7.8] | $83,000 [5.5] | $81,000 [4.0]\n"
            "RESISTANCES: $87,000 [5.5] | $89,000 [4.2] | $91,000 [3.0]\n"
            "MTF BIAS: Bullish (60% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.77,
            "reason": "Deep in HVN ($84K-$86K) = high friction zone where price consolidates. POC at $85,100 — we're sitting at fair value. Buy-dominant volume. HVN breakdowns are rare — price typically consolidates then breaks out in the direction of the trend (bullish 60%).",
            "reasoning": {
                "structure_analysis": "Grade 3 support at $84,500 below. HVN means lots of volume traded here = strong agreement on value. Price doesn't break HVN easily — it consolidates.",
                "data_assessment": "POC $85,100 is fair value. Current $85,200 = at fair value. Buy-dominant in HVN = accumulation. Consolidation is normal and healthy in HVN zones.",
                "risk_factors": "HVN protects against sharp moves. Even if price dips, the high friction slows it. Grade 3 at $84,500 + HVN = double protection. 5x leverage with stop at $83,000 below all structure.",
                "exit_logic": "HVN = patience zone. The market is building energy for a breakout. Don't exit consolidation — that's the worst time to exit. HOLD and wait for the HVN breakout."
            },
            "execution": {"exit_pct": 0, "stop_price": 83000, "order_type": "LIMIT"}
        }
    },
    # ETH LONG entering LVN upward — let momentum carry
    {
        "user": (
            "Position: LONG ETH | Entry: $3,100 | Current: $3,250 | Stop: $3,020 | "
            "Leverage: 5x | R-Multiple: +1.88 | Duration: 12h | "
            "Market: ETH broke above HVN into LVN. Strong momentum.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $3,200 (Grade 2 horizontal 1h, 1.5% below) [priority: 6.0]\n"
            "NEAREST RESISTANCE: $3,350 (Grade 2 horizontal 4h, 3.1% above) [priority: 5.8]\n"
            "VPVR: In LVN ($3,220-$3,380) — LOW friction, fast movement UP. POC $3,150 (3.1% below). Buy-dominant.\n"
            "TRENDLINES: 1h ascending support G2 at $3,180 | 4h flat resistance G2 at $3,350\n"
            "PRESSURE POINT: $3,200 confluence 5.5/10\n"
            "SUPPORTS: $3,200 [6.0] | $3,100 [7.5] | $3,020 [5.0]\n"
            "RESISTANCES: $3,350 [5.8] | $3,450 [4.2] | $3,550 [3.0]\n"
            "MTF BIAS: Bullish (75% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.82,
            "reason": "In LVN moving UP — low friction means fast price movement in our direction. Buy-dominant volume with 75% bullish MTF. In LVN, let momentum carry to the next HVN zone near $3,350 resistance. Do NOT exit in LVN when trend is in your favor.",
            "reasoning": {
                "structure_analysis": "LVN zone = low volume traded here historically. Price moves FAST through LVN. Since we're LONG and price is moving UP through LVN, this is ideal — fast gains with minimal resistance.",
                "data_assessment": "Broke above HVN into LVN = breakout confirmed. Buy-dominant. Next resistance at $3,350 (Grade 2). LVN should carry us close to there before any friction.",
                "risk_factors": "5x at +1.88R. Strong position. LVN risk: if price reverses in LVN, the move DOWN is also fast. Trail stop to $3,180 (ascending trendline G2) to protect profits.",
                "exit_logic": "LVN in our direction = let it run. The next HVN zone near $3,350 is the natural TP target. Don't take profit in the middle of an LVN — that's where momentum delivers."
            },
            "execution": {"exit_pct": 0, "stop_price": 3180, "order_type": "LIMIT"}
        }
    },
    # SOL SHORT in LVN moving down — ride the momentum
    {
        "user": (
            "Position: SHORT SOL | Entry: $158 | Current: $149 | Stop: $165 | "
            "Leverage: 3x | R-Multiple: +1.29 | Duration: 6h | "
            "Market: SOL dropped through HVN into LVN. Accelerating.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $145 (Grade 2 horizontal 4h, 2.7% below) [priority: 6.2]\n"
            "NEAREST RESISTANCE: $152 (Grade 3 horizontal 1h, 2.0% above) [priority: 7.5]\n"
            "VPVR: In LVN ($146-$152) — fast movement DOWN. POC $155 (4.0% above). Sell-dominant.\n"
            "TRENDLINES: 1h descending resistance G2 at $153 | 4h flat support G2 at $145\n"
            "PRESSURE POINT: $152 confluence 7.0/10\n"
            "SUPPORTS: $145 [6.2] | $140 [5.0] | $135 [3.8]\n"
            "RESISTANCES: $152 [7.5] | $155 [5.2] | $158 [6.0]\n"
            "MTF BIAS: Bearish (70% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.80,
            "reason": "SHORT in LVN moving DOWN — fast movement in our favor. Sell-dominant volume with 70% bearish MTF. LVN means low friction — price should accelerate to next HVN/support at $145. Let the momentum work.",
            "reasoning": {
                "structure_analysis": "In LVN between $146-$152. For SHORT, LVN moving down = ideal. Grade 3 resistance at $152 above provides cap. Target the Grade 2 support at $145 where HVN begins.",
                "data_assessment": "Sell-dominant in LVN = strong selling momentum with no friction to slow it. 70% bearish MTF. POC at $155 is well above — price rejected from fair value zone.",
                "risk_factors": "3x leverage at +1.29R. Comfortable position. In LVN, if price bounces, the move back up is also fast. Trail stop to $153 (descending trendline G2 + Grade 3 resistance).",
                "exit_logic": "LVN in SHORT direction = ride the momentum. Target HVN at $145. Do NOT take profit in the middle of LVN — momentum hasn't finished. TP at next friction zone."
            },
            "execution": {"exit_pct": 0, "stop_price": 153, "order_type": "LIMIT"}
        }
    },
    # DOGE at POC — fair value, consolidation expected
    {
        "user": (
            "Position: LONG DOGE | Entry: $0.170 | Current: $0.182 | Stop: $0.160 | "
            "Leverage: 5x | R-Multiple: +1.20 | Duration: 24h | "
            "Market: DOGE settled at POC after rally. Consolidating.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $0.178 (Grade 2 horizontal 1h, 2.2% below) [priority: 5.8]\n"
            "NEAREST RESISTANCE: $0.192 (Grade 3 horizontal 4h, 5.5% above) [priority: 7.5]\n"
            "VPVR: At POC ($0.182) — fair value zone. Deep HVN. Balanced volume.\n"
            "TRENDLINES: 1h ascending support G2 at $0.176 | 4h flat resistance G3 at $0.192\n"
            "PRESSURE POINT: $0.178 confluence 5.5/10\n"
            "SUPPORTS: $0.178 [5.8] | $0.170 [6.5] | $0.162 [4.5]\n"
            "RESISTANCES: $0.192 [7.5] | $0.205 [4.0] | $0.220 [2.8]\n"
            "MTF BIAS: Bullish (58% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.75,
            "reason": "At POC ($0.182) = fair value. In deep HVN. Price consolidates at POC before the next directional move. At +1.20R with 58% bullish MTF, wait for the consolidation to resolve. HVN protects against sharp drops.",
            "reasoning": {
                "structure_analysis": "POC = Point of Control, the price with most volume traded. Being AT POC means price is at fair value — the market agrees this is the right price. Consolidation is expected.",
                "data_assessment": "Balanced volume at POC. Neither buying nor selling dominant. This is a pause, not a reversal. 58% bullish MTF suggests eventual upside resolution.",
                "risk_factors": "5x at +1.20R. Good position. HVN protects against sharp moves. Support at $0.178 + ascending trendline G2 at $0.176 below. Well-protected.",
                "exit_logic": "POC consolidation = patience. Don't exit at fair value. Wait for the breakout from HVN which should come in the bullish direction (58% MTF + uptrend)."
            },
            "execution": {"exit_pct": 0, "stop_price": 0.176, "order_type": "LIMIT"}
        }
    },
    # BTC LONG approaching HVN resistance — TP zone
    {
        "user": (
            "Position: LONG BTC | Entry: $82,000 | Current: $86,800 | Stop: $80,000 | TP1: $87,500 | "
            "Leverage: 10x | R-Multiple: +2.40 | Duration: 18h | "
            "Market: BTC approaching major HVN resistance zone.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $85,500 (Grade 2 horizontal 1h, 1.5% below) [priority: 5.8]\n"
            "NEAREST RESISTANCE: $87,500 (Grade 3 horizontal 4h, 0.8% above) [priority: 8.2]\n"
            "VPVR: Approaching HVN at $87,000-$89,000 — expect STALL. POC $88,000 (1.4% above). Sell-dominant in upper zone.\n"
            "TRENDLINES: 1h ascending support G2 at $85,800 | 4h flat resistance G3 at $87,500\n"
            "PRESSURE POINT: $87,500 confluence 8.0/10\n"
            "SUPPORTS: $85,500 [5.8] | $84,000 [7.0] | $82,000 [5.5]\n"
            "RESISTANCES: $87,500 [8.2] | $89,000 [6.5] | $91,000 [4.0]\n"
            "MTF BIAS: Bullish (65% alignment)"
        ),
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "MEDIUM",
            "confidence": 0.80,
            "reason": "Approaching Grade 3 resistance at $87,500 (confluence 8.0/10) which is the entry to a sell-dominant HVN zone. At 10x leverage with +2.40R, take partial (35%) as price enters the friction zone. HVN will stall momentum — secure profits before the stall.",
            "reasoning": {
                "structure_analysis": "Grade 3 at $87,500 with 8.0/10 confluence = major resistance. Above it is HVN ($87K-$89K) = high friction where price stalls. Sell-dominant in upper zone means selling pressure increases here.",
                "data_assessment": "HVN ahead = momentum will slow. Sell-dominant in the resistance HVN. This is the classic setup for partial TP — price enters friction zone where continuation is uncertain.",
                "risk_factors": "10x at +2.40R = 24% unrealized leveraged gain. In HVN, price stalls and can reverse. At 10x, a 2% reversal = 20% loss on remaining. Secure partial.",
                "exit_logic": "VPVR-informed TP: approaching sell-dominant HVN with Grade 3 resistance. This is the STRUCTURE-based TP signal. Take 35%, trail stop on remainder. If price pushes through HVN, remainder captures TP2."
            },
            "execution": {"exit_pct": 35, "stop_price": 85500, "order_type": "LIMIT"}
        }
    },
    # ADA SHORT in LVN approaching HVN support — TP zone
    {
        "user": (
            "Position: SHORT ADA | Entry: $0.82 | Current: $0.73 | Stop: $0.88 | TP1: $0.72 | "
            "Leverage: 5x | R-Multiple: +1.50 | Duration: 20h | "
            "Market: ADA dropping toward major support HVN zone.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $0.71 (Grade 3 horizontal 4h, 2.7% below) [priority: 8.0]\n"
            "NEAREST RESISTANCE: $0.76 (Grade 2 horizontal 1h, 4.1% above) [priority: 5.5]\n"
            "VPVR: In LVN approaching HVN at $0.70-$0.72 — expect stall at HVN entry. POC $0.72 (1.4% below). Buy-dominant in lower zone.\n"
            "TRENDLINES: 1h descending resistance G2 at $0.76 | 4h flat support G3 at $0.71\n"
            "PRESSURE POINT: $0.71 confluence 7.5/10\n"
            "SUPPORTS: $0.71 [8.0] | $0.68 [5.5] | $0.65 [3.8]\n"
            "RESISTANCES: $0.76 [5.5] | $0.80 [6.0] | $0.82 [5.2]\n"
            "MTF BIAS: Bearish (62% alignment)"
        ),
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "MEDIUM",
            "confidence": 0.78,
            "reason": "SHORT approaching HVN at $0.70-$0.72 with buy-dominant volume. Grade 3 support at $0.71 (confluence 7.5/10). HVN = friction zone where SHORT momentum will stall. At +1.50R, take 30-35% partial before entering the support HVN.",
            "reasoning": {
                "structure_analysis": "For SHORT: approaching buy-dominant HVN = friction. Grade 3 support at $0.71 with 7.5/10 confluence is a strong floor. This is where longs step in. TP zone for shorts.",
                "data_assessment": "LVN now but HVN ahead. Buy-dominant in $0.70-$0.72. Momentum will slow dramatically when we enter HVN. POC at $0.72 = fair value in that zone.",
                "risk_factors": "5x at +1.50R. Solid profits. HVN + Grade 3 support = strong floor. Short could stall or reverse here. Secure gains.",
                "exit_logic": "VPVR-informed TP: LVN momentum approaching buy-dominant HVN with Grade 3 support. This is the structural TP zone for shorts. Take partial, trail remainder with tight stop."
            },
            "execution": {"exit_pct": 35, "stop_price": 0.76, "order_type": "LIMIT"}
        }
    },
    # LINK LONG — sell-dominant HVN = danger zone for longs
    {
        "user": (
            "Position: LONG LINK | Entry: $16.00 | Current: $17.80 | Stop: $14.80 | "
            "Leverage: 5x | R-Multiple: +1.50 | Duration: 36h | "
            "Market: LINK rallying but entering sell-dominant zone.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $17.20 (Grade 2 horizontal 1h, 3.4% below) [priority: 6.0]\n"
            "NEAREST RESISTANCE: $18.50 (Grade 3 horizontal 4h, 3.9% above) [priority: 7.8]\n"
            "VPVR: Entering HVN ($17.50-$18.80) — sell-dominant zone. POC $18.20 (2.2% above). Sell-dominant.\n"
            "TRENDLINES: 1h ascending support G2 at $17.00 | 4h descending resistance G2 at $18.60\n"
            "PRESSURE POINT: $18.50 confluence 7.2/10\n"
            "SUPPORTS: $17.20 [6.0] | $16.50 [5.0] | $15.80 [4.2]\n"
            "RESISTANCES: $18.50 [7.8] | $19.50 [5.5] | $20.50 [3.8]\n"
            "MTF BIAS: Bullish (55% alignment)"
        ),
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "MEDIUM",
            "confidence": 0.76,
            "reason": "Entering sell-dominant HVN zone ($17.50-$18.80). Grade 3 resistance at $18.50 ahead with 7.2/10 confluence. Sell-dominant HVN = distribution zone where sellers overwhelm buyers. At 5x with +1.50R, take 30% partial.",
            "reasoning": {
                "structure_analysis": "HVN with sell-dominant volume = distribution. Historical selling pressure at these levels. Grade 3 resistance at $18.50 + descending trendline G2 at $18.60 = strong ceiling.",
                "data_assessment": "Sell-dominant HVN is the opposite of accumulation — it's distribution. Price enters this zone and faces heavy sell orders. Momentum typically stalls or reverses.",
                "risk_factors": "5x at +1.50R. If price stalls at $18.50 Grade 3 and sells off, a 3% reversal = 15% leveraged loss. Partial TP protects against HVN reversal.",
                "exit_logic": "VPVR sell-dominant HVN ahead + Grade 3 resistance = structural TP signal. The volume profile tells you where sellers are concentrated. Respect the volume."
            },
            "execution": {"exit_pct": 30, "stop_price": 17.00, "order_type": "LIMIT"}
        }
    },
    # XRP entering buy-dominant HVN on SHORT — exit zone
    {
        "user": (
            "Position: SHORT XRP | Entry: $2.60 | Current: $2.38 | Stop: $2.72 | "
            "Leverage: 5x | R-Multiple: +1.83 | Duration: 16h | "
            "Market: XRP dropping into major buy-dominant support zone.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $2.35 (Grade 3 horizontal 4h, 1.3% below) [priority: 8.8]\n"
            "NEAREST RESISTANCE: $2.45 (Grade 2 horizontal 1h, 2.9% above) [priority: 5.5]\n"
            "VPVR: Entering HVN at $2.30-$2.40 — STRONGLY buy-dominant. POC $2.36 (0.8% below). Buy-dominant.\n"
            "TRENDLINES: 4h ascending support G3 at $2.32 | 1h descending resistance G2 at $2.48\n"
            "PRESSURE POINT: $2.35 confluence 8.5/10\n"
            "SUPPORTS: $2.35 [8.8] | $2.28 [5.5] | $2.20 [4.0]\n"
            "RESISTANCES: $2.45 [5.5] | $2.52 [6.0] | $2.60 [5.8]\n"
            "MTF BIAS: Bearish (45% alignment)"
        ),
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "HIGH",
            "confidence": 0.82,
            "reason": "SHORT entering STRONGLY buy-dominant HVN at $2.30-$2.40. Grade 3 support at $2.35 (priority 8.8, confluence 8.5/10) — this is a WALL of buying pressure. At +1.83R, take 40% before hitting this buy wall.",
            "reasoning": {
                "structure_analysis": "For SHORT: buy-dominant HVN = the worst zone to be short. Grade 3 at $2.35 with 8.8 priority and 8.5/10 confluence. Ascending trendline G3 at $2.32. This is a major structural floor.",
                "data_assessment": "Strongly buy-dominant HVN means massive historical buying at $2.30-$2.40. Shorts entering this zone face fierce opposition. Price is likely to bounce here.",
                "risk_factors": "5x at +1.83R. Excellent profits. But Grade 3 + buy-dominant HVN = high probability of bounce. A 3% bounce = 15% of remaining gains lost. Secure now.",
                "exit_logic": "VPVR buy-dominant HVN ahead + Grade 3 support with 8.5/10 confluence. This is the textbook SHORT TP signal — take profits before the buy wall absorbs the selling pressure."
            },
            "execution": {"exit_pct": 40, "stop_price": 2.45, "order_type": "LIMIT"}
        }
    },
]

for ex in vpvr_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))

print(f"Category 3 (VPVR-Informed): {len(vpvr_examples)} examples")


# ============================================================
# CATEGORY 4: Multi-Pair Diversity (20 examples)
# Cover all 10 pairs with varied scenarios — HOLD, EXIT, TP, REDUCE
# Focus on pairs underrepresented in v5: ETH, XRP, DOGE, ADA, LINK, DOT
# ============================================================

multi_pair_examples = [
    # ETH LONG — winning, structure strong, hold
    {
        "user": (
            "Position: LONG ETH | Entry: $2,950 | Current: $3,180 | Stop: $2,850 | "
            "Leverage: 5x | R-Multiple: +2.30 | Duration: 36h | "
            "Market: ETH breakout above consolidation. Higher highs.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $3,100 (Grade 3 horizontal 1h, 2.5% below) [priority: 7.5]\n"
            "NEAREST RESISTANCE: $3,250 (Grade 2 horizontal 4h, 2.2% above) [priority: 5.8]\n"
            "VPVR: In LVN above $3,100 — fast movement up. POC $3,050 (4.1% below). Buy-dominant.\n"
            "TRENDLINES: 1h ascending support G3 at $3,080 | 4h ascending support G2 at $3,020\n"
            "SUPPORTS: $3,100 [7.5] | $3,020 [6.0] | $2,950 [5.5]\n"
            "RESISTANCES: $3,250 [5.8] | $3,380 [4.2] | $3,500 [3.0]\n"
            "MTF BIAS: Bullish (78% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.85,
            "reason": "ETH LONG with strong structure. In LVN moving up = fast gains. Grade 3 support at $3,100 with ascending trendline G3 at $3,080. 78% bullish MTF. At +2.30R, trail stop to $3,080 and let the trend run.",
            "reasoning": {
                "structure_analysis": "Grade 3 support at $3,100 is 2.5% below — excellent buffer. Ascending trendline G3 at $3,080 provides secondary support. Structure firmly intact with multiple layers below.",
                "data_assessment": "LVN = fast movement. Buy-dominant. Breakout above consolidation with higher highs = strong trend. 78% MTF bullish is strong alignment.",
                "risk_factors": "5x at +2.30R. Trail stop to $3,080 (G3 trendline). Even if price reverses, stop protects +1.3R minimum. Very favorable risk/reward.",
                "exit_logic": "Structure intact, LVN momentum, 78% MTF bullish. No exit signals. Target $3,250 Grade 2 resistance. HOLD."
            },
            "execution": {"exit_pct": 0, "stop_price": 3080, "order_type": "LIMIT"}
        }
    },
    # ETH SHORT — structure broken, exit
    {
        "user": (
            "Position: SHORT ETH | Entry: $3,300 | Current: $3,380 | Stop: $3,450 | "
            "Leverage: 10x | R-Multiple: -0.53 | Duration: 5h | "
            "Market: ETH broke above descending trendline. Buyers stepping in.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $3,300 (Grade 1 horizontal 15m, 2.4% below) [priority: 3.5]\n"
            "NEAREST RESISTANCE: $3,350 (Grade 3 descending trendline 1h, BROKEN) [priority: 7.8]\n"
            "VPVR: In LVN above $3,350 — fast movement. POC $3,280 (3.0% below). Buy-dominant.\n"
            "TRENDLINES: 1h descending resistance G3 at $3,350 BROKEN | 4h flat resistance G2 at $3,420\n"
            "SUPPORTS: $3,300 [3.5] | $3,250 [5.2] | $3,180 [6.0]\n"
            "RESISTANCES: $3,420 [5.5] | $3,500 [4.0] | $3,600 [3.0]\n"
            "MTF BIAS: Bullish (70% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.90,
            "reason": "STRUCTURAL BREAK: Grade 3 descending trendline at $3,350 BROKEN for SHORT. Price in LVN with buy-dominant volume and 70% bullish MTF. At 10x leverage, exit immediately before LVN acceleration to $3,420+.",
            "reasoning": {
                "structure_analysis": "For SHORT: Grade 3 trendline broken. $3,380 > $3,350. In LVN = fast upside. Next resistance at $3,420 (Grade 2) — could reach quickly.",
                "data_assessment": "Trendline break with buy-dominant volume in LVN. 70% bullish MTF. All signals point to continuation higher — catastrophic for shorts.",
                "risk_factors": "10x at -0.53R. Stop at $3,450 but in LVN, price moves fast. Better to exit at $3,380 (controlled) than get stopped at $3,450 (10% more leveraged loss).",
                "exit_logic": "MCF Level 2: Grade 3 trendline broken. At 10x in LVN with bullish volume = CRITICAL exit. Structure invalidated."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # XRP LONG — no stop, 5x, losing — REDUCE
    {
        "user": (
            "Position: LONG XRP | Entry: $2.55 | Current: $2.42 | Stop: NONE | "
            "Leverage: 5x | R-Multiple: +0.0 | Duration: 10h | "
            "Market: XRP dropping. No stop set.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $2.38 (Grade 2 horizontal 4h, 1.7% below) [priority: 6.2]\n"
            "NEAREST RESISTANCE: $2.50 (Grade 2 horizontal 1h, 3.3% above) [priority: 5.8]\n"
            "VPVR: In HVN — consolidation. POC $2.44 (0.8% above). Balanced.\n"
            "SUPPORTS: $2.38 [6.2] | $2.30 [5.0] | $2.22 [3.8]\n"
            "RESISTANCES: $2.50 [5.8] | $2.58 [4.5] | $2.65 [3.2]\n"
            "MTF BIAS: Bearish (42% alignment)"
        ),
        "assistant": {
            "action": "REDUCE_SIZE",
            "urgency": "HIGH",
            "confidence": 0.80,
            "reason": "XRP LONG at 5x with NO STOP and losing 5.1% (25.5% leveraged). At 5x losing without a stop, REDUCE 50% immediately. Set stop on remainder at $2.38 (Grade 2 support).",
            "reasoning": {
                "structure_analysis": "Grade 2 support at $2.38 is 1.7% below. This should be the stop for the remaining position. Structure exists but the position has no stop to enforce it.",
                "data_assessment": "HVN = consolidation, which slows the decline. But bearish MTF at 42% and losing 5.1% with no stop is unacceptable risk management.",
                "risk_factors": "5x losing + no stop = compounding risk. Already -25.5% leveraged. Another 2% drop = 10% more. MUST reduce and set stop.",
                "exit_logic": "5x no-stop losing: REDUCE 50%, set stop at nearest structural support ($2.38 Grade 2). The structure provides the natural stop level — use it."
            },
            "execution": {"exit_pct": 50, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # DOGE SHORT — stop breached, bouncing, still exit
    {
        "user": (
            "Position: SHORT DOGE | Entry: $0.200 | Current: $0.218 | Stop: $0.212 | "
            "Leverage: 10x | R-Multiple: -1.50 | Duration: 4h | "
            "Market: DOGE pumped past stop. Now slightly pulling back from $0.225 high. RSI 78.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $0.210 (Grade 1 horizontal 15m, 3.7% below) [priority: 3.5]\n"
            "NEAREST RESISTANCE: $0.225 (Grade 1 horizontal 15m, 3.2% above) [priority: 3.0]\n"
            "VPVR: In LVN — fast movement zone. POC $0.195 (10.5% below). Buy-dominant.\n"
            "SUPPORTS: $0.210 [3.5] | $0.200 [5.5] | $0.192 [4.8]\n"
            "RESISTANCES: $0.225 [3.0] | $0.235 [2.5] | $0.250 [2.0]\n"
            "MTF BIAS: Bullish (75% alignment)"
        ),
        "assistant": {
            "action": "EXIT_FULL",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "SHORT stop at $0.212 BREACHED — current $0.218 is $0.006 ABOVE stop. For SHORT, price above stop = breached. EXIT IMMEDIATELY. RSI overbought pullback is IRRELEVANT when hard stop is violated. At 10x this is a survival situation.",
            "reasoning": {
                "structure_analysis": "Stop breached: $0.218 > $0.212. Past stop by 2.8%. MCF Level 1: Hard Stop is unconditional.",
                "data_assessment": "RSI 78 and pulling back from $0.225 might suggest the pump is fading. But this is IRRELEVANT — the hard stop has been violated. Potential pullback does not override a breached stop.",
                "risk_factors": "10x leverage, -1.50R already. Past the stop. At 10x, every 1% more = 10% loss. DOGE can pump 20%+ in a day. EXIT.",
                "exit_logic": "MCF Level 1: HARD STOP BREACHED. The stop exists for this exact moment. RSI, pullbacks, technical patterns — NONE override a breached hard stop. EXIT 100%."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # ADA SHORT — winning, structure intact, hold
    {
        "user": (
            "Position: SHORT ADA | Entry: $0.85 | Current: $0.78 | Stop: $0.90 | "
            "Leverage: 3x | R-Multiple: +1.40 | Duration: 48h | "
            "Market: ADA in downtrend. Lower highs confirmed.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $0.75 (Grade 2 horizontal 4h, 3.8% below) [priority: 6.5]\n"
            "NEAREST RESISTANCE: $0.82 (Grade 3 descending trendline 1h, 5.1% above) [priority: 7.8]\n"
            "VPVR: In LVN — fast movement down. POC $0.83 (6.4% above). Sell-dominant.\n"
            "TRENDLINES: 1h descending resistance G3 at $0.82 | 4h descending resistance G2 at $0.84\n"
            "SUPPORTS: $0.75 [6.5] | $0.70 [5.0] | $0.65 [3.5]\n"
            "RESISTANCES: $0.82 [7.8] | $0.85 [5.5] | $0.88 [4.2]\n"
            "MTF BIAS: Bearish (72% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "LOW",
            "confidence": 0.82,
            "reason": "ADA SHORT with intact structure. Grade 3 descending trendline at $0.82 caps upside. In LVN moving down with sell-dominant volume. 72% bearish MTF. At +1.40R, let the trend work. Target $0.75 Grade 2 support.",
            "reasoning": {
                "structure_analysis": "For SHORT: Grade 3 resistance at $0.82 is 5.1% above — large buffer. Two descending trendlines (G3, G2) provide overhead pressure. Structure fully intact.",
                "data_assessment": "LVN = fast movement. Sell-dominant. Lower highs confirmed. 72% bearish MTF. All signals aligned for SHORT continuation.",
                "risk_factors": "3x at +1.40R. Stop at $0.90 is well above all resistance. Very comfortable risk position. Low urgency.",
                "exit_logic": "Structure intact. LVN momentum. Bearish MTF. Target $0.75. No exit triggers. HOLD."
            },
            "execution": {"exit_pct": 0, "stop_price": 0.90, "order_type": "LIMIT"}
        }
    },
    # AVAX LONG — near TP1, 10x, take partial
    {
        "user": (
            "Position: LONG AVAX | Entry: $35.00 | Current: $39.80 | Stop: $33.00 | TP1: $40.50 | "
            "Leverage: 10x | R-Multiple: +2.40 | Duration: 12h | "
            "Market: AVAX rallying. TP1 $0.70 away. Momentum slowing.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $38.50 (Grade 2 horizontal 1h, 3.3% below) [priority: 6.0]\n"
            "NEAREST RESISTANCE: $40.50 (Grade 3 horizontal 4h, 1.8% above) [priority: 7.8]\n"
            "VPVR: Approaching HVN at $40-$42. POC $41.00 (3.0% above). Sell-dominant in $40-$42 range.\n"
            "TRENDLINES: 1h ascending support G2 at $38.80 | 4h flat resistance G3 at $40.50\n"
            "SUPPORTS: $38.50 [6.0] | $37.00 [5.2] | $35.50 [4.0]\n"
            "RESISTANCES: $40.50 [7.8] | $42.00 [5.5] | $44.00 [3.5]\n"
            "MTF BIAS: Bullish (60% alignment)"
        ),
        "assistant": {
            "action": "TP_PARTIAL",
            "urgency": "HIGH",
            "confidence": 0.82,
            "reason": "AVAX approaching Grade 3 resistance at $40.50 (TP1) entering sell-dominant HVN zone. At 10x with +2.40R and momentum slowing, take 35% partial. Structure says resistance ahead + sell-dominant volume = stall likely.",
            "reasoning": {
                "structure_analysis": "TP1 at $40.50 = Grade 3 resistance with priority 7.8. Above it is sell-dominant HVN. This is a structural ceiling. Momentum slowing as we approach it.",
                "data_assessment": "Sell-dominant HVN at $40-$42 = distribution zone. Momentum slowing = buyers exhausting. Classic TP setup — approaching resistance with fading momentum.",
                "risk_factors": "10x at +2.40R = 24% leveraged gain. If price stalls at $40.50 and reverses 2%, that's 20% off remaining. Secure partial at resistance.",
                "exit_logic": "Structure-based TP: Grade 3 resistance + sell-dominant HVN + slowing momentum. Take 35%, move stop to $38.50 (Grade 2) on remainder."
            },
            "execution": {"exit_pct": 35, "stop_price": 38.50, "order_type": "LIMIT"}
        }
    },
    # LINK LONG — no stop, 20x, winning — EXIT always
    {
        "user": (
            "Position: LONG LINK | Entry: $17.00 | Current: $17.80 | Stop: NONE | "
            "Leverage: 20x | R-Multiple: +0.0 | Duration: 2h | "
            "Market: LINK pumping. No stop set. At 20x.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $17.20 (Grade 2 horizontal 1h, 3.4% below) [priority: 6.0]\n"
            "NEAREST RESISTANCE: $18.50 (Grade 2 horizontal 4h, 3.9% above) [priority: 5.8]\n"
            "VPVR: In LVN — fast movement. POC $17.50 (1.7% below). Buy-dominant.\n"
            "SUPPORTS: $17.20 [6.0] | $16.50 [5.0] | $15.80 [4.2]\n"
            "RESISTANCES: $18.50 [5.8] | $19.50 [4.5] | $20.50 [3.0]\n"
            "MTF BIAS: Bullish (65% alignment)"
        ),
        "assistant": {
            "action": "EXIT_100_PERCENT_IMMEDIATELY",
            "urgency": "CRITICAL",
            "confidence": 0.95,
            "reason": "LINK at 20x with NO STOP. Despite winning +4.7% (94% leveraged), 20x without a stop is ALWAYS an immediate exit. A 5% reversal = 100% loss (liquidation). Lock in the 94% gain NOW. No exceptions at 20x.",
            "reasoning": {
                "structure_analysis": "Structure is bullish (LVN, buy-dominant). But this is irrelevant — 20x + no stop = EXIT regardless of structure.",
                "data_assessment": "LINK pumping, LVN, bullish. All great. But at 20x, one bad candle can wipe out everything. The gains mean nothing if you get liquidated.",
                "risk_factors": "20x = liquidation at ~5% adverse. No stop means no protection at all. Even in LVN (fast movement), a reversal is equally fast. 94% leveraged gain unprotected.",
                "exit_logic": "20x no-stop: ALWAYS EXIT. This is an absolute rule that overrides all structural analysis, VPVR, trendlines, MTF bias. Lock in the gain. Period."
            },
            "execution": {"exit_pct": 100, "stop_price": None, "order_type": "MARKET"}
        }
    },
    # DOT SHORT — funding crowded but structure intact, hold
    {
        "user": (
            "Position: SHORT DOT | Entry: $8.50 | Current: $7.80 | Stop: $9.20 | "
            "Leverage: 5x | R-Multiple: +1.00 | Duration: 24h | "
            "Market: Funding -0.018% (shorts crowded). But downtrend strong.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $7.50 (Grade 2 horizontal 4h, 3.8% below) [priority: 6.5]\n"
            "NEAREST RESISTANCE: $8.20 (Grade 3 descending trendline 1h, 5.1% above) [priority: 7.5]\n"
            "VPVR: In LVN — fast movement down. POC $8.10 (3.8% above). Sell-dominant.\n"
            "TRENDLINES: 1h descending resistance G3 at $8.20 | 4h descending resistance G2 at $8.40\n"
            "SUPPORTS: $7.50 [6.5] | $7.00 [5.0] | $6.50 [3.5]\n"
            "RESISTANCES: $8.20 [7.5] | $8.50 [5.5] | $8.80 [4.0]\n"
            "MTF BIAS: Bearish (68% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.75,
            "reason": "DOT SHORT with intact bearish structure. Funding -0.018% (crowded shorts) is a risk factor but NOT an exit signal. Downtrend strong with Grade 3 descending trendline capping upside. 68% bearish MTF. Structure > funding.",
            "reasoning": {
                "structure_analysis": "Grade 3 descending trendline at $8.20 is 5.1% above — good buffer. Two descending trendlines provide strong ceiling. Structure intact.",
                "data_assessment": "Negative funding means shorts are paying longs — crowding risk. But price is still making lower lows in LVN with sell-dominant volume. Actual price action overrides funding signal.",
                "risk_factors": "Crowded shorts increase squeeze risk IF price reverses. But the stop at $9.20 handles that. If squeeze occurs, stop triggers — that's what it's for.",
                "exit_logic": "Funding is supplementary data, NOT a structural break. MCF hierarchy requires actual PRICE STRUCTURE breaks for exits. Structure intact = HOLD. Let the stop handle the squeeze if it comes."
            },
            "execution": {"exit_pct": 0, "stop_price": 9.20, "order_type": "LIMIT"}
        }
    },
    # BNB LONG — at breakeven, high leverage, structure neutral
    {
        "user": (
            "Position: LONG BNB | Entry: $640 | Current: $642 | Stop: $620 | "
            "Leverage: 10x | R-Multiple: +0.10 | Duration: 2h | "
            "Market: BNB flat. Just entered. Structure neutral.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $635 (Grade 2 horizontal 1h, 1.1% below) [priority: 6.0]\n"
            "NEAREST RESISTANCE: $655 (Grade 2 horizontal 4h, 2.0% above) [priority: 5.5]\n"
            "VPVR: In HVN — consolidation. POC $640 (0.3% below). Balanced.\n"
            "TRENDLINES: 1h flat support G2 at $634 | 4h flat resistance G2 at $656\n"
            "SUPPORTS: $635 [6.0] | $625 [5.2] | $615 [4.0]\n"
            "RESISTANCES: $655 [5.5] | $668 [4.2] | $680 [3.0]\n"
            "MTF BIAS: Neutral (52% alignment)"
        ),
        "assistant": {
            "action": "HOLD",
            "urgency": "MEDIUM",
            "confidence": 0.68,
            "reason": "BNB at near-breakeven in HVN consolidation. Stop at $620 is below Grade 2 support at $635. At 10x, this is a new position — give it time to develop. HVN protects against sharp moves. Neutral MTF means no strong directional pressure.",
            "reasoning": {
                "structure_analysis": "Grade 2 support at $635 is 1.1% below. Stop at $620 is well below structure. HVN consolidation zone. Position is new — just 2 hours old.",
                "data_assessment": "HVN at POC = fair value zone. Balanced volume. Consolidation. New position needs time for thesis to play out. No immediate threat.",
                "risk_factors": "10x leverage at breakeven. Stop at $620 = 3.1% risk (31% leveraged). Manageable with proper stop. HVN provides friction protection.",
                "exit_logic": "New position in HVN with proper stop below structure. No structural break. Give the position time. HOLD with MEDIUM urgency due to 10x leverage."
            },
            "execution": {"exit_pct": 0, "stop_price": 620, "order_type": "LIMIT"}
        }
    },
    # ETH LONG — massive winner, trail stop to structure
    {
        "user": (
            "Position: LONG ETH | Entry: $2,800 | Current: $3,350 | Stop: $2,700 | TP1: $3,200 | "
            "Leverage: 3x | R-Multiple: +5.50 | Duration: 120h | "
            "Market: ETH extended rally. Past TP1. Momentum still strong.\n\n"
            "STRUCTURAL ANALYSIS (15m/1h/4h):\n"
            "NEAREST SUPPORT: $3,280 (Grade 3 horizontal 1h, 2.1% below) [priority: 7.5]\n"
            "NEAREST RESISTANCE: $3,450 (Grade 2 horizontal 4h, 3.0% above) [priority: 5.8]\n"
            "VPVR: In LVN above $3,300 — fast movement. POC $3,200 (4.5% below). Buy-dominant.\n"
            "TRENDLINES: 1h ascending support G3 at $3,260 | 4h ascending support G2 at $3,180\n"
            "SUPPORTS: $3,280 [7.5] | $3,200 [6.5] | $3,100 [5.0]\n"
            "RESISTANCES: $3,450 [5.8] | $3,550 [4.0] | $3,700 [2.8]\n"
            "MTF BIAS: Bullish (82% alignment)"
        ),
        "assistant": {
            "action": "TRAIL_STOP",
            "urgency": "MEDIUM",
            "confidence": 0.82,
            "reason": "ETH massive winner at +5.50R, past TP1. Trail stop to nearest Grade 3 support at $3,280 — this is the STRUCTURE-BASED trailing stop. NOT an arbitrary ATR distance. If Grade 3 breaks, the trend is weakening. Until then, let the winner run.",
            "reasoning": {
                "structure_analysis": "Grade 3 at $3,280 (priority 7.5) is the natural trail stop level. Ascending trendline G3 at $3,260 is secondary. These are structural levels — not arbitrary distances.",
                "data_assessment": "LVN with buy-dominant volume. 82% bullish MTF. Past TP1 with strong momentum. The trend is intact — don't cut it short with arbitrary trailing.",
                "risk_factors": "3x at +5.50R. Trailing to $3,280 locks in minimum +4.80R. If Grade 3 holds and trend continues, remaining captures TP2/TP3. Excellent risk/reward.",
                "exit_logic": "VPVR-informed trail stop: use Grade 3 support at $3,280 (structural level) NOT ATR-based arbitrary distance. Structure-based trails give positions room to breathe while protecting profits at meaningful levels."
            },
            "execution": {"exit_pct": 0, "stop_price": 3280, "order_type": "LIMIT"}
        }
    },
]

for ex in multi_pair_examples:
    examples.append(make_example(ex["user"], ex["assistant"]))

print(f"Category 4 (Multi-Pair Diversity): {len(multi_pair_examples)} examples")


# ============================================================
# WRITE OUTPUT
# ============================================================

output_file = "bastion_risk_v6_reinforcement.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

total = len(examples)
print(f"\nGenerated {total} v6 training examples")
print(f"Saved to {output_file}")
print()
print("Breakdown:")
cats = {
    "Cat 1 (Structure-Intact HOLD)": len(structure_hold_examples),
    "Cat 2 (Structure-Break EXIT)": len(structure_break_examples),
    "Cat 3 (VPVR-Informed Decisions)": len(vpvr_examples),
    "Cat 4 (Multi-Pair Diversity)": len(multi_pair_examples),
}
for name, count in cats.items():
    print(f"  {name}: {count} examples")
print(f"  TOTAL: {total} examples")
print()
print("Pairs covered: BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, BNB, DOT")
print()

# Also create combined file (v5 base + v6 reinforcement)
import os
v5_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bastion_risk_v5_combined.jsonl")
combined_file = "bastion_risk_v6_combined.jsonl"

if os.path.exists(v5_base):
    with open(v5_base, "r", encoding="utf-8") as f:
        v5_lines = f.readlines()
    with open(combined_file, "w", encoding="utf-8") as f:
        for line in v5_lines:
            f.write(line)
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Combined: {len(v5_lines)} v5 + {total} v6 = {len(v5_lines) + total} total")
    print(f"Saved to {combined_file}")
else:
    print(f"WARNING: v5 base not found at {v5_base}")
    print("Run this script from the training_data directory to auto-combine")
