"""
BASTION-32B Risk Model Backtester
=================================
Brutally tests the fine-tuned model by:
1. Fetching real historical BTC/ETH/SOL price data (1h candles)
2. Simulating positions at various entries with realistic parameters
3. Sending each scenario to the live BASTION-32B model via the API
4. Comparing the model's recommendation against what ACTUALLY happened
5. Computing win rate, expected value, and performance metrics

Usage:
    python backtest/bastion_backtest.py --days 30 --symbol BTC
    python backtest/bastion_backtest.py --days 60 --symbol BTC --scenarios 100
    python backtest/bastion_backtest.py --mode paper  (live paper trading logger)
"""

import asyncio
import json
import time
import os
import sys
import random
import argparse
import logging
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("backtest")

# --- Configuration ----------------------------------------------
BASTION_API_URL = "http://localhost:8003"  # Local backend
MODEL_URL = os.getenv("BASTION_MODEL_URL", "http://70.78.148.208:56333")
MODEL_API_KEY = os.getenv("BASTION_MODEL_API_KEY", "")

# Public APIs for historical data (fallback chain)
BINANCE_API = "https://api.binance.com/api/v3"
BINANCE_US_API = "https://api.binance.us/api/v3"
BYBIT_API = "https://api.bybit.com/v5/market"

# Evaluation horizons: how far ahead we look to judge the model's call
EVAL_HORIZONS = {
    "1h": 1,
    "4h": 4,
    "12h": 12,
    "24h": 24,
}

# --- Data Classes -----------------------------------------------

@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class SimulatedPosition:
    """A simulated position at a point in time."""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    current_price: float
    stop_loss: float
    take_profits: List[float]
    leverage: int
    position_size: float
    entry_time: datetime
    eval_time: datetime  # When we ask the model
    candle_index: int  # Index into candles array
    duration_hours: float = 0
    r_multiple: float = 0.0

@dataclass
class EvaluationResult:
    """Model's evaluation + what actually happened."""
    position: SimulatedPosition
    model_action: str
    model_confidence: float
    model_urgency: str
    model_reason: str
    model_reasoning: Dict = field(default_factory=dict)
    model_exit_pct: int = 0
    model_stop_price: Optional[float] = None

    # What actually happened after the model's call
    price_1h: Optional[float] = None
    price_4h: Optional[float] = None
    price_12h: Optional[float] = None
    price_24h: Optional[float] = None
    max_favorable: float = 0.0  # Best price in our direction
    max_adverse: float = 0.0    # Worst price against us
    hit_stop: bool = False
    hit_tp1: bool = False

    # Scoring
    correct: Optional[bool] = None
    score: float = 0.0
    pnl_if_followed: float = 0.0  # Estimated P&L if we followed the advice

    # Meta
    data_sources: List[str] = field(default_factory=list)
    eval_time_ms: float = 0.0
    error: Optional[str] = None


# --- Historical Data Fetcher ------------------------------------

async def fetch_klines(symbol: str, interval: str = "1h",
                       days: int = 30) -> List[Candle]:
    """
    Fetch historical klines with fallback chain:
    1. Bybit (most reliable, no geo-restrictions)
    2. Binance US
    3. Binance Global
    """
    # Try Bybit first (most reliable, no geo-block)
    candles = await _fetch_bybit_klines(symbol, interval, days)
    if candles:
        return candles

    # Fallback to Binance US
    candles = await _fetch_klines(BINANCE_US_API, symbol, interval, days)
    if candles:
        return candles

    # Last resort: Binance Global
    candles = await _fetch_klines(BINANCE_API, symbol, interval, days)
    return candles


async def _fetch_bybit_klines(symbol: str, interval: str = "1h",
                               days: int = 30) -> List[Candle]:
    """Fetch klines from Bybit V5 API."""
    pair = f"{symbol}USDT"

    # Bybit interval mapping
    interval_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    bybit_interval = interval_map.get(interval, "60")

    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    all_candles = []
    current_end = end_time

    async with httpx.AsyncClient() as client:
        while current_end > start_time:
            try:
                resp = await client.get(
                    f"{BYBIT_API}/kline",
                    params={
                        "category": "linear",
                        "symbol": pair,
                        "interval": bybit_interval,
                        "start": start_time,
                        "end": current_end,
                        "limit": 1000
                    },
                    timeout=30.0
                )

                if resp.status_code != 200:
                    logger.warning(f"Bybit API error: {resp.status_code}")
                    break

                data = resp.json()
                result_list = data.get("result", {}).get("list", [])
                if not result_list:
                    break

                batch = []
                for k in result_list:
                    # Bybit format: [timestamp, open, high, low, close, volume, turnover]
                    batch.append(Candle(
                        timestamp=int(k[0]),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5])
                    ))

                # Bybit returns newest first, reverse for chronological
                batch.reverse()
                all_candles = batch + all_candles if not all_candles else batch + all_candles

                # Bybit paginates backwards
                oldest_ts = int(result_list[-1][0])
                if oldest_ts <= start_time:
                    break
                current_end = oldest_ts - 1

                if len(result_list) < 1000:
                    break

                await asyncio.sleep(0.15)

            except Exception as e:
                logger.warning(f"Bybit fetch error: {e}")
                break

    # Deduplicate and sort
    seen = set()
    unique = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    unique.sort(key=lambda c: c.timestamp)

    # Filter to requested range
    unique = [c for c in unique if c.timestamp >= start_time]

    logger.info(f"[Bybit] Fetched {len(unique)} {interval} candles for {symbol} ({days} days)")
    return unique


async def _fetch_klines(api_base: str, symbol: str,
                                 interval: str = "1h",
                                 days: int = 30) -> List[Candle]:
    """Fetch historical klines from Binance (global or US)."""
    pair = f"{symbol}USDT"
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    all_candles = []
    current_start = start_time

    async with httpx.AsyncClient() as client:
        while current_start < end_time:
            try:
                resp = await client.get(
                    f"{api_base}/klines",
                    params={
                        "symbol": pair,
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": end_time,
                        "limit": 1000
                    },
                    timeout=30.0
                )

                if resp.status_code != 200:
                    logger.warning(f"Binance API ({api_base}) error: {resp.status_code}")
                    break

                data = resp.json()
                if isinstance(data, dict) and data.get("code"):
                    logger.warning(f"Binance API error: {data.get('msg', 'unknown')}")
                    break

                if not data or not isinstance(data, list):
                    break

                for k in data:
                    all_candles.append(Candle(
                        timestamp=k[0],
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5])
                    ))

                current_start = data[-1][0] + 1

                if len(data) < 1000:
                    break

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Binance fetch error: {e}")
                break

    source = "Binance US" if "binance.us" in api_base else "Binance"
    logger.info(f"[{source}] Fetched {len(all_candles)} {interval} candles for {symbol} ({days} days)")
    return all_candles


# --- Scenario Generator ----------------------------------------

def generate_scenarios(candles: List[Candle], symbol: str,
                       num_scenarios: int = 50,
                       min_candles_ahead: int = 24) -> List[SimulatedPosition]:
    """
    Generate realistic trading scenarios from historical data.

    Creates positions at various points with:
    - Random entry points (avoiding the last 24 candles for forward-looking eval)
    - Both LONG and SHORT positions
    - Various leverages (1x, 3x, 5x, 10x, 20x)
    - With and without stop losses
    - Different hold durations (fresh entry vs held for hours)
    """
    scenarios = []
    max_idx = len(candles) - min_candles_ahead - 1

    if max_idx < 50:
        logger.error("Not enough candle data for meaningful backtesting")
        return []

    leverages = [1, 3, 5, 10, 20]

    for i in range(num_scenarios):
        # Pick a random candle as the "current" point
        eval_idx = random.randint(50, max_idx)
        current_candle = candles[eval_idx]
        current_price = current_candle.close

        # Randomly choose LONG or SHORT
        direction = random.choice(["LONG", "SHORT"])

        # Entry was 1-48 candles ago
        lookback = random.randint(1, min(48, eval_idx - 10))
        entry_idx = eval_idx - lookback
        entry_candle = candles[entry_idx]
        entry_price = entry_candle.close

        # Calculate actual P&L direction
        if direction == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        leverage = random.choice(leverages)

        # Stop loss: 70% chance of having one
        stop_loss = 0.0
        if random.random() < 0.70:
            atr = _calc_atr(candles, eval_idx, 14)
            if direction == "LONG":
                stop_loss = entry_price - atr * random.uniform(1.0, 3.0)
            else:
                stop_loss = entry_price + atr * random.uniform(1.0, 3.0)

        # Take profits: 50% chance
        take_profits = []
        if random.random() < 0.50:
            atr = _calc_atr(candles, eval_idx, 14)
            if direction == "LONG":
                take_profits = [
                    round(entry_price + atr * random.uniform(1.5, 3.0), 2),
                    round(entry_price + atr * random.uniform(3.0, 5.0), 2),
                ]
            else:
                take_profits = [
                    round(entry_price - atr * random.uniform(1.5, 3.0), 2),
                    round(entry_price - atr * random.uniform(3.0, 5.0), 2),
                ]

        # R-multiple (if stop loss exists)
        r_multiple = 0.0
        if stop_loss > 0:
            risk = abs(entry_price - stop_loss)
            if risk > 0:
                if direction == "LONG":
                    r_multiple = (current_price - entry_price) / risk
                else:
                    r_multiple = (entry_price - current_price) / risk

        entry_time = datetime.utcfromtimestamp(entry_candle.timestamp / 1000)
        eval_time = datetime.utcfromtimestamp(current_candle.timestamp / 1000)

        scenarios.append(SimulatedPosition(
            symbol=symbol,
            direction=direction,
            entry_price=round(entry_price, 2),
            current_price=round(current_price, 2),
            stop_loss=round(stop_loss, 2) if stop_loss else 0,
            take_profits=take_profits,
            leverage=leverage,
            position_size=round(random.uniform(0.01, 1.0), 4),
            entry_time=entry_time,
            eval_time=eval_time,
            candle_index=eval_idx,
            duration_hours=lookback,
            r_multiple=round(r_multiple, 2)
        ))

    logger.info(f"Generated {len(scenarios)} scenarios "
                f"({sum(1 for s in scenarios if s.direction == 'LONG')} LONG, "
                f"{sum(1 for s in scenarios if s.direction == 'SHORT')} SHORT)")
    return scenarios


def _calc_atr(candles: List[Candle], idx: int, period: int = 14) -> float:
    """Calculate ATR at a given candle index."""
    if idx < period:
        return candles[idx].high - candles[idx].low

    trs = []
    for i in range(idx - period, idx):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i-1].close),
            abs(candles[i].low - candles[i-1].close)
        )
        trs.append(tr)

    return sum(trs) / len(trs) if trs else 0


# --- Model Evaluator -------------------------------------------

async def evaluate_position_via_api(position: SimulatedPosition) -> Dict:
    """Send a position to the BASTION API for evaluation."""
    payload = {
        "position": {
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "position_size": position.position_size,
            "leverage": position.leverage,
            "stop_loss": position.stop_loss if position.stop_loss else None,
            "take_profits": position.take_profits if position.take_profits else [],
            "r_multiple": position.r_multiple,
            "duration_hours": position.duration_hours,
        }
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=30.0)) as client:
        start = time.time()
        resp = await client.post(
            f"{BASTION_API_URL}/api/risk/evaluate",
            json=payload,
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code != 200:
            return {"error": f"API error: {resp.status_code}", "elapsed_ms": elapsed}

        data = resp.json()
        data["elapsed_ms"] = elapsed
        return data


async def evaluate_position_direct(position: SimulatedPosition,
                                    market_context: str = "") -> Dict:
    """
    Send position directly to the vLLM model (bypasses live data fetch).
    Used for pure model testing without needing live data APIs.
    """
    position_state = f"""POSITION STATE:
- Asset: {position.symbol}/USDT
- Direction: {position.direction}
- Entry: ${position.entry_price:,.2f}
- Current Price: ${position.current_price:,.2f}
- P&L: {position.r_multiple:+.1f}R
- Stop Loss: {"$" + f"{position.stop_loss:,.2f}" if position.stop_loss else "NONE (no stop set -- HIGH RISK)"}"""

    if position.take_profits:
        for i, tp in enumerate(position.take_profits):
            position_state += f"\n- TP{i+1}: ${tp:,.2f}"
    position_state += f"\n- Leverage: {position.leverage}x"
    position_state += f"\n- Duration: {position.duration_hours}h"

    system_prompt = f"""You are BASTION Risk Intelligence -- an autonomous trade management AI for institutional crypto trading.

MCF EXIT HIERARCHY (check in this EXACT order -- first trigger wins):
1) HARD STOP -- Maximum loss threshold. NON-NEGOTIABLE. Exit 100% immediately.
2) SAFETY NET BREAK -- Long-term structural support/resistance violated. Exit 100% immediately.
3) GUARDING LINE BREAK -- Dynamic trailing structure broken. Exit 50-75%.
4) TAKE PROFIT TARGETS -- T1: Exit 30-50%. T2: Exit 30-40% remaining. T3: Runners.
5) TRAILING STOP -- ATR-based dynamic stop. Exit remaining if hit.
6) TIME EXIT -- Max holding exceeded with no progress. Exit 50% gradually.

CORE PHILOSOPHY: Exit on STRUCTURE BREAKS, not arbitrary targets. Let winners run when structure holds.

{market_context}

RESPOND WITH ONLY VALID JSON. No markdown, no code fences, no text before or after.

{{
  "action": "HOLD|TP_PARTIAL|TP_FULL|MOVE_STOP_TO_BREAKEVEN|TRAIL_STOP|EXIT_FULL|REDUCE_SIZE|ADJUST_STOP|EXIT_100_PERCENT_IMMEDIATELY",
  "urgency": "LOW|MEDIUM|HIGH|CRITICAL",
  "confidence": 0.0-1.0,
  "reason": "One clear sentence explaining the decision",
  "reasoning": {{
    "structure_analysis": "trend/levels assessment",
    "data_assessment": "market data signals",
    "risk_factors": "risk considerations",
    "exit_logic": "why this specific action"
  }},
  "execution": {{
    "exit_pct": 0,
    "stop_price": null,
    "order_type": "NONE|MARKET|STOP_MARKET"
  }}
}}"""

    user_message = f"""{position_state}

MARKET CONTEXT:
{market_context if market_context else "Limited data available -- base decision on position state, R-multiple, leverage risk, and stop loss placement."}

DECISION REQUIRED: Evaluate this {position.direction} position using MCF exit hierarchy. What action should be taken?"""

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=30.0)) as client:
        headers = {"Content-Type": "application/json"}
        if MODEL_API_KEY:
            headers["Authorization"] = f"Bearer {MODEL_API_KEY}"

        start = time.time()
        resp = await client.post(
            f"{MODEL_URL}/v1/chat/completions",
            headers=headers,
            json={
                "model": "bastion-32b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 800,
                "temperature": 0.3
            }
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code != 200:
            return {"error": f"Model error: {resp.status_code}", "elapsed_ms": elapsed}

        result = resp.json()
        raw = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse JSON from response
        parsed = None
        try:
            parsed = json.loads(raw.strip())
        except:
            try:
                import re
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
            except:
                pass

        return {
            "evaluation": parsed,
            "raw_response": raw,
            "data_sources": ["BASTION-32B-DIRECT"],
            "elapsed_ms": elapsed
        }


# --- Outcome Evaluator -----------------------------------------

def evaluate_outcome(position: SimulatedPosition,
                     model_result: Dict,
                     candles: List[Candle]) -> EvaluationResult:
    """
    Compare model's recommendation against what actually happened.

    Looks at price action in the 1h/4h/12h/24h after the model's call
    to determine if the advice was correct.
    """
    eval = model_result.get("evaluation")

    result = EvaluationResult(
        position=position,
        model_action=eval.get("action", "UNKNOWN") if eval else "PARSE_ERROR",
        model_confidence=eval.get("confidence", 0) if eval else 0,
        model_urgency=eval.get("urgency", "UNKNOWN") if eval else "UNKNOWN",
        model_reason=eval.get("reason", "") if eval else model_result.get("error", ""),
        model_reasoning=eval.get("reasoning", {}) if eval else {},
        model_exit_pct=eval.get("execution", {}).get("exit_pct", 0) if eval else 0,
        model_stop_price=eval.get("execution", {}).get("stop_price") if eval else None,
        data_sources=model_result.get("data_sources", []),
        eval_time_ms=model_result.get("elapsed_ms", 0),
        error=model_result.get("error"),
    )

    if not eval:
        result.error = model_result.get("error", "No evaluation returned")
        return result

    # Get future candles
    idx = position.candle_index
    max_idx = len(candles) - 1

    # Future prices at each horizon
    for horizon_name, hours in EVAL_HORIZONS.items():
        future_idx = min(idx + hours, max_idx)
        future_price = candles[future_idx].close
        setattr(result, f"price_{horizon_name}", future_price)

    # Max favorable and adverse excursion in next 24h
    direction = position.direction
    max_fav = position.current_price
    max_adv = position.current_price

    for i in range(idx + 1, min(idx + 25, max_idx + 1)):
        if direction == "LONG":
            max_fav = max(max_fav, candles[i].high)
            max_adv = min(max_adv, candles[i].low)
        else:
            max_fav = min(max_fav, candles[i].low)
            max_adv = max(max_adv, candles[i].high)

    result.max_favorable = max_fav
    result.max_adverse = max_adv

    # Did price hit the stop loss in next 24h?
    if position.stop_loss:
        for i in range(idx + 1, min(idx + 25, max_idx + 1)):
            if direction == "LONG" and candles[i].low <= position.stop_loss:
                result.hit_stop = True
                break
            elif direction == "SHORT" and candles[i].high >= position.stop_loss:
                result.hit_stop = True
                break

    # Did price hit TP1?
    if position.take_profits:
        tp1 = position.take_profits[0]
        for i in range(idx + 1, min(idx + 25, max_idx + 1)):
            if direction == "LONG" and candles[i].high >= tp1:
                result.hit_tp1 = True
                break
            elif direction == "SHORT" and candles[i].low <= tp1:
                result.hit_tp1 = True
                break

    # -- Score the model's recommendation --
    action = result.model_action.upper()

    # Price movement in our favor or against
    price_4h = result.price_4h or position.current_price
    price_24h = result.price_24h or position.current_price

    if direction == "LONG":
        move_4h_pct = (price_4h - position.current_price) / position.current_price * 100
        move_24h_pct = (price_24h - position.current_price) / position.current_price * 100
        max_drawdown_pct = (max_adv - position.current_price) / position.current_price * 100
        max_runup_pct = (max_fav - position.current_price) / position.current_price * 100
    else:
        move_4h_pct = (position.current_price - price_4h) / position.current_price * 100
        move_24h_pct = (position.current_price - price_24h) / position.current_price * 100
        max_drawdown_pct = (position.current_price - max_adv) / position.current_price * 100
        max_runup_pct = (position.current_price - max_fav) / position.current_price * 100

    # Categorize the action
    is_exit = action in ["EXIT_FULL", "EXIT_100_PERCENT_IMMEDIATELY", "TP_FULL"]
    is_reduce = action in ["TP_PARTIAL", "REDUCE_SIZE"]
    is_hold = action in ["HOLD", "MOVE_STOP_TO_BREAKEVEN", "TRAIL_STOP", "ADJUST_STOP"]

    # -- Check if stop was ALREADY breached at time of model call --
    stop_already_breached = False
    if position.stop_loss:
        if direction == "LONG" and position.current_price <= position.stop_loss:
            stop_already_breached = True
        elif direction == "SHORT" and position.current_price >= position.stop_loss:
            stop_already_breached = True

    # -- Determine correctness --
    if is_exit:
        # If stop was already breached, EXIT is ALWAYS correct (proper risk mgmt)
        if stop_already_breached:
            result.correct = True
            result.score = 1.0
            result.pnl_if_followed = 0  # Flat after exit
        # EXIT was correct if price moved against us within 24h
        # (model saved us from a drawdown)
        elif move_24h_pct < -0.5:  # Price went against us >0.5%
            result.correct = True
            result.score = 1.0
            result.pnl_if_followed = 0  # Flat after exit
        elif max_drawdown_pct < -2.0:  # Big drawdown even if recovered
            result.correct = True
            result.score = 0.7
            result.pnl_if_followed = 0
        elif move_24h_pct > 1.0:  # Price went IN our favor -- exit was wrong
            result.correct = False
            result.score = -1.0
            result.pnl_if_followed = -move_24h_pct * position.leverage  # Missed profit
        else:
            result.correct = None  # Neutral -- marginal move
            result.score = 0.0
            result.pnl_if_followed = 0

    elif is_reduce:
        # REDUCE was correct if price eventually moved against us
        exit_pct = result.model_exit_pct or 30
        remaining_pct = 100 - exit_pct

        if move_24h_pct < -1.0:  # Against us
            result.correct = True
            result.score = 0.8
            result.pnl_if_followed = move_24h_pct * position.leverage * (remaining_pct / 100)
        elif move_24h_pct > 1.5:  # In our favor -- reduce was slightly wrong
            result.correct = False
            result.score = -0.5
            result.pnl_if_followed = move_24h_pct * position.leverage * (remaining_pct / 100)
        else:
            result.correct = None
            result.score = 0.2  # Reducing is generally prudent
            result.pnl_if_followed = move_24h_pct * position.leverage * (remaining_pct / 100)

    elif is_hold:
        # If stop is ALREADY breached and model says HOLD, that's always wrong
        if stop_already_breached:
            result.correct = False
            result.score = -2.0  # HOLD with breached stop = critical error
            result.pnl_if_followed = move_24h_pct * position.leverage
        # HOLD was correct if price moved in our favor
        elif move_4h_pct > 0.3:
            result.correct = True
            result.score = 1.0
            result.pnl_if_followed = move_24h_pct * position.leverage
        elif result.hit_stop:
            result.correct = False
            result.score = -1.5  # HOLD but hit stop = bad
            sl_dist = abs(position.current_price - position.stop_loss) / position.current_price * 100
            result.pnl_if_followed = -sl_dist * position.leverage
        elif move_24h_pct < -2.0:
            result.correct = False
            result.score = -1.0
            result.pnl_if_followed = move_24h_pct * position.leverage
        else:
            result.correct = True  # Sideways is fine for HOLD
            result.score = 0.3
            result.pnl_if_followed = move_24h_pct * position.leverage

    return result


# --- Synthetic Market Context ----------------------------------

def build_synthetic_context(candles: List[Candle], idx: int, symbol: str) -> str:
    """
    Build a realistic market context string from historical candle data.
    This simulates what the live data APIs would provide.
    """
    if idx < 20:
        return "LIMITED DATA"

    current = candles[idx]
    prev = candles[idx - 1]

    # Price change
    change_1h = (current.close - prev.close) / prev.close * 100

    # 24h high/low
    h24 = max(c.high for c in candles[max(0, idx-24):idx+1])
    l24 = min(c.low for c in candles[max(0, idx-24):idx+1])

    # ATR
    atr = _calc_atr(candles, idx, 14)

    # Volatility regime
    atr_pct = atr / current.close * 100
    if atr_pct > 3.0:
        vol_regime = "HIGH"
    elif atr_pct > 1.5:
        vol_regime = "MEDIUM"
    else:
        vol_regime = "LOW"

    # Simple RSI calculation
    gains, losses = [], []
    for i in range(idx - 14, idx):
        diff = candles[i+1].close - candles[i].close
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))
    avg_gain = sum(gains) / 14 if gains else 0
    avg_loss = sum(losses) / 14 if losses else 0.001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Volume trend
    recent_vol = sum(c.volume for c in candles[idx-4:idx+1]) / 5
    older_vol = sum(c.volume for c in candles[idx-24:idx-4]) / 20 if idx > 24 else recent_vol
    vol_ratio = recent_vol / older_vol if older_vol > 0 else 1.0

    # Trend (simple SMA20 vs SMA50)
    sma20 = sum(c.close for c in candles[idx-20:idx]) / 20 if idx >= 20 else current.close
    sma50 = sum(c.close for c in candles[idx-50:idx]) / 50 if idx >= 50 else sma20

    if current.close > sma20 > sma50:
        trend = "BULLISH"
    elif current.close < sma20 < sma50:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL/CHOPPY"

    # CVD approximation (volume delta)
    buy_vol = sum(c.volume for c in candles[idx-4:idx+1] if c.close > c.open)
    sell_vol = sum(c.volume for c in candles[idx-4:idx+1] if c.close <= c.open)
    cvd_bias = "BULLISH" if buy_vol > sell_vol * 1.2 else ("BEARISH" if sell_vol > buy_vol * 1.2 else "NEUTRAL")

    # Random but realistic funding rate
    funding = random.uniform(-0.015, 0.025)

    # Random but realistic OI change
    oi_change = random.uniform(-5, 8)

    context = f"""PRICE: ${current.close:,.2f} | 1h Change: {change_1h:+.2f}%
24H RANGE: ${l24:,.2f} - ${h24:,.2f}
FUNDING RATE: {funding:.4f}% ({("crowded LONG" if funding > 0.01 else "crowded SHORT" if funding < -0.01 else "neutral")})
OI CHANGE (4h): {oi_change:+.1f}% ({("new money entering" if oi_change > 2 else "deleveraging" if oi_change < -2 else "stable")})
CVD 1H: {cvd_bias} confirmation
VOLATILITY: Regime={vol_regime} | ATR(14)=${atr:,.2f} ({atr_pct:.2f}%)
MOMENTUM: RSI(14)={rsi:.1f} | Trend={trend}
VOLUME: {"Elevated" if vol_ratio > 1.5 else "Normal" if vol_ratio > 0.7 else "Low"} ({vol_ratio:.1f}x avg)
SMART MONEY: {"accumulating" if trend == "BULLISH" and rsi < 60 else "distributing" if trend == "BEARISH" and rsi > 40 else "neutral"}"""

    return context


# --- Main Backtest Runner --------------------------------------

async def run_backtest(symbol: str = "BTC", days: int = 30,
                       num_scenarios: int = 50,
                       use_api: bool = True,
                       delay_between: float = 2.0) -> List[EvaluationResult]:
    """
    Run the full backtest pipeline.

    Args:
        symbol: Trading pair (BTC, ETH, SOL)
        days: How many days of historical data
        num_scenarios: Number of simulated positions to test
        use_api: If True, use the full API (with live data).
                 If False, use direct model calls with synthetic context.
        delay_between: Seconds to wait between API calls (rate limiting)
    """
    print(f"\n{'='*70}")
    print(f"  BASTION-32B BACKTEST -- {symbol}/USDT")
    print(f"  {days} days | {num_scenarios} scenarios | {'API mode' if use_api else 'Direct model'}")
    print(f"{'='*70}\n")

    # Step 1: Fetch historical data
    print("[1/4] Fetching historical candle data...")
    candles = await fetch_klines(symbol, "1h", days)
    if len(candles) < 100:
        print(f"ERROR: Only got {len(candles)} candles. Need at least 100.")
        return []

    # Step 2: Generate scenarios
    print(f"[2/4] Generating {num_scenarios} trading scenarios...")
    scenarios = generate_scenarios(candles, symbol, num_scenarios)

    # Step 3: Evaluate each scenario through the model
    print(f"[3/4] Evaluating {len(scenarios)} scenarios through BASTION-32B...")
    results = []

    for i, position in enumerate(scenarios):
        progress = f"[{i+1}/{len(scenarios)}]"
        direction_marker = "^" if position.direction == "LONG" else "v"
        pnl_sign = "+" if position.r_multiple >= 0 else ""

        print(f"  {progress} [{direction_marker}] {position.direction} {symbol} "
              f"entry=${position.entry_price:,.0f} -> ${position.current_price:,.0f} "
              f"({pnl_sign}{position.r_multiple}R) "
              f"lev={position.leverage}x "
              f"SL={'$'+str(int(position.stop_loss)) if position.stop_loss else 'NONE'} ",
              end="", flush=True)

        try:
            if use_api:
                model_result = await evaluate_position_via_api(position)
            else:
                context = build_synthetic_context(candles, position.candle_index, symbol)
                model_result = await evaluate_position_direct(position, context)

            result = evaluate_outcome(position, model_result, candles)
            results.append(result)

            # Print result
            action = result.model_action
            conf = result.model_confidence
            correct_str = "[OK]" if result.correct == True else ("[X]" if result.correct == False else "[-]")
            print(f"-> {action} ({conf:.0%}) {correct_str} "
                  f"[{result.eval_time_ms:.0f}ms]")

        except Exception as e:
            print(f"-> ERROR: {e}")
            results.append(EvaluationResult(
                position=position,
                model_action="ERROR",
                model_confidence=0,
                model_urgency="UNKNOWN",
                model_reason=str(e),
                error=str(e)
            ))

        # Rate limiting
        if delay_between > 0 and i < len(scenarios) - 1:
            await asyncio.sleep(delay_between)

    # Step 4: Analyze results
    print(f"\n[4/4] Analyzing results...")
    print_report(results, symbol, days)

    # Save results
    save_results(results, symbol, days)

    return results


# --- Paper Trading Logger --------------------------------------

async def run_paper_trader(interval_minutes: int = 60):
    """
    Live paper trading mode.
    Periodically evaluates a synthetic BTC position and logs results.
    Checks back on previous predictions to score them.
    """
    print(f"\n{'='*70}")
    print(f"  BASTION-32B PAPER TRADER")
    print(f"  Evaluating every {interval_minutes} minutes")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*70}\n")

    log_file = Path(__file__).parent / "paper_trade_log.jsonl"
    predictions = []

    while True:
        try:
            # Get current BTC price
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{BYBIT_API}/tickers",
                    params={"category": "linear", "symbol": "BTCUSDT"},
                    timeout=10.0
                )
                data = resp.json()
                ticker = data.get("result", {}).get("list", [{}])[0]
                current_price = float(ticker.get("lastPrice", 0))
                change_24h = float(ticker.get("price24hPcnt", 0)) * 100

            # Create a simulated LONG position (typical trader)
            entry_offset = random.uniform(-0.02, 0.02)  # 2% range
            entry_price = current_price * (1 + entry_offset)

            position = SimulatedPosition(
                symbol="BTC",
                direction="LONG",
                entry_price=round(entry_price, 2),
                current_price=round(current_price, 2),
                stop_loss=round(entry_price * 0.97, 2),  # 3% stop
                take_profits=[round(entry_price * 1.05, 2)],  # 5% TP
                leverage=10,
                position_size=0.1,
                entry_time=datetime.utcnow() - timedelta(hours=random.randint(1, 24)),
                eval_time=datetime.utcnow(),
                candle_index=0,
                duration_hours=random.randint(1, 24),
                r_multiple=round((current_price - entry_price) / (entry_price * 0.03), 2)
            )

            # Evaluate
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluating BTC LONG "
                  f"entry=${position.entry_price:,.0f} current=${current_price:,.0f} "
                  f"({position.r_multiple:+.1f}R)")

            result = await evaluate_position_via_api(position)
            eval_data = result.get("evaluation", {})

            if eval_data:
                action = eval_data.get("action", "?")
                confidence = eval_data.get("confidence", 0)
                reason = eval_data.get("reason", "")
                urgency = eval_data.get("urgency", "?")

                print(f"  -> {action} | Confidence: {confidence:.0%} | Urgency: {urgency}")
                print(f"  -> Reason: {reason}")

                # Log prediction
                prediction = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "price_at_eval": current_price,
                    "entry_price": position.entry_price,
                    "direction": position.direction,
                    "leverage": position.leverage,
                    "stop_loss": position.stop_loss,
                    "r_multiple": position.r_multiple,
                    "action": action,
                    "confidence": confidence,
                    "urgency": urgency,
                    "reason": reason,
                    "price_1h_later": None,
                    "price_4h_later": None,
                    "scored": False
                }
                predictions.append(prediction)

                # Save to file
                with open(log_file, "a") as f:
                    f.write(json.dumps(prediction) + "\n")

            # Score previous predictions
            await score_past_predictions(predictions)

            print(f"\n  Waiting {interval_minutes} minutes for next evaluation...")
            await asyncio.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\nPaper trader stopped.")
            print(f"Log saved to: {log_file}")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            await asyncio.sleep(60)


async def score_past_predictions(predictions: List[Dict]):
    """Check back on previous predictions to see if they were correct."""
    if not predictions:
        return

    now = datetime.utcnow()
    scored_count = 0

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BYBIT_API}/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            timeout=10.0
        )
        data = resp.json()
        current_price = float(data.get("result", {}).get("list", [{}])[0].get("lastPrice", 0))

    for pred in predictions:
        if pred.get("scored"):
            continue

        eval_time = datetime.fromisoformat(pred["timestamp"])
        hours_ago = (now - eval_time).total_seconds() / 3600

        if hours_ago >= 1 and pred["price_1h_later"] is None:
            pred["price_1h_later"] = current_price

        if hours_ago >= 4 and pred["price_4h_later"] is None:
            pred["price_4h_later"] = current_price
            pred["scored"] = True
            scored_count += 1

            # Score it
            direction = pred["direction"]
            eval_price = pred["price_at_eval"]

            if direction == "LONG":
                move_pct = (current_price - eval_price) / eval_price * 100
            else:
                move_pct = (eval_price - current_price) / eval_price * 100

            action = pred["action"]
            is_exit = action in ["EXIT_FULL", "EXIT_100_PERCENT_IMMEDIATELY", "TP_FULL"]

            if is_exit and move_pct < -0.5:
                verdict = "[OK] CORRECT EXIT (price dropped)"
            elif is_exit and move_pct > 1.0:
                verdict = "[X] WRONG EXIT (missed rally)"
            elif not is_exit and move_pct > 0.3:
                verdict = "[OK] CORRECT HOLD (price rose)"
            elif not is_exit and move_pct < -1.0:
                verdict = "[X] WRONG HOLD (should have exited)"
            else:
                verdict = "[-] NEUTRAL"

            print(f"\n  [SCORECARD] {pred['timestamp'][:16]} "
                  f"BTC ${eval_price:,.0f} -> ${current_price:,.0f} ({move_pct:+.2f}%)")
            print(f"  Model said: {action} ({pred['confidence']:.0%}) -> {verdict}")

    if scored_count > 0:
        print(f"  Scored {scored_count} past prediction(s)")


# --- Report Generator ------------------------------------------

def print_report(results: List[EvaluationResult], symbol: str, days: int):
    """Print a comprehensive performance report."""

    valid = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    if not valid:
        print("\nNo valid results to analyze!")
        return

    correct = [r for r in valid if r.correct == True]
    wrong = [r for r in valid if r.correct == False]
    neutral = [r for r in valid if r.correct is None]

    total = len(valid)
    scored = len(correct) + len(wrong)

    print(f"\n{'='*70}")
    print(f"  BASTION-32B PERFORMANCE REPORT")
    print(f"  {symbol}/USDT | {days}-day backtest | {total} scenarios")
    print(f"{'='*70}")

    # Overall accuracy
    win_rate = len(correct) / scored * 100 if scored > 0 else 0
    print(f"\n  +--- OVERALL ACCURACY -------------------------")
    print(f"  | Win Rate:     {win_rate:.1f}% ({len(correct)}/{scored} scored)")
    print(f"  | Correct:      {len(correct)} [OK]")
    print(f"  | Wrong:        {len(wrong)} [X]")
    print(f"  | Neutral:      {len(neutral)} [-]")
    print(f"  | Errors:       {len(errors)} [!]")
    print(f"  +-----------------------------------------------")

    # Action distribution
    action_counts = {}
    for r in valid:
        a = r.model_action
        if a not in action_counts:
            action_counts[a] = {"total": 0, "correct": 0, "wrong": 0, "neutral": 0}
        action_counts[a]["total"] += 1
        if r.correct == True:
            action_counts[a]["correct"] += 1
        elif r.correct == False:
            action_counts[a]["wrong"] += 1
        else:
            action_counts[a]["neutral"] += 1

    print(f"\n  +--- ACTION BREAKDOWN -------------------------")
    print(f"  | {'Action':<28} {'Count':>5} {'Win%':>6} {'OK':>4} {'X':>4}")
    print(f"  | {'-'*50}")
    for action, counts in sorted(action_counts.items(), key=lambda x: -x[1]["total"]):
        s = counts["correct"] + counts["wrong"]
        wr = counts["correct"] / s * 100 if s > 0 else 0
        print(f"  | {action:<28} {counts['total']:>5} {wr:>5.1f}% {counts['correct']:>4} {counts['wrong']:>4}")
    print(f"  +-----------------------------------------------")

    # Confidence analysis
    print(f"\n  +--- CONFIDENCE CALIBRATION -------------------")
    conf_buckets = {"Low (0-40%)": [], "Med (40-70%)": [], "High (70-90%)": [], "Very High (90%+)": []}
    for r in valid:
        c = r.model_confidence
        if c < 0.4:
            conf_buckets["Low (0-40%)"].append(r)
        elif c < 0.7:
            conf_buckets["Med (40-70%)"].append(r)
        elif c < 0.9:
            conf_buckets["High (70-90%)"].append(r)
        else:
            conf_buckets["Very High (90%+)"].append(r)

    for bucket_name, bucket_results in conf_buckets.items():
        if not bucket_results:
            continue
        c = sum(1 for r in bucket_results if r.correct == True)
        w = sum(1 for r in bucket_results if r.correct == False)
        s = c + w
        wr = c / s * 100 if s > 0 else 0
        print(f"  | {bucket_name:<20} n={len(bucket_results):>3} | Win: {wr:>5.1f}% ({c}/{s})")
    print(f"  +-----------------------------------------------")

    # P&L analysis
    total_pnl = sum(r.pnl_if_followed for r in valid)
    avg_pnl = total_pnl / len(valid) if valid else 0

    exit_results = [r for r in valid if r.model_action in ["EXIT_FULL", "EXIT_100_PERCENT_IMMEDIATELY", "TP_FULL"]]
    hold_results = [r for r in valid if r.model_action in ["HOLD", "MOVE_STOP_TO_BREAKEVEN", "TRAIL_STOP", "ADJUST_STOP"]]

    print(f"\n  +--- P&L IMPACT (IF FOLLOWED) ----------------")
    print(f"  | Total est. P&L:   {total_pnl:>+.2f}% (leveraged)")
    print(f"  | Avg per trade:    {avg_pnl:>+.2f}%")
    if exit_results:
        exit_pnl = sum(r.pnl_if_followed for r in exit_results)
        print(f"  | EXIT signals:     {exit_pnl:>+.2f}% ({len(exit_results)} trades)")
    if hold_results:
        hold_pnl = sum(r.pnl_if_followed for r in hold_results)
        print(f"  | HOLD signals:     {hold_pnl:>+.2f}% ({len(hold_results)} trades)")
    print(f"  +-----------------------------------------------")

    # Leverage analysis
    print(f"\n  +--- LEVERAGE SENSITIVITY ---------------------")
    for lev in [1, 3, 5, 10, 20]:
        lev_results = [r for r in valid if r.position.leverage == lev]
        if not lev_results:
            continue
        c = sum(1 for r in lev_results if r.correct == True)
        w = sum(1 for r in lev_results if r.correct == False)
        s = c + w
        wr = c / s * 100 if s > 0 else 0
        print(f"  | {lev:>2}x leverage:  n={len(lev_results):>3} | Win: {wr:>5.1f}% ({c}/{s})")
    print(f"  +-----------------------------------------------")

    # Worst calls
    worst = sorted([r for r in valid if r.correct == False], key=lambda r: r.score)[:5]
    if worst:
        print(f"\n  +--- WORST CALLS ------------------------------")
        for r in worst:
            print(f"  | {r.position.direction} {symbol} ${r.position.current_price:,.0f} "
                  f"-> said {r.model_action} ({r.model_confidence:.0%}) "
                  f"| P&L impact: {r.pnl_if_followed:+.1f}%")
            print(f"  |   Reason: {r.model_reason[:80]}")
        print(f"  +-----------------------------------------------")

    # Response time
    times = [r.eval_time_ms for r in valid if r.eval_time_ms > 0]
    if times:
        print(f"\n  +--- RESPONSE TIME ----------------------------")
        print(f"  | Average:  {sum(times)/len(times):,.0f}ms")
        print(f"  | Median:   {sorted(times)[len(times)//2]:,.0f}ms")
        print(f"  | Min:      {min(times):,.0f}ms")
        print(f"  | Max:      {max(times):,.0f}ms")
        print(f"  +-----------------------------------------------")

    print(f"\n{'='*70}")
    print(f"  VERDICT: {'[PASS] MODEL VIABLE' if win_rate >= 60 else '[WARN] NEEDS MORE DATA' if win_rate >= 50 else '[FAIL] MODEL NEEDS RETRAINING'}")
    print(f"  Win Rate: {win_rate:.1f}% | Required: 60%+ for live trading")
    print(f"{'='*70}\n")


def save_results(results: List[EvaluationResult], symbol: str, days: int):
    """Save results to JSON for further analysis."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{symbol}_{days}d_{timestamp}.json"
    filepath = output_dir / filename

    output = {
        "meta": {
            "symbol": symbol,
            "days": days,
            "total_scenarios": len(results),
            "timestamp": datetime.utcnow().isoformat(),
        },
        "results": []
    }

    for r in results:
        output["results"].append({
            "position": {
                "symbol": r.position.symbol,
                "direction": r.position.direction,
                "entry_price": r.position.entry_price,
                "current_price": r.position.current_price,
                "stop_loss": r.position.stop_loss,
                "leverage": r.position.leverage,
                "r_multiple": r.position.r_multiple,
                "duration_hours": r.position.duration_hours,
            },
            "model": {
                "action": r.model_action,
                "confidence": r.model_confidence,
                "urgency": r.model_urgency,
                "reason": r.model_reason,
                "exit_pct": r.model_exit_pct,
            },
            "outcome": {
                "price_1h": r.price_1h,
                "price_4h": r.price_4h,
                "price_12h": r.price_12h,
                "price_24h": r.price_24h,
                "max_favorable": r.max_favorable,
                "max_adverse": r.max_adverse,
                "hit_stop": r.hit_stop,
                "hit_tp1": r.hit_tp1,
                "correct": r.correct,
                "score": r.score,
                "pnl_if_followed": r.pnl_if_followed,
            },
            "meta": {
                "eval_time_ms": r.eval_time_ms,
                "data_sources": r.data_sources,
                "error": r.error,
            }
        })

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {filepath}")


# --- CLI Entry Point --------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="BASTION-32B Risk Model Backtester")
    parser.add_argument("--mode", choices=["backtest", "paper"], default="backtest",
                       help="backtest = historical replay, paper = live paper trading")
    parser.add_argument("--symbol", default="BTC", help="Symbol to test (BTC, ETH, SOL)")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--scenarios", type=int, default=50, help="Number of scenarios to test")
    parser.add_argument("--direct", action="store_true",
                       help="Use direct model calls (bypasses live data APIs)")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Seconds between API calls")
    parser.add_argument("--paper-interval", type=int, default=60,
                       help="Minutes between paper trade evaluations")

    args = parser.parse_args()

    if args.mode == "paper":
        await run_paper_trader(args.paper_interval)
    else:
        results = await run_backtest(
            symbol=args.symbol,
            days=args.days,
            num_scenarios=args.scenarios,
            use_api=not args.direct,
            delay_between=args.delay,
        )

        if results:
            valid = [r for r in results if r.error is None]
            correct = sum(1 for r in valid if r.correct == True)
            wrong = sum(1 for r in valid if r.correct == False)
            scored = correct + wrong
            win_rate = correct / scored * 100 if scored > 0 else 0

            print(f"\n  FINAL: {win_rate:.1f}% win rate across {scored} scored scenarios")
            if win_rate >= 60:
                print("  [PASS] Model passes minimum viability threshold")
            else:
                print("  [FAIL] Model below 60% threshold -- do NOT arm execution")


if __name__ == "__main__":
    asyncio.run(main())
