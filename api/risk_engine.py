"""
BASTION Risk Engine — Autonomous TP/SL Management
===================================================
Automated position monitoring with MCF Exit Hierarchy enforcement.
Adaptive polling, data caching, hard-code safety checks, and AI evaluation.

Phase 1: Position State Store, Data Cache, Evaluation Loop, API endpoints
Phase 3: Execution Engine integration — actions now route to real exchange orders
"""

import asyncio
import json
import time
import os
import logging
import httpx
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger("bastion.engine")


# =============================================================================
# POSITION STATE STORE
# =============================================================================

@dataclass
class PositionState:
    """Full MCF-enriched state for a tracked position."""
    position_id: str
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    guarding_line: Optional[float] = None
    trailing_stop: Optional[float] = None
    take_profits: List[dict] = field(default_factory=list)  # [{price, exit_pct, hit}]
    leverage: float = 1.0
    position_size: float = 0.0
    size_usd: float = 0.0
    exchange: str = ""

    # MCF tracking
    r_multiple: float = 0.0
    opened_at: Optional[str] = None
    duration_hours: float = 0.0
    tps_hit: List[int] = field(default_factory=list)  # indices of TPs hit

    # Engine tracking
    last_evaluated_at: Optional[str] = None
    current_urgency: str = "LOW"
    evaluation_count: int = 0
    last_action: str = "HOLD"
    last_confidence: float = 0.0
    consecutive_holds: int = 0
    auto_execute: bool = False
    engine_active: bool = True

    def to_dict(self):
        return asdict(self)

    def calc_r_multiple(self):
        """Calculate current R-multiple based on entry and stop distance."""
        if not self.stop_loss or not self.entry_price:
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0.0
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) / risk
        else:
            return (self.entry_price - self.current_price) / risk

    def calc_duration(self):
        """Calculate how long position has been open."""
        if not self.opened_at:
            return 0.0
        try:
            opened = datetime.fromisoformat(self.opened_at.replace("Z", "+00:00"))
            return (datetime.utcnow() - opened.replace(tzinfo=None)).total_seconds() / 3600
        except:
            return 0.0


class PositionStateStore:
    """In-memory store for position states, keyed by position_id."""

    def __init__(self):
        self.positions: Dict[str, PositionState] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, pos_id: str, **kwargs) -> PositionState:
        """Create or update a position state."""
        async with self._lock:
            if pos_id in self.positions:
                state = self.positions[pos_id]
                for k, v in kwargs.items():
                    if hasattr(state, k):
                        setattr(state, k, v)
            else:
                state = PositionState(position_id=pos_id, **kwargs)
                self.positions[pos_id] = state
            return state

    async def get(self, pos_id: str) -> Optional[PositionState]:
        return self.positions.get(pos_id)

    async def get_all(self) -> List[PositionState]:
        return list(self.positions.values())

    async def remove(self, pos_id: str):
        async with self._lock:
            self.positions.pop(pos_id, None)

    async def sync_from_exchange(self, exchange_positions: List[dict]):
        """
        Sync exchange positions into the store.
        New positions are added, closed positions are removed.
        Existing positions update current_price but preserve MCF state.
        """
        async with self._lock:
            live_ids = set()
            for pos in exchange_positions:
                pid = pos.get("id", f"{pos.get('symbol','?')}_{pos.get('direction','?')}")
                live_ids.add(pid)
                if pid in self.positions:
                    # Update live fields, keep MCF state
                    state = self.positions[pid]
                    state.current_price = pos.get("current_price", state.current_price)
                    state.size_usd = pos.get("size_usd", state.size_usd)
                    state.position_size = pos.get("size", state.position_size)
                    state.leverage = pos.get("leverage", state.leverage)
                else:
                    # New position — initialize with defaults
                    state = PositionState(
                        position_id=pid,
                        symbol=pos.get("symbol", "UNKNOWN").replace("-PERP", "").replace("USDT", ""),
                        direction=pos.get("direction", "LONG").upper(),
                        entry_price=pos.get("entry_price", 0),
                        current_price=pos.get("current_price", 0),
                        stop_loss=pos.get("stop_loss", 0) or 0,
                        leverage=pos.get("leverage", 1),
                        position_size=pos.get("size", 0),
                        size_usd=pos.get("size_usd", 0),
                        exchange=pos.get("exchange", ""),
                        opened_at=pos.get("updated_at", datetime.utcnow().isoformat()),
                    )
                    self.positions[pid] = state

            # Remove positions no longer on exchange
            closed = [pid for pid in self.positions if pid not in live_ids]
            for pid in closed:
                logger.info(f"[ENGINE] Position {pid} closed on exchange, removing from tracker")
                del self.positions[pid]


# =============================================================================
# DATA CACHE
# =============================================================================

class DataCache:
    """TTL-based cache for market data to avoid hammering APIs."""

    def __init__(self):
        self._cache: Dict[str, dict] = {}
        # TTLs in seconds
        self.ttls = {
            "price": 3,
            "funding": 30,
            "oi": 30,
            "whales": 30,
            "helsinki": 15,
            "whale_alert": 60,
        }

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if not expired."""
        entry = self._cache.get(key)
        if entry and time.time() - entry["ts"] < entry["ttl"]:
            return entry["data"]
        return None

    def set(self, key: str, data: Any, category: str = "helsinki"):
        """Cache data with category-based TTL."""
        ttl = self.ttls.get(category, 15)
        self._cache[key] = {"data": data, "ts": time.time(), "ttl": ttl}

    def invalidate(self, pattern: str = ""):
        """Clear cache entries matching pattern, or all if empty."""
        if not pattern:
            self._cache.clear()
        else:
            keys = [k for k in self._cache if pattern in k]
            for k in keys:
                del self._cache[k]


# =============================================================================
# ENGINE AUDIT LOG
# =============================================================================

class EngineAuditLog:
    """Ring-buffer audit trail for engine actions."""

    def __init__(self, max_entries: int = 500):
        self.entries: deque = deque(maxlen=max_entries)

    def log(self, event_type: str, position_id: str, data: dict):
        self.entries.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "position_id": position_id,
            "data": data
        })

    def get_recent(self, limit: int = 50, position_id: str = None) -> List[dict]:
        entries = list(self.entries)
        if position_id:
            entries = [e for e in entries if e["position_id"] == position_id]
        return entries[-limit:]


# =============================================================================
# ENGINE CONFIGURATION
# =============================================================================

@dataclass
class EngineConfig:
    """Runtime-adjustable engine configuration."""
    enabled: bool = False
    auto_execute: bool = False  # If True, executes actions. If False, advisory only.
    poll_interval_low: int = 120       # seconds
    poll_interval_medium: int = 60
    poll_interval_high: int = 15
    poll_interval_critical: int = 5
    position_sync_interval: int = 30   # How often to refresh exchange positions
    max_evaluations_per_minute: int = 10  # Rate limit on AI calls
    hard_stop_enabled: bool = True     # MCF Level 1 — code-enforced
    safety_net_enabled: bool = True    # MCF Level 2 — code-enforced
    confidence_threshold: float = 0.6  # Min confidence to auto-execute

    def get_interval(self, urgency: str) -> int:
        return {
            "LOW": self.poll_interval_low,
            "MEDIUM": self.poll_interval_medium,
            "HIGH": self.poll_interval_high,
            "CRITICAL": self.poll_interval_critical,
        }.get(urgency, self.poll_interval_low)

    def to_dict(self):
        return asdict(self)


# =============================================================================
# RISK ENGINE CORE
# =============================================================================

class RiskEngine:
    """
    BASTION Autonomous Risk Engine.
    Monitors positions, enforces MCF hierarchy, calls AI for evaluation,
    and optionally executes actions on exchanges.
    """

    def __init__(self):
        self.store = PositionStateStore()
        self.cache = DataCache()
        self.audit = EngineAuditLog()
        self.config = EngineConfig()

        # Runtime state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._eval_count_window: List[float] = []  # timestamps for rate limiting
        self._started_at: Optional[str] = None
        self._total_evaluations = 0
        self._total_actions_taken = 0
        self._last_sync_at: float = 0

        # External dependencies (injected on startup)
        self._get_positions_fn = None  # async () -> List[dict]
        self._evaluate_fn = None       # async (position_dict) -> dict
        self._helsinki = None
        self._coinglass = None
        self._whale_alert = None
        self._execution_engine = None  # ExecutionEngine instance
        self._scope_id = "default"     # User scope for execution routing

    def inject_dependencies(self, get_positions_fn=None, evaluate_fn=None,
                             helsinki=None, coinglass=None, whale_alert=None,
                             execution_engine=None, scope_id=None):
        """Inject external service dependencies."""
        self._get_positions_fn = get_positions_fn
        self._evaluate_fn = evaluate_fn
        self._helsinki = helsinki
        self._coinglass = coinglass
        self._whale_alert = whale_alert
        if execution_engine:
            self._execution_engine = execution_engine
        if scope_id:
            self._scope_id = scope_id

    # ─── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> dict:
        """Start the engine background loop."""
        if self._running:
            return {"success": False, "error": "Engine already running"}

        self.config.enabled = True
        self._running = True
        self._started_at = datetime.utcnow().isoformat()
        self._task = asyncio.create_task(self._engine_loop())
        self.audit.log("ENGINE_START", "SYSTEM", {"config": self.config.to_dict()})
        logger.info("[ENGINE] Risk Engine STARTED")

        return {
            "success": True,
            "message": "BASTION Risk Engine activated",
            "started_at": self._started_at
        }

    async def stop(self) -> dict:
        """Stop the engine gracefully."""
        if not self._running:
            return {"success": False, "error": "Engine not running"}

        self._running = False
        self.config.enabled = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.audit.log("ENGINE_STOP", "SYSTEM", {
            "total_evaluations": self._total_evaluations,
            "total_actions": self._total_actions_taken
        })
        logger.info("[ENGINE] Risk Engine STOPPED")

        return {
            "success": True,
            "message": "Engine stopped",
            "total_evaluations": self._total_evaluations
        }

    def status(self) -> dict:
        """Return current engine state."""
        positions = list(self.store.positions.values())
        urgency_counts = {}
        for p in positions:
            urgency_counts[p.current_urgency] = urgency_counts.get(p.current_urgency, 0) + 1

        return {
            "running": self._running,
            "started_at": self._started_at,
            "config": self.config.to_dict(),
            "positions_tracked": len(positions),
            "urgency_breakdown": urgency_counts,
            "total_evaluations": self._total_evaluations,
            "total_actions_taken": self._total_actions_taken,
            "last_sync": datetime.fromtimestamp(self._last_sync_at).isoformat() if self._last_sync_at else None,
            "positions": [
                {
                    "id": p.position_id,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "pnl_r": round(p.calc_r_multiple(), 2),
                    "urgency": p.current_urgency,
                    "last_action": p.last_action,
                    "last_evaluated": p.last_evaluated_at,
                    "eval_count": p.evaluation_count,
                    "auto_execute": p.auto_execute,
                }
                for p in positions
            ]
        }

    # ─── Main Loop ─────────────────────────────────────────────────────

    async def _engine_loop(self):
        """
        Core engine loop. Runs continuously while engine is active.
        1. Sync positions from exchange
        2. For each position, check if evaluation is due
        3. Run MCF hard checks (Level 1-2)
        4. If needed, call AI evaluation (Level 3-6)
        5. Process action recommendation
        """
        logger.info("[ENGINE] Engine loop started")

        while self._running:
            try:
                # Step 1: Sync positions periodically
                now = time.time()
                if now - self._last_sync_at >= self.config.position_sync_interval:
                    await self._sync_positions()
                    self._last_sync_at = now

                # Step 2: Evaluate each position based on urgency schedule
                positions = await self.store.get_all()
                for state in positions:
                    if not state.engine_active or not self._running:
                        continue

                    # Check if evaluation is due
                    if self._is_evaluation_due(state):
                        # Rate limit check
                        if not self._can_evaluate():
                            continue

                        try:
                            await self._evaluate_position(state)
                        except Exception as e:
                            logger.error(f"[ENGINE] Eval failed for {state.symbol}: {e}")
                            self.audit.log("EVAL_ERROR", state.position_id, {"error": str(e)})

                # Sleep briefly between cycles
                await asyncio.sleep(2)

            except asyncio.CancelledError:
                logger.info("[ENGINE] Engine loop cancelled")
                break
            except Exception as e:
                logger.error(f"[ENGINE] Loop error: {e}")
                await asyncio.sleep(5)

        logger.info("[ENGINE] Engine loop exited")

    # ─── Position Sync ─────────────────────────────────────────────────

    async def _sync_positions(self):
        """Fetch positions from exchange and sync with state store."""
        if not self._get_positions_fn:
            return

        try:
            result = self._get_positions_fn
            if asyncio.iscoroutinefunction(self._get_positions_fn):
                result = await self._get_positions_fn()
            elif callable(self._get_positions_fn):
                result = self._get_positions_fn()

            if result and isinstance(result, dict):
                positions = result.get("positions", [])
            elif result and isinstance(result, list):
                positions = result
            else:
                positions = []

            if positions:
                await self.store.sync_from_exchange(positions)
                logger.debug(f"[ENGINE] Synced {len(positions)} positions")

        except Exception as e:
            logger.error(f"[ENGINE] Position sync failed: {e}")

    # ─── Evaluation Scheduling ─────────────────────────────────────────

    def _is_evaluation_due(self, state: PositionState) -> bool:
        """Check if enough time has passed since last evaluation."""
        if not state.last_evaluated_at:
            return True  # Never evaluated

        try:
            last = datetime.fromisoformat(state.last_evaluated_at)
            interval = self.config.get_interval(state.current_urgency)
            return (datetime.utcnow() - last).total_seconds() >= interval
        except:
            return True

    def _can_evaluate(self) -> bool:
        """Rate limiter — max N evaluations per minute."""
        now = time.time()
        # Clean old entries
        self._eval_count_window = [t for t in self._eval_count_window if now - t < 60]
        if len(self._eval_count_window) >= self.config.max_evaluations_per_minute:
            return False
        self._eval_count_window.append(now)
        return True

    # ─── Position Evaluation ───────────────────────────────────────────

    async def _evaluate_position(self, state: PositionState):
        """
        Full evaluation pipeline for a single position.
        1. MCF Level 1 (Hard Stop) — code check
        2. MCF Level 2 (Safety Net) — code check
        3. MCF Level 3-6 — AI evaluation
        """
        state.current_price = state.current_price or state.entry_price
        state.r_multiple = state.calc_r_multiple()
        state.duration_hours = state.calc_duration()

        # ── MCF Level 1: Hard Stop ──
        if self.config.hard_stop_enabled and state.stop_loss > 0:
            triggered = False
            if state.direction == "LONG" and state.current_price <= state.stop_loss:
                triggered = True
            elif state.direction == "SHORT" and state.current_price >= state.stop_loss:
                triggered = True

            if triggered:
                action = {
                    "action": "EXIT_100_PERCENT_IMMEDIATELY",
                    "urgency": "CRITICAL",
                    "reason": f"HARD STOP HIT — Price {state.current_price} breached stop {state.stop_loss}",
                    "confidence": 1.0,
                    "source": "MCF_LEVEL_1_CODE"
                }
                await self._process_action(state, action)
                return

        # ── MCF Level 2: Safety Net (structural break) ──
        # This requires a structural level — we check guarding_line as proxy
        if self.config.safety_net_enabled and state.guarding_line:
            triggered = False
            if state.direction == "LONG" and state.current_price < state.guarding_line:
                distance_pct = (state.guarding_line - state.current_price) / state.guarding_line * 100
                if distance_pct > 1.5:  # More than 1.5% below guarding line
                    triggered = True
            elif state.direction == "SHORT" and state.current_price > state.guarding_line:
                distance_pct = (state.current_price - state.guarding_line) / state.guarding_line * 100
                if distance_pct > 1.5:
                    triggered = True

            if triggered:
                action = {
                    "action": "EXIT_FULL",
                    "urgency": "HIGH",
                    "reason": f"GUARDING LINE BROKEN — Price {state.current_price} vs guarding {state.guarding_line} ({distance_pct:.1f}% breach)",
                    "confidence": 0.9,
                    "source": "MCF_LEVEL_2_CODE"
                }
                await self._process_action(state, action)
                return

        # ── MCF Levels 3-6: AI Evaluation ──
        if self._evaluate_fn:
            try:
                # Build enriched position dict for the existing risk_evaluate function
                pos_dict = {
                    "symbol": state.symbol,
                    "direction": state.direction,
                    "entry_price": state.entry_price,
                    "current_price": state.current_price,
                    "stop_loss": state.stop_loss,
                    "take_profits": [tp.get("price", 0) for tp in state.take_profits] if state.take_profits else [],
                    "position_size": state.position_size,
                    "leverage": state.leverage,
                    "guarding_line": state.guarding_line,
                    "trailing_stop": state.trailing_stop,
                    "r_multiple": state.r_multiple,
                    "duration_hours": state.duration_hours,
                }

                result = await self._evaluate_fn({"position": pos_dict})

                if result and result.get("success") and result.get("evaluation"):
                    ev = result["evaluation"]
                    ev["source"] = "BASTION_AI"
                    await self._process_action(state, ev)
                else:
                    logger.warning(f"[ENGINE] No valid evaluation for {state.symbol}")

            except Exception as e:
                logger.error(f"[ENGINE] AI evaluation failed for {state.symbol}: {e}")

        # Update tracking
        state.last_evaluated_at = datetime.utcnow().isoformat()
        state.evaluation_count += 1
        self._total_evaluations += 1

    # ─── Action Processing ─────────────────────────────────────────────

    async def _process_action(self, state: PositionState, action: dict):
        """
        Process an action recommendation.
        Updates state, logs to audit, and optionally executes.
        """
        action_type = action.get("action", "HOLD")
        urgency = action.get("urgency", "LOW")
        confidence = action.get("confidence", 0.0)
        reason = action.get("reason", "")
        source = action.get("source", "UNKNOWN")

        # Update position state
        state.last_action = action_type
        state.current_urgency = urgency
        state.last_confidence = confidence
        state.last_evaluated_at = datetime.utcnow().isoformat()
        state.evaluation_count += 1

        if action_type == "HOLD":
            state.consecutive_holds += 1
        else:
            state.consecutive_holds = 0

        # Audit log
        self.audit.log("EVALUATION", state.position_id, {
            "action": action_type,
            "urgency": urgency,
            "confidence": confidence,
            "reason": reason,
            "source": source,
            "r_multiple": state.r_multiple,
            "price": state.current_price,
        })

        self._total_evaluations += 1

        # Auto-execute if enabled and confidence meets threshold
        if (self.config.auto_execute and state.auto_execute and
                action_type != "HOLD" and
                confidence >= self.config.confidence_threshold):

            logger.info(f"[ENGINE] AUTO-EXECUTE: {state.symbol} → {action_type} (conf={confidence:.2f})")

            # ── Phase 3: Route to Execution Engine ──
            if self._execution_engine:
                try:
                    exec_result = await self._execution_engine.execute_action(
                        position_state=state.to_dict(),
                        action=action,
                        scope_id=self._scope_id
                    )
                    if exec_result.success:
                        self.audit.log("EXECUTED", state.position_id, {
                            "action": action_type,
                            "confidence": confidence,
                            "order_id": exec_result.order_id,
                            "exchange": exec_result.exchange,
                            "exit_pct": exec_result.exit_pct,
                            "executed_price": exec_result.executed_price,
                        })
                        self._total_actions_taken += 1
                        logger.info(f"[ENGINE] ✓ EXECUTED {action_type} on {state.symbol} "
                                    f"| order={exec_result.order_id} | exit={exec_result.exit_pct*100:.0f}%")
                    else:
                        self.audit.log("EXECUTION_FAILED", state.position_id, {
                            "action": action_type,
                            "confidence": confidence,
                            "error": exec_result.error,
                        })
                        logger.error(f"[ENGINE] ✗ EXECUTION FAILED {state.symbol}: {exec_result.error}")
                except Exception as exec_err:
                    self.audit.log("EXECUTION_ERROR", state.position_id, {
                        "action": action_type,
                        "error": str(exec_err),
                    })
                    logger.error(f"[ENGINE] Execution exception for {state.symbol}: {exec_err}")
            else:
                # No execution engine wired — log as advisory
                self.audit.log("AUTO_EXECUTE_NO_ENGINE", state.position_id, {
                    "action": action_type,
                    "confidence": confidence,
                    "note": "Execution engine not connected"
                })
                self._total_actions_taken += 1
                logger.warning(f"[ENGINE] AUTO-EXECUTE queued but no execution engine wired: "
                               f"{state.symbol} → {action_type}")

        elif action_type != "HOLD":
            logger.info(f"[ENGINE] ADVISORY: {state.symbol} → {action_type} [{urgency}] {reason}")

    # ─── Manual Controls ───────────────────────────────────────────────

    async def override_position(self, position_id: str, overrides: dict) -> dict:
        """Manual override: update MCF state for a position."""
        state = await self.store.get(position_id)
        if not state:
            return {"success": False, "error": f"Position {position_id} not found"}

        changed = []
        for key in ["stop_loss", "guarding_line", "trailing_stop", "take_profits",
                     "auto_execute", "engine_active", "current_urgency"]:
            if key in overrides:
                old = getattr(state, key, None)
                setattr(state, key, overrides[key])
                changed.append(f"{key}: {old} → {overrides[key]}")

        self.audit.log("OVERRIDE", position_id, {"changes": changed})
        return {"success": True, "changes": changed, "position": state.to_dict()}

    async def force_evaluate(self, position_id: str) -> dict:
        """Force immediate evaluation of a specific position."""
        state = await self.store.get(position_id)
        if not state:
            return {"success": False, "error": f"Position {position_id} not found"}

        # Reset last_evaluated to force eval
        state.last_evaluated_at = None
        await self._evaluate_position(state)
        return {
            "success": True,
            "position_id": position_id,
            "action": state.last_action,
            "urgency": state.current_urgency,
            "confidence": state.last_confidence
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

risk_engine = RiskEngine()
