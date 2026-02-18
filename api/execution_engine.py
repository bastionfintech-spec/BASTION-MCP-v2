"""
BASTION Execution Engine — Action-to-Order Translator
=======================================================
Translates risk engine action recommendations into actual exchange orders.
Respects MCF Exit Hierarchy, position sizing rules, and safety guards.

The execution engine sits between the risk_engine (which decides WHAT to do)
and the exchange_connector (which talks to exchanges). It:
  1. Validates actions before execution
  2. Calculates exact order parameters (qty, price, side)
  3. Routes orders to the correct exchange client
  4. Logs all executions with full audit trail
  5. Enforces safety limits (max position close %, daily loss limit, etc.)
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from collections import deque

logger = logging.getLogger("bastion.execution")

# Structure-aware TP sizer (deterministic math, not LLM)
try:
    from core.tp_sizer import StructureTPSizer
    _tp_sizer = StructureTPSizer()
    logger.info("[EXEC] Structure-aware TP sizer loaded")
except ImportError:
    _tp_sizer = None
    logger.warning("[EXEC] TP sizer not available — falling back to static percentages")


# =============================================================================
# EXECUTION RESULT
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    success: bool
    action: str                   # The MCF action that triggered this
    symbol: str
    exchange: str = ""
    order_id: str = ""
    side: str = ""                # BUY or SELL
    quantity: float = 0.0
    executed_price: Optional[float] = None
    exit_pct: float = 0.0        # What % of position was closed
    error: str = ""
    timestamp: str = ""
    confidence: float = 0.0
    urgency: str = "LOW"
    source: str = ""              # MCF_LEVEL_1_CODE, MCF_LEVEL_2_CODE, AI_EVALUATION
    metadata: Optional[Dict[str, Any]] = None  # TP sizing details, structure data

    def to_dict(self):
        return asdict(self)


# =============================================================================
# SAFETY GUARDS
# =============================================================================

@dataclass
class SafetyConfig:
    """Safety limits for the execution engine."""
    # Maximum % of a position that can be closed in a single action
    max_single_close_pct: float = 1.0       # 100% (no limit by default)
    # Minimum confidence required to auto-execute
    min_confidence: float = 0.6
    # Maximum number of executions per position per hour
    max_executions_per_hour: int = 10
    # Maximum number of total executions per hour across all positions
    max_total_executions_per_hour: int = 50
    # Daily loss limit (USD) — if exceeded, engine stops executing
    daily_loss_limit_usd: float = 0.0       # 0 = disabled
    # Require confirmation for EXIT_FULL and EXIT_100_PERCENT_IMMEDIATELY
    require_confirm_full_exit: bool = False  # If True, queues for user approval
    # Kill switch: if True, ALL execution is halted
    kill_switch: bool = False


# =============================================================================
# EXECUTION AUDIT LOG
# =============================================================================

class ExecutionAuditLog:
    """
    Immutable audit trail for all execution attempts.

    Dual-write: keeps recent entries in memory (fast reads) AND persists
    every entry to Supabase bastion_execution_log table (survives redeploys).
    If Supabase is unavailable, falls back to memory-only gracefully.
    """

    def __init__(self, max_entries: int = 5000):
        self._entries: deque = deque(maxlen=max_entries)
        self._lock = asyncio.Lock()
        self._db_client = None   # Injected from terminal_api.py
        self._db_table = "bastion_execution_log"
        self._total_count_db: int = 0  # Tracks total including DB-only entries

    def set_db_client(self, supabase_client):
        """Inject Supabase client for persistence."""
        self._db_client = supabase_client
        logger.info("[AUDIT] Supabase persistence enabled for execution log")

    async def log(self, result: ExecutionResult, position_state: Optional[dict] = None):
        async with self._lock:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "result": result.to_dict(),
                "position_state_snapshot": position_state,
            }
            self._entries.append(entry)

        # Persist to Supabase (non-blocking, errors don't break execution)
        if self._db_client:
            try:
                row = {
                    "action": result.action,
                    "symbol": result.symbol,
                    "exchange": result.exchange or "",
                    "success": result.success,
                    "order_id": result.order_id or "",
                    "side": result.side or "",
                    "quantity": result.quantity,
                    "executed_price": result.executed_price or 0,
                    "exit_pct": result.exit_pct,
                    "confidence": result.confidence,
                    "urgency": result.urgency or "LOW",
                    "source": result.source or "",
                    "error": result.error or "",
                    "metadata": json.dumps(result.metadata) if result.metadata else "{}",
                    "position_state": json.dumps(position_state) if position_state else "{}",
                }
                self._db_client.table(self._db_table).insert(row).execute()
                self._total_count_db += 1
            except Exception as e:
                logger.warning(f"[AUDIT] Supabase insert failed (memory-only): {e}")

    async def load_recent_from_db(self, limit: int = 200):
        """Load recent entries from Supabase on startup to populate in-memory cache."""
        if not self._db_client:
            return
        try:
            result = self._db_client.table(self._db_table) \
                .select("*", count="exact") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            if result.data:
                # Insert oldest first so newest are at the end
                for row in reversed(result.data):
                    entry = {
                        "timestamp": row.get("created_at", ""),
                        "result": {
                            "action": row.get("action", ""),
                            "symbol": row.get("symbol", ""),
                            "exchange": row.get("exchange", ""),
                            "success": row.get("success", False),
                            "order_id": row.get("order_id", ""),
                            "side": row.get("side", ""),
                            "quantity": row.get("quantity", 0),
                            "executed_price": row.get("executed_price"),
                            "exit_pct": row.get("exit_pct", 0),
                            "confidence": row.get("confidence", 0),
                            "urgency": row.get("urgency", "LOW"),
                            "source": row.get("source", ""),
                            "error": row.get("error", ""),
                            "metadata": row.get("metadata"),
                        },
                        "position_state_snapshot": row.get("position_state"),
                    }
                    self._entries.append(entry)
                self._total_count_db = result.count or len(result.data)
                logger.info(f"[AUDIT] Loaded {len(result.data)} entries from DB ({self._total_count_db} total)")
        except Exception as e:
            logger.warning(f"[AUDIT] Could not load from DB (table may not exist): {e}")

    def get_history(self, limit: int = 100, position_id: Optional[str] = None,
                    action_filter: Optional[str] = None) -> List[dict]:
        entries = list(self._entries)
        if position_id:
            entries = [e for e in entries if
                       e.get("position_state_snapshot", {}).get("position_id") == position_id]
        if action_filter:
            entries = [e for e in entries if e["result"].get("action") == action_filter]
        return entries[-limit:]

    def get_executions_since(self, since: datetime) -> List[dict]:
        return [e for e in self._entries
                if datetime.fromisoformat(e["timestamp"]) >= since]

    @property
    def total_count(self) -> int:
        return max(len(self._entries), self._total_count_db)


# =============================================================================
# ACTION → ORDER MAPPING
# =============================================================================

# MCF action type → execution behavior
ACTION_EXECUTION_MAP = {
    "HOLD": {
        "execute": False,
        "description": "No action — structure holding"
    },
    "TP_PARTIAL": {
        "execute": True,
        "order_type": "close_partial_and_trail",
        "default_exit_pct": 0.33,   # Fallback only — AI should specify via execution.exit_pct
        "description": "Take partial profit and trail stop on remainder"
    },
    "TP_FULL": {
        "execute": True,
        "order_type": "close_full",
        "default_exit_pct": 1.0,
        "description": "Take full profit — all targets hit"
    },
    "MOVE_STOP_TO_BREAKEVEN": {
        "execute": True,
        "order_type": "set_stop",
        "stop_at": "entry",         # Move SL to entry price
        "description": "Lock in — move SL to breakeven"
    },
    "TRAIL_STOP": {
        "execute": True,
        "order_type": "set_stop",
        "stop_at": "dynamic",       # Calculated trailing stop
        "description": "Update trailing stop based on ATR"
    },
    "EXIT_FULL": {
        "execute": True,
        "order_type": "close_full",
        "default_exit_pct": 1.0,
        "description": "Structure break — exit 100%"
    },
    "REDUCE_SIZE": {
        "execute": True,
        "order_type": "close_partial",
        "default_exit_pct": 0.50,   # Fallback only — AI should specify via execution.exit_pct
        "description": "Risk management — reduce exposure"
    },
    "ADJUST_STOP": {
        "execute": True,
        "order_type": "set_stop",
        "stop_at": "dynamic",
        "description": "Move stop to new level"
    },
    "EXIT_100_PERCENT_IMMEDIATELY": {
        "execute": True,
        "order_type": "close_full",
        "default_exit_pct": 1.0,
        "priority": "CRITICAL",
        "description": "EMERGENCY — hard stop breach, exit NOW"
    },
}


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Translates risk engine actions into exchange orders.

    Flow:
    1. Risk engine calls execute_action(position_state, action_dict)
    2. Engine validates action against safety guards
    3. Engine calculates exact order parameters
    4. Engine routes to the correct exchange client
    5. Engine logs result and returns ExecutionResult
    """

    def __init__(self):
        self.safety = SafetyConfig()
        self.audit = ExecutionAuditLog()
        self._user_contexts: Dict[str, Any] = {}  # scope_id → UserContextService
        self._position_map: Dict[str, Any] = {}   # position_id → Position object
        self._execution_counts: Dict[str, List[float]] = {}  # position_id → [timestamps]
        self._daily_pnl: float = 0.0
        self._daily_reset: datetime = datetime.utcnow().replace(hour=0, minute=0, second=0)
        self._pending_confirmations: deque = deque(maxlen=100)
        self._total_executions: int = 0
        self._total_volume_usd: float = 0.0
        self._structure_context_fn: Optional[Callable] = None  # Injected from terminal_api
        self._event_listeners: List[Callable] = []  # For SSE notifications

    # ─── Context Registration ─────────────────────────────────────────

    def register_user_context(self, scope_id: str, user_context):
        """Register a user's exchange context for execution routing."""
        self._user_contexts[scope_id] = user_context
        logger.info(f"[EXEC] Registered user context: {scope_id}")

    def register_position(self, position_id: str, position):
        """Register a position object for order routing."""
        self._position_map[position_id] = position

    def unregister_position(self, position_id: str):
        """Remove a position from the execution map."""
        self._position_map.pop(position_id, None)

    # ─── Safety Checks ────────────────────────────────────────────────

    def _check_kill_switch(self) -> Optional[str]:
        if self.safety.kill_switch:
            return "KILL SWITCH ACTIVE — all execution halted"
        return None

    def _check_rate_limit(self, position_id: str) -> Optional[str]:
        """Check if we've exceeded execution rate limits."""
        now = time.time()
        one_hour_ago = now - 3600

        # Per-position rate limit + evict stale entries
        if position_id in self._execution_counts:
            recent = [t for t in self._execution_counts[position_id] if t > one_hour_ago]
            if recent:
                self._execution_counts[position_id] = recent
            else:
                del self._execution_counts[position_id]  # Evict empty
            if len(recent) >= self.safety.max_executions_per_hour:
                return f"Rate limit: {len(recent)}/{self.safety.max_executions_per_hour} executions per hour for this position"

        # Periodic cleanup: evict all stale position keys
        stale_keys = [k for k, v in self._execution_counts.items()
                      if not any(t > one_hour_ago for t in v)]
        for k in stale_keys:
            del self._execution_counts[k]

        # Global rate limit
        total_recent = sum(
            len([t for t in timestamps if t > one_hour_ago])
            for timestamps in self._execution_counts.values()
        )
        if total_recent >= self.safety.max_total_executions_per_hour:
            return f"Global rate limit: {total_recent}/{self.safety.max_total_executions_per_hour} executions per hour"

        return None

    def _check_daily_loss(self) -> Optional[str]:
        """Check if daily loss limit has been hit."""
        if self.safety.daily_loss_limit_usd <= 0:
            return None
        # Reset at UTC midnight
        now = datetime.utcnow()
        if now.date() > self._daily_reset.date():
            self._daily_pnl = 0.0
            self._daily_reset = now.replace(hour=0, minute=0, second=0)
        if self._daily_pnl < -self.safety.daily_loss_limit_usd:
            return f"Daily loss limit breached: ${abs(self._daily_pnl):.2f} > ${self.safety.daily_loss_limit_usd:.2f}"
        return None

    def _check_confidence(self, confidence: float) -> Optional[str]:
        if confidence < self.safety.min_confidence:
            return f"Confidence {confidence:.2f} below threshold {self.safety.min_confidence:.2f}"
        return None

    # ─── Main Execution Entry Point ───────────────────────────────────

    async def execute_action(
        self,
        position_state: dict,
        action: dict,
        scope_id: str = "default"
    ) -> ExecutionResult:
        """
        Execute a risk engine action on a real exchange.

        Args:
            position_state: PositionState as dict (from risk_engine)
            action: Action dict with {action, urgency, confidence, reason, source, execution}
            scope_id: User scope for routing to correct exchange

        Returns:
            ExecutionResult with success/failure details
        """
        action_type = action.get("action", "HOLD")
        urgency = action.get("urgency", "LOW")
        confidence = action.get("confidence", 0.0)
        reason = action.get("reason", "")
        source = action.get("source", "UNKNOWN")
        execution_hints = action.get("execution", {})
        symbol = position_state.get("symbol", "UNKNOWN")
        position_id = position_state.get("position_id", "")

        # ── Check if action requires execution ──
        action_spec = ACTION_EXECUTION_MAP.get(action_type)
        if not action_spec or not action_spec.get("execute"):
            return ExecutionResult(
                success=True, action=action_type, symbol=symbol,
                error="", timestamp=datetime.utcnow().isoformat(),
                confidence=confidence, urgency=urgency, source=source
            )

        # ── Safety checks (in priority order) ──
        kill_err = self._check_kill_switch()
        if kill_err:
            return self._fail(action_type, symbol, kill_err, confidence, urgency, source)

        # CRITICAL urgency bypasses rate limits and confidence checks
        is_critical = urgency == "CRITICAL" or action_type == "EXIT_100_PERCENT_IMMEDIATELY"

        if not is_critical:
            rate_err = self._check_rate_limit(position_id)
            if rate_err:
                return self._fail(action_type, symbol, rate_err, confidence, urgency, source)

            conf_err = self._check_confidence(confidence)
            if conf_err:
                return self._fail(action_type, symbol, conf_err, confidence, urgency, source)

        loss_err = self._check_daily_loss()
        if loss_err and not is_critical:
            return self._fail(action_type, symbol, loss_err, confidence, urgency, source)

        # ── Resolve exchange client ──
        user_ctx = self._user_contexts.get(scope_id)
        if not user_ctx:
            return self._fail(action_type, symbol, f"No user context for scope: {scope_id}",
                              confidence, urgency, source)

        position = self._position_map.get(position_id)
        if not position:
            # Try to find by symbol in user context
            try:
                positions = await user_ctx.get_all_positions()
                for p in positions:
                    sym_clean = p.symbol.replace("-PERP", "").replace("USDT", "")
                    if sym_clean == symbol or p.symbol == symbol:
                        position = p
                        self._position_map[position_id] = p
                        break
            except Exception:
                pass

        if not position:
            return self._fail(action_type, symbol, f"Position not found: {position_id}",
                              confidence, urgency, source)

        # ── Route to correct order type ──
        order_type = action_spec.get("order_type", "")

        try:
            if order_type == "close_full":
                result = await self._execute_close(
                    user_ctx, position, position_state, action_spec, execution_hints, 1.0
                )
            elif order_type == "close_partial":
                exit_pct = self._resolve_exit_pct(action_spec, execution_hints, position_state)
                result = await self._execute_close(
                    user_ctx, position, position_state, action_spec, execution_hints, exit_pct
                )
            elif order_type == "set_stop":
                result = await self._execute_stop_update(
                    user_ctx, position, position_state, action_spec, execution_hints
                )
            elif order_type == "set_tp":
                result = await self._execute_tp_update(
                    user_ctx, position, position_state, action_spec, execution_hints
                )
            elif order_type == "close_partial_and_trail":
                # TP_PARTIAL with structure-aware sizing
                tp_size_result = None
                if _tp_sizer and self._structure_context_fn:
                    try:
                        # Get fresh structure context for sizing
                        struct_ctx = await self._get_structure_context(symbol, position_state)
                        momentum_data = self._extract_momentum(position_state)
                        tp_size_result = _tp_sizer.calculate_exit_pct(
                            structure_context=struct_ctx,
                            momentum_data=momentum_data,
                            position_state=position_state,
                            tp_index=position_state.get("tps_hit_count", 0)
                        )
                        exit_pct = tp_size_result.exit_pct
                        logger.info(f"[EXEC] Structure-sized TP: {exit_pct:.0%} | {tp_size_result.reasoning}")
                    except Exception as e:
                        logger.warning(f"[EXEC] TP sizer failed, falling back: {e}")
                        exit_pct = self._resolve_exit_pct(action_spec, execution_hints, position_state)
                else:
                    exit_pct = self._resolve_exit_pct(action_spec, execution_hints, position_state)

                result = await self._execute_close(
                    user_ctx, position, position_state, action_spec, execution_hints, exit_pct
                )

                # If partial close succeeded, trail the stop on the remainder
                if result.success:
                    # Use structure-derived trail stop if available
                    trail_price = None
                    if tp_size_result and tp_size_result.trail_stop_price:
                        trail_price = tp_size_result.trail_stop_price
                        execution_hints = {**execution_hints, "stop_price": trail_price}

                    trail_result = await self._execute_stop_update(
                        user_ctx, position, position_state, action_spec, execution_hints
                    )
                    if trail_result.success:
                        logger.info(f"[EXEC] ✓ Trailed SL to ${trail_price or 'dynamic'} after "
                                    f"{exit_pct:.0%} partial close on {symbol}")

                    # Store sizing metadata on the result for notifications
                    if tp_size_result:
                        result.metadata = {
                            "tp_sizing": {
                                "exit_pct": tp_size_result.exit_pct,
                                "adjustments": tp_size_result.adjustments,
                                "reasoning": tp_size_result.reasoning,
                                "trail_stop": tp_size_result.trail_stop_price,
                                "confidence": tp_size_result.confidence,
                            }
                        }
            else:
                result = self._fail(action_type, symbol, f"Unknown order type: {order_type}",
                                    confidence, urgency, source)
        except Exception as e:
            logger.error(f"[EXEC] Execution error for {symbol}: {e}", exc_info=True)
            result = self._fail(action_type, symbol, f"Execution exception: {str(e)}",
                                confidence, urgency, source)

        # ── Post-execution bookkeeping ──
        result.action = action_type
        result.confidence = confidence
        result.urgency = urgency
        result.source = source

        if result.success:
            # Track rate limits
            if position_id not in self._execution_counts:
                self._execution_counts[position_id] = []
            self._execution_counts[position_id].append(time.time())
            self._total_executions += 1
            self._total_volume_usd += (result.quantity * (result.executed_price or position.current_price))

        # Audit log
        await self.audit.log(result, position_state)

        # Broadcast execution event for dashboard notifications
        if result.success and action_type != "HOLD":
            await self._broadcast_event({
                "type": "execution",
                "action": action_type,
                "symbol": symbol,
                "exit_pct": getattr(result, 'exit_pct', None),
                "price": result.executed_price,
                "order_id": result.order_id,
                "confidence": confidence,
                "urgency": urgency,
                "reason": reason,
                "source": source,
                "exchange": result.exchange,
                "metadata": getattr(result, 'metadata', None),
                "timestamp": result.timestamp,
            })

        return result

    # ─── Order Execution Methods ──────────────────────────────────────

    async def _execute_close(
        self, user_ctx, position, position_state: dict,
        action_spec: dict, hints: dict, exit_pct: float
    ) -> ExecutionResult:
        """Execute a position close (full or partial)."""
        symbol = position.symbol
        exchange = position.exchange

        # Clamp exit_pct
        exit_pct = min(exit_pct, self.safety.max_single_close_pct)
        exit_pct = max(exit_pct, 0.01)  # At least 1%

        if exit_pct >= 0.99:
            # Full close
            order_result = await user_ctx.close_position(position)
        else:
            order_result = await user_ctx.close_position_partial(position, exit_pct)

        if order_result.success:
            logger.info(f"[EXEC] ✓ CLOSED {exit_pct*100:.0f}% of {symbol} on {exchange} "
                        f"| order={order_result.order_id}")
            return ExecutionResult(
                success=True, action="", symbol=symbol, exchange=exchange,
                order_id=order_result.order_id, side=order_result.side,
                quantity=order_result.quantity,
                executed_price=order_result.executed_price,
                exit_pct=exit_pct,
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            logger.error(f"[EXEC] ✗ CLOSE FAILED {symbol}: {order_result.error}")
            return ExecutionResult(
                success=False, action="", symbol=symbol, exchange=exchange,
                error=order_result.error, exit_pct=exit_pct,
                timestamp=datetime.utcnow().isoformat()
            )

    async def _execute_stop_update(
        self, user_ctx, position, position_state: dict,
        action_spec: dict, hints: dict
    ) -> ExecutionResult:
        """Execute a stop-loss update."""
        symbol = position.symbol
        exchange = position.exchange
        stop_at = action_spec.get("stop_at", "dynamic")

        # Determine new stop price
        if stop_at == "entry":
            new_stop = position_state.get("entry_price", 0)
        elif stop_at == "dynamic":
            # Priority: execution.stop_price (new) → stop_adjustment (legacy) → position state
            new_stop = self._parse_stop_from_hints(hints, position_state)
        else:
            new_stop = position_state.get("stop_loss", 0)

        if not new_stop or new_stop <= 0:
            return ExecutionResult(
                success=False, action="", symbol=symbol, exchange=exchange,
                error=f"Invalid stop price: {new_stop}",
                timestamp=datetime.utcnow().isoformat()
            )

        # Validate stop direction (don't set stop above entry for SHORT, below entry for LONG in wrong direction)
        direction = position_state.get("direction", "LONG").upper()
        current = position.current_price
        entry = position.entry_price

        # For breakeven moves, the stop IS the entry — always valid
        if stop_at != "entry":
            # For LONG: new stop should be below current price
            if direction == "LONG" and new_stop >= current:
                return ExecutionResult(
                    success=False, action="", symbol=symbol, exchange=exchange,
                    error=f"LONG stop {new_stop} >= current price {current}",
                    timestamp=datetime.utcnow().isoformat()
                )
            # For SHORT: new stop should be above current price
            if direction == "SHORT" and new_stop <= current:
                return ExecutionResult(
                    success=False, action="", symbol=symbol, exchange=exchange,
                    error=f"SHORT stop {new_stop} <= current price {current}",
                    timestamp=datetime.utcnow().isoformat()
                )

        order_result = await user_ctx.set_stop_loss(position, new_stop)

        if order_result.success:
            logger.info(f"[EXEC] ✓ SL updated {symbol} → ${new_stop:,.2f} on {exchange}")
            return ExecutionResult(
                success=True, action="", symbol=symbol, exchange=exchange,
                order_id=order_result.order_id, executed_price=new_stop,
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            logger.error(f"[EXEC] ✗ SL UPDATE FAILED {symbol}: {order_result.error}")
            return ExecutionResult(
                success=False, action="", symbol=symbol, exchange=exchange,
                error=order_result.error,
                timestamp=datetime.utcnow().isoformat()
            )

    async def _execute_tp_update(
        self, user_ctx, position, position_state: dict,
        action_spec: dict, hints: dict
    ) -> ExecutionResult:
        """Execute a take-profit order placement/update."""
        symbol = position.symbol
        exchange = position.exchange

        # Get TP price from execution hints or action spec
        tp_price = None
        if hints:
            tp_price = hints.get("tp_price") or hints.get("take_profit") or hints.get("stop_price")
        if not tp_price:
            # Try take_profits list from position state
            tps = position_state.get("take_profits", [])
            if tps:
                tp_price = tps[0]  # Use first TP target

        if not tp_price or tp_price <= 0:
            return ExecutionResult(
                success=False, action="", symbol=symbol, exchange=exchange,
                error=f"No valid TP price found",
                timestamp=datetime.utcnow().isoformat()
            )

        # Validate TP direction
        direction = position_state.get("direction", "LONG").upper()
        current = position.current_price

        if direction == "LONG" and tp_price <= current:
            return ExecutionResult(
                success=False, action="", symbol=symbol, exchange=exchange,
                error=f"LONG TP {tp_price} <= current price {current}",
                timestamp=datetime.utcnow().isoformat()
            )
        if direction == "SHORT" and tp_price >= current:
            return ExecutionResult(
                success=False, action="", symbol=symbol, exchange=exchange,
                error=f"SHORT TP {tp_price} >= current price {current}",
                timestamp=datetime.utcnow().isoformat()
            )

        # Get optional quantity for partial TP
        qty = None
        if hints:
            exit_pct = hints.get("exit_pct")
            if exit_pct and 0 < exit_pct <= 100:
                qty = round(position.size * (exit_pct / 100.0), 8)

        order_result = await user_ctx.set_take_profit(position, tp_price, quantity=qty)

        if order_result.success:
            logger.info(f"[EXEC] ✓ TP set {symbol} → ${tp_price:,.2f} on {exchange}"
                        + (f" (qty={qty})" if qty else " (full position)"))
            return ExecutionResult(
                success=True, action="", symbol=symbol, exchange=exchange,
                order_id=order_result.order_id, executed_price=tp_price,
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            logger.error(f"[EXEC] ✗ TP SET FAILED {symbol}: {order_result.error}")
            return ExecutionResult(
                success=False, action="", symbol=symbol, exchange=exchange,
                error=order_result.error,
                timestamp=datetime.utcnow().isoformat()
            )

    # ─── Helpers ──────────────────────────────────────────────────────

    def _resolve_exit_pct(self, action_spec: dict, hints: dict,
                          position_state: dict) -> float:
        """
        Determine what % of the position to close.

        Priority:
        1. AI-provided execution.exit_pct (integer 1-100) — new standardized format
        2. AI-provided execution.primary_action text parsing — legacy format
        3. Position state take_profits — structural targets
        4. Action spec default — hardcoded fallback
        """
        # ── 1. New standardized format: execution.exit_pct (integer 1-100) ──
        exit_pct_raw = hints.get("exit_pct")
        if exit_pct_raw is not None:
            try:
                pct = float(exit_pct_raw)
                if 1 <= pct <= 100:
                    return pct / 100.0  # Convert 25 → 0.25
                elif 0 < pct < 1:
                    return pct          # Already a fraction
            except (TypeError, ValueError):
                pass

        # ── 2. Legacy format: parse "Close 50%", "Exit 33%" from primary_action ──
        primary = hints.get("primary_action", "")
        if isinstance(primary, str):
            for word in primary.split():
                word = word.strip("% ,")
                try:
                    pct = float(word)
                    if 1 < pct <= 100:
                        return pct / 100.0
                    elif 0 < pct <= 1:
                        return pct
                except ValueError:
                    continue

        # ── 3. Position state take profits ──
        tps_hit = position_state.get("tps_hit", [])
        take_profits = position_state.get("take_profits", [])
        if take_profits:
            for i, tp in enumerate(take_profits):
                if i not in tps_hit and isinstance(tp, dict):
                    return tp.get("exit_pct", action_spec.get("default_exit_pct", 0.33))

        # ── 4. Hardcoded fallback ──
        return action_spec.get("default_exit_pct", 0.33)

    def set_structure_context_fn(self, fn: Callable):
        """Inject the structure service lookup function from terminal_api."""
        self._structure_context_fn = fn
        logger.info("[EXEC] Structure context function registered for TP sizing")

    async def _get_structure_context(self, symbol: str, position_state: dict = None) -> dict:
        """Get structural context for a symbol (VPVR, S/R grades, etc.).

        The injected function receives (symbol, current_price, direction) and handles
        fetcher creation internally.
        """
        if not self._structure_context_fn:
            return {}
        try:
            pos = position_state or {}
            current_price = float(pos.get("current_price", 0))
            direction = pos.get("direction", "LONG").lower()
            ctx = await self._structure_context_fn(symbol, current_price, direction)
            if ctx and hasattr(ctx, '__dict__'):
                return ctx.__dict__
            elif isinstance(ctx, dict):
                return ctx
        except Exception as e:
            logger.warning(f"[EXEC] Structure context fetch failed for {symbol}: {e}")
        return {}

    def _extract_momentum(self, position_state: dict) -> dict:
        """Extract momentum data from position state or live data."""
        return {
            "rsi": position_state.get("rsi", position_state.get("momentum_rsi", 50)),
            "cvd_trend": position_state.get("cvd_trend", "neutral"),
            "oi_change_pct": position_state.get("oi_change_pct", 0),
        }

    def add_event_listener(self, listener: Callable):
        """Register a callback for execution events (SSE notifications)."""
        self._event_listeners.append(listener)

    async def _broadcast_event(self, event: dict):
        """Broadcast execution event to all listeners."""
        for listener in self._event_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                logger.warning(f"[EXEC] Event listener error: {e}")

    def _parse_stop_from_hints(self, hints: dict, position_state: dict) -> float:
        """
        Extract a stop price from AI execution hints or position state.

        Priority:
        1. AI-provided execution.stop_price (float) — new standardized format
        2. AI-provided execution.stop_adjustment (legacy, numeric or string)
        3. Position state trailing_stop or guarding_line — structural fallback
        """
        # ── 1. New standardized format: execution.stop_price (float) ──
        stop_price = hints.get("stop_price")
        if stop_price is not None:
            try:
                val = float(stop_price)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass

        # ── 2. Legacy format: stop_adjustment (numeric or string) ──
        stop_adj = hints.get("stop_adjustment", "")
        if isinstance(stop_adj, (int, float)) and stop_adj > 0:
            return float(stop_adj)
        if isinstance(stop_adj, str):
            import re
            nums = re.findall(r'[\d,]+\.?\d*', stop_adj.replace(",", ""))
            for n in nums:
                try:
                    val = float(n)
                    if val > 0:
                        return val
                except ValueError:
                    continue

        # ── 3. Structural fallback from position state ──
        trail = position_state.get("trailing_stop")
        if trail and trail > 0:
            return float(trail)
        guard = position_state.get("guarding_line")
        if guard and guard > 0:
            return float(guard)

        return 0.0

    def _fail(self, action: str, symbol: str, error: str,
              confidence: float, urgency: str, source: str) -> ExecutionResult:
        """Helper to create a failed execution result."""
        logger.warning(f"[EXEC] BLOCKED {action} on {symbol}: {error}")
        return ExecutionResult(
            success=False, action=action, symbol=symbol, error=error,
            confidence=confidence, urgency=urgency, source=source,
            timestamp=datetime.utcnow().isoformat()
        )

    # ─── Status & Configuration ───────────────────────────────────────

    def status(self) -> dict:
        """Get execution engine status."""
        return {
            "kill_switch": self.safety.kill_switch,
            "total_executions": self._total_executions,
            "total_volume_usd": round(self._total_volume_usd, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_loss_limit": self.safety.daily_loss_limit_usd,
            "min_confidence": self.safety.min_confidence,
            "max_executions_per_hour": self.safety.max_executions_per_hour,
            "registered_contexts": list(self._user_contexts.keys()),
            "tracked_positions": list(self._position_map.keys()),
            "pending_confirmations": len(self._pending_confirmations),
            "audit_entries": self.audit.total_count,
        }

    def configure(self, **kwargs):
        """Update safety configuration."""
        for key, val in kwargs.items():
            if hasattr(self.safety, key):
                setattr(self.safety, key, val)
                logger.info(f"[EXEC] Config updated: {key} = {val}")

    def activate_kill_switch(self):
        """Emergency stop — halt all execution."""
        self.safety.kill_switch = True
        logger.critical("[EXEC] ⚠ KILL SWITCH ACTIVATED — all execution halted")

    def deactivate_kill_switch(self):
        """Resume execution."""
        self.safety.kill_switch = False
        logger.info("[EXEC] Kill switch deactivated — execution resumed")

    def get_execution_history(self, limit: int = 50, position_id: Optional[str] = None,
                              action_filter: Optional[str] = None) -> List[dict]:
        """Get execution history."""
        return self.audit.get_history(limit, position_id, action_filter)


# =============================================================================
# SINGLETON
# =============================================================================

execution_engine = ExecutionEngine()
