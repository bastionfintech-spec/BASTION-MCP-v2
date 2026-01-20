"""
BASTION Trading Session Manager
================================

Live position tracking with multi-shot entries and dynamic stop evolution.

The MCF Stop Philosophy:
1. Primary Stop (Structural) - Below validated support, 20% ATR buffer
2. Guarding Line (Swing) - Dynamic trailing, activates after 10 bars
3. Safety Net (Max Loss) - Hard cap at 5%

Multi-Shot System:
- Total Risk Cap: 2% default
- Max Shots: 3 entries
- Position sizes: 50% → 30% → 20%

Exit Priority:
1. Opposite MCF signal → immediate exit
2. Guarding line broken → exit remaining
3. Structural support broken → exit
4. Momentum exhaustion → partial exit
5. Volume climax → take profits
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import uuid
import asyncio
import logging

from .risk_engine import GuardingLineManager, StructureHealth
from .adaptive_budget import AdaptiveRiskBudget, TradeBudget, Shot, ShotStatus

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SessionStatus(str, Enum):
    """Status of a trading session."""
    PENDING = "pending"           # Created but no entries yet
    ACTIVE = "active"             # Has open position
    PHASE_1 = "phase_1"           # Bars 0-10, structural stop only
    PHASE_2 = "phase_2"           # Bar 11+, guarding line active
    PARTIAL_EXIT = "partial_exit" # Some targets hit
    CLOSED = "closed"             # Fully exited
    STOPPED = "stopped"           # Hit stop loss
    EXPIRED = "expired"           # Session timeout


class ExitReason(str, Enum):
    """Reason for exit."""
    OPPOSITE_SIGNAL = "opposite_mcf_signal"
    GUARDING_BROKEN = "guarding_line_broken"
    STRUCTURE_BROKEN = "structural_support_broken"
    TARGET_HIT = "target_hit"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    VOLUME_CLIMAX = "volume_climax"
    SAFETY_NET = "safety_net_5pct"
    MANUAL = "manual_exit"
    EXPIRED = "session_expired"


class TradePhase(str, Enum):
    """Current phase of the trade."""
    PRE_ENTRY = "pre_entry"       # Waiting for first shot
    PHASE_1 = "phase_1"           # Bars 0-10, structural stop
    PHASE_2 = "phase_2"           # Bar 11+, guarding active
    TRAILING = "trailing"         # In profit, trailing stops


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SessionEntry:
    """A single entry (shot) in a session."""
    id: str
    shot_number: int              # 1, 2, or 3
    entry_price: float
    size: float                   # Position size in base currency
    risk_amount: float            # Dollar risk for this shot
    stop_price: float             # Stop at time of entry
    entry_bar: int                # Bar number at entry
    entry_time: datetime = field(default_factory=datetime.utcnow)
    status: ShotStatus = ShotStatus.ACTIVE
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    pnl: float = 0.0


@dataclass
class PartialExit:
    """Record of a partial exit."""
    price: float
    size: float                   # Amount exited
    percentage: float             # % of position exited
    reason: str
    pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionState:
    """Current state of a trading session."""
    
    # Identifiers
    id: str
    symbol: str
    direction: str                # "long" or "short"
    timeframe: str
    
    # Account context
    account_balance: float
    total_risk_cap_pct: float = 2.0
    max_shots: int = 3
    
    # Status
    status: SessionStatus = SessionStatus.PENDING
    phase: TradePhase = TradePhase.PRE_ENTRY
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Entries (shots)
    entries: List[SessionEntry] = field(default_factory=list)
    shots_taken: int = 0
    
    # Position aggregate
    total_size: float = 0.0
    average_entry: float = 0.0
    risk_used: float = 0.0
    risk_remaining: float = 0.0
    
    # Stop levels
    structural_stop: float = 0.0          # Grade 3-4 trendline support
    current_stop: float = 0.0             # Active stop (may be guarding)
    safety_net_stop: float = 0.0          # 5% max loss
    guarding_line: Optional[Dict] = None  # {slope, intercept, activation_bar}
    guarding_level: Optional[float] = None
    
    # Target levels
    targets: List[Dict] = field(default_factory=list)
    targets_hit: int = 0
    
    # Tracking
    bars_in_trade: int = 0
    current_price: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    
    # Partial exits
    partial_exits: List[PartialExit] = field(default_factory=list)
    remaining_size: float = 0.0
    
    # Alerts/signals
    pending_signals: List[Dict] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction,
            'timeframe': self.timeframe,
            'status': self.status.value,
            'phase': self.phase.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            
            'shots_taken': self.shots_taken,
            'max_shots': self.max_shots,
            'entries': [
                {
                    'shot': e.shot_number,
                    'price': e.entry_price,
                    'size': e.size,
                    'risk': e.risk_amount,
                    'status': e.status.value,
                    'pnl': e.pnl,
                }
                for e in self.entries
            ],
            
            'position': {
                'total_size': self.total_size,
                'remaining_size': self.remaining_size,
                'average_entry': self.average_entry,
                'current_price': self.current_price,
            },
            
            'risk': {
                'total_cap': self.account_balance * (self.total_risk_cap_pct / 100),
                'used': self.risk_used,
                'remaining': self.risk_remaining,
            },
            
            'stops': {
                'structural': self.structural_stop,
                'current': self.current_stop,
                'safety_net': self.safety_net_stop,
                'guarding_level': self.guarding_level,
                'guarding_active': self.phase == TradePhase.PHASE_2,
            },
            
            'targets': self.targets,
            'targets_hit': self.targets_hit,
            
            'tracking': {
                'bars_in_trade': self.bars_in_trade,
                'highest': self.highest_since_entry,
                'lowest': self.lowest_since_entry,
            },
            
            'pnl': {
                'unrealized': self.unrealized_pnl,
                'unrealized_pct': self.unrealized_pnl_pct,
                'realized': self.realized_pnl,
                'total': self.unrealized_pnl + self.realized_pnl,
            },
            
            'partial_exits': [
                {
                    'price': p.price,
                    'size': p.size,
                    'pct': p.percentage,
                    'reason': p.reason,
                    'pnl': p.pnl,
                }
                for p in self.partial_exits
            ],
            
            'last_update': self.last_update.isoformat(),
        }


@dataclass
class SessionUpdate:
    """Result of updating a session with new price data."""
    session_id: str
    status: SessionStatus
    phase: TradePhase
    
    # Price info
    current_price: float
    bars_in_trade: int
    
    # Stop status
    current_stop: float
    guarding_level: Optional[float]
    stop_moved: bool = False
    
    # Signals
    exit_signal: bool = False
    exit_reason: Optional[ExitReason] = None
    exit_percentage: float = 0.0           # How much to exit (0-100%)
    
    # Target status
    target_hit: bool = False
    target_index: int = -1
    
    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Alerts
    alerts: List[str] = field(default_factory=list)


# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """
    Manages live trading sessions with multi-shot entries and dynamic stops.
    
    Usage:
        manager = SessionManager()
        
        # Create session
        session = manager.create_session(
            symbol='BTCUSDT',
            direction='long',
            timeframe='4h',
            account_balance=100000,
            structural_support=93200,
            targets=[...],
        )
        
        # Take first shot
        entry = manager.take_shot(
            session_id=session.id,
            entry_price=94500,
            current_atr=600,
        )
        
        # Update on each bar
        update = manager.update_session(
            session_id=session.id,
            current_price=95000,
            current_bar=5,
            ohlcv_data=df,
        )
        
        # Check for exit signals
        if update.exit_signal:
            manager.execute_exit(session.id, update.exit_reason, update.exit_percentage)
    """
    
    def __init__(
        self,
        default_risk_cap: float = 2.0,
        default_max_shots: int = 3,
        guarding_activation_bars: int = 10,
        safety_net_pct: float = 5.0,
        session_timeout_hours: int = 168,  # 1 week default
    ):
        self.default_risk_cap = default_risk_cap
        self.default_max_shots = default_max_shots
        self.guarding_activation_bars = guarding_activation_bars
        self.safety_net_pct = safety_net_pct
        self.session_timeout_hours = session_timeout_hours
        
        # Active sessions
        self._sessions: Dict[str, SessionState] = {}
        
        # Budget manager for multi-shot
        self._budget_manager = AdaptiveRiskBudget(
            max_shots=default_max_shots,
            total_risk_cap=default_risk_cap
        )
        
        # Guarding line manager
        self._guarding_manager = GuardingLineManager(
            activation_bars=guarding_activation_bars
        )
    
    def create_session(
        self,
        symbol: str,
        direction: str,
        timeframe: str,
        account_balance: float,
        structural_support: float,
        targets: List[Dict[str, Any]],
        risk_cap_pct: float = None,
        max_shots: int = None,
        timeout_hours: int = None,
    ) -> SessionState:
        """
        Create a new trading session.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            direction: 'long' or 'short'
            timeframe: Candle timeframe
            account_balance: Account size in USD
            structural_support: Grade 3-4 validated support level
            targets: List of target levels [{price, exit_percentage, reason}]
            risk_cap_pct: Total risk cap (default 2%)
            max_shots: Max entries (default 3)
            timeout_hours: Session timeout (default 168h / 1 week)
        """
        session_id = str(uuid.uuid4())[:8]
        risk_cap = risk_cap_pct or self.default_risk_cap
        max_shots = max_shots or self.default_max_shots
        timeout = timeout_hours or self.session_timeout_hours
        
        # Calculate safety net (5% max loss from anticipated entry)
        # Will be updated when first shot is taken
        safety_net = structural_support * 0.95 if direction == "long" else structural_support * 1.05
        
        session = SessionState(
            id=session_id,
            symbol=symbol,
            direction=direction,
            timeframe=timeframe,
            account_balance=account_balance,
            total_risk_cap_pct=risk_cap,
            max_shots=max_shots,
            status=SessionStatus.PENDING,
            phase=TradePhase.PRE_ENTRY,
            expires_at=datetime.utcnow() + timedelta(hours=timeout),
            structural_stop=structural_support,
            current_stop=structural_support,
            safety_net_stop=safety_net,
            targets=targets,
            risk_remaining=account_balance * (risk_cap / 100),
        )
        
        self._sessions[session_id] = session
        logger.info(f"Created session {session_id} for {symbol} {direction}")
        
        return session
    
    def take_shot(
        self,
        session_id: str,
        entry_price: float,
        current_atr: float,
        stop_override: float = None,
    ) -> Optional[SessionEntry]:
        """
        Take a shot (entry) in the session.
        
        Args:
            session_id: Session ID
            entry_price: Entry price for this shot
            current_atr: Current ATR for buffer calculation
            stop_override: Override stop price (defaults to structural)
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return None
        
        if session.shots_taken >= session.max_shots:
            logger.warning(f"Session {session_id}: Max shots ({session.max_shots}) already taken")
            return None
        
        if session.status == SessionStatus.STOPPED:
            logger.warning(f"Session {session_id}: Already stopped out")
            return None
        
        # Calculate shot allocation (50% → 30% → 20%)
        shot_allocations = [0.50, 0.30, 0.20]
        shot_number = session.shots_taken + 1
        allocation = shot_allocations[shot_number - 1] if shot_number <= 3 else 0.20
        
        # Calculate risk for this shot
        total_budget = session.account_balance * (session.total_risk_cap_pct / 100)
        shot_risk = total_budget * allocation
        
        # Cap at remaining risk
        shot_risk = min(shot_risk, session.risk_remaining)
        if shot_risk <= 0:
            logger.warning(f"Session {session_id}: No risk budget remaining")
            return None
        
        # Calculate stop price
        stop_price = stop_override or session.structural_stop
        
        # Add ATR buffer for structural stop
        if not stop_override:
            if session.direction == "long":
                stop_price = session.structural_stop - (current_atr * 0.2)
            else:
                stop_price = session.structural_stop + (current_atr * 0.2)
        
        # Calculate position size
        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0:
            logger.error(f"Session {session_id}: Invalid risk distance")
            return None
        
        size = shot_risk / risk_distance
        
        # Create entry
        entry = SessionEntry(
            id=str(uuid.uuid4())[:8],
            shot_number=shot_number,
            entry_price=entry_price,
            size=size,
            risk_amount=shot_risk,
            stop_price=stop_price,
            entry_bar=session.bars_in_trade,
        )
        
        # Update session
        session.entries.append(entry)
        session.shots_taken = shot_number
        session.total_size += size
        session.remaining_size = session.total_size
        session.risk_used += shot_risk
        session.risk_remaining -= shot_risk
        
        # Recalculate average entry
        total_value = sum(e.entry_price * e.size for e in session.entries if e.status == ShotStatus.ACTIVE)
        total_size = sum(e.size for e in session.entries if e.status == ShotStatus.ACTIVE)
        session.average_entry = total_value / total_size if total_size > 0 else entry_price
        
        # Update status
        if session.status == SessionStatus.PENDING:
            session.status = SessionStatus.ACTIVE
            session.phase = TradePhase.PHASE_1
            session.highest_since_entry = entry_price
            session.lowest_since_entry = entry_price
            
            # Calculate safety net from actual entry
            if session.direction == "long":
                session.safety_net_stop = entry_price * (1 - self.safety_net_pct / 100)
            else:
                session.safety_net_stop = entry_price * (1 + self.safety_net_pct / 100)
        
        # Update current stop (use tightest for later entries)
        if session.direction == "long":
            session.current_stop = max(session.current_stop, stop_price)
        else:
            session.current_stop = min(session.current_stop, stop_price)
        
        logger.info(
            f"Session {session_id}: Shot {shot_number} taken - "
            f"Entry: {entry_price}, Size: {size:.4f}, Risk: ${shot_risk:.2f}"
        )
        
        return entry
    
    def update_session(
        self,
        session_id: str,
        current_price: float,
        current_bar: int,
        recent_lows: List[float] = None,
        recent_highs: List[float] = None,
        opposing_signal: bool = False,
        momentum_exhaustion: bool = False,
        volume_climax: bool = False,
    ) -> SessionUpdate:
        """
        Update session with new bar data.
        
        Call this on each new bar to:
        - Update tracking (high/low, P&L)
        - Check for phase transition (Phase 1 → Phase 2)
        - Update guarding line if active
        - Check for exit signals
        
        Args:
            session_id: Session ID
            current_price: Current price
            current_bar: Bar number since session start
            recent_lows: Recent swing lows (for guarding line)
            recent_highs: Recent swing highs (for guarding line)
            opposing_signal: True if opposite MCF signal detected
            momentum_exhaustion: True if momentum exhausting
            volume_climax: True if volume climax detected
        """
        session = self._sessions.get(session_id)
        if not session:
            return SessionUpdate(session_id=session_id, status=SessionStatus.CLOSED, 
                               phase=TradePhase.PRE_ENTRY, current_price=current_price,
                               bars_in_trade=0, current_stop=0, guarding_level=None)
        
        if session.status in [SessionStatus.CLOSED, SessionStatus.STOPPED, SessionStatus.EXPIRED]:
            return SessionUpdate(
                session_id=session_id,
                status=session.status,
                phase=session.phase,
                current_price=current_price,
                bars_in_trade=session.bars_in_trade,
                current_stop=session.current_stop,
                guarding_level=session.guarding_level,
            )
        
        # Check expiration
        if session.expires_at and datetime.utcnow() > session.expires_at:
            session.status = SessionStatus.EXPIRED
            return SessionUpdate(
                session_id=session_id,
                status=SessionStatus.EXPIRED,
                phase=session.phase,
                current_price=current_price,
                bars_in_trade=session.bars_in_trade,
                current_stop=session.current_stop,
                guarding_level=session.guarding_level,
                exit_signal=True,
                exit_reason=ExitReason.EXPIRED,
                exit_percentage=100.0,
                alerts=["Session expired"],
            )
        
        # Update tracking
        session.current_price = current_price
        session.bars_in_trade = current_bar
        session.highest_since_entry = max(session.highest_since_entry, current_price)
        session.lowest_since_entry = min(session.lowest_since_entry, current_price)
        session.last_update = datetime.utcnow()
        
        # Calculate P&L
        if session.remaining_size > 0:
            if session.direction == "long":
                session.unrealized_pnl = (current_price - session.average_entry) * session.remaining_size
            else:
                session.unrealized_pnl = (session.average_entry - current_price) * session.remaining_size
            
            session.unrealized_pnl_pct = (session.unrealized_pnl / session.account_balance) * 100
        
        alerts = []
        update = SessionUpdate(
            session_id=session_id,
            status=session.status,
            phase=session.phase,
            current_price=current_price,
            bars_in_trade=current_bar,
            current_stop=session.current_stop,
            guarding_level=session.guarding_level,
            unrealized_pnl=session.unrealized_pnl,
            unrealized_pnl_pct=session.unrealized_pnl_pct,
        )
        
        # === CHECK EXIT SIGNALS (Priority Order) ===
        
        # 1. Opposite MCF Signal → Immediate Exit
        if opposing_signal:
            update.exit_signal = True
            update.exit_reason = ExitReason.OPPOSITE_SIGNAL
            update.exit_percentage = 100.0
            alerts.append("🚨 OPPOSITE MCF SIGNAL - Exit immediately")
            update.alerts = alerts
            return update
        
        # 2. Check Phase Transition (Phase 1 → Phase 2)
        if session.phase == TradePhase.PHASE_1 and current_bar >= self.guarding_activation_bars:
            session.phase = TradePhase.PHASE_2
            session.status = SessionStatus.PHASE_2
            
            # Initialize guarding line
            if recent_lows and session.direction == "long":
                session.guarding_line = self._guarding_manager.calculate_initial_line(
                    session.average_entry, session.direction, recent_lows
                )
            elif recent_highs and session.direction == "short":
                session.guarding_line = self._guarding_manager.calculate_initial_line(
                    session.average_entry, session.direction, recent_highs
                )
            
            alerts.append(f"📈 Phase 2 activated - Guarding line now trailing")
            update.phase = TradePhase.PHASE_2
        
        # 3. Update Guarding Line (if Phase 2)
        if session.phase == TradePhase.PHASE_2 and session.guarding_line:
            session.guarding_level = self._guarding_manager.get_current_level(
                session.guarding_line, current_bar
            )
            update.guarding_level = session.guarding_level
            
            # Check if guarding line broken
            is_broken, reason = self._guarding_manager.check_break(
                current_price, session.guarding_level, session.direction
            )
            
            if is_broken:
                update.exit_signal = True
                update.exit_reason = ExitReason.GUARDING_BROKEN
                update.exit_percentage = 100.0
                alerts.append(f"🛑 GUARDING LINE BROKEN - {reason}")
                update.alerts = alerts
                return update
            
            # Update current stop to max of structural and guarding
            if session.direction == "long":
                new_stop = max(session.structural_stop, session.guarding_level)
                if new_stop > session.current_stop:
                    session.current_stop = new_stop
                    update.stop_moved = True
                    update.current_stop = new_stop
                    alerts.append(f"⬆️ Stop raised to ${new_stop:,.2f}")
            else:
                new_stop = min(session.structural_stop, session.guarding_level)
                if new_stop < session.current_stop:
                    session.current_stop = new_stop
                    update.stop_moved = True
                    update.current_stop = new_stop
                    alerts.append(f"⬇️ Stop lowered to ${new_stop:,.2f}")
        
        # 4. Check Structural Stop
        if session.direction == "long" and current_price < session.structural_stop:
            update.exit_signal = True
            update.exit_reason = ExitReason.STRUCTURE_BROKEN
            update.exit_percentage = 100.0
            alerts.append(f"🛑 STRUCTURAL SUPPORT BROKEN at ${session.structural_stop:,.2f}")
            update.alerts = alerts
            return update
        elif session.direction == "short" and current_price > session.structural_stop:
            update.exit_signal = True
            update.exit_reason = ExitReason.STRUCTURE_BROKEN
            update.exit_percentage = 100.0
            alerts.append(f"🛑 STRUCTURAL RESISTANCE BROKEN at ${session.structural_stop:,.2f}")
            update.alerts = alerts
            return update
        
        # 5. Check Safety Net (5% max loss)
        if session.direction == "long" and current_price < session.safety_net_stop:
            update.exit_signal = True
            update.exit_reason = ExitReason.SAFETY_NET
            update.exit_percentage = 100.0
            alerts.append(f"🚨 SAFETY NET HIT - 5% max loss at ${session.safety_net_stop:,.2f}")
            update.alerts = alerts
            return update
        elif session.direction == "short" and current_price > session.safety_net_stop:
            update.exit_signal = True
            update.exit_reason = ExitReason.SAFETY_NET
            update.exit_percentage = 100.0
            alerts.append(f"🚨 SAFETY NET HIT - 5% max loss at ${session.safety_net_stop:,.2f}")
            update.alerts = alerts
            return update
        
        # 6. Check Targets
        for i, target in enumerate(session.targets):
            if i < session.targets_hit:
                continue  # Already hit
            
            target_price = target['price']
            target_hit = False
            
            if session.direction == "long" and current_price >= target_price:
                target_hit = True
            elif session.direction == "short" and current_price <= target_price:
                target_hit = True
            
            if target_hit:
                update.target_hit = True
                update.target_index = i
                update.exit_signal = True
                update.exit_reason = ExitReason.TARGET_HIT
                update.exit_percentage = target.get('exit_percentage', 33)
                alerts.append(f"🎯 TARGET {i+1} HIT at ${target_price:,.2f} - Exit {update.exit_percentage:.0f}%")
                break
        
        # 7. Check Momentum Exhaustion (partial exit)
        if momentum_exhaustion and not update.exit_signal:
            update.exit_signal = True
            update.exit_reason = ExitReason.MOMENTUM_EXHAUSTION
            update.exit_percentage = 33.0  # Partial exit
            alerts.append("⚠️ Momentum exhaustion detected - Partial exit 33%")
        
        # 8. Check Volume Climax (take profits)
        if volume_climax and not update.exit_signal:
            update.exit_signal = True
            update.exit_reason = ExitReason.VOLUME_CLIMAX
            update.exit_percentage = 50.0  # Take significant profits
            alerts.append("📊 Volume climax detected - Taking 50% profit")
        
        update.alerts = alerts
        return update
    
    def execute_exit(
        self,
        session_id: str,
        exit_price: float,
        exit_reason: ExitReason,
        exit_percentage: float = 100.0,
    ) -> Optional[PartialExit]:
        """
        Execute an exit (full or partial).
        
        Args:
            session_id: Session ID
            exit_price: Price to exit at
            exit_reason: Reason for exit
            exit_percentage: Percentage of remaining position to exit (0-100)
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if session.remaining_size <= 0:
            logger.warning(f"Session {session_id}: No remaining position to exit")
            return None
        
        # Calculate exit size
        exit_pct = min(exit_percentage, 100.0) / 100.0
        exit_size = session.remaining_size * exit_pct
        
        # Calculate P&L for this exit
        if session.direction == "long":
            pnl = (exit_price - session.average_entry) * exit_size
        else:
            pnl = (session.average_entry - exit_price) * exit_size
        
        # Create partial exit record
        partial = PartialExit(
            price=exit_price,
            size=exit_size,
            percentage=exit_percentage,
            reason=exit_reason.value,
            pnl=pnl,
        )
        
        session.partial_exits.append(partial)
        session.remaining_size -= exit_size
        session.realized_pnl += pnl
        
        # Update entries
        if exit_percentage >= 99.9:  # Full exit
            for entry in session.entries:
                if entry.status == ShotStatus.ACTIVE:
                    entry.status = ShotStatus.FULL_EXIT
                    entry.exit_price = exit_price
                    entry.exit_time = datetime.utcnow()
                    entry.exit_reason = exit_reason
                    
                    if session.direction == "long":
                        entry.pnl = (exit_price - entry.entry_price) * entry.size
                    else:
                        entry.pnl = (entry.entry_price - exit_price) * entry.size
        
        # Update target hit count
        if exit_reason == ExitReason.TARGET_HIT:
            session.targets_hit += 1
        
        # Update session status
        if session.remaining_size <= 0.0001:  # Fully closed
            session.status = SessionStatus.CLOSED
            session.remaining_size = 0
        elif exit_reason in [ExitReason.STRUCTURE_BROKEN, ExitReason.SAFETY_NET, ExitReason.GUARDING_BROKEN]:
            session.status = SessionStatus.STOPPED
        else:
            session.status = SessionStatus.PARTIAL_EXIT
        
        logger.info(
            f"Session {session_id}: Exited {exit_pct*100:.0f}% at ${exit_price:,.2f} "
            f"({exit_reason.value}) - P&L: ${pnl:,.2f}"
        )
        
        return partial
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def get_active_sessions(self, symbol: str = None) -> List[SessionState]:
        """Get all active sessions, optionally filtered by symbol."""
        active = [
            s for s in self._sessions.values()
            if s.status in [SessionStatus.ACTIVE, SessionStatus.PHASE_1, SessionStatus.PHASE_2, SessionStatus.PARTIAL_EXIT]
        ]
        
        if symbol:
            active = [s for s in active if s.symbol == symbol]
        
        return active
    
    def close_session(self, session_id: str, reason: str = "manual") -> bool:
        """Close a session without exiting (e.g., on expiration)."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.CLOSED
        logger.info(f"Session {session_id} closed: {reason}")
        return True
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get a summary of session performance."""
        session = self._sessions.get(session_id)
        if not session:
            return {}
        
        total_pnl = session.realized_pnl + session.unrealized_pnl
        total_pnl_pct = (total_pnl / session.account_balance) * 100
        
        return {
            'session_id': session_id,
            'symbol': session.symbol,
            'direction': session.direction,
            'status': session.status.value,
            'duration_bars': session.bars_in_trade,
            'shots_taken': session.shots_taken,
            'average_entry': session.average_entry,
            'current_price': session.current_price,
            'total_size': session.total_size,
            'remaining_size': session.remaining_size,
            'risk_used': session.risk_used,
            'realized_pnl': session.realized_pnl,
            'unrealized_pnl': session.unrealized_pnl,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'targets_hit': session.targets_hit,
            'partial_exits': len(session.partial_exits),
        }

