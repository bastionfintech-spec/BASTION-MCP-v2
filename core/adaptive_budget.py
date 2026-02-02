"""
Adaptive Risk Budget Manager
============================

Multi-shot entry system with capped total risk.
Instead of all-or-nothing entries, allows multiple attempts
while maintaining strict risk discipline.

Key Principles:
1. Cap total risk across all re-entries (e.g., 2% max)
2. Decay allocation per shot (1st: 1%, 2nd: 0.6%, 3rd: 0.4%)
3. Only allow re-entry if structure still valid
4. Track aggregate P&L across all shots
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import uuid


class ShotStatus(Enum):
    """Status of an individual entry."""
    ACTIVE = "active"
    STOPPED_OUT = "stopped_out"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"
    CANCELLED = "cancelled"


@dataclass
class Shot:
    """A single entry attempt within a budget."""
    id: str
    entry_price: float
    size: float
    risk_allocated: float      # Risk % allocated to this shot
    stop_price: float
    status: ShotStatus
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class TradeBudget:
    """
    Risk budget for a trade (may contain multiple shots).
    """
    id: str
    symbol: str
    direction: str
    total_risk_cap: float          # Maximum risk as % of account
    risk_used: float = 0.0         # Risk consumed so far
    max_shots: int = 3
    shots: List[Shot] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"         # "active", "exhausted", "completed"
    
    @property
    def shots_remaining(self) -> int:
        active_shots = [s for s in self.shots if s.status == ShotStatus.ACTIVE]
        return self.max_shots - len(self.shots) + len(active_shots)
    
    @property
    def risk_remaining(self) -> float:
        return max(0, self.total_risk_cap - self.risk_used)
    
    @property
    def can_take_shot(self) -> bool:
        return (
            self.shots_remaining > 0 and 
            self.risk_remaining > 0.1 and  # At least 0.1% remaining
            self.status == "active"
        )
    
    @property
    def aggregate_pnl(self) -> float:
        return sum(shot.pnl for shot in self.shots)
    
    @property
    def aggregate_pnl_pct(self) -> float:
        total_allocated = sum(shot.risk_allocated for shot in self.shots)
        if total_allocated == 0:
            return 0
        return self.aggregate_pnl / total_allocated * 100


class AdaptiveRiskBudget:
    """
    Manages risk budgets for multi-shot trading.
    
    Usage:
        budget_mgr = AdaptiveRiskBudget(max_shots=3, total_risk_cap=2.0)
        
        # Create a budget for a trade
        budget = budget_mgr.create_budget("BTCUSDT", "long")
        
        # Take first shot
        shot1 = budget_mgr.take_shot(budget.id, entry=50000, stop=49000)
        
        # If stopped out, take another shot
        budget_mgr.record_stop(budget.id, shot1.id, exit_price=49000)
        shot2 = budget_mgr.take_shot(budget.id, entry=49500, stop=48500)
    """
    
    def __init__(
        self,
        max_shots: int = 3,
        total_risk_cap: float = 2.0,  # 2% total risk
        decay_factor: float = 0.6,     # Each shot gets 60% of previous
        min_shot_risk: float = 0.2     # Minimum 0.2% per shot
    ):
        self.max_shots = max_shots
        self.total_risk_cap = total_risk_cap
        self.decay_factor = decay_factor
        self.min_shot_risk = min_shot_risk
        
        self._budgets: Dict[str, TradeBudget] = {}
    
    def create_budget(
        self,
        symbol: str,
        direction: str,
        total_risk_cap: Optional[float] = None,
        max_shots: Optional[int] = None
    ) -> TradeBudget:
        """
        Create a new risk budget for a trade.
        """
        budget_id = str(uuid.uuid4())[:8]
        budget = TradeBudget(
            id=budget_id,
            symbol=symbol,
            direction=direction,
            total_risk_cap=total_risk_cap or self.total_risk_cap,
            max_shots=max_shots or self.max_shots
        )
        self._budgets[budget_id] = budget
        return budget
    
    def get_budget(self, budget_id: str) -> Optional[TradeBudget]:
        """Get a budget by ID."""
        return self._budgets.get(budget_id)
    
    def calculate_next_shot_risk(self, budget_id: str) -> float:
        """
        Calculate risk allocation for the next shot.
        Uses decaying allocation to cap total risk.
        """
        budget = self._budgets.get(budget_id)
        if not budget:
            return 0
        
        shot_number = len(budget.shots) + 1
        
        if shot_number == 1:
            # First shot: Use base risk
            base_risk = budget.total_risk_cap / (1 + self.decay_factor + self.decay_factor**2)
            return min(base_risk, budget.risk_remaining)
        
        # Subsequent shots: Decay
        prev_risk = budget.shots[-1].risk_allocated if budget.shots else 1.0
        next_risk = prev_risk * self.decay_factor
        
        # Ensure minimum
        next_risk = max(self.min_shot_risk, next_risk)
        
        # Cap at remaining budget
        return min(next_risk, budget.risk_remaining)
    
    def take_shot(
        self,
        budget_id: str,
        entry_price: float,
        stop_price: float,
        account_balance: float = 10000
    ) -> Optional[Shot]:
        """
        Take a shot (entry) against the budget.
        
        Returns the Shot details including calculated size.
        """
        budget = self._budgets.get(budget_id)
        if not budget or not budget.can_take_shot:
            return None
        
        # Calculate risk for this shot
        risk_pct = self.calculate_next_shot_risk(budget_id)
        risk_amount = account_balance * (risk_pct / 100)
        
        # Calculate position size
        risk_distance = abs(entry_price - stop_price)
        if risk_distance == 0:
            return None
        
        size = risk_amount / risk_distance
        
        # Create shot
        shot = Shot(
            id=str(uuid.uuid4())[:8],
            entry_price=entry_price,
            size=size,
            risk_allocated=risk_pct,
            stop_price=stop_price,
            status=ShotStatus.ACTIVE,
            entry_time=datetime.now()
        )
        
        # Update budget
        budget.shots.append(shot)
        budget.risk_used += risk_pct
        
        if budget.risk_remaining < self.min_shot_risk:
            budget.status = "exhausted"
        
        return shot
    
    def record_stop(
        self,
        budget_id: str,
        shot_id: str,
        exit_price: float
    ) -> Optional[Shot]:
        """Record a stop-out for a shot."""
        budget = self._budgets.get(budget_id)
        if not budget:
            return None
        
        shot = next((s for s in budget.shots if s.id == shot_id), None)
        if not shot:
            return None
        
        shot.status = ShotStatus.STOPPED_OUT
        shot.exit_price = exit_price
        shot.exit_time = datetime.now()
        
        # Calculate P&L
        if budget.direction == "long":
            shot.pnl = (exit_price - shot.entry_price) * shot.size
            shot.pnl_pct = (exit_price - shot.entry_price) / shot.entry_price * 100
        else:
            shot.pnl = (shot.entry_price - exit_price) * shot.size
            shot.pnl_pct = (shot.entry_price - exit_price) / shot.entry_price * 100
        
        return shot
    
    def record_exit(
        self,
        budget_id: str,
        shot_id: str,
        exit_price: float,
        exit_pct: float = 100
    ) -> Optional[Shot]:
        """Record a take-profit or manual exit."""
        budget = self._budgets.get(budget_id)
        if not budget:
            return None
        
        shot = next((s for s in budget.shots if s.id == shot_id), None)
        if not shot:
            return None
        
        if exit_pct >= 100:
            shot.status = ShotStatus.FULL_EXIT
        else:
            shot.status = ShotStatus.PARTIAL_EXIT
        
        shot.exit_price = exit_price
        shot.exit_time = datetime.now()
        
        # Calculate P&L (on the exited portion)
        exit_size = shot.size * (exit_pct / 100)
        if budget.direction == "long":
            shot.pnl = (exit_price - shot.entry_price) * exit_size
            shot.pnl_pct = (exit_price - shot.entry_price) / shot.entry_price * 100
        else:
            shot.pnl = (shot.entry_price - exit_price) * exit_size
            shot.pnl_pct = (shot.entry_price - exit_price) / shot.entry_price * 100
        
        # Check if all shots are closed
        all_closed = all(
            s.status in [ShotStatus.STOPPED_OUT, ShotStatus.FULL_EXIT, ShotStatus.CANCELLED]
            for s in budget.shots
        )
        if all_closed:
            budget.status = "completed"
        
        return shot
    
    def get_active_shots(self, budget_id: str) -> List[Shot]:
        """Get all active shots for a budget."""
        budget = self._budgets.get(budget_id)
        if not budget:
            return []
        return [s for s in budget.shots if s.status == ShotStatus.ACTIVE]
    
    def get_budget_summary(self, budget_id: str) -> Dict:
        """Get a summary of the budget status."""
        budget = self._budgets.get(budget_id)
        if not budget:
            return {}
        
        return {
            "id": budget.id,
            "symbol": budget.symbol,
            "direction": budget.direction,
            "status": budget.status,
            "total_risk_cap": budget.total_risk_cap,
            "risk_used": budget.risk_used,
            "risk_remaining": budget.risk_remaining,
            "shots_taken": len(budget.shots),
            "shots_remaining": budget.shots_remaining,
            "can_take_shot": budget.can_take_shot,
            "aggregate_pnl": budget.aggregate_pnl,
            "aggregate_pnl_pct": budget.aggregate_pnl_pct,
            "shots": [
                {
                    "id": s.id,
                    "entry": s.entry_price,
                    "stop": s.stop_price,
                    "size": s.size,
                    "status": s.status.value,
                    "pnl": s.pnl
                }
                for s in budget.shots
            ]
        }
    
    def close_budget(self, budget_id: str, reason: str = "manual") -> bool:
        """Close a budget (cancel any remaining shots)."""
        budget = self._budgets.get(budget_id)
        if not budget:
            return False
        
        for shot in budget.shots:
            if shot.status == ShotStatus.ACTIVE:
                shot.status = ShotStatus.CANCELLED
        
        budget.status = "completed"
        return True












