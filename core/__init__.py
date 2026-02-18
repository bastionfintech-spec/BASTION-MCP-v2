"""
BASTION Core Module
===================

Strategy-agnostic risk management engine with advanced detection systems.
"""

# Main Risk Engine (the only engine you need)
from .risk_engine import (
    RiskEngine,
    RiskEngineConfig,
    RiskLevels,
)

# Detection Systems
from .structure_detector import StructureDetector, StructureAnalysis, StructureGrade
from .vpvr_analyzer import VPVRAnalyzer, VPVRAnalysis, VolumeNode, ValueArea
from .mtf_structure import MTFStructureAnalyzer, MTFAlignment
from .orderflow_detector import OrderFlowDetector, OrderFlowAnalysis, FlowDirection

# MCF Structure-Based Exits (Auto-Support + Structure Service)
from .auto_support import AutoSupportDetector, AutoSupportAnalysis, AutoLevel
from .structure_service import StructureService, StructuralContext

# Multi-shot entry system (separate feature)
from .adaptive_budget import AdaptiveRiskBudget, TradeBudget, Shot, ShotStatus

# Session Management (live position tracking)
try:
    from .session import (
        SessionManager, SessionState, SessionEntry, SessionUpdate,
        SessionStatus, ExitReason, TradePhase, PartialExit
    )
except (ImportError, AttributeError):
    SessionManager = SessionState = SessionEntry = SessionUpdate = None
    SessionStatus = ExitReason = TradePhase = PartialExit = None

__all__ = [
    # Main Engine
    "RiskEngine",
    "RiskEngineConfig",
    "RiskLevels",
    # Structure Detection
    "StructureDetector",
    "StructureAnalysis",
    "StructureGrade",
    # Volume Profile
    "VPVRAnalyzer",
    "VPVRAnalysis",
    "VolumeNode",
    "ValueArea",
    # Multi-Timeframe
    "MTFStructureAnalyzer",
    "MTFAlignment",
    # Order Flow
    "OrderFlowDetector",
    "OrderFlowAnalysis",
    "FlowDirection",
    # Auto-Support
    "AutoSupportDetector",
    "AutoSupportAnalysis",
    "AutoLevel",
    # Structure Service
    "StructureService",
    "StructuralContext",
    # Multi-Shot
    "AdaptiveRiskBudget",
    "TradeBudget",
    "Shot",
    "ShotStatus",
    # Session Management
    "SessionManager",
    "SessionState",
    "SessionEntry",
    "SessionUpdate",
    "SessionStatus",
    "ExitReason",
    "TradePhase",
    "PartialExit",
]
