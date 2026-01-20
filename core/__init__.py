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
    RiskUpdate,
    TradeSetup,
    PositionUpdate,
    StopType,
    TargetType,
    StructureHealth,
    GuardingLineManager,
)

# Detection Systems
from .structure_detector import StructureDetector, StructureAnalysis, StructureGrade
from .vpvr_analyzer import VPVRAnalyzer, VPVRAnalysis, VolumeNode, ValueArea
from .mtf_structure import MTFStructureAnalyzer, MTFAlignment
from .orderflow_detector import OrderFlowDetector, OrderFlowAnalysis, FlowDirection

# Multi-shot entry system (separate feature)
from .adaptive_budget import AdaptiveRiskBudget, TradeBudget, Shot, ShotStatus

# Session Management (live position tracking)
from .session import (
    SessionManager, SessionState, SessionEntry, SessionUpdate,
    SessionStatus, ExitReason, TradePhase, PartialExit
)

__all__ = [
    # Main Engine
    "RiskEngine",
    "RiskEngineConfig",
    "RiskLevels",
    "RiskUpdate",
    "TradeSetup",
    "PositionUpdate",
    "StopType",
    "TargetType",
    "StructureHealth",
    "GuardingLineManager",
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
