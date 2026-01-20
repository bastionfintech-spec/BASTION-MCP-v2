"""
BASTION API Tests
=================

Test suite for the BASTION risk management API.

Run with: pytest tests/test_api.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


@pytest.fixture(scope="function")
def client():
    """Create a fresh test client for each test."""
    from api.server import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_correct_status(self, client):
        """Health endpoint should return status 'ok'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
    
    def test_health_returns_service_name(self, client):
        """Health endpoint should return service name 'BASTION'."""
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "BASTION"
    
    def test_health_returns_version(self, client):
        """Health endpoint should return version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "2.1.0"


class TestRootEndpoint:
    """Tests for the root / endpoint."""
    
    def test_root_returns_200(self, client):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_api_info(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")
        data = response.json()
        assert data["service"] == "BASTION API"
        assert "docs" in data
        assert "health" in data
        assert "calculate" in data


class TestCalculateEndpointValidation:
    """Tests for /calculate endpoint input validation."""
    
    def test_calculate_requires_entry_price(self, client):
        """Calculate endpoint should require entry_price."""
        response = client.post("/calculate", json={
            "symbol": "BTCUSDT",
            "direction": "long",
            "timeframe": "4h"
        })
        assert response.status_code == 422  # Validation error
    
    def test_calculate_requires_direction(self, client):
        """Calculate endpoint should require direction."""
        response = client.post("/calculate", json={
            "symbol": "BTCUSDT",
            "entry_price": 95000,
            "timeframe": "4h"
        })
        assert response.status_code == 422
    
    def test_calculate_validates_direction(self, client):
        """Calculate endpoint should only accept 'long' or 'short'."""
        response = client.post("/calculate", json={
            "symbol": "BTCUSDT",
            "entry_price": 95000,
            "direction": "sideways",  # Invalid
            "timeframe": "4h"
        })
        assert response.status_code == 422
    
    def test_calculate_validates_positive_entry_price(self, client):
        """Calculate endpoint should require positive entry price."""
        response = client.post("/calculate", json={
            "symbol": "BTCUSDT",
            "entry_price": -1000,
            "direction": "long",
            "timeframe": "4h"
        })
        assert response.status_code == 422


# Integration tests that require network access
@pytest.mark.skipif(
    True,  # Skip by default - set to False to run integration tests
    reason="Integration tests require network access"
)
class TestCalculateEndpointIntegration:
    """Integration tests for /calculate endpoint (require network)."""
    
    def test_calculate_btc_long(self, client):
        """Calculate endpoint should work for BTC long position."""
        response = client.post("/calculate", json={
            "symbol": "BTCUSDT",
            "entry_price": 95000,
            "direction": "long",
            "timeframe": "4h",
            "account_balance": 100000,
            "risk_per_trade_pct": 1.0
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "BTCUSDT"
        assert data["direction"] == "long"
        assert len(data["stops"]) > 0
        assert len(data["targets"]) > 0
        assert "market_context" in data
    
    def test_calculate_btc_short(self, client):
        """Calculate endpoint should work for BTC short position."""
        response = client.post("/calculate", json={
            "symbol": "BTCUSDT",
            "entry_price": 95000,
            "direction": "short",
            "timeframe": "4h",
            "account_balance": 100000,
            "risk_per_trade_pct": 1.0
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "BTCUSDT"
        assert data["direction"] == "short"


class TestCoreComponents:
    """Tests for core risk engine components (no network required)."""
    
    def test_risk_engine_import(self):
        """RiskEngine should be importable."""
        from core.risk_engine import RiskEngine
        engine = RiskEngine()
        assert engine is not None
    
    def test_trade_setup_import(self):
        """TradeSetup model should be importable."""
        from core.risk_engine import TradeSetup
        setup = TradeSetup(
            symbol="BTCUSDT",
            entry_price=95000,
            direction="long",
            timeframe="4h"
        )
        assert setup.entry_price == 95000
        assert setup.direction == "long"
    
    def test_risk_levels_structure(self):
        """RiskLevels should have correct structure."""
        from core.risk_engine import RiskLevels
        
        levels = RiskLevels(
            entry_price=95000,
            direction="long",
            timeframe="4h",
            symbol="BTCUSDT"
        )
        
        assert levels.entry_price == 95000
        assert levels.stops == []
        assert levels.targets == []
        assert levels.structure_quality == 0.0
        assert levels.volume_profile_score == 0.0
        assert levels.orderflow_bias == "neutral"
        assert levels.mtf_alignment == 0.0
    
    def test_stop_type_enum(self):
        """StopType enum should have correct values."""
        from core.risk_engine import StopType
        
        assert StopType.PRIMARY.value == "primary"
        assert StopType.SECONDARY.value == "secondary"
        assert StopType.SAFETY_NET.value == "safety_net"
        assert StopType.GUARDING.value == "guarding"
    
    def test_target_type_enum(self):
        """TargetType enum should have correct values."""
        from core.risk_engine import TargetType
        
        assert TargetType.STRUCTURAL.value == "structural"
        assert TargetType.VPVR.value == "vpvr"
        assert TargetType.EXTENSION.value == "extension"
        assert TargetType.DYNAMIC.value == "dynamic"


class TestGuardingLineManager:
    """Tests for integrated guarding line."""
    
    def test_guarding_line_import(self):
        """GuardingLineManager should be importable."""
        from core.risk_engine import GuardingLineManager
        gl = GuardingLineManager()
        assert gl is not None
    
    def test_guarding_line_calculation(self):
        """GuardingLineManager should calculate initial line."""
        from core.risk_engine import GuardingLineManager
        
        gl = GuardingLineManager(activation_bars=10)
        
        # Mock lows for a long position
        lows = [94000, 93800, 93900, 94100, 94000, 93700, 93800, 94000, 94200, 94100,
                94300, 94500, 94400, 94600, 94700, 94500, 94800, 94900, 95000, 95100]
        
        line = gl.calculate_initial_line(
            entry_price=95000,
            direction="long",
            price_data=lows
        )
        
        assert "slope" in line
        assert "intercept" in line
        assert "activation_bar" in line
    
    def test_guarding_line_level(self):
        """GuardingLineManager should return current level."""
        from core.risk_engine import GuardingLineManager
        
        gl = GuardingLineManager(activation_bars=5)
        
        line = {
            "slope": 50,
            "intercept": 94000,
            "activation_bar": 5,
            "buffer_pct": 0.3
        }
        
        # Before activation
        level_before = gl.get_current_level(line, bars_since_entry=3)
        
        # After activation
        level_after = gl.get_current_level(line, bars_since_entry=10)
        
        # After activation, level should be higher (for positive slope)
        assert level_after > level_before
    
    def test_guarding_break_detection(self):
        """GuardingLineManager should detect breaks."""
        from core.risk_engine import GuardingLineManager
        
        gl = GuardingLineManager()
        
        # Long position - price below guarding = break
        is_broken, reason = gl.check_break(93000, 94000, "long")
        assert is_broken
        assert "broke below" in reason
        
        # Long position - price above guarding = no break
        is_broken, _ = gl.check_break(95000, 94000, "long")
        assert not is_broken


class TestAdaptiveBudget:
    """Tests for adaptive risk budget system."""
    
    def test_budget_creation(self):
        """Should create a risk budget."""
        from core.adaptive_budget import AdaptiveRiskBudget
        
        budget_mgr = AdaptiveRiskBudget(max_shots=3, total_risk_cap=2.0)
        budget = budget_mgr.create_budget("BTCUSDT", "long")
        
        assert budget is not None
        assert budget.symbol == "BTCUSDT"
        assert budget.direction == "long"
        assert budget.can_take_shot
    
    def test_shot_taking(self):
        """Should take shots against budget."""
        from core.adaptive_budget import AdaptiveRiskBudget
        
        budget_mgr = AdaptiveRiskBudget(max_shots=3, total_risk_cap=2.0)
        budget = budget_mgr.create_budget("BTCUSDT", "long")
        
        # Take first shot
        shot = budget_mgr.take_shot(
            budget.id,
            entry_price=95000,
            stop_price=93000,
            account_balance=100000
        )
        
        assert shot is not None
        assert shot.entry_price == 95000
        assert shot.size > 0
        assert budget.risk_used > 0


class TestSessionManager:
    """Tests for session management."""
    
    def test_session_manager_import(self):
        """SessionManager should be importable."""
        from core.session import SessionManager
        manager = SessionManager()
        assert manager is not None
    
    def test_create_session(self):
        """Should create a trading session."""
        from core.session import SessionManager, SessionStatus
        
        manager = SessionManager()
        session = manager.create_session(
            symbol="BTCUSDT",
            direction="long",
            timeframe="4h",
            account_balance=100000,
            structural_support=93200,
            targets=[
                {"price": 97500, "exit_percentage": 33, "reason": "T1"},
                {"price": 99800, "exit_percentage": 33, "reason": "T2"},
            ],
        )
        
        assert session.id is not None
        assert session.symbol == "BTCUSDT"
        assert session.direction == "long"
        assert session.status == SessionStatus.PENDING
        assert session.structural_stop == 93200
    
    def test_take_shot(self):
        """Should take a shot in session."""
        from core.session import SessionManager, SessionStatus
        
        manager = SessionManager()
        session = manager.create_session(
            symbol="BTCUSDT",
            direction="long",
            timeframe="4h",
            account_balance=100000,
            structural_support=93200,
            targets=[],
        )
        
        entry = manager.take_shot(
            session_id=session.id,
            entry_price=94500,
            current_atr=600,
        )
        
        assert entry is not None
        assert entry.shot_number == 1
        assert entry.entry_price == 94500
        assert entry.size > 0
        assert session.status == SessionStatus.ACTIVE
    
    def test_multi_shot_allocation(self):
        """Multi-shot should allocate 50% -> 30% -> 20%."""
        from core.session import SessionManager
        
        manager = SessionManager()
        session = manager.create_session(
            symbol="BTCUSDT",
            direction="long",
            timeframe="4h",
            account_balance=100000,
            structural_support=93200,
            targets=[],
        )
        
        # Shot 1: 50% of $2000 = $1000
        shot1 = manager.take_shot(session.id, 94500, 600)
        # Shot 2: 30% of $2000 = $600
        shot2 = manager.take_shot(session.id, 93850, 600)
        # Shot 3: 20% of $2000 = $400
        shot3 = manager.take_shot(session.id, 95200, 600)
        
        assert shot1.risk_amount == 1000  # 50%
        assert shot2.risk_amount == 600   # 30%
        assert shot3.risk_amount == 400   # 20%
        assert session.shots_taken == 3
    
    def test_phase_transition(self):
        """Should transition from Phase 1 to Phase 2 at bar 10."""
        from core.session import SessionManager, TradePhase
        
        manager = SessionManager()
        session = manager.create_session(
            symbol="BTCUSDT",
            direction="long",
            timeframe="4h",
            account_balance=100000,
            structural_support=93200,
            targets=[],
        )
        
        manager.take_shot(session.id, 94500, 600)
        
        # Update at bar 5 (Phase 1)
        update = manager.update_session(session.id, 95000, 5)
        assert update.phase == TradePhase.PHASE_1
        
        # Update at bar 11 (Phase 2)
        update = manager.update_session(
            session.id, 96000, 11,
            recent_lows=[94000, 94200, 94100, 94300, 94500]
        )
        assert update.phase == TradePhase.PHASE_2
        assert update.guarding_level is not None
    
    def test_exit_on_structure_break(self):
        """Should signal exit when structure breaks."""
        from core.session import SessionManager, ExitReason
        
        manager = SessionManager()
        session = manager.create_session(
            symbol="BTCUSDT",
            direction="long",
            timeframe="4h",
            account_balance=100000,
            structural_support=93200,
            targets=[],
        )
        
        manager.take_shot(session.id, 94500, 600)
        
        # Price breaks below structural support
        update = manager.update_session(session.id, 93000, 5)
        
        assert update.exit_signal == True
        assert update.exit_reason == ExitReason.STRUCTURE_BROKEN
        assert update.exit_percentage == 100.0
    
    def test_exit_on_target_hit(self):
        """Should signal exit when target hit."""
        from core.session import SessionManager, ExitReason
        
        manager = SessionManager()
        session = manager.create_session(
            symbol="BTCUSDT",
            direction="long",
            timeframe="4h",
            account_balance=100000,
            structural_support=93200,
            targets=[
                {"price": 97500, "exit_percentage": 33, "reason": "T1"},
            ],
        )
        
        manager.take_shot(session.id, 94500, 600)
        
        # Price hits target
        update = manager.update_session(session.id, 97600, 8)
        
        assert update.exit_signal == True
        assert update.exit_reason == ExitReason.TARGET_HIT
        assert update.exit_percentage == 33


class TestDetectionSystems:
    """Tests for detection system imports."""
    
    def test_structure_detector_import(self):
        """StructureDetector should be importable."""
        from core.structure_detector import StructureDetector
        detector = StructureDetector()
        assert detector is not None
    
    def test_vpvr_analyzer_import(self):
        """VPVRAnalyzer should be importable."""
        from core.vpvr_analyzer import VPVRAnalyzer
        analyzer = VPVRAnalyzer()
        assert analyzer is not None
    
    def test_mtf_analyzer_import(self):
        """MTFStructureAnalyzer should be importable."""
        from core.mtf_structure import MTFStructureAnalyzer
        analyzer = MTFStructureAnalyzer()
        assert analyzer is not None
    
    def test_orderflow_detector_import(self):
        """OrderFlowDetector should be importable."""
        from core.orderflow_detector import OrderFlowDetector
        detector = OrderFlowDetector()
        assert detector is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
