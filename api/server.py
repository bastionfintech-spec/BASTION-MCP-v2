"""
BASTION API Server
==================

FastAPI application for BASTION risk management.

Endpoints:
    GET  /health   - Health check
    POST /calculate - Calculate risk levels for a trade
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add parent directory to path for imports
bastion_path = Path(__file__).parent.parent
sys.path.insert(0, str(bastion_path))

from api.models import (
    CalculateRiskRequest,
    RiskLevelsResponse,
    ErrorResponse,
    HealthResponse,
    StopLevelResponse,
    TargetLevelResponse,
    MarketContextResponse
)
from core.risk_engine import RiskEngine
from data.fetcher import LiveDataFetcher

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
risk_engine: RiskEngine = None
data_fetcher: LiveDataFetcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global risk_engine, data_fetcher
    
    # Startup
    logger.info("BASTION API starting up...")
    risk_engine = RiskEngine()
    data_fetcher = LiveDataFetcher()
    logger.info("Risk engine initialized (Advanced Mode)")
    logger.info("Data fetcher ready")
    
    yield
    
    # Shutdown
    logger.info("BASTION API shutting down...")
    await data_fetcher.close()
    await risk_engine.close()


# Initialize FastAPI with lifespan
app = FastAPI(
    title="BASTION API",
    description="Strategy-Agnostic Risk Management Infrastructure",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "service": "BASTION API",
        "version": "2.0.0",
        "mode": "Advanced (VPVR + Structure + MTF + OrderFlow)",
        "docs": "/docs",
        "health": "/health",
        "calculate": "/calculate"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service="BASTION",
        version="2.0.0"
    )


@app.post("/calculate", response_model=RiskLevelsResponse, tags=["Risk"])
async def calculate_risk(request: CalculateRiskRequest):
    """
    Calculate risk levels for a trade setup.
    
    Uses advanced detection:
    - Structure Detection (Grade 1-4 trendlines)
    - Volume Profile (VPVR - HVN/LVN)
    - Multi-Timeframe Alignment
    - Order Flow Analysis
    
    Returns:
    - Structural stop-loss levels
    - Volume-informed take-profit targets
    - Volatility-adjusted position sizing
    - Market context (informational, not judgmental)
    """
    try:
        logger.info(f"Calculating risk for {request.symbol} {request.direction} @ {request.entry_price}")
        
        # Fetch market data for multiple timeframes
        try:
            # Primary timeframe
            primary_df = await data_fetcher.get_ohlcv(
                symbol=request.symbol,
                interval=request.timeframe,
                limit=200
            )
            
            if primary_df.empty:
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not fetch market data for {request.symbol}"
                )
            
            # Build OHLCV data dict
            ohlcv_data = {request.timeframe: primary_df}
            
            # Try to fetch higher timeframe for MTF analysis
            higher_tf = _get_higher_timeframe(request.timeframe)
            if higher_tf:
                try:
                    higher_df = await data_fetcher.get_ohlcv(
                        symbol=request.symbol,
                        interval=higher_tf,
                        limit=100
                    )
                    if not higher_df.empty:
                        ohlcv_data[higher_tf] = higher_df
                except Exception:
                    pass  # Higher TF is optional
            
            logger.info(f"Fetched data for {list(ohlcv_data.keys())}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch market data: {str(e)}"
            )
        
        # Calculate risk levels using advanced engine
        try:
            levels = await risk_engine.calculate_risk_levels(
                symbol=request.symbol,
                entry_price=request.entry_price,
                direction=request.direction,
                timeframe=request.timeframe,
                account_balance=request.account_balance,
                ohlcv_data=ohlcv_data,
                risk_per_trade_pct=request.risk_per_trade_pct
            )
            
            logger.info(f"Risk calculated: {len(levels.stops)} stops, {len(levels.targets)} targets")
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Risk calculation error: {str(e)}"
            )
        
        # Format response
        response = RiskLevelsResponse(
            symbol=levels.symbol,
            entry_price=levels.entry_price,
            direction=levels.direction,
            current_price=levels.current_price,
            stops=[
                StopLevelResponse(
                    type=s.get('type', 'unknown'),
                    price=s['price'],
                    distance_pct=s.get('distance_pct', 0),
                    reason=s.get('reason', ''),
                    confidence=s.get('confidence', 0.5)
                )
                for s in levels.stops
            ],
            targets=[
                TargetLevelResponse(
                    price=t['price'],
                    type=t.get('type', 'unknown'),
                    exit_percentage=t.get('exit_percentage', 33),
                    distance_pct=t.get('distance_pct', 0),
                    reason=t.get('reason', ''),
                    confidence=t.get('confidence', 0.5)
                )
                for t in levels.targets
            ],
            position_size=levels.position_size,
            position_size_pct=levels.position_size_pct,
            risk_amount=levels.risk_amount,
            risk_reward_ratio=levels.risk_reward_ratio,
            max_risk_reward_ratio=levels.max_risk_reward_ratio,
            guarding_line=levels.guarding_line,
            market_context=MarketContextResponse(
                structure_quality=levels.structure_quality,
                volume_profile_score=levels.volume_profile_score,
                orderflow_bias=levels.orderflow_bias,
                mtf_alignment=levels.mtf_alignment
            )
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /calculate: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


def _get_higher_timeframe(timeframe: str) -> str:
    """Get the next higher timeframe for MTF analysis."""
    tf_hierarchy = {
        "1m": "5m",
        "5m": "15m",
        "15m": "1h",
        "1h": "4h",
        "4h": "1d",
        "1d": "1w",
    }
    return tf_hierarchy.get(timeframe.lower(), None)


# Mount static files for web calculator
web_dir = bastion_path / "web"
if web_dir.exists():
    app.mount("/app", StaticFiles(directory=str(web_dir), html=True), name="web")
    logger.info(f"Mounted web calculator at /app")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
