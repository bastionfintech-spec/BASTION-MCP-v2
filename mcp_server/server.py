"""
BASTION MCP Server — Core
Exposes the full BASTION platform as MCP tools for Claude agents.

Tools (108):
  CORE AI (auth optional — pass api_key for user-scoped results)
    bastion_evaluate_risk        — AI risk intelligence for a position
    bastion_chat                 — Neural chat (ask anything about markets)
    bastion_evaluate_all_positions — Evaluate all open positions at once
    bastion_scan_signals         — Scan for trading signals across pairs

  MARKET DATA (public — no auth needed)
    bastion_get_price            — Live crypto price
    bastion_get_market_data      — Aggregated market intelligence
    bastion_get_klines           — Candlestick OHLCV data
    bastion_get_volatility       — Volatility metrics + regime detection
    bastion_get_volatility_regime — Volatility regime classification
    bastion_get_btc_dominance    — BTC dominance + altseason score
    bastion_get_correlation_matrix — Cross-asset correlation matrix
    bastion_get_confluence       — Multi-timeframe confluence scanner
    bastion_get_sector_rotation  — Sector rotation tracker

  DERIVATIVES & ORDER FLOW (public)
    bastion_get_open_interest    — Open interest across exchanges
    bastion_get_oi_changes       — OI changes across all pairs
    bastion_get_cvd              — Cumulative Volume Delta
    bastion_get_orderflow        — Order flow analysis
    bastion_get_funding_rates    — Cross-exchange funding rates
    bastion_get_funding_arb      — Funding rate arbitrage
    bastion_get_liquidations     — Liquidation events + clusters
    bastion_get_liquidations_by_exchange — Liquidations per exchange
    bastion_get_heatmap          — Liquidation heatmap
    bastion_get_taker_ratio      — Taker buy/sell ratio
    bastion_get_top_traders      — Top trader positioning
    bastion_get_market_maker_magnet — MM gamma magnet levels
    bastion_get_options          — Options OI, P/C ratio, max pain

  ON-CHAIN & INTELLIGENCE (public)
    bastion_get_whale_activity   — Whale transactions (11 chains)
    bastion_get_exchange_flow    — Exchange inflow/outflow
    bastion_get_onchain          — On-chain metrics
    bastion_get_news             — Aggregated crypto news
    bastion_get_smart_money      — Smart money flow analysis
    bastion_get_hyperliquid_whales — Top Hyperliquid whale positions

  MACRO & SENTIMENT (public)
    bastion_get_fear_greed       — Fear & Greed Index
    bastion_get_macro_signals    — Macro signals (DXY, yields, equities)
    bastion_get_etf_flows        — BTC/ETH ETF flow data
    bastion_get_stablecoin_markets — Stablecoin supply + flows
    bastion_get_economic_data    — FRED economic data series
    bastion_get_polymarket       — Prediction market data

  RESEARCH (auth optional)
    bastion_generate_report      — Generate MCF Labs research report
    bastion_get_reports          — List existing reports
    bastion_calculate_position   — Position sizing + Monte Carlo
    bastion_get_kelly_sizing     — Kelly Criterion optimal sizing
    bastion_log_trade            — Log trade to performance journal
    bastion_get_trade_journal    — Journal stats + real Kelly sizing
    bastion_backtest_strategy    — Backtest strategies on-demand
    bastion_get_risk_parity      — Portfolio risk parity analysis
    bastion_strategy_builder     — Natural language → backtest pipeline
    bastion_risk_replay          — Historical position time-travel analysis
    bastion_risk_card            — Interactive risk visualization widget
    bastion_get_leaderboard      — Model performance leaderboard
    bastion_log_prediction       — Log prediction for leaderboard tracking
    bastion_subscribe_alert      — Subscribe to price/condition alerts
    bastion_check_alerts         — Check active & triggered alerts
    bastion_cancel_alert         — Cancel an alert subscription
    bastion_create_risk_card     — Shareable risk score card with unique URL
    bastion_get_performance      — Equity curve + Sharpe + drawdown analytics
    bastion_record_equity        — Record equity snapshot for tracking
    bastion_add_webhook          — Register Discord/Telegram/URL webhook
    bastion_list_webhooks        — List notification webhooks
    bastion_send_notification    — Push notification to webhooks
    bastion_get_agent_analytics  — Agent usage + latency dashboard
    bastion_format_risk          — Beautiful terminal-style risk output

  PORTFOLIO (auth required — 'read' scope)
    bastion_get_positions        — All open positions
    bastion_get_balance          — Total portfolio balance
    bastion_get_exchanges        — Connected exchanges
    bastion_engine_status        — Trading engine status
    bastion_engine_history       — Engine execution history
    bastion_get_alerts           — Active alerts & notifications
    bastion_get_session_stats    — Trading session statistics

  TRADING ACTIONS (auth required — 'trade' scope)
    bastion_emergency_exit       — Close ALL positions immediately
    bastion_partial_close        — Close % of a position
    bastion_set_take_profit      — Set/update take-profit
    bastion_set_stop_loss        — Set/update stop-loss
    bastion_move_to_breakeven    — Move stops to entry price
    bastion_flatten_winners      — Close all winning positions

  ENGINE CONTROL (auth required — 'engine' scope)
    bastion_engine_start         — Start autonomous risk engine
    bastion_engine_arm           — Arm engine for auto-execution
    bastion_engine_disarm        — Disarm engine (advisory only)

  WAR ROOM (multi-agent intelligence hub)
    bastion_war_room_post        — Post signal to War Room
    bastion_war_room_read        — Read War Room feed
    bastion_war_room_consensus   — Get agent consensus
    bastion_war_room_vote        — Cast weighted vote (accuracy-based)
    bastion_war_room_consensus_weighted — Get weighted consensus

  INTELLIGENCE (protocol-level innovation)
    bastion_quick_intel          — Instant intelligence snapshot (~200ms)
    bastion_deep_analysis        — Multi-step reasoning chain (15 sources in one call)
    bastion_get_regime_tools     — Discover market-regime adaptive tools
    bastion_execute_regime_tool  — Execute regime-specific analysis
    bastion_live_feed            — Real-time aggregated market feed
    bastion_get_server_card      — MCP Server Card for agent discovery
    bastion_risk_confirm         — Interactive risk confirmation + warnings

  CHALLENGE CARDS (viral prediction market)
    bastion_create_challenge     — Create a "Prove Me Wrong" prediction
    bastion_counter_challenge    — Take the opposite side
    bastion_get_challenges       — List active challenges
    bastion_score_challenge      — Auto-score a challenge

  PERSISTENT MEMORY (cross-session intelligence)
    bastion_memory_store         — Store episodic/semantic/procedural memory
    bastion_memory_recall        — Search and recall memories
    bastion_memory_profile       — Get learned trader profile

  HEAT INDEX (attention market score)
    bastion_heat_index           — Composite attention score for a symbol
    bastion_heat_scan            — Scan all pairs for hottest activity

  COPY-ANALYSIS WORKFLOWS
    bastion_save_workflow        — Save a reusable tool chain
    bastion_get_workflows        — Browse public workflows
    bastion_run_workflow         — Execute a saved workflow

  AUDIT & COMPLIANCE (enterprise-grade)
    bastion_audit_log            — Log hash-chained audit entry
    bastion_audit_trail          — Query immutable audit trail
    bastion_audit_verify         — Verify chain integrity
    bastion_decision_provenance  — Record full decision reasoning chain
    bastion_get_provenance       — Query decision provenance
    bastion_tool_safety          — Get safety annotations for all tools

Resources (8):
  bastion://status, bastion://supported-symbols, bastion://model-info,
  bastion://tools, bastion://exchanges, bastion://capabilities,
  bastion://live/{symbol}, bastion://regime

Prompts (10):
  evaluate_my_position, market_analysis, risk_check, portfolio_risk_scan,
  whale_tracker, macro_briefing, pre_trade_analysis, strategy_lab,
  full_risk_dashboard, deep_intelligence_brief
"""
import json
import logging
import httpx
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import config

logger = logging.getLogger("bastion.mcp.server")

# ── Create FastMCP Server ───────────────────────────────────────
mcp = FastMCP(
    config.MCP_SERVER_NAME,
    instructions=config.MCP_SERVER_DESCRIPTION,
)

# ── HTTP Client ─────────────────────────────────────────────────
_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=config.API_BASE_URL,
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={"User-Agent": f"BASTION-MCP/{config.MCP_SERVER_VERSION}"}
        )
    return _client


async def api_get(path: str, params: dict = None, auth_headers: dict = None) -> dict:
    """Make a GET request to the BASTION API."""
    client = await get_client()
    try:
        resp = await client.get(path, params=params, headers=auth_headers)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API returned {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


async def api_post(path: str, data: dict = None, auth_headers: dict = None) -> dict:
    """Make a POST request to the BASTION API."""
    client = await get_client()
    try:
        resp = await client.post(path, json=data, headers=auth_headers)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API returned {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


async def resolve_auth(api_key: Optional[str], required_scope: str = "read") -> tuple:
    """
    Validate a bst_ API key and return auth headers for the backend.

    Returns:
        (auth_headers dict, error_string or None)
        If auth fails, auth_headers is None and error is set.
        If no api_key provided, returns empty headers (guest mode).
    """
    from .auth import validate_bst_key, check_scope

    if not api_key:
        return {}, None  # Guest mode — no auth headers

    key_info = await validate_bst_key(api_key)
    if not key_info:
        return None, "Invalid or expired API key."

    if not check_scope(key_info, required_scope):
        return None, f"API key lacks required scope: '{required_scope}'. Your scopes: {key_info.get('scopes', [])}"

    user_id = key_info.get("user_id")
    if not user_id:
        return {}, None  # Legacy key — no user identity

    # Build internal auth headers for the backend
    headers = {
        "X-Bastion-Internal": config.MCP_INTERNAL_SECRET,
        "X-Bastion-User-Id": user_id,
    }
    return headers, None


# ═════════════════════════════════════════════════════════════════
# TOOLS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_evaluate_risk(
    symbol: str,
    direction: str,
    entry_price: float,
    current_price: float,
    leverage: float = 1.0,
    stop_loss: float = 0.0,
    take_profit_1: float = 0.0,
    take_profit_2: float = 0.0,
    take_profit_3: float = 0.0,
    position_size_usd: float = 1000.0,
    duration_hours: float = 0.0,
    api_key: str = "",
) -> str:
    """Evaluate a crypto futures position using BASTION's AI risk intelligence engine.

    This is the core tool — sends a position to the fine-tuned 72B model which analyzes
    560+ real-time signals (liquidations, whale flows, funding rates, order flow, VPVR,
    market structure) and returns a specific action with reasoning.

    Args:
        symbol: Trading pair symbol (e.g. BTC, ETH, SOL)
        direction: Position direction — LONG or SHORT
        entry_price: Price at which the position was opened
        current_price: Current market price
        leverage: Position leverage multiplier (default 1.0)
        stop_loss: Stop loss price (0 = no stop set)
        take_profit_1: First take profit target (0 = none)
        take_profit_2: Second take profit target (0 = none)
        take_profit_3: Third take profit target (0 = none)
        position_size_usd: Position size in USD (default 1000)
        duration_hours: How long the position has been open in hours
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        JSON with: action (HOLD/EXIT_FULL/TP_PARTIAL/EXIT_100%/REDUCE_SIZE/TRAIL_STOP),
        urgency (LOW/MEDIUM/HIGH/CRITICAL), confidence (0-1), reason, detailed reasoning,
        and execution details.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    symbol = symbol.upper().replace("USDT", "").replace("USD", "").replace("-PERP", "")

    # Build take profits list
    tps = [tp for tp in [take_profit_1, take_profit_2, take_profit_3] if tp > 0]

    # Calculate unrealized PnL
    if direction.upper() == "LONG":
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
    else:
        pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage

    payload = {
        "position": {
            "symbol": symbol,
            "direction": direction.upper(),
            "entry_price": entry_price,
            "current_price": current_price,
            "stop_loss": stop_loss if stop_loss > 0 else None,
            "take_profits": tps if tps else None,
            "leverage": leverage,
            "duration_hours": duration_hours,
            "unrealized_pnl_pct": round(pnl_pct, 2),
            "position_size_usd": position_size_usd,
        }
    }

    result = await api_post("/api/risk/evaluate", payload, auth_headers=auth_headers)

    if "error" in result:
        return json.dumps(result, indent=2)

    # Format response for Claude
    output = {
        "action": result.get("action", "UNKNOWN"),
        "urgency": result.get("urgency", "UNKNOWN"),
        "confidence": result.get("confidence", 0),
        "reason": result.get("reason", ""),
        "reasoning": result.get("reasoning", ""),
        "execution": result.get("execution", {}),
        "unrealized_pnl_pct": round(pnl_pct, 2),
    }
    return json.dumps(output, indent=2)


@mcp.tool()
async def bastion_chat(
    query: str,
    symbol: str = "BTC",
    api_key: str = "",
) -> str:
    """Ask BASTION's neural AI anything about crypto markets, trading, or risk.

    Powered by the same fine-tuned 72B model used for risk intelligence.
    Context-aware — knows current market conditions, whale activity, and structure.

    Args:
        query: Your question (e.g. "Is BTC a buy right now?", "Where are whales moving?")
        symbol: Symbol context for the query (default BTC)
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        AI-generated analysis in markdown format.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})

    result = await api_post("/api/neural/chat", {
        "query": query,
        "symbol": symbol.upper(),
        "include_positions": bool(api_key),
        "session_id": "mcp-session",
        "user_id": "mcp-user",
    }, auth_headers=auth_headers)

    if "error" in result:
        return f"Error: {result['error']}"

    return result.get("response", "No response from neural engine.")


@mcp.tool()
async def bastion_get_price(symbol: str = "BTC") -> str:
    """Get the current live price of a cryptocurrency.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL, AVAX, DOGE)

    Returns:
        Current price, 24h change, and basic market data.
    """
    result = await api_get(f"/api/price/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_market_data(symbol: str = "BTC") -> str:
    """Get aggregated market intelligence for a cryptocurrency.

    Fetches multiple data points in one call: price, CVD (order flow),
    open interest, funding rates, and volatility metrics.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Comprehensive market data including order flow, derivatives, and volatility.
    """
    sym = symbol.upper()

    # Fetch multiple endpoints in parallel via the market endpoint
    result = await api_get(f"/api/market/{sym}")

    if "error" in result:
        # Fallback: fetch individual endpoints
        price = await api_get(f"/api/price/{sym}")
        funding = await api_get("/api/funding")
        return json.dumps({
            "price": price,
            "funding": funding,
            "note": "Partial data — some endpoints unavailable"
        }, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_liquidations(symbol: str = "BTC") -> str:
    """Get liquidation data for a cryptocurrency.

    Shows recent liquidations, liquidation clusters, and cascade risk zones.
    Critical for understanding where leveraged positions are getting wiped.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Liquidation data including recent events, clusters, and risk zones.
    """
    result = await api_get(f"/api/liquidations/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_whale_activity(
    min_value_usd: int = 1000000,
    limit: int = 20,
) -> str:
    """Get recent whale transactions across major blockchains.

    Tracks large transfers (>$1M) across BTC, ETH, SOL and 8 other chains.
    Includes exchange attribution (which exchange funds are moving to/from).

    Args:
        min_value_usd: Minimum transaction value in USD (default 1,000,000)
        limit: Number of transactions to return (default 20, max 50)

    Returns:
        Recent whale transactions with amounts, from/to, exchange attribution.
    """
    result = await api_get("/api/whales", {
        "min_value": min_value_usd,
        "limit": min(limit, 50),
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_funding_rates() -> str:
    """Get cross-exchange funding rates for major crypto pairs.

    Funding rates indicate market sentiment — positive = longs paying shorts (bullish
    overcrowding), negative = shorts paying longs (bearish overcrowding).

    Returns:
        Funding rates for BTC, ETH, SOL across Binance, Bybit, OKX, and more.
    """
    result = await api_get("/api/funding")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_generate_report(
    report_type: str = "institutional_research",
    symbol: str = "BTC",
    api_key: str = "",
) -> str:
    """Generate an AI-powered MCF Labs research report.

    Creates institutional-grade analysis with ratings, conviction scores,
    target prices, risk factors, trade structures, and valuation scenarios.

    Args:
        report_type: Type of report — institutional_research (default), market_structure,
            whale_intelligence, options_flow, or cycle_position
        symbol: Symbol to analyze (default BTC)
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        Full report with bias, confidence, sections (thesis, key drivers, risks, scenarios).
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    valid_types = [
        "institutional_research", "market_structure",
        "whale_intelligence", "options_flow", "cycle_position"
    ]
    if report_type not in valid_types:
        return json.dumps({
            "error": f"Invalid report type. Choose from: {', '.join(valid_types)}"
        })

    result = await api_post(
        f"/api/mcf/generate/{report_type}",
        {"symbol": symbol.upper()},
        auth_headers=auth_headers,
    )

    if "error" in result:
        return json.dumps(result, indent=2)

    report = result.get("report", {})
    # Return summary for MCP (full report can be very large)
    output = {
        "success": True,
        "id": report.get("id"),
        "type": report.get("type"),
        "title": report.get("title"),
        "bias": report.get("bias"),
        "confidence": report.get("confidence"),
        "summary": report.get("summary"),
        "sections": report.get("sections", {}),
    }
    return json.dumps(output, indent=2)


@mcp.tool()
async def bastion_get_reports(
    limit: int = 10,
    report_type: str = "",
    symbol: str = "",
    api_key: str = "",
) -> str:
    """List existing MCF Labs research reports.

    Args:
        limit: Number of reports to return (default 10)
        report_type: Filter by type (institutional_research, market_structure, etc.)
        symbol: Filter by symbol (btc, eth, sol)
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        List of reports with id, title, bias, type, timestamp, and tags.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})

    params = {"limit": min(limit, 50)}
    if report_type:
        params["type"] = report_type
    if symbol:
        params["symbol"] = symbol.lower()

    result = await api_get("/api/mcf/reports", params, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_calculate_position(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    leverage: float = 1.0,
    account_size: float = 10000.0,
    risk_percent: float = 2.0,
    api_key: str = "",
) -> str:
    """Calculate optimal position size with Monte Carlo risk simulation.

    Runs 50,000 simulations to estimate probability of hitting TP vs SL,
    expected value, and quality score for the trade setup.

    Args:
        symbol: Trading pair (e.g. BTC, ETH, SOL)
        direction: LONG or SHORT
        entry_price: Planned entry price
        stop_loss: Stop loss price
        take_profit: Take profit target price
        leverage: Position leverage (default 1.0)
        account_size: Account size in USD (default 10000)
        risk_percent: Max risk per trade as % of account (default 2.0)
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        Position size, risk/reward ratio, TP probability, SL probability,
        expected value, and trade quality score.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/pre-trade-calculator", {
        "symbol": symbol.upper(),
        "direction": direction.upper(),
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "leverage": leverage,
        "account_size": account_size,
        "risk_percent": risk_percent,
    }, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOLS — DERIVATIVES & ORDER FLOW
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_open_interest(symbol: str = "BTC") -> str:
    """Get open interest data for a cryptocurrency.

    Shows total open interest across exchanges — a key measure of how much
    money is in leveraged positions. Rising OI = new money entering, falling OI = closing.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Total open interest, OI changes, and exchange breakdown.
    """
    result = await api_get(f"/api/oi/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_oi_changes() -> str:
    """Get open interest changes across all major crypto pairs.

    Shows which pairs are seeing the biggest OI increases/decreases.
    Useful for identifying where new leveraged money is flowing.

    Returns:
        OI changes for major pairs across exchanges.
    """
    result = await api_get("/api/oi-changes")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_cvd(symbol: str = "BTC") -> str:
    """Get Cumulative Volume Delta (CVD) for a cryptocurrency.

    CVD tracks the difference between aggressive buying and selling volume.
    Rising CVD = buyers dominating, falling CVD = sellers dominating.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        CVD data showing buy/sell pressure over time.
    """
    result = await api_get(f"/api/cvd/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_orderflow(symbol: str = "BTC") -> str:
    """Get order flow analysis for a cryptocurrency.

    Detailed order flow showing aggressive buyers vs sellers, large orders,
    and imbalances in the order book.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Order flow analysis with buy/sell pressure metrics.
    """
    result = await api_get(f"/api/orderflow/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_volatility(symbol: str = "BTC") -> str:
    """Get volatility metrics and regime detection for a cryptocurrency.

    Includes historical volatility, implied volatility indicators,
    and volatility regime classification (low/normal/high/extreme).

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Volatility metrics, regime classification, and ATR data.
    """
    result = await api_get(f"/api/volatility/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_heatmap(symbol: str = "BTC") -> str:
    """Get the liquidation heatmap for a cryptocurrency.

    Shows where liquidation clusters are building above and below the current price.
    These clusters act as magnets — price tends to sweep them.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Heatmap data showing liquidation density at various price levels.
    """
    result = await api_get(f"/api/heatmap/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_taker_ratio(symbol: str = "BTC") -> str:
    """Get taker buy/sell ratio for a cryptocurrency.

    Taker ratio > 1 means more aggressive buying, < 1 means more aggressive selling.
    A key real-time sentiment indicator from the derivatives markets.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Taker buy/sell ratio across exchanges.
    """
    result = await api_get(f"/api/taker-ratio/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_funding_arb(symbol: str = "BTC") -> str:
    """Get funding rate arbitrage data for a cryptocurrency.

    Shows funding rate differentials across exchanges — opportunities
    where you can earn funding by going long on one exchange and short on another.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Funding rates per exchange with arbitrage spread analysis.
    """
    result = await api_get(f"/api/funding-arb/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_top_traders(symbol: str = "BTC") -> str:
    """Get top trader positions and sentiment for a cryptocurrency.

    Shows how the biggest traders on Binance/Bybit are positioned —
    their long/short ratio and account ratio.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Top trader long/short ratios and positioning data.
    """
    result = await api_get(f"/api/top-traders/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_market_maker_magnet(symbol: str = "BTC") -> str:
    """Get market maker magnet levels for a cryptocurrency.

    Identifies price levels where market maker gamma exposure creates
    magnetic attraction — price tends to pin at these levels.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Key magnet levels and their relative strength.
    """
    result = await api_get(f"/api/mm-magnet/{symbol.upper()}")
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOLS — ON-CHAIN & INTELLIGENCE
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_exchange_flow(symbol: str = "BTC") -> str:
    """Get exchange inflow/outflow for a cryptocurrency.

    Tracks coins moving onto and off of exchanges. Large inflows often
    precede selling; large outflows suggest accumulation.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH)

    Returns:
        Exchange inflow/outflow data with net flow analysis.
    """
    result = await api_get(f"/api/exchange-flow/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_onchain() -> str:
    """Get on-chain metrics for major cryptocurrencies.

    Includes active addresses, transaction counts, MVRV, NVT,
    and other fundamental on-chain indicators.

    Returns:
        On-chain metrics across major networks.
    """
    result = await api_get("/api/onchain")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_fear_greed() -> str:
    """Get the Crypto Fear & Greed Index.

    A composite sentiment indicator from 0 (extreme fear) to 100 (extreme greed).
    Based on volatility, volume, social media, dominance, and trends.

    Returns:
        Current and historical Fear & Greed index values.
    """
    result = await api_get("/api/fear-greed")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_news(limit: int = 20) -> str:
    """Get aggregated crypto news from major sources.

    Pulls news from CoinDesk, Cointelegraph, and other major crypto
    news outlets. Useful for understanding current narratives.

    Args:
        limit: Number of articles to return (default 20)

    Returns:
        Recent news articles with titles, sources, and timestamps.
    """
    result = await api_get("/api/news", {"limit": min(limit, 50)})
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOLS — MACRO & SENTIMENT
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_macro_signals() -> str:
    """Get macro market signals affecting crypto.

    Includes DXY (dollar index), yields, equity indices, gold,
    and their correlation/impact on crypto markets.

    Returns:
        Macro signals with correlation analysis to crypto.
    """
    result = await api_get("/api/macro-signals")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_etf_flows() -> str:
    """Get Bitcoin and Ethereum ETF flow data.

    Tracks daily inflows/outflows for spot BTC and ETH ETFs.
    Large inflows = institutional buying, outflows = institutional selling.

    Returns:
        ETF flow data with daily net flows and cumulative totals.
    """
    result = await api_get("/api/etf-flows")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_stablecoin_markets() -> str:
    """Get stablecoin market data.

    Tracks USDT, USDC, DAI and other stablecoin supply and flows.
    Stablecoin inflows to exchanges often precede buying pressure.

    Returns:
        Stablecoin market caps, supply changes, and exchange reserves.
    """
    result = await api_get("/api/stablecoin-markets")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_economic_data(series_id: str = "DFF") -> str:
    """Get FRED economic data series.

    Access Federal Reserve Economic Data — rates, CPI, unemployment,
    GDP, and hundreds of other macro indicators.

    Args:
        series_id: FRED series ID (e.g. DFF=Fed Funds Rate, CPIAUCSL=CPI,
            T10Y2Y=yield curve, UNRATE=unemployment)

    Returns:
        Economic data series with recent values.
    """
    result = await api_get("/api/fred-data", {"series_id": series_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_polymarket(limit: int = 15) -> str:
    """Get prediction market data from Polymarket.

    Shows active prediction markets relevant to crypto — regulatory
    decisions, ETF approvals, price targets, election outcomes, etc.

    Args:
        limit: Number of markets to return (default 15)

    Returns:
        Active prediction markets with current probabilities and volumes.
    """
    result = await api_get("/api/polymarket", {
        "closed": "false",
        "order": "volume",
        "ascending": "false",
        "limit": min(limit, 50),
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOLS — PORTFOLIO & POSITIONS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_positions(api_key: str = "") -> str:
    """Get all open positions across connected exchanges.

    Returns current positions with entry price, current price,
    unrealized PnL, leverage, and other details.

    Args:
        api_key: Your BASTION API key (bst_...) to access YOUR positions

    Returns:
        All open futures positions across Binance, Bybit, Bitunix, etc.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_get("/api/positions/all", auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_balance(api_key: str = "") -> str:
    """Get total portfolio balance across all connected exchanges.

    Returns aggregated balance with per-exchange breakdown.

    Args:
        api_key: Your BASTION API key (bst_...) to access YOUR balance

    Returns:
        Total balance, available margin, and exchange breakdown.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_get("/api/balance/total", auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_exchanges(api_key: str = "") -> str:
    """List all connected exchanges.

    Shows which exchanges are linked to BASTION with connection status.

    Args:
        api_key: Your BASTION API key (bst_...) to see YOUR exchanges

    Returns:
        Connected exchanges with status information.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_get("/api/exchange/list", auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_evaluate_all_positions(api_key: str = "") -> str:
    """Run AI risk evaluation on ALL open positions simultaneously.

    Sends every open position through the 72B model for evaluation.
    Returns a risk assessment for each position with recommended actions.

    Args:
        api_key: Your BASTION API key (bst_...) to evaluate YOUR positions

    Returns:
        Risk evaluation results for every open position.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/risk/evaluate-all", auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_scan_signals(api_key: str = "") -> str:
    """Scan for trading signals across all supported pairs.

    Uses the AI engine to identify potential trade setups based on
    current market conditions, structure, and flow data.

    Args:
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        Active trading signals with confidence scores and reasoning.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/signals/scan", auth_headers=auth_headers)
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOLS — TRADING ENGINE
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_engine_status(api_key: str = "") -> str:
    """Get the autonomous trading engine status.

    Shows whether the engine is running, armed, and its current configuration
    including safety parameters and execution history.

    Args:
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        Engine state, armed status, configuration, and recent execution stats.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_get("/api/engine/status", auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_engine_history(limit: int = 20, api_key: str = "") -> str:
    """Get trading engine execution history.

    Shows recent actions taken by the autonomous engine — which positions
    it evaluated, what actions it took, and the reasoning.

    Args:
        limit: Number of history entries to return (default 20)
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        Recent engine execution events with timestamps and details.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_get("/api/engine/execution-history", {"limit": min(limit, 100)}, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOLS — ALERTS & KLINES
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_alerts(api_key: str = "") -> str:
    """Get active alerts and notifications.

    Returns risk alerts, liquidation warnings, whale movement alerts,
    and user-configured price alerts.

    Args:
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        Active alerts with severity, type, and details.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_get("/api/alerts", auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_klines(
    symbol: str = "BTC",
    interval: str = "1h",
    limit: int = 100,
) -> str:
    """Get candlestick (OHLCV) data for a cryptocurrency.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)
        interval: Candle interval — 1m, 5m, 15m, 1h, 4h, 1d (default 1h)
        limit: Number of candles to return (default 100, max 500)

    Returns:
        OHLCV candle data with timestamps.
    """
    result = await api_get(f"/api/klines/{symbol.upper()}", {
        "interval": interval,
        "limit": min(limit, 500),
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_options(symbol: str = "BTC") -> str:
    """Get options data for a cryptocurrency.

    Includes options open interest, put/call ratio, max pain, and
    notable options flow that may indicate institutional positioning.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH)

    Returns:
        Options data including OI, P/C ratio, max pain, and flow.
    """
    result = await api_get(f"/api/options/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_session_stats(api_key: str = "") -> str:
    """Get current trading session statistics.

    Shows performance metrics for the current trading session including
    win rate, PnL, number of trades, and risk metrics.

    Args:
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        Session statistics with performance breakdown.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_get("/api/session/stats", auth_headers=auth_headers)
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TRADING TOOLS (require 'trade' scope)
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_emergency_exit(api_key: str) -> str:
    """EMERGENCY: Close ALL open positions across all exchanges immediately.

    This is a destructive action — it market-closes every open position.
    Use only in emergencies (flash crash, black swan, risk limit breached).

    Args:
        api_key: Your BASTION API key (bst_...) with 'trade' scope — REQUIRED

    Returns:
        Number of positions closed and any errors.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "trade")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/actions/emergency-exit", {}, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_partial_close(
    symbol: str,
    exit_pct: int = 50,
    api_key: str = "",
) -> str:
    """Close a percentage of a specific position.

    Use this for scaling out of positions — e.g. take 50% off at TP1,
    then let the rest run with a trailing stop.

    Args:
        symbol: Trading pair to close (e.g. BTC, BTCUSDT)
        exit_pct: Percentage to close (1-100, default 50)
        api_key: Your BASTION API key (bst_...) with 'trade' scope — REQUIRED

    Returns:
        Confirmation with closed quantity and remaining position.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "trade")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/actions/partial-close", {
        "symbol": symbol.upper(),
        "exit_pct": max(1, min(100, exit_pct)),
    }, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_set_take_profit(
    symbol: str,
    tp_price: float,
    exit_pct: int = 0,
    api_key: str = "",
) -> str:
    """Set or update take-profit for a position.

    Optionally set a partial TP (e.g. close 30% at $70k).

    Args:
        symbol: Trading pair (e.g. BTC, BTCUSDT)
        tp_price: Take profit price target
        exit_pct: Percentage to close at TP (0 = full position, 1-100 = partial)
        api_key: Your BASTION API key (bst_...) with 'trade' scope — REQUIRED

    Returns:
        Confirmation with TP details.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "trade")
    if auth_error:
        return json.dumps({"error": auth_error})
    payload = {"symbol": symbol.upper(), "tp_price": tp_price}
    if exit_pct > 0:
        payload["exit_pct"] = max(1, min(100, exit_pct))
    result = await api_post("/api/actions/set-take-profit", payload, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_set_stop_loss(
    symbol: str,
    sl_price: float,
    api_key: str = "",
) -> str:
    """Set or update stop-loss for a position.

    Args:
        symbol: Trading pair (e.g. BTC, BTCUSDT)
        sl_price: Stop loss price
        api_key: Your BASTION API key (bst_...) with 'trade' scope — REQUIRED

    Returns:
        Confirmation with SL details.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "trade")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/actions/set-stop-loss", {
        "symbol": symbol.upper(),
        "sl_price": sl_price,
    }, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_move_to_breakeven(
    symbol: str = "",
    api_key: str = "",
) -> str:
    """Move stop loss to entry price (breakeven) for profitable positions.

    Only moves stops for positions that are currently in profit.
    If no symbol specified, moves all profitable positions to breakeven.

    Args:
        symbol: Specific trading pair (empty = all profitable positions)
        api_key: Your BASTION API key (bst_...) with 'trade' scope — REQUIRED

    Returns:
        Number of positions moved to breakeven.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "trade")
    if auth_error:
        return json.dumps({"error": auth_error})
    payload = {}
    if symbol:
        payload["symbol"] = symbol.upper()
    result = await api_post("/api/actions/move-to-breakeven", payload, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_flatten_winners(api_key: str = "") -> str:
    """Close all winning positions (positions currently in profit).

    Leaves losing positions untouched — useful for locking in gains
    when you expect a reversal.

    Args:
        api_key: Your BASTION API key (bst_...) with 'trade' scope — REQUIRED

    Returns:
        Number of winning positions closed and total profit locked.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "trade")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/actions/flatten-winners", {}, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# ENGINE TOOLS (require 'engine' scope)
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_engine_start(api_key: str = "") -> str:
    """Start the BASTION autonomous risk engine.

    The engine monitors your positions in real-time and evaluates them
    using the 72B AI model on a continuous loop. Does NOT auto-execute
    trades until you arm it.

    Args:
        api_key: Your BASTION API key (bst_...) with 'engine' scope — REQUIRED

    Returns:
        Engine start confirmation with status details.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "engine")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/engine/start", {}, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_engine_arm(
    confidence_threshold: float = 0.7,
    daily_loss_limit_usd: float = 5000.0,
    api_key: str = "",
) -> str:
    """Arm the engine for autonomous trade execution.

    CRITICAL: This enables the engine to execute trades on your behalf.
    Your exchange must be connected with WRITE-enabled API keys.

    Args:
        confidence_threshold: Minimum AI confidence to execute (0.0-1.0, default 0.7)
        daily_loss_limit_usd: Max daily loss before engine pauses (default $5000)
        api_key: Your BASTION API key (bst_...) with 'engine' scope — REQUIRED

    Returns:
        Arm confirmation with safety parameters.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "engine")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/engine/arm", {
        "auto_execute": True,
        "confidence_threshold": max(0.0, min(1.0, confidence_threshold)),
        "daily_loss_limit_usd": daily_loss_limit_usd,
    }, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_engine_disarm(api_key: str = "") -> str:
    """Disarm the engine — stop autonomous trade execution.

    The engine continues monitoring but will NOT execute trades.
    Positions remain open; you'll need to manage them manually.

    Args:
        api_key: Your BASTION API key (bst_...) with 'engine' scope — REQUIRED

    Returns:
        Disarm confirmation.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "engine")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/engine/disarm", {}, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# ADVANCED ANALYTICS TOOLS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_volatility_regime(symbol: str = "BTC") -> str:
    """Get the current volatility regime classification for a symbol.

    Classifies the market into HIGH_VOL, LOW_VOL, or NORMAL based on
    realized volatility, ATR, and Bollinger band width. Critical for
    position sizing and stop placement.

    Args:
        symbol: Crypto symbol (default BTC)

    Returns:
        Volatility regime with confidence, current vol metrics, and regime history.
    """
    result = await api_get(f"/api/volatility-regime/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_kelly_sizing() -> str:
    """Get Kelly Criterion optimal position sizing based on your historical performance.

    Calculates the mathematically optimal bet size using your win rate and
    average win/loss ratio. Returns full Kelly, half Kelly (recommended),
    and quarter Kelly sizing with risk-of-ruin analysis.

    Returns:
        Kelly percentages (full/half/quarter), win rate, R-ratio, and recommendation.
    """
    result = await api_get("/api/kelly")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_liquidations_by_exchange(symbol: str = "BTC") -> str:
    """Get liquidation data broken down by exchange.

    Shows which exchanges are seeing the most liquidations — useful for
    identifying where leveraged positions are concentrated and getting wiped.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Liquidations per exchange with long/short breakdown and dominant side.
    """
    result = await api_get(f"/api/liq-exchange/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_smart_money(symbol: str = "BTC") -> str:
    """Get smart money flow analysis for a cryptocurrency.

    Tracks institutional and smart money flow direction using order flow
    analysis, large trade clustering, and volume profile. Shows whether
    smart money is accumulating or distributing.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Smart money bias (BULLISH/BEARISH/NEUTRAL), score, institutional flow, and divergence signals.
    """
    result = await api_get(f"/api/smart-money/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_hyperliquid_whales(symbol: str = "BTC") -> str:
    """Get top whale positions on Hyperliquid DEX.

    Shows the top 20 largest positions on Hyperliquid with their entry price,
    leverage, and unrealized PnL. These whales often front-run major moves.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Top 20 whale positions with side, size, entry, leverage, PnL, and net bias.
    """
    result = await api_get(f"/api/hyperliquid-whales?symbol={symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_btc_dominance() -> str:
    """Get Bitcoin dominance, altseason score, and total crypto market cap.

    Tracks BTC's share of the total crypto market. Falling BTC dominance
    often signals altseason (capital rotating into alts). Rising dominance
    means flight to quality (BTC season).

    Returns:
        BTC/ETH/alt dominance percentages, total market cap, volume,
        and altseason classification (STRONG_ALTSEASON to BTC_SEASON).
    """
    result = await api_get("/api/btc-dominance")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_correlation_matrix(
    symbols: str = "BTC,ETH,SOL,AVAX,DOGE",
    period: str = "30d",
) -> str:
    """Get a real-time correlation matrix across crypto assets and macro indicators.

    Shows how different assets move together. High correlation (>0.85) means
    you're effectively doubling your exposure. Negative correlation means
    natural hedge. Essential for portfolio risk management.

    Args:
        symbols: Comma-separated symbols (crypto + macro). Crypto: BTC,ETH,SOL,AVAX,DOGE.
            Macro: DXY,SPX,GOLD,VIX. Max 10 symbols. (default: BTC,ETH,SOL,AVAX,DOGE)
        period: Lookback period — 7d, 14d, 30d, or 90d (default: 30d)

    Returns:
        Full correlation matrix, highest/lowest correlated pairs, and risk warnings
        for dangerously correlated positions.
    """
    result = await api_get("/api/correlation-matrix", {"symbols": symbols, "period": period})
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_confluence(symbol: str = "BTC") -> str:
    """Get multi-timeframe confluence analysis for a cryptocurrency.

    Checks if 15m, 1h, 4h, and 1D timeframes are aligned on direction.
    Analyzes trend, VWAP, momentum, and market structure on each timeframe.
    High confluence (3-4 timeframes aligned) = high conviction setup.

    Args:
        symbol: Crypto symbol (e.g. BTC, ETH, SOL)

    Returns:
        Per-timeframe bias, overall confluence score, alignment count,
        and recommendation (trade or wait).
    """
    result = await api_get(f"/api/confluence/{symbol.upper()}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_sector_rotation() -> str:
    """Track capital rotation across crypto sectors.

    Compares 7-day performance of L1s, L2s, DeFi, AI, Memes, and Gaming sectors.
    Identifies where money is flowing in and out — critical for catching
    sector rotations early (e.g. money leaving memes, flowing into AI tokens).

    Returns:
        Sector rankings by 7d performance, capital inflow/outflow detection,
        best/worst tokens per sector, and rotation signal.
    """
    result = await api_get("/api/sector-rotation")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_risk_parity(api_key: str = "") -> str:
    """Analyze portfolio risk parity across all open positions.

    Calculates concentration risk (HHI index), correlation-adjusted exposure,
    effective leverage, directional bias, and maximum drawdown estimate.
    Tells you if you're accidentally 10x exposed to the same trade.

    Args:
        api_key: Your BASTION API key (bst_...) for authenticated access

    Returns:
        Risk level, concentration analysis, drawdown estimate, directional exposure,
        correlation warnings, and actionable recommendations.
    """
    auth_headers, auth_error = await resolve_auth(api_key, "read")
    if auth_error:
        return json.dumps({"error": auth_error})
    result = await api_post("/api/risk-parity", {}, auth_headers=auth_headers)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_log_trade(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl_usd: float,
    size_usd: float = 1000.0,
    leverage: float = 1.0,
    ai_recommendation: str = "",
    ai_followed: bool = True,
    notes: str = "",
) -> str:
    """Log a completed trade to your performance journal.

    Builds your real trading track record over time. The journal feeds
    into Kelly Criterion calculations and win rate statistics, replacing
    simulated data with YOUR actual performance.

    Args:
        symbol: Trading pair (e.g. BTC, ETH, SOL)
        direction: LONG or SHORT
        entry_price: Price at entry
        exit_price: Price at exit
        pnl_usd: Realized PnL in USD (positive = profit, negative = loss)
        size_usd: Position size in USD (default 1000)
        leverage: Leverage used (default 1.0)
        ai_recommendation: What did BASTION AI recommend? (HOLD, EXIT_FULL, etc.)
        ai_followed: Did you follow the AI recommendation? (default true)
        notes: Any notes about the trade

    Returns:
        Confirmation with trade ID.
    """
    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if direction.upper() == "LONG" \
        else ((entry_price - exit_price) / entry_price * 100)
    result = await api_post("/api/trade-journal/log", {
        "symbol": symbol.upper(),
        "direction": direction.upper(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct * leverage,
        "size_usd": size_usd,
        "leverage": leverage,
        "ai_recommendation": ai_recommendation,
        "ai_followed": ai_followed,
        "notes": notes,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_trade_journal(symbol: str = "", last_n: int = 0) -> str:
    """Get your trade journal performance statistics.

    Shows real win rate, average R-ratio, expectancy per trade, Kelly sizing
    from actual performance data, win/loss streaks, AI accuracy, and
    per-symbol breakdown. Replaces simulated Kelly data with reality.

    Args:
        symbol: Filter by symbol (empty = all symbols)
        last_n: Only analyze last N trades (0 = all trades)

    Returns:
        Win rate, R-ratio, Kelly sizing (real), expectancy, streaks,
        AI accuracy, and per-symbol performance breakdown.
    """
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    if last_n > 0:
        params["last_n"] = last_n
    result = await api_get("/api/trade-journal/stats", params)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_backtest_strategy(
    symbol: str = "BTC",
    strategy: str = "funding_spike",
    direction: str = "SHORT",
    leverage: float = 1.0,
    lookback_days: int = 30,
    tp_pct: float = 2.0,
    sl_pct: float = 1.0,
) -> str:
    """Backtest a trading strategy against historical data.

    Test if a strategy would have been profitable over the past N days.
    Supports multiple strategy types with configurable TP/SL and leverage.

    Args:
        symbol: Crypto symbol (default BTC)
        strategy: Strategy type — 'funding_spike' (mean reversion on price deviations),
            'mean_reversion' (Bollinger band reversal), 'momentum' (breakout following),
            'volume_spike' (enter on volume anomalies)
        direction: Trade direction — LONG or SHORT (default SHORT)
        leverage: Leverage multiplier (default 1.0)
        lookback_days: How many days to backtest (default 30, max 20)
        tp_pct: Take profit percentage (default 2.0)
        sl_pct: Stop loss percentage (default 1.0)

    Returns:
        Total trades, win rate, total PnL, best/worst trades, and verdict
        (PROFITABLE, MARGINAL, or UNPROFITABLE).
    """
    result = await api_post("/api/backtest-strategy", {
        "symbol": symbol.upper(),
        "strategy": strategy,
        "direction": direction.upper(),
        "leverage": leverage,
        "lookback_days": min(lookback_days, 20),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# RESOURCES
# ═════════════════════════════════════════════════════════════════


@mcp.resource("bastion://status")
async def get_status() -> str:
    """BASTION system status — model version, GPU health, API latency."""
    result = await api_get("/api/status")

    status = {
        "model_version": config.MODEL_VERSION,
        "model_base": config.MODEL_BASE,
        "gpu_cluster": config.MODEL_GPU,
        "api_status": result.get("status", "unknown"),
        "mcp_server_version": config.MCP_SERVER_VERSION,
        "services": result.get("services", {}),
    }
    return json.dumps(status, indent=2)


@mcp.resource("bastion://supported-symbols")
async def get_symbols() -> str:
    """List of cryptocurrency symbols supported by BASTION."""
    return json.dumps({
        "symbols": config.SUPPORTED_SYMBOLS,
        "count": len(config.SUPPORTED_SYMBOLS),
        "note": "Primary pairs (BTC, ETH, SOL) have the most data coverage. "
                "Other pairs may have limited structure analysis."
    }, indent=2)


@mcp.resource("bastion://model-info")
async def get_model_info() -> str:
    """BASTION AI model details — version, accuracy, training methodology."""
    return json.dumps({
        "version": config.MODEL_VERSION,
        "model": config.MODEL_BASE,
        "parameters": "72 billion",
        "accuracy": {
            "combined": config.MODEL_ACCURACY,
            "btc": "71.7%",
            "eth": "72.7%",
            "sol": "81.8%",
        },
        "actions": [
            "HOLD", "TP_PARTIAL", "EXIT_FULL", "EXIT_100%",
            "REDUCE_SIZE", "TRAIL_STOP", "MOVE_STOP_TO_BREAKEVEN"
        ],
        "signals_per_evaluation": "560+",
        "data_sources": [
            "Quantitative derivatives analytics (33 endpoints)",
            "Liquidations, open interest, funding, whale positions",
            "On-chain whale tracking (11 blockchains)",
            "Market structure service (VPVR, pivots, auto-support, trendlines)"
        ],
        "infrastructure": config.MODEL_GPU,
    }, indent=2)


@mcp.resource("bastion://tools")
async def get_tools() -> str:
    """Complete list of all 108 BASTION MCP tools with descriptions."""
    tools = {
        "core_ai": [
            "bastion_evaluate_risk — AI risk evaluation for a position",
            "bastion_chat — Ask anything about crypto markets",
            "bastion_evaluate_all_positions — Evaluate all positions at once",
            "bastion_scan_signals — Scan for trading signals",
        ],
        "market_data": [
            "bastion_get_price — Live crypto price",
            "bastion_get_market_data — Aggregated market intelligence",
            "bastion_get_klines — Candlestick OHLCV data",
            "bastion_get_volatility — Volatility metrics + regime",
            "bastion_get_volatility_regime — Volatility regime classification",
            "bastion_get_btc_dominance — BTC dominance + altseason score",
            "bastion_get_correlation_matrix — Cross-asset correlation matrix",
            "bastion_get_confluence — Multi-timeframe confluence scanner",
            "bastion_get_sector_rotation — Sector rotation tracker",
        ],
        "derivatives_orderflow": [
            "bastion_get_open_interest — OI across exchanges",
            "bastion_get_oi_changes — OI changes across all pairs",
            "bastion_get_cvd — Cumulative Volume Delta",
            "bastion_get_orderflow — Order flow analysis",
            "bastion_get_funding_rates — Cross-exchange funding rates",
            "bastion_get_funding_arb — Funding rate arbitrage",
            "bastion_get_liquidations — Liquidation events + clusters",
            "bastion_get_liquidations_by_exchange — Liquidations per exchange",
            "bastion_get_heatmap — Liquidation heatmap",
            "bastion_get_taker_ratio — Taker buy/sell ratio",
            "bastion_get_top_traders — Top trader positioning",
            "bastion_get_market_maker_magnet — MM gamma magnet levels",
            "bastion_get_options — Options OI, P/C ratio, max pain",
        ],
        "onchain_intelligence": [
            "bastion_get_whale_activity — Whale transactions (11 chains)",
            "bastion_get_exchange_flow — Exchange inflow/outflow",
            "bastion_get_onchain — On-chain metrics",
            "bastion_get_news — Aggregated crypto news",
            "bastion_get_smart_money — Smart money flow analysis",
            "bastion_get_hyperliquid_whales — Top Hyperliquid whale positions",
        ],
        "macro_sentiment": [
            "bastion_get_fear_greed — Fear & Greed Index",
            "bastion_get_macro_signals — DXY, yields, equities, gold",
            "bastion_get_etf_flows — BTC/ETH ETF flows",
            "bastion_get_stablecoin_markets — Stablecoin supply + flows",
            "bastion_get_economic_data — FRED economic data",
            "bastion_get_polymarket — Prediction market data",
        ],
        "research": [
            "bastion_generate_report — MCF Labs research report",
            "bastion_get_reports — List existing reports",
            "bastion_calculate_position — Position sizing + Monte Carlo",
            "bastion_get_kelly_sizing — Kelly Criterion optimal sizing",
            "bastion_log_trade — Log trade to performance journal",
            "bastion_get_trade_journal — Journal stats + real Kelly",
            "bastion_backtest_strategy — Backtest strategies on-demand",
            "bastion_get_risk_parity — Portfolio risk parity analysis",
        ],
        "portfolio": [
            "bastion_get_positions — All open positions",
            "bastion_get_balance — Portfolio balance",
            "bastion_get_exchanges — Connected exchanges",
            "bastion_engine_status — Engine status",
            "bastion_engine_history — Engine execution history",
            "bastion_get_alerts — Active alerts",
            "bastion_get_session_stats — Session statistics",
        ],
        "trading_actions": [
            "bastion_emergency_exit — Close ALL positions",
            "bastion_partial_close — Close % of a position",
            "bastion_set_take_profit — Set/update TP",
            "bastion_set_stop_loss — Set/update SL",
            "bastion_move_to_breakeven — Move stops to entry",
            "bastion_flatten_winners — Close all winning positions",
        ],
        "engine_control": [
            "bastion_engine_start — Start risk engine",
            "bastion_engine_arm — Arm for auto-execution",
            "bastion_engine_disarm — Disarm engine",
        ],
        "war_room": [
            "bastion_war_room_post — Post signal to War Room",
            "bastion_war_room_read — Read War Room feed",
            "bastion_war_room_consensus — Get consensus",
            "bastion_war_room_vote — Cast weighted vote (accuracy-based)",
            "bastion_war_room_consensus_weighted — Get weighted consensus",
        ],
        "advanced": [
            "bastion_risk_replay — Historical position time-travel analysis",
            "bastion_strategy_builder — Natural language → backtest pipeline",
            "bastion_risk_card — Interactive risk visualization widget",
            "bastion_get_leaderboard — Model performance leaderboard",
            "bastion_log_prediction — Log prediction for tracking",
            "bastion_subscribe_alert — Subscribe to price/condition alerts",
            "bastion_check_alerts — Check active & triggered alerts",
            "bastion_cancel_alert — Cancel an alert subscription",
            "bastion_create_risk_card — Shareable risk score card",
            "bastion_get_performance — Portfolio performance analytics",
            "bastion_record_equity — Record equity snapshot",
            "bastion_add_webhook — Register notification webhook",
            "bastion_list_webhooks — List all webhooks",
            "bastion_send_notification — Push notification to webhooks",
            "bastion_get_agent_analytics — Agent usage analytics",
            "bastion_format_risk — Beautiful terminal output formatter",
        ],
        "intelligence": [
            "bastion_quick_intel — Instant intelligence snapshot (~200ms)",
            "bastion_deep_analysis — Multi-step reasoning chain (gathers 15 sources in one call)",
            "bastion_get_regime_tools — Discover market-regime adaptive tools",
            "bastion_execute_regime_tool — Execute regime-specific analysis tools",
            "bastion_live_feed — Real-time aggregated market feed snapshot",
            "bastion_war_room_vote — Cast weighted vote in multi-agent consensus",
            "bastion_war_room_consensus_weighted — Get accuracy-weighted consensus",
            "bastion_get_server_card — MCP Server Card for agent-to-agent discovery",
            "bastion_risk_confirm — Interactive risk confirmation with warnings",
        ],
        "challenge_cards": [
            "bastion_create_challenge — Create a 'Prove Me Wrong' prediction",
            "bastion_counter_challenge — Take the opposite side of a challenge",
            "bastion_get_challenges — List active challenges",
            "bastion_score_challenge — Auto-score a challenge",
        ],
        "persistent_memory": [
            "bastion_memory_store — Store episodic/semantic/procedural memory",
            "bastion_memory_recall — Search and recall memories",
            "bastion_memory_profile — Get learned trader profile",
        ],
        "heat_index": [
            "bastion_heat_index — Composite attention score for a symbol",
            "bastion_heat_scan — Scan all pairs for hottest activity",
        ],
        "copy_analysis": [
            "bastion_save_workflow — Save a reusable tool chain",
            "bastion_get_workflows — Browse public workflows",
            "bastion_run_workflow — Execute a saved workflow",
        ],
        "audit_compliance": [
            "bastion_audit_log — Log hash-chained audit entry",
            "bastion_audit_trail — Query immutable audit trail",
            "bastion_audit_verify — Verify chain integrity",
            "bastion_decision_provenance — Record decision reasoning chain",
            "bastion_get_provenance — Query decision provenance",
            "bastion_tool_safety — Get safety annotations for all tools",
        ],
    }
    return json.dumps({"tools": tools, "total": 108}, indent=2)


@mcp.resource("bastion://exchanges")
async def get_exchanges() -> str:
    """List of supported exchanges for BASTION integration."""
    return json.dumps({
        "supported_exchanges": [
            {"name": "Binance", "type": "CEX", "features": ["futures", "spot", "read", "write"]},
            {"name": "Bybit", "type": "CEX", "features": ["futures", "spot", "read", "write"]},
            {"name": "OKX", "type": "CEX", "features": ["futures", "spot", "read", "write"]},
            {"name": "Bitunix", "type": "CEX", "features": ["futures", "read", "write"]},
            {"name": "Hyperliquid", "type": "DEX", "features": ["futures", "read"]},
            {"name": "Deribit", "type": "CEX", "features": ["options", "futures", "read"]},
            {"name": "BloFin", "type": "CEX", "features": ["futures", "read", "write"]},
        ],
        "note": "Connect exchanges in the BASTION dashboard. Read-only for monitoring, "
                "write-enabled for autonomous engine execution."
    }, indent=2)


@mcp.resource("bastion://live/{symbol}")
async def get_live_feed(symbol: str = "BTC") -> str:
    """Live market feed for a symbol — price, funding, volume, fear/greed in one snapshot.

    Subscribe to this resource for real-time market monitoring. Returns all key
    metrics aggregated into a single response. Poll every 30-60 seconds.
    """
    result = await api_get(f"/api/live-feed/{symbol}")
    return json.dumps(result, indent=2)


@mcp.resource("bastion://regime")
async def get_regime() -> str:
    """Current market regime and active adaptive tools.

    Shows which specialized tools are currently activated based on market conditions.
    Tools appear during: extreme fear, extreme greed, high volatility, low volatility,
    and extreme funding rate conditions.
    """
    result = await api_get("/api/regime/tools")
    return json.dumps(result, indent=2)


@mcp.resource("bastion://capabilities")
async def get_capabilities() -> str:
    """Complete BASTION platform capabilities and data sources."""
    return json.dumps({
        "ai_engine": {
            "model": "72B parameter fine-tuned Qwen",
            "gpu_cluster": "4x RTX 5090 (128GB VRAM)",
            "accuracy": "75.4% combined (BTC 71.7%, ETH 72.7%, SOL 81.8%)",
            "signals_analyzed": "560+ per evaluation",
        },
        "data_sources": {
            "helsinki_vm": "33 quantitative endpoints (derivatives, order flow, volatility)",
            "coinglass": "13 premium endpoints (OI, liquidations, whale positions)",
            "whale_alert": "11 blockchain whale tracking",
            "market_structure": "VPVR, pivot detection, auto-support, trendlines",
            "macro": "Yahoo Finance, FRED, CoinGecko, Polymarket",
        },
        "mcp_tools": 108,
        "resources": 8,
        "prompts": 10,
        "features": {
            "deep_analysis": "Server-side multi-step reasoning — gathers 15 sources in one call",
            "dynamic_tools": "Market-regime adaptive tools that appear/disappear based on conditions",
            "live_feed": "Aggregated real-time market feed for resource subscriptions",
            "war_room_consensus": "Multi-agent weighted voting with accuracy-based weighting",
            "server_discovery": "MCP Server Card at /.well-known/mcp.json for agent-to-agent discovery",
            "risk_elicitation": "Interactive risk confirmation with structured warnings and questions",
        },
        "trading_actions": ["HOLD", "EXIT_FULL", "TP_PARTIAL", "EXIT_100%", "REDUCE_SIZE", "TRAIL_STOP"],
        "supported_symbols": config.SUPPORTED_SYMBOLS,
    }, indent=2)


# ═════════════════════════════════════════════════════════════════
# WAR ROOM — Multi-Agent Intelligence Hub
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_war_room_post(
    content: str,
    api_key: str,
    type: str = "signal",
    agent_name: str = "",
    symbol: str = "",
    direction: str = "",
    tools_used: str = "",
) -> str:
    """Post a signal, alert, thesis, or counter-thesis to the BASTION War Room.

    The War Room is a shared intelligence feed where Claude agents post market observations
    backed by BASTION data. Other agents (and their users) can see your signals in real-time.

    Args:
        content: Your signal/observation (max 2000 chars). Should cite BASTION data.
        api_key: Your bst_ API key (MCP users only — required for posting).
        type: Message type — 'signal', 'alert', 'thesis', or 'counter'.
        agent_name: Your agent's display name (e.g., 'scout_alpha', 'risk_guard').
        symbol: Primary asset (e.g., 'BTC', 'ETH', 'SOL'). Helps consensus engine.
        direction: Your directional bias — 'BULLISH' or 'BEARISH'. Powers consensus.
        tools_used: Which BASTION tools informed this signal (e.g., 'whale_activity + funding_rates').

    Returns:
        Confirmation with message_id, or error if auth fails.
    """
    client = await get_client()
    try:
        resp = await client.post("/api/warroom/post", json={
            "api_key": api_key,
            "content": content,
            "type": type,
            "agent_name": agent_name,
            "symbol": symbol,
            "direction": direction,
            "tools_used": tools_used,
        })
        data = resp.json()
        if resp.status_code != 200:
            return json.dumps({"error": data.get("error", "Post failed")})
        return json.dumps({"ok": True, "message_id": data.get("message_id"), "message": f"Signal posted to War Room. Type: {type.upper()}, Symbol: {symbol or 'N/A'}"})
    except Exception as e:
        return json.dumps({"error": f"War Room post failed: {str(e)}"})


@mcp.tool()
async def bastion_war_room_read(
    limit: int = 20,
    symbol: str = "",
    type: str = "",
) -> str:
    """Read the latest signals from the BASTION War Room.

    The War Room is a shared intelligence feed where multiple Claude agents post
    market observations, risk alerts, and trade theses — all backed by BASTION data.
    Use this to see what other agents are seeing before making your own decisions.

    No API key needed to read — the feed is visible to all MCP-connected agents.

    Args:
        limit: Number of recent messages to fetch (default 20, max 50).
        symbol: Filter by symbol (e.g., 'BTC'). Leave empty for all.
        type: Filter by type ('signal', 'alert', 'thesis', 'consensus', 'counter'). Leave empty for all.

    Returns:
        JSON array of recent War Room messages with agent names, signals, and timestamps.
    """
    client = await get_client()
    params = {"limit": min(limit, 50)}
    if symbol:
        params["symbol"] = symbol.upper()
    if type:
        params["msg_type"] = type.lower()
    try:
        resp = await client.get("/api/warroom/feed", params=params)
        data = resp.json()
        messages = data.get("messages", [])
        if not messages:
            return json.dumps({"messages": [], "note": "War Room is quiet. No recent signals. Be the first to post!"})
        # Format for readability
        formatted = []
        for m in messages:
            formatted.append({
                "type": m.get("type", "signal").upper(),
                "agent": m.get("agent", "unknown"),
                "symbol": m.get("symbol", ""),
                "direction": m.get("direction", ""),
                "content": m.get("content", ""),
                "tools": m.get("tools", ""),
                "id": m.get("id", ""),
            })
        return json.dumps({"messages": formatted, "total_in_room": data.get("total", 0)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to read War Room: {str(e)}"})


@mcp.tool()
async def bastion_war_room_consensus(
    symbol: str = "",
) -> str:
    """Get the current War Room consensus for a symbol or all tracked assets.

    The consensus engine aggregates directional signals from all active agents
    over the last 30 minutes and calculates whether the room leans BULLISH,
    BEARISH, or NEUTRAL — with a confidence rating.

    Args:
        symbol: Asset to check consensus for (e.g., 'BTC'). Leave empty for all symbols.

    Returns:
        Consensus data: direction, bullish/bearish count, confidence level.
    """
    client = await get_client()
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    try:
        resp = await client.get("/api/warroom/consensus", params=params)
        data = resp.json()
        if symbol:
            return json.dumps({
                "symbol": data.get("symbol", symbol.upper()),
                "consensus": data.get("direction", "NEUTRAL"),
                "bullish_signals": data.get("bullish", 0),
                "bearish_signals": data.get("bearish", 0),
                "total_signals": data.get("total", 0),
                "confidence": data.get("confidence", "NONE"),
                "note": "Based on agent signals from the last 30 minutes."
            }, indent=2)
        # All symbols
        result = {}
        for sym, info in data.items():
            result[sym] = {
                "consensus": info.get("direction", "NEUTRAL"),
                "bullish": info.get("bullish", 0),
                "bearish": info.get("bearish", 0),
                "total": info.get("total", 0),
                "confidence": info.get("confidence", "NONE"),
            }
        return json.dumps({"consensus": result, "note": "Based on agent signals from the last 30 minutes."}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get consensus: {str(e)}"})


# ═════════════════════════════════════════════════════════════════
# RISK REPLAY & STRATEGY BUILDER
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_risk_replay(
    symbol: str = "BTC",
    direction: str = "LONG",
    entry_price: float = 0,
    timestamp: str = "",
    lookback_hours: int = 4,
) -> str:
    """Replay a historical position — reconstruct market state at a past time and analyze what would have happened.

    Time-travel risk analysis: go back in time and see the market conditions,
    price action, and PnL outcome of a position that was (or could have been) taken.

    Args:
        symbol: Trading pair (e.g., 'BTC', 'ETH', 'SOL')
        direction: LONG or SHORT
        entry_price: Entry price of the position (0 = use historical price)
        timestamp: ISO timestamp to replay from (e.g., '2025-02-20T14:00:00Z'). Empty = lookback_hours ago.
        lookback_hours: Hours to look back if no timestamp given (default 4)

    Returns:
        Historical market snapshot, price evolution, and hindsight PnL analysis.
    """
    result = await api_get(f"/api/risk-replay/{symbol}", {
        "direction": direction,
        "entry_price": entry_price,
        "timestamp": timestamp,
        "lookback_hours": lookback_hours,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_strategy_builder(
    description: str,
    symbol: str = "BTC",
    lookback_days: int = 30,
) -> str:
    """Build and backtest a trading strategy from natural language description.

    Describe your strategy in plain English and BASTION will parse it into rules
    and run a backtest against historical data. Supports conditions like
    funding rates, RSI, volume spikes, OI changes, mean reversion, momentum, and pullbacks.

    Examples:
        - "When funding is above 0.1%, go SHORT with 2x leverage, TP 3%, SL 2%"
        - "Buy the dip when RSI is oversold, 5x leverage, TP 5%, SL 1.5%"
        - "Short when volume spikes and OI is rising"
        - "Mean reversion on BTC, long oversold bounces"

    Args:
        description: Natural language description of your strategy
        symbol: Trading pair to backtest on (default BTC)
        lookback_days: How many days of history to test (default 30)

    Returns:
        Parsed rules, backtest results (win rate, PnL, trades), and interpretation.
    """
    result = await api_post("/api/strategy-builder", {
        "description": description,
        "symbol": symbol,
        "lookback_days": lookback_days,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# ALERT SUBSCRIPTIONS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_subscribe_alert(
    symbol: str = "BTC",
    condition: str = "price_above",
    threshold: float = 0,
    notes: str = "",
) -> str:
    """Subscribe to a price or market condition alert.

    Set alerts that trigger when conditions are met. Check triggered alerts
    with bastion_check_alerts.

    Conditions:
        - price_above: Alert when price goes above threshold
        - price_below: Alert when price drops below threshold
        - funding_spike: Alert on extreme funding rate
        - volume_spike: Alert on unusual volume

    Args:
        symbol: Trading pair (e.g., 'BTC', 'ETH', 'SOL')
        condition: Alert condition type (price_above, price_below, funding_spike, volume_spike)
        threshold: Price or value threshold to trigger the alert
        notes: Optional notes about why you set this alert

    Returns:
        Alert confirmation with ID for tracking.
    """
    result = await api_post("/api/alerts/subscribe", {
        "symbol": symbol,
        "condition": condition,
        "threshold": threshold,
        "notes": notes,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_check_alerts() -> str:
    """Check all active alerts and see which ones have triggered.

    Returns active (untriggered) alerts, recently triggered alerts,
    and overall alert status. Call this periodically to monitor your alerts.

    Returns:
        Active alerts, triggered alerts, and counts.
    """
    result = await api_get("/api/alerts/active")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_cancel_alert(alert_id: str) -> str:
    """Cancel an active alert by its ID.

    Args:
        alert_id: The alert ID returned when subscribing (e.g., 'alert_1234567890_0')

    Returns:
        Confirmation of cancellation.
    """
    client = await get_client()
    try:
        resp = await client.delete(f"/api/alerts/{alert_id}")
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to cancel alert: {str(e)}"})


# ═════════════════════════════════════════════════════════════════
# MODEL LEADERBOARD
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_leaderboard() -> str:
    """Get the BASTION model performance leaderboard.

    Shows accuracy stats — overall, per-symbol (BTC/ETH/SOL), and per-action
    (HOLD/EXIT/TP_PARTIAL). Includes both backtest baseline and live prediction
    tracking when available.

    Returns:
        Model accuracy, per-symbol breakdown, per-action breakdown, recent predictions, win streaks.
    """
    result = await api_get("/api/leaderboard")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_log_prediction(
    symbol: str = "BTC",
    direction: str = "LONG",
    action: str = "HOLD",
    confidence: float = 0,
    entry_price: float = 0,
    outcome: str = "pending",
    outcome_price: float = 0,
    pnl_pct: float = 0,
) -> str:
    """Log a model prediction and its outcome to the leaderboard.

    Track BASTION's live predictions and their outcomes for transparency
    and performance monitoring.

    Args:
        symbol: Trading pair
        direction: LONG or SHORT
        action: Model's recommended action (HOLD, EXIT_FULL, TP_PARTIAL, etc.)
        confidence: Model confidence score (0-1)
        entry_price: Position entry price
        outcome: 'correct', 'incorrect', or 'pending'
        outcome_price: Price when outcome was determined
        pnl_pct: PnL percentage if applicable

    Returns:
        Logged prediction confirmation.
    """
    result = await api_post("/api/leaderboard/log", {
        "symbol": symbol,
        "direction": direction,
        "action": action,
        "confidence": confidence,
        "entry_price": entry_price,
        "outcome": outcome,
        "outcome_price": outcome_price,
        "pnl_pct": pnl_pct,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# INTERACTIVE RISK VISUALIZATION
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_risk_card(
    symbol: str = "BTC",
    direction: str = "LONG",
    entry_price: float = 0,
    current_price: float = 0,
    stop_loss: float = 0,
    leverage: float = 1,
) -> str:
    """Generate an interactive risk visualization card for a position.

    Creates a visual risk card with BASTION branding showing entry/exit prices,
    PnL, risk level gauge, and position metrics. Returns a link to the
    interactive HTML widget.

    Args:
        symbol: Trading pair (e.g., 'BTC', 'ETH', 'SOL')
        direction: LONG or SHORT
        entry_price: Position entry price
        current_price: Current market price (0 = fetch live)
        stop_loss: Stop loss price (0 = no stop)
        leverage: Position leverage (default 1)

    Returns:
        Position summary with link to interactive risk visualization.
    """
    # Get current price if not provided
    if current_price == 0:
        price_data = await api_get(f"/api/price/{symbol}")
        current_price = float(price_data.get("price", 0)) if isinstance(price_data, dict) else 0

    if entry_price == 0:
        entry_price = current_price

    # Calculate metrics
    if direction.upper() == "LONG":
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    else:
        pnl_pct = ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0

    effective_pnl = pnl_pct * leverage
    risk_level = "LOW" if abs(effective_pnl) < 5 else "MEDIUM" if abs(effective_pnl) < 15 else "HIGH" if abs(effective_pnl) < 30 else "CRITICAL"

    params = f"symbol={symbol}&direction={direction}&entry_price={entry_price}&current_price={current_price}&stop_loss={stop_loss}&leverage={leverage}"

    return json.dumps({
        "position": {
            "symbol": symbol.upper(),
            "direction": direction.upper(),
            "entry_price": entry_price,
            "current_price": current_price,
            "stop_loss": stop_loss,
            "leverage": leverage,
        },
        "metrics": {
            "pnl_pct": round(pnl_pct, 2),
            "effective_pnl": round(effective_pnl, 2),
            "risk_level": risk_level,
        },
        "visualization": f"https://bastionfi.tech/api/widget/risk-card?{params}",
        "note": "Open the visualization URL to see an interactive risk card with BASTION branding."
    }, indent=2)


# ═════════════════════════════════════════════════════════════════
# SHAREABLE RISK CARDS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_create_risk_card(
    symbol: str = "BTC",
    direction: str = "LONG",
    entry_price: float = 0,
    current_price: float = 0,
    stop_loss: float = 0,
    leverage: float = 1,
    action: str = "HOLD",
    risk_score: int = 50,
    reasoning: str = "",
    confidence: float = 0.75,
) -> str:
    """Create a shareable risk score card with a unique URL.

    Generates a beautiful branded card (like Spotify Wrapped for trades) that can be
    shared on Twitter/X, Discord, and any platform with OG meta tag support. Each card
    gets a unique URL with full embed previews.

    Args:
        symbol: Trading pair (e.g., 'BTC', 'ETH', 'SOL')
        direction: LONG or SHORT
        entry_price: Position entry price
        current_price: Current market price
        stop_loss: Stop loss price (0 = no stop)
        leverage: Position leverage
        action: AI recommendation (HOLD, EXIT_FULL, TP_PARTIAL, etc.)
        risk_score: Risk score 0-100 (higher = more dangerous)
        reasoning: Brief reasoning for the recommendation
        confidence: Model confidence 0-1

    Returns:
        Shareable card URL, embed URL, and card data.
    """
    result = await api_post("/api/risk-card/create", {
        "symbol": symbol, "direction": direction,
        "entry_price": entry_price, "current_price": current_price,
        "stop_loss": stop_loss, "leverage": leverage,
        "action": action, "risk_score": risk_score,
        "reasoning": reasoning, "confidence": confidence,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# PERFORMANCE ANALYTICS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_performance(
    period: str = "7d",
) -> str:
    """Get portfolio performance analytics: equity curve, win rate, Sharpe ratio, max drawdown.

    Comprehensive performance metrics calculated from trade journal entries and
    equity snapshots. Includes per-symbol breakdown, profit factor, streak analysis,
    and risk-adjusted returns.

    Args:
        period: Time period — '1d', '7d', '30d', '90d', or 'all' (default '7d')

    Returns:
        Win rate, Sharpe ratio, max drawdown, equity curve, per-symbol stats, profit factor.
    """
    result = await api_get("/api/analytics/performance", {"period": period})
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_record_equity(
    equity_usd: float,
    open_positions: int = 0,
    daily_pnl: float = 0,
    win_count: int = 0,
    loss_count: int = 0,
    total_trades: int = 0,
) -> str:
    """Record an equity snapshot for performance tracking over time.

    Call this periodically to build an equity curve. Snapshots are used to
    calculate max drawdown, Sharpe ratio, and other time-series metrics.

    Args:
        equity_usd: Current portfolio equity in USD
        open_positions: Number of open positions
        daily_pnl: PnL for today in USD
        win_count: Total wins to date
        loss_count: Total losses to date
        total_trades: Total trades to date

    Returns:
        Snapshot confirmation.
    """
    result = await api_post("/api/analytics/snapshot", {
        "equity_usd": equity_usd, "open_positions": open_positions,
        "daily_pnl": daily_pnl, "win_count": win_count,
        "loss_count": loss_count, "total_trades": total_trades,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# WEBHOOK NOTIFICATIONS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_add_webhook(
    url: str,
    webhook_type: str = "custom",
    events: str = "risk_alert,price_alert,whale_alert",
) -> str:
    """Register a webhook for push notifications to Discord, Telegram, or any URL.

    When alerts trigger or risk conditions change, BASTION will POST to your
    webhook URL with the event details.

    Args:
        url: Webhook URL (Discord webhook URL, Telegram bot URL, or any HTTPS endpoint)
        webhook_type: Type — 'discord', 'telegram', or 'custom' (default 'custom')
        events: Comma-separated events to subscribe to: risk_alert, price_alert, whale_alert, trade_alert

    Returns:
        Webhook registration confirmation with ID.
    """
    event_list = [e.strip() for e in events.split(",") if e.strip()]
    result = await api_post("/api/notifications/webhook", {
        "type": webhook_type, "url": url, "events": event_list,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_list_webhooks() -> str:
    """List all registered notification webhooks.

    Returns:
        All webhooks with their status, event subscriptions, and fire counts.
    """
    result = await api_get("/api/notifications/webhooks")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_send_notification(
    event: str = "risk_alert",
    message: str = "",
) -> str:
    """Send a notification to all matching webhooks.

    Triggers a push notification to all webhooks subscribed to the given event type.
    Useful for sending custom alerts from your agent to Discord/Telegram.

    Args:
        event: Event type — risk_alert, price_alert, whale_alert, trade_alert
        message: Human-readable alert message

    Returns:
        Number of webhooks notified.
    """
    result = await api_post("/api/notifications/send", {
        "event": event, "message": message, "payload": {},
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# AGENT ANALYTICS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_agent_analytics() -> str:
    """Get agent performance analytics — tool usage, latency, error rates.

    Shows which MCP tools are called most often, average response times,
    error rates, P95 latency, and connected agent counts. Like Stripe's
    dashboard but for MCP usage.

    Returns:
        Tool usage stats, latency metrics, error rates, connected agents.
    """
    result = await api_get("/api/analytics/agents")
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TERMINAL OUTPUT FORMATTER
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_format_risk(
    symbol: str = "BTC",
    direction: str = "LONG",
    action: str = "HOLD",
    risk_score: int = 50,
    pnl_pct: float = 0,
    leverage: float = 1,
    entry_price: float = 0,
    current_price: float = 0,
    reasoning: str = "",
    confidence: float = 0.75,
) -> str:
    """Format a risk evaluation as beautiful terminal-style output.

    Creates a clean, screenshot-worthy ASCII box with risk meter, PnL indicator,
    and action recommendation. Perfect for sharing in terminal screenshots.

    Args:
        symbol: Trading pair
        direction: LONG or SHORT
        action: AI recommendation
        risk_score: Risk score 0-100
        pnl_pct: PnL percentage
        leverage: Position leverage
        entry_price: Entry price
        current_price: Current price
        reasoning: Brief reasoning
        confidence: Model confidence 0-1

    Returns:
        Beautifully formatted terminal output string.
    """
    result = await api_post("/api/format/risk", {
        "symbol": symbol, "direction": direction, "action": action,
        "risk_score": risk_score, "pnl_pct": pnl_pct, "leverage": leverage,
        "entry_price": entry_price, "current_price": current_price,
        "reasoning": reasoning, "confidence": confidence,
    })
    if isinstance(result, dict) and "formatted" in result:
        return result["formatted"]
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER — Protocol-Level Innovation
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_quick_intel(
    symbol: str = "BTC",
) -> str:
    """Instant intelligence snapshot — price, regime, funding, fear/greed in one call.

    The fastest way to get market context. Returns a compact intelligence packet
    with current price, market regime, funding signal, and sentiment in ~200ms.
    Use this before any trading decision for quick context.

    Args:
        symbol: Crypto symbol (BTC, ETH, SOL, etc.)

    Returns:
        Compact intel: price, change, regime, funding signal, fear/greed, and risk level.
    """
    result = await api_get(f"/api/live-feed/{symbol}")
    if isinstance(result, dict) and not result.get("error"):
        fg = result.get("fear_greed", 50)
        regime = "NEUTRAL"
        if isinstance(fg, (int, float)):
            if fg < 20: regime = "CAPITULATION"
            elif fg < 35: regime = "FEAR"
            elif fg > 80: regime = "EUPHORIA"
            elif fg > 65: regime = "GREED"
        funding = result.get("funding_rate", 0)
        funding_signal = "EXTREME" if abs(funding) > 0.03 else "ELEVATED" if abs(funding) > 0.01 else "NORMAL"
        return json.dumps({
            "symbol": symbol,
            "price": result.get("price", 0),
            "change_24h": result.get("change_24h", 0),
            "regime": regime,
            "fear_greed": fg,
            "sentiment": result.get("sentiment", "Neutral"),
            "funding_rate": round(funding, 6),
            "funding_signal": funding_signal,
            "risk_level": "HIGH" if regime in ("CAPITULATION", "EUPHORIA") else "ELEVATED" if regime in ("FEAR", "GREED") else "NORMAL",
        }, indent=2)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_deep_analysis(
    symbol: str = "BTC",
    focus: str = "full",
    timeframe: str = "4h",
) -> str:
    """Deep multi-step analysis — gathers 5-15 data sources in parallel and synthesizes into one intelligence brief.

    This is BASTION's most powerful tool. Instead of calling 10 tools separately,
    call this once — BASTION orchestrates all data gathering internally and returns
    a unified analysis. Server-side reasoning chain (MCP Sampling pattern).

    Args:
        symbol: Crypto symbol (BTC, ETH, SOL, etc.)
        focus: Analysis focus — "full" (everything), "risk" (risk-focused), "flow" (order flow + derivatives), "macro" (macro environment), "structure" (market structure)
        timeframe: Primary timeframe for analysis (1h, 4h, 1d)

    Returns:
        Comprehensive intelligence brief with regime assessment, risk signals, and synthesized data from multiple sources.
    """
    result = await api_post("/api/deep-analysis", {
        "symbol": symbol, "focus": focus, "timeframe": timeframe,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# DYNAMIC TOOL DISCOVERY — Market-Regime Adaptive Tools
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_regime_tools() -> str:
    """Discover which specialized tools are currently active based on market conditions.

    Tools appear and disappear based on real-time market regime:
    - Extreme Fear → bastion_capitulation_scanner activates
    - Extreme Greed → bastion_euphoria_detector activates
    - High Volatility → bastion_crisis_mode activates
    - Low Volatility → bastion_range_scanner activates
    - Extreme Funding → bastion_funding_arb_scanner activates

    Call this first to see what's available, then use bastion_execute_regime_tool.

    Returns:
        List of currently active regime-adaptive tools with descriptions.
    """
    result = await api_get("/api/regime/tools")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_execute_regime_tool(
    tool: str = "bastion_market_pulse",
    symbol: str = "BTC",
) -> str:
    """Execute a regime-adaptive tool discovered via bastion_get_regime_tools.

    These are specialized tools that only activate during specific market conditions.
    Use bastion_get_regime_tools first to see what's available.

    Args:
        tool: Tool name from bastion_get_regime_tools (e.g., bastion_crisis_mode, bastion_capitulation_scanner, bastion_market_pulse)
        symbol: Crypto symbol to analyze

    Returns:
        Specialized analysis based on current market regime.
    """
    result = await api_post("/api/regime/execute", {
        "tool": tool, "symbol": symbol,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# LIVE MARKET FEED — Resource Subscription Data
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_live_feed(
    symbol: str = "BTC",
) -> str:
    """Get a real-time market feed snapshot — price, funding, volume, fear/greed in ONE call.

    Designed for high-frequency polling. Returns all key metrics aggregated into
    a single response. Use this instead of calling 5 separate tools when you need
    a quick market pulse.

    Subscribe pattern: Call this every 30-60 seconds for live monitoring.

    Args:
        symbol: Crypto symbol (BTC, ETH, SOL, etc.)

    Returns:
        Unified feed: price, 24h change, volume, funding rate, fear/greed, sentiment.
    """
    result = await api_get(f"/api/live-feed/{symbol}")
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# WAR ROOM UPGRADE — Weighted Consensus Voting
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_war_room_vote(
    symbol: str = "BTC",
    action: str = "HOLD",
    confidence: float = 0.7,
    reasoning: str = "",
    agent_id: str = "",
    historical_accuracy: float = 0.5,
) -> str:
    """Cast a weighted vote in the multi-agent War Room.

    Each agent's vote is weighted by their historical accuracy. Multiple agents
    analyze the same symbol independently, then the War Room synthesizes their
    disagreements into a consensus decision. Emergent intelligence.

    Args:
        symbol: Crypto symbol to vote on
        action: Your recommended action (HOLD, EXIT_FULL, TP_PARTIAL, REDUCE_SIZE, etc.)
        confidence: Your confidence in this recommendation (0.0 to 1.0)
        reasoning: Brief explanation of your analysis
        agent_id: Your unique agent identifier (auto-generated if empty)
        historical_accuracy: Your historical prediction accuracy from leaderboard (0.0 to 1.0)

    Returns:
        Vote confirmation with current vote tally.
    """
    result = await api_post("/api/war-room/vote", {
        "symbol": symbol, "action": action, "confidence": confidence,
        "reasoning": reasoning, "agent_id": agent_id or f"agent_{int(__import__('time').time()) % 10000}",
        "historical_accuracy": historical_accuracy,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_war_room_consensus_weighted(
    symbol: str = "BTC",
) -> str:
    """Get weighted consensus from the multi-agent War Room.

    Aggregates all agent votes with accuracy-based weighting. Shows which action
    the collective intelligence favors, how strong the consensus is, and which
    agents agree/disagree. True multi-agent decision-making.

    Args:
        symbol: Crypto symbol to check consensus for

    Returns:
        Consensus action, strength percentage, agreement level, and agent breakdown.
    """
    result = await api_get(f"/api/war-room/consensus/{symbol}")
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# MCP SERVER CARD — Agent-to-Agent Discovery
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_server_card() -> str:
    """Get BASTION's MCP Server Card for agent-to-agent discovery.

    Returns the full server card with capabilities, tools count, auth info,
    model details, and endpoint URLs. Other MCP agents can use this to
    discover and connect to BASTION automatically.

    Implements the /.well-known/mcp.json specification for MCP server discovery.

    Returns:
        Complete MCP Server Card with capabilities, auth, endpoints, and model info.
    """
    result = await api_get("/.well-known/mcp.json")
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# ELICITATION — Interactive Risk Confirmation
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_risk_confirm(
    symbol: str = "BTC",
    direction: str = "LONG",
    leverage: float = 1,
    entry_price: float = 0,
    stop_loss: float = 0,
    position_size_usd: float = 1000,
    portfolio_pct: float = 0,
) -> str:
    """Interactive risk confirmation — analyzes position and surfaces warnings that need acknowledgment.

    Before entering a trade, run this to get a structured risk assessment with
    specific warnings and confirmation questions. Implements the MCP Elicitation
    pattern: BASTION identifies risks and asks for human confirmation.

    Args:
        symbol: Crypto symbol
        direction: LONG or SHORT
        leverage: Position leverage
        entry_price: Planned entry price
        stop_loss: Stop loss price (0 = no stop)
        position_size_usd: Position size in USD
        portfolio_pct: What percentage of total portfolio this trade represents

    Returns:
        Risk level, warnings, confirmation questions, and recommendation.
    """
    result = await api_post("/api/risk/confirm", {
        "symbol": symbol, "direction": direction, "leverage": leverage,
        "entry_price": entry_price, "stop_loss": stop_loss,
        "position_size_usd": position_size_usd, "portfolio_pct": portfolio_pct,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# "PROVE ME WRONG" CHALLENGE CARDS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_create_challenge(
    symbol: str = "BTC",
    prediction: str = "BULLISH",
    timeframe_hours: int = 24,
    target_pct: float = 0,
    reasoning: str = "",
    agent_id: str = "",
) -> str:
    """Create a public "Prove Me Wrong" prediction challenge with a timestamped call.

    Publish your prediction as a shareable challenge card. Other agents can take
    the opposite side. After the timeframe expires, BASTION auto-scores it.
    Creates a shareable URL with OG meta tags for Twitter/Discord.

    Args:
        symbol: Crypto symbol to predict on
        prediction: BULLISH or BEARISH
        timeframe_hours: How many hours until the challenge expires (default 24)
        target_pct: Target percentage move (0 = just direction)
        reasoning: Your analysis/reasoning for this prediction
        agent_id: Your unique agent identifier

    Returns:
        Challenge card with unique shareable URL.
    """
    result = await api_post("/api/challenges/create", {
        "symbol": symbol, "prediction": prediction,
        "timeframe_hours": timeframe_hours, "target_pct": target_pct,
        "reasoning": reasoning, "agent_id": agent_id,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_counter_challenge(
    challenge_id: str,
    reasoning: str = "",
    agent_id: str = "",
) -> str:
    """Take the opposite side of a "Prove Me Wrong" challenge.

    Args:
        challenge_id: The challenge ID to counter
        reasoning: Your counter-argument
        agent_id: Your agent identifier

    Returns:
        Counter confirmation with current tally.
    """
    result = await api_post("/api/challenges/counter", {
        "challenge_id": challenge_id, "reasoning": reasoning, "agent_id": agent_id,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_challenges(
    status: str = "active",
    symbol: str = "",
) -> str:
    """List active prediction challenges — see what other agents are calling.

    Args:
        status: Filter by status — "active", "scored", "expired"
        symbol: Filter by symbol (empty = all symbols)

    Returns:
        List of challenges with predictions, counters, and scores.
    """
    params = {"status": status}
    if symbol:
        params["symbol"] = symbol
    result = await api_get("/api/challenges", params=params)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_score_challenge(
    challenge_id: str,
) -> str:
    """Score a challenge by checking the current price against the prediction.

    Auto-determines if the prediction was correct based on price movement
    since the challenge was created. Awards points to the winner.

    Args:
        challenge_id: The challenge ID to score

    Returns:
        Scoring result with price change, correctness verdict, and target hit status.
    """
    result = await api_post(f"/api/challenges/score/{challenge_id}", {})
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# PERSISTENT TRADER MEMORY
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_memory_store(
    content: str,
    memory_type: str = "episodic",
    key: str = "",
    user_id: str = "_default",
    symbol: str = "",
    action: str = "",
    outcome: str = "",
    workflow_name: str = "",
    tools_sequence: str = "",
) -> str:
    """Store a persistent memory — the agent remembers across sessions.

    Three memory types:
    - episodic: "What happened" — timestamped trading events and their outcomes
    - semantic: "What I know about you" — trader preferences, risk tolerance, patterns
    - procedural: "Learned workflows" — tool sequences the trader commonly uses

    Args:
        content: The memory content to store
        memory_type: "episodic", "semantic", or "procedural"
        key: Key name for semantic memories (e.g., "risk_tolerance", "preferred_leverage")
        user_id: User identifier for memory scoping
        symbol: Symbol context (for episodic memories)
        action: Action taken (for episodic memories)
        outcome: Whether it was "correct" or "incorrect" (for episodic memories)
        workflow_name: Name of the workflow (for procedural memories)
        tools_sequence: Comma-separated list of tools in the workflow (for procedural memories)

    Returns:
        Storage confirmation with memory counts.
    """
    metadata = {}
    if symbol: metadata["symbol"] = symbol
    if action: metadata["action"] = action
    if outcome: metadata["outcome"] = outcome
    if workflow_name: metadata["workflow_name"] = workflow_name
    if tools_sequence: metadata["tools_sequence"] = [t.strip() for t in tools_sequence.split(",") if t.strip()]

    result = await api_post("/api/memory/store", {
        "user_id": user_id, "type": memory_type,
        "content": content, "key": key, "metadata": metadata,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_memory_recall(
    query: str = "",
    memory_type: str = "all",
    user_id: str = "_default",
    limit: int = 10,
) -> str:
    """Recall memories — search what the agent remembers about past trades and preferences.

    Search across episodic events, semantic facts about the trader, and learned workflows.
    Use this at the start of a conversation to recall context from previous sessions.

    Args:
        query: Search query (searches content, symbols, keys). Empty = return recent memories.
        memory_type: "all", "episodic", "semantic", or "procedural"
        user_id: User identifier
        limit: Max results per category

    Returns:
        Matching memories across all categories.
    """
    result = await api_get("/api/memory/recall", params={
        "user_id": user_id, "query": query,
        "memory_type": memory_type, "limit": str(limit),
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_memory_profile(
    user_id: str = "_default",
) -> str:
    """Get the agent's learned profile of a trader — preferences, patterns, accuracy.

    Returns a summary of everything the agent has learned: most traded symbols,
    action distribution, prediction accuracy, semantic facts (risk tolerance,
    preferred leverage, override patterns), and learned workflows.

    Args:
        user_id: User identifier

    Returns:
        Complete trader profile derived from memory.
    """
    result = await api_get(f"/api/memory/profile/{user_id}")
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# BASTION HEAT INDEX
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_heat_index(
    symbol: str = "BTC",
) -> str:
    """Get the BASTION Heat Index — a composite attention/activity score (0-100).

    Combines whale activity, funding extremes, liquidation clusters, volatility regime,
    and price action into a single attention score. High heat = high activity = opportunity
    or danger. When the model says EXIT_FULL on a 90+ heat score, pay attention.

    Labels: QUIET (0-20), LOW (20-40), MODERATE (40-65), HIGH (65-85), EXTREME (85-100)

    Args:
        symbol: Crypto symbol

    Returns:
        Heat index score, label, component breakdown, and raw data.
    """
    result = await api_get(f"/api/heat-index/{symbol}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_heat_scan() -> str:
    """Scan heat index across top 10 crypto pairs — find where the action is.

    Returns a ranked list of all major pairs by their heat index score.
    The hottest pair has the most combined activity (whales, funding, volatility, price action).
    Use this to identify which symbols need attention right now.

    Returns:
        Ranked list of symbols by heat index with scores and labels.
    """
    result = await api_get("/api/heat-scan")
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# COPY-ANALYSIS WORKFLOWS
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_save_workflow(
    name: str,
    tools: str,
    description: str = "",
    creator: str = "",
    public: bool = True,
) -> str:
    """Save a named analysis workflow — a reusable sequence of BASTION tools.

    Create shareable analysis chains that other agents can copy and run.
    Top leaderboard analysts share their tool sequences; followers run them with one call.

    Args:
        name: Workflow name (e.g., "Alpha Whale's BTC Setup")
        tools: JSON string of tool sequence — [{"tool": "bastion_get_price", "params": {"symbol": "BTC"}}, ...]
        description: What this workflow does
        creator: Creator name/ID
        public: Whether other agents can discover and run this workflow

    Returns:
        Saved workflow with unique ID.
    """
    try:
        tools_list = json.loads(tools) if isinstance(tools, str) else tools
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid tools JSON. Format: [{\"tool\": \"bastion_get_price\", \"params\": {\"symbol\": \"BTC\"}}]"})

    result = await api_post("/api/workflows/save", {
        "name": name, "tools": tools_list,
        "description": description, "creator": creator, "public": public,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_workflows() -> str:
    """List all public analysis workflows, sorted by popularity.

    Browse what analysis chains other analysts have created. Find popular
    tool sequences for different market conditions and copy them.

    Returns:
        List of public workflows with names, descriptions, tool sequences, and run counts.
    """
    result = await api_get("/api/workflows")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_run_workflow(
    workflow_id: str,
    symbol: str = "",
) -> str:
    """Execute a saved analysis workflow — runs all tools in sequence.

    Run another analyst's tool chain on any symbol. Each tool in the workflow
    executes in order, and all results are returned together.

    Args:
        workflow_id: The workflow ID to execute
        symbol: Override symbol for all tools in the workflow (empty = use workflow defaults)

    Returns:
        All tool results from the workflow execution with total latency.
    """
    result = await api_post(f"/api/workflows/run/{workflow_id}", {
        "symbol": symbol,
    })
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# IMMUTABLE AUDIT TRAIL + DECISION PROVENANCE
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_audit_log(
    tool: str,
    action: str = "tool_call",
    input_summary: str = "",
    output_summary: str = "",
    latency_ms: int = 0,
    user_id: str = "",
    session_id: str = "",
    success: bool = True,
) -> str:
    """Log an immutable audit entry with hash chain integrity.

    Every tool invocation should be logged here for compliance. Creates a tamper-evident
    record with cryptographic hash chaining — each entry includes the hash of the
    previous entry, creating a blockchain-like audit trail.

    Args:
        tool: Tool name that was called
        action: Action type (tool_call, trade_execution, risk_evaluation, etc.)
        input_summary: Brief summary of inputs (redact sensitive data)
        output_summary: Brief summary of output
        latency_ms: Response time in milliseconds
        user_id: User identifier
        session_id: Session identifier
        success: Whether the call succeeded

    Returns:
        Audit event ID and hash for verification.
    """
    result = await api_post("/api/audit/log", {
        "tool": tool, "action": action,
        "input_summary": input_summary, "output_summary": output_summary,
        "latency_ms": latency_ms, "user_id": user_id,
        "session_id": session_id, "success": success,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_audit_trail(
    user_id: str = "",
    tool: str = "",
    limit: int = 50,
) -> str:
    """Query the immutable audit trail — hash-chained event history.

    Returns tamper-evident audit records. Each event includes a cryptographic hash
    linking it to the previous event. Any modification breaks the chain, which
    bastion_audit_verify can detect. SOC 2 / GDPR exportable.

    Args:
        user_id: Filter by user (empty = all users)
        tool: Filter by tool name (empty = all tools)
        limit: Max events to return

    Returns:
        Hash-chained audit events with integrity status.
    """
    params = {"limit": str(limit)}
    if user_id: params["user_id"] = user_id
    if tool: params["tool"] = tool
    result = await api_get("/api/audit/trail", params=params)
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_audit_verify() -> str:
    """Verify the integrity of the entire audit chain.

    Walks the complete hash chain and verifies every link. If any event has been
    tampered with, the chain breaks and this tool reports exactly where.
    Use this for compliance audits and integrity checks.

    Returns:
        Chain length, integrity status (INTACT/BROKEN), and break point if any.
    """
    result = await api_post("/api/audit/verify", {})
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_decision_provenance(
    symbol: str = "",
    decision: str = "",
    confidence: float = 0,
    tools_called: str = "",
    data_context: str = "",
    model_output: str = "",
    user_id: str = "",
    user_followed: bool = True,
    actual_outcome: str = "",
    outcome_correct: bool = True,
) -> str:
    """Record complete decision provenance — why the AI recommended what it did.

    Creates an auditable record of the full decision chain: which tools were called,
    what data was seen, what the model output, whether the user followed the recommendation,
    and what the actual outcome was. The institutional selling point: "We can show auditors
    exactly why the AI said EXIT."

    Args:
        symbol: Symbol the decision was about
        decision: The AI's recommendation (e.g., "EXIT_FULL", "HOLD")
        confidence: Model confidence (0-1)
        tools_called: Comma-separated list of tools called during analysis
        data_context: Key data points that influenced the decision (JSON string)
        model_output: The model's raw output/reasoning
        user_id: User identifier
        user_followed: Whether the user followed the recommendation
        actual_outcome: What actually happened (e.g., "price_dropped_5pct")
        outcome_correct: Whether the AI's recommendation was correct

    Returns:
        Provenance record ID for future reference.
    """
    tools_list = [t.strip() for t in tools_called.split(",") if t.strip()] if tools_called else []
    try:
        context = json.loads(data_context) if data_context else {}
    except json.JSONDecodeError:
        context = {"raw": data_context}

    result = await api_post("/api/audit/provenance", {
        "symbol": symbol, "decision": decision, "confidence": confidence,
        "tools_called": tools_list, "data_context": context,
        "model_output": model_output, "user_id": user_id,
        "user_followed": user_followed, "actual_outcome": actual_outcome,
        "outcome_correct": outcome_correct,
    })
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_provenance(
    user_id: str = "",
    symbol: str = "",
    limit: int = 20,
) -> str:
    """Query decision provenance — the complete 'why' chain for AI recommendations.

    Shows the full reasoning trail: which tools were called, what data the model saw,
    what it recommended, whether the user followed, and whether it was right.
    Essential for institutional compliance and model evaluation.

    Args:
        user_id: Filter by user
        symbol: Filter by symbol
        limit: Max provenance records

    Returns:
        Decision provenance records with full reasoning chains.
    """
    params = {"limit": str(limit)}
    if user_id: params["user_id"] = user_id
    if symbol: params["symbol"] = symbol
    result = await api_get("/api/audit/provenance", params=params)
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOL SAFETY ANNOTATIONS QUERY
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_tool_safety() -> str:
    """Get the safety annotation matrix for all BASTION MCP tools.

    Returns readOnlyHint, destructiveHint, and idempotentHint for every tool.
    Clients can use this to auto-gate destructive tools behind confirmation prompts
    while letting safe read-only tools run freely.

    Returns:
        Safety matrix with annotations for all 100+ tools organized by category.
    """
    return json.dumps({
        "annotations": {
            "read_only_safe": {
                "description": "Safe to auto-approve. No side effects.",
                "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                "tools": [
                    "bastion_get_price", "bastion_get_market_data", "bastion_get_klines",
                    "bastion_get_volatility", "bastion_get_volatility_regime",
                    "bastion_get_btc_dominance", "bastion_get_correlation_matrix",
                    "bastion_get_confluence", "bastion_get_sector_rotation",
                    "bastion_get_open_interest", "bastion_get_oi_changes",
                    "bastion_get_cvd", "bastion_get_orderflow",
                    "bastion_get_funding_rates", "bastion_get_funding_arb",
                    "bastion_get_liquidations", "bastion_get_liquidations_by_exchange",
                    "bastion_get_heatmap", "bastion_get_taker_ratio",
                    "bastion_get_top_traders", "bastion_get_market_maker_magnet",
                    "bastion_get_options", "bastion_get_whale_activity",
                    "bastion_get_exchange_flow", "bastion_get_onchain",
                    "bastion_get_news", "bastion_get_smart_money",
                    "bastion_get_hyperliquid_whales", "bastion_get_fear_greed",
                    "bastion_get_macro_signals", "bastion_get_etf_flows",
                    "bastion_get_stablecoin_markets", "bastion_get_economic_data",
                    "bastion_get_polymarket", "bastion_get_positions",
                    "bastion_get_balance", "bastion_get_exchanges",
                    "bastion_engine_status", "bastion_engine_history",
                    "bastion_get_alerts", "bastion_get_session_stats",
                    "bastion_get_reports", "bastion_get_trade_journal",
                    "bastion_get_leaderboard", "bastion_check_alerts",
                    "bastion_get_performance", "bastion_list_webhooks",
                    "bastion_get_agent_analytics", "bastion_get_regime_tools",
                    "bastion_get_challenges", "bastion_get_workflows",
                    "bastion_get_provenance", "bastion_quick_intel",
                    "bastion_live_feed", "bastion_heat_index", "bastion_heat_scan",
                    "bastion_war_room_read", "bastion_war_room_consensus",
                    "bastion_war_room_consensus_weighted", "bastion_get_server_card",
                    "bastion_memory_recall", "bastion_memory_profile",
                    "bastion_tool_safety", "bastion_audit_trail",
                ],
            },
            "ai_analysis": {
                "description": "Calls AI model. Read-only but uses compute. Safe to approve.",
                "readOnlyHint": True, "destructiveHint": False, "idempotentHint": False,
                "tools": [
                    "bastion_evaluate_risk", "bastion_chat",
                    "bastion_evaluate_all_positions", "bastion_scan_signals",
                    "bastion_deep_analysis", "bastion_execute_regime_tool",
                    "bastion_risk_confirm", "bastion_risk_card",
                ],
            },
            "write_safe": {
                "description": "Creates/modifies data but not destructive. Low risk.",
                "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False,
                "tools": [
                    "bastion_generate_report", "bastion_calculate_position",
                    "bastion_get_kelly_sizing", "bastion_log_trade",
                    "bastion_backtest_strategy", "bastion_get_risk_parity",
                    "bastion_strategy_builder", "bastion_risk_replay",
                    "bastion_log_prediction", "bastion_subscribe_alert",
                    "bastion_cancel_alert", "bastion_create_risk_card",
                    "bastion_record_equity", "bastion_add_webhook",
                    "bastion_send_notification", "bastion_format_risk",
                    "bastion_war_room_post", "bastion_war_room_vote",
                    "bastion_create_challenge", "bastion_counter_challenge",
                    "bastion_score_challenge", "bastion_save_workflow",
                    "bastion_run_workflow", "bastion_memory_store",
                    "bastion_audit_log", "bastion_audit_verify",
                    "bastion_decision_provenance",
                ],
            },
            "destructive_trading": {
                "description": "EXECUTES TRADES. Requires explicit user confirmation.",
                "readOnlyHint": False, "destructiveHint": True, "idempotentHint": False,
                "tools": [
                    "bastion_emergency_exit", "bastion_partial_close",
                    "bastion_set_take_profit", "bastion_set_stop_loss",
                    "bastion_move_to_breakeven", "bastion_flatten_winners",
                ],
            },
            "engine_control": {
                "description": "Controls autonomous trading engine. High impact.",
                "readOnlyHint": False, "destructiveHint": True, "idempotentHint": False,
                "tools": [
                    "bastion_engine_start", "bastion_engine_arm",
                    "bastion_engine_disarm",
                ],
            },
        },
        "total_tools": 108,
    }, indent=2)


# ═════════════════════════════════════════════════════════════════
# PROMPTS
# ═════════════════════════════════════════════════════════════════


@mcp.prompt()
async def evaluate_my_position(
    symbol: str,
    direction: str,
    entry_price: str,
    current_price: str,
    leverage: str = "1",
    stop_loss: str = "0",
) -> str:
    """Evaluate a trading position using BASTION Risk Intelligence.

    Args:
        symbol: Trading pair (e.g. BTC, ETH, SOL)
        direction: LONG or SHORT
        entry_price: Entry price
        current_price: Current price
        leverage: Leverage multiplier
        stop_loss: Stop loss price (0 = no stop)
    """
    return (
        f"I have a {direction.upper()} position on {symbol.upper()}.\n\n"
        f"Entry: ${entry_price}\n"
        f"Current: ${current_price}\n"
        f"Leverage: {leverage}x\n"
        f"Stop Loss: {'$' + stop_loss if float(stop_loss) > 0 else 'None set'}\n\n"
        f"Use the bastion_evaluate_risk tool to analyze this position. "
        f"Tell me whether I should hold, take profit, reduce size, or exit — and explain why."
    )


@mcp.prompt()
async def market_analysis(symbol: str = "BTC") -> str:
    """Get a complete market analysis for a cryptocurrency.

    Args:
        symbol: Crypto symbol to analyze (default BTC)
    """
    return (
        f"Give me a complete market analysis for {symbol.upper()}.\n\n"
        f"Use the following BASTION tools:\n"
        f"1. bastion_get_price — get current price\n"
        f"2. bastion_get_market_data — get order flow, OI, volatility\n"
        f"3. bastion_get_funding_rates — check funding sentiment\n"
        f"4. bastion_get_whale_activity — check whale movements\n"
        f"5. bastion_get_liquidations — check liquidation risk zones\n\n"
        f"Synthesize all the data into a clear market outlook with key levels, "
        f"sentiment assessment, and risk factors."
    )


@mcp.prompt()
async def risk_check(symbol: str = "BTC") -> str:
    """Quick risk check for a cryptocurrency.

    Args:
        symbol: Crypto symbol to check (default BTC)
    """
    return (
        f"Run a quick risk check on {symbol.upper()} using BASTION tools:\n\n"
        f"1. Check liquidation clusters (bastion_get_liquidations)\n"
        f"2. Check whale activity (bastion_get_whale_activity)\n"
        f"3. Check funding rates (bastion_get_funding_rates)\n\n"
        f"Summarize: Is this a high-risk or low-risk environment right now? "
        f"What are the key threats?"
    )


@mcp.prompt()
async def portfolio_risk_scan() -> str:
    """Run a comprehensive risk scan on all open positions.

    Evaluates every position, checks whale activity, funding rates,
    liquidation risk, and macro conditions — returns a full risk briefing.
    """
    return (
        "Run a comprehensive risk scan using BASTION tools:\n\n"
        "1. Get all open positions (bastion_get_positions)\n"
        "2. Run AI evaluation on ALL positions (bastion_evaluate_all_positions)\n"
        "3. Check whale activity (bastion_get_whale_activity)\n"
        "4. Check funding rates (bastion_get_funding_rates)\n"
        "5. Check Fear & Greed Index (bastion_get_fear_greed)\n"
        "6. Check macro signals (bastion_get_macro_signals)\n\n"
        "Synthesize into a clear risk briefing:\n"
        "- Which positions need immediate action?\n"
        "- What's the overall portfolio risk level (LOW/MEDIUM/HIGH/CRITICAL)?\n"
        "- Are there any macro headwinds or tailwinds?\n"
        "- What's the recommended course of action?"
    )


@mcp.prompt()
async def whale_tracker(symbol: str = "BTC") -> str:
    """Track whale activity and smart money flows for a cryptocurrency.

    Args:
        symbol: Crypto symbol to track (default BTC)
    """
    return (
        f"Track whale activity and smart money flows for {symbol.upper()}:\n\n"
        f"1. Get whale transactions (bastion_get_whale_activity)\n"
        f"2. Get exchange inflow/outflow (bastion_get_exchange_flow)\n"
        f"3. Get Hyperliquid whale positions (bastion_get_hyperliquid_whales)\n"
        f"4. Get smart money flow (bastion_get_smart_money)\n"
        f"5. Get top trader positioning (bastion_get_top_traders)\n\n"
        f"Summarize: Where is the smart money flowing? Are whales accumulating "
        f"or distributing? What do Hyperliquid whale positions tell us about "
        f"institutional conviction? Is there divergence between retail and smart money?"
    )


@mcp.prompt()
async def macro_briefing() -> str:
    """Get a complete macro + crypto market briefing.

    Covers traditional markets, crypto sentiment, BTC dominance,
    stablecoins, prediction markets, and risk environment.
    """
    return (
        "Give me a complete macro + crypto market briefing using BASTION tools:\n\n"
        "1. Get macro signals — DXY, VIX, yields, equities (bastion_get_macro_signals)\n"
        "2. Get Fear & Greed Index (bastion_get_fear_greed)\n"
        "3. Get BTC dominance and altseason score (bastion_get_btc_dominance)\n"
        "4. Get ETF flows (bastion_get_etf_flows)\n"
        "5. Get stablecoin markets (bastion_get_stablecoin_markets)\n"
        "6. Get BTC volatility regime (bastion_get_volatility_regime)\n"
        "7. Get prediction market sentiment (bastion_get_polymarket)\n\n"
        "Synthesize into a clear briefing:\n"
        "- Is it risk-on or risk-off in traditional markets?\n"
        "- What's the crypto sentiment and where in the cycle are we?\n"
        "- Is capital flowing into or out of crypto (ETFs + stablecoins)?\n"
        "- Are we in altseason or BTC season?\n"
        "- What are prediction markets pricing in?"
    )


@mcp.prompt()
async def pre_trade_analysis(
    symbol: str,
    direction: str,
    entry_price: str,
    stop_loss: str,
    take_profit: str,
) -> str:
    """Run a complete pre-trade analysis before entering a position.

    Args:
        symbol: Trading pair (e.g. BTC, ETH, SOL)
        direction: LONG or SHORT
        entry_price: Planned entry price
        stop_loss: Stop loss price
        take_profit: Take profit target
    """
    return (
        f"I'm considering a {direction.upper()} on {symbol.upper()} "
        f"with entry ${entry_price}, stop ${stop_loss}, TP ${take_profit}.\n\n"
        f"Run a complete pre-trade analysis using BASTION tools:\n\n"
        f"1. Calculate position sizing (bastion_calculate_position)\n"
        f"2. Get Kelly Criterion sizing (bastion_get_kelly_sizing)\n"
        f"3. Get current market data (bastion_get_market_data)\n"
        f"4. Check liquidation clusters near entry/stop (bastion_get_liquidations)\n"
        f"5. Check funding sentiment (bastion_get_funding_rates)\n"
        f"6. Check whale positioning (bastion_get_hyperliquid_whales)\n"
        f"7. Check volatility regime (bastion_get_volatility_regime)\n\n"
        f"Tell me:\n"
        f"- Is this a good entry? What's the probability of hitting TP vs SL?\n"
        f"- What's the optimal position size?\n"
        f"- Are there liquidation clusters that could sweep my stop?\n"
        f"- Is the market structure supportive of a {direction.upper()}?\n"
        f"- Final verdict: TAKE THE TRADE or PASS?"
    )


@mcp.prompt()
async def strategy_lab(
    description: str = "Short when funding is high and OI is rising, 3x leverage, TP 4%, SL 2%",
    symbol: str = "BTC",
) -> str:
    """Build and backtest a trading strategy from a natural language description.

    Args:
        description: Your strategy described in plain English
        symbol: Trading pair to test on (default BTC)
    """
    return (
        f"I want to test this strategy on {symbol.upper()}:\n\n"
        f'"{description}"\n\n'
        f"Use BASTION tools to build and backtest this:\n\n"
        f"1. Parse and backtest the strategy (bastion_strategy_builder)\n"
        f"2. Check current market conditions (bastion_get_market_data)\n"
        f"3. Get correlation context (bastion_get_correlation_matrix)\n"
        f"4. Check confluence (bastion_get_confluence)\n\n"
        f"Tell me:\n"
        f"- What rules were extracted from my description?\n"
        f"- How did the strategy perform in backtesting (win rate, PnL)?\n"
        f"- Is the current market favorable for this strategy?\n"
        f"- Any suggestions to improve it?"
    )


@mcp.prompt()
async def full_risk_dashboard(
    symbol: str = "BTC",
    direction: str = "LONG",
    entry_price: str = "0",
) -> str:
    """Generate a full risk intelligence dashboard with visualizations.

    Args:
        symbol: Trading pair (e.g. BTC, ETH, SOL)
        direction: LONG or SHORT
        entry_price: Entry price (0 = use current price)
    """
    return (
        f"Build a complete risk dashboard for my {direction.upper()} {symbol.upper()} "
        f"position (entry: {'current price' if entry_price == '0' else '$' + entry_price}).\n\n"
        f"Use these BASTION tools in sequence:\n\n"
        f"1. Generate risk visualization card (bastion_risk_card)\n"
        f"2. Get AI risk evaluation (bastion_evaluate_risk)\n"
        f"3. Check multi-timeframe confluence (bastion_get_confluence)\n"
        f"4. Get correlation matrix (bastion_get_correlation_matrix)\n"
        f"5. Check model leaderboard accuracy (bastion_get_leaderboard)\n"
        f"6. Check liquidation risk (bastion_get_liquidations)\n"
        f"7. Check whale positioning (bastion_get_hyperliquid_whales)\n\n"
        f"Synthesize everything into a comprehensive risk briefing with:\n"
        f"- Current risk level and AI recommendation\n"
        f"- Multi-timeframe analysis\n"
        f"- Correlation exposure risks\n"
        f"- Model confidence and accuracy context\n"
        f"- Key levels to watch (support, resistance, liquidation clusters)\n"
        f"- Final action recommendation"
    )


@mcp.prompt()
async def deep_intelligence_brief(
    symbol: str = "BTC",
) -> str:
    """Run BASTION's most comprehensive analysis — deep multi-step intelligence brief.

    Args:
        symbol: Crypto symbol to analyze (default BTC)
    """
    return (
        f"Run a DEEP intelligence brief on {symbol.upper()} using BASTION's most powerful tools:\n\n"
        f"1. Deep multi-step analysis (bastion_deep_analysis) — gathers 15 data sources in one call\n"
        f"2. Check what regime-adaptive tools are active (bastion_get_regime_tools)\n"
        f"3. Execute any active regime tools (bastion_execute_regime_tool)\n"
        f"4. Get real-time feed snapshot (bastion_live_feed)\n"
        f"5. Check War Room consensus if available (bastion_war_room_consensus_weighted)\n\n"
        f"Synthesize into an institutional-grade intelligence brief:\n"
        f"- Current market regime and risk level\n"
        f"- Key signals from 15+ data sources\n"
        f"- Regime-specific insights (crisis mode, capitulation, euphoria, etc.)\n"
        f"- Multi-agent consensus (if available)\n"
        f"- Actionable recommendation with conviction level"
    )
