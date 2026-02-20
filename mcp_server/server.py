"""
BASTION MCP Server — Core
Exposes the full BASTION platform as MCP tools for Claude agents.

Tools (52):
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

  DERIVATIVES & ORDER FLOW (public)
    bastion_get_open_interest    — Open interest across exchanges
    bastion_get_oi_changes       — OI changes across all pairs
    bastion_get_cvd              — Cumulative Volume Delta
    bastion_get_orderflow        — Order flow analysis
    bastion_get_funding_rates    — Cross-exchange funding rates
    bastion_get_funding_arb      — Funding rate arbitrage
    bastion_get_liquidations     — Liquidation events + clusters
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

Resources:
  bastion://status, bastion://supported-symbols, bastion://model-info

Prompts:
  evaluate_my_position, market_analysis, risk_check
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
