"""
BASTION MCP Server — Core
Exposes BASTION Risk Intelligence as MCP tools for Claude agents.

Tools:
  1. bastion_evaluate_risk     — Core risk intelligence for a position
  2. bastion_chat              — Neural chat (ask anything about markets)
  3. bastion_get_price         — Live crypto price
  4. bastion_get_market_data   — Aggregated market data (CVD, OI, funding, vol)
  5. bastion_get_liquidations  — Liquidation data
  6. bastion_get_whale_activity — Whale transactions
  7. bastion_get_funding_rates — Cross-exchange funding
  8. bastion_generate_report   — Generate MCF Labs research report
  9. bastion_get_reports       — List existing reports
  10. bastion_calculate_position — Position sizing + Monte Carlo

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


async def api_get(path: str, params: dict = None) -> dict:
    """Make a GET request to the BASTION API."""
    client = await get_client()
    try:
        resp = await client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API returned {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


async def api_post(path: str, data: dict = None) -> dict:
    """Make a POST request to the BASTION API."""
    client = await get_client()
    try:
        resp = await client.post(path, json=data)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API returned {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


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
) -> str:
    """Evaluate a crypto futures position using BASTION's AI risk intelligence engine.

    This is the core tool — sends a position to the fine-tuned 32B model which analyzes
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

    Returns:
        JSON with: action (HOLD/EXIT_FULL/TP_PARTIAL/EXIT_100%/REDUCE_SIZE/TRAIL_STOP),
        urgency (LOW/MEDIUM/HIGH/CRITICAL), confidence (0-1), reason, detailed reasoning,
        and execution details.
    """
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

    result = await api_post("/api/risk/evaluate", payload)

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
) -> str:
    """Ask BASTION's neural AI anything about crypto markets, trading, or risk.

    Powered by the same fine-tuned 32B model used for risk intelligence.
    Context-aware — knows current market conditions, whale activity, and structure.

    Args:
        query: Your question (e.g. "Is BTC a buy right now?", "Where are whales moving?")
        symbol: Symbol context for the query (default BTC)

    Returns:
        AI-generated analysis in markdown format.
    """
    result = await api_post("/api/neural/chat", {
        "query": query,
        "symbol": symbol.upper(),
        "include_positions": False,
        "session_id": "mcp-session",
        "user_id": "mcp-user",
    })

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
) -> str:
    """Generate an AI-powered MCF Labs research report.

    Creates institutional-grade analysis with ratings, conviction scores,
    target prices, risk factors, trade structures, and valuation scenarios.

    Args:
        report_type: Type of report — institutional_research (default), market_structure,
            whale_intelligence, options_flow, or cycle_position
        symbol: Symbol to analyze (default BTC)

    Returns:
        Full report with bias, confidence, sections (thesis, key drivers, risks, scenarios).
    """
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
        {"symbol": symbol.upper()}
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
) -> str:
    """List existing MCF Labs research reports.

    Args:
        limit: Number of reports to return (default 10)
        report_type: Filter by type (institutional_research, market_structure, etc.)
        symbol: Filter by symbol (btc, eth, sol)

    Returns:
        List of reports with id, title, bias, type, timestamp, and tags.
    """
    params = {"limit": min(limit, 50)}
    if report_type:
        params["type"] = report_type
    if symbol:
        params["symbol"] = symbol.lower()

    result = await api_get("/api/mcf/reports", params)
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

    Returns:
        Position size, risk/reward ratio, TP probability, SL probability,
        expected value, and trade quality score.
    """
    result = await api_post("/api/pre-trade-calculator", {
        "symbol": symbol.upper(),
        "direction": direction.upper(),
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "leverage": leverage,
        "account_size": account_size,
        "risk_percent": risk_percent,
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
        "base_model": config.MODEL_BASE,
        "method": "QLoRA (4-bit NF4 quantized base + LoRA rank=32, alpha=64)",
        "training_examples": config.MODEL_TRAINING_EXAMPLES,
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
        "data_signals_per_evaluation": "560+",
        "data_sources": [
            "Helsinki VM (33 quant endpoints)",
            "Coinglass Premium (liquidations, OI, funding, whale positions)",
            "Whale Alert (on-chain whale tracking, 11 blockchains)",
            "MCF Structure Service (VPVR, pivots, auto-support, trendlines)"
        ],
        "infrastructure": config.MODEL_GPU,
        "inference_engine": "vLLM with tensor parallelism=4",
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
