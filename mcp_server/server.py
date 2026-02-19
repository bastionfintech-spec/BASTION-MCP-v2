"""
BASTION MCP Server — Core
Exposes the full BASTION platform as MCP tools for Claude agents.

Tools (40):
  CORE AI
    bastion_evaluate_risk        — AI risk intelligence for a position
    bastion_chat                 — Neural chat (ask anything about markets)
    bastion_evaluate_all_positions — Evaluate all open positions at once
    bastion_scan_signals         — Scan for trading signals across pairs

  MARKET DATA
    bastion_get_price            — Live crypto price
    bastion_get_market_data      — Aggregated market intelligence
    bastion_get_klines           — Candlestick OHLCV data
    bastion_get_volatility       — Volatility metrics + regime detection

  DERIVATIVES & ORDER FLOW
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

  ON-CHAIN & INTELLIGENCE
    bastion_get_whale_activity   — Whale transactions (11 chains)
    bastion_get_exchange_flow    — Exchange inflow/outflow
    bastion_get_onchain          — On-chain metrics
    bastion_get_news             — Aggregated crypto news

  MACRO & SENTIMENT
    bastion_get_fear_greed       — Fear & Greed Index
    bastion_get_macro_signals    — Macro signals (DXY, yields, equities)
    bastion_get_etf_flows        — BTC/ETH ETF flow data
    bastion_get_stablecoin_markets — Stablecoin supply + flows
    bastion_get_economic_data    — FRED economic data series
    bastion_get_polymarket       — Prediction market data

  RESEARCH
    bastion_generate_report      — Generate MCF Labs research report
    bastion_get_reports          — List existing reports
    bastion_calculate_position   — Position sizing + Monte Carlo

  PORTFOLIO & TRADING
    bastion_get_positions        — All open positions
    bastion_get_balance          — Total portfolio balance
    bastion_get_exchanges        — Connected exchanges
    bastion_engine_status        — Trading engine status
    bastion_engine_history       — Engine execution history
    bastion_get_alerts           — Active alerts & notifications
    bastion_get_session_stats    — Trading session statistics

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

    Powered by the same fine-tuned 72B model used for risk intelligence.
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
async def bastion_get_positions() -> str:
    """Get all open positions across connected exchanges.

    Returns current positions with entry price, current price,
    unrealized PnL, leverage, and other details.

    Returns:
        All open futures positions across Binance, Bybit, Bitunix, etc.
    """
    result = await api_get("/api/positions/all")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_balance() -> str:
    """Get total portfolio balance across all connected exchanges.

    Returns aggregated balance with per-exchange breakdown.

    Returns:
        Total balance, available margin, and exchange breakdown.
    """
    result = await api_get("/api/balance/total")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_get_exchanges() -> str:
    """List all connected exchanges.

    Shows which exchanges are linked to BASTION with connection status.

    Returns:
        Connected exchanges with status information.
    """
    result = await api_get("/api/exchange/list")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_evaluate_all_positions() -> str:
    """Run AI risk evaluation on ALL open positions simultaneously.

    Sends every open position through the 72B model for evaluation.
    Returns a risk assessment for each position with recommended actions.

    Returns:
        Risk evaluation results for every open position.
    """
    result = await api_post("/api/risk/evaluate-all")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_scan_signals() -> str:
    """Scan for trading signals across all supported pairs.

    Uses the AI engine to identify potential trade setups based on
    current market conditions, structure, and flow data.

    Returns:
        Active trading signals with confidence scores and reasoning.
    """
    result = await api_post("/api/signals/scan")
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOLS — TRADING ENGINE
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_engine_status() -> str:
    """Get the autonomous trading engine status.

    Shows whether the engine is running, armed, and its current configuration
    including safety parameters and execution history.

    Returns:
        Engine state, armed status, configuration, and recent execution stats.
    """
    result = await api_get("/api/engine/status")
    return json.dumps(result, indent=2)


@mcp.tool()
async def bastion_engine_history(limit: int = 20) -> str:
    """Get trading engine execution history.

    Shows recent actions taken by the autonomous engine — which positions
    it evaluated, what actions it took, and the reasoning.

    Args:
        limit: Number of history entries to return (default 20)

    Returns:
        Recent engine execution events with timestamps and details.
    """
    result = await api_get("/api/engine/execution-history", {"limit": min(limit, 100)})
    return json.dumps(result, indent=2)


# ═════════════════════════════════════════════════════════════════
# TOOLS — ALERTS & KLINES
# ═════════════════════════════════════════════════════════════════


@mcp.tool()
async def bastion_get_alerts() -> str:
    """Get active alerts and notifications.

    Returns risk alerts, liquidation warnings, whale movement alerts,
    and user-configured price alerts.

    Returns:
        Active alerts with severity, type, and details.
    """
    result = await api_get("/api/alerts")
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
async def bastion_get_session_stats() -> str:
    """Get current trading session statistics.

    Shows performance metrics for the current trading session including
    win rate, PnL, number of trades, and risk metrics.

    Returns:
        Session statistics with performance breakdown.
    """
    result = await api_get("/api/session/stats")
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
