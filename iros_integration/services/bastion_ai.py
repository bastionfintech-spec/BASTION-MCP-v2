"""
BASTION AI - Intelligent Trading Analysis System
=================================================
Powered by IROS infrastructure - 32B LLM + 33 quant endpoints

Usage:
    from iros_integration.services.bastion_ai import BastionAI
    
    bastion = BastionAI()
    result = await bastion.process_query("Should I long BTC with $50K?")
    print(result["response"])
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import httpx

from ..config.settings import settings
from .helsinki import HelsinkiClient, AggregatedMarketData
from .query_processor import QueryProcessor, QueryContext


@dataclass
class BastionQueryResult:
    """Result from a Bastion AI query"""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
    data_sources: List[str] = None
    latency: Dict[str, int] = None
    context: Optional[QueryContext] = None


class BastionAI:
    """
    Main Bastion AI service - institutional-grade crypto analysis.
    
    Combines:
    - Helsinki VM: 33 real-time quant endpoints (FREE)
    - Vast.ai GPU: 32B parameter LLM
    - Query context extraction
    - Dynamic system prompt building
    
    Usage:
        bastion = BastionAI()
        result = await bastion.process_query(
            query="Should I long BTC at $97K with $50K?",
            comprehensive=False  # Use True for full 33-endpoint data
        )
    """
    
    def __init__(
        self,
        helsinki_url: Optional[str] = None,
        model_url: Optional[str] = None,
        model_api_key: Optional[str] = None,
    ):
        # Helsinki client for market data
        self.helsinki = HelsinkiClient(base_url=helsinki_url)
        
        # Model configuration
        self.model_url = model_url or settings.model.base_url
        self.model_api_key = model_api_key or settings.model.api_key
        self.model_timeout = settings.model.timeout
        
        # Query processor for context extraction
        self.query_processor = QueryProcessor()
    
    async def process_query(
        self,
        query: str,
        comprehensive: bool = False,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> BastionQueryResult:
        """
        Process a trading query through Bastion AI.
        
        Args:
            query: User's question (e.g., "Should I long BTC with $50K?")
            comprehensive: If True, fetches all 33 endpoints (slower but more data)
            user_context: Optional dict to override extracted context
            
        Returns:
            BastionQueryResult with response, sources, and latency info
        """
        start_time = time.time()
        data_sources: List[str] = []
        
        try:
            # 1. Extract query context
            context = self.query_processor.extract_context(query)
            
            # Override with user-provided context if given
            if user_context:
                if "capital" in user_context:
                    context.capital = user_context["capital"]
                if "timeframe" in user_context:
                    context.timeframe = user_context["timeframe"]
                if "risk_tolerance" in user_context:
                    context.risk_tolerance = user_context["risk_tolerance"]
            
            # 2. Detect query type for specialized fetching
            query_type = self._detect_query_type(query)
            
            # 3. Fetch market data from Helsinki
            data_fetch_start = time.time()
            
            if query_type == "cvd":
                cvd_data = await self.helsinki.fetch_cvd_data(context.symbol)
                market_context = cvd_data["formatted"]
                data_sources.extend(["Helsinki:cvd", "Helsinki:options-iv"])
            elif query_type == "volatility":
                vol_data = await self.helsinki.fetch_volatility_data(context.symbol)
                market_context = self._format_volatility_data(vol_data, context.symbol)
                data_sources.extend(["Helsinki:volatility", "Helsinki:iv-rv-spread", "Helsinki:options-iv"])
            elif query_type == "liquidation":
                liq_data = await self.helsinki.fetch_liquidation_data(context.symbol)
                market_context = self._format_liquidation_data(liq_data, context.symbol)
                data_sources.extend(["Helsinki:liquidation-estimate", "Helsinki:options-iv"])
            else:
                # General query - fetch priority or comprehensive data
                if comprehensive:
                    market_data = await self.helsinki.fetch_comprehensive_data(context.symbol)
                else:
                    market_data = await self.helsinki.fetch_full_data(context.symbol)
                
                market_context = self.helsinki.format_for_prompt(market_data)
                data_sources.extend([f"Helsinki:{s}" for s in market_data.meta.get("sources", [])])
            
            data_fetch_time = int((time.time() - data_fetch_start) * 1000)
            
            # 4. Build system prompt
            system_prompt = self._build_system_prompt(market_context, context)
            
            # 5. Query the AI model
            inference_start = time.time()
            model_result = await self._query_model(query, system_prompt)
            inference_time = int((time.time() - inference_start) * 1000)
            
            data_sources.append("Bastion:32B-LLM")
            
            total_time = int((time.time() - start_time) * 1000)
            
            return BastionQueryResult(
                success=True,
                response=model_result["response"],
                tokens_used=model_result.get("tokens_used", 0),
                data_sources=data_sources,
                latency={
                    "data_fetch": data_fetch_time,
                    "inference": inference_time,
                    "total": total_time,
                },
                context=context,
            )
            
        except Exception as e:
            return BastionQueryResult(
                success=False,
                error=str(e),
                data_sources=data_sources,
                latency={"total": int((time.time() - start_time) * 1000)},
            )
    
    def _detect_query_type(self, query: str) -> str:
        """Detect if query is about a specific topic"""
        q = query.lower()
        
        if any(x in q for x in ["cvd", "cumulative volume", "volume delta"]):
            return "cvd"
        if any(x in q for x in ["volatility", "vol regime", "atr", "iv ", "rv "]):
            return "volatility"
        if any(x in q for x in ["liquidation", "cascade", "squeeze", "liq"]):
            return "liquidation"
        if any(x in q for x in ["momentum", "trend strength", "roc"]):
            return "momentum"
        if any(x in q for x in ["orderbook", "order book", "bid ask", "imbalance"]):
            return "orderbook"
        
        return "general"
    
    def _build_system_prompt(self, market_context: str, context: QueryContext) -> str:
        """Build the institutional system prompt with context"""
        
        # Build context section
        context_section = self.query_processor.build_context_section(context)
        context_block = f"\n## USER CONTEXT\n{context_section}\n" if context_section else ""
        
        return f"""You are BASTION, a senior quant analyst for a $500M+ hedge fund. Provide institutional-grade analysis.
{context_block}
CRITICAL RULES:
- USE THE CURRENT PRICE FROM THE DATA. NEVER MAKE UP PRICES.
- No emojis. Use probabilities and confidence scores.
- Be precise, quantified, actionable.
- Reject bad setups with clear reasoning.
- ADAPT YOUR RESPONSE TO THE USER'S CONTEXT (timeframe, risk tolerance, trade type).
- IF USER ASKS ABOUT EXIT/SELL, focus on exit strategy, not new entries.
- IF USER ASKS ABOUT DCA, provide accumulation zones with allocation percentages.

RESPONSE FORMAT:

## Key Structural Levels
- Resistance: $X (Grade 1-3, touches)
- Support: $X (Grade 1-3, touches)

## Entry Setup (Test → Break → Retest)
- Current Phase: [Awaiting break / Testing / Confirmed]
- Entry Trigger: Bullish: [condition] ; Bearish: [condition]
- Confirmation Required: [candle pattern needed]

## Trading Scenarios
BULLISH (Break + Retest of $RESISTANCE):
- Entry: $X (after retest confirmation)
- Target 1: $X (+X% from entry)
- Target 2: $X (+X% from entry)
- Target 3: $X (+X% from entry)
- Stop: $X (below breakdown level, 0.5x ATR buffer)
- Risk: Reward: 1:X

BEARISH (Rejection at $RESISTANCE):
- Entry: $X (break below support)
- Target 1: $X (-X% from entry)
- Target 2: $X (-X% from entry)
- Stop: $X (above resistance, 0.5x ATR buffer)
- Risk: Reward: 1:X

## Risk Shield Position Sizing
- User Capital: $[USE USER'S STATED CAPITAL or assume $10,000]
- Risk Budget: 2% of capital per trade
- Position Size: (Capital × Risk%) / (Entry - Stop) = X units
- Dollar Value: $X at current price
- Volatility Adjustment: [None/Reduce 25%/50%] (based on current regime)
- Final Position: X units ($X USD)

## VERDICT
Bias: [BULLISH/BEARISH/NEUTRAL] | Confidence: X% | Action: [Specific instruction with entry, stop, target]

LIVE DATA:
{market_context}"""
    
    async def _query_model(
        self,
        user_query: str,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Query the Bastion AI model on Vast.ai"""
        
        # Calculate dynamic max tokens
        input_text = system_prompt + user_query
        estimated_input_tokens = len(input_text) // 3  # Conservative estimate
        max_context = 4096
        max_output_tokens = min(1200, max(800, max_context - estimated_input_tokens - 100))
        
        request_body = {
            "model": settings.model.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            "max_tokens": max_output_tokens,
            "temperature": settings.model.temperature,
            "top_p": settings.model.top_p,
        }
        
        model_url = f"{self.model_url}/v1/chat/completions"
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.model_api_key:
                headers["Authorization"] = f"Bearer {self.model_api_key}"
            
            async with httpx.AsyncClient(timeout=self.model_timeout / 1000) as client:
                response = await client.post(
                    model_url,
                    headers=headers,
                    json=request_body,
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    raise Exception(f"Model error: {response.status_code} - {error_text}")
                
                result = response.json()
                
                return {
                    "response": (
                        result.get("response") or 
                        result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    ),
                    "tokens_used": (
                        result.get("tokens_used") or 
                        result.get("usage", {}).get("total_tokens", 0)
                    ),
                }
                
        except httpx.TimeoutException:
            raise Exception("Model inference timed out")
        except Exception as e:
            # Return fallback response with market data
            return {
                "response": f"""[WARNING] **BASTION MODEL TEMPORARILY UNAVAILABLE**

The Bastion 32B model at {self.model_url} is not reachable.

**Error:** {str(e)}

**However, here's your live market data:**

{system_prompt.split("LIVE DATA:")[-1] if "LIVE DATA:" in system_prompt else "Market data available in context."}

---
**To fix this:**
1. Check if your Vast.ai instance is running
2. Verify the Cloudflare tunnel URL hasn't changed
3. Ensure the API key is correct

Once the model is back online, you'll get full institutional analysis.""",
                "tokens_used": 0,
            }
    
    def _format_volatility_data(self, data: Dict[str, Any], symbol: str) -> str:
        """Format volatility data for prompt"""
        lines = [f"## VOLATILITY ANALYSIS - {symbol}"]
        
        if data.get("price"):
            lines.append(f"**Price:** ${data['price']:,.2f}\n")
        
        vol = data.get("volatility")
        if vol:
            lines.append(f"**Regime:** {vol.get('current_regime', 'N/A')}")
            lines.append(f"**7D Volatility:** {vol.get('volatility_7d_pct', 0):.1f}%")
            lines.append(f"**30D Volatility:** {vol.get('volatility_30d_pct', 0):.1f}%")
            lines.append(f"**Percentile:** {vol.get('volatility_percentile', 0):.0f}th")
        
        iv_rv = data.get("iv_rv")
        if iv_rv:
            lines.append(f"**IV vs RV Spread:** {iv_rv.get('spread_pct', 0):.1f}%")
            lines.append(f"**IV:** {iv_rv.get('implied_volatility_pct', 0):.1f}%")
            lines.append(f"**RV:** {iv_rv.get('realized_volatility_pct', 0):.1f}%")
        
        return "\n".join(lines)
    
    def _format_liquidation_data(self, data: Dict[str, Any], symbol: str) -> str:
        """Format liquidation data for prompt"""
        lines = [f"## LIQUIDATION MAP - {symbol}"]
        
        if data.get("price"):
            lines.append(f"**Price:** ${data['price']:,.2f}\n")
        
        liq = data.get("liquidation")
        if liq:
            oi = liq.get("open_interest_usd", 0) / 1e9
            lines.append(f"**Open Interest:** ${oi:.2f}B")
            lines.append(f"**Long/Short Ratio:** {liq.get('long_short_ratio', 0):.2f}")
            lines.append(f"**Cascade Bias:** {liq.get('cascade_bias', 'N/A')}\n")
            
            down_zones = liq.get("downside_liquidation_zones", [])
            if down_zones:
                lines.append("**Downside Zones:**")
                for zone in down_zones[:3]:
                    lines.append(
                        f"  ↓ ${zone.get('price', 0):.0f} "
                        f"({zone.get('distance_pct', 0)}%) - "
                        f"${zone.get('estimated_usd_at_risk', 0) / 1e6:.0f}M at risk"
                    )
            
            up_zones = liq.get("upside_liquidation_zones", [])
            if up_zones:
                lines.append("**Upside Zones:**")
                for zone in up_zones[:3]:
                    lines.append(
                        f"  ↑ ${zone.get('price', 0):.0f} "
                        f"(+{zone.get('distance_pct', 0)}%) - "
                        f"${zone.get('estimated_usd_at_risk', 0) / 1e6:.0f}M at risk"
                    )
        
        return "\n".join(lines)


# Convenience function for quick queries
async def ask_bastion(query: str, **kwargs) -> str:
    """
    Quick helper to query Bastion AI.
    
    Usage:
        response = await ask_bastion("Should I long BTC with $50K?")
        print(response)
    """
    bastion = BastionAI()
    result = await bastion.process_query(query, **kwargs)
    
    if result.success:
        return result.response
    else:
        return f"Error: {result.error}"






