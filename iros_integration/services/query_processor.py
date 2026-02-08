"""
Query Context Extraction
=========================
Parses user queries to extract trading context for smarter AI responses
"""

import re
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class QueryContext:
    """Extracted context from user query"""
    capital: Optional[float] = None
    timeframe: Optional[Literal["scalp", "day", "swing", "position", "longterm"]] = None
    trade_type: Optional[Literal["spot", "futures"]] = None
    leverage: Optional[int] = None
    risk_tolerance: Optional[Literal["conservative", "moderate", "aggressive"]] = None
    query_intent: Literal["entry", "exit", "analysis", "dca", "compare"] = "analysis"
    specific_price: Optional[float] = None
    symbol: str = "BTC"


class QueryProcessor:
    """
    Extracts relevant context from user queries for smarter AI responses.
    
    Usage:
        processor = QueryProcessor()
        context = processor.extract_context("Should I long BTC at $97K with $50K?")
    """
    
    # Symbol detection patterns
    SYMBOL_PATTERNS = {
        "ETH": ["eth", "ethereum", "ether"],
        "SOL": ["sol", "solana"],
        "BNB": ["bnb", "binance coin"],
        "XRP": ["xrp", "ripple"],
        "ADA": ["ada", "cardano"],
        "DOGE": ["doge", "dogecoin"],
        "AVAX": ["avax", "avalanche"],
        "LINK": ["link", "chainlink"],
        "DOT": ["dot", "polkadot"],
        "MATIC": ["matic", "polygon"],
        "ATOM": ["atom", "cosmos"],
        "ARB": ["arb", "arbitrum"],
        "OP": ["op", "optimism"],
    }
    
    def extract_context(self, query: str) -> QueryContext:
        """Extract all relevant context from a user query"""
        q = query.lower()
        
        return QueryContext(
            capital=self._extract_capital(query),
            timeframe=self._extract_timeframe(q),
            trade_type=self._extract_trade_type(q),
            leverage=self._extract_leverage(q),
            risk_tolerance=self._extract_risk_tolerance(q),
            query_intent=self._extract_query_intent(q),
            specific_price=self._extract_specific_price(query),
            symbol=self._detect_symbol(q),
        )
    
    def _extract_capital(self, query: str) -> Optional[float]:
        """Extract capital amount from query"""
        patterns = [
            r"\$?([\d,]+)\s*(?:dollars?|usd)?",
            r"\$?([\d]+)k\b",
            r"\$?([\d]+)\s*(?:thousand|grand)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                amount = match.group(1).replace(",", "")
                if "k" in query.lower() and "000" not in amount:
                    return float(amount) * 1000
                return float(amount)
        return None
    
    def _extract_timeframe(self, q: str) -> Optional[str]:
        """Extract trading timeframe"""
        if any(x in q for x in ["scalp", "quick", "5 min", "15 min"]):
            return "scalp"
        if any(x in q for x in ["day trade", "daytrade", "intraday", "today"]):
            return "day"
        if any(x in q for x in ["swing", "few days", "this week"]):
            return "swing"
        if any(x in q for x in ["position", "weeks", "month"]):
            return "position"
        if any(x in q for x in ["long term", "longterm", "hodl", "invest", "hold"]):
            return "longterm"
        return None
    
    def _extract_trade_type(self, q: str) -> Optional[str]:
        """Extract spot vs futures"""
        if any(x in q for x in ["spot", "no leverage", "without leverage"]):
            return "spot"
        if any(x in q for x in ["futures", "perp", "perpetual", "leverage"]) or re.search(r"\d+x", q):
            return "futures"
        return None
    
    def _extract_leverage(self, q: str) -> Optional[int]:
        """Extract leverage amount"""
        match = re.search(r"(\d+)x\s*(?:leverage)?", q, re.IGNORECASE)
        if match:
            return int(match.group(1))
        if "max leverage" in q:
            return 20
        if "high leverage" in q:
            return 10
        if "low leverage" in q:
            return 3
        return None
    
    def _extract_risk_tolerance(self, q: str) -> Optional[str]:
        """Extract risk tolerance level"""
        if any(x in q for x in ["conservative", "safe", "low risk", "careful"]):
            return "conservative"
        if any(x in q for x in ["aggressive", "risky", "high risk", "yolo"]):
            return "aggressive"
        if any(x in q for x in ["moderate", "balanced", "normal risk"]):
            return "moderate"
        return None
    
    def _extract_query_intent(self, q: str) -> str:
        """Extract what the user wants to do"""
        if any(x in q for x in ["sell", "exit", "take profit", "close", "get out"]):
            return "exit"
        if any(x in q for x in ["dca", "dollar cost", "average in", "split entry"]):
            return "dca"
        if any(x in q for x in ["compare", " vs ", "versus", "which is better"]):
            return "compare"
        if any(x in q for x in ["buy", "entry", "long", "short", "where to"]):
            return "entry"
        return "analysis"
    
    def _extract_specific_price(self, query: str) -> Optional[float]:
        """Extract specific price level mentioned"""
        match = re.search(
            r"(?:at|below|above|around|near)\s*\$?([\d,]+)k?",
            query,
            re.IGNORECASE
        )
        if match:
            price = float(match.group(1).replace(",", ""))
            if "k" in query.lower():
                price *= 1000
            return price
        return None
    
    def _detect_symbol(self, q: str) -> str:
        """Detect which asset the user is asking about"""
        for symbol, patterns in self.SYMBOL_PATTERNS.items():
            if any(p in q for p in patterns):
                return symbol
        
        # Check for BTC explicitly or default
        if "btc" in q or "bitcoin" in q:
            return "BTC"
        
        return "BTC"  # Default
    
    def build_context_section(self, context: QueryContext) -> str:
        """Build the context section for the system prompt"""
        lines = []
        
        if context.capital:
            lines.append(
                f"USER CAPITAL: ${context.capital:,.0f} - "
                "Calculate ALL position sizes based on this."
            )
        
        if context.timeframe:
            timeframe_map = {
                "scalp": "SCALP TRADE (minutes to hours) - Use tight stops, small targets",
                "day": "DAY TRADE (hours, close by EOD) - Intraday levels only",
                "swing": "SWING TRADE (days to 1-2 weeks) - Use daily levels",
                "position": "POSITION TRADE (weeks to months) - Use weekly levels",
                "longterm": "LONG-TERM HOLD (months+) - Focus on accumulation zones, DCA levels",
            }
            lines.append(f"TIMEFRAME: {timeframe_map[context.timeframe]}")
        
        if context.trade_type:
            leverage_str = f" with {context.leverage}x leverage" if context.leverage else ""
            lines.append(f"TRADE TYPE: {context.trade_type.upper()}{leverage_str}")
        
        if context.leverage and context.leverage > 1:
            lines.append(
                f"⚠️ LEVERAGE WARNING: {context.leverage}x increases liquidation risk. "
                "Adjust stops accordingly."
            )
        
        if context.risk_tolerance:
            risk_map = {
                "conservative": "CONSERVATIVE - Use 1% risk per trade, wider stops, fewer trades",
                "moderate": "MODERATE - Use 2% risk per trade, standard position sizing",
                "aggressive": "AGGRESSIVE - Use 3-5% risk per trade, but warn about drawdown risk",
            }
            lines.append(f"RISK TOLERANCE: {risk_map[context.risk_tolerance]}")
        
        if context.query_intent == "exit":
            lines.append(
                "QUERY TYPE: EXIT/SELL - Focus on take-profit levels and exit strategy, not entry."
            )
        elif context.query_intent == "dca":
            lines.append(
                "QUERY TYPE: DCA - Provide multiple accumulation zones with % allocation per zone."
            )
        
        if context.specific_price:
            lines.append(
                f"USER ASKS ABOUT PRICE: ${context.specific_price:,.0f} - "
                "Analyze if this is a good level."
            )
        
        return "\n".join(lines) if lines else ""









