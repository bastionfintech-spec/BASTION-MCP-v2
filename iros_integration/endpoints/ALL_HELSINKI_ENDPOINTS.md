# 📊 Helsinki VM - Complete Endpoint Reference

**Base URL:** `http://77.42.29.188:5002`  
**Auth:** None required (free!)  
**Rate Limit:** Unlimited  

---

## 🔄 Symbol-Specific Endpoints

Replace `{symbol}` with: `BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `ADA`, `DOGE`, `AVAX`, `LINK`, etc.

---

### Order Flow Endpoints

#### `/quant/cvd/{symbol}`
**Cumulative Volume Delta** - Measures net buying/selling pressure

```json
{
  "symbol": "BTCUSDT",
  "current_price": 94500,
  "cvd_1h": 1234.567,
  "cvd_4h": 5678.901,
  "cvd_total": 12345.678,
  "cvd_1h_usd": 116500000,
  "price_change_1h_pct": 0.45,
  "divergence": "BULLISH_DIVERGENCE",
  "interpretation": "Price down but buying pressure increasing",
  "signal": "BUY"
}
```

#### `/quant/orderbook/{symbol}`
**Level 2 Order Book Analysis**

```json
{
  "best_bid": 94520.00,
  "best_ask": 94525.00,
  "spread": 5.00,
  "spread_pct": 0.0053,
  "buy_pressure": 0.62,
  "bid_volume_10": 45.234,
  "ask_volume_10": 28.123,
  "imbalance_ratio": 1.61
}
```

#### `/quant/large-trades/{symbol}`
**Whale Transaction Detection**

```json
{
  "large_buy_volume_1h": 234.5,
  "large_sell_volume_1h": 189.2,
  "net_large_flow": 45.3,
  "whale_buy_ratio": 0.55,
  "significant_trades": [
    {"size": 50.2, "side": "buy", "price": 94500, "time": "2026-01-07T12:30:00Z"}
  ]
}
```

#### `/quant/smart-money/{symbol}`
**Institutional Flow Indicators**

```json
{
  "smart_money_bias": "BULLISH",
  "divergence": 0.23,
  "top_trader_ls_ratio": 1.45,
  "trend": "ACCUMULATING",
  "interpretation": "Institutions are buying the dip"
}
```

#### `/quant/whale-flow/{symbol}`
**Large Wallet Movement Tracking**

```json
{
  "inflow_24h": 2345.67,
  "outflow_24h": 1890.12,
  "net_flow": 455.55,
  "exchange_netflow": -234.5,
  "whale_wallets_active": 47
}
```

---

### Derivatives Endpoints

#### `/quant/basis/{symbol}`
**Spot/Futures Basis Analysis**

```json
{
  "spot_price": 94500,
  "futures_price": 94650,
  "basis_percent": 0.16,
  "structure": "contango",
  "annualized_yield": 5.84,
  "interpretation": "Moderate bullish positioning"
}
```

#### `/quant/open-interest/{symbol}`
**Open Interest Analysis**

```json
{
  "oi_value": 28500000000,
  "oi_change_24h": 2.34,
  "trend": "increasing",
  "concentration": "distributed",
  "interpretation": "New money entering the market"
}
```

#### `/quant/greeks/{symbol}`
**Options Greeks & Sentiment**

```json
{
  "symbol": "BTC",
  "underlying_price": 94500,
  "atm_implied_vol_pct": 58.5,
  "put_call_ratio": 0.78,
  "options_sentiment": "BULLISH",
  "note": "Low put/call suggests call buying dominance"
}
```

#### `/quant/liquidation-map/{symbol}`
**Liquidation Cluster Mapping**

```json
{
  "current_price": 94500,
  "upside_liquidation_levels": [
    {"price": 96000, "volume": 45000000},
    {"price": 98000, "volume": 120000000}
  ],
  "downside_liquidation_levels": [
    {"price": 92000, "volume": 89000000},
    {"price": 90000, "volume": 234000000}
  ]
}
```

#### `/quant/liquidation-estimate/{symbol}`
**Estimated Liquidation Cascade Zones** ⭐ CRITICAL ENDPOINT

```json
{
  "current_price": 94500,
  "open_interest_usd": 28500000000,
  "long_short_ratio": 1.23,
  "cascade_bias": "DOWNSIDE",
  "downside_liquidation_zones": [
    {
      "price": 91800,
      "distance_pct": -2.86,
      "estimated_usd_at_risk": 847000000,
      "leverage_concentration": "10-25x"
    }
  ],
  "upside_liquidation_zones": [
    {
      "price": 97200,
      "distance_pct": 2.86,
      "estimated_usd_at_risk": 234000000,
      "leverage_concentration": "10-20x"
    }
  ]
}
```

#### `/quant/options-iv/{symbol}`
**Options Implied Volatility** ⭐ BEST SOURCE FOR CURRENT PRICE

```json
{
  "symbol": "BTC",
  "underlying_price": 94523.45,
  "atm_implied_volatility_pct": 48.5,
  "put_call_ratio": 0.82,
  "skew_interpretation": "NEUTRAL",
  "term_structure": "contango"
}
```

---

### Volatility Endpoints

#### `/quant/iv-rv-spread/{symbol}`
**Implied vs Realized Volatility**

```json
{
  "implied_volatility_pct": 48.5,
  "realized_volatility_pct": 32.1,
  "spread_pct": 16.4,
  "interpretation": "Options expensive relative to realized moves",
  "strategy_suggestion": "Consider selling premium"
}
```

#### `/quant/volatility/{symbol}`
**Volatility Regime Detection**

```json
{
  "current_regime": "NORMAL",
  "confidence": 85,
  "volatility_7d_pct": 42.5,
  "volatility_30d_pct": 38.2,
  "volatility_percentile": 58,
  "thresholds": {
    "low": 25,
    "high": 75
  },
  "position_size_multiplier": 1.0
}
```

---

### Technical Endpoints

#### `/quant/vwap/{symbol}`
**Volume-Weighted Average Price**

```json
{
  "symbol": "BTCUSDT",
  "current_price": 94500,
  "vwap": 94123.45,
  "vwap_upper_1sd": 95234.56,
  "vwap_lower_1sd": 93012.34,
  "distance_from_vwap_pct": 0.40,
  "position": "ABOVE_VWAP",
  "interpretation": "Trading above fair value"
}
```

#### `/quant/momentum/{symbol}`
**Momentum Score & Indicators**

```json
{
  "symbol": "BTCUSDT",
  "current_price": 94500,
  "momentum_score": 72,
  "momentum": "BULLISH",
  "interpretation": "Strong upward momentum",
  "roc": {
    "7d_pct": 3.45,
    "14d_pct": 5.67,
    "30d_pct": 12.34
  },
  "rsi_14": 62.5,
  "atr_14": 1823.45,
  "atr_pct": 1.93
}
```

#### `/quant/mean-reversion/{symbol}`
**Mean Reversion Signals**

```json
{
  "symbol": "BTCUSDT",
  "current_price": 94500,
  "z_scores": {
    "7d": 1.23,
    "14d": 0.89,
    "30d": 0.45
  },
  "primary_z_score": 1.23,
  "signal": "NEUTRAL",
  "interpretation": "Within normal range",
  "action": "HOLD",
  "bollinger_bands": {
    "upper": 96500,
    "middle": 94000,
    "lower": 91500,
    "position": "MIDDLE"
  }
}
```

#### `/quant/drawdown/{symbol}`
**Drawdown Analysis**

```json
{
  "symbol": "BTCUSDT",
  "current_price": 94500,
  "period_high": 108000,
  "current_drawdown_pct": -12.5,
  "max_drawdown_pct": -21.3,
  "max_drawdown_date": "2024-08-05",
  "recovery_assessment": "In recovery phase",
  "risk_note": "Within normal correction range"
}
```

---

### Full Context Endpoint

#### `/quant/full/{symbol}` ⭐ MOST IMPORTANT ENDPOINT

Returns ALL symbol data in one call!

```json
{
  "symbol": "BTC",
  "price": 94500,
  "volatility": {
    "regime": "NORMAL",
    "percentile": 58,
    "atr_14": 1823.45
  },
  "liquidation": {
    "open_interest_usd": 28500000000,
    "long_short_ratio": 1.23,
    "cascade_bias": "DOWNSIDE",
    "upside_liquidation_zones": [...],
    "downside_liquidation_zones": [...]
  },
  "smart_money": {
    "signal": "BULLISH",
    "whale_buy_ratio": 0.68
  },
  "funding_arb": {
    "opportunities": [...],
    "all_rates": {...}
  },
  "options_iv": {
    "underlying_price": 94523.45,
    "atm_implied_volatility_pct": 48.5
  }
}
```

---

## 📊 Static Endpoints (No Symbol Required)

---

### Derivatives

#### `/derivatives/funding`
Multi-exchange funding rates

#### `/derivatives/oi`
Open interest breakdown by exchange

#### `/derivatives/long-short`
Long/short ratios across exchanges

#### `/derivatives/basis`
Basis comparison across exchanges

#### `/quant/funding-arb`
Cross-exchange funding arbitrage opportunities

---

### Macro

#### `/quant/dominance`
**Market Dominance**

```json
{
  "btc_dominance_pct": 52.3,
  "eth_dominance_pct": 17.8,
  "others_dominance_pct": 29.9,
  "total_market_cap_usd": 3200000000000,
  "alt_season_score": 35,
  "season": "BTC_SEASON",
  "strategy": "Focus on large caps"
}
```

#### `/quant/defi-tvl`
**DeFi Total Value Locked**

```json
{
  "total_defi_tvl_usd": 180000000000,
  "ethereum_tvl_usd": 90000000000,
  "ethereum_dominance_pct": 49.9,
  "top_protocols": [
    {"name": "Lido", "tvl_usd": 35000000000, "category": "Liquid Staking"},
    {"name": "AAVE", "tvl_usd": 18000000000, "category": "Lending"}
  ]
}
```

#### `/quant/gas`
**Network Gas Fees**

```json
{
  "ethereum": {
    "safe_gwei": 15,
    "standard_gwei": 22,
    "fast_gwei": 35,
    "congestion": "LOW"
  },
  "bitcoin": {
    "slow_sat_vb": 8,
    "medium_sat_vb": 15,
    "fast_sat_vb": 28,
    "congestion": "NORMAL",
    "estimated_transfer_usd": 1.25
  }
}
```

#### `/quant/stablecoin-supply`
Stablecoin supply tracking (USDT, USDC, DAI)

---

### Sentiment

#### `/sentiment/fear-greed`
**Fear & Greed Index**

```json
{
  "value": 72,
  "label": "greed",
  "classification": "Greed",
  "trading_implication": "Market may be overextended, consider profit-taking",
  "trend": "increasing",
  "yesterday": 68,
  "last_week": 55
}
```

#### `/sentiment/stablecoin-dominance`
Stablecoin dominance trend

---

### Options

#### `/options/skew`
Options skew across strikes

---

### Full Context

#### `/context/full`
Complete market context in one call

---

## 🧪 Testing Endpoints

### Health Check
```bash
curl http://77.42.29.188:5002/health
```

### Test Full Data
```bash
curl http://77.42.29.188:5002/quant/full/BTC
```

### Test Options IV (Best Price Source)
```bash
curl http://77.42.29.188:5002/quant/options-iv/BTC
```

### Test Fear & Greed
```bash
curl http://77.42.29.188:5002/sentiment/fear-greed
```

---

## 📈 Recommended Usage

### Priority Endpoints (Fast - 5 calls)
For quick analysis, use:
1. `/quant/full/{symbol}` - All symbol data
2. `/quant/options-iv/{symbol}` - Reliable current price
3. `/sentiment/fear-greed` - Market sentiment
4. `/quant/dominance` - Market structure
5. `/quant/gas` - Network activity

### Comprehensive Endpoints (Slow - 33 calls)
For deep analysis, fetch all endpoints in parallel.

---

**Helsinki VM - Free institutional-grade quant data 📊**












