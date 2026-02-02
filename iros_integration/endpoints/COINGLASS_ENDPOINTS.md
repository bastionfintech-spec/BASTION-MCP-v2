# 📊 Coinglass Premium API - Complete Endpoint Reference

**Base URL:** `https://open-api-v3.coinglass.com/api`  
**Auth:** `CG-API-KEY` header  
**API Key:** `03e5a43afaa4489384cb935b9b2ea16b` (pre-configured)  
**Plan:** $299/month (Whale Plan)  
**Docs:** https://coinglass.com/api  

---

## 🔥 Liquidation Endpoints

### `/futures/liquidation/aggregated-history`
**Aggregated Liquidation History** - Time-series liquidation data

**Parameters:**
- `symbol`: BTC, ETH, SOL, etc.
- `interval`: h1, h4, h12, h24
- `limit`: Number of records (max 500)

```json
{
  "code": "0",
  "msg": "success",
  "data": [
    {
      "time": 1706803200000,
      "longLiquidationUsd": 12500000,
      "shortLiquidationUsd": 8900000,
      "longLiquidationVolume": 132.5,
      "shortLiquidationVolume": 94.2
    }
  ]
}
```

### `/futures/liquidation/heatmap`
**Liquidation Heatmap** - Cluster visualization data

**Parameters:**
- `symbol`: BTC, ETH, etc.

```json
{
  "code": "0",
  "data": {
    "levels": [
      {"price": 91000, "longLiquidation": 145000000, "shortLiquidation": 0},
      {"price": 98000, "longLiquidation": 0, "shortLiquidation": 89000000}
    ],
    "currentPrice": 94500
  }
}
```

### `/futures/liquidation/exchange-list`
**Liquidations by Exchange** - Real-time per-exchange data

```json
{
  "data": [
    {"exchange": "Binance", "h24LongLiqUsd": 45000000, "h24ShortLiqUsd": 32000000},
    {"exchange": "OKX", "h24LongLiqUsd": 23000000, "h24ShortLiqUsd": 18000000}
  ]
}
```

---

## 📈 Open Interest Endpoints

### `/futures/openInterest/exchange-list`
**OI by Exchange** - Current open interest breakdown

**Parameters:**
- `symbol`: BTC, ETH, etc.

```json
{
  "data": [
    {
      "exchange": "Binance",
      "openInterest": 28500000000,
      "openInterestAmount": 301234.5,
      "h24Change": 2.34
    },
    {
      "exchange": "CME",
      "openInterest": 8900000000,
      "h24Change": -1.2
    }
  ]
}
```

### `/futures/openInterest/ohlc-aggregated-history`
**OI OHLC History** - Open interest time series

**Parameters:**
- `symbol`: BTC, ETH, etc.
- `interval`: h1, h4, h12, h24
- `limit`: Number of records

```json
{
  "data": [
    {
      "time": 1706803200000,
      "open": 28100000000,
      "high": 28600000000,
      "low": 27900000000,
      "close": 28500000000
    }
  ]
}
```

### `/futures/openInterest/ohlc-history`
**OI History by Exchange** - Per-exchange OI time series

---

## 💰 Funding Rate Endpoints

### `/futures/fundingRate/exchange-list`
**Current Funding Rates** - All exchanges

**Parameters:**
- `symbol`: BTC, ETH, etc.

```json
{
  "data": [
    {
      "exchange": "Binance",
      "symbol": "BTCUSDT",
      "fundingRate": 0.0001,
      "nextFundingTime": 1706832000000
    },
    {
      "exchange": "dYdX",
      "symbol": "BTC-USD",
      "fundingRate": 0.000125
    }
  ]
}
```

### `/futures/fundingRate/history`
**Funding Rate History** - Historical rates

**Parameters:**
- `symbol`: BTC, ETH, etc.
- `limit`: Number of records

```json
{
  "data": [
    {
      "time": 1706803200000,
      "avgFundingRate": 0.0001,
      "openInterest": 28500000000
    }
  ]
}
```

### `/futures/fundingRate/oi-weight`
**OI-Weighted Funding** - True aggregate funding

---

## 📊 Long/Short Ratio Endpoints

### `/futures/globalLongShortAccountRatio/history`
**Global L/S Account Ratio** - Retail sentiment

**Parameters:**
- `symbol`: BTC, ETH, etc.

```json
{
  "data": [
    {
      "time": 1706803200000,
      "longAccount": 52.3,
      "shortAccount": 47.7,
      "longShortRatio": 1.10
    }
  ]
}
```

### `/futures/topLongShortAccountRatio/history`
**Top Trader L/S Ratio** - Whale sentiment

```json
{
  "data": [
    {
      "time": 1706803200000,
      "longAccount": 58.2,
      "shortAccount": 41.8,
      "longShortRatio": 1.39
    }
  ]
}
```

### `/futures/topLongShortPositionRatio/history`
**Top Trader Position Ratio** - Size-weighted

---

## 🎯 Options Endpoints

### `/options/info`
**Options Overview** - Put/call ratio, max pain

```json
{
  "data": {
    "symbol": "BTC",
    "putCallRatio": 0.78,
    "maxPainPrice": 95000,
    "totalOpenInterest": 12500000000,
    "totalVolume24h": 890000000
  }
}
```

### `/options/openInterest/history`
**Options OI History**

### `/options/volume/history`
**Options Volume History**

---

## 📉 Grayscale & ETF Endpoints

### `/index/gbtc`
**GBTC Holdings** - Grayscale Bitcoin Trust

```json
{
  "data": {
    "totalBtc": 215000,
    "totalUsd": 20350000000,
    "premium": -2.34,
    "h24Change": 0.5
  }
}
```

### `/index/bitcoin-etf`
**Bitcoin ETF Flows** - All spot ETFs

```json
{
  "data": [
    {"name": "IBIT", "totalBtc": 285000, "h24Flow": 2340},
    {"name": "FBTC", "totalBtc": 175000, "h24Flow": 890}
  ]
}
```

---

## 🔄 Market Data Endpoints

### `/futures/price`
**Futures Prices** - All exchanges

### `/futures/vol`
**Volume Data** - 24h volumes

### `/futures/taker-buy-sell-ratio`
**Taker Buy/Sell Ratio** - Order flow

```json
{
  "data": {
    "symbol": "BTC",
    "buyRatio": 0.52,
    "sellRatio": 0.48,
    "netFlow": "BUY"
  }
}
```

---

## 🧪 Python Usage Examples

### Get Liquidation Data
```python
from iros_integration import CoinglassClient

cg = CoinglassClient()  # API key pre-configured!

# Recent liquidations
liqs = await cg.get_liquidation_history("BTC", interval="h1", limit=24)
print(f"Last 24h long liquidations: ${sum(l['longLiquidationUsd'] for l in liqs.data):,.0f}")

# Liquidation heatmap
heatmap = await cg.get_liquidation_map("BTC")
```

### Get Open Interest
```python
# Current OI by exchange
oi = await cg.get_open_interest("ETH")
for exchange in oi.data:
    print(f"{exchange['exchange']}: ${exchange['openInterest']:,.0f}")

# OI history
oi_hist = await cg.get_open_interest_history("BTC", interval="h4", limit=100)
```

### Get Funding Rates
```python
# Current rates
funding = await cg.get_funding_rates("BTC")
for rate in funding.data:
    print(f"{rate['exchange']}: {rate['fundingRate']*100:.4f}%")

# Historical rates
hist = await cg.get_funding_history("BTC", limit=200)
```

### Get Market Overview (All Data)
```python
# Fetches OI, funding, L/S ratio, and liquidations in parallel
overview = await cg.get_market_overview("BTC")
print(f"Open Interest: {overview['open_interest']}")
print(f"Funding Rates: {overview['funding_rates']}")
print(f"Long/Short: {overview['long_short_ratio']}")
```

---

## 🎯 Best Practices

### Rate Limits
- 100 requests/minute on Whale plan
- Use batch endpoints where possible
- Cache data for 1 minute minimum

### Key Endpoints for Trading
1. `/futures/liquidation/heatmap` - Find liquidation clusters
2. `/futures/fundingRate/exchange-list` - Spot arbitrage opportunities
3. `/futures/topLongShortAccountRatio/history` - Smart money positioning
4. `/futures/openInterest/exchange-list` - Track institutional activity

### Combining with Helsinki
```python
from iros_integration import HelsinkiClient, CoinglassClient

helsinki = HelsinkiClient()
coinglass = CoinglassClient()

# Helsinki: Quick overview + price
helsinki_data = await helsinki.fetch_full_data("BTC")

# Coinglass: Deep liquidation analysis
liq_data = await coinglass.get_liquidation_map("BTC")

# Combine for complete picture
```

---

**Coinglass - Real liquidation data that moves markets 🔥**







