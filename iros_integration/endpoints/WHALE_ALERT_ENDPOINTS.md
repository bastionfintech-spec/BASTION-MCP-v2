# 🐋 Whale Alert Premium API - Complete Endpoint Reference

**REST Base URL:** `https://api.whale-alert.io/v1`  
**WebSocket URL:** `wss://ws.whale-alert.io`  
**Auth:** `api_key` query parameter  
**API Key:** `OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ` (pre-configured)  
**Plan:** $29.95/month (Pro Plan)  
**Docs:** https://whale-alert.io/documentation  

---

## 🌐 Supported Blockchains

| Blockchain | Symbol | Min Value |
|------------|--------|-----------|
| Bitcoin | btc | $100,000 |
| Ethereum | eth | $100,000 |
| Ripple | xrp | $100,000 |
| Tether | usdt | $100,000 |
| USDC | usdc | $100,000 |
| Binance Chain | bnb | $100,000 |
| Cardano | ada | $100,000 |
| Solana | sol | $100,000 |
| Avalanche | avax | $100,000 |
| Polygon | matic | $100,000 |
| Tron | trx | $100,000 |

---

## 📡 REST API Endpoints

### `/status`
**API Status** - Check rate limits and account info

**Parameters:**
- `api_key`: Your API key

```json
{
  "result": "success",
  "blockchain_count": 11,
  "transaction_count_24h": 847,
  "rate_limit_remaining": 98,
  "rate_limit_reset": 1706803260
}
```

---

### `/transactions`
**Get Recent Transactions** ⭐ PRIMARY ENDPOINT

**Parameters:**
- `api_key`: Your API key (required)
- `min_value`: Minimum USD value (default: 500000)
- `start`: Start timestamp (Unix)
- `end`: End timestamp (Unix)
- `cursor`: Pagination cursor
- `limit`: Max transactions (default: 100)

**Example Request:**
```
GET /transactions?api_key=YOUR_KEY&min_value=10000000&limit=50
```

**Response:**
```json
{
  "result": "success",
  "cursor": "abc123...",
  "count": 50,
  "transactions": [
    {
      "id": "5a1b2c3d4e5f",
      "blockchain": "bitcoin",
      "symbol": "btc",
      "transaction_type": "transfer",
      "hash": "abc123def456...",
      "timestamp": 1706803200,
      "amount": 500.25,
      "amount_usd": 47273625,
      "from": {
        "address": "bc1q...",
        "owner": "binance",
        "owner_type": "exchange"
      },
      "to": {
        "address": "bc1p...",
        "owner": "unknown",
        "owner_type": "unknown"
      }
    }
  ]
}
```

---

### `/transaction/{blockchain}/{hash}`
**Get Specific Transaction** - Lookup by hash

**Example:**
```
GET /transaction/bitcoin/abc123def456...?api_key=YOUR_KEY
```

```json
{
  "result": "success",
  "transaction": {
    "id": "5a1b2c3d4e5f",
    "blockchain": "bitcoin",
    "symbol": "btc",
    "amount": 500.25,
    "amount_usd": 47273625,
    "from": {...},
    "to": {...}
  }
}
```

---

## 🔌 WebSocket Streaming

### Connection
```
wss://ws.whale-alert.io
```

### Subscribe Message
```json
{
  "type": "subscribe",
  "api_key": "OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ",
  "min_value": 1000000,
  "symbols": ["btc", "eth"]
}
```

### Transaction Event
```json
{
  "type": "transaction",
  "transaction": {
    "id": "5a1b2c3d4e5f",
    "blockchain": "bitcoin",
    "symbol": "btc",
    "amount": 1250.5,
    "amount_usd": 118547500,
    "from": {
      "address": "bc1q...",
      "owner": "unknown",
      "owner_type": "unknown"
    },
    "to": {
      "address": "1A1z...",
      "owner": "binance",
      "owner_type": "exchange"
    },
    "timestamp": 1706803200
  }
}
```

---

## 🏷️ Owner Types

| Type | Description | Trading Signal |
|------|-------------|----------------|
| `exchange` | Known exchange wallet | Deposit = bearish, Withdraw = bullish |
| `unknown` | Unidentified wallet | Could be OTC, cold storage |
| `custodian` | Custody service | Institutional activity |
| `whale` | Known large holder | Watch for patterns |
| `smart_contract` | DeFi protocol | Liquidity movement |

---

## 📊 Transaction Types

| Type | Description |
|------|-------------|
| `transfer` | Standard wallet-to-wallet transfer |
| `mint` | New tokens created (stablecoins) |
| `burn` | Tokens destroyed |
| `lock` | Tokens locked in smart contract |
| `unlock` | Tokens released from contract |

---

## 🧪 Python Usage Examples

### Basic Transaction Fetch
```python
from iros_integration import WhaleAlertClient

whale = WhaleAlertClient()  # API key pre-configured!

# Get $10M+ transactions from last hour
txs = await whale.get_transactions(min_value=10_000_000)

for tx in txs.transactions:
    print(f"🐋 {tx.amount:,.2f} {tx.symbol.upper()} (${tx.amount_usd:,.0f})")
    print(f"   {tx.from_owner or 'Unknown'} → {tx.to_owner or 'Unknown'}")
```

### Exchange Flow Analysis
```python
# Calculate BTC exchange inflows/outflows over 24h
flows = await whale.get_exchange_flows(symbol="btc", hours=24)

print(f"BTC Exchange Flows (24h):")
print(f"  Inflows:  ${flows['inflows_usd']:,.0f}")
print(f"  Outflows: ${flows['outflows_usd']:,.0f}")
print(f"  Net Flow: ${flows['net_flow_usd']:,.0f}")  # Positive = leaving exchanges (bullish)

# Breakdown by exchange
for exchange, volume in flows['exchange_breakdown'].items():
    print(f"  {exchange}: ${volume:,.0f}")
```

### Real-Time Streaming
```python
async def handle_whale_tx(tx):
    """Called for each whale transaction"""
    formatted = whale.format_transaction(tx)
    print(formatted)
    
    # Alert on major exchange movements
    if tx.amount_usd > 50_000_000:
        if tx.to_owner_type == "exchange":
            print("⚠️ MAJOR DEPOSIT - Potential sell pressure!")
        elif tx.from_owner_type == "exchange":
            print("✅ MAJOR WITHDRAW - Bullish signal!")

# Start streaming $5M+ transactions
await whale.stream_transactions(
    callback=handle_whale_tx,
    min_value=5_000_000,
    symbols=["btc", "eth"]
)
```

### Stablecoin Monitoring
```python
# Monitor stablecoin mints/burns
txs = await whale.get_transactions(min_value=100_000_000)

mints = [t for t in txs.transactions if t.transaction_type == "mint" and t.symbol in ["usdt", "usdc"]]
burns = [t for t in txs.transactions if t.transaction_type == "burn" and t.symbol in ["usdt", "usdc"]]

total_minted = sum(t.amount_usd for t in mints)
total_burned = sum(t.amount_usd for t in burns)

print(f"Stablecoin Activity:")
print(f"  Minted: ${total_minted:,.0f}")
print(f"  Burned: ${total_burned:,.0f}")
print(f"  Net: ${total_minted - total_burned:,.0f}")  # Positive = new capital entering
```

---

## 🎯 Trading Signals

### Bullish Signals
1. **Large exchange withdrawals** - BTC/ETH leaving exchanges
2. **Stablecoin mints** - New capital entering crypto
3. **Unknown → Unknown transfers** - Potential accumulation
4. **OTC activity** - Institutional buying

### Bearish Signals
1. **Large exchange deposits** - Potential selling pressure
2. **Stablecoin burns** - Capital exiting crypto
3. **Whale → Exchange transfers** - Distribution
4. **Multiple deposits in short time** - Coordinated selling

---

## 🔄 Combining with Other Data

### With Helsinki + Coinglass
```python
from iros_integration import HelsinkiClient, CoinglassClient, WhaleAlertClient

helsinki = HelsinkiClient()
coinglass = CoinglassClient()
whale = WhaleAlertClient()

# Get complete market picture
async def get_full_analysis(symbol: str):
    # Price & technicals from Helsinki
    market = await helsinki.fetch_full_data(symbol)
    
    # Liquidation levels from Coinglass
    liqs = await coinglass.get_liquidation_map(symbol)
    
    # Recent whale movements
    txs = await whale.get_transactions(min_value=10_000_000)
    symbol_txs = [t for t in txs.transactions if t.symbol.lower() == symbol.lower()]
    
    # Analyze
    exchange_deposits = sum(
        t.amount_usd for t in symbol_txs 
        if t.to_owner_type == "exchange"
    )
    exchange_withdrawals = sum(
        t.amount_usd for t in symbol_txs 
        if t.from_owner_type == "exchange"
    )
    
    return {
        "price": market.price,
        "smart_money_bias": market.smart_money.get("signal"),
        "liquidation_bias": liqs.data.get("cascade_bias") if liqs.success else None,
        "whale_net_flow": exchange_withdrawals - exchange_deposits,
        "whale_signal": "BULLISH" if exchange_withdrawals > exchange_deposits else "BEARISH"
    }
```

---

## ⚙️ Rate Limits & Best Practices

### Pro Plan Limits
- 10 requests/minute REST API
- 1 WebSocket connection
- Historical data: 30 days

### Best Practices
1. **Use WebSocket for real-time** - Don't poll REST constantly
2. **Cache REST results** - Minimum 1 minute
3. **Filter by min_value** - Start high ($10M+) to reduce noise
4. **Use exchange_flows helper** - Pre-computed analytics

### Alert Thresholds by Asset
| Asset | Minor | Major | Massive |
|-------|-------|-------|---------|
| BTC | $10M | $50M | $100M+ |
| ETH | $5M | $25M | $50M+ |
| Stables | $50M | $100M | $500M+ |

---

**Whale Alert - See what the big players are doing 🐋**









