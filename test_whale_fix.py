"""Test the whale position parsing fix"""
import asyncio
from iros_integration.services.coinglass import CoinglassClient
from mcf_labs.generator import ReportGenerator

async def test_whale_parsing():
    client = CoinglassClient()
    gen = ReportGenerator(client)
    
    # Test whale position parsing
    print("=== Testing Whale Position Parsing ===")
    result = await client.get_hyperliquid_whale_positions("BTC")
    
    positions = gen._parse_whale_positions(result, "BTC")
    print(f"Parsed {len(positions)} BTC positions")
    
    if positions:
        print("\nTop 3 positions:")
        for p in positions[:3]:
            side = p["side"]
            size_m = p["size_usd"] / 1e6
            entry = p["entry_price"]
            lev = p["leverage"]
            pnl_m = p["pnl_usd"] / 1e6
            print(f"  #{p['rank']} {side} ${size_m:.1f}M @ ${entry:,.0f} ({lev}x) PnL: ${pnl_m:.2f}M")
    
    # Test whale analysis
    print("\n=== Whale Analysis ===")
    analysis = gen._analyze_whales(result, "BTC")
    print(f"Net Bias: {analysis['net_bias']}")
    print(f"Total Long: ${analysis['total_long_usd']/1e6:.1f}M")
    print(f"Total Short: ${analysis['total_short_usd']/1e6:.1f}M")
    print(f"Position Count: {analysis['position_count']}")
    
    # Test ETH
    print("\n=== ETH Positions ===")
    eth_positions = gen._parse_whale_positions(result, "ETH")
    print(f"Parsed {len(eth_positions)} ETH positions")
    if eth_positions:
        print("\nTop 3 ETH positions:")
        for p in eth_positions[:3]:
            side = p["side"]
            size_m = p["size_usd"] / 1e6
            pnl_m = p["pnl_usd"] / 1e6
            print(f"  #{p['rank']} {side} ${size_m:.1f}M PnL: ${pnl_m:.2f}M")

if __name__ == "__main__":
    asyncio.run(test_whale_parsing())

