"""Clean bad reports and regenerate quality reports"""
import asyncio
import os
import sys

# Set model URL
os.environ["BASTION_MODEL_URL"] = "https://7dbfac97c5ec.tail2a2463.ts.net"

from iros_integration.services.coinglass import CoinglassClient
from mcf_labs.iros_generator import create_iros_generator, SUPPORTED_COINS
from mcf_labs.storage import get_storage

async def clean_and_regenerate():
    coinglass = CoinglassClient()
    storage = get_storage()
    
    # Step 1: Clear bad reports
    print("=" * 60)
    print("STEP 1: CLEARING BAD REPORTS")
    print("=" * 60)
    
    deleted = storage.clear_bad_reports()
    print(f"Deleted {deleted} bad reports")
    
    # Show remaining reports
    remaining = storage.list_reports(limit=100)
    print(f"Remaining reports: {len(remaining)}")
    
    # Step 2: Create generator (without IROS for now to test data quality)
    print("\n" + "=" * 60)
    print("STEP 2: TESTING REPORT GENERATION WITH FIXED PARSING")
    print("=" * 60)
    
    # First test without IROS to verify data quality
    from mcf_labs.generator import ReportGenerator
    gen = ReportGenerator(coinglass)
    
    # Test Market Structure for BTC
    print("\n--- Testing BTC Market Structure ---")
    try:
        report = await gen.generate_market_structure("BTC")
        print(f"SUCCESS: {report.title}")
        print(f"  Price: ${report.sections['key_levels']['current_price']:,.0f}")
        print(f"  Whale Bias: {report.sections['whale_positioning']['net_bias']}")
        print(f"  Long: ${report.sections['whale_positioning']['total_long_usd']/1e6:.1f}M")
        print(f"  Short: ${report.sections['whale_positioning']['total_short_usd']/1e6:.1f}M")
        print(f"  Summary: {report.summary[:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test Whale Report for BTC
    print("\n--- Testing BTC Whale Report ---")
    try:
        report = await gen.generate_whale_report("BTC")
        print(f"SUCCESS: {report.title}")
        print(f"  Summary: {report.summary[:100]}...")
        positions = report.sections.get("top_positions", [])
        print(f"  Positions: {len(positions)}")
        if positions:
            p = positions[0]
            print(f"  Top: {p['side']} ${p['size_usd']/1e6:.1f}M @ ${p['entry_price']:,.0f}")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test Cycle Report
    print("\n--- Testing BTC Cycle Report ---")
    try:
        report = await gen.generate_cycle_report("BTC")
        print(f"SUCCESS: {report.title}")
        print(f"  Summary: {report.summary[:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING QUALITY REPORTS WITH IROS")
    print("=" * 60)
    
    # Create IROS generator
    generator = create_iros_generator(
        coinglass_client=coinglass,
        model_url=os.environ.get("BASTION_MODEL_URL")
    )
    
    # Generate Market Structure for top coins
    successful = 0
    failed = 0
    
    for symbol in ["BTC", "ETH", "SOL", "BNB", "XRP"]:
        try:
            print(f"\nGenerating Market Structure for {symbol}...", end=" ")
            report = await generator.generate_market_structure(symbol)
            
            # Validate before saving
            price = report.sections["key_levels"]["current_price"]
            if price > 0 and "UNKNOWN" not in report.summary:
                storage.save_report(report)
                print(f"OK - ${price:,.0f}")
                successful += 1
            else:
                print("SKIPPED - invalid data")
                failed += 1
        except Exception as e:
            print(f"FAILED - {e}")
            failed += 1
        await asyncio.sleep(2)
    
    # Generate Whale reports for coins with good data
    print("\n--- Generating Whale Reports ---")
    for symbol in ["BTC", "ETH", "SOL"]:
        try:
            print(f"Generating Whale Report for {symbol}...", end=" ")
            report = await generator.generate_whale_report(symbol)
            
            # Validate
            positions = report.sections.get("top_positions", [])
            valid = [p for p in positions if p.get("size_usd", 0) > 0]
            
            if len(valid) >= 3 and "UNKNOWN" not in report.summary:
                storage.save_report(report)
                print(f"OK - {len(valid)} positions")
                successful += 1
            else:
                print(f"SKIPPED - only {len(valid)} valid positions")
                failed += 1
        except ValueError as e:
            print(f"SKIPPED - {e}")
            failed += 1
        except Exception as e:
            print(f"FAILED - {e}")
            failed += 1
        await asyncio.sleep(2)
    
    # Generate Cycle report
    print("\n--- Generating Cycle Report ---")
    try:
        print("Generating BTC Cycle Report...", end=" ")
        report = await generator.generate_cycle_report("BTC")
        storage.save_report(report)
        print(f"OK - {report.title}")
        successful += 1
    except Exception as e:
        print(f"FAILED - {e}")
        failed += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed/Skipped: {failed}")
    
    all_reports = storage.list_reports(limit=100)
    print(f"\nTotal reports in storage: {len(all_reports)}")
    
    # Group by type
    by_type = {}
    for r in all_reports:
        t = r.type.value
        by_type[t] = by_type.get(t, 0) + 1
    
    print("\nBreakdown by type:")
    for t, count in by_type.items():
        print(f"  - {t}: {count}")
    
    print("\nRecent reports:")
    for r in all_reports[:5]:
        print(f"  - [{r.type.value}] {r.title}")
        print(f"    {r.summary[:80]}...")

if __name__ == "__main__":
    asyncio.run(clean_and_regenerate())

