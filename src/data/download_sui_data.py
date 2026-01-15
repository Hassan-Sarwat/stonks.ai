"""
Download SUI token historical data from DexPaprika using the official Python SDK.

This script downloads historical OHLCV data for specified tokens across multiple intervals.

Usage:
    # Install the SDK first
    pip install dexpaprika-sdk
    
    # Download single token with default intervals (1d, 1h, 15m, 1m)
    python download_sui_data_v3.py --token SUI
    
    # Download multiple tokens with default intervals
    python download_sui_data_v3.py --tokens SUI USDC CETUS
    
    # Download with specific intervals only
    python download_sui_data_v3.py --token SUI --intervals 1d 1h
    
    # Download multiple tokens with custom intervals
    python download_sui_data_v3.py --tokens SUI USDC USDT --intervals 1h 15m 1m
    
    # Specify custom date range
    python download_sui_data_v3.py --token SUI --start-date 2025-08-01
    
    # All options together
    python download_sui_data_v3.py --tokens SUI USDC --intervals 1d 1h --start-date 2025-06-08 --output-dir my_data
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from dexpaprika_fetcher import DexPaprikaFetcher, CETUS_POOLS


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Download SUI token historical data from DexPaprika',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single token, all default intervals (1d, 1h, 15m, 1m)
  %(prog)s --token SUI
  
  # Multiple tokens, all default intervals
  %(prog)s --tokens SUI USDC CETUS
  
  # Single token, specific intervals
  %(prog)s --token SUI --intervals 1d 1h
  
  # Multiple tokens, custom intervals and date
  %(prog)s --tokens SUI USDC --intervals 1h 15m --start-date 2025-08-01
  
  # All options
  %(prog)s --tokens SUI USDC USDT --intervals 1d 1h 15m 1m --start-date 2025-06-08 --output-dir data_raw
        """
    )
    
    # Token arguments (mutually exclusive - either --token or --tokens)
    token_group = parser.add_mutually_exclusive_group(required=True)
    token_group.add_argument(
        '--token',
        type=str,
        help='Single token symbol to download (e.g., SUI, USDC, CETUS)'
    )
    token_group.add_argument(
        '--tokens',
        nargs='+',
        type=str,
        help='Multiple token symbols to download (e.g., SUI USDC CETUS)'
    )
    
    # Interval arguments
    parser.add_argument(
        '--intervals',
        nargs='+',
        type=str,
        default=['1d', '1h', '15m', '1m'],
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '24h', '1d'],
        help='Time intervals to download (default: 1d 1h 15m 1m)'
    )
    
    # Date range arguments
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-06-08',
        help='Start date for historical data (YYYY-MM-DD, default: 2025-06-08 - post-Cetus exploit)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for historical data (YYYY-MM-DD, default: today)'
    )
    
    # Output directory
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data_raw',
        help='Output directory for downloaded data (default: data_raw)'
    )
    
    # Optional: max candles limit
    parser.add_argument(
        '--max-candles',
        type=int,
        default=None,
        help='Maximum candles to fetch per token per interval (default: unlimited)'
    )
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.end_date:
            datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        parser.error(f"Invalid date format: {e}. Use YYYY-MM-DD format.")
    
    return args


def main():
    """Main entry point for downloading SUI token data."""
    args = parse_arguments()
    
    # Determine which tokens to download
    if args.token:
        tokens = [args.token]
    else:
        tokens = args.tokens
    
    # Display configuration
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  STONKS.AI - SUI DATA DOWNLOADER                 ║
║                 Powered by DexPaprika Python SDK                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("📋 Configuration:")
    print(f"   Tokens: {', '.join(tokens)}")
    print(f"   Intervals: {', '.join(args.intervals)}")
    print(f"   Date range: {args.start_date} to {args.end_date or 'today'}")
    print(f"   Output directory: {args.output_dir}")
    if args.max_candles:
        print(f"   Max candles per interval: {args.max_candles:,}")
    print()
    
    # Initialize fetcher
    fetcher = DexPaprikaFetcher(output_dir=args.output_dir)
    
    # Track overall results
    total_success = 0
    total_failed = 0
    all_results = {}
    
    # Download each token with each interval
    for token in tokens:
        print(f"\n{'='*70}")
        print(f"🎯 DOWNLOADING DATA FOR {token.upper()}")
        print(f"{'='*70}")
        
        token_results = {}
        
        for interval in args.intervals:
            print(f"\n📊 Interval: {interval}")
            print(f"{'─'*70}")
            
            try:
                # Get pool address from known pools, or search if not found
                pool_address = "0x51e883ba7c0b566a26cbc8a94cd33eb0abd418a77cc1e60ad22fd9b1f29cd2ab"# CETUS_POOLS.get(token.upper())
                
                filepath = fetcher.download_token_data(
                    symbol=token,
                    pool_address=pool_address,  # Use known address or None to search
                    start_date=args.start_date,
                    end_date=args.end_date,
                    interval=interval,
                    max_candles=args.max_candles,
                )
                
                if filepath:
                    token_results[interval] = filepath
                    total_success += 1
                    print(f"✅ Successfully saved {interval} data for {token}")
                else:
                    total_failed += 1
                    print(f"❌ Failed to download {interval} data for {token}")
                
                # Rate limiting between intervals
                import time
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error downloading {interval} data for {token}: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1
        
        all_results[token] = token_results
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Total successful downloads: {total_success}")
    print(f"❌ Total failed downloads: {total_failed}")
    print(f"📁 Data saved to: {args.output_dir}/")
    
    # Detailed breakdown
    print(f"\n📋 Detailed breakdown:")
    for token, results in all_results.items():
        print(f"\n   {token.upper()}:")
        if results:
            for interval, filepath in results.items():
                print(f"      ✓ {interval}: {filepath}")
        else:
            print(f"      ✗ No data downloaded")
    
    # Next steps
    print(f"\n🎯 Next steps:")
    print(f"   1. Verify data: ls -lh {args.output_dir}/*/")
    print(f"   2. Check data quality: head {args.output_dir}/*/*_1h.csv")
    print(f"   3. Run backtests with your trading strategies")
    
    # Exit code based on results
    if total_success > 0:
        print(f"\n✅ SUCCESS! Downloaded {total_success} datasets")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: No data downloaded")
        sys.exit(1)


if __name__ == "__main__":
    main()