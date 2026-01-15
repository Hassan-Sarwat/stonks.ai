"""
DexPaprika Data Fetcher for SUI Chain (SDK Version)

Fetches historical OHLCV data from DexPaprika using the official Python SDK.
Converts data to Binance-compatible format for stonks_ai backtesting.

Author: Claude (Stonks AI)
Date: January 15, 2026
SDK: dexpaprika-sdk (pip install dexpaprika-sdk)
API: https://api.dexpaprika.com (FREE, no key required)
"""

from dexpaprika_sdk import DexPaprikaClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path


# Known Cetus pool addresses on SUI (verified from GeckoTerminal)
# These are the actual on-chain pool contract addresses
CETUS_POOLS = {
    # Main liquidity pools (Source: GeckoTerminal, Jan 2026)
    "SUI": "0xcf994611fd4c48e277ce3ffd4d4364c914af2c3cbb05f7bf6facd371de688630",    # SUI/USDC pool ($261K liquidity)
    "USDC": "0xcf994611fd4c48e277ce3ffd4d4364c914af2c3cbb05f7bf6facd371de688630",   # SUI/USDC pool (same as SUI)
    "USDT": "0xcf994611fd4c48e277ce3ffd4d4364c914af2c3cbb05f7bf6facd371de688630",   # Using SUI/USDC as proxy (USDT pools less liquid)
    "CETUS": "0x2e041f3fd93646dcc877f783c1f2b7fa62d30271bdef1f21ef002cebf857bded",  # CETUS/SUI pool ($839K liquidity)
    
    # Note: These tokens need pool addresses to be found manually or via search
    # Check GeckoTerminal or Cetus app for current pool addresses
    "DEEP": None,     # DeepBook/SUI pool - search required
    "SCA": None,      # Scallop/SUI pool - search required
    "WAL": None,      # Walrus/SUI pool - search required
    "HAEDAL": None,   # Haedal/SUI pool - search required
    "LBTC": None,     # LBTC/SUI pool - search required
    "NS": None,       # Navi/SUI pool - search required
}


class DexPaprikaFetcher:
    """
    Fetches historical OHLCV data from DexPaprika for SUI DEX pools using the SDK.
    """
    
    NETWORK = "sui"  # SUI blockchain network ID
    
    def __init__(self, output_dir: str = "data_raw"):
        """
        Initialize the fetcher with the official DexPaprika SDK.
        
        Args:
            output_dir: Directory for storing downloaded data
        """
        self.output_dir = Path(output_dir)
        self.client = DexPaprikaClient()
        print("✅ DexPaprika SDK client initialized")
        
    def search_pool(self, token_symbol: str) -> Optional[str]:
        """
        Smart Search: Finds the BEST pool for a pair (e.g., 'SUI USDC').
        Fixes the issue of grabbing stablecoin pools or returning no results.
        """
        # 1. Parse the input (e.g. "SUI USDC" -> ["SUI", "USDC"])
        tokens = token_symbol.upper().replace("/", " ").replace("-", " ").split()
        
        # If user just typed "SUI", we assume they want the main pair (SUI/USDC)
        # to avoid the "Stablecoin Trap" (getting a $1.00 stable pool).
        if len(tokens) == 1 and tokens[0] == "SUI":
            tokens = ["SUI", "USDC"]
            print(f"🔎 inferred pair: SUI + USDC (to avoid stablecoin pools)")
            
        primary_token = tokens[0]
        
        print(f"🔍 Searching for pair: {' + '.join(tokens)} on Cetus...")
        
        try:
            # 2. Broad Search: Search for the primary token to get all candidates
            # We search for "SUI" to get the list, then we filter locally.
            # This bypasses the API's inability to handle complex strings like "SUI USDC".
            results = self.client.search.search(primary_token)
            
            candidates = []
            
            if hasattr(results, 'pools') and results.pools:
                for pool in results.pools:
                    # Filter 1: Must be on SUI network
                    if not (hasattr(pool, 'network_id') and pool.network_id == self.NETWORK):
                        continue
                        
                    # Filter 2: Must be on Cetus (or the specific DEX you want)
                    # The API might return pools from other DEXes like Turbos/DeepBook
                    dex_name = getattr(pool, 'dex_id', '').lower()
                    if 'cetus' not in dex_name:
                        continue
                        
                    # Filter 3: Must contain ALL requested tokens
                    # We check if the pool's name or symbol contains our tokens
                    pool_name = getattr(pool, 'name', '').upper()
                    pool_symbols = [
                        getattr(pool, 'base_symbol', '').upper(),
                        getattr(pool, 'quote_symbol', '').upper()
                    ]
                    
                    # Check match: Do we have SUI? Do we have USDC?
                    # We check if all our search tokens appear in the pool's symbols
                    match = True
                    for t in tokens:
                        if t not in pool_symbols and t not in pool_name:
                            match = False
                            break
                    
                    if match:
                        candidates.append(pool)

            if not candidates:
                print(f"⚠️  No Cetus pool found for {'+'.join(tokens)}")
                return None
                
            # 3. Sort by Liquidity (Find the "Main" pool)
            # We want the deep SUI/USDC pool, not a low-liquidity clone.
            candidates_sorted = sorted(
                candidates,
                key=lambda x: float(getattr(x, 'liquidity_usd', 0) or 0),
                reverse=True
            )
            
            best_pool = candidates_sorted[0]
            addr = best_pool.address
            liq = float(getattr(best_pool, 'liquidity_usd', 0) or 0)
            name = getattr(best_pool, 'name', 'Unknown')
            
            print(f"✅ Found Best Pool: {name}")
            print(f"   Address: {addr}")
            print(f"   Liquidity: ${liq:,.0f}")
            
            return addr

        except Exception as e:
            print(f"❌ Search Error: {e}")
            return None
    
    def _calculate_interval_hours(self, interval: str) -> float:
        """Calculate the number of hours in an interval."""
        interval_map = {
            "1m": 1/60,
            "5m": 5/60,
            "15m": 15/60,
            "30m": 30/60,
            "1h": 1,
            "4h": 4,
            "6h": 6,
            "12h": 12,
            "24h": 24,
            "1d": 24,
        }
        return interval_map.get(interval, 1)
    
    def get_pool_ohlcv(
        self,
        pool_address: str,
        start_date: str = "2025-06-08",
        end_date: Optional[str] = None,
        interval: str = "1h",
        max_candles: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data using RAW REQUESTS to bypass SDK validation errors.
        Fixes: 'Field required: volume' error on sparse data.
        """
        import requests  # Import here or at top of file
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        print(f"📥 Fetching OHLCV data from {start_date} to {end_date} (Raw Mode)...")
        
        all_data = []
        last_stored_time = None
        current_date_pointer = start_dt
        
        # Base URL for DexPaprika (bypassing SDK wrapper)
        BASE_URL = "https://api.dexpaprika.com"
        
        while current_date_pointer <= end_dt:
            # 1. Prepare Parameters
            start_str = current_date_pointer.strftime("%Y-%m-%d")
            req_end_dt = min(current_date_pointer + timedelta(days=30), end_dt)
            end_str = req_end_dt.strftime("%Y-%m-%d")
            
            print(f"   Fetching window {start_str} to {end_str}...", end=" ", flush=True)
            
            # 2. Fetch using requests (Bypass SDK)
            batch_data = []
            max_retries = 5
            
            for attempt in range(max_retries):
                try:
                    # Construct URL manually
                    url = f"{BASE_URL}/networks/{self.NETWORK}/pools/{pool_address}/ohlcv"
                    params = {
                        "start": start_str,
                        "end": end_str,
                        "limit": 366,
                        "interval": interval
                    }
                    
                    resp = requests.get(url, params=params, timeout=10)
                    
                    if resp.status_code == 200:
                        batch_data = resp.json() # This is a list of dicts
                        break
                    elif resp.status_code == 429:
                        # Rate limit
                        time.sleep((attempt + 1) * 2)
                    else:
                        print(f"[Status {resp.status_code}]", end=" ")
                        time.sleep(1)
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep((attempt + 1) * 1)
                    else:
                        print(f"❌ Connection Error: {e}")
            
            # 3. Handle Empty
            if not batch_data:
                print("⚠ No data. Advancing 1 day...")
                current_date_pointer += timedelta(days=1)
                continue

            # 4. Filter & Clean Data (The "Manual Validator")
            new_candles = []
            max_batch_time = None
            
            for candle in batch_data:
                # Manual Extraction with Defaults
                # API returns keys like: time_open, open, high, low, close, volume
                raw_time = candle.get('time_open')
                if not raw_time:
                    continue
                    
                c_time = pd.to_datetime(raw_time)
                if c_time.tzinfo is not None:
                    c_time = c_time.tz_localize(None)
                
                # Update local max
                if max_batch_time is None or c_time > max_batch_time:
                    max_batch_time = c_time
                
                # Filter duplicates
                if last_stored_time is None or c_time > last_stored_time:
                    # CLEANING: Set volume to 0 if missing (Fixes your error)
                    if 'volume' not in candle or candle['volume'] is None:
                        candle['volume'] = 0.0
                    
                    new_candles.append(candle)

            # 5. Advance Pointer
            if new_candles:
                print(f"✓ {len(new_candles)} new")
                all_data.extend(new_candles)
                
                # Update High-Water Mark
                last_item = new_candles[-1]
                last_stored_time = pd.to_datetime(last_item.get('time_open'))
                if last_stored_time.tzinfo is not None:
                    last_stored_time = last_stored_time.tz_localize(None)
                
                # Set next start
                next_ptr = last_stored_time.replace(hour=0, minute=0, second=0, microsecond=0)
                if next_ptr < current_date_pointer:
                     next_ptr = current_date_pointer
                current_date_pointer = next_ptr
                
                # Anti-stuck logic
                if len(new_candles) < 10 and current_date_pointer == start_dt:
                     current_date_pointer += timedelta(days=1)

            else:
                print("⚠ Duplicates only. Pushing forward 1 day...")
                current_date_pointer += timedelta(days=1)
            
            if max_candles and len(all_data) >= max_candles:
                print(f"⚠️  Limit reached")
                break
                
            time.sleep(0.5)
            
        if not all_data:
            print(f"⚠️  No data returned for pool {pool_address[:10]}...")
            return None
            
        # 6. Final DataFrame Creation
        data_list = []
        for candle in all_data:
            try:
                # Safe Dict Access
                vol = float(candle.get('volume', 0))
                o = float(candle.get('open', 0))
                # Fallback logic: if high/low missing, use open/close
                c = float(candle.get('close', o))
                h = float(candle.get('high', max(o, c)))
                l = float(candle.get('low', min(o, c)))
                t = candle.get('time_open')
                
                if t:
                    dt_t = pd.to_datetime(t)
                    if dt_t.tzinfo is not None:
                        dt_t = dt_t.tz_localize(None)
                        
                    data_list.append({
                        'open_time': dt_t, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': vol
                    })
            except Exception as e:
                continue
                
        df = pd.DataFrame(data_list)
        df = df.sort_values('open_time').drop_duplicates(subset=['open_time']).reset_index(drop=True)
        
        return df

    def convert_to_binance_format(
        self,
        ohlcv_df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Convert to Binance format and FILL MISSING GAPS with flat candles.
        """
        if ohlcv_df is None or ohlcv_df.empty:
            return pd.DataFrame()
        
        df = ohlcv_df.copy()
        df = df.set_index('open_time')
        
        # 1. Infer Interval to create a perfect time grid
        # We calculate the most common time difference between consecutive candles
        if len(df) > 1:
            inferred_freq = df.index.to_series().diff().mode()[0]
        else:
            # Fallback default if only 1 candle exists
            inferred_freq = pd.Timedelta('1h') 
            
        # 2. Reindex to fill gaps (The "Gap Filler")
        # Create a full range from start to end
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=inferred_freq)
        df = df.reindex(full_idx)
        
        # 3. Fill Logic
        # Forward fill Close price (if no trades, price stays same)
        df['close'] = df['close'].ffill()
        
        # Fill Open, High, Low with the Close price (Flat candle)
        df['open'] = df['open'].fillna(df['close'])
        df['high'] = df['high'].fillna(df['close'])
        df['low'] = df['low'].fillna(df['close'])
        
        # Fill Volume with 0 (No trades occurred)
        df['volume'] = df['volume'].fillna(0)
        
        # Reset index to get open_time back as column
        df = df.reset_index().rename(columns={'index': 'open_time'})

        # 4. Standard Binance Calculations
        df["quote_volume"] = df["volume"] * df["close"]
        
        # Estimate Taker Buy/Sell (Proxy)
        df["price_change"] = df["close"] - df["open"]
        df["buy_pressure"] = (df["price_change"] > 0).astype(float)
        
        # If volume is 0, taker buy is 0
        df["taker_buy_base"] = df["volume"] * (0.5 + 0.1 * df["buy_pressure"])
        df["taker_buy_quote"] = df["taker_buy_base"] * df["close"]
        
        # Trade count proxy
        df["trades"] = (df["volume"] / df["close"]).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 10000).astype(int)
        
        # Select final columns
        binance_df = df[[
            "open_time", "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_base", "taker_buy_quote", "trades"
        ]].copy()
        
        return binance_df

    def save_to_csv(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> str:
        """
        Save data to CSV in stonks_ai format.
        
        Args:
            df: DataFrame with Binance-format data
            symbol: Token symbol
            interval: Candle interval
            
        Returns:
            Path to saved file
        """
        # Create directory: data_raw/symbol_lower/
        symbol_lower = symbol.lower()
        symbol_dir = self.output_dir / symbol_lower
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: SYMBOLUSDT_interval.csv
        filename = f"{symbol.upper()}USDT_{interval}.csv"
        filepath = symbol_dir / filename
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"💾 Saved {len(df)} candles to {filepath}")
        
        return str(filepath)
    
    def download_token_data(
        self,
        symbol: str,
        pool_address: Optional[str] = None,
        start_date: str = "2025-06-08",
        end_date: Optional[str] = None,
        interval: str = "1h",
        max_candles: int = None,
    ) -> Optional[str]:
        """
        Download and process data for a single token.
        
        Args:
            symbol: Token symbol
            pool_address: Pool address (if None, will search)
            start_date: Start date for historical data
            end_date: End date for historical data (None = today)
            interval: Candle interval
            max_candles: Maximum candles to fetch (None = unlimited)
            
        Returns:
            Path to saved CSV or None if failed
        """
        print(f"\n{'='*70}")
        print(f"📊 Processing {symbol}")
        print(f"{'='*70}")
        
        # Find pool if not provided
        if pool_address is None:
            pool_address = self.search_pool(symbol)
            if pool_address is None:
                print(f"❌ Cannot proceed without pool address for {symbol}")
                print(f"💡 Tip: Check GeckoTerminal (https://www.geckoterminal.com/sui-network/pools)")
                print(f"         or Cetus app (https://app.cetus.zone/pools) for pool addresses")
                return None
        else:
            print(f"ℹ️  Using pool address: {pool_address}")
            print(f"   (from CETUS_POOLS dictionary)")
        
        # Fetch OHLCV data
        ohlcv_df = self.get_pool_ohlcv(
            pool_address=pool_address,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            max_candles=max_candles,
        )
        
        if ohlcv_df is None or ohlcv_df.empty:
            print(f"❌ No data fetched for {symbol}")
            return None
        
        # Convert to Binance format
        print(f"🔄 Converting to Binance format...")
        binance_df = self.convert_to_binance_format(ohlcv_df, symbol)
        
        if binance_df.empty:
            print(f"❌ Failed to convert data for {symbol}")
            return None
        
        print(f"✅ Converted {len(binance_df)} candles")
        
        # Save to CSV
        filepath = self.save_to_csv(binance_df, symbol, interval)
        
        # Print summary
        print(f"\n📈 Summary for {symbol}:")
        print(f"   Date range: {binance_df['open_time'].min()} to {binance_df['open_time'].max()}")
        print(f"   Price range: ${binance_df['close'].min():.4f} - ${binance_df['close'].max():.4f}")
        print(f"   Avg volume: {binance_df['volume'].mean():,.0f}")
        
        return filepath
    
    def download_all_tokens(
        self,
        tokens: Optional[List[str]] = None,
        start_date: str = "2025-06-08",
        interval: str = "1h",
    ) -> Dict[str, str]:
        """
        Download data for all tokens.
        
        Args:
            tokens: List of token symbols (if None, use all from CETUS_POOLS)
            start_date: Start date for historical data
            interval: Candle interval
            
        Returns:
            Dictionary mapping symbol to filepath
        """
        if tokens is None:
            tokens = list(CETUS_POOLS.keys())
        
        results = {}
        failed = []
        
        print(f"\n🚀 Starting download for {len(tokens)} tokens...")
        print(f"📅 Date range: {start_date} to {datetime.now().strftime('%Y-%m-%d')}")
        print(f"⏱️  Interval: {interval}")
        
        for i, symbol in enumerate(tokens, 1):
            print(f"\n[{i}/{len(tokens)}] Processing {symbol}...")
            
            pool_address = CETUS_POOLS.get(symbol)
            
            try:
                filepath = self.download_token_data(
                    symbol=symbol,
                    pool_address=pool_address,
                    start_date=start_date,
                    interval=interval,
                )
                
                if filepath:
                    results[symbol] = filepath
                else:
                    failed.append(symbol)
                
                # Rate limiting: be nice to the API
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                failed.append(symbol)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"📊 DOWNLOAD SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Successfully downloaded: {len(results)}/{len(tokens)} tokens")
        if results:
            print(f"\nDownloaded tokens:")
            for symbol, filepath in results.items():
                print(f"   - {symbol}: {filepath}")
        
        if failed:
            print(f"\n❌ Failed tokens ({len(failed)}):")
            for symbol in failed:
                print(f"   - {symbol}")
        
        return results


def main():
    """
    Main entry point for downloading SUI token data.
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  STONKS.AI - SUI DATA DOWNLOADER                 ║
║                 Powered by DexPaprika Python SDK                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize fetcher
    fetcher = DexPaprikaFetcher(output_dir="data_raw")
    
    # Define tokens to download
    tokens = [
        "SUI",
        "USDC",
        "USDT",
        "CETUS",
        "DEEP",
        "SCA",
        "WAL",
        "HAEDAL",
        "LBTC",
        "NS",
    ]
    
    # Download all tokens
    results = fetcher.download_all_tokens(
        tokens=tokens,
        start_date="2025-06-08",  # Post-Cetus exploit (June 8, 2025)
        interval="1h",  # Hourly candles
    )
    
    print(f"\n🎉 Download complete! {len(results)} tokens ready for backtesting.")
    print(f"📁 Data saved to: {fetcher.output_dir}/")


if __name__ == "__main__":
    main()