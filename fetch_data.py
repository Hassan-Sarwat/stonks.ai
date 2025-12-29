"""
Simple script to fetch Binance data and save to CSV
"""

import requests
import pandas as pd
import time
import os

# Configuration
SYMBOL = "XRPUSDT"  #ETHUSDT, SOLUSDT, TRXUSDT, BNBUSDT
INTERVALS = ["1m", "15m", "1h", "1d"]
END_TIME = 1766936750534  # Your timestamp
START_TIME = END_TIME - (365 * 24 * 60 * 60 * 1000)  # 1 year back
OUTPUT_DIR = "data"

# Rate limiting - Binance allows 6000 weight per minute, klines endpoint weight is 2
REQUEST_DELAY = 0.5  # Conservative delay between requests
MAX_RETRIES = 5


def get_last_timestamp(symbol, interval):
    """Get the last timestamp from existing CSV file if it exists"""
    filename = f"{OUTPUT_DIR}/{symbol}_{interval}.csv"
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if len(df) > 0:
                # Get last close_time and convert to milliseconds
                last_time = pd.to_datetime(df["close_time"].iloc[-1])
                last_timestamp = int(last_time.timestamp() * 1000)
                print(f"  Found existing data, resuming from {last_time}")
                return last_timestamp + 1  # Start from next candle
        except Exception as e:
            print(f"  Error reading existing file: {e}")
    return None


def fetch_klines(symbol, interval, start_time, end_time):
    """Fetch klines from Binance API with proper rate limit handling"""
    all_data = []
    save_threshold = 20000  # Save every 5000 lines

    # Check if we should resume from existing data
    last_timestamp = get_last_timestamp(symbol, interval)
    if last_timestamp:
        current_start = last_timestamp
        print(f"Fetching {symbol} {interval} (resuming)...")
    else:
        current_start = start_time
        print(f"Fetching {symbol} {interval} (new)...")

    while current_start < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,
        }

        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                response = requests.get(url, params=params, timeout=30)

                # Check for rate limit headers
                used_weight = response.headers.get("X-MBX-USED-WEIGHT-1M", "N/A")

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    print(f"  Rate limit hit! Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                elif response.status_code == 418:
                    retry_after = int(response.headers.get("Retry-After", 180))
                    print(f"  IP banned! Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue

                response.raise_for_status()
                data = response.json()

                # Show rate limit usage occasionally
                if len(all_data) % 10000 == 0 and all_data:
                    print(f"  Weight used (1min): {used_weight}")

                if not data:
                    break

                all_data.extend(data)
                current_start = data[-1][6] + 1  # Last close time + 1

                print(f"  Fetched {len(all_data)} candles...")

                # Save periodically to avoid losing data
                if len(all_data) >= save_threshold:
                    print(f"  Saving checkpoint at {len(all_data)} candles...")
                    save_to_csv(all_data, symbol, interval)
                    all_data = []  # Clear buffer after saving

                if len(data) < 1000:
                    break

                time.sleep(REQUEST_DELAY)  # Rate limiting
                break  # Success, exit retry loop

            except requests.exceptions.RequestException as e:
                retry_count += 1
                print(f"  Error: {e}. Retry {retry_count}/{MAX_RETRIES}")
                if retry_count >= MAX_RETRIES:
                    # Save what we have before crashing
                    if all_data:
                        print(f"  Saving {len(all_data)} candles before exit...")
                        save_to_csv(all_data, symbol, interval)
                    raise
                time.sleep(2**retry_count)  # Exponential backoff

    # Save any remaining data
    if all_data:
        print(f"  Saving final {len(all_data)} candles...")
        save_to_csv(all_data, symbol, interval)

    print(f"  Complete!\n")


def save_to_csv(data, symbol, interval):
    """Convert to DataFrame and save/append to CSV"""
    if not data:
        print(f"  No new data to save")
        return

    new_df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    # Convert timestamps to datetime
    new_df["open_time"] = pd.to_datetime(new_df["open_time"], unit="ms")
    new_df["close_time"] = pd.to_datetime(new_df["close_time"], unit="ms")

    # Convert to numeric
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]:
        new_df[col] = new_df[col].astype(float)

    # Save or append
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}/{symbol}_{interval}.csv"

    if os.path.exists(filename):
        # Append to existing file
        existing_df = pd.read_csv(filename)
        existing_df["open_time"] = pd.to_datetime(existing_df["open_time"])
        existing_df["close_time"] = pd.to_datetime(existing_df["close_time"])

        # Concatenate and remove duplicates based on open_time
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["open_time"], keep="last")
        combined_df = combined_df.sort_values("open_time").reset_index(drop=True)

        combined_df.to_csv(filename, index=False)
        print(
            f"✓ Appended {len(new_df)} rows to {filename} (total: {len(combined_df)} rows)"
        )
    else:
        # Create new file
        new_df.to_csv(filename, index=False)
        print(f"✓ Saved {len(new_df)} rows to {filename}")


def main():
    print("Starting data fetch...\n")

    for interval in INTERVALS:
        fetch_klines(SYMBOL, interval, START_TIME, END_TIME)
        # Data is saved incrementally inside fetch_klines, no need to save again

    print("\nDone! Files saved in 'data/' directory")


if __name__ == "__main__":
    main()
