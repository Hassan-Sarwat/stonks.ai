import pandas as pd
import os
import glob
from src.backtesting.engine import Backtester
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy

def load_data(data_dir: str = "data_raw") -> pd.DataFrame:
    """
    Load the most recent CSV file from the data directory.
    """
    # Find all csv files recursively
    search_pattern = os.path.join(data_dir, "**", "*.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
    # Just pick the first one for now, or the largest/most recent
    # Let's try to find one that looks like a 1m candle file
    latest_file = files[0] # Simplification
    print(f"Loading data from {latest_file}...")
    
    df = pd.read_csv(latest_file)
    
    # Ensure correct types
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    
    return df

def main():
    print("=== Stonks.ai Trader Bot ===")
    
    try:
        # Load Data
        data = load_data()
        print(f"Loaded {len(data)} candles.")
        
        # 1. Run Momentum Strategy
        print("\n--- Running Momentum Strategy (OFI) ---")
        momentum = MomentumStrategy(imbalance_threshold=0.0)
        bot_a = Backtester(momentum)
        bot_a.run(data)
        
        # 2. Run Mean Reversion Strategy
        print("\n--- Running Mean Reversion Strategy (BB) ---")
        mean_rev = MeanReversionStrategy(window=20, num_std=2.0)
        bot_b = Backtester(mean_rev)
        bot_b.run(data)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python fetch_data.py' first to download some data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
