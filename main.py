import pandas as pd
import json
import os
import glob
from typing import Dict, List
from src.backtesting.engine import Backtester
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.xgboost_strategy import XGBoostStrategy


def load_config(config_path: str = "config.json") -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_data(
    coins: List[str], interval: str, data_dir: str = "data_raw"
) -> Dict[str, pd.DataFrame]:
    """
    Load data for specified coins and interval.
    Returns a dictionary of {symbol: DataFrame}.
    path: data_raw/{symbol_lower}/{symbol_upper}USDT_{interval}.csv
    """
    data = {}
    for coin in coins:
        symbol_lower = coin.lower()
        symbol_upper = coin.upper()
        file_path = os.path.join(
            data_dir, symbol_lower, f"{symbol_upper}USDT_{interval}.csv"
        )

        if os.path.exists(file_path):
            print(f"Loading {coin} data from {file_path}...")
            df = pd.read_csv(file_path)
            # Ensure correct types
            df["open_time"] = pd.to_datetime(df["open_time"])
            df = df.sort_values("open_time").reset_index(drop=True)
            data[coin] = df
        else:
            print(f"Warning: Data file not found for {coin} ({file_path})")

    if not data:
        raise FileNotFoundError("No data files loaded for any of the requested coins.")

    return data


def main():
    print("=== Stonks.ai Trader Bot (Multi-Coin) ===")

    try:
        # 1. Load Config
        print("Loading configuration...")
        config = load_config()
        print(f"Config: {json.dumps(config, indent=2)}")

        coins = config.get("coins", [])
        interval = config.get("interval", "1m")
        initial_capital = config.get("initial_capital", 10000.0)

        # 2. Load Data for all coins
        data = load_data(coins, interval)
        print(f"Loaded data for {len(data)} coins.")

        # 3. Run Strategies
        strategies_to_run = config.get("strategies", ["momentum"])

        if "momentum" in strategies_to_run:
            print("\n====================================")
            print("Running Momentum Strategy (OFI)")
            print("====================================")
            # Create strategy instance
            momentum = MomentumStrategy(imbalance_threshold=0.0)
            # Run backtest with Unified Portfolio
            bot_a = Backtester(momentum, initial_capital=initial_capital)
            bot_a.run(data)

        if "mean_reversion" in strategies_to_run:
            print("\n====================================")
            print("Running Mean Reversion Strategy (BB)")
            print("====================================")
            mean_rev = MeanReversionStrategy(window=20, num_std=2.0)
            bot_b = Backtester(mean_rev, initial_capital=initial_capital)
            bot_b.run(data)

        if "xgboost" in strategies_to_run:
            print("\n====================================")
            print("Running XGBoost ML Strategy")
            print("====================================")
            xgb_strategy = XGBoostStrategy(
                train_split=0.7,
                lookback_window=60,
                prediction_horizon=5,
                return_threshold=0.001,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
            )
            bot_xgb = Backtester(xgb_strategy, initial_capital=initial_capital)
            bot_xgb.run(data)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
