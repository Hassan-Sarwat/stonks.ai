"""
Live trading script for the SUI trading bot.
"""

import asyncio
import json
import signal
import sys
from typing import Dict, Any

from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.xgboost_strategy import XGBoostStrategy
from src.live.trading_bot import SuiTradingBot


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


async def main():
    print("=== Stonks.ai Live Trading Bot for Cetus DEX ===")

    # Load configuration
    config = load_config()
    print("Configuration loaded")

    # Check if live trading is enabled
    live_config = config.get("live_trading", {})
    if not live_config.get("enabled", False):
        print("Live trading is disabled in config. Exiting.")
        return

    # Get trading parameters
    coins = config.get("coins", [])
    interval = config.get("interval", "1m")
    strategies_to_run = config.get("strategies", ["mean_reversion"])

    # Convert coins to trading pairs
    symbols = [f"{coin}_USDC" for coin in coins]

    # Get API credentials
    private_key = live_config.get("private_key", "")
    network = live_config.get("network", "testnet")
    dry_run = live_config.get("dry_run", True)

    if not private_key:
        print("Private key not configured. Exiting.")
        return

    # Get risk configuration
    risk_config = config.get("risk_management", {})

    # Get notification configuration
    notification_config = config.get("notifications", {})

    # Initialize strategy
    strategy = None
    if "momentum" in strategies_to_run:
        print("Using Momentum Strategy")
        strategy = MomentumStrategy(imbalance_threshold=0.0)
    elif "mean_reversion" in strategies_to_run:
        print("Using Mean Reversion Strategy")
        strategy = MeanReversionStrategy(window=20, num_std=2.0)
    elif "xgboost" in strategies_to_run:
        print("Using XGBoost ML Strategy")
        strategy = XGBoostStrategy(
            train_split=0.7,
            lookback_window=60,
            prediction_horizon=5,
            return_threshold=0.001,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
        )
    else:
        print(f"No valid strategy selected. Exiting.")
        return

    # Initialize trading bot
    bot = SuiTradingBot(
        private_key=private_key,
        strategy=strategy,
        symbols=symbols,
        interval=interval,
        network=network,
        dry_run=dry_run,
        notification_config=notification_config,
        risk_config=risk_config,
    )

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("Shutting down...")
        asyncio.create_task(shutdown())

    async def shutdown():
        await bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start bot
    try:
        await bot.start()

        # Keep main thread alive
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Error: {e}")
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
