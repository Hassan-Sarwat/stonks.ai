# Stonks.ai Trader Bot

A high-performance, vectorized algorithmic trading backtester and live trading bot using the **Strategy Design Pattern**.

## 🚀 Getting Started

### Prerequisites
- **uv** (Python package manager)
- Python 3.12+

### Installation
Clone the repository and install dependencies:
```bash
uv sync
```

### Running Backtests
To run the backtester with the default configuration:
```bash
uv run main.py
```

### Running Live Trading
To run the live trading bot with Cetus DEX on SUI blockchain:
```bash
uv run live_trading.py
```

## ⚙️ Configuration
The bot is configured via `config.json`. You can control the active strategies, coins, timeframe, capital, and live trading settings.

```json
{
    "coins": ["SUI"],
    "interval": "1m",
    "start_date": null, 
    "end_date": null,
    "initial_capital": 10000.0,
    "strategies": ["momentum", "mean_reversion"],
    "live_trading": {
        "enabled": false,
        "network": "testnet",
        "private_key": "YOUR_PRIVATE_KEY_HERE",
        "dry_run": true
    },
    "risk_management": {
        "max_position_size": 0.1,
        "max_daily_drawdown": 0.05,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.05
    },
    "notifications": {
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "email": ""
    }
}
```

## 🧠 Strategies

### Available Strategies
- **Momentum (OFI)**: Trades based on Order Flow Imbalance.
- **Mean Reversion (BB)**: Trades based on Bollinger Bands deviations.

### Adding a New Strategy
1.  **Create a file** in `src/strategies/` (e.g., `my_strategy.py`).
2.  **Inherit from `IStrategy`** (`src/strategies/base.py`).
3.  **Implement `generate_signals`**:
    ```python
    from src.strategies.base import IStrategy
    
    class MyStrategy(IStrategy):
        def generate_signals(self) -> Dict[str, pd.DataFrame]:
            signals = {}
            for symbol, df in self.data.items():
                df = df.copy()
                df['signal'] = 0
                # Logic: Build your signal col (1=Buy, -1=Sell)
                df.loc[condition_buy, 'signal'] = 1
                df.loc[condition_sell, 'signal'] = -1
                signals[symbol] = df[['signal']]
            return signals
            
        # on_tick is deprecated/optional for vectorized mode
        def on_tick(self, timestamp, prices, rows):
            pass
    ```
4.  **Register your strategy** in `config.json` and run `uv run main.py`.

## 🏗️ Architecture
- **`src/backtesting/engine.py`**: A Numpy-optimized execution engine that runs simulations across multiple assets simultaneously.
- **`src/strategies/`**: Contains all trading logic. Strategies are independent and pluggable.
- **`src/blockchain/`**: SUI blockchain integration components.
- **`src/dex/`**: Cetus DEX integration for executing trades on SUI blockchain.
- **`src/data/`**: Data handling components for both historical and real-time data.
- **`src/execution/`**: Order execution components for live trading.
- **`src/risk/`**: Risk management components for position sizing and stop-loss/take-profit.
- **`src/monitoring/`**: Monitoring and notification components.
- **`src/live/`**: Live trading components that integrate all the above.

## 🚀 Live Trading with Cetus DEX

The bot now supports live trading on the Cetus DEX on the SUI blockchain. To use this feature:

1. Configure your `config.json` with the appropriate settings:
   - Set `"enabled": true` in the `live_trading` section
   - Provide your SUI wallet private key
   - Choose the network (`mainnet`, `testnet`, or `devnet`)
   - Set `dry_run` to `true` for testing without executing actual trades

2. Run the live trading bot:
   ```bash
   uv run live_trading.py
   ```

### Features
- **Real-time Data**: Fetches real-time price data from Cetus DEX
- **Order Execution**: Executes trades directly on Cetus DEX
- **Risk Management**: Implements position sizing, stop-loss, and take-profit
- **Monitoring**: Real-time monitoring with notifications via Telegram and email

## 🛡️ Security Considerations

1. **Private Key Security**:
   - Never store private keys in the repository
   - Use environment variables or a secure secret manager
   - Consider using a dedicated wallet with limited funds

2. **Server Security**:
   - Keep the server updated
   - Use SSH key authentication
   - Configure a firewall
   - Use a non-root user

## 📊 Deployment

For reliable operation, deploy the bot on a server with the following considerations:

### Server Requirements
- **Operating System**: Linux (Ubuntu 20.04 or later recommended)
- **CPU**: 2+ cores
- **RAM**: 4+ GB
- **Storage**: 20+ GB SSD
- **Network**: Stable internet connection

### Deployment Steps
1. Set up the server and install dependencies:
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install dependencies
   sudo apt install -y python3-pip git
   
   # Install uv
   pip install uv
   
   # Clone repository
   git clone https://github.com/yourusername/stonks.ai.git
   cd stonks.ai
   
   # Install dependencies
   uv sync
   ```

2. Configure the bot:
   ```bash
   # Edit config.json with your settings
   nano config.json
   ```

3. Set up a systemd service for automatic startup and restart:
   ```bash
   sudo nano /etc/systemd/system/stonks-bot.service
   ```

   Add the following content:
   ```
   [Unit]
   Description=Stonks.ai Trading Bot
   After=network.target
   
   [Service]
   User=yourusername
   WorkingDirectory=/path/to/stonks.ai
   ExecStart=/path/to/python /path/to/stonks.ai/live_trading.py
   Restart=on-failure
   RestartSec=5s
   
   [Install]
   WantedBy=multi-user.target
   ```

4. Enable and start the service:
   ```bash
   sudo systemctl enable stonks-bot
   sudo systemctl start stonks-bot
   ```

5. Monitor the bot:
   ```bash
   # Check status
   sudo systemctl status stonks-bot
   
   # View logs
   sudo journalctl -u stonks-bot -f
   ```

## ⚠️ Disclaimer

Cryptocurrency trading involves significant risk and may result in loss of funds.
