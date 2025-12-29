# Stonks.ai Trader Bot

A high-performance, vectorized algorithmic trading backtester using the **Strategy Design Pattern**.

## 🚀 Getting Started

### Prerequisites
- **uv** (Python package manager)
- Python 3.10+

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

## ⚙️ Configuration
The bot is configured via `config.json`. You can control the active strategies, coins, timeframe, and capital.

```json
{
    "coins": ["XRP", "SOL", "ETH", "BNB", "TRX"],
    "interval": "1m",
    "start_date": null, 
    "end_date": null,
    "initial_capital": 10000.0,
    "strategies": ["momentum", "mean_reversion"]
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
4.  **Register your strategy** in `main.py` and run it.

## 🏗️ Architecture
- **`src/backtesting/engine.py`**: A Numpy-optimized execution engine that runs simulations across multiple assets simultaneously.
- **`src/strategies/`**: Contains all trading logic. Strategies are independent and pluggable.
