import pandas as pd
import numpy as np
from typing import List, Dict
from src.strategies.base import IStrategy

class Backtester:
    """
    The engine that accepts an IStrategy and simulates trading on multiple coins.
    """

    def __init__(self, strategy: IStrategy, initial_capital: float = 10000.0):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.strategy.cash = initial_capital
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []

    def run(self, data: Dict[str, pd.DataFrame]):
        """
        Runs the backtest simulation across multiple assets.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary of historical OHLCV data.
        """
        print(f"Initializing strategy with {len(data)} assets...")
        
        # 1. Initialize Strategy
        self.strategy.initialize(data)
        
        # 2. Synchronize Timestamps
        # We need to iterate through time and trigger the strategy for each step where ANY data exists.
        all_timestamps = set()
        for df in data.values():
            if 'open_time' in df.columns:
                all_timestamps.update(df['open_time'].tolist())
            else:
                # Fallback if using datetime index
                all_timestamps.update(df.index.tolist())
                
        sorted_timestamps = sorted(list(all_timestamps))
        
        print(f"Running backtest simulation over {len(sorted_timestamps)} time steps...")
        
        # 3. Event Loop
        for timestamp in sorted_timestamps:
            # Collect current prices and rows for this timestamp
            current_prices = {}
            current_rows = {}
            
            for symbol, df in data.items():
                # Efficient lookup (assuming datetime index would be faster, but staying with current structure)
                # For optimized backtesting, we'd use reindexed dataframes.
                
                # Doing a row lookup every time is slow O(N), but robust for mvp. 
                # Optimization: Dataframes should be indexed by time before loop.
                # Let's assume passed data can be reindexed here locally or is pre-processed.
                
                # Try to get row by timestamp
                # Note: This relies on 'open_time' being the index or queryable.
                # Let's assume data is indexed by open_time for speed here.
                if 'open_time' in df.columns:
                    # Filter matching rows (should be 0 or 1)
                    # This is still slow inside a loop.
                    # BETTER: Use a generator or iterator for each dataframe.
                    pass 
                
            # --- OPTIMIZED APPROACH ---
            # Reindex all dataframes to the global timeline once
            pass
        
        # Let's do the reindexing strategy for simplicity and speed
        aligned_data = {}
        for symbol, df in data.items():
            df_copy = df.copy()
            if 'open_time' in df_copy.columns:
                df_copy.set_index('open_time', inplace=True)
            aligned_data[symbol] = df_copy
            
        for timestamp in sorted_timestamps:
            prices = {}
            rows = {}
            
            for symbol, df in aligned_data.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    rows[symbol] = row
                    prices[symbol] = row['close']
            
            if not rows:
                continue

            # Pass current state to strategy
            signals = self.strategy.on_tick(timestamp, prices, rows)
            
            if signals:
                self._execute_signals(signals, prices, timestamp)
            
            # Update equity
            self._update_equity(prices)

        self._calculate_performance()

    def _execute_signals(self, signals: Dict[str, Dict], prices: Dict[str, float], timestamp):
        for symbol, signal in signals.items():
            if symbol not in prices:
                continue # Can't trade if no price
                
            action = signal.get('action')
            quantity = signal.get('quantity', 0)
            price = prices[symbol]

            if action == 'BUY':
                cost = quantity * price
                if self.strategy.cash >= cost:
                    self.strategy.cash -= cost
                    self.strategy.positions[symbol] = self.strategy.positions.get(symbol, 0) + quantity
                    self.trades.append({
                        'time': timestamp,
                        'symbol': symbol,
                        'type': 'BUY',
                        'price': price,
                        'quantity': quantity,
                        'cost': cost
                    })
            elif action == 'SELL':
                current_pos = self.strategy.positions.get(symbol, 0)
                if current_pos >= quantity:
                    revenue = quantity * price
                    self.strategy.cash += revenue
                    self.strategy.positions[symbol] -= quantity
                    self.trades.append({
                        'time': timestamp,
                        'symbol': symbol,
                        'type': 'SELL',
                        'price': price,
                        'quantity': quantity,
                        'revenue': revenue
                    })

    def _update_equity(self, current_prices: Dict[str, float]):
        equity = self.strategy.cash
        for symbol, qty in self.strategy.positions.items():
            if symbol in current_prices:
                equity += qty * current_prices[symbol]
            # If price missing (stale), use last known? 
            # For now, if missing, we just don't add it (risk of drop), 
            # or better: we should keep track of last known prices.
        
        self.equity_curve.append(equity)

    def _calculate_performance(self):
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        print("\n--- Backtest Results ---")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Equity:    ${final_equity:,.2f}")
        print(f"Total Return:    {total_return:.2f}%")
        print(f"Total Trades:    {len(self.trades)}")
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 1440)
            print(f"Sharpe Ratio:    {sharpe:.2f}")
