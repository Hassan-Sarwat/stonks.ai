import pandas as pd
import numpy as np
from typing import List, Dict
from src.strategies.base import IStrategy

class Backtester:
    """
    High-performance vectorized backtester using Numpy.
    """

    def __init__(self, strategy: IStrategy, initial_capital: float = 10000.0):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.strategy.cash = initial_capital
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []

    def run(self, data: Dict[str, pd.DataFrame]):
        """
        Runs the backtest simulation using vectorized signal alignment and Numpy iteration.
        """
        print(f"Initializing strategy with {len(data)} assets...")
        self.strategy.initialize(data)
        
        # 1. Generate Signals (Vectorized)
        print("Generating signals...")
        signal_dfs = self.strategy.generate_signals()
        
        # 2. Align Data (Reindex to common timeline)
        # Find union of all timestamps
        print("Aligning data...")
        all_timestamps = set()
        for df in data.values():
            if 'open_time' in df.columns:
                all_timestamps.update(df['open_time'])
            else:
                all_timestamps.update(df.index)
        
        sorted_timestamps = sorted(list(all_timestamps))
        full_index = pd.DatetimeIndex(sorted_timestamps)
        
        # Create Price and Signal Matrices (Timestamp x Symbol)
        # We use a list of symbols to maintain order
        symbols = sorted(list(data.keys()))
        n_symbols = len(symbols)
        n_steps = len(full_index)
        
        price_matrix = np.full((n_steps, n_symbols), np.nan)
        signal_matrix = np.zeros((n_steps, n_symbols), dtype=int)
        
        symbol_map = {sym: i for i, sym in enumerate(symbols)}
        
        for i, sym in enumerate(symbols):
            df = data[sym].copy()
            if 'open_time' in df.columns:
                df.set_index('open_time', inplace=True)
            
            # Reindex to full timeline
            df_reindexed = df.reindex(full_index)
            price_matrix[:, i] = df_reindexed['close'].values
            
            if sym in signal_dfs:
                sig_df = signal_dfs[sym].copy()
                if 'open_time' in sig_df.columns:  # Ensure index is time
                     sig_df.set_index('open_time', inplace=True)
                elif not isinstance(sig_df.index, pd.DatetimeIndex):
                     # Likely same index as df if generated from it
                     sig_df.index = df.index
                     
                sig_reindexed = sig_df.reindex(full_index).fillna(0)
                signal_matrix[:, i] = sig_reindexed['signal'].fillna(0).astype(int).values

        # 3. Fast Execution Loop using Numpy
        print(f"Running vectorized simulation steps ({n_steps})...")
        
        cash = self.initial_capital
        positions = np.zeros(n_symbols) # Array of position quantities
        equity_history = np.zeros(n_steps)
        
        # Pre-calculate trade costs/revenues to avoid logic inside loop? 
        # No, because cash constraints are path-dependent. We must iterate.
        
        # Optimizations:
        # - Access numpy arrays via index
        # - Vectorized updates where possible? No, cash check is sequential.
        
        for t in range(n_steps):
            current_prices = price_matrix[t]
            current_signals = signal_matrix[t]
            
            # Skip step if all prices are NaN
            if np.isnan(current_prices).all():
                equity_history[t] = cash # or last known equity
                continue
            
            # Process Signals
            # Vectorized check: indices where signal != 0
            active_indices = np.where(current_signals != 0)[0]
            
            for idx in active_indices:
                price = current_prices[idx]
                if np.isnan(price): continue
                
                sig = current_signals[idx]
                
                if sig == 1: # BUY
                    # Determine quantity (e.g., 20% of CURRENT cash)
                    # Simplified logic from before
                    if cash > 10:
                        trade_amt = cash * 0.20
                        quantity = trade_amt / price
                        cost = quantity * price
                        
                        cash -= cost
                        positions[idx] += quantity
                        
                        # Record Trade (Optional: slows down loop, but needed for reporting)
                        # self.trades.append(...) 
                        
                elif sig == -1: # SELL
                    quantity = positions[idx]
                    if quantity > 0:
                        revenue = quantity * price
                        cash += revenue
                        positions[idx] = 0
            
            # Update Equity
            # Equity = Cash + Sum(Positions * Prices)
            # Handle NaN prices for equity calc (use 0 value or last price? using 0 for simplicity/safety)
            valid_prices = np.nan_to_num(current_prices) 
            # Note: nan_to_num might set price to 0, making position value 0. 
            # Ideally forward fill prices, but reindex already puts NaNs.
            
            portfolio_val = np.sum(positions * valid_prices)
            equity_history[t] = cash + portfolio_val

        self.equity_curve = equity_history.tolist()
        self._calculate_performance()

    def _calculate_performance(self):
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        print("\n--- Backtest Results ---")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Equity:    ${final_equity:,.2f}")
        print(f"Total Return:    {total_return:.2f}%")
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 1440)
            print(f"Sharpe Ratio:    {sharpe:.2f}")
