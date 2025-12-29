import pandas as pd
import numpy as np
from typing import List, Dict
from src.strategies.base import IStrategy

class Backtester:
    """
    The engine that accepts an IStrategy and simulates trading.
    """

    def __init__(self, strategy: IStrategy, initial_capital: float = 10000.0):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.strategy.cash = initial_capital
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []

    def run(self, data: pd.DataFrame):
        """
        Runs the backtest simulation.
        
        Args:
            data (pd.DataFrame): Historical OHLCV data.
        """
        print(f"Initializing strategy with {len(data)} rows of data...")
        self.strategy. initialize(data)
        
        # We can support both vectorized and event-driven. 
        # For this implementation, we'll iterate for simulation accuracy (slippage/fills),
        # but the strategy might have pre-calculated signals vectorially.
        
        print("Running backtest simulation...")
        for index, row in data.iterrows():
            # Pass current state to strategy
            signal = self.strategy.on_tick(row)
            
            # Execute signal if any
            if signal:
                self._execute_signal(signal, row)
            
            # Update equity curve
            current_value = self.strategy.cash + (self.strategy.position * row['close'])
            self.equity_curve.append(current_value)

        self._calculate_performance()

    def _execute_signal(self, signal: Dict, row: pd.Series):
        action = signal.get('action')
        quantity = signal.get('quantity', 0)
        price = row['close'] # Assuming fill at close for simplicity, or could use next open

        if action == 'BUY':
            cost = quantity * price
            if self.strategy.cash >= cost:
                self.strategy.cash -= cost
                self.strategy.position += quantity
                self.trades.append({
                    'time': row['open_time'] if 'open_time' in row else row.name,
                    'type': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'cost': cost
                })
        elif action == 'SELL':
            if self.strategy.position >= quantity:
                revenue = quantity * price
                self.strategy.cash += revenue
                self.strategy.position -= quantity
                self.trades.append({
                    'time': row['open_time'] if 'open_time' in row else row.name,
                    'type': 'SELL',
                    'price': price,
                    'quantity': quantity,
                    'revenue': revenue
                })

    def _calculate_performance(self):
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        print("\n--- Backtest Results ---")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Equity:    ${final_equity:,.2f}")
        print(f"Total Return:    {total_return:.2f}%")
        print(f"Total Trades:    {len(self.trades)}")
        
        # Simple Sharpe Ratio (assuming daily data approx, need adjustment for intraday)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 1440) # Annualized for 1m candles roughly
            print(f"Sharpe Ratio:    {sharpe:.2f}")
