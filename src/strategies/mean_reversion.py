from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.strategies.base import IStrategy

class MeanReversionStrategy(IStrategy):
    """
    Strategy B (Mean Reversion): Uses Bollinger Bands.
    Multi-coin implementation.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__()
        self.window = window
        self.num_std = num_std
        self.indicators = {} # symbol -> DataFrame with bands

    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Calculate Bollinger Bands for all symbols.
        """
        for symbol, df in data.items():
            df = df.copy()
            df['sma'] = df['close'].rolling(window=self.window).mean()
            df['std'] = df['close'].rolling(window=self.window).std()
            df['upper_band'] = df['sma'] + (df['std'] * self.num_std)
            df['lower_band'] = df['sma'] - (df['std'] * self.num_std)
            
            if 'open_time' in df.columns:
                df.set_index('open_time', inplace=True)
                
            self.indicators[symbol] = df

    def on_tick(self, timestamp: pd.Timestamp, prices: Dict[str, float], rows: Dict[str, pd.Series]) -> Optional[Dict[str, Dict[str, Any]]]:
        signals = {}
        
        for symbol, _ in rows.items():
            price = prices.get(symbol)
            if price is None: continue
            
            # Lookup indicators
            if symbol not in self.indicators or timestamp not in self.indicators[symbol].index:
                continue
                
            ind_row = self.indicators[symbol].loc[timestamp]
            
            if pd.isna(ind_row['upper_band']):
                continue
                
            upper = ind_row['upper_band']
            lower = ind_row['lower_band']
            
            signal = None
            
            if price < lower:
                # Buy
                if self.positions.get(symbol, 0) <= 0:
                    trade_cash = self.cash * 0.20
                    if trade_cash > 10:
                        quantity = trade_cash / price
                        signal = {'action': 'BUY', 'quantity': quantity}
                        
            elif price > upper:
                # Sell
                pos_qty = self.positions.get(symbol, 0)
                if pos_qty > 0:
                     signal = {'action': 'SELL', 'quantity': pos_qty}
                     
            if signal:
                signals[symbol] = signal
                
        return signals if signals else None

    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Vectorized signal generation using Bollinger Bands.
        """
        signals = {}
        
        # We need to make sure indicators are calculated
        if not self.indicators:
            self.initialize(self.data)
            
        for symbol, df in self.indicators.items():
            # df already has upper_band and lower_band from initialize()
            df = df.copy()
            df['signal'] = 0
            
            # Buy when price < lower band
            df.loc[df['close'] < df['lower_band'], 'signal'] = 1
            
            # Sell when price > upper band
            df.loc[df['close'] > df['upper_band'], 'signal'] = -1
            
            signals[symbol] = df[['signal']]
            
        return signals
