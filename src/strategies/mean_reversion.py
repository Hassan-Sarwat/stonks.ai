from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.strategies.base import IStrategy

class MeanReversionStrategy(IStrategy):
    """
    Strategy B (Mean Reversion): Uses Bollinger Bands.
    Buy when price < Lower Band.
    Sell when price > Upper Band.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__()
        self.window = window
        self.num_std = num_std

    def initialize(self, data: pd.DataFrame) -> None:
        """
        Calculate Bollinger Bands.
        """
        self.data = data.copy()
        
        # Calculate SME and StdDev
        self.data['sma'] = self.data['close'].rolling(window=self.window).mean()
        self.data['std'] = self.data['close'].rolling(window=self.window).std()
        
        self.data['upper_band'] = self.data['sma'] + (self.data['std'] * self.num_std)
        self.data['lower_band'] = self.data['sma'] - (self.data['std'] * self.num_std)

    def on_tick(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Check price against Bollinger Bands.
        Note: on_tick usually receives a row from the original dataframe, 
        so we need to make sure 'upper_band' and 'lower_band' are in it.
        The Backtester passes the row from the dataframe passed to initialize(), 
        so if we modified self.data in initialize, we might need to look up pre-calculated values
        or calculate them on the fly if we want a pure online algorithm.
        
        For this implementation, we assume the Backtester iterates over the ENRICHED dataframe
        if we return it, or we need to access self.data by index/timestamp.
        
        Actually, the Backtester iterates the *input* data.
        To access the indicators calculated in initialize, we can look them up by index,
        assuming the row index matches.
        """
        
        # Safer way: Look up the indicators from our internal self.data using the index
        if row.name not in self.data.index:
            return None
            
        current_data = self.data.loc[row.name]
        
        # Check if we have enough data (not NaN)
        if pd.isna(current_data['upper_band']):
            return None
            
        price = row['close']
        upper = current_data['upper_band']
        lower = current_data['lower_band']
        
        signal = None
        
        if price < lower:
            # Price is oversold -> BUY
            if self.cash > 0:
                quantity = (self.cash * 0.99) / price
                signal = {'action': 'BUY', 'quantity': quantity}
                
        elif price > upper:
            # Price is overbought -> SELL
            if self.position > 0:
                signal = {'action': 'SELL', 'quantity': self.position}
                
        return signal

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Data not initialized")
        
        df = self.data.copy()
        df['signal'] = 0
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1
        return df
