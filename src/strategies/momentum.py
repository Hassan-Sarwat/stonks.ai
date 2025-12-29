from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.strategies.base import IStrategy

class MomentumStrategy(IStrategy):
    """
    Strategy A (Momentum): Uses Order Flow Imbalance (OFI).
    If taker_buy_base > taker_sell_base significantly -> Buy.
    """
    
    def __init__(self, imbalance_threshold: float = 0.0):
        super().__init__()
        self.imbalance_threshold = imbalance_threshold

    def initialize(self, data: pd.DataFrame) -> None:
        """
        Pre-calculate OFI and signals.
        """
        self.data = data.copy()
        
        # Calculate Taker Sell Volume (Base)
        # volume = taker_buy_base + taker_sell_base
        # therefore: taker_sell_base = volume - taker_buy_base
        self.data['taker_sell_base'] = self.data['volume'] - self.data['taker_buy_base']
        
        # Calculate Net Order Flow (OFI)
        self.data['ofi'] = self.data['taker_buy_base'] - self.data['taker_sell_base']
        
        # Pre-calculate signals vectorially for efficiency reference, 
        # though on_tick will process them event-by-event.
        self.data['signal_strength'] = self.data['ofi']

    def on_tick(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Check OFI for the current tick and generate signals.
        """
        current_ofi = row['taker_buy_base'] - (row['volume'] - row['taker_buy_base'])
        
        # Simple Logic:
        # If OFI is positive and we have cash -> BUY
        # If OFI is negative and we have position -> SELL
        
        # NOTE: A real strategy would likely check the *accumulated* OFI or a moving average of OFI.
        # For this requirement, we look at the immediate imbalance.
        
        signal = None
        
        if current_ofi > self.imbalance_threshold:
            # Bullish pressure
            if self.cash > 0:
                # Buy as much as possible with current cash
                # In a real system, we'd have risk management here.
                quantity = (self.cash * 0.99) / row['close'] # 99% usage to leave room for fees/slippage
                signal = {'action': 'BUY', 'quantity': quantity}
                
        elif current_ofi < -self.imbalance_threshold:
            # Bearish pressure
            if self.position > 0:
                signal = {'action': 'SELL', 'quantity': self.position}
                
        return signal

    def generate_signals(self) -> pd.DataFrame:
        """
        Vectorized signal generation for analysis.
        """
        if self.data is None:
            raise ValueError("Data not initialized")
            
        df = self.data.copy()
        df['signal'] = 0
        df.loc[df['ofi'] > self.imbalance_threshold, 'signal'] = 1  # Buy signal target
        df.loc[df['ofi'] < -self.imbalance_threshold, 'signal'] = -1 # Sell signal target
        return df
