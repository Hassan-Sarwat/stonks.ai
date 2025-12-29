from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.strategies.base import IStrategy

class MomentumStrategy(IStrategy):
    """
    Strategy A (Momentum): Uses Order Flow Imbalance (OFI).
    Multi-coin implementation.
    """
    
    def __init__(self, imbalance_threshold: float = 0.0):
        super().__init__()
        self.imbalance_threshold = imbalance_threshold
        self.ofi_data = {} # Pre-calculated OFI series per symbol

    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Pre-calculate OFI for all symbols.
        """
        self.data = data
        
        for symbol, df in data.items():
            # Calculate metrics
            taker_sell_base = df['volume'] - df['taker_buy_base']
            ofi = df['taker_buy_base'] - taker_sell_base
            
            # Store optimized lookups
            # We assume df has 'open_time' or we rely on reindexing in engine.
            # Ideally the engine passes clean rows, but we can store series here.
            # Let's modify the dataframe safely.
            df = df.copy()
            df['ofi'] = ofi
            
            # If engine reindexes main data, we might lose these columns if we don't pass them back
            # OR we just assume 'data' is mutable or referenced. 
            # In our Engine, we copied data to reindex.
            # So looking up 'ofi' in on_tick from self.data might fail if index mismatch.
            
            # Strategy:
            # We will perform online calculation in on_tick for simplicity and correctness 
            # without complex data syncing, OR we store a map of timestamp -> ofi.
            
            # Let's map timestamp -> ofi for O(1) lookup
            if 'open_time' in df.columns:
                self.ofi_data[symbol] = pd.Series(ofi.values, index=df['open_time'])
            else:
                 self.ofi_data[symbol] = ofi # Assuming index is time

    def on_tick(self, timestamp: pd.Timestamp, prices: Dict[str, float], rows: Dict[str, pd.Series]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Check OFI for all available symbols and allocate capital.
        """
        signals = {}
        
        # Simple Logic: 
        # Divide cash equally among opportunities or just take first one?
        # Let's try to grab any opportunity.
        
        for symbol, row in rows.items():
            # Get pre-calc OFI if available, else calc on fly
            current_ofi = 0
            if symbol in self.ofi_data and timestamp in self.ofi_data[symbol].index:
                current_ofi = self.ofi_data[symbol].loc[timestamp]
            else:
                # Fallback calculation
                vol = row.get('volume', 0)
                buy = row.get('taker_buy_base', 0)
                sell = vol - buy
                current_ofi = buy - sell
            
            signal = None
            
            if current_ofi > self.imbalance_threshold:
                # Buy signal
                # Check if we already have a position
                if self.positions.get(symbol, 0) <= 0:
                    # Allocate portion of cash. E.g. 10% of total portfolio or all available?
                    # Let's act conservatively: max 20% of CURRENT cash per trade
                    trade_cash = self.cash * 0.20
                    if trade_cash > 10: # Min trade size
                         quantity = trade_cash / row['close']
                         signal = {'action': 'BUY', 'quantity': quantity}
            
            elif current_ofi < -self.imbalance_threshold:
                # Sell signal
                pos_qty = self.positions.get(symbol, 0)
                if pos_qty > 0:
                    signal = {'action': 'SELL', 'quantity': pos_qty} # Close entire position
            
            if signal:
                signals[symbol] = signal
        
        return signals if signals else None

    def generate_signals(self) -> pd.DataFrame:
        # Not implemented for multi-coin vectorization yet
        pass
