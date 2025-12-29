from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class IStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Enforces the Strategy Pattern as defined in the project architecture.
    """

    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
        # Positions: symbol -> quantity (positive for long, negative for short)
        self.positions: Dict[str, float] = {} 
        self.cash: float = 10000.0  # Starting cash, will be overridden by Backtester
        self.portfolio_value: float = self.cash

    @abstractmethod
    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize the strategy with historical data for all coins.
        Used for pre-calculating indicators or setting up caches.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping symbol to its historical dataframe.
        """
        pass

    @abstractmethod
    def on_tick(self, timestamp: pd.Timestamp, prices: Dict[str, float], rows: Dict[str, pd.Series]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Process a single market step (event-driven simulation) for all coins simultaneously.
        
        Args:
            timestamp (pd.Timestamp): Current simulation time.
            prices (Dict[str, float]): Current price for each symbol.
            rows (Dict[str, pd.Series]): Current row data for each symbol.
            
        Returns:
            Optional[Dict[str, Dict[str, Any]]]: A dictionary of signals keyed by symbol.
            Example: {'BTC': {'action': 'BUY', 'quantity': 0.1}, 'ETH': {'action': 'SELL', ...}}
        """
        pass
