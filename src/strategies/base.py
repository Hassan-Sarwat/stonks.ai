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
    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Vectorized signal generation.
        Should calculate signals for the entire dataset at once.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are symbols and values are DataFrames
                                     containing at least a 'signal' column (1=Buy, -1=Sell, 0=Hold).
        """
        pass

    @abstractmethod
    def on_tick(self, timestamp: pd.Timestamp, prices: Dict[str, float], rows: Dict[str, pd.Series]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Process a single market step (event-driven simulation).
        Deprecated for high-performance backtesting but kept for live trading compatibility.
        """
        pass
