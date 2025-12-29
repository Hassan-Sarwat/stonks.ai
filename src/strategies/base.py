from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class IStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Enforces the Strategy Pattern as defined in the project architecture.
    """

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.position: int = 0  # Current position size
        self.cash: float = 10000.0  # Starting cash, can be overridden
        self.portfolio_value: float = self.cash

    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with historical data.
        Used for pre-calculating indicators or setting up caches.
        
        Args:
            data (pd.DataFrame): The full dataset for the backtest.
        """
        pass

    @abstractmethod
    def on_tick(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Process a single market step (event-driven simulation).
        
        Args:
            row (pd.Series): The current row of data (candle/tick).
            
        Returns:
            Optional[Dict[str, Any]]: A signal dictionary (e.g., {'action': 'BUY', 'quantity': 1.0}) or None.
        """
        pass

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Vectorized signal generation.
        Should calculate signals for the entire dataset at once if possible.
        
        Returns:
            pd.DataFrame: The original dataframe with an added 'signal' column.
        """
        pass
