"""
SUI Data Feed for fetching real-time market data from Cetus DEX.
"""
import asyncio
import time
import pandas as pd
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime, timedelta

from src.dex.cetus_client import CetusDexClient


class SuiDataFeed:
    def __init__(
        self,
        cetus_client: CetusDexClient,
        symbols: List[str],
        interval: str = '1m'
    ):
        """
        Initialize SUI data feed.
        
        Args:
            cetus_client: Initialized Cetus DEX client
            symbols: List of symbols to track (e.g., ["SUI_USDC"])
            interval: Data interval (e.g., "1m", "5m", "1h")
        """
        self.cetus_client = cetus_client
        self.symbols = symbols
        self.interval = interval
        self.data: Dict[str, pd.DataFrame] = {}
        self.running = False
        self.callbacks: List[Callable] = []
        
        # Initialize data structures with all required columns
        for symbol in symbols:
            self.data[symbol] = pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'taker_buy_base', 'taker_buy_quote', 'quote_volume', 'trades',
                'open_time', 'close_time'
            ])
        
        # Convert interval to seconds
        interval_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        self.interval_seconds = interval_map.get(interval, 60)
        
        # Interval has already been initialized
            
    def register_callback(self, callback: Callable):
        """Register a callback to be called when new data arrives."""
        self.callbacks.append(callback)
        
    async def start(self):
        """Start the data feed."""
        self.running = True
        await self._update_loop()
        
    async def stop(self):
        """Stop the data feed."""
        self.running = False
        
    async def _fetch_historical_data(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        Note: Cetus doesn't provide historical OHLCV data directly through
        an API.
        This would typically require indexing events from the blockchain or
        using a third-party data provider. This is a simplified placeholder
        implementation.
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Use a third-party API that indexes Cetus trading data
        # 2. Or build your own indexer that processes Cetus events from the
        #    blockchain
        
        # For now, we'll create synthetic data based on current price
        current_price = await self.cetus_client.get_price(symbol)
        # Convert Decimal to float to avoid type issues
        current_price_float = float(current_price)
        
        # Create synthetic OHLCV data
        now = datetime.now()
        data = []
        
        for i in range(limit):
            timestamp = int((now - timedelta(
                minutes=i * self.interval_seconds // 60)).timestamp() * 1000)
            # Create some price variation
            variation = (i % 5 - 2) / 100  # -2% to +2%
            price = current_price_float * (1 + variation)
            
            data.append({
                'timestamp': timestamp,
                'open': price * 0.998,
                'high': price * 1.005,
                'low': price * 0.995,
                'close': price,
                'volume': 1000 + (i % 10) * 100,
                'taker_buy_base': 500 + (i % 10) * 50,  # 50% of volume
                # Quote value
                'taker_buy_quote': (500 + (i % 10) * 50) * price,
                # Total quote value
                'quote_volume': (1000 + (i % 10) * 100) * price,
                'trades': 50 + (i % 10) * 5,  # Add trades
                'open_time': pd.to_datetime(timestamp, unit='ms'),
                'close_time': pd.to_datetime(
                    timestamp + self.interval_seconds * 1000, unit='ms')
            })
            
        # Convert to DataFrame and sort by time
        df = pd.DataFrame(data)
        df = df.sort_values('open_time')
        
        return df
        
    async def _update_price_data(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Update price data for a symbol."""
        try:
            # Get current price
            price = await self.cetus_client.get_price(symbol)
            # Convert Decimal to float
            price_float = float(price)
            
            # Create new candle data
            now = int(time.time() * 1000)
            candle_start = now - (now % (self.interval_seconds * 1000))
            
            new_data = {
                'timestamp': candle_start,
                'open': price_float * 0.998,  # Synthetic open price
                'high': price_float * 1.005,  # Synthetic high
                'low': price_float * 0.995,   # Synthetic low
                'close': price_float,
                'volume': 1000,  # Synthetic volume
                'taker_buy_base': 500,  # 50% of volume
                'taker_buy_quote': 500 * price_float,  # Quote value
                'quote_volume': 1000 * price_float,  # Total quote value
                'trades': 50,  # Number of trades
                'open_time': pd.to_datetime(candle_start, unit='ms'),
                'close_time': pd.to_datetime(
                    candle_start + self.interval_seconds * 1000, unit='ms')
            }
            
            return new_data
            
        except Exception as e:
            print(f"Error updating price data for {symbol}: {e}")
            return None
            
    async def _update_loop(self):
        """Main update loop that fetches data periodically."""
        # First, fetch historical data
        for symbol in self.symbols:
            try:
                historical_data = await self._fetch_historical_data(symbol)
                self.data[symbol] = historical_data
                print(f"Loaded historical data for {symbol}: "
                      f"{len(historical_data)} candles")
            except Exception as e:
                print(f"Error loading historical data for {symbol}: {e}")
                
        # Then start the update loop
        while self.running:
            try:
                for symbol in self.symbols:
                    # Update price data
                    new_data = await self._update_price_data(symbol)
                    
                    if new_data:
                        # Check if we already have this candle
                        df = self.data[symbol]
                        existing_candle = df[
                            df['timestamp'] == new_data['timestamp']
                        ]
                        
                        if len(existing_candle) > 0:
                            # Update existing candle
                            idx = existing_candle.index[0]
                            df.at[idx, 'high'] = max(
                                df.at[idx, 'high'], new_data['high'])
                            df.at[idx, 'low'] = min(
                                df.at[idx, 'low'], new_data['low'])
                            df.at[idx, 'close'] = new_data['close']
                            # Increment volume
                            df.at[idx, 'volume'] += new_data['volume'] * 0.1
                        else:
                            # Add new candle
                            self.data[symbol] = pd.concat([
                                df, 
                                pd.DataFrame([new_data])
                            ]).sort_values('open_time').reset_index(drop=True)
                            
                            # Keep only the last 1000 candles
                            if len(self.data[symbol]) > 1000:
                                self.data[symbol] = self.data[symbol].iloc[
                                    -1000:
                                ]
                                
                        # Notify callbacks
                        for callback in self.callbacks:
                            await callback(
                                symbol,
                                self.data[symbol],
                                new_data['close']
                            )
                            
                # Sleep until next update
                await asyncio.sleep(self.interval_seconds)
                
            except Exception as e:
                print(f"Error in data update loop: {e}")
                # Wait full interval on error
                await asyncio.sleep(self.interval_seconds)