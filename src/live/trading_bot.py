"""
Main trading bot class for live trading on Cetus DEX.
"""
import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.blockchain.sui_client import SuiBlockchainClient
from src.dex.cetus_client import CetusDexClient
from src.data.sui_data_feed import SuiDataFeed
from src.execution.sui_order_manager import SuiOrderManager
from src.risk.sui_risk_manager import SuiRiskManager
from src.monitoring.monitor import Monitor
from src.strategies.base import IStrategy


class SuiTradingBot:
    def __init__(
        self,
        private_key: str,
        strategy: IStrategy,
        symbols: List[str],
        interval: str = '1m',
        network: str = "testnet",
        dry_run: bool = True,
        notification_config: Optional[Dict[str, str]] = None,
        risk_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the SUI trading bot.
        
        Args:
            private_key: Private key for the SUI wallet
            strategy: Trading strategy to use
            symbols: List of symbols to trade (e.g., ["SUI_USDC"])
            interval: Data interval (e.g., "1m", "5m", "1h")
            network: Network to connect to ("mainnet", "testnet", or "devnet")
            dry_run: If True, simulate orders without executing them
            notification_config: Dictionary with notification configuration
            risk_config: Dictionary with risk management configuration
        """
        self.private_key = private_key
        self.strategy = strategy
        self.symbols = symbols
        self.interval = interval
        self.network = network
        self.dry_run = dry_run
        self.notification_config = notification_config or {}
        self.risk_config = risk_config or {}
        
        # Initialize components
        self.sui_client = SuiBlockchainClient(private_key, network)
        self.cetus_client = CetusDexClient(self.sui_client)
        self.order_manager = SuiOrderManager(self.cetus_client, dry_run)
        
        # Initialize risk manager with configuration
        self.risk_manager = SuiRiskManager(
            self.cetus_client,
            max_position_size=self.risk_config.get('max_position_size', 0.1),
            max_daily_drawdown=self.risk_config.get('max_daily_drawdown', 0.05),
            stop_loss_pct=self.risk_config.get('stop_loss_pct', 0.03),
            take_profit_pct=self.risk_config.get('take_profit_pct', 0.05)
        )
        
        # Initialize data feed
        self.data_feed = SuiDataFeed(self.cetus_client, symbols, interval)
        
        # Initialize monitor
        self.monitor = Monitor(self, self.notification_config)
        
        # Trading state
        self.running = False
        self.main_task = None
        
    async def on_data_update(self, symbol: str, data: Any, current_price: float):
        """Callback for when new data arrives."""
        try:
            # Update strategy with new data
            self.strategy.data[symbol] = data
            
            # Check stop loss / take profit for existing positions
            action = self.risk_manager.check_stop_loss_take_profit(symbol, current_price)
            if action:
                print(f"{action.upper()} triggered for {symbol} at {current_price}")
                await self._close_position(symbol, current_price)
                return
            
            # Generate signals
            signals = self.strategy.generate_signals()
                        
            if symbol in signals:
                signal_df = signals[symbol]
                if not signal_df.empty:
                    latest_signal = signal_df.iloc[-1]['signal']
                    print(f"Latest signal for {symbol}: {latest_signal}")
                    
                    if latest_signal == 1:  # Buy signal
                        print(f"BUY SIGNAL detected for {symbol} at {current_price}")
                        await self._handle_buy_signal(symbol, current_price)
                    elif latest_signal == -1:  # Sell signal
                        print(f"SELL SIGNAL detected for {symbol} at {current_price}")
                        await self._handle_sell_signal(symbol, current_price)
                    else:
                        print(f"No actionable signal for {symbol} (signal = {latest_signal})")
                else:
                    print(f"Empty signal dataframe for {symbol}")
            else:
                print(f"No signals generated for {symbol}")
        except Exception as e:
            print(f"Error in data update callback: {e}")
            
    async def _handle_buy_signal(self, symbol: str, price: float):
        """Handle a buy signal."""
        # Check if we should take this trade
        if not await self.risk_manager.should_take_trade(symbol, 'buy', price):
            return
            
        # Calculate position size
        quantity = await self.risk_manager.calculate_position_size(symbol, price)
        
        # Execute buy order
        order_id = await self.order_manager.execute_buy(symbol, quantity)
        if order_id:
            # Get order details
            order = await self.order_manager.get_order(order_id)
            if order.get('status') == 'filled':
                # Register position
                self.risk_manager.register_position(
                    symbol, 
                    order.get('price', price), 
                    order.get('quantity', quantity)
                )
                print(f"Registered new position in {symbol}")
                
    async def _handle_sell_signal(self, symbol: str, price: float):
        """Handle a sell signal."""
        position = self.risk_manager.get_position(symbol)
        if position:
            await self._close_position(symbol, price)
            
    async def _close_position(self, symbol: str, price: float):
        """Close an existing position."""
        position = self.risk_manager.get_position(symbol)
        if position:
            quantity = position['quantity']
            
            # Execute sell order
            order_id = await self.order_manager.execute_sell(symbol, quantity)
            if order_id:
                # Get order details
                order = await self.order_manager.get_order(order_id)
                if order.get('status') == 'filled':
                    # Close position in risk manager
                    self.risk_manager.close_position(
                        symbol, 
                        order.get('price', price)
                    )
                    print(f"Closed position in {symbol}")
                
    async def start(self):
        """Start the trading bot."""
        print(f"Starting trading bot at {datetime.now()}")
        
        # Set initial balance
        await self.risk_manager.set_starting_balance()
        
        # Register callback for data updates
        self.data_feed.register_callback(self.on_data_update)
        
        # Start data feed
        self.running = True
        await self.data_feed.start()
        
        # Start monitor
        await self.monitor.start()
        
        print("Trading bot started successfully")
        
    async def stop(self):
        """Stop the trading bot."""
        print(f"Stopping trading bot at {datetime.now()}")
        
        self.running = False
        
        # Stop data feed
        await self.data_feed.stop()
        
        # Stop monitor
        await self.monitor.stop()
        
        print("Trading bot stopped successfully")