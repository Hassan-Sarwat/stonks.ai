"""
SUI Risk Manager for managing trading risk on Cetus DEX.
"""
import asyncio
from typing import Dict, Any, List, Optional
from decimal import Decimal
import time

from src.dex.cetus_client import CetusDexClient

class SuiRiskManager:
    def __init__(
        self,
        cetus_client: CetusDexClient,
        max_position_size: float = 0.1,  # Max 10% of portfolio in one position
        max_daily_drawdown: float = 0.05,  # Max 5% daily drawdown
        stop_loss_pct: float = 0.03,  # 3% stop loss
        take_profit_pct: float = 0.05  # 5% take profit
    ):
        """
        Initialize SUI risk manager.
        
        Args:
            cetus_client: Initialized Cetus DEX client
            max_position_size: Maximum position size as fraction of portfolio
            max_daily_drawdown: Maximum daily drawdown as fraction of portfolio
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.cetus_client = cetus_client
        self.max_position_size = max_position_size
        self.max_daily_drawdown = max_daily_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.daily_pnl = 0.0
        self.starting_balance = 0.0
        self.last_balance_check = 0
        self.portfolio_value = 0.0
        
    async def set_starting_balance(self):
        """Set the starting balance for the day."""
        # Get SUI balance
        sui_balance = await self.cetus_client.sui_client.get_balance()
        
        # Get USDC balance (if available)
        usdc_balance = 0
        try:
            usdc_balance = await self.cetus_client.sui_client.get_balance(
                self.cetus_client.tokens.get("USDC", "")
            )
        except:
            pass
            
        # Get current SUI price in USDC
        sui_price = 1.0
        try:
            sui_price = float(await self.cetus_client.get_price("SUI_USDC"))
        except:
            pass
            
        # Calculate total portfolio value in USDC
        self.portfolio_value = (sui_balance * sui_price) + usdc_balance
        self.starting_balance = self.portfolio_value
        self.daily_pnl = 0.0
        self.last_balance_check = time.time()
        
        return self.portfolio_value
        
    async def update_portfolio_value(self):
        """Update the current portfolio value."""
        # Only update once every 5 minutes
        if time.time() - self.last_balance_check < 300:
            return self.portfolio_value
            
        await self.set_starting_balance()
        return self.portfolio_value
        
    async def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate the appropriate position size based on risk parameters."""
        # Update portfolio value
        portfolio_value = await self.update_portfolio_value()
        
        # Calculate maximum position size
        max_amount = portfolio_value * self.max_position_size
        
        # Calculate quantity based on price
        quantity = max_amount / price
        
        return quantity
        
    async def should_take_trade(self, symbol: str, side: str, price: float) -> bool:
        """Determine if a trade should be taken based on risk parameters."""
        # Check daily drawdown limit
        if self.daily_pnl < -self.max_daily_drawdown * self.starting_balance:
            print(f"Daily drawdown limit reached: {self.daily_pnl}")
            return False
            
        # Check if we already have a position in this symbol
        if symbol in self.positions and side == 'buy':
            print(f"Already have a position in {symbol}")
            return False
            
        # Check portfolio value
        portfolio_value = await self.update_portfolio_value()
        
        # Additional risk checks can be added here
        
        return True
        
    def register_position(self, symbol: str, entry_price: float, quantity: float):
        """Register a new position."""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': entry_price * (1 - self.stop_loss_pct),
            'take_profit': entry_price * (1 + self.take_profit_pct),
            'entry_time': time.time()
        }
        
    def close_position(self, symbol: str, exit_price: float):
        """Close a position and update PnL."""
        if symbol in self.positions:
            position = self.positions[symbol]
            pnl = (exit_price - position['entry_price']) * position['quantity']
            self.daily_pnl += pnl
            print(f"Closed position in {symbol}: PnL = {pnl}")
            del self.positions[symbol]
            return pnl
        return 0.0
        
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> str:
        """Check if stop loss or take profit has been hit."""
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        if current_price <= position['stop_loss']:
            return 'stop_loss'
        elif current_price >= position['take_profit']:
            return 'take_profit'
            
        return None
        
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position details for a symbol."""
        return self.positions.get(symbol)
        
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all positions."""
        return self.positions