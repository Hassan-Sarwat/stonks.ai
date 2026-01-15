"""
SUI Order Manager for executing trades on Cetus DEX.
"""
import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional
from decimal import Decimal

from src.dex.cetus_client import CetusDexClient

class SuiOrderManager:
    def __init__(self, cetus_client: CetusDexClient, dry_run: bool = False):
        """
        Initialize SUI order manager.
        
        Args:
            cetus_client: Initialized Cetus DEX client
            dry_run: If True, simulate orders without executing them
        """
        self.cetus_client = cetus_client
        self.dry_run = dry_run
        self.orders: Dict[str, Dict[str, Any]] = {}
        
    async def execute_buy(
        self,
        symbol: str,
        quantity: float,
        max_price: Optional[float] = None,
        slippage_pct: float = 0.5
    ) -> str:
        """
        Execute a buy order on Cetus DEX.
        
        Args:
            symbol: Trading pair symbol (e.g., "SUI_USDC")
            quantity: Amount of base token to buy
            max_price: Maximum price willing to pay (optional)
            slippage_pct: Maximum acceptable slippage percentage
            
        Returns:
            Order ID (local tracking ID)
        """
        order_id = str(uuid.uuid4())
        
        # Parse symbol to get token_in and token_out
        tokens = symbol.split('_')
        if len(tokens) != 2:
            raise ValueError(f"Invalid symbol format: {symbol}. Expected format: TOKEN_IN_TOKEN_OUT")
            
        token_out, token_in = tokens  # For buying SUI with USDC, token_in=USDC, token_out=SUI
        
        if self.dry_run:
            # Simulate order execution
            current_price = await self.cetus_client.get_price(symbol)
            amount_in = quantity * float(current_price)
            
            print(f"[DRY RUN] BUY {quantity} {token_out} with {amount_in} {token_in} at price {current_price}")
            
            self.orders[order_id] = {
                'symbol': symbol,
                'side': 'buy',
                'quantity': quantity,
                'price': float(current_price),
                'status': 'filled',
                'timestamp': time.time(),
                'token_in': token_in,
                'token_out': token_out,
                'amount_in': amount_in,
                'amount_out': quantity
            }
            
            return order_id
            
        try:
            # Get current price
            current_price = await self.cetus_client.get_price(symbol)
            
            # Check max_price if provided
            if max_price is not None and float(current_price) > max_price:
                print(f"Current price {current_price} exceeds max price {max_price}")
                self.orders[order_id] = {
                    'symbol': symbol,
                    'side': 'buy',
                    'quantity': quantity,
                    'price': float(current_price),
                    'status': 'rejected',
                    'timestamp': time.time(),
                    'error': 'Price exceeds max_price'
                }
                return order_id
            
            # Calculate amount_in based on quantity and price
            amount_in = quantity * float(current_price)
            
            # Apply slippage to calculate min_amount_out
            min_amount_out = int(quantity * (1 - slippage_pct / 100))
            
            # Execute swap
            result = await self.cetus_client.swap(
                token_in=token_in,
                token_out=token_out,
                amount_in=int(amount_in * 10**9),  # Convert to smallest unit (e.g., 9 decimals for SUI)
                min_amount_out=min_amount_out * 10**9  # Convert to smallest unit
            )
            
            # Store order details
            self.orders[order_id] = {
                'symbol': symbol,
                'side': 'buy',
                'quantity': quantity,
                'price': float(current_price),
                'status': 'filled' if result.get('effects', {}).get('status', {}).get('status') == 'success' else 'failed',
                'timestamp': time.time(),
                'transaction_id': result.get('certificate', {}).get('transactionDigest'),
                'token_in': token_in,
                'token_out': token_out,
                'amount_in': amount_in,
                'amount_out': quantity
            }
            
            print(f"BUY order executed: {quantity} {token_out} with {amount_in} {token_in}, Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            print(f"Error executing buy order: {e}")
            
            # Store failed order
            self.orders[order_id] = {
                'symbol': symbol,
                'side': 'buy',
                'quantity': quantity,
                'status': 'failed',
                'timestamp': time.time(),
                'error': str(e)
            }
            
            return order_id
            
    async def execute_sell(
        self,
        symbol: str,
        quantity: float,
        min_price: Optional[float] = None,
        slippage_pct: float = 0.5
    ) -> str:
        """
        Execute a sell order on Cetus DEX.
        
        Args:
            symbol: Trading pair symbol (e.g., "SUI_USDC")
            quantity: Amount of base token to sell
            min_price: Minimum price willing to accept (optional)
            slippage_pct: Maximum acceptable slippage percentage
            
        Returns:
            Order ID (local tracking ID)
        """
        order_id = str(uuid.uuid4())
        
        # Parse symbol to get token_in and token_out
        tokens = symbol.split('_')
        if len(tokens) != 2:
            raise ValueError(f"Invalid symbol format: {symbol}. Expected format: TOKEN_IN_TOKEN_OUT")
            
        token_in, token_out = tokens  # For selling SUI for USDC, token_in=SUI, token_out=USDC
        
        if self.dry_run:
            # Simulate order execution
            current_price = await self.cetus_client.get_price(symbol)
            amount_out = quantity * float(current_price)
            
            print(f"[DRY RUN] SELL {quantity} {token_in} for {amount_out} {token_out} at price {current_price}")
            
            self.orders[order_id] = {
                'symbol': symbol,
                'side': 'sell',
                'quantity': quantity,
                'price': float(current_price),
                'status': 'filled',
                'timestamp': time.time(),
                'token_in': token_in,
                'token_out': token_out,
                'amount_in': quantity,
                'amount_out': amount_out
            }
            
            return order_id
            
        try:
            # Get current price
            current_price = await self.cetus_client.get_price(symbol)
            
            # Check min_price if provided
            if min_price is not None and float(current_price) < min_price:
                print(f"Current price {current_price} is below min price {min_price}")
                self.orders[order_id] = {
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': quantity,
                    'price': float(current_price),
                    'status': 'rejected',
                    'timestamp': time.time(),
                    'error': 'Price below min_price'
                }
                return order_id
            
            # Calculate expected amount_out based on quantity and price
            expected_amount_out = quantity * float(current_price)
            
            # Apply slippage to calculate min_amount_out
            min_amount_out = int(expected_amount_out * (1 - slippage_pct / 100))
            
            # Execute swap
            result = await self.cetus_client.swap(
                token_in=token_in,
                token_out=token_out,
                amount_in=int(quantity * 10**9),  # Convert to smallest unit (e.g., 9 decimals for SUI)
                min_amount_out=min_amount_out * 10**9  # Convert to smallest unit
            )
            
            # Store order details
            self.orders[order_id] = {
                'symbol': symbol,
                'side': 'sell',
                'quantity': quantity,
                'price': float(current_price),
                'status': 'filled' if result.get('effects', {}).get('status', {}).get('status') == 'success' else 'failed',
                'timestamp': time.time(),
                'transaction_id': result.get('certificate', {}).get('transactionDigest'),
                'token_in': token_in,
                'token_out': token_out,
                'amount_in': quantity,
                'amount_out': expected_amount_out
            }
            
            print(f"SELL order executed: {quantity} {token_in} for {expected_amount_out} {token_out}, Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            print(f"Error executing sell order: {e}")
            
            # Store failed order
            self.orders[order_id] = {
                'symbol': symbol,
                'side': 'sell',
                'quantity': quantity,
                'status': 'failed',
                'timestamp': time.time(),
                'error': str(e)
            }
            
            return order_id
            
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details by ID."""
        return self.orders.get(order_id, {})
        
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all orders, optionally filtered by symbol and/or status."""
        filtered_orders = []
        
        for order_id, order in self.orders.items():
            if symbol and order.get('symbol') != symbol:
                continue
                
            if status and order.get('status') != status:
                continue
                
            filtered_orders.append({**order, 'order_id': order_id})
            
        return filtered_orders