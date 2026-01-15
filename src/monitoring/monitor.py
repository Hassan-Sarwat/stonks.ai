"""
Monitoring module for the trading bot.
"""
import asyncio
import time
import requests
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime

class Monitor:
    def __init__(self, bot, notification_config: Dict[str, str] = None):
        """
        Initialize the monitoring system.
        
        Args:
            bot: The trading bot instance
            notification_config: Dictionary with notification configuration
        """
        self.bot = bot
        self.notification_config = notification_config or {}
        self.running = False
        self.monitor_task = None
        self.last_notification_time = {}
        
    async def start(self):
        """Start the monitoring system."""
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Stop the monitoring system."""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            
    def send_telegram_notification(self, message: str, level: str = 'info') -> bool:
        """Send a notification via Telegram."""
        bot_token = self.notification_config.get('telegram_bot_token')
        chat_id = self.notification_config.get('telegram_chat_id')
        
        if not bot_token or not chat_id:
            print("Telegram notification not configured")
            return False
            
        # Rate limit notifications (no more than 1 per minute)
        current_time = time.time()
        if 'telegram' in self.last_notification_time:
            if current_time - self.last_notification_time['telegram'] < 60:
                return False
                
        self.last_notification_time['telegram'] = current_time
        
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': f"[{level.upper()}] {message}",
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")
            return False
            
    def send_email_notification(self, message: str, level: str = 'info') -> bool:
        """Send a notification via email."""
        # This is a placeholder - in a real implementation, you would use an email library
        email = self.notification_config.get('email')
        
        if not email:
            print("Email notification not configured")
            return False
            
        # Rate limit notifications (no more than 1 per 5 minutes)
        current_time = time.time()
        if 'email' in self.last_notification_time:
            if current_time - self.last_notification_time['email'] < 300:
                return False
                
        self.last_notification_time['email'] = current_time
        
        print(f"[MOCK] Sending email to {email}: [{level.upper()}] {message}")
        return True
            
    def send_notification(self, message: str, level: str = 'info') -> bool:
        """Send a notification through all configured channels."""
        success = False
        
        # Try Telegram
        if self.notification_config.get('telegram_bot_token') and self.notification_config.get('telegram_chat_id'):
            if self.send_telegram_notification(message, level):
                success = True
                
        # Try Email
        if self.notification_config.get('email'):
            if self.send_email_notification(message, level):
                success = True
                
        # If no notifications were sent, print to console
        if not success:
            print(f"[{level.upper()}] {message}")
            
        return success
            
    async def _monitor_loop(self):
        """Main monitoring loop."""
        # Send startup notification
        self.send_notification("Trading bot started")
        
        while self.running:
            try:
                # Check data feed status
                if not self.bot.data_feed.running:
                    self.send_notification("Data feed is not running!", 'error')
                    
                # Check account balance
                portfolio_value = await self.bot.risk_manager.update_portfolio_value()
                
                # Check for significant balance changes
                if hasattr(self, 'last_portfolio_value'):
                    if self.last_portfolio_value > 0:
                        change_pct = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
                        if abs(change_pct) > 0.05:  # 5% change
                            direction = "increased" if change_pct > 0 else "decreased"
                            self.send_notification(
                                f"Portfolio value {direction} by {abs(change_pct)*100:.2f}% to {portfolio_value:.2f}",
                                'warning' if change_pct < 0 else 'info'
                            )
                            
                self.last_portfolio_value = portfolio_value
                
                # Check active positions
                positions = self.bot.risk_manager.get_all_positions()
                for symbol, position in positions.items():
                    entry_time = position.get('entry_time', 0)
                    duration = time.time() - entry_time
                    
                    # Alert for positions held for more than 24 hours
                    if duration > 86400:  # 24 hours
                        self.send_notification(
                            f"Position in {symbol} has been held for more than 24 hours",
                            'warning'
                        )
                        
                # Daily summary (at midnight)
                now = datetime.now()
                if now.hour == 0 and now.minute == 0:
                    daily_pnl = self.bot.risk_manager.daily_pnl
                    self.send_notification(
                        f"Daily summary: PnL = {daily_pnl:.2f}, Portfolio Value = {portfolio_value:.2f}",
                        'info'
                    )
                    # Reset daily PnL
                    self.bot.risk_manager.daily_pnl = 0.0
                    
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Sleep longer on error