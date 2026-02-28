"""
Trading Engine for executing arbitrage trades automatically.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from exchange_manager import ExchangeManager, ArbitrageOpportunity
from config import TRADING_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)

class TradeStatus(Enum):
    """Trade execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"

class OrderStatus(Enum):
    """Individual order status."""
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class Order:
    """Represents an individual order."""
    id: str
    exchange: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float]
    order_type: str  # 'limit' or 'market'
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    filled_amount: float = 0.0
    filled_price: Optional[float] = None
    fee: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None

@dataclass
class ArbitrageTrade:
    """Represents a complete arbitrage trade (buy + sell)."""
    id: str
    opportunity: ArbitrageOpportunity
    buy_order: Order
    sell_order: Order
    status: TradeStatus = TradeStatus.PENDING
    actual_profit_usd: float = 0.0
    execution_time_ms: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None

class TradingEngine:
    """Executes arbitrage trades automatically."""
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.active_trades: Dict[str, ArbitrageTrade] = {}
        self.completed_trades: List[ArbitrageTrade] = []
        self.is_trading_enabled = False
        self.daily_stats = {
            'trades_executed': 0,
            'total_profit_usd': 0.0,
            'total_loss_usd': 0.0,
            'successful_trades': 0,
            'failed_trades': 0,
            'start_time': time.time()
        }
        
        # Risk management
        self.circuit_breaker_triggered = False
        self.last_balance_check = 0
        
        # Performance tracking
        self.execution_times = []
        self.order_fill_rates = {'buy': [], 'sell': []}
    
    def enable_trading(self):
        """Enable automatic trading."""
        if self.circuit_breaker_triggered:
            logger.error("Cannot enable trading: circuit breaker is triggered")
            return False
        
        self.is_trading_enabled = True
        logger.info("Trading enabled")
        return True
    
    def disable_trading(self):
        """Disable automatic trading."""
        self.is_trading_enabled = False
        logger.info("Trading disabled")
    
    def trigger_circuit_breaker(self, reason: str):
        """Trigger emergency stop."""
        self.circuit_breaker_triggered = True
        self.is_trading_enabled = False
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
        
        # Cancel all active orders
        asyncio.create_task(self._emergency_cancel_all_orders())
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Optional[ArbitrageTrade]:
        """Execute an arbitrage trade."""
        if not self.is_trading_enabled:
            logger.debug("Trading is disabled, skipping opportunity")
            return None
        
        if self.circuit_breaker_triggered:
            logger.warning("Circuit breaker triggered, cannot execute trades")
            return None
        
        # Check daily limits
        if not self._check_daily_limits():
            return None
        
        # Check risk limits
        if not self._check_risk_limits(opportunity):
            return None
        
        # Calculate optimal trade size
        trade_amount = self._calculate_trade_size(opportunity)
        if trade_amount <= 0:
            logger.debug("Trade amount too small, skipping")
            return None
        
        # Create trade object
        trade_id = str(uuid.uuid4())
        trade = self._create_arbitrage_trade(trade_id, opportunity, trade_amount)
        
        # Add to active trades
        self.active_trades[trade_id] = trade
        
        # Execute the trade
        try:
            await self._execute_trade(trade)
            return trade
        except Exception as e:
            logger.error(f"Failed to execute trade {trade_id}: {e}")
            trade.status = TradeStatus.FAILED
            trade.error_message = str(e)
            self._move_to_completed(trade)
            return trade
    
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits are exceeded."""
        if self.daily_stats['trades_executed'] >= TRADING_CONFIG['max_daily_trades']:
            logger.warning("Daily trade limit reached")
            return False
        
        if self.daily_stats['total_loss_usd'] >= RISK_CONFIG['max_daily_loss_usd']:
            logger.warning("Daily loss limit reached")
            self.trigger_circuit_breaker("Daily loss limit exceeded")
            return False
        
        return True
    
    def _check_risk_limits(self, opportunity: ArbitrageOpportunity) -> bool:
        """Check risk limits for the opportunity."""
        # Check maximum open positions
        if len(self.active_trades) >= RISK_CONFIG['max_open_positions']:
            logger.debug("Maximum open positions reached")
            return False
        
        # Check if we have sufficient balance
        base_currency = opportunity.symbol.split('/')[0]
        quote_currency = opportunity.symbol.split('/')[1]
        
        # Check balances
        buy_balance = self.exchange_manager.get_balance(
            opportunity.buy_exchange, quote_currency
        )
        sell_balance = self.exchange_manager.get_balance(
            opportunity.sell_exchange, base_currency
        )
        
        required_quote = opportunity.max_quantity * opportunity.buy_price
        
        if buy_balance < required_quote or sell_balance < opportunity.max_quantity:
            logger.debug("Insufficient balance for trade")
            return False
        
        return True
    
    def _calculate_trade_size(self, opportunity: ArbitrageOpportunity) -> float:
        """Calculate optimal trade size based on available capital, risk limits, and opportunity liquidity."""
        # The max_quantity from the opportunity is already calculated based on order book depth and profitability
        # in price_monitor.py. We will use this as the primary determinant.
        
        # Get available balances
        base_currency = opportunity.symbol.split("/")[0]
        quote_currency = opportunity.symbol.split("/")[1]
        
        buy_balance = self.exchange_manager.get_balance(
            opportunity.buy_exchange, quote_currency
        )
        sell_balance = self.exchange_manager.get_balance(
            opportunity.sell_exchange, base_currency
        )
        
        # Calculate maximum tradeable amount based on balances
        max_by_buy_balance = buy_balance / opportunity.buy_price if opportunity.buy_price > 0 else 0
        max_by_sell_balance = sell_balance
        
        # Take the minimum of the opportunity's max_quantity and available balances
        max_amount = min(opportunity.max_quantity, max_by_buy_balance, max_by_sell_balance)
        
        # Apply overall trade size limits from config (e.g., MAX_TRADE_AMOUNT_USD)
        max_trade_value_usd = TRADING_CONFIG["max_trade_amount_usd"]
        max_by_trade_limit = max_trade_value_usd / opportunity.buy_price if opportunity.buy_price > 0 else 0
        
        final_amount = min(max_amount, max_by_trade_limit)
        
        # Ensure minimum trade size (e.g., $10 equivalent)
        min_trade_value = 10.0
        min_amount = min_trade_value / opportunity.buy_price if opportunity.buy_price > 0 else 0
        
        if final_amount < min_amount:
            return 0.0
        
        # Dynamic capital allocation based on opportunity score (higher score, potentially larger trade)
        # This is a simple linear scaling; more complex models could be used.
        # Assuming score is between 0 and 100
        score_factor = opportunity.score / 100.0 if opportunity.score else 0.5 # Default to 0.5 if no score
        final_amount *= score_factor

        return final_amount
    
    def _create_arbitrage_trade(
        self, 
        trade_id: str, 
        opportunity: ArbitrageOpportunity, 
        amount: float
    ) -> ArbitrageTrade:
        """Create an ArbitrageTrade object."""
        
        # Create buy order
        buy_order = Order(
            id=f"{trade_id}_buy",
            exchange=opportunity.buy_exchange,
            symbol=opportunity.symbol,
            side='buy',
            amount=amount,
            price=opportunity.buy_price,
            order_type='limit'
        )
        
        # Create sell order
        sell_order = Order(
            id=f"{trade_id}_sell",
            exchange=opportunity.sell_exchange,
            symbol=opportunity.symbol,
            side='sell',
            amount=amount,
            price=opportunity.sell_price,
            order_type='limit'
        )
        
        return ArbitrageTrade(
            id=trade_id,
            opportunity=opportunity,
            buy_order=buy_order,
            sell_order=sell_order
        )
    
    async def _execute_trade(self, trade: ArbitrageTrade):
        """Execute a complete arbitrage trade."""
        start_time = time.time()
        trade.status = TradeStatus.EXECUTING
        
        logger.info(f"Executing arbitrage trade {trade.id}: "
                   f"{trade.opportunity.symbol} - "
                   f"Buy {trade.buy_order.amount:.6f} on {trade.buy_order.exchange} "
                   f"at {trade.buy_order.price:.6f}, "
                   f"Sell {trade.sell_order.amount:.6f} on {trade.sell_order.exchange} "
                   f"at {trade.sell_order.price:.6f}")
        
        try:
            # Pre-trade slippage estimation
            estimated_buy_slippage = await self._estimate_slippage(
                trade.buy_order.exchange, trade.buy_order.symbol, trade.buy_order.side, 
                trade.buy_order.amount, trade.buy_order.price
            )
            estimated_sell_slippage = await self._estimate_slippage(
                trade.sell_order.exchange, trade.sell_order.symbol, trade.sell_order.side, 
                trade.sell_order.amount, trade.sell_order.price
            )

            total_estimated_slippage = estimated_buy_slippage + estimated_sell_slippage
            expected_profit_usd = trade.opportunity.potential_profit_usd

            if expected_profit_usd > 0 and total_estimated_slippage > TRADING_CONFIG["pre_trade_slippage_estimation_threshold"] * expected_profit_usd:
                logger.warning(f"Trade {trade.id} skipped due to high estimated slippage: {total_estimated_slippage:.4f} > {TRADING_CONFIG['pre_trade_slippage_estimation_threshold'] * expected_profit_usd:.4f}")
                trade.status = TradeStatus.FAILED
                trade.error_message = "High estimated slippage"
                self._move_to_completed(trade)
                return

            # Adaptive Limit Orders
            buy_price = trade.buy_order.price
            sell_price = trade.sell_order.price

            if trade.opportunity.score and trade.opportunity.score > 70: # Example: high score opportunities
                # Make buy order slightly more aggressive (higher price)
                buy_price *= (1 + TRADING_CONFIG["adaptive_limit_order_aggressiveness"])
                # Make sell order slightly more aggressive (lower price)
                sell_price *= (1 - TRADING_CONFIG["adaptive_limit_order_aggressiveness"])
                logger.info(f"Adjusting limit orders for high-score opportunity {trade.id}. New buy price: {buy_price:.6f}, New sell price: {sell_price:.6f}")

            trade.buy_order.price = buy_price
            trade.sell_order.price = sell_price

            # Place both orders simultaneously
            buy_task = asyncio.create_task(self._place_order(trade.buy_order))
            sell_task = asyncio.create_task(self._place_order(trade.sell_order))
            
            # Wait for both orders to be placed
            await asyncio.gather(buy_task, sell_task)
            # Monitor order execution
            await self._monitor_trade_execution(trade)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # in milliseconds
            trade.execution_time_ms = execution_time
            self.execution_times.append(execution_time)
            
            # Calculate actual profit
            self._calculate_actual_profit(trade)
            
            # Update statistics
            self._update_trade_statistics(trade)
            
            logger.info(f"Trade {trade.id} completed with status: {trade.status.value}, "
                       f"Actual profit: ${trade.actual_profit_usd:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trade {trade.id}: {e}")
            trade.status = TradeStatus.FAILED
            trade.error_message = str(e)
            
            # Try to cancel any pending orders
            await self._cancel_trade_orders(trade)
        
        finally:
            # Move to completed trades
            self._move_to_completed(trade)
    
    async def _place_order(self, order: Order):
        """Place an individual order."""
        try:
            logger.debug(f"Placing {order.side} order {order.id} on {order.exchange}")
            
            # Place the order
            exchange_order = await self.exchange_manager.place_order(
                exchange_name=order.exchange,
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                price=order.price,
                order_type=order.order_type
            )
            
            # Update order status
            order.exchange_order_id = exchange_order['id']
            order.status = OrderStatus.PLACED
            
            logger.debug(f"Order {order.id} placed successfully: {order.exchange_order_id}")
            
        except Exception as e:
            logger.error(f"Failed to place order {order.id}: {e}")
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            raise
    
    async def _monitor_trade_execution(self, trade: ArbitrageTrade):
        """Monitor the execution of both orders in a trade."""
        timeout = TRADING_CONFIG['order_timeout_seconds']
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check buy order status
            if trade.buy_order.status == OrderStatus.PLACED:
                await self._update_order_status(trade.buy_order)
            
            # Check sell order status
            if trade.sell_order.status == OrderStatus.PLACED:
                await self._update_order_status(trade.sell_order)
            
            # Check if both orders are filled
            if (trade.buy_order.status == OrderStatus.FILLED and 
                trade.sell_order.status == OrderStatus.FILLED):
                trade.status = TradeStatus.COMPLETED
                break
            
            # Check if any order failed
            if (trade.buy_order.status == OrderStatus.FAILED or 
                trade.sell_order.status == OrderStatus.FAILED):
                trade.status = TradeStatus.FAILED
                break
            
            # Small delay before next check
            await asyncio.sleep(0.1)
        
        # Handle timeout
        if trade.status == TradeStatus.EXECUTING:
            logger.warning(f"Trade {trade.id} timed out")
            await self._handle_timeout(trade)
    
    async def _update_order_status(self, order: Order):
        """Update the status of an individual order."""
        try:
            exchange_order = await self.exchange_manager.get_order_status(
                order.exchange, order.exchange_order_id, order.symbol
            )
            
            # Update order details
            if exchange_order['status'] == 'closed':
                order.status = OrderStatus.FILLED
                order.filled_amount = exchange_order['filled']
                order.filled_price = exchange_order['average']
                order.fee = exchange_order.get('fee', {}).get('cost', 0)
            elif exchange_order['status'] == 'canceled':
                order.status = OrderStatus.CANCELLED
            elif exchange_order['filled'] > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_amount = exchange_order['filled']
                order.filled_price = exchange_order['average']
                order.fee = exchange_order.get('fee', {}).get('cost', 0)
            
        except Exception as e:
            logger.error(f"Failed to update order status for {order.id}: {e}")
    
    async def _handle_timeout(self, trade: ArbitrageTrade):
        """Handle trade timeout by cancelling unfilled orders."""
        logger.warning(f"Handling timeout for trade {trade.id}")
        
        # Cancel unfilled orders
        await self._cancel_trade_orders(trade)
        
        # Determine final trade status
        buy_filled = trade.buy_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
        sell_filled = trade.sell_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
        
        if buy_filled and sell_filled:
            trade.status = TradeStatus.PARTIALLY_FILLED
        elif buy_filled or sell_filled:
            trade.status = TradeStatus.PARTIALLY_FILLED
            # Log exposure warning
            logger.warning(f"Trade {trade.id} has unmatched position - manual intervention may be required")
        else:
            trade.status = TradeStatus.CANCELLED
    
    async def _cancel_trade_orders(self, trade: ArbitrageTrade):
        """Cancel all orders in a trade."""
        tasks = []
        
        if (trade.buy_order.status == OrderStatus.PLACED and 
            trade.buy_order.exchange_order_id):
            tasks.append(self._cancel_order(trade.buy_order))
        
        if (trade.sell_order.status == OrderStatus.PLACED and 
            trade.sell_order.exchange_order_id):
            tasks.append(self._cancel_order(trade.sell_order))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _cancel_order(self, order: Order):
        """Cancel an individual order."""
        try:
            success = await self.exchange_manager.cancel_order(
                order.exchange, order.exchange_order_id, order.symbol
            )
            if success:
                order.status = OrderStatus.CANCELLED
                logger.debug(f"Cancelled order {order.id}")
        except Exception as e:
            logger.error(f"Failed to cancel order {order.id}: {e}")
    
    def _calculate_actual_profit(self, trade: ArbitrageTrade):
        """Calculate the actual profit/loss from a completed trade."""
        buy_order = trade.buy_order
        sell_order = trade.sell_order
        
        if (buy_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED] and
            sell_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]):
            
            # Use the minimum filled amount
            executed_amount = min(buy_order.filled_amount, sell_order.filled_amount)
            
            if executed_amount > 0:
                # Calculate revenue and costs
                revenue = executed_amount * (sell_order.filled_price or sell_order.price)
                cost = executed_amount * (buy_order.filled_price or buy_order.price)
                fees = buy_order.fee + sell_order.fee
                
                trade.actual_profit_usd = revenue - cost - fees
            else:
                trade.actual_profit_usd = -(buy_order.fee + sell_order.fee)  # Only fees lost
        else:
            trade.actual_profit_usd = 0.0
    
    def _update_trade_statistics(self, trade: ArbitrageTrade):
        """Update daily trading statistics."""
        self.daily_stats['trades_executed'] += 1
        
        if trade.status == TradeStatus.COMPLETED:
            self.daily_stats['successful_trades'] += 1
            if trade.actual_profit_usd > 0:
                self.daily_stats['total_profit_usd'] += trade.actual_profit_usd
            else:
                self.daily_stats['total_loss_usd'] += abs(trade.actual_profit_usd)
        else:
            self.daily_stats['failed_trades'] += 1
            if trade.actual_profit_usd < 0:
                self.daily_stats['total_loss_usd'] += abs(trade.actual_profit_usd)
        
        # Check if single trade loss limit is exceeded
        if abs(trade.actual_profit_usd) > RISK_CONFIG['max_single_trade_loss_usd']:
            logger.warning(f"Single trade loss limit exceeded: ${abs(trade.actual_profit_usd):.2f}")
    
    def _move_to_completed(self, trade: ArbitrageTrade):
        """Move a trade from active to completed."""
        if trade.id in self.active_trades:
            del self.active_trades[trade.id]
        self.completed_trades.append(trade)
        
        # Keep only recent completed trades in memory
        if len(self.completed_trades) > 1000:
            self.completed_trades = self.completed_trades[-500:]
    
    async def _emergency_cancel_all_orders(self):
        """Cancel all active orders in emergency situations."""
        logger.critical("Emergency cancellation of all active orders")
        
        cancel_tasks = []
        for trade in self.active_trades.values():
            cancel_tasks.append(self._cancel_trade_orders(trade))
        
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
    
    def get_trading_statistics(self) -> Dict:
        """Get current trading statistics."""
        runtime_hours = (time.time() - self.daily_stats['start_time']) / 3600
        
        stats = self.daily_stats.copy()
        stats.update({
            'active_trades': len(self.active_trades),
            'completed_trades': len(self.completed_trades),
            'net_profit_usd': stats['total_profit_usd'] - stats['total_loss_usd'],
            'success_rate': (
                stats['successful_trades'] / max(stats['trades_executed'], 1) * 100
            ),
            'trades_per_hour': stats['trades_executed'] / max(runtime_hours, 0.01),
            'avg_execution_time_ms': (
                sum(self.execution_times) / len(self.execution_times)
                if self.execution_times else 0
            ),
            'is_trading_enabled': self.is_trading_enabled,
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
        })
        
        return stats
    
    def reset_daily_statistics(self):
        """Reset daily statistics (typically called at start of new day)."""
        self.daily_stats = {
            'trades_executed': 0,
            'total_profit_usd': 0.0,
            'total_loss_usd': 0.0,
            'successful_trades': 0,
            'failed_trades': 0,
            'start_time': time.time()
        }
        logger.info("Daily statistics reset")



    async def _estimate_slippage(self, exchange_id: str, symbol: str, side: str, amount: float, price: float) -> float:
        """Estimates potential slippage for a given order."""
        order_book = await self.exchange_manager.get_order_book(exchange_id, symbol, limit=TRADING_CONFIG["order_book_depth"])
        if not order_book:
            logger.warning(f"Could not fetch order book for slippage estimation on {exchange_id} {symbol}")
            return 0.0 # Cannot estimate slippage

        if side == "buy":
            # For a buy order, we are consuming asks
            levels = order_book.get("asks", [])
        else:
            # For a sell order, we are consuming bids
            levels = order_book.get("bids", [])

        filled_amount = 0.0
        cost = 0.0
        for level_price, level_quantity in levels:
            if filled_amount + level_quantity >= amount:
                # This level fills the remaining amount
                remaining_amount = amount - filled_amount
                cost += remaining_amount * level_price
                filled_amount = amount
                break
            else:
                # Consume the entire level
                cost += level_quantity * level_price
                filled_amount += level_quantity
        
        if filled_amount < amount:
            logger.warning(f"Not enough liquidity in order book to fill {amount} {symbol} on {exchange_id} for {side} order.")
            # If not enough liquidity, assume significant slippage or partial fill
            return 1.0 # Return a high slippage to indicate insufficient liquidity

        average_fill_price = cost / amount
        slippage_abs = abs(average_fill_price - price)
        slippage_pct = (slippage_abs / price) if price > 0 else 0.0
        return slippage_pct


