import asyncio
import time
import logging
import logging.handlers
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import uuid
import enum

from exchange_manager import ExchangeManager, ArbitrageOpportunity
from price_monitor import PriceMonitor
from safety_manager import SafetyManager
from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from monitoring import MonitoringSystem
from websocket_manager import WebSocketManager
from config import TRADING_CONFIG, RISK_CONFIG, PERFORMANCE_CONFIG, LOGGING_CONFIG

logger = logging.getLogger(__name__)

class TradeStatus(enum.Enum):
    PENDING = "PENDING"
    EXECUTING_BUY = "EXECUTING_BUY"
    BUY_FILLED = "BUY_FILLED"
    EXECUTING_SELL = "EXECUTING_SELL"
    SELL_FILLED = "SELL_FILLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    id: str
    opportunity: ArbitrageOpportunity
    amount: float
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    buy_price: Optional[float] = None
    sell_price: Optional[float] = None
    actual_profit_usd: float = 0.0
    execution_time_ms: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None

class TradingEngine:
    def __init__(self, exchange_manager: ExchangeManager, safety_manager: SafetyManager, error_handler: ErrorHandler, monitoring_system: MonitoringSystem):
        self.exchange_manager = exchange_manager
        self.safety_manager = safety_manager
        self.error_handler = error_handler
        self.monitoring_system = monitoring_system
        self.active_trades: Dict[str, Trade] = {}
        self.completed_trades: List[Trade] = []
        self.trading_enabled = True
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit_usd = 0.0
        self.todays_trades = 0
        self.last_day_reset = datetime.now().day

    def enable_trading(self) -> bool:
        if self.safety_manager.is_circuit_breaker_active():
            logger.warning("Cannot enable trading: Circuit breaker is active.")
            self.monitoring_system.alert_manager.create_alert(
                "Trading Enable Failed", "Attempted to enable trading while circuit breaker is active.", "warning", "TradingEngine"
            )
            return False
        self.trading_enabled = True
        logger.info("Trading enabled.")
        self.monitoring_system.alert_manager.create_alert(
            "Trading Enabled", "Trading has been re-enabled.", "info", "TradingEngine"
        )
        return True

    def disable_trading(self):
        self.trading_enabled = False
        logger.warning("Trading disabled.")
        self.monitoring_system.alert_manager.create_alert(
            "Trading Disabled", "Trading has been manually disabled.", "warning", "TradingEngine"
        )


    async def execute_arbitrage_trade(self, opportunity: ArbitrageOpportunity):
      if not self.trading_enabled:
         logger.info(f"Skipping trade for {opportunity.symbol} due to trading being disabled.")
         return
  
      if self.safety_manager.is_circuit_breaker_active():
         logger.warning(f"Skipping trade for {opportunity.symbol}: Circuit breaker active.")
         self.monitoring_system.alert_manager.create_alert(
             "Trade Skipped", f"Trade for {opportunity.symbol} skipped due to active circuit breaker.", "warning", "TradingEngine"
         )
         return
  
      trade_id = str(uuid.uuid4())
      
      # Dynamic position sizing
      dynamic_trade_amount_usd = self.safety_manager.get_dynamic_trade_size(
          opportunity.symbol,
          opportunity.potential_profit_pct,
          self.exchange_manager.get_exchange_volatility(opportunity.buy_exchange, opportunity.symbol) # Using buy exchange for volatility
      )
      
      # Convert USD amount to base currency amount using the buy price
      # Ensure buy_price is not zero to avoid division by zero
      if opportunity.buy_price and opportunity.buy_price > 0:
          trade_amount = dynamic_trade_amount_usd / opportunity.buy_price
      else:
          logger.warning(f"Invalid buy price ({opportunity.buy_price}) for {opportunity.symbol}. Cannot determine trade amount.")
          return

      trade = Trade(id=trade_id, opportunity=opportunity, amount=trade_amount)
      self.active_trades[trade_id] = trade
      self.total_trades += 1
  
      # Reset daily trades if day changed
      if datetime.now().day != self.last_day_reset:
         self.todays_trades = 0
         self.last_day_reset = datetime.now().day
      self.todays_trades += 1
  
      logger.info(f"Attempting arbitrage trade {trade_id} for {opportunity.symbol} between {opportunity.buy_exchange} and {opportunity.sell_exchange} with amount {trade.amount:.4f}")
      self.monitoring_system.alert_manager.create_alert(
         "Trade Attempt", f"Attempting arbitrage for {opportunity.symbol}. Profit: {opportunity.potential_profit_pct:.2f}%", "info", "TradingEngine"
     )
  
      buy_fee = 0.0
      sell_fee = 0.0
  
      try:
         ticker = await self.exchange_manager.fetch_ticker(opportunity.buy_exchange, opportunity.symbol)
         if ticker is None or ticker.get("ask") is None:
            raise ValueError(f"Could not fetch ticker or ask price for {opportunity.symbol} on {opportunity.buy_exchange}")
         opportunity.buy_price = float(ticker["ask"])  # Use the ask price for buying
        
        
         # --- STEP 1: PLACE BUY ORDER ---
         trade.status = TradeStatus.EXECUTING_BUY
         logger.info(f"Placing buy order for {trade.amount} {opportunity.symbol} on {opportunity.buy_exchange} at {opportunity.buy_price}")

         buy_order = await self.exchange_manager.place_order(
             opportunity.buy_exchange, opportunity.symbol, "limit", "buy", trade.amount, opportunity.buy_price
         )
 
         # If missing price, fetch details
         if not buy_order or buy_order.get("price") is None:
             if buy_order and buy_order.get("id"):
                 logger.warning(f"Buy order missing price, fetching details for {buy_order['id']}...")
                 buy_order = await self.exchange_manager.fetch_order(opportunity.buy_exchange, opportunity.symbol, buy_order["id"])
 
         # Validate
         if not buy_order or buy_order.get("price") is None:
             raise ValueError(f"Buy order failed or returned invalid price after fetch: {buy_order}")
 
         trade.buy_order_id = buy_order["id"]
         trade.buy_price = float(buy_order.get("price", opportunity.buy_price))
         trade.status = TradeStatus.BUY_FILLED
         logger.info(f"Buy order {trade.buy_order_id} filled on {opportunity.buy_exchange}.")
 
         # --- STEP 2: PLACE SELL ORDER ---
         trade.status = TradeStatus.EXECUTING_SELL
         logger.info(f"Placing sell order for {trade.amount} {opportunity.symbol} on {opportunity.sell_exchange} at {opportunity.sell_price}")
 
         sell_order = await self.exchange_manager.place_order(
             opportunity.sell_exchange, opportunity.symbol, "limit", "sell", trade.amount, opportunity.sell_price
         )
 
         # If missing price, fetch details
         if not sell_order or sell_order.get("price") is None:
             if sell_order and sell_order.get("id"):
                 logger.warning(f"Sell order missing price, fetching details for {sell_order['id']}...")
                 sell_order = await self.exchange_manager.fetch_order(opportunity.sell_exchange, opportunity.symbol, sell_order["id"])
 
         trade.sell_order_id = sell_order["id"] if sell_order else None
         trade.sell_price = float(sell_order.get("price", opportunity.sell_price)) if sell_order else opportunity.sell_price
         trade.status = TradeStatus.SELL_FILLED
         logger.info(f"Sell order {trade.sell_order_id} filled on {opportunity.sell_exchange}.")
 
         # --- STEP 3: CALCULATE PROFIT ---
         revenue = trade.amount * trade.sell_price
         cost = trade.amount * trade.buy_price
 
         buy_fee = trade.amount * trade.buy_price * self.exchange_manager.get_exchange_trading_fee(opportunity.buy_exchange)
         sell_fee = trade.amount * trade.sell_price * self.exchange_manager.get_exchange_trading_fee(opportunity.sell_exchange)
         total_fees = buy_fee + sell_fee
 
         trade.actual_profit_usd = revenue - cost - total_fees
         trade.execution_time_ms = (time.time() - trade.timestamp) * 1000
         trade.status = TradeStatus.COMPLETED
 
         self.successful_trades += 1
         self.total_profit_usd += trade.actual_profit_usd
         self.safety_manager.record_profit(trade.actual_profit_usd)
 
         logger.info(f"Trade {trade_id} completed. Profit: ${trade.actual_profit_usd:.2f}")
         self.monitoring_system.alert_manager.create_alert(
             "Trade Completed", f"Trade {trade_id} for {opportunity.symbol} completed. Profit: ${trade.actual_profit_usd:.2f}", "success", "TradingEngine"
         )
  
      except Exception as e:
         trade.status = TradeStatus.FAILED
         trade.error_message = str(e)
         potential_loss = -(trade.amount * trade.buy_price) if trade.buy_price else 0.0
         self.safety_manager.record_loss(potential_loss)
         logger.error(f"Trade {trade_id} failed: {e}")
         self.error_handler.handle_error(e, ErrorCategory.TRADING, ErrorSeverity.HIGH, "TradingEngine", f"Trade execution failed for {opportunity.symbol}")
         self.monitoring_system.alert_manager.create_alert(
             "Trade Failed", f"Trade {trade_id} for {opportunity.symbol} failed: {e}", "error", "TradingEngine"
         )
      finally:
         self.completed_trades.append(trade)
         if trade_id in self.active_trades:
            del self.active_trades[trade_id]

    def get_trading_statistics(self) -> Dict[str, Any]:
        success_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "success_rate": success_rate,
            "total_profit_usd": self.total_profit_usd,
            "todays_trades": self.todays_trades,
            "trading_enabled": self.trading_enabled
        }


class ArbitrageBot:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange_manager = ExchangeManager(config["EXCHANGES"])
        self.monitoring_system = MonitoringSystem(config)
        self.error_handler = ErrorHandler(self.monitoring_system)
        self.safety_manager = SafetyManager(self.monitoring_system, config["RISK_CONFIG"])
        self.websocket_manager = None # Initialize as None, set later
        self.price_monitor = PriceMonitor(self.exchange_manager, self.monitoring_system, config["PERFORMANCE_CONFIG"])
        self.trading_engine = TradingEngine(self.exchange_manager, self.safety_manager, self.error_handler, self.monitoring_system)
        
        self.is_running = False
        self.is_initialized = False
        self.shutdown_event = asyncio.Event()

        # Configure logging for the bot
        self._setup_logging()

    def set_websocket_manager(self, manager):
        self.websocket_manager = manager
        logger.info("WebSocketManager set on ArbitrageBot.")
        self.price_monitor.set_websocket_manager(manager)
        logger.info("WebSocketManager set on PriceMonitor.")



    def _setup_logging(self):
        log_level = self.config["LOGGING_CONFIG"].get("level", "INFO").upper()
        log_file = self.config["LOGGING_CONFIG"].get("file")

        # Remove existing handlers to prevent duplicate logs
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config["LOGGING_CONFIG"].get("max_bytes", 5 * 1024 * 1024), # 5 MB
                backupCount=self.config["LOGGING_CONFIG"].get("backup_count", 5)
            )
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(file_handler)

        logger.info(f"Logging configured. Level: {log_level}, File: {log_file if log_file else 'Console'}")

    async def initialize(self):
        if self.is_initialized:
            logger.info("Bot already initialized.")
            return
        
        logger.info("Initializing bot components...")
        await self.exchange_manager.initialize_exchanges()
        await self.safety_manager.initialize_balances(self.exchange_manager)
        self.is_initialized = True
        logger.info("Bot initialization complete.")

    # ArbitrageBot.start()
    async def start(self):
        if not self.websocket_manager:
            raise RuntimeError("WebSocketManager must be set BEFORE bot.start() is called")
        
        if self.is_running:
            logger.info("Bot is already running.")
            return

        logger.info("Starting Crypto Arbitrage Bot...")
        self.is_running = True
        self.shutdown_event.clear()

        await self.initialize()
        await self.monitoring_system.start()

        # Only now: start the websocket manager (already set)
        if self.websocket_manager:
            await self.websocket_manager.start()

        # Now it's safe to start monitoring:
        asyncio.create_task(self.price_monitor.start_monitoring())
        # ...etc.


        # Main arbitrage loop
        while not self.shutdown_event.is_set():
            try:
                opportunities = self.price_monitor.get_arbitrage_opportunities()
                # Sort opportunities by score (highest first)
                opportunities.sort(key=lambda x: x.score, reverse=True)

                for opportunity in opportunities:
                    if not self.shutdown_event.is_set():
                        logger.info(f"Found opportunity: {opportunity.symbol} profit {opportunity.potential_profit_pct:.2f}% (Score: {opportunity.score:.2f})")
                        asyncio.create_task(self.trading_engine.execute_arbitrage_trade(opportunity))
                        await asyncio.sleep(PERFORMANCE_CONFIG.get("opportunity_scan_interval", 0.05)) # Small delay to prevent overwhelming

                await asyncio.sleep(PERFORMANCE_CONFIG.get("main_loop_interval", 1))

            except Exception as e:
                logger.error(f"Error in main arbitrage loop: {e}")
                self.error_handler.handle_error(e, ErrorCategory.SYSTEM, ErrorSeverity.HIGH, "ArbitrageBot", "Main Arbitrage Loop Error")
                await asyncio.sleep(5) # Wait before retrying

        logger.info("Bot shutdown initiated.")
        await self.stop()


    async def stop(self):
        if not self.is_running:
            logger.info("Bot is not running.")
            return

        logger.info("Stopping Crypto Arbitrage Bot...")
        self.is_running = False
        self.shutdown_event.set() # Signal shutdown

        # Stop monitoring system
        await self.monitoring_system.stop()

        # Close WebSocket Manager
        if self.websocket_manager:
            await self.websocket_manager.close()

        # Cancel all active tasks (if any)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Crypto Arbitrage Bot stopped.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "trading_enabled": self.trading_engine.trading_enabled,
            "circuit_breaker_active": self.safety_manager.is_circuit_breaker_active(),
            "active_trades_count": len(self.trading_engine.active_trades)
        }

    async def run_forever(self):
        """Runs the bot until a shutdown signal is received."""
        await self.initialize()
        await self.start()
        # Keep the event loop running until shutdown is signaled
        await self.shutdown_event.wait()
        logger.info("Bot run_forever completed.")


