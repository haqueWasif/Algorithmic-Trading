import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

from exchange_manager import ExchangeManager, ArbitrageOpportunity
from config import TRADING_CONFIG, PERFORMANCE_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class PriceAlert:
    """Represents a price alert or anomaly."""
    exchange: str
    symbol: str
    alert_type: str  # 'spike', 'drop', 'stale_data', 'spread_anomaly'
    current_price: float
    reference_price: Optional[float]
    deviation_pct: float
    timestamp: float
    message: str

@dataclass
class MarketStats:
    """Market statistics for a trading pair."""
    symbol: str
    exchanges: List[str] = field(default_factory=list)
    prices: Dict[str, float] = field(default_factory=dict)
    spreads: Dict[str, float] = field(default_factory=dict)  # bid-ask spread per exchange
    price_history: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    last_update: Dict[str, float] = field(default_factory=dict)
    
    def add_price_data(self, exchange: str, bid: float, ask: float, timestamp: float):
        """Add new price data for an exchange."""
        mid_price = (bid + ask) / 2
        spread = ask - bid
        
        self.prices[exchange] = mid_price
        self.spreads[exchange] = spread
        self.price_history[exchange].append((timestamp, mid_price))
        self.last_update[exchange] = timestamp
        
        if exchange not in self.exchanges:
            self.exchanges.append(exchange)
    
    def get_price_volatility(self, exchange: str, window_minutes: int = 5) -> Optional[float]:
        """Calculate price volatility over a time window."""
        if exchange not in self.price_history:
            return None
        
        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)
        
        # Get recent prices
        recent_prices = [
            price for timestamp, price in self.price_history[exchange]
            if timestamp >= cutoff_time
        ]
        
        if len(recent_prices) < 2:
            return None
        
        # Calculate standard deviation as volatility measure
        return statistics.stdev(recent_prices)
    
    def get_cross_exchange_spread(self) -> Optional[float]:
        """Calculate the spread between highest and lowest prices across exchanges."""
        if len(self.prices) < 2:
            return None
        
        prices = list(self.prices.values())
        return max(prices) - min(prices)

    
class PriceMonitor:
    def __init__(self, exchange_manager: ExchangeManager, monitoring_system: Any, performance_config: Dict[str, Any], websocket_manager=None):
        self.exchange_manager = exchange_manager
        self.monitoring_system = monitoring_system
        self.performance_config = performance_config
        self.websocket_manager = websocket_manager # Will be set by ArbitrageBot
        self.tickers: Dict[str, Dict[str, Any]] = {}
        self.order_books: Dict[str, Dict[str, Any]] = {}
        self.opportunities: deque[ArbitrageOpportunity] = deque(maxlen=100)
        self.last_scan_time = time.time()

    def set_websocket_manager(self, manager):
        self.websocket_manager = manager
        logger.info("WebSocketManager has been set on PriceMonitor!")  # <-- confirms manager is set

    async def start_monitoring(self):
        logger.info(f"Starting price monitoring, websocket_manager: {self.websocket_manager}")
        while True:
            try:
                await self._scan_for_opportunities()
            except Exception as e:
                logger.error(f"Error in price monitoring loop: {e}")
                self.monitoring_system.alert_manager.create_alert(
                    "Price Monitor Error", f"Error in price monitoring loop: {e}", "error", "PriceMonitor"
                )
            await asyncio.sleep(self.performance_config.get("price_update_interval", 0.1))


    async def _scan_for_opportunities(self):
        if not self.websocket_manager:
            logger.warning("WebSocketManager not set. Cannot scan for opportunities.")
            return

        # Fetch tickers from WebSocketManager
        self.tickers = {}
        for exchange_id in self.exchange_manager.exchanges_config.keys():
            for symbol in TRADING_CONFIG["trade_symbols"]:
                market_data = await self.websocket_manager.get_latest_market_data(exchange_id, symbol)
                if market_data and market_data.get("bid") is not None and market_data.get("ask") is not None:
                    if exchange_id not in self.tickers:
                        self.tickers[exchange_id] = {}
                    self.tickers[exchange_id][symbol] = {
                        "bid": market_data["bid"],
                        "ask": market_data["ask"],
                        "timestamp": market_data["timestamp"],
                        "bids": market_data.get("bids", []),
                        "asks": market_data.get("asks", []),
                    }
                else:
                    # This warning is expected if data isn't immediately available, but should resolve as data streams in.
                    logger.debug(f"No valid WebSocket data for {symbol} on {exchange_id} yet.")

        # Find arbitrage opportunities
        self.opportunities.clear() # Clear old opportunities
        opportunities_found_total = 0

        for symbol in TRADING_CONFIG["trade_symbols"]:
            for buy_exchange_id, buy_exchange_tickers in self.tickers.items():
                if symbol not in buy_exchange_tickers or not buy_exchange_tickers[symbol]:
                    continue
                
                buy_price = buy_exchange_tickers[symbol].get("ask")
                
                # Ensure buy_price is not None and is a valid number
                if buy_price is None or not isinstance(buy_price, (int, float)) or buy_price <= 0:
                    continue

                for sell_exchange_id, sell_exchange_tickers in self.tickers.items():
                    if buy_exchange_id == sell_exchange_id:
                        continue
                    if symbol not in sell_exchange_tickers or not sell_exchange_tickers[symbol]:
                        continue
                    
                    sell_price = sell_exchange_tickers[symbol].get("bid")

                    # Ensure sell_price is not None and is a valid number
                    if sell_price is None or not isinstance(sell_price, (int, float)) or sell_price <= 0:
                        continue

                    # Calculate potential profit (simplified, without fees)
                    if sell_price > buy_price: # Only consider if sell price is higher than buy price
                        # Get trading fees for both exchanges
                        buy_fee = self.exchange_manager.get_exchange_trading_fee(buy_exchange_id)
                        sell_fee = self.exchange_manager.get_exchange_trading_fee(sell_exchange_id)

                        # Calculate profit after fees
                        # Assuming fees are a percentage of the trade amount
                        effective_buy_price = buy_price * (1 + buy_fee)
                        effective_sell_price = sell_price * (1 - sell_fee)

                        if effective_sell_price > effective_buy_price:
                            potential_profit_pct = ((effective_sell_price - effective_buy_price) / effective_buy_price) * 100
                            
                            # Apply minimum profit threshold from config
                            min_profit_threshold = TRADING_CONFIG.get("min_profit_threshold", 0.001)

                            if potential_profit_pct > min_profit_threshold * 100:
                                # Dynamic max_quantity based on order book depth and volume
                                buy_order_book = buy_exchange_tickers[symbol].get("asks", [])
                                sell_order_book = sell_exchange_tickers[symbol].get("bids", [])

                                # Calculate max tradable quantity considering order book depth
                                max_quantity = self._calculate_max_tradable_quantity(
                                    buy_price, sell_price, buy_order_book, sell_order_book, min_profit_threshold
                                )

                                if max_quantity <= 0:
                                    continue # No profitable quantity found

                                potential_profit_usd = (effective_sell_price - effective_buy_price) * max_quantity
                                
                                # Opportunity Scoring
                                opportunity_score = self._score_opportunity(
                                    potential_profit_pct, max_quantity, 
                                    buy_fee,
                                    sell_fee,
                                    # Placeholder for actual volatility, need to implement in MarketStats
                                    0.0, # self.exchange_manager.get_exchange_volatility(buy_exchange_id, symbol),
                                    0.0  # self.exchange_manager.get_exchange_volatility(sell_exchange_id, symbol)
                                )

                                opportunity = ArbitrageOpportunity(
                                    symbol=symbol,
                                    buy_exchange=buy_exchange_id,
                                    sell_exchange=sell_exchange_id,
                                    buy_price=buy_price,
                                    sell_price=sell_price,
                                    potential_profit_pct=potential_profit_pct,
                                    potential_profit_usd=potential_profit_usd,
                                    max_quantity=max_quantity,
                                    timestamp=time.time(),
                                    score=opportunity_score
                                )
                                self.opportunities.append(opportunity)
                                opportunities_found_total += 1
            
            if opportunities_found_total > 0:
                logger.info(f"Found {opportunities_found_total} arbitrage opportunities for {symbol}.")

        self.last_scan_time = time.time()
        self.monitoring_system.update_performance_metrics(
            active_trades_count=0, # This should come from trading_engine
            opportunities_found=opportunities_found_total,
            trade_execution_times=[] # This should come from trading_engine
        )


    def get_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        return list(self.opportunities)

    def get_market_summary(self) -> Dict[str, Any]:
        summary = {"tickers": self.tickers, "last_scan": self.last_scan_time}
        return summary

    def _calculate_max_tradable_quantity(self, buy_price, sell_price, buy_order_book, sell_order_book, min_profit_threshold):
        # This is a more robust calculation considering order book depth.
        # It finds the maximum quantity that can be traded while maintaining the min_profit_threshold.

        max_buy_quantity = 0.0
        current_buy_cost = 0.0
        for price, quantity in buy_order_book:
            if price <= buy_price: # Only consider asks at or below our desired buy price
                max_buy_quantity += quantity
                current_buy_cost += price * quantity
            else:
                break # Order book is sorted by price, so we can stop
        
        max_sell_quantity = 0.0
        current_sell_revenue = 0.0
        for price, quantity in sell_order_book:
            if price >= sell_price: # Only consider bids at or above our desired sell price
                max_sell_quantity += quantity
                current_sell_revenue += price * quantity
            else:
                break # Order book is sorted by price, so we can stop

        # The actual tradable quantity is limited by the liquidity on both sides
        tradable_quantity = min(max_buy_quantity, max_sell_quantity)

        # Further refine max_quantity based on MAX_TRADE_AMOUNT_USD from config
        max_trade_amount_usd = TRADING_CONFIG.get("max_trade_amount_usd", 100.0)
        max_quantity_from_usd = max_trade_amount_usd / buy_price if buy_price > 0 else 0
        max_quantity = min(tradable_quantity, max_quantity_from_usd)

        return max(0.0, max_quantity)

    def _score_opportunity(self, potential_profit_pct, max_quantity, buy_fee, sell_fee, buy_volatility, sell_volatility):
        # Implement a comprehensive scoring system for arbitrage opportunities
        # This is a simplified example, weights can be adjusted in config.py
        
        profit_weight = RISK_CONFIG["opportunity_scoring_weights"]["profit"]
        liquidity_weight = RISK_CONFIG["opportunity_scoring_weights"]["liquidity"]
        volatility_weight = RISK_CONFIG["opportunity_scoring_weights"]["volatility"]
        historical_success_weight = RISK_CONFIG["opportunity_scoring_weights"]["historical_success"]

        # Normalize profit (example: scale to 0-100)
        normalized_profit = min(potential_profit_pct * 10, 100) # Assuming 10% profit is high

        # Normalize liquidity (example: scale based on a max quantity, e.g., 10 BTC)
        normalized_liquidity = min(max_quantity / 10.0 * 100, 100) # Assuming 10 BTC is high liquidity

        # Volatility (lower is better, so invert score)
        # Assuming volatility is a small number, e.g., 0.01 for 1% volatility
        normalized_volatility = max(0, 100 - (buy_volatility + sell_volatility) * 1000) # Scale and invert

        # Historical success (placeholder, would come from trade history analysis)
        # For now, assume a default high value, or integrate with a real historical success rate
        historical_success_score = 80 # Placeholder

        score = (
            normalized_profit * profit_weight +
            normalized_liquidity * liquidity_weight +
            normalized_volatility * volatility_weight +
            historical_success_score * historical_success_weight
        )
        return score 

