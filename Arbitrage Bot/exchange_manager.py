import ccxt
import os
import math
import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    potential_profit_pct: float
    potential_profit_usd: float
    max_quantity: float
    timestamp: float
    score: float = 0.0 # Added score to dataclass

class ExchangeManager:
    def __init__(self, exchanges_config: Dict[str, Any]):
        self.exchanges_config = exchanges_config
        self.exchanges: Dict[str, Any] = {}
        self.initialized = False
        self.trading_fees: Dict[str, float] = {}

    async def initialize_exchanges(self):
        if self.initialized:
            logger.info("Exchanges already initialized.")
            return

        logger.info("Initializing exchanges...")
        for exchange_id, config in self.exchanges_config.items():
            try:
                if not config.get("api_key") or not config.get("secret"):
                    logger.warning(f"Skipping {exchange_id}: Missing API credentials")
                    continue

                exchange_class = getattr(ccxt, exchange_id)
                exchange_config = {
                    "apiKey": config["api_key"],
                    "secret": config["secret"],
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot", "adjustForTimeDifference": True}
                }

                if config.get("passphrase"):
                    exchange_config["password"] = config["passphrase"]

                exchange = exchange_class(exchange_config)

                # Sandbox mode
                if config.get("sandbox", False):
                    try:
                        exchange.set_sandbox_mode(True)
                        logger.info(f"Set {exchange_id} to SANDBOX mode.")
                    except Exception as sandbox_error:
                        logger.warning(f"Sandbox mode not supported: {sandbox_error}")

                # Load markets
                markets = await asyncio.to_thread(exchange.load_markets)
                logger.info(f"Loaded {len(markets)} markets for {exchange_id}")
                self.exchanges[exchange_id] = exchange

                # Fetch and store trading fees
                try:
                    # Attempt to fetch actual trading fees
                    # This might require specific exchange methods or a generic fetch_trading_fees
                    # For now, we'll use a placeholder or a default from config
                    fee_info = await asyncio.to_thread(exchange.fetch_trading_fees)
                    # Assuming fee_info structure, adjust as per actual CCXT response
                    # This part needs to be adapted based on how CCXT returns fees for the specific exchange
                    # For simplicity, we'll just use the default from config for now if not explicitly fetched
                    self.trading_fees[exchange_id] = config.get("trading_fee", 0.001)
                    logger.info(f"Fetched trading fees for {exchange_id}: {self.trading_fees[exchange_id]}")
                except Exception as fee_e:
                    logger.warning(f"Could not fetch trading fees for {exchange_id}: {fee_e}. Using default from config.")
                    self.trading_fees[exchange_id] = config.get("trading_fee", 0.001)

            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id}: {e}")

        self.initialized = True

    async def fetch_order(self, exchange_name: str, symbol: str, order_id: str):
        exchange = self.exchanges[exchange_name]

        try:
            # Bybit-specific fix: pass acknowledged=True
            params = {}
            if exchange_name == 'bybit':
                params['acknowledged'] = True

            # Call fetch_order safely (async or sync)
            result = exchange.fetch_order(order_id, symbol, params)
            if asyncio.iscoroutine(result):
                order = await result
            else:
                order = result

            logger.debug(f"Fetched order details for {order_id} on {exchange_name}: {order}")
            return order

        except Exception as e:
            logger.error(f"Failed to fetch order {order_id} on {exchange_name}: {e}")

            # Fallback: try fetch_open_orders / fetch_closed_orders safely
            try:
                open_result = exchange.fetch_open_orders(symbol)
                closed_result = exchange.fetch_closed_orders(symbol)

                if asyncio.iscoroutine(open_result):
                    open_orders = await open_result
                else:
                    open_orders = open_result

                if asyncio.iscoroutine(closed_result):
                    closed_orders = await closed_result
                else:
                    closed_orders = closed_result

                combined = open_orders + closed_orders
                for o in combined:
                    if o['id'] == order_id:
                        return o

            except Exception as e2:
                logger.error(f"Fallback fetch failed for {order_id} on {exchange_name}: {e2}")

            return None

    async def place_order(self, exchange_id: str, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Optional[Dict[str, Any]]:
        def round_up(value, decimals):
            multiplier = 10 ** decimals
            return math.ceil(value * multiplier) / multiplier
    
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            logger.warning(f"Exchange {exchange_id} not initialized.")
            return None
    
        # --- Get market info ---
        market = exchange.markets.get(symbol)
        if not market:
            logger.error(f"Market {symbol} not found on {exchange_id}")
            return None
    
        # --- Get min notional ---
        min_notional = None
        if market.get('limits'):
            if market['limits'].get('notional'):
                min_notional = market['limits']['notional'].get('min')
            elif market['limits'].get('cost'):
                min_notional = market['limits']['cost'].get('min')
    
        logger.info(f"min_notional (or cost min) for {symbol} on {exchange_id}: {min_notional}")
    
        # --- Ensure price is available ---
        if price is None:
            try:
                ticker = await exchange.fetch_ticker(symbol)
                price = ticker.get('last') or ticker.get('ask') or ticker.get('bid')
                logger.info(f"Fetched price for {symbol} on {exchange_id}: {price}")
            except Exception as e:
                logger.error(f"Failed to fetch price for {symbol} on {exchange_id}: {e}")
                return None
    
        if price is None:
            logger.error(f"No price data available for {symbol} on {exchange_id}")
            return None
    
        # --- Adjust amount to meet min notional ---
        amount_precision = market.get('precision', {}).get('amount', 6)
    
        if min_notional:
            notional = amount * price
            if notional < min_notional:
                required_amount = min_notional / price
                adjusted_amount = round_up(required_amount, amount_precision)
    
                # Cap based on MAX_TRADE_AMOUNT_USD
                MAX_TRADE_AMOUNT_USD = float(os.getenv("MAX_TRADE_AMOUNT_USD", 1.0))
                max_amount = MAX_TRADE_AMOUNT_USD / price
                if adjusted_amount > max_amount:
                    logger.warning(f"Adjusted amount {adjusted_amount} > max allowed ({max_amount}), using max_amount.")
                    adjusted_amount = max_amount
    
                logger.warning(
                    f"Adjusting amount for {symbol} on {exchange_id} to meet min notional. "
                    f"Original: {amount}, New: {adjusted_amount}"
                )
                amount = adjusted_amount
    
        # --- Final check ---
        order_cost = amount * price
        if min_notional and order_cost < min_notional:
            raise ValueError(f"Order cost {order_cost} USDT below min notional {min_notional} for {symbol} on {exchange_id}")
    
        # --- Place order ---
        try:
            if order_type == "limit":
                if side == "buy":
                    order_creation_method = exchange.create_limit_buy_order
                elif side == "sell":
                    order_creation_method = exchange.create_limit_sell_order
                else:
                    logger.error(f"Unsupported side for limit order: {side}")
                    return None
                args = (symbol, amount, price)
            elif order_type == "market":
                if side == "buy":
                    order_creation_method = exchange.create_market_buy_order
                elif side == "sell":
                    order_creation_method = exchange.create_market_sell_order
                else:
                    logger.error(f"Unsupported side for market order: {side}")
                    return None
                args = (symbol, amount)
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return None
    
            order = order_creation_method(*args)
            if asyncio.iscoroutine(order):
                order = await order
    
            logger.info(f"Placed {side} {order_type} order {order.get('id', 'N/A')} for {amount} {symbol} on {exchange_id}.")
            return order
    
        except Exception as e:
            logger.error(f"Failed to place {side} {order_type} order for {amount} {symbol} on {exchange_id}: {e}")
            return None

    async def fetch_ticker(self, exchange_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        exchange = self.exchanges.get(exchange_id)
        try:
            return await asyncio.to_thread(exchange.fetch_ticker, symbol)
        except Exception as e:
            logger.error(f"Failed to fetch ticker: {e}")
            return None

    async def get_balance(self, exchange_id: str, currency: str) -> float:
        exchange = self.exchanges.get(exchange_id)
        try:
            balance = await asyncio.to_thread(exchange.fetch_balance)
            return balance["free"].get(currency, 0.0)
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0

    async def get_all_balances(self) -> Dict[str, Dict[str, float]]:
        all_balances = {}
        for exchange_id, exchange in self.exchanges.items():
            try:
                balance = await asyncio.to_thread(exchange.fetch_balance)
                if asyncio.iscoroutine(balance):
                    balance = await balance
            except Exception as e:
                logger.error(f"Failed to fetch all balances for {exchange_id}: {e}")
                all_balances[exchange_id] = {"error": str(e)}
                continue # Continue to next exchange if this one fails

            all_balances[exchange_id] = {
                "free": balance["free"],
                "used": balance["used"],
                "total": balance["total"]
            }
        return all_balances

    async def get_order_book(self, exchange_id: str, symbol: str, limit: int = 10) -> Optional[Dict[str, Any]]:
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            logger.warning(f"Exchange {exchange_id} not initialized.")
            return None
        try:
            order_book = await exchange.fetch_order_book(symbol, limit=limit)
        except TypeError:
            order_book = exchange.fetch_order_book(symbol, limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol} on {exchange_id}: {e}")
            return None
        return order_book

    async def close(self):
        logger.info("Closing exchange connections...")
        for exchange_id, exchange in self.exchanges.items():
            try:
                # Some exchanges might have a close method, others might not
                if hasattr(exchange, "close"):
                    await exchange.close()
                logger.info(f"Closed connection for {exchange_id}.")
            except Exception as e:
                logger.error(f"Error closing connection for {exchange_id}: {e}")
                
    def get_exchange_trading_fee(self, exchange_id: str) -> float:
        """Returns the configured trading fee for a given exchange."""
        return self.trading_fees.get(exchange_id, 0.001) # Default to 0.001 if not found

    def get_exchange_volatility(self, exchange_id: str, symbol: str) -> float:
        """Placeholder for getting exchange-specific volatility. 
        In a real scenario, this would be calculated based on historical price data
        or fetched from a market data provider.
        """
        # For now, return a dummy value. Lower value means lower volatility.
        # This can be made dynamic based on actual market data later.
        return 0.005 # Example: 0.5% volatility



