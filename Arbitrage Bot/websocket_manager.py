import asyncio
import logging
import json
import websockets
import ccxt.async_support as ccxt
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self, config):
        self.config = config
        self.exchanges_config = config["EXCHANGES"]
        self.trade_symbols = config["TRADING_CONFIG"]["trade_symbols"]
        self.market_data = {}
        self.lock = asyncio.Lock()
        self.exchange_ws_clients = {}
        self.binance_ws_url = "wss://stream.binance.com:9443/ws/" # Default for production
        if self.exchanges_config.get("binance", {}).get("sandbox", False):
            self.binance_ws_url = "wss://stream.testnet.binance.vision/ws/" # Binance testnet WebSocket URL
        self.binance_update_count = 0
        self.bybit_update_count = 0

    async def start(self):
        """Start WebSocket connections for all configured exchanges."""
        tasks = []
        for exchange_id, exchange_config in self.exchanges_config.items():
            if exchange_config.get("api_key") and exchange_config.get("secret"):
                if exchange_id == "binance":
                    tasks.append(self._connect_binance_native_ws())
                else:
                    tasks.append(self._connect_and_subscribe(exchange_id, exchange_config))
        await asyncio.gather(*tasks)
        logger.info("WebSocket Manager started for all configured exchanges.")

    async def _connect_binance_native_ws(self):
        """Connects to Binance native WebSocket and subscribes to all-ticker stream."""
        stream_name = "!ticker@arr"
        uri = f"{self.binance_ws_url}{stream_name}"
        logger.info(f"Connecting to Binance native WebSocket: {uri}")

        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    logger.info(f"Connected to Binance native WebSocket: {uri}")
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if "data" in data and isinstance(data["data"], list):
                            for ticker_data in data["data"]:
                                symbol = ticker_data["s"] # Symbol, e.g., BTCUSDT
                                if symbol in self.trade_symbols:
                                    async with self.lock:
                                        self.market_data.setdefault("binance", {})[symbol] = {
                                            "bid": float(ticker_data["b"]),
                                            "ask": float(ticker_data["a"]),
                                            "timestamp": ticker_data["E"],
                                            "datetime": datetime.fromtimestamp(ticker_data["E"] / 1000).isoformat(),
                                            "high": float(ticker_data["h"]),
                                            "low": float(ticker_data["l"]),
                                            "volume": float(ticker_data["v"]),
                                            "quoteVolume": float(ticker_data["q"]),
                                            "info": ticker_data,
                                            "bids": [],
                                            "asks": []
                                        }
                                    self.binance_update_count += 1
                                    if self.binance_update_count % 100 == 0: # Log every 100 updates
                                        logger.info(f"Binance native WS update for {symbol}: Bid={ticker_data['b']}, Ask={ticker_data['a']}. Total updates: {self.binance_update_count}")
            except Exception as e:
                logger.error(f"Binance native WebSocket error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def _connect_and_subscribe(self, exchange_id, exchange_config):
        """Connects to exchange WebSocket using CCXT and subscribes to tickers with retry logic."""
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'apiKey': exchange_config['api_key'],
                    'secret': exchange_config['secret'],
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                })

                if exchange_config.get("sandbox", False):
                    try:
                        exchange.set_sandbox_mode(True)
                        logger.info(f"Set {exchange_id} to SANDBOX mode.")
                    except Exception as sandbox_error:
                        logger.warning(f"Sandbox mode not supported for {exchange_id}: {sandbox_error}")

                self.exchange_ws_clients[exchange_id] = exchange
                logger.info(f"Initialized CCXT client for {exchange_id}.")

                await exchange.load_markets()

                for symbol in self.trade_symbols:
                    asyncio.create_task(self._watch_ticker(exchange, exchange_id, symbol))
                return # Exit loop on successful connection

            except Exception as e:
                retries += 1
                delay = 2 ** retries # Exponential backoff
                logger.error(f"Failed to connect or subscribe to {exchange_id} WebSocket (Attempt {retries}/{max_retries}): {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        logger.critical(f"Failed to connect to {exchange_id} WebSocket after {max_retries} attempts. Please check API keys and network access.")

    async def _watch_ticker(self, exchange, exchange_id, symbol):
        """Watches ticker data for a given exchange and symbol using CCXT."""
        while True:
            try:
                ticker = await exchange.watch_ticker(symbol)
                async with self.lock:
                    self.market_data.setdefault(exchange_id, {})[symbol] = {
                        "bid": ticker["bid"],
                        "ask": ticker["ask"],
                        "timestamp": ticker["timestamp"],
                        "datetime": ticker["datetime"],
                        "high": ticker["high"],
                        "low": ticker["low"],
                        "volume": ticker["baseVolume"],
                        "quoteVolume": ticker["quoteVolume"],
                        "info": ticker["info"],
                        "bids": [],
                        "asks": []
                    }
                self.bybit_update_count += 1
                if self.bybit_update_count % 100 == 0: # Log every 100 updates
                    logger.info(f"Updated {symbol} on {exchange_id}: Bid={ticker['bid']}, Ask={ticker['ask']}. Total updates: {self.bybit_update_count}")
            except Exception as e:
                logger.error(f"Error watching ticker for {symbol} on {exchange_id}: {e}")
                await asyncio.sleep(5)

    async def get_latest_market_data(self, exchange_id, symbol):
        logger.debug(f"Attempting to retrieve market data for {symbol} on {exchange_id}")
        async with self.lock:
            return self.market_data.get(exchange_id, {}).get(symbol, None)

    async def close(self):
        logger.info("Stopping WebSocket Manager...")
        for exchange_id, client in self.exchange_ws_clients.items():
            if client and hasattr(client, 'close'):
                await client.close()
                logger.info(f"Closed CCXT WebSocket client for {exchange_id}.")
        logger.info("WebSocket Manager stopped.")


