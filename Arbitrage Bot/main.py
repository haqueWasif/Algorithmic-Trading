# main.py

import asyncio
import signal
import logging
from arbitrage_bot import ArbitrageBot
from config import load_config
from websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}. Initiating shutdown...")
    if hasattr(signal_handler, 'bot') and signal_handler.bot:
        signal_handler.bot.shutdown_event.set()

async def main():
    config = load_config()
    websocket_manager = WebSocketManager(config)
    bot = ArbitrageBot(config)
    bot.set_websocket_manager(websocket_manager)  # <-- ENSURE THIS HAPPENS BEFORE STARTING BOT

    signal_handler.bot = bot
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting Crypto Arbitrage Bot...")
    await bot.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
