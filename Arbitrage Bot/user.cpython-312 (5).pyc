"""
Main entry point for the Flask dashboard application.
"""
import asyncio
import sys
import platform

# PATCH: Fix for Windows + aiohttp/aiodns + python-binance event loop bug
# See: https://github.com/saghul/aiodns/issues/86
if platform.system() == "Windows":
    if sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
from flask import Flask, render_template
from flask_cors import CORS
import os
import sys
import logging

# Add the parent directory to the Python path to import bot modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from arbitrage_bot import ArbitrageBot
from monitoring import MonitoringSystem
from websocket_manager import WebSocketManager      # <-- Import WebSocketManager
from config import load_config

from routes.bot_api import bot_api, set_bot_instances

# Configure logging for the dashboard
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="static")
CORS(app)  # Enable CORS for all routes

# Load bot configuration
config = load_config()

# Initialize global bot, monitoring system, and WebSocketManager for the Flask app at startup
try:
    arbitrage_bot_instance = ArbitrageBot(config)
    monitoring_system_instance = MonitoringSystem(config)
    ws_manager_instance = WebSocketManager(config)   # <-- Initialize WS manager

    # Pass all three instances to the bot_api blueprint
    set_bot_instances(arbitrage_bot_instance, monitoring_system_instance, ws_manager_instance)
    logger.info("ArbitrageBot, MonitoringSystem, and WebSocketManager instances initialized and passed to bot_api.")

except Exception as e:
    logger.error(f"Failed to initialize bot, monitoring system, or WebSocketManager: {e}")
    # Prevent further errors if initialization fails
    arbitrage_bot_instance = None
    monitoring_system_instance = None
    ws_manager_instance = None

# Register blueprints
app.register_blueprint(bot_api, url_prefix="/api/bot")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    # Flask app runs as a dashboard only.
    app.run(host="0.0.0.0", port=5000, debug=True)
