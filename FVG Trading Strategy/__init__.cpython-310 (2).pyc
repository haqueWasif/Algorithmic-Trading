import MetaTrader5 as mt5
from secrets_1 import MT5_CONFIG, TELEGRAM_CONFIG  # NEW: Import from secrets.py

# MetaTrader5 configuration
MT5_LOGIN = MT5_CONFIG['login']  # NEW: Load from secrets
MT5_PASSWORD = MT5_CONFIG['password']  # NEW: Load from secrets
MT5_SERVER = MT5_CONFIG['server']  # NEW: Load from secrets
MT5_TERMINAL_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# Telegram configuration
TELEGRAM_BOT_TOKEN = TELEGRAM_CONFIG['bot_token']  # NEW: Load from secrets
TELEGRAM_CHAT_ID = TELEGRAM_CONFIG['chat_id']  # NEW: Load from secrets

# Trading and plotting settings
SYMBOL = 'BTCUSD'
LIMIT = 1000  # Historical data for navigation
DISPLAY_CANDLES = 100  # Candles to display at a time
PLOT_TIMEFRAMES = ['M1', 'M5', 'M15', 'H1']

# Timeframe mapping
TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}