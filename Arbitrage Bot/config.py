import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()



def load_config():
    """Loads and returns the configuration for the arbitrage bot."""

    # Exchange API Credentials
    EXCHANGES = {
        "binance": {
            "api_key": os.getenv("BINANCE_API_KEY"),
            "secret": os.getenv("BINANCE_SECRET"),
            "sandbox": os.getenv("BINANCE_SANDBOX", "false").lower() == "true",
            "rate_limit": 1200,  # requests per minute
            "trading_fee": 0.001  # 0.1%
        },
        # "bybit": {
        #     "api_key": os.getenv("BYBIT_API_KEY"),
        #     "secret": os.getenv("BYBIT_SECRET"),
        #     "sandbox": os.getenv("BYBIT_SANDBOX", "false").lower() == "true",
        #     "rate_limit": 60,  # requests per minute
        #     "trading_fee": 0.001 # 0.1%
        # },
        # Add other exchanges here as needed
    }

    # Trading Configuration
    TRADING_CONFIG = {
        "min_profit_threshold": float(os.getenv("MIN_PROFIT_THRESHOLD", 0.001)), # 0.1% minimum profit
        "max_trade_amount_usd": float(os.getenv("MAX_TRADE_AMOUNT_USD", 100.0)), # Max trade amount in USD
        "max_daily_trades": int(os.getenv("MAX_DAILY_TRADES", 1000)),
        "order_timeout_seconds": int(os.getenv("ORDER_TIMEOUT_SECONDS", 10)),
        "max_slippage_tolerance": float(os.getenv("MAX_SLIPPAGE_TOLERANCE", 0.002)), # 0.2% max slippage
        "pre_trade_slippage_estimation_threshold": float(os.getenv("PRE_TRADE_SLIPPAGE_ESTIMATION_THRESHOLD", 0.001)), # 0.1% of expected profit
        "adaptive_limit_order_aggressiveness": float(os.getenv("ADAPTIVE_LIMIT_ORDER_AGGRESSIVENESS", 0.0005)), # 0.05% closer to market
        # Symbols for native websocket APIs (no slash, uppercase)
        "trade_symbols": [s.strip().replace("/", "").upper() for s in os.getenv("TRADE_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")]
    }

    # Risk Management
    RISK_CONFIG = {
        "max_daily_loss_usd": float(os.getenv("MAX_DAILY_LOSS_USD", 500.0)),
        "max_single_trade_loss_usd": float(os.getenv("MAX_SINGLE_TRADE_LOSS_USD", 50.0)),
        "max_open_positions": int(os.getenv("MAX_OPEN_POSITIONS", 5)),
        "emergency_stop_loss_pct": float(os.getenv("EMERGENCY_STOP_LOSS_PCT", 0.05)), # 5%
        "balance_check_interval_minutes": int(os.getenv("BALANCE_CHECK_INTERVAL_MINUTES", 5)),
        "max_consecutive_losses": int(os.getenv("MAX_CONSECUTIVE_LOSSES", 3)), # Max consecutive losing trades before pause
        "circuit_breaker_enabled": os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true",
        "granular_circuit_breaker_cooldown_minutes": int(os.getenv("GRANULAR_CIRCUIT_BREAKER_COOLDOWN_MINUTES", 10)),
        "dynamic_loss_threshold_factor": float(os.getenv("DYNAMIC_LOSS_THRESHOLD_FACTOR", 0.1)), # Factor to adjust daily loss limit based on profit
        "rebalancing_threshold_pct": float(os.getenv("REBALANCING_THRESHOLD_PCT", 0.1)), # 10% imbalance triggers rebalancing
        "opportunity_scoring_weights": {
            "profit": float(os.getenv("OPPORTUNITY_SCORING_WEIGHT_PROFIT", 0.4)),
            "liquidity": float(os.getenv("OPPORTUNITY_SCORING_WEIGHT_LIQUIDITY", 0.3)),
            "volatility": float(os.getenv("OPPORTUNITY_SCORING_WEIGHT_VOLATILITY", 0.2)),
            "historical_success": float(os.getenv("OPPORTUNITY_SCORING_WEIGHT_HISTORICAL_SUCCESS", 0.1))
        }
    }

    # Database Configuration
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "name": os.getenv("DB_NAME", "arbitrage_bot"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "password")
    }

    # Redis Configuration
    REDIS_CONFIG = {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "db": int(os.getenv("REDIS_DB", 0)),
        "password": os.getenv("REDIS_PASSWORD", "")
    }

    # Logging
    LOGGING_CONFIG = {
        "level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "file": os.getenv("LOG_FILE", "arbitrage_bot.log"),
        "max_bytes": int(os.getenv("LOG_MAX_FILE_SIZE_MB", 100)) * 1024 * 1024,
        "backup_count": int(os.getenv("LOG_BACKUP_COUNT", 5))
    }

    # Monitoring & Alerts
    MONITORING_CONFIG = {
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
        "email_smtp_server": os.getenv("EMAIL_SMTP_SERVER"),
        "email_smtp_port": int(os.getenv("EMAIL_SMTP_PORT", 587)),
        "email_username": os.getenv("EMAIL_USERNAME"),
        "email_password": os.getenv("EMAIL_PASSWORD"),
        "email_recipients": [r.strip() for r in os.getenv("EMAIL_RECIPIENTS", "").split(",")] if os.getenv("EMAIL_RECIPIENTS") else [],
        "alert_thresholds": {
            "cpu_usage_percent": float(os.getenv("ALERT_THRESHOLD_CPU", 80)),
            "memory_usage_percent": float(os.getenv("ALERT_THRESHOLD_MEMORY", 80))
        }
    }

    # Flask Web Interface
    FLASK_CONFIG = {
        "port": int(os.getenv("FLASK_PORT", 5000)),
        "debug": os.getenv("FLASK_DEBUG", "false").lower() == "true",
        "secret_key": os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")
    }

    # Performance Tuning + new websocket options
    PERFORMANCE_CONFIG = {
        "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", 50)),
        "request_timeout_seconds": float(os.getenv("REQUEST_TIMEOUT_SECONDS", 10)),
        "websocket_ping_interval": int(os.getenv("WEBSOCKET_PING_INTERVAL", 30)),
        "order_book_depth": int(os.getenv("ORDER_BOOK_DEPTH", 20)),
        "price_update_interval": float(os.getenv("PRICE_UPDATE_INTERVAL", 0.1)),
        "main_loop_interval": float(os.getenv("MAIN_LOOP_INTERVAL", 1)),
        "opportunity_scan_interval": float(os.getenv("OPPORTUNITY_SCAN_INTERVAL", 0.05)),
        "websocket_data_source": os.getenv("WEBSOCKET_DATA_SOURCE", "native_websocket"),  # use native exchange websockets
        "websocket_urls": {
            "binance": os.getenv("BINANCE_WS_URL", ""),
            "bybit": os.getenv("BYBIT_WS_URL", "")
        },
        "WEBSOCKET_CONFIG": {
            "binance": {
                "ping_interval": int(os.getenv("BINANCE_WS_PING_INTERVAL", 20)),
            },
            "bybit": {
                "ping_interval": int(os.getenv("BYBIT_WS_PING_INTERVAL", 25)),
            }
        }
    }

    return {
        "EXCHANGES": EXCHANGES,
        "TRADING_CONFIG": TRADING_CONFIG,
        "RISK_CONFIG": RISK_CONFIG,
        "DB_CONFIG": DB_CONFIG,
        "REDIS_CONFIG": REDIS_CONFIG,
        "LOGGING_CONFIG": LOGGING_CONFIG,
        "MONITORING_CONFIG": MONITORING_CONFIG,
        "FLASK_CONFIG": FLASK_CONFIG,
        "PERFORMANCE_CONFIG": PERFORMANCE_CONFIG
    }


CONFIG = load_config()
EXCHANGES = CONFIG["EXCHANGES"]
TRADING_CONFIG = CONFIG["TRADING_CONFIG"]
RISK_CONFIG = CONFIG["RISK_CONFIG"]
PERFORMANCE_CONFIG = CONFIG["PERFORMANCE_CONFIG"]
MONITORING_CONFIG = CONFIG["MONITORING_CONFIG"]
LOGGING_CONFIG = CONFIG["LOGGING_CONFIG"]


