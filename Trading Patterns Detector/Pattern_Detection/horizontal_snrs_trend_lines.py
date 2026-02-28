import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import asyncio
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque, OrderedDict
import mplfinance as mpf
from matplotlib.widgets import Slider, CheckButtons
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.cluster import DBSCAN
from itertools import product
from joblib import Parallel, delayed
import pickle
import os
import sqlite3
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple, Dict
import lru

# Configure logging with reduced verbosity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Configuration
CONFIG = {
    'MT5_PATH' : r"C:\Program Files\MetaTrader\terminal64.exe",
    'MT5_LOGIN' : 97109110,
    'MT5_PASSWORD' : "Gb*l5fZg",
    'MT5_SERVER' : "MetaQuotes-Demo",
    'SYMBOLS': ['XAUUSD'],
    'LIMIT': 2000,
    'DISPLAY_CANDLES': 50,
    'PLOT_TIMEFRAMES': ['H1'],
    'AGGREGATE_TIMEFRAMES': ['M1', 'M5', 'M15'],
    'SR_LEVELS_FILE': 'historical_sr_levels.pkl',
    'DB_FILE': 'ohlcv_cache.db',
    'DEFAULT_EPS': 0.0005,
    'MIN_CANDLES': 14,
    'UPDATE_INTERVAL': 1000,  # ms
    'RETRY_DELAYS': [2, 4, 8],  # Seconds for exponential backoff
    'CACHE_SIZE': 100,
}

# Timeframe mappings
TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1,
}
TIMEFRAME_MINUTES = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
CACHE_EXPIRY = {'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800, 'H1': 3600, 'H4': 14400, 'D1': 86400}

# Global state
global_mt5_connected = False
ohlcv_cache = lru.LRU(CONFIG['CACHE_SIZE'])  # LRU cache for OHLCV data
fallback_data = {}  # Last valid data per symbol/timeframe
historical_sr_levels = deque(maxlen=50)
indicator_cache = {}  # Cache for ATR, MACD, RSI, Bollinger Bands

# SQLite cache setup
def init_db():
    """Initialize SQLite database for OHLCV caching."""
    with sqlite3.connect(CONFIG['DB_FILE']) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT, timeframe TEXT, timestamp INTEGER, open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )''')
        conn.commit()

# S/R Level class with reduced memory footprint
class SRLevel:
    __slots__ = ['price', 'timestamp', 'is_support', 'touches', 'significance']

    def __init__(self, price: float, timestamp: datetime, is_support: bool, touches: int = 1, significance: float = 1.0):
        self.price = price
        self.timestamp = timestamp
        self.is_support = is_support
        self.touches = touches
        self.significance = significance

    def update_touch(self):
        self.touches += 1
        self.significance += 0.2

# Initialize MetaTrader5 with persistent connection
def initialize_mt5() -> bool:
    """Initialize MT5 connection with retry logic."""
    global global_mt5_connected
    for attempt, delay in enumerate(CONFIG['RETRY_DELAYS'], 1):
        try:
            if mt5.initialize(login=CONFIG['MT5_LOGIN'], password=CONFIG['MT5_PASSWORD'], 
                            server=CONFIG['MT5_SERVER'], timeout=30000):
                global_mt5_connected = True
                for symbol in CONFIG['SYMBOLS']:
                    if not mt5.symbol_select(symbol, True):
                        logger.error(f"Symbol {symbol} not available: {mt5.last_error()}")
                        mt5.shutdown()
                        global_mt5_connected = False
                        return False
                if not any(s.name == 'EURUSD' for s in mt5.symbols_get()):
                    logger.error("EURUSD not found in market watch")
                    mt5.shutdown()
                    global_mt5_connected = False
                    return False
                logger.info("MT5 initialized successfully")
                return True
            logger.warning(f"MT5 initialization failed: {mt5.last_error()}")
        except Exception as e:
            logger.error(f"MT5 initialization exception (attempt {attempt}): {e}")
        time.sleep(delay)
    global_mt5_connected = False
    logger.error("Failed to initialize MT5 after retries")
    return False

def fetch_ohlcv_from_mt5(symbol: str, timeframe: str, limit: int, 
                        last_timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from MT5 with incremental fetching and robust error handling."""
    global global_mt5_connected
    try:
        start_time = time.time()
        if not global_mt5_connected or not mt5.terminal_info():
            logger.warning("MT5 connection lost, attempting to reconnect")
            if not initialize_mt5():
                logger.error("Reconnection failed, cannot fetch data")
                return None
        timeframe_val = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_H1)
        now = datetime.now(timezone.utc).replace(microsecond=0)  # MT5 expects clean datetime
        minutes_per_candle = TIMEFRAME_MINUTES.get(timeframe, 60)
        
        if last_timestamp:  # Incremental fetch
            # Ensure last_timestamp is timezone-aware and MT5-compatible
            if not isinstance(last_timestamp, datetime):
                logger.error(f"Invalid last_timestamp type: {type(last_timestamp)}")
                return None
            if last_timestamp.tzinfo is None:
                last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
            from_date = last_timestamp
            candles = 50 if timeframe in ['M1', 'M5', 'M15'] else 10  # Increased for M5 stability
        else:
            candles = limit
            from_date = now - timedelta(minutes=candles * minutes_per_candle)
        
        # Convert to naive UTC datetime for MT5 compatibility
        from_date = from_date.replace(tzinfo=None) if from_date.tzinfo else from_date
        now = now.replace(tzinfo=None)
        
        logger.debug(f"Fetching {candles} candles for {symbol} on {timeframe} from {from_date} to {now}")
        rates = mt5.copy_rates_range(symbol, timeframe_val, from_date, now)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"copy_rates_range returned no data for {symbol} on {timeframe}, trying copy_rates_from_pos")
            rates = mt5.copy_rates_from_pos(symbol, timeframe_val, 0, candles)
        
        if rates is None or len(rates) == 0:
            logger.error(f"No data fetched for {symbol} on {timeframe}: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if len(df) < CONFIG['MIN_CANDLES']:
            logger.error(f"Insufficient data ({len(df)} candles) for {symbol} on {timeframe}")
            return None
        
        df.set_index('timestamp', inplace=True)
        df = df.astype({'open': 'float16', 'high': 'float16', 'low': 'float16', 'close': 'float16', 'volume': 'int16'})
        df = df.sort_index().ffill().dropna()
        
        if df[['open', 'high', 'low', 'close']].le(0).any().any():
            logger.error(f"Invalid price data for {symbol} on {timeframe}")
            return None
        df['volume'] = df['volume'].clip(lower=1)
        
        logger.info(f"Fetched {len(df)} candles for {symbol} on {timeframe}, took {time.time() - start_time:.3f}s")
        fallback_data[(symbol, timeframe)] = df
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol} on {timeframe}: {e}, MT5 error: {mt5.last_error()}")
        global_mt5_connected = False
        return None

def fetch_ohlcv(symbol: str, timeframe: str, limit: int, retries: int = 3, 
                last_timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data with caching, retries, and fallback."""
    cache_key = f"{symbol}_{timeframe}_{limit}"
    if cache_key in ohlcv_cache:
        cached_df, cache_time = ohlcv_cache[cache_key]
        if (datetime.now(timezone.utc) - cache_time).total_seconds() < CACHE_EXPIRY.get(timeframe, 3600):
            logger.debug(f"Using cached data for {symbol} on {timeframe}")
            return cached_df
    
    # Check SQLite cache first
    df_db = fetch_from_db(symbol, timeframe, limit)
    if df_db is not None and len(df_db) >= CONFIG['MIN_CANDLES']:
        ohlcv_cache[cache_key] = (df_db, datetime.now(timezone.utc))
        logger.info(f"Using SQLite data for {symbol} on {timeframe}")
        return df_db
    
    # Retry fetching from MT5
    for attempt in range(retries):
        df = fetch_ohlcv_from_mt5(symbol, timeframe, limit, last_timestamp)
        if df is not None:
            cache_to_db(symbol, timeframe, df)
            ohlcv_cache[cache_key] = (df, datetime.now(timezone.utc))
            return df
        logger.warning(f"Attempt {attempt + 1} failed for {symbol} on {timeframe}")
        time.sleep(CONFIG['RETRY_DELAYS'][attempt])
    
    # Fallback to SQLite or stored data
    if (symbol, timeframe) in fallback_data:
        logger.warning(f"Using fallback data for {symbol} on {timeframe}")
        return fallback_data[(symbol, timeframe)]
    
    df_db = fetch_from_db(symbol, timeframe, limit)
    if df_db is not None:
        logger.warning(f"Using SQLite cached data for {symbol} on {timeframe}")
        ohlcv_cache[cache_key] = (df_db, datetime.now(timezone.utc))
        return df_db
    
    logger.error(f"Failed to fetch data for {symbol} on {timeframe} after {retries} attempts")
    return None

def cache_to_db(symbol: str, timeframe: str, df: pd.DataFrame):
    """Batch insert OHLCV data to SQLite."""
    with sqlite3.connect(CONFIG['DB_FILE']) as conn:
        df.reset_index()[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].to_sql(
            'ohlcv', conn, if_exists='append', index=False, method='multi')
        conn.commit()

def fetch_from_db(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from SQLite cache."""
    with sqlite3.connect(CONFIG['DB_FILE']) as conn:
        query = "SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
    if df.empty:
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype({
        'open': 'float16', 'high': 'float16', 'low': 'float16', 'close': 'float16', 'volume': 'int16'
    })
    return df.sort_index()


def calculate_indicators(df: pd.DataFrame, timeframe: str, cache_key: str) -> Dict:
    """Calculate and cache technical indicators."""
    if cache_key in indicator_cache:
        cached_data, cache_time = indicator_cache[cache_key]
        if (datetime.now(timezone.utc) - cache_time).total_seconds() < CACHE_EXPIRY.get(timeframe, 3600):
            return cached_data
    data = df.copy()
    indicators = {}
    indicators['atr'] = calculate_atr(data, timeframe=timeframe)
    indicators['macd'] = calculate_macd(data)
    indicators['rsi'] = calculate_rsi(data)
    indicators['bollinger'] = calculate_bollinger_bands(data)
    indicator_cache[cache_key] = (indicators, datetime.now(timezone.utc))
    return indicators

def calculate_fibonacci_levels(df: pd.DataFrame, initial_candles: int) -> List[float]:
    """Calculate Fibonacci retracement levels."""
    data = df[-initial_candles:]
    if len(data) < 2:
        return []
    swing_high, swing_low = data['high'].max(), data['low'].min()
    diff = swing_high - swing_low
    if diff <= 0:
        return []
    return [swing_low + diff * level for level in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]]

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    """Calculate MACD indicator."""
    if len(df) < max(fast, slow, signal):
        return False
    data = df['close'].ewm(span=fast, adjust=False).mean() - df['close'].ewm(span=slow, adjust=False).mean()
    signal_line = data.ewm(span=signal, adjust=False).mean()
    return data.iloc[-1] > signal_line.iloc[-1]

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate RSI indicator."""
    if len(df) < period:
        return 50.0
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Tuple[Optional[float], ...]:
    """Calculate Bollinger Bands."""
    if len(df) < period:
        return None, None, None
    sma = df['close'].rolling(window=period).mean().iloc[-1]
    std = df['close'].rolling(window=period).std().iloc[-1]
    return sma, sma + std * std_dev, sma - std * std_dev

def calculate_atr(df: pd.DataFrame, period: int = 14, timeframe: str = 'H1') -> float:
    """Calculate ATR with timeframe adjustment."""
    if len(df) < period or df.empty:
        return CONFIG['DEFAULT_EPS']
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift()))
    ).clip(lower=CONFIG['DEFAULT_EPS'] / 10).fillna(CONFIG['DEFAULT_EPS'] / 10)
    atr = tr.rolling(window=period).mean().iloc[-1]
    if np.isnan(atr) or atr <= 0:
        return CONFIG['DEFAULT_EPS']
    return atr * {'M1': 0.5, 'M5': 0.75, 'M15': 1.0, 'H1': 1.5}.get(timeframe, 1.0)

def calculate_volume_profile(df: pd.DataFrame, bins: int = 30) -> Tuple[List[float], List[float]]:
    """Optimized volume profile calculation."""
    if len(df) < CONFIG['MIN_CANDLES']:
        return [], []
    price_range = df['high'].max() - df['low'].min()
    if price_range <= 0:
        return [], []
    hist, bin_edges = np.histogram(df['close'], bins=bins, weights=df['volume'], range=(df['low'].min(), df['high'].max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    threshold = np.percentile(hist, 80)
    levels = bin_centers[hist > threshold].tolist()
    return levels, levels

def calculate_pivot_points(df: pd.DataFrame, initial_candles: int) -> Tuple[List[float], List[float]]:
    """Calculate pivot points for S/R."""
    data = df[-initial_candles:]
    if len(data) < 2:
        return [], []
    pivot = (data['high'].max() + data['low'].min() + data['close'].iloc[-1]) / 3
    return [2 * pivot - data['high'].max()], [2 * pivot - data['low'].min()]

def optimize_snr_parameters(df: pd.DataFrame, cache_key: str, use_grid_search: bool, 
                          timeframe: str) -> Tuple[float, int]:
    """Optimize DBSCAN parameters with caching."""
    cache_file = f"{cache_key}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            params, cache_time = pickle.load(f)
        if (datetime.now(timezone.utc) - cache_time).total_seconds() < 7200:  # 2-hour cache
            return params
    if not use_grid_search:
        return CONFIG['DEFAULT_EPS'], 2
    atr = calculate_atr(df, timeframe=timeframe)
    eps_range = [atr * 0.75] if not np.isnan(atr) else [CONFIG['DEFAULT_EPS']]  # Reduced search space
    min_samples_range = [2]
    results = [
        (backtest_sr_levels(df, deque(
            calculate_snr_line(df, initial_candles=50, eps=eps, min_samples=ms)[0:2] + [SRLevel(eps, df.index[-1], True)],
            maxlen=50)), (eps, ms))
        for eps, ms in product(eps_range, min_samples_range)
    ]
    best_params = max(results, key=lambda x: x[0])[1] if results else (CONFIG['DEFAULT_EPS'], 2)
    with open(cache_file, 'wb') as f:
        pickle.dump((best_params, datetime.now(timezone.utc)), f)
    return best_params

def calculate_snr_line(df: pd.DataFrame, initial_candles: int, eps: float, min_samples: int, 
                      use_volume_profile: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate support/resistance lines."""
    data = df[-initial_candles:].copy()
    if len(data) < CONFIG['MIN_CANDLES'] or data.empty:
        support, resistance = calculate_pivot_points(df, initial_candles)
        if not support or not resistance:
            current_price = data['close'].iloc[-1] if not data.empty else 1.0
            x = np.arange(max(len(data), 1))
            return np.full(len(x), current_price - CONFIG['DEFAULT_EPS']), np.full(len(x), current_price + CONFIG['DEFAULT_EPS'])
        return np.full(len(data), support[0]), np.full(len(data), resistance[0])
    
    if use_volume_profile:
        support, resistance = calculate_volume_profile(data)
        if not support or not resistance:
            support, resistance = calculate_pivot_points(data, initial_candles)
            if not support or not resistance:
                current_price = data['close'].iloc[-1]
                return np.full(len(data), current_price - CONFIG['DEFAULT_EPS']), np.full(len(data), current_price + CONFIG['DEFAULT_EPS'])
            return np.full(len(data), support[0]), np.full(len(data), resistance[0])
        current_price = data['close'].iloc[-1]
        return np.full(len(data), min(support, key=lambda x: abs(x - current_price))), \
               np.full(len(data), min(resistance, key=lambda x: abs(x - current_price)))
    
    indicators = calculate_indicators(data, 'H1', f"{data.index[-1]}_indicators")
    data['high_ema'] = data['high'].ewm(span=15, adjust=False).mean()
    data['low_ema'] = data['low'].ewm(span=15, adjust=False).mean()
    
    quantile = 0.75 if indicators['rsi'] > 50 else 0.70
    high_quantile = quantile if indicators['macd'] else quantile + 0.05
    low_quantile = 1 - quantile if indicators['macd'] else 1 - (quantile + 0.05)
    
    high_threshold = data['high_ema'].quantile(high_quantile) + indicators['atr'] * 0.2
    low_threshold = data['low_ema'].quantile(low_quantile) - indicators['atr'] * 0.2
    
    highs, lows = data['high_ema'][data['high_ema'] >= high_threshold].values, \
                  data['low_ema'][data['low_ema'] <= low_threshold].values
    if len(highs) == 0 or len(lows) == 0:
        support, resistance = calculate_pivot_points(data, initial_candles)
        if not support or not resistance:
            return np.full(len(data), data['close'].iloc[-1] - indicators['atr'] * 0.5), \
                   np.full(len(data), data['close'].iloc[-1] + indicators['atr'] * 0.5)
        return np.full(len(data), support[0]), np.full(len(data), resistance[0])
    
    highs, lows = highs.reshape(-1, 1), lows.reshape(-1, 1)
    db_highs = DBSCAN(eps=eps, min_samples=min_samples).fit(highs)
    db_lows = DBSCAN(eps=eps, min_samples=min_samples).fit(lows)
    
    resistance_levels = [highs[db_highs.labels_ == label].mean() for label in set(db_highs.labels_) if label != -1]
    support_levels = [lows[db_lows.labels_ == label].mean() for label in set(db_lows.labels_) if label != -1]
    
    current_price = data['close'].iloc[-1]
    resistance_price = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else high_threshold
    support_price = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else low_threshold
    
    return np.full(len(data), support_price), np.full(len(data), resistance_price)

def calculate_trend_line(df: pd.DataFrame, initial_candles: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate trend lines using RANSAC."""
    from sklearn.linear_model import RANSACRegressor
    data = df[-initial_candles:].copy()
    if len(data) < 3:
        x = np.arange(max(len(data), 1))
        low = data['low'].min() if not data.empty else 1.0
        high = data['high'].max() if not data.empty else 1.0
        return np.full(len(x), low), np.full(len(x), high)
    
    x = np.arange(len(data)).reshape(-1, 1)
    atr = calculate_atr(data)
    high_pivot, high_second = np.argmax(data['high'].values), np.argsort(data['high'].values)[-2]
    low_pivot, low_second = np.argmin(data['low'].values), np.argsort(data['low'].values)[1]
    
    high_prices = data['high'].values + atr * 0.2
    low_prices = data['low'].values - atr * 0.2
    
    ransac_support = RANSACRegressor().fit(x[[low_pivot, low_second]], low_prices[[low_pivot, low_second]])
    ransac_resistance = RANSACRegressor().fit(x[[high_pivot, high_second]], high_prices[[high_pivot, high_second]])
    
    return ransac_support.predict(x), ransac_resistance.predict(x)

def backtest_sr_levels(df: pd.DataFrame, levels: deque) -> float:
    """Backtest S/R levels for accuracy."""
    accuracy, total = 0, 0
    prices = df['close'].values
    for i in range(1, len(df) - 1):
        price = prices[i]
        for level in levels:
            if abs(price - level.price) < 0.0005:
                prev_price, next_price = prices[i-1], prices[i+1]
                if (prev_price > level.price and next_price < level.price) or \
                   (prev_price < level.price and next_price > level.price):
                    accuracy += 1
                total += 1
    return accuracy / total if total > 0 else 0

def generate_trading_signal(df: pd.DataFrame, support_snr: np.ndarray, resistance_snr: np.ndarray, 
                          support_trend: np.ndarray, resistance_trend: np.ndarray) -> str:
    """Generate trading signal based on S/R and Bollinger Bands."""
    if len(df) < 2:
        return "Hold"
    current_price = df['close'].iloc[-1]
    indicators = calculate_indicators(df, 'H1', f"{df.index[-1]}_signal")
    if indicators['bollinger'][0] is None:
        return "Hold"
    sma, upper_bb, lower_bb = indicators['bollinger']
    if (abs(current_price - support_snr[-1]) < indicators['atr'] * 0.1 and
        current_price > support_trend[-1] and current_price < sma and current_price > lower_bb):
        return "Buy"
    if (abs(current_price - resistance_snr[-1]) < indicators['atr'] * 0.1 and
        current_price < resistance_trend[-1] and current_price > sma and current_price < upper_bb):
        return "Sell"
    return "Hold"

def update_historical_sr_levels(df: pd.DataFrame, support_snr: np.ndarray, resistance_snr: np.ndarray, 
                              levels: deque = None, show_fib: bool = True):
    """Update historical S/R levels with Fibonacci integration."""
    levels = levels or historical_sr_levels
    latest_time = df.index[-1] if not df.empty else datetime.now(timezone.utc)
    support_price, resistance_price = support_snr[-1], resistance_snr[-1]
    price_diff = 0.0005
    
    if show_fib:
        for level in calculate_fibonacci_levels(df, initial_candles=50):
            if not any(abs(level - sr.price) < price_diff for sr in levels):
                levels.append(SRLevel(level, latest_time, True, significance=0.5))
    
    for level in levels:
        age_hours = (latest_time - level.timestamp).total_seconds() / 3600
        level.significance *= np.exp(-age_hours / 24)
        if level.is_support and abs(level.price - support_price) < price_diff or \
           not level.is_support and abs(level.price - resistance_price) < price_diff:
            level.update_touch()
            return
    
    levels.append(SRLevel(support_price, latest_time, True))
    levels.append(SRLevel(resistance_price, latest_time, False))

def save_sr_levels(levels: deque, filename: str = CONFIG['SR_LEVELS_FILE']):
    """Save S/R levels to file."""
    with open(filename, 'wb') as f:
        pickle.dump(list(levels), f)

def load_sr_levels(filename: str = CONFIG['SR_LEVELS_FILE']) -> deque:
    """Load S/R levels from file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return deque(pickle.load(f), maxlen=50)
    return deque(maxlen=50)

def aggregate_sr_levels(symbol: str, timeframes: List[str], limit: int) -> List[SRLevel]:
    """Aggregate S/R levels across timeframes."""
    all_levels = []
    weights = {'M1': 0.2, 'M5': 0.3, 'M15': 0.5}
    for timeframe in timeframes:
        df = fetch_ohlcv(symbol, timeframe, limit)
        if df is None:
            continue
        support_snr, resistance_snr = calculate_snr_line(df, initial_candles=50)
        weight = weights.get(timeframe, 1.0 / len(timeframes))
        all_levels.append(SRLevel(support_snr[-1], df.index[-1], True, significance=weight))
        all_levels.append(SRLevel(resistance_snr[-1], df.index[-1], False, significance=weight))
    if not all_levels:
        df_m1 = fetch_ohlcv(symbol, 'M1', limit)
        if df_m1 is not None:
            support_snr, resistance_snr = calculate_snr_line(df_m1, initial_candles=50)
            all_levels.extend([
                SRLevel(support_snr[-1], df_m1.index[-1], True, significance=0.2),
                SRLevel(resistance_snr[-1], df_m1.index[-1], False, significance=0.2)
            ])
    merged_levels = []
    price_diff = 0.0005
    for level in all_levels:
        matched = False
        for merged in merged_levels:
            if abs(level.price - merged.price) < price_diff and level.is_support == merged.is_support:
                merged.significance += level.significance
                merged.touches += 1
                matched = True
                break
        if not matched:
            merged_levels.append(level)
    return merged_levels

async def plot(plot_df: pd.DataFrame, symbol: str, timeframe: str, snr_candles: int = CONFIG['DISPLAY_CANDLES'], 
              trend_candles: int = CONFIG['DISPLAY_CANDLES'], backtest_index: int = 0):
    """Real-time plotting of OHLCV with S/R and trend lines."""
    if plot_df is None or len(plot_df) < CONFIG['DISPLAY_CANDLES']:
        logger.error(f"Insufficient data ({len(plot_df) if plot_df is not None else 0} candles)")
        return
    
    cache_key = f"{symbol}_{timeframe}"
    fig, ax_candles = plt.subplots(figsize=(12, 8))
    gs = fig.add_gridspec(7, 1, height_ratios=[4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], hspace=0.5)
    ax_candles = fig.add_subplot(gs[0])
    
    sliders = {
        'snr': Slider(fig.add_axes([0.15, 0.35, 0.65, 0.04]), 'SNR Candles', 10, 200, valinit=snr_candles, valstep=1),
        'candles': Slider(fig.add_axes([0.15, 0.30, 0.65, 0.04]), 'Candles', 10, 200, valinit=CONFIG['DISPLAY_CANDLES'], valstep=1),
        'trend': Slider(fig.add_axes([0.15, 0.25, 0.65, 0.04]), 'Trend Candles', 10, 200, valinit=trend_candles, valstep=1),
        'backtest': Slider(fig.add_axes([0.15, 0.20, 0.65, 0.04]), 'Backtest', 0, len(plot_df) - CONFIG['DISPLAY_CANDLES'], valinit=backtest_index, valstep=1),
    }
    checks = {
        'grid': CheckButtons(fig.add_axes([0.15, 0.15, 0.65, 0.04]), ['Grid Search', 'Volume Profile'], [True, False]),
        'fib': CheckButtons(fig.add_axes([0.15, 0.10, 0.65, 0.04]), ['Fibonacci'], [True])
    }
    
    plot_cache = {'support_snr': None, 'resistance_snr': None, 'support_trend': None, 'resistance_trend': None}
    
    async def update(frame=None):
        nonlocal plot_df
        try:
            ax_candles.clear()
            new_df = fetch_ohlcv(symbol, timeframe, CONFIG['LIMIT'], last_timestamp=plot_df.index[-1] if not plot_df.empty else None)
            if new_df is not None:
                plot_df = pd.concat([plot_df, new_df]).drop_duplicates().sort_index()[-CONFIG['LIMIT']:]
            
            if plot_df is None or len(plot_df) < CONFIG['DISPLAY_CANDLES']:
                logger.error(f"Plot data invalid ({len(plot_df) if plot_df is not None else 0} candles)")
                return
            
            snr_candles, candles, trend_candles = int(sliders['snr'].val), int(sliders['candles'].val), int(sliders['trend'].val)
            backtest_index = int(sliders['backtest'].val)
            use_grid_search, use_volume_profile = checks['grid'].get_status()
            show_fib = checks['fib'].get_status()[0]
            
            CONFIG['DISPLAY_CANDLES'] = candles
            display_df = plot_df.iloc[max(0, backtest_index):backtest_index + CONFIG['DISPLAY_CANDLES']]
            snr_data = plot_df.iloc[max(0, backtest_index - snr_candles):backtest_index + CONFIG['DISPLAY_CANDLES']]
            trend_data = plot_df.iloc[max(0, backtest_index - trend_candles):backtest_index + CONFIG['DISPLAY_CANDLES']]
            
            best_eps, best_min_samples = optimize_snr_parameters(snr_data, cache_key, use_grid_search, timeframe)
            support_snr, resistance_snr = calculate_snr_line(snr_data, min(snr_candles, len(snr_data)), 
                                                           best_eps, best_min_samples, use_volume_profile)
            support_trend, resistance_trend = calculate_trend_line(trend_data, min(trend_candles, len(trend_data)))
            
            update_historical_sr_levels(plot_df, support_snr, resistance_snr, show_fib=show_fib)
            sr_accuracy = backtest_sr_levels(plot_df, historical_sr_levels)
            signal = generate_trading_signal(plot_df, support_snr, resistance_snr, support_trend, resistance_trend)
            
            plot_cache.update({
                'support_snr': support_snr[-len(display_df):],
                'resistance_snr': resistance_snr[-len(display_df):],
                'support_trend': support_trend[-len(display_df):],
                'resistance_trend': resistance_trend[-len(display_df):]
            })
            
            addplots = [
                mpf.make_addplot(pd.DataFrame({
                    'support_snr': plot_cache['support_snr'], 'resistance_snr': plot_cache['resistance_snr'],
                    'support_trend': plot_cache['support_trend'], 'resistance_trend': plot_cache['resistance_trend']
                }), ax=ax_candles, colors=['green', 'red', 'blue', 'orange'], width=1.5)
            ]
            
            indicators = calculate_indicators(display_df, timeframe, f"{display_df.index[-1]}_plot")
            if indicators['bollinger'][0] is not None:
                sma, upper_bb, lower_bb = indicators['bollinger']
                addplots.append(mpf.make_addplot(pd.DataFrame({
                    'sma': np.full(len(display_df), sma), 'upper_bb': np.full(len(display_df), upper_bb),
                    'lower_bb': np.full(len(display_df), lower_bb)
                }), ax=ax_candles, colors=['purple', 'purple', 'purple'], width=1, linestyles=['-', '--', '--']))
            
            for level in historical_sr_levels:
                if level.timestamp >= display_df.index[0] and level.timestamp <= display_df.index[-1]:
                    addplots.append(mpf.make_addplot(np.full(len(display_df), level.price), 
                                                  color='darkgreen' if level.is_support else 'darkred', 
                                                  width=0.8, linestyle='--', ax=ax_candles))
                    ax_candles.annotate(
                        f"{'S' if level.is_support else 'R'}:{level.price:.5f} ({level.touches})",
                        xy=(display_df.index[-1], level.price), xytext=(5, 0), textcoords="offset points",
                        color='darkgreen' if level.is_support else 'darkred'
                    )
            
            ax_candles.set_ylim(
                display_df['low'].min() - max((display_df['high'].max() - display_df['low'].min()) * 0.15, indicators['atr'] * 0.5),
                display_df['high'].max() + max((display_df['high'].max() - display_df['low'].min()) * 0.15, indicators['atr'] * 0.5)
            )
            
            mpf.plot(
                display_df, type='candle', title=f"{symbol} - {timeframe} (S/R Accuracy: {sr_accuracy:.2%}, eps={best_eps:.6f}, Signal={signal})",
                ylabel='Price', addplot=addplots, style='classic', show_nontrading=False, ax=ax_candles
            )
            
            save_sr_levels(historical_sr_levels)
            fig.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Plot update error: {e}")
    
    for slider in sliders.values():
        slider.on_changed(lambda _: asyncio.create_task(update()))
    for check in checks.values():
        check.on_clicked(lambda _: asyncio.create_task(update()))
    
    ani = FuncAnimation(fig, lambda _: asyncio.create_task(update()), interval=CONFIG['UPDATE_INTERVAL'], cache_frame_data=False)
    
    try:
        await update()
        plt.show()
    except Exception as e:
        logger.error(f"Initial plot failed: {e}")
        raise

async def main():
    """Main entry point for the trading bot."""
    init_db()
    if not initialize_mt5():
        logger.error("Failed to initialize MT5, exiting")
        return
    
    global historical_sr_levels
    historical_sr_levels = load_sr_levels()
    
    for symbol in CONFIG['SYMBOLS']:
        historical_sr_levels.extend(aggregate_sr_levels(symbol, CONFIG['AGGREGATE_TIMEFRAMES'], CONFIG['LIMIT']))
        data, metadata = fetch_all_data([symbol], CONFIG['PLOT_TIMEFRAMES'], CONFIG['LIMIT'])
        for df, (_, timeframe) in zip(data, metadata):
            if df is not None:
                await plot(df, symbol, timeframe)
    
    mt5.shutdown()

if __name__ == '__main__':
    asyncio.run(main())