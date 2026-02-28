import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from telegram import Bot
from telegram.error import TelegramError
import asyncio
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter, deque
import mplfinance as mpf
from matplotlib.widgets import Slider, Button
import threading
import time
import argparse
import os
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MT5_LOGIN = 5038239857
MT5_PASSWORD = '-rCv0yJj'
MT5_SERVER = 'MetaQuotes-Demo'
TELEGRAM_BOT_TOKEN = '7451195794:AAFEWGL-Aejx1ZmxT2d1CIJV3b4Hfs6nvpo'
TELEGRAM_CHAT_ID = '5260699946'
SYMBOLS = ['EURGBP']
LIMIT = 2000
DISPLAY_CANDLES = 50
PLOT_TIMEFRAMES = ['M1']

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

# Global plotting variables
fig = None
axes = None
slider = None
data_dict_global = None
current_symbol = SYMBOLS[0]
buttons = []

# Risk management parameters
RISK_PER_TRADE = 0.05
BINARY_PAYOUT = 0.85
DEFAULT_ACCOUNT_BALANCE = 2000

# Class to store S/R levels
class SRLevel:
    def __init__(self, price, timestamp, is_support, touches=1, significance=1.0):
        self.price = price
        self.timestamp = timestamp
        self.is_support = is_support
        self.touches = touches
        self.significance = significance

    def update_touch(self):
        self.touches += 1
        self.significance += 0.2

# Global storage for historical S/R levels
historical_sr_levels = deque(maxlen=50)

def initialize_mt5(retries=3):
    for attempt in range(retries):
        try:
            if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, timeout=30000):
                for symbol in SYMBOLS:
                    if not mt5.symbol_select(symbol, True):
                        logger.error(f"Symbol {symbol} not available: {mt5.last_error()}")
                        mt5.shutdown()
                        return False
                logger.info("MT5 initialized successfully")
                return True
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        except Exception as e:
            logger.error(f"MT5 initialization attempt {attempt + 1}/{retries} failed: {e}")
        time.sleep(2)
    logger.error("All MT5 initialization attempts failed")
    return False

def get_account_balance():
    try:
        account_info = mt5.account_info()
        if account_info:
            return account_info.balance
        logger.warning("Failed to fetch account balance, using default")
        return DEFAULT_ACCOUNT_BALANCE
    except Exception as e:
        logger.error(f"Error fetching account balance: {e}")
        return DEFAULT_ACCOUNT_BALANCE

# Initialize Telegram bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

def timeframe_to_minutes(timeframe):
    timeframe_minutes = {
        mt5.TIMEFRAME_M1: 1,
        mt5.TIMEFRAME_M5: 5,
        mt5.TIMEFRAME_M15: 15,
        mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60,
        mt5.TIMEFRAME_H4: 240,
        mt5.TIMEFRAME_D1: 1440
    }
    return timeframe_minutes.get(timeframe, 1)

def fetch_ohlcv(symbol, timeframe, limit, retries=3):
    for attempt in range(retries):
        try:
            start_time = time.time()
            if not mt5.terminal_info():
                logger.warning("MT5 connection lost, attempting to reconnect")
                if not initialize_mt5():
                    return None
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, limit)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data fetched for {symbol} on {timeframe}: {mt5.last_error()}")
                if limit > 100:
                    logger.info(f"Retrying with reduced limit: {limit // 2}")
                    return fetch_ohlcv(symbol, timeframe, limit // 2, retries)
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df = df.rename(columns={
                'time': 'timestamp', 'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'tick_volume': 'volume'
            })
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            if df.empty:
                logger.error(f"Empty DataFrame for {symbol} on {timeframe}")
                return None
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            logger.debug(f"Fetched {len(df)} candles for {symbol} on {timeframe}, took {time.time() - start_time:.3f}s")
            logger.debug(f"DataFrame info: shape={df.shape}, columns={df.columns}, index_type={type(df.index)}")
            return df
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{retries} failed for {symbol} on {timeframe}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    logger.error(f"All {retries} attempts failed for {symbol} on {timeframe}")
    return None

def get_td(timeframe):
    td_map = {
        mt5.TIMEFRAME_M1: timedelta(minutes=1),
        mt5.TIMEFRAME_M5: timedelta(minutes=5),
        mt5.TIMEFRAME_M15: timedelta(minutes=15),
        mt5.TIMEFRAME_M30: timedelta(minutes=30),
        mt5.TIMEFRAME_H1: timedelta(hours=1),
        mt5.TIMEFRAME_H4: timedelta(hours=4),
        mt5.TIMEFRAME_D1: timedelta(days=1)
    }
    return td_map.get(timeframe, timedelta(minutes=1))

def calculate_snr(df, timeframe):
    snr_data = {'support': [], 'resistance': [], 'mini_snr': [], 'vssnr': [], 'ssnr': []}
    if timeframe in PLOT_TIMEFRAMES:
        recent = df.iloc[-20:]
        
        high_threshold = recent['high'].quantile(0.75)
        low_threshold = recent['low'].quantile(0.25)
        
        highs = recent['high'][recent['high'] >= high_threshold].values.reshape(-1, 1)
        lows = recent['low'][recent['low'] <= low_threshold].values.reshape(-1, 1)
        
        eps = 0.0002
        db_highs = DBSCAN(eps=eps, min_samples=2).fit(highs) if len(highs) >= 2 else None
        db_lows = DBSCAN(eps=eps, min_samples=2).fit(lows) if len(lows) >= 2 else None
        
        resistance_levels = [highs[db_highs.labels_ == label].mean() for label in set(db_highs.labels_) if label != -1] if db_highs else []
        support_levels = [lows[db_lows.labels_ == label].mean() for label in set(db_lows.labels_) if label != -1] if db_lows else []
        
        current_price = recent['close'].iloc[-1]
        resistance_price = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else high_threshold
        support_price = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else low_threshold
        
        if pd.notna(support_price) and np.isfinite(support_price):
            snr_data['support'].append(support_price)
        if pd.notna(resistance_price) and np.isfinite(resistance_price):
            snr_data['resistance'].append(resistance_price)
        
        for level in historical_sr_levels:
            if level.timestamp >= recent.index[0] and pd.notna(level.price) and np.isfinite(level.price):
                snr_data['mini_snr'].append(level.price)
        
        update_historical_sr_levels(recent, support_price, resistance_price)
    
    logger.debug(f"SNR for {timeframe}: {snr_data}")
    return snr_data

def update_historical_sr_levels(df, support_price, resistance_price):
    latest_time = df.index[-1]
    price_diff = 0.0002
    
    if not pd.notna(support_price) or not np.isfinite(support_price):
        logger.warning(f"Invalid support_price: {support_price}")
        support_price = None
    if not pd.notna(resistance_price) or not np.isfinite(resistance_price):
        logger.warning(f"Invalid resistance_price: {resistance_price}")
        resistance_price = None
    
    for level in historical_sr_levels:
        if support_price is not None and level.is_support and abs(level.price - support_price) < price_diff:
            level.update_touch()
            support_price = None
        if resistance_price is not None and not level.is_support and abs(level.price - resistance_price) < price_diff:
            level.update_touch()
            resistance_price = None
    
    if support_price is not None:
        historical_sr_levels.append(SRLevel(support_price, latest_time, True))
    if resistance_price is not None:
        historical_sr_levels.append(SRLevel(resistance_price, latest_time, False))

def is_hammer(candle):
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    if body == 0:
        return upper_wick > 0 or lower_wick > 0
    return max(upper_wick, lower_wick) >= 1.5 * body

def get_market_type(df):
    if len(df) < 20:
        return 'unknown'
    greens_20 = sum(1 for i in range(-20, 0) if df['close'].iloc[i] > df['open'].iloc[i])
    reds_20 = sum(1 for i in range(-20, 0) if df['close'].iloc[i] < df['open'].iloc[i])
    if greens_20 > 12:
        overall_trend = 'up'
    elif reds_20 > 12:
        overall_trend = 'down'
    else:
        overall_trend = 'no_clear_trend'

    greens_5 = sum(1 for i in range(-5, 0) if df['close'].iloc[i] > df['open'].iloc[i])
    reds_5 = sum(1 for i in range(-5, 0) if df['close'].iloc[i] < df['open'].iloc[i])
    dojis_5 = sum(1 for i in range(-5, 0) if df['high'].iloc[i] != df['low'].iloc[i] and
                  abs(df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i]) < 0.1)

    if overall_trend == 'no_clear_trend':
        return 'stack' if dojis_5 > 2 else 'ranging'
    if overall_trend == 'up':
        return 'trending_up'
    return 'trending_down'

def find_touch_snr(df, second_last_idx, resistance, support, direction):
    if second_last_idx < 0:
        return False
    second_last_close = df.iloc[second_last_idx]['close']
    max_lookback = 100
    for i in range(second_last_idx - 1, max(-1, second_last_idx - max_lookback), -1):
        candle = df.iloc[i]
        if direction == 'buy':
            if candle['close'] < second_last_close:
                return False
            if candle['close'] > second_last_close:
                return any(df.iloc[i:second_last_idx]['high'] >= resistance)
        else:
            if candle['close'] > second_last_close:
                return False
            if candle['close'] < second_last_close:
                return any(df.iloc[i:second_last_idx]['low'] <= support)
    return any(df.iloc[max(0, second_last_idx - max_lookback):second_last_idx]['high'] >= resistance) if direction == 'buy' else \
           any(df.iloc[max(0, second_last_idx - max_lookback):second_last_idx]['low'] <= support)

def detect_ss1(df):
    market_type = get_market_type(df)
    if market_type not in ['trending_up', 'trending_down'] or len(df) < 4:
        return None
    
    third_last = df.iloc[-4]
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    
    third_size = abs(third_last['close'] - third_last['open'])
    second_size = abs(second_last['close'] - second_last['open'])
    last_size = abs(last['close'] - last['open'])

    if market_type == 'trending_up':
        if (second_last['close'] < second_last['open']) and (last['close'] > last['open']) and third_last['close'] > third_last['open'] and (second_size < third_size and second_size < last_size):
            return {'type': 'Buy', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-1'}
    elif market_type == 'trending_down':
        if second_last['close'] > second_last['open'] and last['close'] < last['open'] and third_last['close'] < third_last['open'] and (second_size < third_size and second_size < last_size):
            return {'type': 'Sell', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-1'}
    return None

def detect_ss2(df, snr_data):
    market_type = get_market_type(df)
    if market_type not in ['ranging'] or len(df) < 8:
        return None
    
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    
    support_levels = snr_data['support'].copy()
    resistance_levels = snr_data['resistance'].copy()
    for level in snr_data['ssnr'] + snr_data['vssnr']:
        if pd.notna(level) and np.isfinite(level):
            if level < last['close']:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    resistance = min(resistance_levels) if resistance_levels else float('inf')
    support = max(support_levels) if support_levels else -float('inf')
    
    upper_wick_last = last['high'] - max(last['open'], last['close'])
    lower_wick_last = min(last['open'], last['close']) - last['low']

    if second_last['close'] < second_last['open'] and (second_last['low'] <= support or last['low'] <= support) and (last['close'] > last['open']) and lower_wick_last == 0:
        touched_resistance = find_touch_snr(df, -3, resistance, support, 'buy')
        if touched_resistance:
            logger.debug(f"SS-2 Buy: Candle at support {support}, came from resistance {resistance}")
            return {
                'type': 'Buy',
                'timestamp': last.name,
                'price': last['close'],
                'sure_shot': 'SS-2'
            }
    
    elif second_last['close'] > second_last['open'] and (second_last['high'] >= resistance or last['high'] >= resistance) and (last['close'] < last['open']) and upper_wick_last == 0:
        touched_support = find_touch_snr(df, -3, resistance, support, 'sell')
        if touched_support:
            logger.debug(f"SS-2 Sell: Candle at resistance {resistance}, came from support {support}")
            return {
                'type': 'Sell',
                'timestamp': last.name,
                'price': last['close'],
                'sure_shot': 'SS-2'
            }
    
    return None

def detect_ss3(df, snr_data):
    market_type = get_market_type(df)
    if market_type not in ['trending_up', 'trending_down'] or len(df) < 2:
        return None
    
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    
    support_levels = snr_data['support'].copy()
    resistance_levels = snr_data['resistance'].copy()
    for level in snr_data['vssnr']:
        if pd.notna(level) and np.isfinite(level):
            if level < last['close']:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    resistance = min(resistance_levels) if resistance_levels else float('inf')
    support = max(support_levels) if support_levels else -float('inf')

    if market_type == 'trending_up':
        if second_last['close'] < second_last['open'] and last['close'] > last['open']:
            if last['close'] > resistance:
                return {'type': 'Buy', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-3'}
    elif market_type == 'trending_down':
        if second_last['close'] > second_last['open'] and last['close'] < last['open']:
            if last['close'] < support:
                return {'type': 'Sell', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-3'}
    return None

def detect_ss4(df):
    market_type = get_market_type(df)
    if market_type != 'stack' or len(df) < 2:
        return None
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    if is_hammer(second_last):
        if last['close'] > second_last['high']:
            return {'type': 'Buy', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-4'}
        elif last['close'] < second_last['low']:
            return {'type': 'Sell', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-4'}
    return None

def detect_ss5(df, snr_data):
    market_type = get_market_type(df)
    if market_type not in ['trending_up', 'trending_down'] or len(df) < 2:
        return None
    
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    
    support_levels = snr_data['support'].copy()
    resistance_levels = snr_data['resistance'].copy()
    for level in snr_data['vssnr']:
        if pd.notna(level) and np.isfinite(level):
            if level < last['close']:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    resistance = min(resistance_levels) if resistance_levels else float('inf')
    support = max(support_levels) if support_levels else -float('inf')
    
    if market_type == 'trending_up':
        if second_last['close'] > second_last['open'] and second_last['close'] > resistance:
            if last['close'] < last['open'] and last['close'] >= support and last['low'] <= support:
                return {'type': 'Buy', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-5'}
    elif market_type == 'trending_down':
        if second_last['close'] < second_last['open'] and second_last['close'] < support:
            if last['close'] > last['open'] and last['close'] <= resistance and last['high'] >= resistance:
                return {'type': 'Sell', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-5'}
    return None

def detect_ss6(df, snr_data):
    market_type = get_market_type(df)
    if market_type not in ['ranging'] or len(df) < 6:
        return None
    
    last = df.iloc[-2]
    
    support_levels = snr_data['support'].copy()
    resistance_levels = snr_data['resistance'].copy()
    for level in snr_data['ssnr'] + snr_data['vssnr']:
        if pd.notna(level) and np.isfinite(level):
            if level < last['close']:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    resistance = min(resistance_levels) if resistance_levels else float('inf')
    support = max(support_levels) if support_levels else -float('inf')
    
    upper_wick = last['high'] - max(last['open'], last['close'])
    lower_wick = min(last['open'], last['close']) - last['low']
    if is_hammer(last):
        if last['low'] <= support and last['close'] >= support and upper_wick >= lower_wick:
            touched_resistance = find_touch_snr(df, -2, resistance, support, 'buy')
            if touched_resistance:
                logger.debug(f"SS-6 Buy: Hammer at support {support}, came from resistance {resistance}")
                return {
                    'type': 'Buy',
                    'timestamp': last.name,
                    'price': last['close'],
                    'sure_shot': 'SS-6'
                }
        elif last['high'] >= resistance and last['close'] <= resistance and lower_wick >= upper_wick:
            touched_support = find_touch_snr(df, -2, resistance, support, 'sell')
            if touched_support:
                logger.debug(f"SS-6 Sell: Hammer at resistance {resistance}, came from support {support}")
                return {
                    'type': 'Sell',
                    'timestamp': last.name,
                    'price': last['close'],
                    'sure_shot': 'SS-6'
                }
    
    return None

def detect_ss7(df):
    market_type = get_market_type(df)
    if market_type not in ['trending_up', 'trending_down'] or len(df) < 2:
        return None

    last = df.iloc[-2]
    
    upper_wick = last['high'] - max(last['open'], last['close'])
    lower_wick = min(last['open'], last['close']) - last['low']

    if market_type == 'trending_up' and is_hammer(last) and last['close'] > last['open'] and upper_wick > lower_wick:
        return {'type': 'Buy', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-7'}
    elif market_type == 'trending_down' and is_hammer(last) and last['close'] < last['open'] and lower_wick > upper_wick:
        return {'type': 'Sell', 'timestamp': last.name, 'price': last['close'], 'sure_shot': 'SS-7'}
    return None

def detect_ss8(df, snr_data):
    market_type = get_market_type(df)
    if market_type not in ['trending_up', 'trending_down'] or len(df) < 6:
        return None
    
    third_last = df.iloc[-4]
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    
    support_levels = snr_data['support'].copy()
    resistance_levels = snr_data['resistance'].copy()
    for level in snr_data['ssnr'] + snr_data['vssnr']:
        if pd.notna(level) and np.isfinite(level):
            if level < last['close']:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    resistance = min(resistance_levels) if resistance_levels else float('inf')
    support = max(support_levels) if support_levels else -float('inf')

    if market_type == 'trending_up':
        if third_last['close'] < third_last['open'] and third_last['close'] <= support and third_last['high'] >= support:
            if second_last['close'] > second_last['open'] and second_last['close'] > support:
                if last['close'] < last['open'] and last['low'] <= support and last['close'] == third_last['close']:
                    return {
                        'type': 'Buy',
                        'timestamp': last.name,
                        'price': last['close'],
                        'sure_shot': 'SS-8'
                    }
    elif market_type == 'trending_down':
        if third_last['close'] > third_last['open'] and third_last['close'] >= resistance and third_last['low'] <= resistance:
            if second_last['close'] < second_last['open'] and second_last['close'] < resistance:
                if last['close'] > last['open'] and last['low'] >= resistance and last['close'] == third_last['close']:
                    return {
                        'type': 'Sell',
                        'timestamp': last.name,
                        'price': last['close'],
                        'sure_shot': 'SS-8'
                    }
    
    return None

def detect_ss_ya(df, snr_data):
    market_type = get_market_type(df)
    if market_type not in ['ranging'] or len(df) < 8:
        return None
    
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    
    support_levels = snr_data['support'].copy()
    resistance_levels = snr_data['resistance'].copy()
    for level in snr_data['ssnr'] + snr_data['vssnr']:
        if pd.notna(level) and np.isfinite(level):
            if level < last['close']:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    resistance = min(resistance_levels) if resistance_levels else float('inf')
    support = max(support_levels) if support_levels else -float('inf')

    upper_wick_last = last['high'] - max(last['open'], last['close'])
    upper_wick_second_last = second_last['high'] - max(second_last['open'], second_last['close'])
    lower_wick_last = min(last['open'], last['close']) - last['low']
    lower_wick_second_last = min(second_last['open'], second_last['close']) - second_last['low']

    if is_hammer(last) and is_hammer(second_last):
        if last['low'] <= support and last['close'] > support and lower_wick_last >= upper_wick_last and upper_wick_second_last >= lower_wick_second_last and second_last['close'] > support:
            touched_resistance = find_touch_snr(df, -3, resistance, support, 'buy')
            if touched_resistance:
                logger.debug(f"SS-YA Buy: Hammer at support {support}, came from resistance {resistance}")
                return {
                    'type': 'Buy',
                    'timestamp': last.name,
                    'price': last['close'],
                    'sure_shot': 'SS-YA'
                }
        elif last['high'] >= resistance and last['close'] < resistance and lower_wick_last <= upper_wick_last and upper_wick_second_last <= lower_wick_second_last and second_last['close'] < resistance:
            touched_support = find_touch_snr(df, -3, resistance, support, 'sell')
            if touched_support:
                logger.debug(f"SS-YA Sell: Hammer at resistance {resistance}, came from support {support}")
                return {
                    'type': 'Sell',
                    'timestamp': last.name,
                    'price': last['close'],
                    'sure_shot': 'SS-YA'
                }
    
    return None

def detect_ss_ys(df, snr_data):
    market_type = get_market_type(df)
    if market_type not in ['ranging'] or len(df) < 8:
        logger.debug(f"SS-YS failed: market_type={market_type}, df_length={len(df)}")
        return None
    
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    
    support_levels = snr_data['support'].copy()
    resistance_levels = snr_data['resistance'].copy()
    for level in snr_data['ssnr'] + snr_data['vssnr']:
        if pd.notna(level) and np.isfinite(level):
            if level < last['close']:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    resistance = min(resistance_levels) if resistance_levels else float('inf')
    support = max(support_levels) if support_levels else -float('inf')

    if not is_hammer(last):
        logger.debug(f"SS-YS failed: last candle not a hammer at {last.name}")
        return None
    
    if is_hammer(second_last) and is_hammer(last):
        if last['low'] <= support and last['close'] > support and second_last['low'] <= support and second_last['close'] > support:
            touched_resistance = find_touch_snr(df, -3, resistance, support, 'buy')
            if touched_resistance:
                logger.debug(f"SS-YS Buy: Hammer at support {support}, came from resistance {resistance}")
                return {
                    'type': 'Buy',
                    'timestamp': last.name,
                    'price': last['close'],
                    'sure_shot': 'SS-YS'
                }
    
        elif last['high'] >= resistance and last['close'] < resistance and second_last['high'] >= resistance and second_last['close'] < resistance:
            touched_support = find_touch_snr(df, -3, resistance, support, 'sell')
            if touched_support:
                logger.debug(f"SS-YS Sell: Hammer at resistance {resistance}, came from support {support}")
                return {
                    'type': 'Sell',
                    'timestamp': last.name,
                    'price': last['close'],
                    'sure_shot': 'SS-YS'
                }
    
    logger.debug(f"SS-YS failed: no SNR touch or invalid candle position at {last.name}")
    return None

sure_shots = [
    {'name': 'SS-1', 'detect': detect_ss1, 'requires_snr': False},
    {'name': 'SS-2', 'detect': detect_ss2, 'requires_snr': True},
    {'name': 'SS-3', 'detect': detect_ss3, 'requires_snr': True},
    {'name': 'SS-4', 'detect': detect_ss4, 'requires_snr': False},
    {'name': 'SS-5', 'detect': detect_ss5, 'requires_snr': True},
    {'name': 'SS-6', 'detect': detect_ss6, 'requires_snr': True},
    {'name': 'SS-7', 'detect': detect_ss7, 'requires_snr': False},
    {'name': 'SS-8', 'detect': detect_ss8, 'requires_snr': True},
    {'name': 'SS-YA', 'detect': detect_ss_ya, 'requires_snr': True},
    {'name': 'SS-YS', 'detect': detect_ss_ys, 'requires_snr': True},
]

def generate_signal(df, timeframe, symbol, sure_shot_name=None):
    logger.debug(f"Generating signal for {symbol} on {timeframe}, sure_shot={sure_shot_name}")
    snr_data = calculate_snr(df, timeframe)
    if sure_shot_name:
        for ss in sure_shots:
            if ss['name'] == sure_shot_name:
                signal = ss['detect'](df, snr_data) if ss['requires_snr'] else ss['detect'](df)
                if signal:
                    signal['sure_shot'] = ss['name']
                    signal['timeframe'] = timeframe
                    signal['symbol'] = symbol
                    signal['snr_data'] = snr_data
                    logger.debug(f"Signal generated for {symbol} on {timeframe}: {signal}")
                    return signal
                logger.debug(f"No signal for {ss['name']} on {symbol} {timeframe}")
                return None
        return None
    else:
        for ss in sure_shots:
            signal = ss['detect'](df, snr_data) if ss['requires_snr'] else ss['detect'](df)
            if signal:
                signal['sure_shot'] = ss['name']
                signal['timeframe'] = timeframe
                signal['symbol'] = symbol
                signal['snr_data'] = snr_data
                logger.debug(f"Signal generated for {symbol} on {timeframe}: {signal}")
                return signal
        logger.debug(f"No signals for {symbol} on {timeframe}")
        return None

async def send_telegram_message(message, retries=3):
    for attempt in range(retries):
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.debug(f"Sent Telegram message")
            return
        except TelegramError as e:
            logger.error(f"Attempt {attempt + 1}/{retries} failed to send Telegram message: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
    logger.error("All attempts to send Telegram message failed")

def set_symbol(symbol):
    global current_symbol
    current_symbol = symbol
    logger.info(f"Switched to symbol: {symbol}")
    if data_dict_global:
        plot_signals_multi(data_dict_global)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from matplotlib.widgets import Slider, Button
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SYMBOLS = ['EURGBP']
PLOT_TIMEFRAMES = ['M1']
DISPLAY_CANDLES = 50
current_symbol = SYMBOLS[0]

def plot_signals_multi(data_dict, initial=False):
    """Plot candlesticks, SNR lines, and signals for multiple timeframes with a slider."""
    global fig, axes, slider, data_dict_global, buttons
    data_dict_global = data_dict

    if initial or fig is None:
        plt.close('all')
        try:
            fig = plt.figure(figsize=(12, 8), dpi=150)
            n_timeframes = len(PLOT_TIMEFRAMES)
            if n_timeframes == 0:
                logger.error("PLOT_TIMEFRAMES is empty")
                fig.text(0.5, 0.5, "No timeframes specified", ha='center', va='center')
                plt.show(block=True)
                return
            
            # Create subplot grid
            rows = min(2, (n_timeframes + 1) // 2)
            cols = min(2, n_timeframes)
            gs = fig.add_gridspec(rows, cols, top=0.92, bottom=0.18, hspace=0.4, wspace=0.3)
            axes = [fig.add_subplot(gs[i//cols, i%cols]) for i in range(n_timeframes)]
            logger.debug(f"Created {n_timeframes} subplots: rows={rows}, cols={cols}")
            
            # Button axes for symbol selection
            button_axes = [fig.add_axes([0.05 + i*0.15, 0.96, 0.1, 0.04]) for i in range(len(SYMBOLS))]
            buttons = []
            for i, symbol in enumerate(SYMBOLS):
                logger.debug(f"Creating button for symbol: {symbol}")
                button = Button(button_axes[i], symbol)
                button.on_clicked(lambda event, s=symbol: set_symbol(s))
                buttons.append(button)
            
            # Compute maximum candles for slider
            valid_lengths = []
            for symbol in SYMBOLS:
                for tf in PLOT_TIMEFRAMES:
                    try:
                        if (symbol in data_dict and tf in data_dict[symbol] and 
                            data_dict[symbol][tf]['df'] is not None and not data_dict[symbol][tf]['df'].empty):
                            valid_lengths.append(len(data_dict[symbol][tf]['df']))
                    except KeyError as e:
                        logger.warning(f"KeyError accessing data_dict[{symbol}][{tf}]: {e}")
                        continue
            max_candles = min(valid_lengths) if valid_lengths else 0
            if max_candles <= DISPLAY_CANDLES:
                logger.warning("Not enough data for sliding window")
                for ax, tf in zip(axes, PLOT_TIMEFRAMES):
                    ax.text(0.5, 0.5, f"No data for {tf}", ha='center', va='center')
                    ax.set_title(f"{tf} Chart ({current_symbol})")
                plt.tight_layout(pad=2.0)
                plt.show(block=True)
                return
            
            # Create slider
            slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.04])
            slider = Slider(slider_ax, 'Candle Index', 0, max_candles - DISPLAY_CANDLES, 
                            valinit=max_candles - DISPLAY_CANDLES, valstep=1)
            slider.on_changed(update)
            logger.debug(f"Slider initialized with range 0 to {max_candles - DISPLAY_CANDLES}")
        
        except Exception as e:
            logger.error(f"Failed to initialize plot: {e}")
            fig = plt.figure(figsize=(8, 6))
            fig.text(0.5, 0.5, f"Plot initialization failed: {e}", ha='center', va='center')
            plt.show(block=True)
            return
    
    update(slider.val if slider else 0)
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.tight_layout(pad=2.0)
        plt.pause(0.01)  # Ensure event loop is processed
        logger.debug("Plot initialized or updated")
    except Exception as e:
        logger.error(f"Failed to draw plot: {e}")
    plt.show(block=False)

def update(val):
    global axes, data_dict_global, current_symbol
    if data_dict_global is None or current_symbol not in data_dict_global:
        logger.warning(f"No data to update plot for {current_symbol}")
        return
    
    start_idx = int(val)
    end_idx = start_idx + DISPLAY_CANDLES
    for ax_idx, (ax, tf) in enumerate(zip(axes, PLOT_TIMEFRAMES)):
        ax.clear()
        data = data_dict_global.get(current_symbol, {}).get(tf, {})
        df = data.get('df')
        signals = data.get('signals', [])
        
        if df is None or df.empty:
            logger.warning(f"No data for {current_symbol} on {tf}")
            ax.text(0.5, 0.5, f"No data for {tf}", ha='center', va='center')
            ax.set_title(f"{tf} Chart ({current_symbol})")
            continue
        
        logger.debug(f"Processing {current_symbol} on {tf}: df length={len(df)}, start_idx={start_idx}, end_idx={end_idx}")
        window_df = df.iloc[max(0, start_idx):end_idx].copy()
        if window_df.empty:
            logger.warning(f"Empty window for {current_symbol} on {tf} at start_idx={start_idx}")
            ax.text(0.5, 0.5, f"No data in window for {tf}", ha='center', va='center')
            ax.set_title(f"{tf} Chart ({current_symbol})")
            continue
        
        # Validate DataFrame
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in window_df.columns for col in required_cols):
            logger.error(f"Invalid columns for {current_symbol} on {tf}: {window_df.columns}")
            ax.text(0.5, 0.5, f"Invalid data for {tf}", ha='center', va='center')
            ax.set_title(f"{tf} Chart ({current_symbol})")
            continue
        if not isinstance(window_df.index, pd.DatetimeIndex):
            logger.error(f"Invalid index type for {current_symbol} on {tf}: {type(window_df.index)}")
            ax.text(0.5, 0.5, f"Invalid index for {tf}", ha='center', va='center')
            ax.set_title(f"{tf} Chart ({current_symbol})")
            continue
        if not window_df[required_cols].apply(pd.to_numeric, errors='coerce').notna().all().all():
            logger.error(f"Invalid numeric data in {required_cols} for {current_symbol} on {tf}")
            ax.text(0.5, 0.5, f"Invalid data for {tf}", ha='center', va='center')
            ax.set_title(f"{tf} Chart ({current_symbol})")
            continue
        
        logger.debug(f"Window for {current_symbol} on {tf}: shape={window_df.shape}, columns={window_df.columns}")
        
        # Calculate SNR
        snr_window_df = df.iloc[:end_idx]
        if len(snr_window_df) < 20:
            logger.warning(f"Insufficient data for SNR calculation for {current_symbol} on {tf}: {len(snr_window_df)} candles")
            snr_data = {'support': [], 'resistance': [], 'mini_snr': [], 'vssnr': [], 'ssnr': []}
        else:
            try:
                snr_data = calculate_snr(snr_window_df, tf)
                logger.debug(f"SNR data for {current_symbol} on {tf}: {snr_data}")
            except Exception as e:
                logger.error(f"Error calculating SNR for {current_symbol} on {tf}: {e}")
                snr_data = {'support': [], 'resistance': [], 'mini_snr': [], 'vssnr': [], 'ssnr': []}
        
        # Prepare addplots for SNR lines
        apds = []
        support_labeled = False
        resistance_labeled = False
        mini_snr_labeled = False
        vssnr_labeled = False
        ssnr_labeled = False
        buy_labeled = False
        sell_labeled = False
        
        # Add SNR lines using pd.Series
        for level in snr_data.get('support', []):
            if pd.notna(level) and np.isfinite(level):
                try:
                    apds.append(mpf.make_addplot(
                        pd.Series(level, index=window_df.index),
                        color='green', linestyle='--'
                    ))
                    logger.debug(f"Added support for {current_symbol} on {tf}: {level}")
                    support_labeled = True
                except Exception as e:
                    logger.error(f"Error adding support for {current_symbol} on {tf}: {e}")
        for level in snr_data.get('resistance', []):
            if pd.notna(level) and np.isfinite(level):
                try:
                    apds.append(mpf.make_addplot(
                        pd.Series(level, index=window_df.index),
                        color='red', linestyle='--',
                    ))
                    logger.debug(f"Added resistance for {current_symbol} on {tf}: {level}")
                    resistance_labeled = True
                except Exception as e:
                    logger.error(f"Error adding resistance for {current_symbol} on {tf}: {e}")
        for level in snr_data.get('mini_snr', []):
            if pd.notna(level) and np.isfinite(level):
                try:
                    apds.append(mpf.make_addplot(
                        pd.Series(level, index=window_df.index),
                        color='blue', linestyle=':', 
                    ))
                    logger.debug(f"Added mini_snr for {current_symbol} on {tf}: {level}")
                    mini_snr_labeled = True
                except Exception as e:
                    logger.error(f"Error adding mini_snr for {current_symbol} on {tf}: {e}")
        for level in snr_data.get('vssnr', []):
            if pd.notna(level) and np.isfinite(level):
                try:
                    apds.append(mpf.make_addplot(
                        pd.Series(level, index=window_df.index),
                        color='purple', linestyle='-',
                    ))
                    logger.debug(f"Added vssnr for {current_symbol} on {tf}: {level}")
                    vssnr_labeled = True
                except Exception as e:
                    logger.error(f"Error adding vssnr for {current_symbol} on {tf}: {e}")
        for level in snr_data.get('ssnr', []):
            if pd.notna(level) and np.isfinite(level):
                try:
                    apds.append(mpf.make_addplot(
                        pd.Series(level, index=window_df.index),
                        color='orange', linestyle='-', 
                    ))
                    logger.debug(f"Added ssnr for {current_symbol} on {tf}: {level}")
                    ssnr_labeled = True
                except Exception as e:
                    logger.error(f"Error adding ssnr for {current_symbol} on {tf}: {e}")
        
        # Plot candlesticks
        plot_kwargs = {
            'type': 'candle',
            'volume': False,
            'ax': ax,
            'ylabel': 'Price',
            'show_nontrading': False,
            'title': f"{tf} Chart ({current_symbol})",
            'style': 'classic'
        }
        if apds:
            plot_kwargs['addplot'] = apds
        
        try:
            logger.debug(f"Plotting candlesticks with {len(apds)} addplots for {current_symbol} on {tf}: {len(window_df)} candles")
            mpf.plot(window_df[['open', 'high', 'low', 'close']], **plot_kwargs)
            logger.debug(f"Candlesticks and SNR lines plotted successfully for {current_symbol} on {tf}")
        except Exception as e:
            logger.error(f"Error plotting candlesticks with addplots for {current_symbol} on {tf}: {e}")
            # Fallback: Plot candlesticks without addplots
            try:
                mpf.plot(window_df[['open', 'high', 'low', 'close']],
                         type='candle', volume=False, ax=ax, show_nontrading=False,
                         ylabel='Price', title=f"{tf} Chart ({current_symbol})", style='classic')
                logger.debug(f"Fallback: Candlesticks plotted without addplots for {current_symbol} on {tf}")
                # Add SNR lines manually
                for level in snr_data.get('support', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'Support' if not support_labeled else None
                        ax.axhline(y=level, color='green', linestyle='--', alpha=0.7)
                        support_labeled = True
                for level in snr_data.get('resistance', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'Resistance' if not resistance_labeled else None
                        ax.axhline(y=level, color='red', linestyle='--', alpha=0.7)
                        resistance_labeled = True
                for level in snr_data.get('mini_snr', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'Mini SNR' if not mini_snr_labeled else None
                        ax.axhline(y=level, color='blue', linestyle=':', alpha=0.7)
                        mini_snr_labeled = True
                for level in snr_data.get('vssnr', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'VSSNR' if not vssnr_labeled else None
                        ax.axhline(y=level, color='purple', linestyle='-', alpha=0.7)
                        vssnr_labeled = True
                for level in snr_data.get('ssnr', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'SSNR' if not ssnr_labeled else None
                        ax.axhline(y=level, color='orange', linestyle='-', alpha=0.7)
                        ssnr_labeled = True
            except Exception as e2:
                logger.error(f"Fallback plotting failed for {current_symbol} on {tf}: {e2}")
                ax.text(0.5, 0.5, f"Plot failed for {tf}", ha='center', va='center')
                ax.set_title(f"{tf} Chart ({current_symbol})")
                continue
        
        # Plot signals
        label_positions = {}
        for signal in signals:
            try:
                ts = signal.get('timestamp')
                if ts in window_df.index:
                    idx = window_df.index.get_loc(ts)
                    signal_type = signal['type']
                    ss_name = signal['sure_shot']
                    price = signal['price']
                    if signal_type == 'Buy':
                        y_pos_marker = window_df.iloc[idx]['low'] * 0.9999
                        base_y_pos_label = window_df.iloc[idx]['low'] * 0.9996
                        color = 'green'
                        marker = '^'
                        label = 'Buy Signal' if not buy_labeled else None
                        buy_labeled = True
                        key = 'buy'
                        va = 'bottom'
                        offset_direction = -1
                    else:
                        y_pos_marker = window_df.iloc[idx]['high'] * 1.0001
                        base_y_pos_label = window_df.iloc[idx]['high'] * 1.0004
                        color = 'red'
                        marker = 'v'
                        label = 'Sell Signal' if not sell_labeled else None
                        sell_labeled = True
                        key = 'sell'
                        va = 'top'
                        offset_direction = 1
                    if idx not in label_positions:
                        label_positions[idx] = {'buy': 0, 'sell': 0}
                    signal_count = label_positions[idx][key]
                    price_level = window_df.iloc[idx]['close']
                    offset_percentage = 0.02
                    text_offset = (price_level * offset_percentage / 100) * signal_count * offset_direction
                    y_pos_label = base_y_pos_label + text_offset
                    label_positions[idx][key] += 1
                    ax.scatter(window_df.index[idx], y_pos_marker, marker=marker, color=color, s=100, zorder=10,)
                    ax.text(window_df.index[idx], y_pos_label, ss_name, fontsize=8, color=color, ha='center', va=va, zorder=10)
                    logger.debug(f"Plotted {signal_type} signal for {current_symbol} on {tf} at {ts}")
            except Exception as e:
                logger.warning(f"Error plotting signal for {current_symbol} on {tf}: {e}")
                continue
        
        # Adjust y-limits
        try:
            y_min = window_df['low'].min()
            y_max = window_df['high'].max()
            if pd.notna(y_min) and pd.notna(y_max) and np.isfinite(y_min) and np.isfinite(y_max):
                all_levels = []
                for key in ['support', 'resistance', 'mini_snr', 'vssnr', 'ssnr']:
                    all_levels.extend([x for x in snr_data.get(key, []) if pd.notna(x) and np.isfinite(x)])
                for signal in signals:
                    ts = signal.get('timestamp')
                    if ts in window_df.index:
                        idx = window_df.index.get_loc(ts)
                        signal_type = signal['type']
                        price_level = window_df.iloc[idx]['close']
                        offset_percentage = 0.02
                        if signal_type == 'Buy':
                            y_pos_marker = window_df.iloc[idx]['low'] * 0.9999
                            base_y_pos_label = window_df.iloc[idx]['low'] * 0.9996
                            offset_direction = -1
                        else:
                            y_pos_marker = window_df.iloc[idx]['high'] * 1.0001
                            base_y_pos_label = window_df.iloc[idx]['high'] * 1.0004
                            offset_direction = 1
                        signal_count = label_positions.get(idx, {'buy': 0, 'sell': 0})[signal_type.lower()]
                        y_pos_label = base_y_pos_label + (price_level * offset_percentage / 100) * signal_count * offset_direction
                        all_levels.extend([y_pos_marker, y_pos_label])
                if all_levels:
                    y_min = min(y_min, min(all_levels)) * 0.9995
                    y_max = max(y_max, max(all_levels)) * 1.0005
                price_range = y_max - y_min
                y_min -= price_range * 0.02
                y_max += price_range * 0.02
                ax.set_ylim(y_min, y_max)
            if buy_labeled or sell_labeled or support_labeled or resistance_labeled or mini_snr_labeled or vssnr_labeled or ssnr_labeled:
                ax.legend()
        except Exception as e:
            logger.error(f"Error setting y-limits for {current_symbol} on {tf}: {e}")
    
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)
        logger.info(f"Updated plot with slider at index {start_idx}")
    except Exception as e:
        logger.error(f"Error updating canvas for {current_symbol}: {e}")
    

def plot_backtest_signals(symbols=SYMBOLS, timeframes=PLOT_TIMEFRAMES, limit=LIMIT, display_candles=50):
    global fig, axes, slider, data_dict_global, current_symbol
    logger.info(f"Plotting backtest signals for symbols {symbols} on timeframes {timeframes}")
    
    if not initialize_mt5():
        logger.error("MT5 initialization failed")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "MT5 initialization failed", ha='center', va='center')
        plt.show(block=True)
        return

    try:
        results_df = pd.read_csv('backtest_results.csv')
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'], utc=True)
        logger.info(f"Loaded {len(results_df)} signals from backtest_results.csv")
        logger.debug(f"Backtest results columns: {results_df.columns}")
    except FileNotFoundError:
        logger.error("backtest_results.csv not found")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "backtest_results.csv not found", ha='center', va='center')
        plt.show(block=True)
        mt5.shutdown()
        return
    except Exception as e:
        logger.error(f"Error loading backtest_results.csv: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error loading CSV: {e}", ha='center', va='center')
        plt.show(block=True)
        mt5.shutdown()
        return

    data_dict_global = {}
    min_candles = float('inf')
    for symbol in symbols:
        data_dict_global[symbol] = {}
        for timeframe in timeframes:
            df = fetch_ohlcv(symbol, TIMEFRAME_MAP[timeframe], limit)
            if df is None or df.empty:
                logger.error(f"No data for {symbol} on {timeframe}")
                continue
            
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error(f"Invalid index type for {symbol} on {timeframe}: {type(df.index)}")
                continue
            
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol} on {timeframe}: {df.columns}")
                continue
            
            signals_df = results_df[(results_df['symbol'] == symbol) & (results_df['timeframe'] == timeframe)]
            signals = []
            for _, row in signals_df.iterrows():
                if row['timestamp'] in df.index:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'type': row['type'],
                        'sure_shot': row['sure_shot'],
                        'price': row['entry_price']
                    })
            logger.debug(f"Prepared {len(signals)} signals for {symbol} on {timeframe}")
            
            data_dict_global[symbol][timeframe] = {
                'df': df,
                'signals': signals,
                'snr_data': None
            }
            min_candles = min(min_candles, len(df))
    
    if not data_dict_global:
        logger.error("No valid data available for plotting")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No valid data available", ha='center', va='center')
        plt.show(block=True)
        mt5.shutdown()
        return

    plt.close('all')
    try:
        fig = plt.figure(figsize=(10, 6), dpi=100)
    except Exception as e:
        logger.error(f"Failed to create figure: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Figure creation failed: {e}", ha='center', va='center')
        plt.show(block=True)
        mt5.shutdown()
        return
    
    try:
        ax = fig.add_subplot(111)
        axes = [ax]
    except Exception as e:
        logger.error(f"Failed to create subplot: {e}")
        fig.text(0.5, 0.5, f"Subplot creation failed: {e}", ha='center', va='center')
        plt.show(block=True)
        mt5.shutdown()
        return
    
    try:
        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.04])
    except Exception as e:
        logger.error(f"Failed to create slider axis: {e}")
        slider_ax = None
    
    max_candles = min_candles
    display_candles = min(DISPLAY_CANDLES, max_candles)
    if max_candles < 10:
        logger.warning(f"Very limited data: max_candles={max_candles}, proceeding without slider")
        slider = None
    else:
        try:
            slider = Slider(slider_ax, 'Candle Index', 0, max_candles - display_candles, 
                            valinit=max_candles - display_candles, valstep=1)
        except Exception as e:
            logger.error(f"Failed to create slider: {e}")
            slider = None
    
    def update(val=0):
        plt.clf()
        axes.clear()
        axes.append(fig.add_subplot(111))
        ax = axes[0]
        
        start_idx = int(val) if slider else max(0, max_candles - display_candles)
        end_idx = start_idx + display_candles
        symbol = symbols[0]
        timeframe = timeframes[0]
        
        if symbol not in data_dict_global or timeframe not in data_dict_global[symbol]:
            ax.text(0.5, 0.5, f"No data for {timeframe}", ha='center', va='center')
            ax.set_title(f"{timeframe} Chart ({symbol})")
            fig.canvas.draw()
            return
        
        df = data_dict_global[symbol][timeframe]['df']
        signals = data_dict_global[symbol][timeframe]['signals']
        window_df = df.iloc[start_idx:end_idx]
        if window_df.empty:
            ax.text(0.5, 0.5, f"No data in window for {timeframe}", ha='center', va='center')
            ax.set_title(f"{timeframe} Chart ({symbol})")
            fig.canvas.draw()
            return
        
        snr_window_df = df.iloc[:end_idx]
        snr_data = calculate_snr(snr_window_df, timeframe) if len(snr_window_df) >= 10 else {
            'support': [], 'resistance': [], 'mini_snr': [], 'vssnr': [], 'ssnr': []
        }
        
        logger.debug(f"Plotting {len(window_df)} candles for {symbol} on {timeframe}, SNR: {snr_data}")
        
        apds = []
        support_labeled = False
        resistance_labeled = False
        mini_snr_labeled = False
        for level in snr_data.get('support', []):
            if pd.notna(level) and np.isfinite(level):
                label = 'Support' if not support_labeled else ''
                apds.append(mpf.make_addplot([level] * len(window_df), color='green', linestyle='--',  width=1.5))
                support_labeled = True
        for level in snr_data.get('resistance', []):
            if pd.notna(level) and np.isfinite(level):
                label = 'Resistance' if not resistance_labeled else ''
                apds.append(mpf.make_addplot([level] * len(window_df), color='red', linestyle='--', width=1.5))
                resistance_labeled = True
        for level in snr_data.get('mini_snr', []):
            if pd.notna(level) and np.isfinite(level):
                label = 'Mini SNR' if not mini_snr_labeled else ''
                apds.append(mpf.make_addplot([level] * len(window_df), color='blue', linestyle=':', width=1.0))
                mini_snr_labeled = True
        
        try:
            mpf.plot(
                window_df[['open', 'high', 'low', 'close']],
                type='candle',
                volume=False,
                ax=ax,
                ylabel='Price',
                show_nontrading=False,
                addplot=apds if apds else None,
                title=f"{timeframe} Chart ({symbol})",
                figscale=1.0
            )
        except Exception as e:
            logger.error(f"mplfinance plotting failed for {symbol} on {timeframe}: {e}")
            ax.clear()
            ax.plot(window_df.index, window_df['close'])
            for level in snr_data.get('support', []):
                if pd.notna(level) and np.isfinite(level):
                    ax.axhline(y=level, color='green', linestyle='--')
                    support_labeled = True
            for level in snr_data.get('resistance', []):
                if pd.notna(level) and np.isfinite(level):
                    ax.axhline(y=level, color='red', linestyle='--')
                    resistance_labeled = True
            ax.set_title(f"{timeframe} Chart ({symbol})")
            ax.set_ylabel('Price')
        
        buy_labeled = False
        sell_labeled = False
        label_positions = {}
        for signal in signals:
            ts = signal['timestamp']
            if ts in window_df.index:
                idx = window_df.index.get_loc(ts)
                signal_type = signal['type']
                ss_name = signal['sure_shot']
                price = signal['price']
                if signal_type == 'Buy':
                    y_pos_signal = window_df.iloc[idx]['low'] * 0.9999
                    base_y_pos_label = window_df.iloc[idx]['low'] * 0.9996
                    color = 'green'
                    marker = '^'
                    label = 'Buy Signal' if not buy_labeled else None
                    buy_labeled = True
                    key = 'buy'
                    va = 'bottom'
                    offset_direction = -1
                else:
                    y_pos_signal = window_df.iloc[idx]['high'] * 1.0001
                    base_y_pos_label = window_df.iloc[idx]['high'] * 1.0004
                    color = 'red'
                    marker = 'v'
                    label = 'Sell Signal' if not sell_labeled else None
                    sell_labeled = True
                    key = 'sell'
                    va = 'top'
                    offset_direction = 1
                if idx not in label_positions:
                    label_positions[idx] = {'buy': 0, 'sell': 0}
                signal_count = label_positions[idx][key]
                price_level = window_df.iloc[idx]['close']
                offset_percentage = 0.02
                text_offset = (price_level * offset_percentage / 100) * signal_count * offset_direction
                y_pos_label = base_y_pos_label + text_offset
                label_positions[idx][key] += 1
                ax.scatter(idx, y_pos_signal, marker=marker, color=color, s=100, zorder=10)
                ax.text(idx, y_pos_label, ss_name, fontsize=8, color=color, ha='center', va=va, zorder=10)
        
        y_min = window_df['low'].min()
        y_max = window_df['high'].max()
        all_levels = []
        for key in ['support', 'resistance', 'mini_snr']:
            all_levels.extend([x for x in snr_data.get(key, []) if pd.notna(x) and np.isfinite(x)])
        for signal in signals:
            ts = signal['timestamp']
            if ts in window_df.index:
                all_levels.append(signal['price'])
        if all_levels:
            y_min = min(y_min, min(all_levels)) * 0.999
            y_max = max(y_max, max(all_levels)) * 1.001
        price_range = y_max - y_min
        y_min -= price_range * 0.05
        y_max += price_range * 0.05
        ax.set_ylim(y_min, y_max)
        if buy_labeled or sell_labeled or support_labeled or resistance_labeled or mini_snr_labeled:
            ax.legend()
        
        fig.canvas.draw()
    
    if slider:
        slider.on_changed(update)
    update(slider.val if slider else 0)
    try:
        plt.tight_layout(pad=1.0)
        plt.show(block=True)
    except Exception as e:
        logger.error(f"Failed to display plot: {e}")
    mt5.shutdown()

class Backtester:
    def __init__(self):
        self.trades = []
        self.total_trades = 0
        self.wins = 0
        self.total_profit_loss = 0.0

    def filter_trade(self, df, idx, signal):
        sure_shot = signal['sure_shot']
        if sure_shot in ['SS-2', 'SS-6', 'SS-8', 'SS-YA', 'SS-YS']:
            snr_data = calculate_snr(df.iloc[:idx + 1], signal['timeframe'])
            support_levels = snr_data.get('support', [])
            resistance_levels = snr_data.get('resistance', [])
            if idx >= len(df):
                return False
            candle = df.iloc[idx]
            body_low = min(candle['open'], candle['close'])
            body_high = max(candle['open'], candle['close'])
            if signal['type'] == 'Buy':
                for support in support_levels:
                    if abs(body_low - support) / support < 0.0001 and support > 0.0001:
                        logger.debug(f"Filtered Buy {sure_shot} at {signal['timestamp']}: body_low={body_low}, support={support}")
                        return False
            else:
                for resistance in resistance_levels:
                    if abs(body_high - resistance) / resistance < 0.0001 and resistance > 0.0001:
                        logger.debug(f"Filtered Sell {sure_shot} at {signal['timestamp']}: body_high={body_high}, resistance={resistance}")
                        return False
        return True

    def add_trade(self, signal, outcome, profit_loss, entry_price, exit_price, entry_idx, df, symbol, timeframe):
        self.trades.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'type': signal['type'],
            'sure_shot': signal['sure_shot'],
            'timestamp': signal['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'outcome': outcome,
            'profit_loss': profit_loss
        })
        self.total_trades += 1
        if outcome == 'Win':
            self.wins += 1
        self.total_profit_loss += profit_loss
        logger.debug(f"Added trade: {symbol} {timeframe} {signal['type']} {signal['sure_shot']} at {signal['timestamp']}, "
                     f"Outcome={outcome}, Profit/Loss={profit_loss}")

    def calculate_metrics(self):
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit_loss': 0.0,
                'avg_profit_per_trade': 0.0
            }
        win_rate = (self.wins / self.total_trades) * 100
        avg_profit_per_trade = self.total_profit_loss / self.total_trades
        return {
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_profit_loss': self.total_profit_loss,
            'avg_profit_per_trade': avg_profit_per_trade
        }

def backtest_sure_shots(symbols=SYMBOLS, timeframes=['M1'], limit=500, risk_per_trade=0.05, binary_payout=0.85):
    logger.info(f"Starting backtest for symbols {symbols} on timeframes {timeframes}")
    if not initialize_mt5():
        logger.error("MT5 initialization failed")
        return None
    
    backtester = Backtester()
    account_balance = get_account_balance()
    stake = account_balance * risk_per_trade
    results = []

    for symbol in symbols:
        for timeframe in timeframes:
            mt5_timeframe = TIMEFRAME_MAP.get(timeframe)
            if not mt5_timeframe:
                logger.error(f"Invalid timeframe: {timeframe}")
                continue
            
            df = fetch_ohlcv(symbol, mt5_timeframe, limit)
            if df is None or df.empty or len(df) < 20:
                logger.error(f"Insufficient data for {symbol} on {timeframe}: {len(df) if df is not None else 'None'} candles")
                continue
            
            for i in range(10, len(df) - 5):
                window_df = df.iloc[:i + 1]
                signal = generate_signal(window_df, timeframe, symbol)
                if signal and backtester.filter_trade(window_df, i, signal):
                    entry_price = signal['price']
                    entry_idx = i
                    signal_type = signal['type']
                    sure_shot = signal['sure_shot']
                    logger.debug(f"Testing signal {sure_shot} for {symbol} on {timeframe} at index {i}: {signal}")

                    future_candles = df.iloc[i + 1:i + 6]
                    if future_candles.empty:
                        logger.debug(f"No future candles for {symbol} on {timeframe} at index {i}")
                        continue
                    
                    outcome = None
                    exit_price = None
                    profit_loss = 0.0
                    
                    if signal_type == 'Buy':
                        for j, candle in future_candles.iterrows():
                            if candle['high'] >= entry_price * (1 + binary_payout * 0.01):
                                outcome = 'Win'
                                exit_price = candle['high']
                                profit_loss = stake * binary_payout
                                break
                            elif candle['low'] <= entry_price * 0.999:
                                outcome = 'Loss'
                                exit_price = candle['low']
                                profit_loss = -stake
                                break
                    else:
                        for j, candle in future_candles.iterrows():
                            if candle['low'] <= entry_price * (1 - binary_payout * 0.01):
                                outcome = 'Win'
                                exit_price = candle['low']
                                profit_loss = stake * binary_payout
                                break
                            elif candle['high'] >= entry_price * 1.001:
                                outcome = 'Loss'
                                exit_price = candle['high']
                                profit_loss = -stake
                                break
                    
                    if outcome:
                        backtester.add_trade(signal, outcome, profit_loss, entry_price, exit_price, entry_idx, df, symbol, timeframe)
                        results.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'type': signal_type,
                            'sure_shot': sure_shot,
                            'timestamp': signal['timestamp'],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'outcome': outcome,
                            'profit_loss': profit_loss
                        })
                        logger.debug(f"Trade result for {symbol} on {timeframe}: {outcome}, Profit/Loss={profit_loss}")
            
            logger.info(f"Completed backtest for {symbol} on {timeframe}: {backtester.total_trades} trades")

    mt5.shutdown()
    
    try:
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.to_csv('backtest_results.csv', index=False)
            logger.info("Saved backtest results to backtest_results.csv")
            
            metrics = backtester.calculate_metrics()
            summary_df = pd.DataFrame([{
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'total_profit_loss': metrics['total_profit_loss'],
                'avg_profit_per_trade': metrics['avg_profit_per_trade']
            }])
            summary_df.to_csv('backtest_summary.csv', index=False)
            logger.info("Saved backtest summary to backtest_summary.csv")
        else:
            logger.warning("No trades to save")
    except Exception as e:
        logger.error(f"Error saving backtest results: {e}")

    return backtester

async def main(backtest_mode=False):
    # backtest_mode = True  # Note: This overrides the argument, consider removing if you want to use the passed argument
    if backtest_mode:
        logger.info("Running in backtest mode")
        backtest_sure_shots()
        # Plot for all symbols and timeframes
        plot_backtest_signals(symbols=SYMBOLS, timeframes=PLOT_TIMEFRAMES)

    logger.info(f"Starting live bot for {', '.join(SYMBOLS)} on {', '.join(PLOT_TIMEFRAMES)}")
    last_processed_times = {symbol: {tf: None for tf in PLOT_TIMEFRAMES} for symbol in SYMBOLS}
    signal_history = {symbol: {tf: [] for tf in PLOT_TIMEFRAMES} for symbol in SYMBOLS}
    
    if not initialize_mt5():
        raise Exception("MT5 initialization failed")
    
    plt.ion()
    first_run = True
    
    while True:
        try:
            start_time = time.time()
            data_dict = {symbol: {tf: {'df': None, 'signals': [], 'snr_data': None} for tf in PLOT_TIMEFRAMES} for symbol in SYMBOLS}
            for symbol in SYMBOLS:
                for timeframe in PLOT_TIMEFRAMES:
                    fetch_start = time.time()
                    df = fetch_ohlcv(symbol, TIMEFRAME_MAP[timeframe], LIMIT)
                    fetch_time = time.time() - fetch_start
                    if df is None or df.empty:
                        logger.warning(f"No data for {symbol} on {timeframe}")
                        continue
                    logger.debug(f"Data for {symbol} on {timeframe}: {len(df)} candles")
                    snr_data = calculate_snr(df, timeframe)
                    current_last_time = df.index[-1]
                    if last_processed_times[symbol][timeframe] is None or current_last_time > last_processed_times[symbol][timeframe]:
                        market_type = get_market_type(df)
                        signal = generate_signal(df, timeframe, symbol)
                        if signal:
                            message = (
                                f"🚨 {signal['type']} Signal for {signal['symbol']} ({signal['timeframe']}) using {signal['sure_shot']}\n"
                                f"📅 Time: {signal['timestamp']}\n"
                                f"💰 Entry Price: {signal['price']:.5f}"
                            )
                            await send_telegram_message(message)
                            signal_history[symbol][timeframe].append(signal)
                            signal_history[symbol][timeframe] = signal_history[symbol][timeframe][-10:]
                        data_dict[symbol][timeframe] = {
                            'df': df,
                            'signals': signal_history[symbol][timeframe],
                            'snr_data': snr_data
                        }
                        last_processed_times[symbol][timeframe] = current_last_time
                    else:
                        data_dict[symbol][timeframe] = {
                            'df': df,
                            'signals': signal_history[symbol][timeframe],
                            'snr_data': snr_data
                        }
            
            plot_start = time.time()
            plot_signals_multi(data_dict, initial=first_run)
            plot_time = time.time() - plot_start
            if first_run:
                first_run = False
            
            loop_time = time.time() - start_time
            await asyncio.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sure Shot Signals Trading Bot")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest mode")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(backtest_mode=args.backtest))
    except KeyboardInterrupt:
        logger.info("Shutting down bot")
        mt5.shutdown()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
