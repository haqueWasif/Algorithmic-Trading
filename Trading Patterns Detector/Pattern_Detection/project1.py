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
from collections import Counter
import mplfinance as mpf
from matplotlib.widgets import Slider, Button
import threading
import time
import argparse
import os

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
LIMIT = 500
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

# Risk management parameters
RISK_PER_TRADE = 0.05  # 5% of account balance
BINARY_PAYOUT = 0.85  # 85% payout for winning binary options trade
DEFAULT_ACCOUNT_BALANCE = 2000

# Initialize MetaTrader5
def initialize_mt5():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, timeout=30000):
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    for symbol in SYMBOLS:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Symbol {symbol} not available: {mt5.last_error()}")
            mt5.shutdown()
            return False
    logger.info("MT5 initialized successfully")
    return True

# Fetch account balance
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

# Global plotting variables
fig = None
axes = None
slider = None
data_dict_global = None
current_symbol = SYMBOLS[0]
buttons = []

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

def fetch_ohlcv(symbol, timeframe, limit):
    try:
        start_time = time.time()
        if not mt5.terminal_info():
            logger.warning("MT5 connection lost, attempting to reconnect")
            if not initialize_mt5():
                return None
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, limit)
        if rates is None or len(rates) == 0:
            logger.error(f"No data fetched for {symbol} on {timeframe}: {mt5.last_error()}")
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
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol} on {timeframe}: {e}")
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

def is_hammer(candle, abnormal=False):
    """Check if a candle is a hammer, optionally an abnormal one with a long wick."""
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    if body == 0:
        return upper_wick > 0 or lower_wick > 0
    if abnormal:
        return max(upper_wick, lower_wick) >= 3 * body  # Long wick for abnormal hammer
    return max(upper_wick, lower_wick) >= 1.5 * body

def is_doji(candle):
    """Check if a candle is a Doji (small body relative to range)."""
    body = abs(candle['close'] - candle['open'])
    candle_range = candle['high'] - candle['low']
    return body <= 0.1 * candle_range if candle_range > 0 else False

def is_bald_snr_candle(candle, avg_body_size):
    """Check if a candle qualifies as a Bald SNR (strong rejection)."""
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    # Large wick or body indicates strong rejection
    return max(upper_wick, lower_wick, body) >= 2 * avg_body_size

def get_market_type(df):
    if len(df) < 50:
        return 'unknown'
    greens_50 = sum(1 for i in range(-50, 0) if df['close'].iloc[i] > df['open'].iloc[i])
    reds_50 = sum(1 for i in range(-50, 0) if df['close'].iloc[i] < df['open'].iloc[i])
    if greens_50 > 30:
        overall_trend = 'up'
    elif reds_50 > 30:
        overall_trend = 'down'
    else:
        overall_trend = 'no_clear_trend'

    greens_10 = sum(1 for i in range(-10, 0) if df['close'].iloc[i] > df['open'].iloc[i])
    reds_10 = sum(1 for i in range(-10, 0) if df['close'].iloc[i] < df['open'].iloc[i])
    dojis_10 = sum(1 for i in range(-10, 0) if df['high'].iloc[i] != df['low'].iloc[i] and
                   abs(df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i]) < 0.1)

    if overall_trend == 'no_clear_trend':
        return 'stack' if dojis_10 > 4 else 'ranging'
    if overall_trend == 'up':
        return 'trending_up'
    return 'trending_down'

def detect_pivots(df, pivot_window):
    """
    Detect pivot highs and lows in the DataFrame using a specified window size.
    Returns lists of (index, price) tuples for highs and lows.
    """
    highs = []
    lows = []
    for i in range(pivot_window, len(df) - pivot_window):
        if all(df['high'].iloc[i] >= df['high'].iloc[i - pivot_window:i + pivot_window + 1]):
            highs.append((i, df['high'].iloc[i]))
        if all(df['low'].iloc[i] <= df['low'].iloc[i - pivot_window:i + pivot_window + 1]):
            lows.append((i, df['low'].iloc[i]))
    return highs, lows

def calculate_snr(df, timeframe):
    """
    Calculate VSSNR and SSNR levels based on 'The Golden 30-S', including Doji body, abnormal hammer wicks,
    Bald SNR, and now trend lines for Three Rivers in a channeling market.
    """
    snr_data = {
        'support': [], 'resistance': [], 'mini_snr': [], 'vssnr': [], 'ssnr': [],
        'vssnr_upper': [], 'vssnr_lower': [], 'ssnr_upper': [], 'ssnr_lower': [],
        'snr_upper': [], 'snr_lower': []
    }
    if df.empty or len(df) < 20:
        logger.warning(f"Insufficient data for SNR calculation on {timeframe}")
        return snr_data

    # --- Existing Logic (Unchanged) ---
    long_lookback = min(200, len(df))
    short_lookback = min(50, len(df))
    pivot_window = 5
    cluster_threshold = 0.001
    touch_threshold = 0.0001
    max_breach_ratio = 0.3
    min_touches = 3
    default_atr_period = 14
    avg_body_size = df[-short_lookback:]['close'].sub(df['open']).abs().mean()

    # Existing helper functions (unchanged)
    def calculate_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.iloc[-1] if not atr.empty else 0

    def cluster_levels(levels, price_threshold):
        if not levels:
            return []
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= price_threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        return clustered

    def count_touches(df, level, is_support):
        touches = 0
        for i in range(len(df)):
            if is_support:
                if abs(df['low'].iloc[i] - level) / level < touch_threshold:
                    touches += 1
            else:
                if abs(df['high'].iloc[i] - level) / level < touch_threshold:
                    touches += 1
        return touches

    def count_breaches(df, level, is_support):
        breaches = 0
        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            if is_support:
                if prev_close >= level and curr_close < level and curr_low < level:
                    breaches += 1
            else:
                if prev_close <= level and curr_close > level and curr_high > level:
                    breaches += 1
        return breaches

    def get_market_type(df, atr_period):
        recent_prices = df['close'][-20:]
        price_range = recent_prices.max() - recent_prices.min()
        atr = calculate_atr(df[-20:], period=atr_period)
        if price_range < atr * 1.5:
            return 'stack'
        elif price_range < atr * 3:
            return 'ranging'
        else:
            return 'trendy'

    def is_hammer(candle, abnormal=False):
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        if body == 0:
            return False
        if abnormal:
            return (upper_wick >= 2 * body or lower_wick >= 2 * body) and body > avg_body_size * 1.5
        return upper_wick >= 2 * body or lower_wick >= 2 * body

    def is_doji(candle):
        body = abs(candle['close'] - candle['open'])
        return body <= (candle['high'] - candle['low']) * 0.1

    def is_bald_snr_candle(candle, avg_body_size):
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        return body >= avg_body_size and upper_wick <= body * 0.1 and lower_wick <= body * 0.1

    market_type = get_market_type(df[-short_lookback:], atr_period=default_atr_period)
    timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}.get(timeframe.lower(), 60)
    atr_period_map = {
        '1m': {'stack': 7, 'ranging': 10, 'trendy': 14},
        '5m': {'stack': 10, 'ranging': 12, 'trendy': 16},
        '15m': {'stack': 10, 'ranging': 12, 'trendy': 16},
        '30m': {'stack': 12, 'ranging': 14, 'trendy': 18},
        '1h': {'stack': 14, 'ranging': 16, 'trendy': 20},
        '4h': {'stack': 16, 'ranging': 18, 'trendy': 22},
        '1d': {'stack': 18, 'ranging': 20, 'trendy': 25}
    }
    atr_period = atr_period_map.get(timeframe.lower(), {'stack': 14, 'ranging': 14, 'trendy': 14})[market_type]
    atr = calculate_atr(df[-short_lookback:], period=atr_period)
    min_range_threshold = atr * 2

    # Existing VSSNR and SSNR calculation (unchanged)
    vssnr_support = []
    vssnr_resistance = []
    recent_df = df.iloc[-long_lookback:] if len(df) >= long_lookback else df

    if market_type == 'ranging':
        highs = []
        lows = []
        for i in range(pivot_window, len(recent_df) - pivot_window):
            is_high = all(recent_df['high'].iloc[i] >= recent_df['high'].iloc[i - pivot_window:i + pivot_window + 1])
            is_low = all(recent_df['low'].iloc[i] <= recent_df['low'].iloc[i - pivot_window:i + pivot_window + 1])
            if is_high:
                highs.append(recent_df['high'].iloc[i])
            if is_low:
                lows.append(recent_df['low'].iloc[i])
        highs = cluster_levels(highs, cluster_threshold)
        lows = cluster_levels(lows, cluster_threshold)
        vssnr_resistance.extend(highs)
        vssnr_support.extend(lows)

    if market_type == 'stack':
        for i in range(len(recent_df) - 10, len(recent_df)):
            candle = recent_df.iloc[i]
            if is_hammer(candle, abnormal=True):
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                if upper_wick >= lower_wick:
                    vssnr_resistance.append(candle['high'])
                if lower_wick >= upper_wick:
                    vssnr_support.append(candle['low'])

    highs = []
    lows = []
    doji_levels = []
    for i in range(pivot_window, len(recent_df) - pivot_window):
        is_high = all(recent_df['high'].iloc[i] >= recent_df['high'].iloc[i - pivot_window:i + pivot_window + 1])
        is_low = all(recent_df['low'].iloc[i] <= recent_df['low'].iloc[i - pivot_window:i + pivot_window + 1])
        if is_high:
            highs.append(recent_df['high'].iloc[i])
        if is_low:
            lows.append(recent_df['low'].iloc[i])
        candle = recent_df.iloc[i]
        if is_doji(candle):
            doji_body = (candle['open'] + candle['close']) / 2
            if doji_body < recent_df['close'].iloc[-1]:
                doji_levels.append((doji_body, True))
            else:
                doji_levels.append((doji_body, False))
    highs = cluster_levels(highs, cluster_threshold)
    lows = cluster_levels(lows, cluster_threshold)
    vssnr_resistance.extend(highs)
    vssnr_support.extend(lows)
    for level, is_support in doji_levels:
        if is_support:
            vssnr_support.append(level)
        else:
            vssnr_resistance.append(level)

    filtered_vssnr_support = []
    filtered_vssnr_resistance = []
    for level in vssnr_support:
        touches = count_touches(recent_df, level, True)
        breaches = count_breaches(recent_df, level, True)
        breach_ratio = breaches / touches if touches > 0 else 0
        if touches >= min_touches and breach_ratio <= max_breach_ratio:
            filtered_vssnr_support.append((level, touches))
    for level in vssnr_resistance:
        touches = count_touches(recent_df, level, False)
        breaches = count_breaches(recent_df, level, False)
        breach_ratio = breaches / touches if touches > 0 else 0
        if touches >= min_touches and breach_ratio <= max_breach_ratio:
            filtered_vssnr_resistance.append((level, touches))

    vssnr_support = [level for level, touches in sorted(filtered_vssnr_support, key=lambda x: x[1], reverse=True)[:1]]
    vssnr_resistance = [level for level, touches in sorted(filtered_vssnr_resistance, key=lambda x: x[1], reverse=True)[:1]]

    ssnr_support = []
    ssnr_resistance = []
    short_df = df.iloc[-short_lookback:] if len(df) >= short_lookback else df
    highs = []
    lows = []
    for i in range(pivot_window, len(short_df) - pivot_window):
        is_high = all(short_df['high'].iloc[i] >= short_df['high'].iloc[i - pivot_window:i + pivot_window + 1])
        is_low = all(short_df['low'].iloc[i] <= short_df['low'].iloc[i - pivot_window:i + pivot_window + 1])
        candle = short_df.iloc[i]
        if is_high:
            if is_bald_snr_candle(candle, avg_body_size):
                highs.append(candle['high'])
            else:
                highs.append(candle['high'])
        if is_low:
            if is_bald_snr_candle(candle, avg_body_size):
                lows.append(candle['low'])
            else:
                lows.append(candle['low'])
    highs = cluster_levels(highs, cluster_threshold)
    lows = cluster_levels(lows, cluster_threshold)
    ssnr_resistance.extend(highs)
    ssnr_support.extend(lows)

    filtered_ssnr_support = []
    filtered_ssnr_resistance = []
    for level in ssnr_support:
        touches = count_touches(short_df, level, True)
        breaches = count_breaches(short_df, level, True)
        breach_ratio = breaches / touches if touches > 0 else 0
        if touches >= min_touches and breach_ratio <= max_breach_ratio:
            filtered_ssnr_support.append((level, touches))
    for level in ssnr_resistance:
        touches = count_touches(short_df, level, False)
        breaches = count_breaches(short_df, level, False)
        breach_ratio = breaches / touches if touches > 0 else 0
        if touches >= min_touches and breach_ratio <= max_breach_ratio:
            filtered_ssnr_resistance.append((level, touches))

    ssnr_support = [level for level, touches in sorted(filtered_ssnr_support, key=lambda x: x[1], reverse=True)[:1]]
    ssnr_resistance = [level for level, touches in sorted(filtered_ssnr_resistance, key=lambda x: x[1], reverse=True)[:1]]

    final_support = []
    final_resistance = []
    if vssnr_support and vssnr_resistance:
        support_level = vssnr_support[0]
        resistance_level = vssnr_resistance[0]
        distance = abs(resistance_level - support_level)
        if distance < min_range_threshold:
            support_touches = count_touches(recent_df, support_level, True)
            resistance_touches = count_touches(recent_df, resistance_level, False)
            if support_touches > resistance_touches:
                final_support = [support_level]
            else:
                final_resistance = [resistance_level]
        else:
            final_support = vssnr_support
            final_resistance = vssnr_resistance
    else:
        final_support = vssnr_support
        final_resistance = vssnr_resistance

    if ssnr_support and ssnr_resistance and not (final_support and final_resistance):
        support_level = ssnr_support[0]
        resistance_level = ssnr_resistance[0]
        distance = abs(resistance_level - support_level)
        if distance < min_range_threshold:
            support_touches = count_touches(short_df, support_level, True)
            resistance_touches = count_touches(short_df, resistance_level, False)
            if support_touches > resistance_touches:
                final_support = [support_level]
            else:
                final_resistance = [resistance_level]
        else:
            if not final_support:
                final_support = ssnr_support
            if not final_resistance:
                final_resistance = ssnr_resistance

    snr_data['vssnr'] = vssnr_support + vssnr_resistance
    snr_data['ssnr'] = ssnr_support + ssnr_resistance
    snr_data['support'] = final_support
    snr_data['resistance'] = final_resistance
    snr_data['mini_snr'] = []
    # --- End of Existing Logic ---

    # --- New Logic for Three Rivers Trend Lines ---
    market_type = get_market_type(df, atr_period=default_atr_period)  # Use existing function
    if market_type == 'ranging':
        # Define window sizes for Three Rivers
        pivot_window_vssnr = 10  # Big River
        pivot_window_ssnr = 5   # Medium River
        pivot_window_snr = 2    # Small River
        min_distance = 10       # Minimum candles between pivots

        # VSSNR Trend Lines
        highs_vssnr, lows_vssnr = detect_pivots(df, pivot_window_vssnr)
        sorted_highs_vssnr = sorted(highs_vssnr, key=lambda x: x[0], reverse=True)
        if len(sorted_highs_vssnr) >= 2 and sorted_highs_vssnr[0][0] - sorted_highs_vssnr[1][0] >= min_distance:
            snr_data['vssnr_upper'] = sorted_highs_vssnr[:2]  # [(idx1, price1), (idx2, price2)]
        sorted_lows_vssnr = sorted(lows_vssnr, key=lambda x: x[0], reverse=True)
        if len(sorted_lows_vssnr) >= 2 and sorted_lows_vssnr[0][0] - sorted_lows_vssnr[1][0] >= min_distance:
            snr_data['vssnr_lower'] = sorted_lows_vssnr[:2]

        # SSNR Trend Lines
        highs_ssnr, lows_ssnr = detect_pivots(df, pivot_window_ssnr)
        sorted_highs_ssnr = sorted(highs_ssnr, key=lambda x: x[0], reverse=True)
        if len(sorted_highs_ssnr) >= 2 and sorted_highs_ssnr[0][0] - sorted_highs_ssnr[1][0] >= min_distance:
            snr_data['ssnr_upper'] = sorted_highs_ssnr[:2]
        sorted_lows_ssnr = sorted(lows_ssnr, key=lambda x: x[0], reverse=True)
        if len(sorted_lows_ssnr) >= 2 and sorted_lows_ssnr[0][0] - sorted_lows_ssnr[1][0] >= min_distance:
            snr_data['ssnr_lower'] = sorted_lows_ssnr[:2]

        # SNR Trend Lines
        highs_snr, lows_snr = detect_pivots(df, pivot_window_snr)
        sorted_highs_snr = sorted(highs_snr, key=lambda x: x[0], reverse=True)
        if len(sorted_highs_snr) >= 2 and sorted_highs_snr[0][0] - sorted_highs_snr[1][0] >= min_distance:
            snr_data['snr_upper'] = sorted_highs_snr[:2]
        sorted_lows_snr = sorted(lows_snr, key=lambda x: x[0], reverse=True)
        if len(sorted_lows_snr) >= 2 and sorted_lows_snr[0][0] - sorted_lows_snr[1][0] >= min_distance:
            snr_data['snr_lower'] = sorted_lows_snr[:2]
    else:
        # No trend lines for non-ranging markets
        snr_data['vssnr_upper'] = []
        snr_data['vssnr_lower'] = []
        snr_data['ssnr_upper'] = []
        snr_data['ssnr_lower'] = []
        snr_data['snr_upper'] = []
        snr_data['snr_lower'] = []

    # Ensure all levels are valid
    for key in snr_data:
        if key in ['vssnr_upper', 'vssnr_lower', 'ssnr_upper', 'ssnr_lower', 'snr_upper', 'snr_lower']:
            # For trend lines, keep as list of tuples
            snr_data[key] = snr_data[key] if snr_data[key] else []
        else:
            # For horizontal levels, filter as before
            snr_data[key] = [x for x in snr_data[key] if pd.notna(x) and np.isfinite(x)]

    logger.debug(f"SNR for {timeframe}: {snr_data}")
    return snr_data


def find_touch_snr(df, last_idx, resistance, support, direction):
    last_close = df.iloc[last_idx]['close']
    last_open = df.iloc[last_idx]['open']
    max_lookback = 200
    lookback_start = -max_lookback

    # Initialize variables
    price_condition_met = True
    snr_touched = False
    snr_touch_idx = last_idx  # Default to last_idx if no touch found

    
    # Check candles backward from last_idx-1 until SNR touch or lookback limit

    i = last_idx - 1
    while(True):
        candle = df.iloc[i]
        if direction == 'buy':
            last_min = min(last_close, last_open)
            # Check price condition: open and close must be above last_min
            if candle['close'] <= last_min:
                price_condition_met = False
                logger.debug(f"Buy signal at idx {last_idx}: Price condition failed at idx {i}, "
                             f"open={candle['open']}, close={candle['close']}, last_min={last_min}")
                break
            # Check for SNR touch (sellers loop)
            if candle['high'] >= resistance:
                snr_touched = True
                snr_touch_idx = i
                break
        else:  # sell
            last_max = max(last_close, last_open)
            # Check price condition: open and close must be below last_max
            if candle['close'] >= last_max:
                price_condition_met = False
                logger.debug(f"Sell signal at idx {last_idx}: Price condition failed at idx {i}, "
                             f"open={candle['open']}, close={candle['close']}, last_max={last_max}")
                break
            # Check for SNR touch (buyers loop)
            if candle['low'] <= support:
                snr_touched = True
                snr_touch_idx = i
                break

        i-=1

    # If no SNR touch was found, check the entire lookback for SNR touch
    if not snr_touched:
        prior_candles = df.iloc[lookback_start:last_idx]
        snr_touched = any(prior_candles['high'] >= resistance) if direction == 'buy' else \
                      any(prior_candles['low'] <= support)
        snr_touch_idx = lookback_start if snr_touched else last_idx
        logger.debug(f"{direction.capitalize()} signal at idx {last_idx}: No SNR touch in loop, "
                     f"full lookback snr_touched={snr_touched}, snr_touch_idx={snr_touch_idx}")

    # Verify price condition for all candles from last_idx-1 to snr_touch_idx
    if snr_touched and price_condition_met:
        for i in range(last_idx - 1, snr_touch_idx - 1, -1):
            candle = df.iloc[i]
            if direction == 'buy':
                if candle['open'] <= last_min or candle['close'] <= last_min:
                    price_condition_met = False
                    logger.debug(f"Buy signal at idx {last_idx}: Price condition failed at idx {i} "
                                 f"after SNR touch, open={candle['open']}, close={candle['close']}, last_min={last_min}")
                    break
            else:  # sell
                if candle['open'] >= last_max or candle['close'] >= last_max:
                    price_condition_met = False
                    logger.debug(f"Sell signal at idx {last_idx}: Price condition failed at idx {i} "
                                 f"after SNR touch, open={candle['open']}, close={candle['close']}, last_max={last_max}")
                    break

    logger.debug(f"{direction.capitalize()} signal at idx {last_idx}: "
                 f"price_condition_met={price_condition_met}, snr_touched={snr_touched}, "
                 f"snr_touch_idx={snr_touch_idx}")

    return price_condition_met and snr_touched

def detect_ss1(df):
    market_type = get_market_type(df)
    if market_type not in ['trending_up', 'trending_down'] or len(df) < 2:
        return None
    
    third_last = df.iloc[-4]
    second_last = df.iloc[-3]
    last = df.iloc[-2]
    
    third_size = abs(third_last['close'] - third_last['open'])
    second_size = abs(second_last['close'] - second_last['open'])
    last_size = abs(last['close'] - last['open'])

    if market_type == 'trending_up':
        if (second_last['close'] < second_last['open']) and (last['close'] > last['open']) and third_last['close'] > third_last['open'] and (second_size < third_size and second_size < last_size) :
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

    if (second_last['close'] < second_last['open']) and (second_last['low'] <= support or last['low'] <= support) and (last['close'] > last['open']) and (lower_wick_last == 0) and (last['open'] >= support):
        touched_resistance = find_touch_snr(df, -2, resistance, support, 'buy')
        if touched_resistance:
            logger.debug(f"SS-2 Buy: Candle at support {support}, came from resistance {resistance}")
            return {
                'type': 'Buy',
                'timestamp': last.name,
                'price': last['close'],
                'sure_shot': 'SS-2'
            }
    
    elif (second_last['close'] > second_last['open']) and (second_last['high'] >= resistance or last['high'] >= resistance) and (last['close'] < last['open']) and (upper_wick_last == 0) and (last['open'] <= resistance):
        touched_support = find_touch_snr(df, -2, resistance, support, 'sell')
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

    if market_type == 'trending_up' :
        if third_last['close'] < third_last['open'] and third_last['close'] <= support and third_last['high'] >= support:
            if second_last['close'] > second_last['open'] and second_last['close'] > support:
                if last['close'] < last['open'] and last['low'] <= support and last['close'] == third_last['close']:
                    return {
                        'type': 'Buy',
                        'timestamp': last.name,
                        'price': last['close'],
                        'sure_shot': 'SS-8'
                    }
    elif market_type == 'trending_down' :
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
            touched_resistance = find_touch_snr(df, -2, resistance, support, 'buy')
            if touched_resistance:
                logger.debug(f"SS-YA Buy: Hammer at support {support}, came from resistance {resistance}")
                return {
                    'type': 'Buy',
                    'timestamp': last.name,
                    'price': last['close'],
                    'sure_shot': 'SS-YA'
                }
        elif last['high'] >= resistance and last['close'] < resistance and lower_wick_last <= upper_wick_last  and upper_wick_second_last <= lower_wick_second_last and second_last['close'] < resistance:
            touched_support = find_touch_snr(df, -2, resistance, support, 'sell')
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
            touched_resistance = find_touch_snr(df, -2, resistance, support, 'buy')
            if touched_resistance:
                logger.debug(f"SS-YS Buy: Hammer at support {support}, came from resistance {resistance}")
                return {
                    'type': 'Buy',
                    'timestamp': last.name,
                    'price': last['close'],
                    'sure_shot': 'SS-YS'
                }
    
        elif last['high'] >= resistance and last['close'] < resistance and second_last['high'] >= resistance and second_last['close'] < resistance:
            touched_support = find_touch_snr(df, -2, resistance, support, 'sell')
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

async def send_telegram_message(message):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logger.debug(f"Sent Telegram message")
    except TelegramError as e:
        logger.error(f"Failed to send Telegram message: {e}")

def plot_backtest_signals(symbols=SYMBOLS, timeframes=PLOT_TIMEFRAMES, limit=LIMIT, display_candles=50):
    """Plot backtest signals for multiple symbols and timeframes with a slider, including dynamic SNR lines."""
    global fig, axes, slider, data_dict_global, current_symbol
    logger.info(f"Plotting backtest signals for symbols {symbols} on timeframes {timeframes}")
    
    # Initialize MT5
    if not initialize_mt5():
        logger.error("MT5 initialization failed")
        plt.text(0.5, 0.5, "MT5 initialization failed", ha='center', va='center')
        plt.show()
        return

    # Load backtest results
    try:
        results_df = pd.read_csv('backtest_results.csv')
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'], utc=True)
        logger.info(f"Loaded {len(results_df)} signals from backtest_results.csv")
    except FileNotFoundError:
        logger.error("backtest_results.csv not found")
        plt.text(0.5, 0.5, "backtest_results.csv not found", ha='center', va='center')
        plt.show()
        mt5.shutdown()
        return
    except Exception as e:
        logger.error(f"Error loading backtest_results.csv: {e}")
        plt.text(0.5, 0.5, f"Error loading CSV: {e}", ha='center', va='center')
        plt.show()
        mt5.shutdown()
        return

    # Prepare data for each symbol-timeframe combination
    data_dict_global = {}
    min_candles = float('inf')
    for symbol in symbols:
        data_dict_global[symbol] = {}
        for timeframe in timeframes:
            # Fetch OHLCV data
            df = fetch_ohlcv(symbol, TIMEFRAME_MAP[timeframe], limit)
            if df is None or df.empty or len(df) < display_candles:
                logger.error(f"Insufficient data for {symbol} on {timeframe}: {len(df) if df is not None else 'None'} candles")
                continue
            
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error(f"Invalid index type for {symbol} on {timeframe}: {type(df.index)}")
                continue
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol} on {timeframe}: {df.columns}")
                continue
            
            # Filter signals for this symbol and timeframe
            signals_df = results_df[(results_df['symbol'] == symbol) & (results_df['timeframe'] == timeframe)]
            signals = []
            for _, row in signals_df.iterrows():
                if row['timestamp'] in df.index:  # Ensure signal timestamp is in DataFrame index
                    signals.append({
                        'timestamp': row['timestamp'],
                        'type': row['type'],
                        'sure_shot': row['sure_shot'],
                        'price': row['entry_price']
                    })
            logger.debug(f"Prepared {len(signals)} valid signals for {symbol} on {timeframe}")
            
            data_dict_global[symbol][timeframe] = {
                'df': df,
                'signals': signals,
                'snr_data': None  # Will be calculated dynamically in update
            }
            min_candles = min(min_candles, len(df))
    
    if not data_dict_global:
        logger.error("No valid data available for plotting")
        plt.text(0.5, 0.5, "No valid data available", ha='center', va='center')
        plt.show()
        mt5.shutdown()
        return

    # Initialize plot
    plt.close('all')
    fig = plt.figure(figsize=(12, 8), dpi=150)
    
    # Create subplot grid: one subplot per symbol-timeframe combination
    n_combinations = sum(len(data_dict_global[symbol]) for symbol in data_dict_global)
    if n_combinations == 0:
        logger.error("No symbol-timeframe combinations to plot")
        fig.text(0.5, 0.5, "No symbol-timeframe combinations", ha='center', va='center')
        plt.show()
        mt5.shutdown()
        return
    
    # Determine grid layout
    rows = min(2, (n_combinations + 1) // 2)
    cols = min(2, (n_combinations + 1) // 2)
    if rows * cols < n_combinations:
        cols = (n_combinations + rows - 1) // rows
    gs = fig.add_gridspec(rows, cols, top=0.92, bottom=0.18, hspace=0.4, wspace=0.3)
    
    # Create subplots
    axes = []
    idx = 0
    for symbol in symbols:
        for timeframe in timeframes:
            if symbol in data_dict_global and timeframe in data_dict_global[symbol]:
                ax = fig.add_subplot(gs[idx // cols, idx % cols])
                axes.append((symbol, timeframe, ax))
                idx += 1
    
    if not axes:
        logger.error("No valid subplots created")
        fig.text(0.5, 0.5, "No valid data to plot", ha='center', va='center')
        plt.show()
        mt5.shutdown()
        return
    
    slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.04])
    
    # Set up slider
    max_candles = min_candles
    if max_candles <= display_candles:
        logger.warning("Not enough data for sliding window: max_candles={max_candles}")
        for _, timeframe, ax in axes:
            ax.text(0.5, 0.5, f"Insufficient data for {timeframe}", ha='center', va='center')
            ax.set_title(f"{timeframe} Chart")
        plt.show()
        mt5.shutdown()
        return
    
    slider = Slider(slider_ax, 'Candle Index', 0, max_candles - display_candles, 
                    valinit=max_candles - display_candles, valstep=1)
    
    def update(val):
        """Update the plot based on slider position, including dynamic SNR lines, candlesticks, and signals."""
        global axes, data_dict_global
        if data_dict_global is None:
            logger.warning("No data in data_dict_global")
            return

        start_idx = int(val)
        end_idx = start_idx + display_candles
        for symbol, timeframe, ax in axes:
            ax.clear()

            if symbol not in data_dict_global or timeframe not in data_dict_global[symbol]:
                logger.warning(f"No data for {symbol} on {timeframe}")
                ax.text(0.5, 0.5, f"No data for {timeframe}", ha='center', va='center')
                ax.set_title(f"{timeframe} Chart ({symbol})")
                continue
            
            df = data_dict_global[symbol][timeframe]['df']
            signals = data_dict_global[symbol][timeframe]['signals']

            if df is None or df.empty:
                logger.warning(f"Empty DataFrame for {symbol} on {timeframe}")
                ax.text(0.5, 0.5, f"No data for {timeframe}", ha='center', va='center')
                ax.set_title(f"{timeframe} Chart ({symbol})")
                continue
            
            # Slice window for plotting
            window_df = df.iloc[start_idx:end_idx]
            if window_df.empty:
                logger.warning(f"Empty window for {symbol} on {timeframe} at start_idx={start_idx}, end_idx={end_idx}")
                ax.text(0.5, 0.5, f"No data in window for {timeframe}", ha='center', va='center')
                ax.set_title(f"{timeframe} Chart ({symbol})")
                continue
            
            # Validate window_df
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in window_df.columns for col in required_cols):
                logger.error(f"Missing columns in window_df for {symbol} on {timeframe}: {window_df.columns}")
                ax.text(0.5, 0.5, f"Invalid data for {timeframe}", ha='center', va='center')
                ax.set_title(f"{timeframe} Chart ({symbol})")
                continue
            
            if not isinstance(window_df.index, pd.DatetimeIndex):
                logger.error(f"Invalid index type for {symbol} on {timeframe}: {type(window_df.index)}")
                ax.text(0.5, 0.5, f"Invalid index for {timeframe}", ha='center', va='center')
                ax.set_title(f"{timeframe} Chart ({symbol})")
                continue
            
            # Calculate SNR for data up to end_idx
            snr_window_df = df.iloc[:end_idx]
            if len(snr_window_df) < 20:
                logger.warning(f"Insufficient data for SNR calculation for {symbol} on {timeframe}: {len(snr_window_df)} candles")
                snr_data = {'support': [], 'resistance': [], 'mini_snr': [], 'vssnr': [], 'ssnr': []}
            else:
                try:
                    snr_data = calculate_snr(snr_window_df, timeframe)
                    logger.debug(f"Recalculated SNR for {symbol} on {timeframe} at end_idx={end_idx}: {snr_data}")
                except Exception as e:
                    logger.error(f"Error calculating SNR for {symbol} on {timeframe}: {e}")
                    snr_data = {'support': [], 'resistance': [], 'mini_snr': [], 'vssnr': [], 'ssnr': []}
            
            # Prepare addplots for SNR lines
            apds = []
            support_labeled = False
            resistance_labeled = False
            mini_snr_labeled = False
            vssnr_labeled = False
            ssnr_labeled = False
            
            if snr_data:
                for level in snr_data.get('support', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'Support' if not support_labeled else ''
                        apds.append(mpf.make_addplot([level] * len(window_df), color='green', linestyle='--', label=label))
                        logger.debug(f"Added support for {symbol} on {timeframe}: {level}")
                        support_labeled = True
                for level in snr_data.get('resistance', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'Resistance' if not resistance_labeled else ''
                        apds.append(mpf.make_addplot([level] * len(window_df), color='red', linestyle='--', label=label))
                        logger.debug(f"Added resistance for {symbol} on {timeframe}: {level}")
                        resistance_labeled = True
                for level in snr_data.get('mini_snr', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'Mini SNR' if not mini_snr_labeled else ''
                        apds.append(mpf.make_addplot([level] * len(window_df), color='blue', linestyle=':', label=label))
                        logger.debug(f"Added mini_snr for {symbol} on {timeframe}: {level}")
                        mini_snr_labeled = True
                for level in snr_data.get('vssnr', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'VSSNR' if not vssnr_labeled else ''
                        apds.append(mpf.make_addplot([level] * len(window_df), color='purple', linestyle='-', label=label))
                        logger.debug(f"Added vssnr for {symbol} on {timeframe}: {level}")
                        vssnr_labeled = True
                for level in snr_data.get('ssnr', []):
                    if pd.notna(level) and np.isfinite(level):
                        label = 'SSNR' if not ssnr_labeled else ''
                        apds.append(mpf.make_addplot([level] * len(window_df), color='orange', linestyle='-', label=label))
                        logger.debug(f"Added ssnr for {symbol} on {timeframe}: {level}")
                        ssnr_labeled = True
            
            # Plot candlesticks first
            try:
                mpf.plot(
                    window_df[['open', 'high', 'low', 'close']],
                    type='candle',
                    volume=False,
                    ax=ax,
                    ylabel='Price',
                    show_nontrading=False,
                    tight_layout=False
                )
                logger.debug(f"Candlesticks plotted successfully for {symbol} on {timeframe}")
            except Exception as e:
                logger.error(f"Error plotting candlesticks for {symbol} on {timeframe}: {e}")
                ax.text(0.5, 0.5, f"Failed to plot candlesticks for {timeframe}", ha='center', va='center')
                ax.set_title(f"{timeframe} Chart ({symbol})")
                continue
            
            # Add SNR lines separately
            if apds:
                try:
                    for ap in apds:
                        mpf.plot(
                            window_df[['open', 'high', 'low', 'close']],
                            type='candle',
                            addplot=[ap],
                            ax=ax,
                            ylabel='Price',
                            show_nontrading=False,
                            tight_layout=False
                        )
                    logger.debug(f"SNR lines added for {symbol} on {timeframe}")
                except Exception as e:
                    logger.error(f"Error adding SNR lines for {symbol} on {timeframe}: {e}")
                    # Fallback: Add SNR lines manually
                    for level in snr_data.get('support', []):
                        if pd.notna(level) and np.isfinite(level):
                            label = 'Support' if not support_labeled else None
                            ax.axhline(y=level, color='green', linestyle='--', label=label, alpha=0.7)
                            support_labeled = True
                    for level in snr_data.get('resistance', []):
                        if pd.notna(level) and np.isfinite(level):
                            label = 'Resistance' if not resistance_labeled else None
                            ax.axhline(y=level, color='red', linestyle='--', label=label, alpha=0.7)
                            resistance_labeled = True
                    for level in snr_data.get('mini_snr', []):
                        if pd.notna(level) and np.isfinite(level):
                            label = 'Mini SNR' if not mini_snr_labeled else None
                            ax.axhline(y=level, color='blue', linestyle=':', label=label, alpha=0.7)
                            mini_snr_labeled = True
                    for level in snr_data.get('vssnr', []):
                        if pd.notna(level) and np.isfinite(level):
                            label = 'VSSNR' if not vssnr_labeled else None
                            ax.axhline(y=level, color='purple', linestyle='-', label=label, alpha=0.7)
                            vssnr_labeled = True
                    for level in snr_data.get('ssnr', []):
                        if pd.notna(level) and np.isfinite(level):
                            label = 'SSNR' if not ssnr_labeled else None
                            ax.axhline(y=level, color='orange', linestyle='-', label=label, alpha=0.7)
                            ssnr_labeled = True
            
            # Plot signals
            buy_labeled = False
            sell_labeled = False
            label_positions = {}  # {index: {'buy': count, 'sell': count}}

            for signal in signals:
                try:
                    ts = signal['timestamp']
                    if ts in window_df.index:
                        idx = window_df.index.get_loc(ts)
                        signal_type = signal['type']
                        ss_name = signal['sure_shot']
                        price = signal['price']

                        # Define marker and label positions
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
                        else:  # Sell
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

                        ax.scatter(idx, y_pos_signal, marker=marker, color=color, s=100, label=label, zorder=10)
                        ax.text(idx, y_pos_label, ss_name, fontsize=8, color=color, ha='center', va=va, zorder=10)
                        logger.debug(f"Plotted {signal_type} signal (label='{ss_name}') for {symbol} on {timeframe} at {ts}")
                    else:
                        logger.debug(f"Signal timestamp {ts} not in window for {symbol} on {timeframe}")
                except Exception as e:
                    logger.warning(f"Error plotting signal for {symbol} on {timeframe}: {e}")
                    continue
            
            # Set y-axis limits
            try:
                if not window_df.empty:
                    y_min = window_df['low'].min()
                    y_max = window_df['high'].max()
                    all_levels = []
                    if snr_data:
                        for key in ['support', 'resistance', 'mini_snr', 'vssnr', 'ssnr']:
                            all_levels.extend([x for x in snr_data.get(key, []) if pd.notna(x) and np.isfinite(x)])
                    signal_positions = []
                    label_positions_for_ylim = {}
                    for signal in signals:
                        ts = signal['timestamp']
                        if ts in window_df.index:
                            idx = window_df.index.get_loc(ts)
                            price_level = window_df.iloc[idx]['close']
                            offset_percentage = 0.02
                            if signal['type'] == 'Buy':
                                base_y_pos = window_df.iloc[idx]['low'] * 0.9996
                                if idx not in label_positions_for_ylim:
                                    label_positions_for_ylim[idx] = {'buy': 0, 'sell': 0}
                                count = label_positions_for_ylim[idx]['buy']
                                y_adjusted = base_y_pos - (price_level * offset_percentage / 100) * count
                                signal_positions.append(y_adjusted)
                                label_positions_for_ylim[idx]['buy'] += 1
                            else:
                                base_y_pos = window_df.iloc[idx]['high'] * 1.0004
                                if idx not in label_positions_for_ylim:
                                    label_positions_for_ylim[idx] = {'buy': 0, 'sell': 0}
                                count = label_positions_for_ylim[idx]['sell']
                                y_adjusted = base_y_pos + (price_level * offset_percentage / 100) * count
                                signal_positions.append(y_adjusted)
                                label_positions_for_ylim[idx]['sell'] += 1
                    if signal_positions:
                        all_levels.extend(signal_positions)
                    if all_levels:
                        y_min = min(y_min, min(all_levels)) * 0.999
                        y_max = max(y_max, max(all_levels)) * 1.001
                    price_range = y_max - y_min
                    y_min -= price_range * 0.05
                    y_max += price_range * 0.05
                    ax.set_ylim(y_min, y_max)
                    logger.debug(f"Set y-axis limits for {symbol} on {timeframe}: {y_min} to {y_max}")
            except Exception as e:
                logger.error(f"Error setting y-axis limits for {symbol} on {timeframe}: {e}")

            if buy_labeled or sell_labeled or support_labeled or resistance_labeled or mini_snr_labeled or vssnr_labeled or ssnr_labeled:
                ax.legend()
            ax.set_title(f"{timeframe} Chart ({symbol})", pad=10)
    
    slider.on_changed(update)
    update(slider.val if slider else 0)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.tight_layout(pad=2.0)
    plt.show()
    mt5.shutdown()

def plot_signals_multi(data_dict, initial=False):
    """Plot candlesticks, SNR lines, and signals for multiple timeframes with a slider."""
    global fig, axes, slider, data_dict_global, buttons
    data_dict_global = data_dict

    if initial or fig is None:
        plt.close('all')
        fig = plt.figure(figsize=(12, 8), dpi=150)
        n_timeframes = len(PLOT_TIMEFRAMES)
        if n_timeframes == 0:
            logger.error("PLOT_TIMEFRAMES is empty")
            fig.text(0.5, 0.5, "No timeframes specified", ha='center', va='center')
            return
        
        # Create subplot grid based on number of timeframes
        rows = min(2, (n_timeframes + 1) // 2)
        cols = min(2, n_timeframes)
        gs = fig.add_gridspec(rows, cols, top=0.92, bottom=0.18, hspace=0.4, wspace=0.3)
        axes = [fig.add_subplot(gs[i//cols, i%cols]) for i in range(n_timeframes)]
        
        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.04])
        
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
            fig.canvas.draw()
            return
        
        slider = Slider(slider_ax, 'Candle Index', 0, max_candles - DISPLAY_CANDLES, 
                        valinit=max_candles - DISPLAY_CANDLES, valstep=1)
        slider.on_changed(update)
        logger.debug(f"Slider initialized with range 0 to {max_candles - DISPLAY_CANDLES}")
    
    update(slider.val if slider else 0)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.tight_layout(pad=2.0)
    logger.debug("Plot initialized or updated")

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
        window_df = df.iloc[max(0, start_idx):end_idx]
        if window_df.empty:
            logger.warning(f"Empty window for {current_symbol} on {tf} at start_idx={start_idx}")
            ax.text(0.5, 0.5, f"No data in window for {tf}", ha='center', va='center')
            ax.set_title(f"{tf} Chart ({current_symbol})")
            continue
        
        snr_window_df = df.iloc[:end_idx]
        if len(snr_window_df) < 20:
            logger.warning(f"Insufficient data for SNR calculation for {current_symbol} on {tf}: {len(snr_window_df)} candles")
            snr_data = {
                'support': [], 'resistance': [], 'mini_snr': [], 'vssnr': [], 'ssnr': [],
                'vssnr_upper': [], 'vssnr_lower': [], 'ssnr_upper': [], 'ssnr_lower': [],
                'snr_upper': [], 'snr_lower': []
            }
        else:
            snr_data = calculate_snr(snr_window_df, tf)
            logger.debug(f"Recalculated SNR for {current_symbol} on {tf} at end_idx={end_idx}: {snr_data}")
        
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
        
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(window_df[col]):
                logger.error(f"Non-numeric data in {col} for {current_symbol} on {tf}")
                ax.text(0.5, 0.5, f"Invalid data in {col} for {tf}", ha='center', va='center')
                ax.set_title(f"{tf} Chart ({current_symbol})")
                continue
        
        # Prepare addplots
        apds = []
        support_labeled = False
        resistance_labeled = False
        mini_snr_labeled = False
        vssnr_labeled = False
        ssnr_labeled = False
        buy_labeled = False
        sell_labeled = False

        # Existing horizontal SNR lines
        if snr_data:
            for level in snr_data.get('support', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'Support' if not support_labeled else ''
                    apds.append(mpf.make_addplot([level] * len(window_df), color='green', linestyle='--', label=label))
                    support_labeled = True
            for level in snr_data.get('resistance', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'Resistance' if not resistance_labeled else ''
                    apds.append(mpf.make_addplot([level] * len(window_df), color='red', linestyle='--', label=label))
                    resistance_labeled = True
            for level in snr_data.get('mini_snr', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'Mini SNR' if not mini_snr_labeled else ''
                    apds.append(mpf.make_addplot([level] * len(window_df), color='blue', linestyle=':', label=label))
                    mini_snr_labeled = True
            for level in snr_data.get('vssnr', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'VSSNR' if not vssnr_labeled else ''
                    apds.append(mpf.make_addplot([level] * len(window_df), color='purple', linestyle='-', label=label))
                    vssnr_labeled = True
            for level in snr_data.get('ssnr', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'SSNR' if not ssnr_labeled else ''
                    apds.append(mpf.make_addplot([level] * len(window_df), color='orange', linestyle='-', label=label))
                    ssnr_labeled = True

        # Add trend lines
        trend_line_types = [
            ('vssnr_upper', 'purple', 'VSSNR Upper'),
            ('vssnr_lower', 'purple', 'VSSNR Lower'),
            ('ssnr_upper', 'yellow', 'SSNR Upper'),
            ('ssnr_lower', 'yellow', 'SSNR Lower'),
            ('snr_upper', 'gray', 'SNR Upper'),
            ('snr_lower', 'gray', 'SNR Lower'),
        ]
        for trend_type, color, label in trend_line_types:
            if trend_type in snr_data and snr_data[trend_type]:
                try:
                    (x1, y1), (x2, y2) = snr_data[trend_type]
                    if x2 != x1:  # Avoid division by zero
                        m = (y2 - y1) / (x2 - x1)
                        b = y1 - m * x1
                        window_x = np.arange(start_idx, min(end_idx, len(df)))
                        y_values = m * window_x + b
                        # Ensure the series aligns with window_df
                        trend_line_series = pd.Series(y_values, index=df.index[start_idx:min(end_idx, len(df))])
                        # Trim or pad to match window_df length
                        if len(trend_line_series) > len(window_df):
                            trend_line_series = trend_line_series[:len(window_df)]
                        elif len(trend_line_series) < len(window_df):
                            last_y = trend_line_series.iloc[-1]
                            trend_line_series = trend_line_series.reindex(window_df.index, method='ffill').fillna(last_y)
                        else:
                            trend_line_series = trend_line_series.reindex(window_df.index)
                        apds.append(mpf.make_addplot(trend_line_series, color=color, label=label))
                        logger.debug(f"Added {trend_type} trend line for {current_symbol} on {tf}")
                    else:
                        logger.warning(f"Vertical trend line for {trend_type} at x={x1}, skipping")
                except Exception as e:
                    logger.error(f"Error plotting {trend_type} for {current_symbol} on {tf}: {e}")

        # Plot candlesticks with all addplots
        plot_kwargs = {
            'type': 'candle',
            'volume': False,
            'ax': ax,
            'ylabel': 'Price',
            'show_nontrading': False,
            'tight_layout': False
        }
        if apds:
            plot_kwargs['addplot'] = apds
        
        try:
            mpf.plot(window_df[['open', 'high', 'low', 'close']], **plot_kwargs)
            logger.debug(f"Plotted candlesticks with {len(apds)} addplots for {current_symbol} on {tf}")
        except Exception as e:
            logger.error(f"Error plotting for {current_symbol} on {tf}: {e}")
            # Fallback plotting
            mpf.plot(window_df[['open', 'high', 'low', 'close']], type='candle', volume=False, ax=ax, show_nontrading=False, ylabel='Price')
            # Add horizontal lines manually
            for level in snr_data.get('support', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'Support' if not support_labeled else None
                    ax.axhline(y=level, color='green', linestyle='--', label=label, alpha=0.7)
                    support_labeled = True
            for level in snr_data.get('resistance', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'Resistance' if not resistance_labeled else None
                    ax.axhline(y=level, color='red', linestyle='--', label=label, alpha=0.7)
                    resistance_labeled = True
            for level in snr_data.get('mini_snr', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'Mini SNR' if not mini_snr_labeled else None
                    ax.axhline(y=level, color='blue', linestyle=':', label=label, alpha=0.7)
                    mini_snr_labeled = True
            for level in snr_data.get('vssnr', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'VSSNR' if not vssnr_labeled else None
                    ax.axhline(y=level, color='purple', linestyle='-', label=label, alpha=0.7)
                    vssnr_labeled = True
            for level in snr_data.get('ssnr', []):
                if pd.notna(level) and np.isfinite(level):
                    label = 'SSNR' if not ssnr_labeled else None
                    ax.axhline(y=level, color='orange', linestyle='-', label=label, alpha=0.7)
                    ssnr_labeled = True

        # Plot signals (unchanged)
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
                    
                    ax.scatter(idx, y_pos_marker, marker=marker, color=color, s=100, label=label, zorder=10)
                    ax.text(idx, y_pos_label, ss_name, fontsize=8, color=color, ha='center', va=va, zorder=10)
            except Exception as e:
                logger.warning(f"Error plotting signal for {current_symbol} on {tf}: {e}")
                continue
        
        # Adjust y-axis limits (unchanged)
        try:
            if not window_df.empty:
                y_min = window_df['low'].min()
                y_max = window_df['high'].max()
                if pd.notna(y_min) and pd.notna(y_max) and np.isfinite(y_min) and np.isfinite(y_max):
                    all_levels = []
                    if snr_data:
                        for key in ['support', 'resistance', 'mini_snr', 'vssnr', 'ssnr']:
                            all_levels.extend([x for x in snr_data.get(key, []) if pd.notna(x) and np.isfinite(x)])
                    label_positions = {}
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
                                key = 'buy'
                                offset_direction = -1
                            else:
                                y_pos_marker = window_df.iloc[idx]['high'] * 1.0001
                                base_y_pos_label = window_df.iloc[idx]['high'] * 1.0004
                                key = 'sell'
                                offset_direction = 1
                            if idx not in label_positions:
                                label_positions[idx] = {'buy': 0, 'sell': 0}
                            signal_count = label_positions[idx][key]
                            y_pos_label = base_y_pos_label + (price_level * offset_percentage / 100) * signal_count * offset_direction
                            label_positions[idx][key] += 1
                            all_levels.extend([y_pos_marker, y_pos_label])
                    # Include trend line points within the window
                    for key in ['vssnr_upper', 'vssnr_lower', 'ssnr_upper', 'ssnr_lower', 'snr_upper', 'snr_lower']:
                        if snr_data.get(key):
                            (x1, y1), (x2, y2) = snr_data[key]
                            if start_idx <= x1 < end_idx:
                                all_levels.append(y1)
                            if start_idx <= x2 < end_idx:
                                all_levels.append(y2)
                            # Extrapolate to window edges
                            if x2 != x1:
                                m = (y2 - y1) / (x2 - x1)
                                b = y1 - m * x1
                                y_start = m * start_idx + b
                                y_end = m * (end_idx - 1) + b
                                all_levels.extend([y_start, y_end])
                    if all_levels:
                        y_min = min(y_min, min(all_levels)) * 0.9995
                        y_max = max(y_max, max(all_levels)) * 1.0005
                    price_range = y_max - y_min
                    y_min -= price_range * 0.02
                    y_max += price_range * 0.02
                    ax.set_ylim(y_min, y_max)
                else:
                    logger.warning(f"Invalid y_min/y_max for {current_symbol} on {tf}: {y_min}, {y_max}")
            
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
        
        except Exception as e:
            logger.error(f"Error setting y-limits or legend for {current_symbol} on {tf}: {e}")
        
        ax.set_title(f"{tf} Chart ({current_symbol})", pad=10)
    
    fig.canvas.draw_idle()
    logger.info(f"Updated plot with slider at index {start_idx}")

def set_symbol(symbol):
    """Update the current symbol and refresh the plot."""
    global current_symbol
    logger.info(f"Switching to symbol: {symbol}")
    current_symbol = symbol
    update(slider.val if slider else 0)  # Redraw the plot with the new symbol
    fig.canvas.draw_idle()

class Backtester:
    def __init__(self):
        self.trades = []
        self.total_trades = 0
        self.wins = 0
        self.total_profit_loss = 0.0

    def filter_trade(self, df, entry_idx, signal):
        sure_shot = signal['sure_shot']
        market_type = get_market_type(df.iloc[:entry_idx + 1])
        
        valid_markets = {
            'SS-1': ['trending_up', 'trending_down'],
            'SS-2': ['ranging', 'stack'],
            'SS-3': ['trending_up', 'trending_down'],
            'SS-4': ['stack'],
            'SS-5': ['trending_up', 'trending_down'],
            'SS-6': ['ranging', 'stack'],
            'SS-7': ['trending_up', 'trending_down'],
            'SS-8': ['retracing_down', 'retracing_up'],
            'SS-YA': ['ranging', 'stack'],
            'SS-YS': ['ranging', 'stack']
        }
        if market_type not in valid_markets.get(sure_shot, []):
            logger.debug(f"Filtered {sure_shot} at {signal['timestamp']}: invalid market_type={market_type}")
            return False
        
        if sure_shot in ['SS-2', 'SS-6', 'SS-8', 'SS-YA', 'SS-YS']:
            snr_data = signal.get('snr_data', {})
            support_levels = snr_data.get('support', []).copy()
            resistance_levels = snr_data.get('resistance', []).copy()
            for level in snr_data.get('ssnr', []) + snr_data.get('vssnr', []):
                if pd.notna(level) and np.isfinite(level):
                    if level < df.iloc[entry_idx]['close']:
                        support_levels.append(level)
                    else:
                        resistance_levels.append(level)
            
            resistance = min(resistance_levels) if resistance_levels else float('inf')
            support = max(support_levels) if support_levels else -float('inf')
            
            candle = df.iloc[entry_idx]
            if signal['type'] == 'Buy':
                body_low = min(candle['open'], candle['close'])
                if support == -float('inf') or abs(body_low - support) / support > 0.0001:
                    logger.debug(f"Filtered Buy {sure_shot} at {signal['timestamp']}: body_low={body_low}, support={support}")
                    return False
            else:
                body_high = max(candle['open'], candle['close'])
                if resistance == float('inf') or abs(body_high - resistance) / resistance > 0.0001:
                    logger.debug(f"Filtered Sell {sure_shot} at {signal['timestamp']}: body_high={body_high}, resistance={resistance}")
                    return False
        
        return True

    def simulate_trade(self, df, entry_idx, signal):
        trade_type = signal['type']
        entry_price = signal['price']
        entry_time = signal['timestamp']
        
        if entry_idx + 1 >= len(df):
            logger.debug(f"No next candle for trade at index {entry_idx}, skipping")
            return 0.0

        if not self.filter_trade(df, entry_idx, signal):
            return 0.0

        # Dynamic stake: 1% of account balance
        account_balance = get_account_balance()
        stake = account_balance * RISK_PER_TRADE

        # Binary options: evaluate at next candle (1-minute expiration for M1)
        next_candle = df.iloc[entry_idx + 1]
        exit_price = next_candle['close']
        
        # Determine win/loss
        if trade_type == 'Buy':
            if exit_price > entry_price:
                profit_loss = stake * BINARY_PAYOUT  # Win: 85% payout
                self.wins += 1
                logger.debug(f"Buy trade won: entry={entry_price}, exit={exit_price}, profit={profit_loss}")
            else:
                profit_loss = -stake  # Loss: 100% of stake
                logger.debug(f"Buy trade lost: entry={entry_price}, exit={exit_price}, loss={profit_loss}")
        else:  # Sell
            if exit_price < entry_price:
                profit_loss = stake * BINARY_PAYOUT
                self.wins += 1
                logger.debug(f"Sell trade won: entry={entry_price}, exit={exit_price}, profit={profit_loss}")
            else:
                profit_loss = -stake
                logger.debug(f"Sell trade lost: entry={entry_price}, exit={exit_price}, loss={profit_loss}")

        # Record trade
        self.trades.append({
            'symbol': signal['symbol'],
            'timeframe': signal['timeframe'],
            'sure_shot': signal['sure_shot'],
            'type': trade_type,
            'timestamp': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'stake': stake
        })
        self.total_trades += 1
        self.total_profit_loss += profit_loss
        logger.debug(f"Simulated trade: {self.trades[-1]}")
        return profit_loss

    def run_backtest(self, df, symbol, timeframe, sure_shot_name):
        logger.info(f"Starting backtest for {symbol} on {timeframe}, sure_shot={sure_shot_name}, df_length={len(df)}")
        for i in range(50, len(df) - 2):
            try:
                if i % 100 == 0:
                    logger.info(f"Processing candle {i}/{len(df) - 2} for {symbol} {timeframe} {sure_shot_name}")
                window_df = df.iloc[:i + 1]
                signal = generate_signal(window_df, timeframe, symbol, sure_shot_name)
                if signal and signal['sure_shot'] == sure_shot_name:
                    logger.debug(f"Detected signal at index {i}: {signal}")
                    self.simulate_trade(df, i, signal)
                    i += 1
            except Exception as e:
                logger.error(f"Error at index {i} for {symbol} {timeframe} {sure_shot_name}: {e}")
                continue
        logger.info(f"Completed backtest for {symbol} on {timeframe}, sure_shot={sure_shot_name}, trades={self.total_trades}")

    def get_results(self):
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit_loss': 0.0,
                'avg_profit_loss_per_trade': 0.0
            }
        win_rate = (self.wins / self.total_trades) * 100 if self.total_trades > 0 else 0.0
        avg_profit_loss = self.total_profit_loss / self.total_trades if self.total_trades > 0 else 0.0
        return {
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_profit_loss': self.total_profit_loss,
            'avg_profit_loss_per_trade': avg_profit_loss
        }

def backtest_sure_shots():
    try:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        output_dir = os.getcwd()
    logger.info(f"Output directory: {output_dir}")

    results_file = os.path.join(output_dir, 'backtest_results.csv')
    summary_file = os.path.join(output_dir, 'backtest_summary.csv')
    logger.info(f"Will save trade records to: {results_file}")
    logger.info(f"Will save summary to: {summary_file}")

    try:
        test_file = os.path.join(output_dir, 'test_write.csv')
        pd.DataFrame({'test': [1]}).to_csv(test_file, index=False)
        logger.info(f"Test write successful: {test_file}")
        os.remove(test_file)
    except Exception as e:
        logger.error(f"Failed to write test file: {e}")
        raise PermissionError(f"Cannot write to {output_dir}: {e}")

    if not initialize_mt5():
        logger.error("Failed to initialize MT5 for backtesting")
        empty_df = pd.DataFrame(columns=['symbol', 'timeframe', 'sure_shot', 'timestamp', 'type', 'entry_price', 'exit_price', 'profit_loss', 'stake'])
        empty_summary = pd.DataFrame(columns=['symbol', 'timeframe', 'sure_shot', 'total_trades', 'win_rate', 'total_profit_loss', 'avg_profit_loss_per_trade'])
        try:
            empty_df.to_csv(results_file, index=False)
            empty_summary.to_csv(summary_file, index=False)
            logger.info(f"Saved empty results to {results_file} and {summary_file} due to MT5 failure")
        except Exception as e:
            logger.error(f"Failed to save empty results: {e}")
        return

    results = []
    trade_records = []
    try:
        for symbol in SYMBOLS:
            for timeframe in PLOT_TIMEFRAMES:
                logger.info(f"Backtesting {symbol} on {timeframe}")
                df = fetch_ohlcv(symbol, TIMEFRAME_MAP[timeframe], LIMIT)
                if df is None or df.empty or len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol} on {timeframe}, skipping")
                    continue
                logger.debug(f"Data for {symbol} {timeframe}: shape={df.shape}, last={df.tail(2)}")
                
                for ss in sure_shots:
                    backtester = Backtester()
                    backtester.run_backtest(df, symbol, timeframe, ss['name'])
                    result = backtester.get_results()
                    results.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'sure_shot': ss['name'],
                        'total_trades': result['total_trades'],
                        'win_rate': result['win_rate'],
                        'total_profit_loss': result['total_profit_loss'],
                        'avg_profit_loss_per_trade': result['avg_profit_loss_per_trade']
                    })
                    trade_records.extend(backtester.trades)
                    logger.info(
                        f"{ss['name']} on {symbol} {timeframe}: "
                        f"Trades={result['total_trades']}, "
                        f"WinRate={result['win_rate']:.2f}%, "
                        f"TotalProfitLoss={result['total_profit_loss']:.2f}, "
                        f"AvgProfitLoss={result['avg_profit_loss_per_trade']:.2f}"
                    )
                    logger.debug(f"Trades generated: {len(backtester.trades)} for {symbol} {timeframe} {ss['name']}")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
    finally:
        try:
            if trade_records:
                results_df = pd.DataFrame(trade_records)
                results_df.to_csv(results_file, index=False)
                logger.info(f"Saved {len(trade_records)} trade records to {results_file}")
            else:
                logger.warning("No trade records generated")
                pd.DataFrame(columns=['symbol', 'timeframe', 'sure_shot', 'timestamp', 'type', 'entry_price', 'exit_price', 'profit_loss', 'stake']).to_csv(results_file, index=False)
                logger.info(f"Saved empty trade records to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save trade records to {results_file}: {e}")

        try:
            if results:
                summary_df = pd.DataFrame(results)
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"Saved {len(results)} summary records to {summary_file}")
            else:
                logger.warning("No summary records generated")
                pd.DataFrame(columns=['symbol', 'timeframe', 'sure_shot', 'total_trades', 'win_rate', 'total_profit_loss', 'avg_profit_loss_per_trade']).to_csv(summary_file, index=False)
                logger.info(f"Saved empty summary to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save summary to {summary_file}: {e}")

    print("\nBacktest Summary:")
    for ss_name in [ss['name'] for ss in sure_shots]:
        ss_results = pd.DataFrame(results)[pd.DataFrame(results)['sure_shot'] == ss_name]
        if not ss_results.empty:
            total_trades = ss_results['total_trades'].sum()
            avg_win_rate = ss_results['win_rate'].mean()
            total_profit_loss = ss_results['total_profit_loss'].sum()
            avg_profit_loss = ss_results['avg_profit_loss_per_trade'].mean()
            print(
                f"{ss_name}: "
                f"Total Trades={total_trades}, "
                f"Avg Win Rate={avg_win_rate:.2f}%, "
                f"Total Profit/Loss={total_profit_loss:.2f}, "
                f"Avg Profit/Loss/Trade={avg_profit_loss:.2f}"
            )

    mt5.shutdown()

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

