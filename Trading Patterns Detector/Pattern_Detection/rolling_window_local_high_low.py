import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import asyncio
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.animation import FuncAnimation
import mplfinance as mpf
import threading
import time
import argparse
import os
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Configuration
MT5_LOGIN = 97109110
MT5_PASSWORD = "Gb*l5fZg"
MT5_SERVER = "MetaQuotes-Demo"
SYMBOLS = ['XAUUSD']
LIMIT = 5000
DISPLAY_CANDLES = 50
PLOT_TIMEFRAMES = ['M5', 'M15']
UPDATE_INTERVAL = 0.01
DEFAULT_ORDER = 10
FIGURE_DPI = 150
FIGURE_SIZE_PER_CHART = (12, 6)

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

def fetch_ohlcv_range(symbol, timeframe, start_date, end_date):
    try:
        start_time = time.time()
        if not mt5.terminal_info():
            logger.warning("MT5 connection lost, attempting to reconnect")
            if not initialize_mt5():
                return None
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            logger.error(f"No data fetched for {symbol} on {timeframe} from {start_date} to {end_date}: {mt5.last_error()}")
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
        logger.debug(f"Fetched {len(df)} candles for {symbol} on {timeframe} from {start_date} to {end_date}, took {time.time() - start_time:.3f}s")
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

def loss_derivative(y, y_pred):
    err = y_pred - y
    return np.sum(err) / len(err)

def optimize_slope(x, y, slope, pivot):
    intercept = y.iloc[pivot] - slope * x[pivot]
    y_pred = slope * x + intercept
    temp = 0
    MAX_ITERATION = 1000
    i = 0
    while i <= MAX_ITERATION:
        dj_dw = loss_derivative(y, y_pred)
        if abs(dj_dw - temp) <= 1e-5:
            break
        slope = slope - 0.01 * dj_dw
        y_pred = slope * x + intercept
        temp = dj_dw
        i += 1
    return slope

def optimize_intercept(x, y, intercept):
    y_pred = 0 * x + intercept
    temp = 0
    MAX_ITERATION = 1000
    i = 1
    while i <= MAX_ITERATION:
        dj_dw = loss_derivative(y, y_pred)
        if abs(dj_dw - temp) <= 1e-5:
            break
        intercept = intercept - 0.01 * dj_dw
        y_pred = 0 * x + intercept
        temp = dj_dw
        i += 1
    return intercept

def calculate_trend_line(df, initial_candles, order):
    data = df[-initial_candles:].copy()
    local_high, local_low = find_local_high_low(data, order)
    high_indices = [i for i, is_high in enumerate(local_high) if is_high]
    low_indices = [i for i, is_low in enumerate(local_low) if is_low]
    x = np.arange(len(data))
    if len(high_indices) < 2 or len(low_indices) < 2:
        support_trend_line, resistance_trend_line = calculate_trend_line(df, initial_candles, order-1)
        return support_trend_line, resistance_trend_line
    high_prices = [data['high'].iloc[i] for i in high_indices]
    sorted_highs = sorted(zip(high_prices, high_indices), reverse=True)[:2]
    high_pivot1_price, high_pivot1_idx = sorted_highs[0]
    high_pivot2_price, high_pivot2_idx = sorted_highs[1]
    low_prices = [data['low'].iloc[i] for i in low_indices]
    sorted_lows = sorted(zip(low_prices, low_indices))[:2]
    low_pivot1_price, low_pivot1_idx = sorted_lows[0]
    low_pivot2_price, low_pivot2_idx = sorted_lows[1]
    resistance_slope = (high_pivot1_price - high_pivot2_price) / (high_pivot1_idx - high_pivot2_idx)
    support_slope = (low_pivot1_price - low_pivot2_price) / (low_pivot1_idx - low_pivot2_idx)
    resistance_intercept = high_pivot1_price - resistance_slope * high_pivot1_idx
    support_intercept = low_pivot1_price - support_slope * low_pivot1_idx
    support_trend_line = support_slope * x + support_intercept
    resistance_trend_line = resistance_slope * x + resistance_intercept
    return support_trend_line, resistance_trend_line

def calculate_snr_line(df, initial_candles, order):
    data = df[-initial_candles:].copy()
    local_high, local_low = find_local_high_low(data, order)
    high_indices = [i for i, is_high in enumerate(local_high) if is_high]
    low_indices = [i for i, is_low in enumerate(local_low) if is_low]
    if len(high_indices) < 2 or len(low_indices) < 2:
        support_snr_line, resistance_snr_line = calculate_snr_line(df, initial_candles, order-1)
        return support_snr_line, resistance_snr_line
    high_prices = [data['high'].iloc[i] for i in high_indices]
    low_prices = [data['low'].iloc[i] for i in low_indices]
    coeff_r = high_prices[0] if len(high_prices) == 1 else np.polyfit(np.arange(len(high_prices)), high_prices, 1)
    coeff_s = low_prices[0] if len(low_prices) == 1 else np.polyfit(np.arange(len(low_prices)), low_prices, 1)
    x = np.arange(len(data))
    resistance_snr_line = 0 * x + optimize_intercept(np.arange(len(high_prices)), high_prices, coeff_r[1] if len(high_prices) > 1 else coeff_r)
    support_snr_line = 0 * x + optimize_intercept(np.arange(len(low_prices)), low_prices, coeff_s[1] if len(low_prices) > 1 else coeff_s)
    return support_snr_line, resistance_snr_line

def find_local_high_low(data, order):
    local_high = [False for _ in range(len(data))]
    local_low = [False for _ in range(len(data))]
    for i in range(0, len(data)):
        if (i < order) or (i >= len(data) - order):
            continue
        count_high = 0
        count_low = 0
        for j in range(1, order + 1):
            if (data['high'].iloc[i - j] <= data['high'].iloc[i]) and (data['high'].iloc[i] >= data['high'].iloc[i + j]):
                count_high += 1
            if (data['low'].iloc[i - j] >= data['low'].iloc[i]) and (data['low'].iloc[i] <= data['low'].iloc[i + j]):
                count_low += 1
        if count_high == order:
            local_high[i] = True
        elif count_low == order:
            local_low[i] = True
    return local_high, local_low

class LiveData:
    def __init__(self, symbol, timeframe, limit):
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.df = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.fetch_loop)
        self.thread.daemon = True
        self.thread.start()

    def fetch_loop(self):
        while self.running:
            df_new = fetch_ohlcv(self.symbol, self.timeframe, self.limit)
            if df_new is not None:
                with self.lock:
                    self.df = df_new
            time.sleep(UPDATE_INTERVAL)

    def get_data(self):
        with self.lock:
            return self.df.copy() if self.df is not None else None

    def stop(self):
        self.running = False
        self.thread.join()

def plot_common(ax_candles, plot_df, snr_data, trend_data, snr_candles, trend_candles, order, symbol, timeframe, title):
    try:
        if plot_df.empty or len(plot_df) < DISPLAY_CANDLES:
            ax_candles.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
            ax_candles.set_title(title)
            return False
        support_snr_line, resistance_snr_line = calculate_snr_line(snr_data, min(snr_candles, len(snr_data)), order)
        support_trend_line, resistance_trend_line = calculate_trend_line(trend_data, min(trend_candles, len(trend_data)), order)
        local_high, local_low = find_local_high_low(plot_df, order)
        high_indices = [i for i, is_high in enumerate(local_high) if is_high]
        low_indices = [i for i, is_low in enumerate(local_low) if is_low]
        addplots = [
            mpf.make_addplot(support_snr_line[-len(plot_df):], color='green', width=1.5, ax=ax_candles),
            mpf.make_addplot(resistance_snr_line[-len(plot_df):], color='red', width=1.5, ax=ax_candles),
            mpf.make_addplot(support_trend_line[-len(plot_df):], color='blue', width=1.5, ax=ax_candles),
            mpf.make_addplot(resistance_trend_line[-len(plot_df):], color='orange', width=1.5, ax=ax_candles),
            mpf.make_addplot(
                pd.Series([plot_df['high'].iloc[i] if i in high_indices else np.nan for i in range(len(plot_df))], index=plot_df.index),
                type='scatter', markersize=100, marker='v', color='red', ax=ax_candles
            ),
            mpf.make_addplot(
                pd.Series([plot_df['low'].iloc[i] if i in low_indices else np.nan for i in range(len(plot_df))], index=plot_df.index),
                type='scatter', markersize=100, marker='^', color='green', ax=ax_candles
            )
        ]
        mpf.plot(
            plot_df,
            type='candle',
            title=title,
            ylabel='Price',
            addplot=addplots,
            style='classic',
            show_nontrading=False,
            ax=ax_candles,
            tight_layout=False
        )
        y_min = min(plot_df['low'].min(), support_snr_line.min(), support_trend_line.min()) * 0.999
        y_max = max(plot_df['high'].max(), resistance_snr_line.max(), resistance_trend_line.max()) * 1.001
        ax_candles.set_ylim(y_min, y_max)
        return True
    except Exception as e:
        logger.error(f"Error in plot_common: {e}")
        ax_candles.text(0.5, 0.5, f"Error plotting: {str(e)}", ha='center', va='center')
        ax_candles.set_title(title)
        return False

def plot_live_multi(data_sources, symbols, timeframes, snr_candles=DISPLAY_CANDLES, trend_candles=DISPLAY_CANDLES, order=DEFAULT_ORDER):
    global DISPLAY_CANDLES
    n_timeframes = len(timeframes)
    n_cols = min(n_timeframes, 3)
    n_rows = 1
    fig_width = FIGURE_SIZE_PER_CHART[0] * n_cols
    fig_height = FIGURE_SIZE_PER_CHART[1] * (n_rows + 2)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)
    
    gs = fig.add_gridspec(n_rows + 2, n_cols, height_ratios=[4] * n_rows + [0.5, 0.5], hspace=0.4, wspace=0.2, top=0.95, bottom=0.1)
    
    axes = []
    for j in range(n_cols):
        if j < n_timeframes:
            ax = fig.add_subplot(gs[0, j])
            axes.append(ax)
        else:
            axes.append(None)
    
    slider_ax_snr = fig.add_axes([0.15, 0.08, 0.65, 0.03])
    slider_ax_candles = fig.add_axes([0.15, 0.11, 0.65, 0.03])
    slider_ax_trend = fig.add_axes([0.15, 0.14, 0.65, 0.03])
    slider_ax_order = fig.add_axes([0.15, 0.17, 0.65, 0.03])
    radio_ax = fig.add_axes([0.1, 0.02, 0.8, 0.06])
    
    snr_slider = Slider(slider_ax_snr, 'SNR Candles', 10, 200, valinit=DISPLAY_CANDLES, valstep=1)
    candles_slider = Slider(slider_ax_candles, 'Candles', 10, 200, valinit=DISPLAY_CANDLES, valstep=1)
    trend_slider = Slider(slider_ax_trend, 'Trend Candles', 10, 200, valinit=trend_candles, valstep=1)
    order_slider = Slider(slider_ax_order, 'Order', 1, 50, valinit=order, valstep=1)
    
    radio = RadioButtons(radio_ax, symbols, active=0)
    for label in radio.labels:
        label.set_fontsize(6)
        label.set_fontweight(1000)
    
    current_symbol = symbols[0]
    
    plot_configs = []
    for j, timeframe in enumerate(timeframes):
        if j < len(axes) and axes[j] is not None:
            plot_configs.append({
                'symbol': current_symbol,
                'timeframe': timeframe,
                'data_source': data_sources[current_symbol][timeframe],
                'ax': axes[j]
            })
    
    def update_plot(frame):
        try:
            global DISPLAY_CANDLES
            snr_candles = int(snr_slider.val)
            candles = int(candles_slider.val)
            trend_candles = int(trend_slider.val)
            order = int(order_slider.val)
            DISPLAY_CANDLES = candles
            
            for ax in axes:
                if ax is not None:
                    ax.clear()
            
            for config in plot_configs:
                ax = config['ax']
                symbol = config['symbol']
                timeframe = config['timeframe']
                data_source = config['data_source']
                
                if symbol != current_symbol:
                    continue
                
                df = data_source.get_data()
                if df is None or len(df) < DISPLAY_CANDLES:
                    ax.text(0.5, 0.5, "No data available", ha='center', va='center')
                    ax.set_title(f"{symbol} - {timeframe} (Live)")
                    continue
                
                start_idx = max(0, len(df) - DISPLAY_CANDLES)
                end_idx = len(df)
                plot_df = df.iloc[start_idx:end_idx]
                snr_data = df.iloc[max(0, start_idx - snr_candles):end_idx]
                trend_data = df.iloc[max(0, start_idx - trend_candles):end_idx]
                
                success = plot_common(
                    ax, plot_df, snr_data, trend_data, snr_candles, trend_candles, order,
                    symbol, timeframe, f"{symbol} - {timeframe} (Live)"
                )
                if success:
                    logger.info(f"Updated live plot for {symbol} - {timeframe} at index {start_idx} with order {order}")
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception as e:
            logger.error(f"Error in live plot update: {e}")
            for config in plot_configs:
                if config['symbol'] == current_symbol:
                    config['ax'].clear()
                    config['ax'].text(0.5, 0.5, f"Error updating plot: {str(e)}", ha='center', va='center')
                    config['ax'].set_title(f"{config['symbol']} - {config['timeframe']} (Live)")
            fig.canvas.draw()

    def update_symbol(label):
        nonlocal current_symbol, plot_configs
        current_symbol = label
        plot_configs = []
        for j, timeframe in enumerate(timeframes):
            if j < len(axes) and axes[j] is not None:
                plot_configs.append({
                    'symbol': current_symbol,
                    'timeframe': timeframe,
                    'data_source': data_sources[current_symbol][timeframe],
                    'ax': axes[j]
                })
        logger.info(f"Switched to symbol {current_symbol}")
        update_plot(None)
    
    def update_sliders(val):
        update_plot(None)

    radio.on_clicked(update_symbol)
    snr_slider.on_changed(update_sliders)
    candles_slider.on_changed(update_sliders)
    trend_slider.on_changed(update_sliders)
    order_slider.on_changed(update_sliders)
    
    ani = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
    plt.tight_layout(pad=2.0)
    plt.show()
    return ani, data_sources

def plot_backtest(data_dict, symbols, timeframes, start_date, end_date, snr_candles=DISPLAY_CANDLES, trend_candles=DISPLAY_CANDLES, order=DEFAULT_ORDER):
    global DISPLAY_CANDLES
    n_timeframes = len(timeframes)
    n_cols = min(n_timeframes, 3)
    n_rows = 1
    fig_width = FIGURE_SIZE_PER_CHART[0] * n_cols
    fig_height = FIGURE_SIZE_PER_CHART[1] * (n_rows + 3)  # Adjusted for additional slider
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)
    
    # Adjusted grid spec for 5 control rows (4 sliders + radio buttons)
    gs = fig.add_gridspec(n_rows + 5, n_cols, height_ratios=[4] * n_rows + [0.5, 0.5, 0.5, 0.5, 0.5], hspace=0.4, wspace=0.2, top=0.95, bottom=0.05)
    
    axes = []
    for j in range(n_cols):
        if j < n_timeframes:
            ax = fig.add_subplot(gs[0, j])
            axes.append(ax)
        else:
            axes.append(None)
    
    # Create sliders and radio buttons
    slider_ax_candle_index = fig.add_axes([0.15, 0.20, 0.65, 0.03])
    slider_ax_snr = fig.add_axes([0.15, 0.17, 0.65, 0.03])
    slider_ax_candles = fig.add_axes([0.15, 0.14, 0.65, 0.03])
    slider_ax_trend = fig.add_axes([0.15, 0.11, 0.65, 0.03])
    slider_ax_order = fig.add_axes([0.15, 0.08, 0.65, 0.03])
    radio_ax = fig.add_axes([0.1, 0.02, 0.8, 0.06])
    
    # Initialize sliders
    max_candles = max(len(data_dict[symbol][timeframe]) for symbol in symbols for timeframe in timeframes if data_dict[symbol][timeframe] is not None)
    candle_index_slider = Slider(slider_ax_candle_index, 'Candle Index', 0, max_candles - DISPLAY_CANDLES, valinit=max_candles - DISPLAY_CANDLES, valstep=1)
    snr_slider = Slider(slider_ax_snr, 'SNR Candles', 10, 200, valinit=snr_candles, valstep=1)
    candles_slider = Slider(slider_ax_candles, 'Candles', 10, 200, valinit=DISPLAY_CANDLES, valstep=1)
    trend_slider = Slider(slider_ax_trend, 'Trend Candles', 10, 200, valinit=trend_candles, valstep=1)
    order_slider = Slider(slider_ax_order, 'Order', 1, 50, valinit=order, valstep=1)
    
    radio = RadioButtons(radio_ax, symbols, active=0)
    for label in radio.labels:
        label.set_fontsize(6)
        label.set_fontweight(1000)
    
    current_symbol = symbols[0]
    
    plot_configs = []
    for j, timeframe in enumerate(timeframes):
        if j < len(axes) and axes[j] is not None:
            plot_configs.append({
                'symbol': current_symbol,
                'timeframe': timeframe,
                'data': data_dict[current_symbol][timeframe],
                'ax': axes[j]
            })
    
    def update_plot(val=None):
        try:
            global DISPLAY_CANDLES
            candle_index = int(candle_index_slider.val)
            snr_candles = int(snr_slider.val)
            candles = int(candles_slider.val)
            trend_candles = int(trend_slider.val)
            order = int(order_slider.val)
            DISPLAY_CANDLES = candles
            
            for ax in axes:
                if ax is not None:
                    ax.clear()
            
            for config in plot_configs:
                ax = config['ax']
                symbol = config['symbol']
                timeframe = config['timeframe']
                df = config['data']
                
                if symbol != current_symbol:
                    continue
                
                if df is None or len(df) < DISPLAY_CANDLES:
                    ax.text(0.5, 0.5, "No data available", ha='center', va='center')
                    ax.set_title(f"{symbol} - {timeframe} (Backtest {start_date.date()} to {end_date.date()})")
                    continue
                
                # Use candle_index to select the range of candles
                start_idx = max(0, min(candle_index, len(df) - DISPLAY_CANDLES))
                end_idx = min(start_idx + DISPLAY_CANDLES, len(df))
                plot_df = df.iloc[start_idx:end_idx]
                snr_data = df.iloc[max(0, start_idx - snr_candles):end_idx]
                trend_data = df.iloc[max(0, start_idx - trend_candles):end_idx]
                
                success = plot_common(
                    ax, plot_df, snr_data, trend_data, snr_candles, trend_candles, order,
                    symbol, timeframe, f"{symbol} - {timeframe} (Backtest {start_date.date()} to {end_date.date()})"
                )
                if success:
                    logger.info(f"Updated backtest plot for {symbol} - {timeframe} at index {start_idx} with order {order}")
            
            fig.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error in backtest plot update: {e}")
            for config in plot_configs:
                if config['symbol'] == current_symbol:
                    config['ax'].clear()
                    config['ax'].text(0.5, 0.5, f"Error updating plot: {str(e)}", ha='center', va='center')
                    config['ax'].set_title(f"{config['symbol']} - {config['timeframe']} (Backtest {start_date.date()} to {end_date.date()})")
            fig.canvas.draw()

    def update_symbol(label):
        nonlocal current_symbol, plot_configs
        current_symbol = label
        plot_configs = []
        for j, timeframe in enumerate(timeframes):
            if j < len(axes) and axes[j] is not None:
                plot_configs.append({
                    'symbol': current_symbol,
                    'timeframe': timeframe,
                    'data': data_dict[current_symbol][timeframe],
                    'ax': axes[j]
                })
        logger.info(f"Switched to symbol {current_symbol}")
        update_plot()

    radio.on_clicked(update_symbol)
    candle_index_slider.on_changed(update_plot)
    snr_slider.on_changed(update_plot)
    candles_slider.on_changed(update_plot)
    trend_slider.on_changed(update_plot)
    order_slider.on_changed(update_plot)
    
    update_plot()
    plt.tight_layout(pad=2.0)
    plt.show()
    return data_dict

async def main(mode='live', start_date=None, end_date=None):
    if not initialize_mt5():
        logger.error("Failed to initialize MT5, exiting")
        return
    data_sources = {symbol: {} for symbol in SYMBOLS}
    try:
        if mode == 'live':
            for symbol in SYMBOLS:
                for timeframe in PLOT_TIMEFRAMES:
                    data_sources[symbol][timeframe] = LiveData(symbol, TIMEFRAME_MAP[timeframe], LIMIT)
            ani, data_sources = plot_live_multi(data_sources, SYMBOLS, PLOT_TIMEFRAMES)
            plt.show()
        elif mode == 'backtest':
            if start_date is None or end_date is None:
                logger.error("Start date and end date must be provided for backtest mode")
                return
            for symbol in SYMBOLS:
                for timeframe in PLOT_TIMEFRAMES:
                    df = fetch_ohlcv_range(symbol, TIMEFRAME_MAP[timeframe], start_date, end_date)
                    data_sources[symbol][timeframe] = df
            plot_backtest(data_sources, SYMBOLS, PLOT_TIMEFRAMES, start_date, end_date)
        else:
            logger.error(f"Unknown mode: {mode}")
    except Exception as e:
        logger.error(f"Error in {mode} plotting: {e}")
    finally:
        if mode == 'live':
            for symbol in SYMBOLS:
                for timeframe in PLOT_TIMEFRAMES:
                    if symbol in data_sources and timeframe in data_sources[symbol]:
                        data_sources[symbol][timeframe].stop()
        mt5.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live and Backtest Multi-Timeframe Chart for MT5 with Symbol Toggle')
    parser.add_argument('--mode', choices=['live', 'backtest'], default='live', help='Mode: live or backtest')
    parser.add_argument('--start_date', type=str, help='Start date for backtest (YYYY-MM-DD)', default=None)
    parser.add_argument('--end_date', type=str, help='End date for backtest (YYYY-MM-DD)', default=None)
    args = parser.parse_args()
    
    start_date = None
    end_date = None
    if args.mode == 'backtest':
        if args.start_date is None or args.end_date is None:
            logger.error("Both --start_date and --end_date must be provided for backtest mode. Use format YYYY-MM-DD.")
            exit(1)
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            if end_date <= start_date:
                logger.error("End date must be after start date")
                exit(1)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
            exit(1)
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(args.mode, start_date, end_date))
    except KeyboardInterrupt:
        logger.info("Shutting down bot")
        mt5.shutdown()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()