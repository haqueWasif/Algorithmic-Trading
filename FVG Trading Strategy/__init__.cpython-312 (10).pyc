import pandas as pd
from datetime import timedelta
import logging
from src.utils.mt5_utils import timeframe_to_minutes  # CHANGED
# ... rest of fvg_analysis.py remains unchanged ...

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_fvg(df, timeframe_name):
    """Detect Fair Value Gaps in the DataFrame."""
    fvg_zones = []
    if df.empty:
        logger.error(f"Empty DataFrame passed to detect_fvg for {timeframe_name}")
        return df, fvg_zones
    df = df.copy()
    df['next_low'] = df['low'].shift(-1)
    df['next_high'] = df['high'].shift(-1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['FVG'] = (df['next_low'] > df['prev_high']) | (df['next_high'] < df['prev_low'])
    
    for i in range(1, len(df)-1):
        if df['next_low'].iloc[i] > df['prev_high'].iloc[i]:
            fvg_zones.append((df.index[i-1], df.index[i+1], df['prev_high'].iloc[i], df['next_low'].iloc[i], 'bullish'))
        elif df['next_high'].iloc[i] < df['prev_low'].iloc[i]:
            fvg_zones.append((df.index[i-1], df.index[i+1], df['next_high'].iloc[i], df['prev_low'].iloc[i], 'bearish'))
    logger.info(f"Detected {len(fvg_zones)} FVG zones for {timeframe_name}")
    return df, fvg_zones

def generate_signal(df, fvg_zones, timeframe):
    """Generate trading signals based on FVG zones after candle close."""
    if df.empty or not fvg_zones:
        logger.warning("Empty DataFrame or no FVG zones for signal generation")
        return None
    
    latest_candle = df.iloc[-1]
    current_time = df.index[-1]
    signal_window = timedelta(minutes=timeframe_to_minutes(timeframe) * 10)
    
    # NEW: Only consider FVGs where the end_time is before the latest candle
    for start_time, end_time, fvg_low, fvg_high, fvg_type in fvg_zones[-5:]:
        # CHANGED: Ensure FVG candle has closed by checking end_time < current_time
        if end_time < current_time:
            # Check if current time is within signal window after FVG formation
            if current_time <= end_time + signal_window:
                if fvg_type == 'bullish':
                    # CHANGED: Adjusted conditions to confirm signal after FVG close
                    if (latest_candle['close'] >= fvg_low and 
                        latest_candle['low'] <= fvg_high):
                        signal = {
                            'type': 'Buy',
                            'price': latest_candle['close'],
                            'fvg_low': fvg_low,
                            'fvg_high': fvg_high,
                            'timestamp': current_time,
                            'stop_loss': fvg_low - 0.0001,
                            'take_profit': latest_candle['close'] + 2 * (latest_candle['close'] - fvg_low)
                        }
                        logger.info(f"Bullish signal generated at {current_time} for FVG {fvg_low}-{fvg_high}")
                        return signal
                else:  # bearish
                    # CHANGED: Adjusted conditions to confirm signal after FVG close
                    if (latest_candle['close'] <= fvg_high and 
                        latest_candle['high'] >= fvg_low):
                        signal = {
                            'type': 'Sell',
                            'price': latest_candle['close'],
                            'fvg_low': fvg_low,
                            'fvg_high': fvg_high,
                            'timestamp': current_time,
                            'stop_loss': fvg_high + 0.0001,
                            'take_profit': latest_candle['close'] - 2 * (fvg_high - latest_candle['close'])
                        }
                        logger.info(f"Bearish signal generated at {current_time} for FVG {fvg_low}-{fvg_high}")
                        return signal
            else:
                logger.debug(f"FVG at {start_time}-{end_time} outside signal window")
        else:
            logger.debug(f"Skipping FVG at {start_time}-{end_time}: Candle not closed")
    
    logger.debug("No valid signal generated")
    return None