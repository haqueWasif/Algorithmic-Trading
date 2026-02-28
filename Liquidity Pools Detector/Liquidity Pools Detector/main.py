# full_integrated_snr_trendline_pools_fixed.py
"""
Consolidated, cleaned version of your SNR + trendline -> liquidity pools -> LSTM pipeline.
Key fixes:
 - consistent method names
 - Timestamp handling fixed (use .value)
 - avoid DataFrame fragmentation (use pd.concat when adding many columns)
 - limit number of pool features to avoid OOM / memory errors
 - Plotly buttons keep candlestick visible
 - safe DBSCAN eps search bounded by ATR
 - optional model training to avoid heavy memory usage during debug
"""

import os
import math
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, Input
from tensorflow.keras.metrics import BinaryAccuracy
import plotly.graph_objects as go

# ---------- CONFIG ----------
MT5_PATH = r"C:\Program Files\MetaTrader\terminal64.exe"
MT5_LOGIN = 97109110
MT5_PASSWORD = "Gb*l5fZg"
MT5_SERVER = "MetaQuotes-Demo"

# Adjustable to avoid huge memory use:
MAX_POOL_FEATURES = 40    # keep at most this many pools as features (others will still be in liquidity_pools_all)
MAX_PLOTS_POOLS = 80      # how many pools to draw in "Show All" (to avoid extremely heavy plots)

# ---------- Helpers ----------
def calculate_atr(df: pd.DataFrame, period: int = 14, default_eps: float = 1e-5) -> float:
    if df is None or len(df) < 2:
        return default_eps
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(default_eps)
    atr = tr.rolling(window=min(period, len(tr))).mean().iloc[-1]
    if np.isnan(atr) or atr <= 0:
        return default_eps
    return float(atr)

def find_local_high_low(df: pd.DataFrame, order: int):
    n = len(df)
    local_high = [False] * n
    local_low = [False] * n
    for i in range(order, n - order):
        win_h = df['high'].iloc[i - order:i + order + 1]
        win_l = df['low'].iloc[i - order:i + order + 1]
        if df['high'].iloc[i] == win_h.max():
            local_high[i] = True
        if df['low'].iloc[i] == win_l.min():
            local_low[i] = True
    return local_high, local_low

def calculate_snr_line_dbscan(df: pd.DataFrame, initial_candles: int, eps: float, min_samples: int):
    """Cluster highs (resistance) and lows (support) separately. eps in absolute price units."""
    if df is None or len(df) == 0:
        return [], []
    data = df[-initial_candles:].copy()
    if len(data) < min_samples:
        return [], []

    X_high = data['high'].values.reshape(-1, 1)
    X_low = data['low'].values.reshape(-1, 1)

    db_high = DBSCAN(eps=eps, min_samples=min_samples).fit(X_high)
    db_low = DBSCAN(eps=eps, min_samples=min_samples).fit(X_low)

    resistances = []
    for label in sorted(set(db_high.labels_)):
        if label == -1:
            continue
        cluster_vals = X_high[db_high.labels_ == label].flatten()
        if len(cluster_vals) >= min_samples:
            resistances.append(float(cluster_vals.mean()))

    supports = []
    for label in sorted(set(db_low.labels_)):
        if label == -1:
            continue
        cluster_vals = X_low[db_low.labels_ == label].flatten()
        if len(cluster_vals) >= min_samples:
            supports.append(float(cluster_vals.mean()))

    return sorted(supports), sorted(resistances)

def optimize_dbscan_eps(df_sub: pd.DataFrame, atr_period: int = 14, default_eps: float = 0.002, min_samples_range=(2,3)):
    """Pick eps candidates based on ATR and choose one producing most levels (simple heuristic)."""
    atr = calculate_atr(df_sub, period=atr_period, default_eps=default_eps)
    # Bound eps candidates so it is never ridiculously large or tiny
    eps_candidates = [atr * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]
    # but ensure positive and reasonable
    eps_candidates = [max(default_eps * 0.1, min(c, max(default_eps * 100, atr * 2))) for c in eps_candidates]
    best = (default_eps, min_samples_range[0])
    best_score = -1
    for eps in eps_candidates:
        for min_s in min_samples_range:
            supports, resistances = calculate_snr_line_dbscan(df_sub, initial_candles=len(df_sub), eps=eps, min_samples=min_s)
            score = len(supports) + len(resistances)
            if score > best_score:
                best_score = score
                best = (eps, min_s)
    return best

def project_trendline_to_timestamp(idx1, price1, idx2, price2, df_timestamps, target_timestamp):
    """Project a straight line between (idx1,price1) and (idx2,price2) to the target_timestamp.
       df_timestamps is a pd.Series of Timestamps; target_timestamp is a single Timestamp."""
    t1 = int(df_timestamps.iloc[idx1].value // 10**9)
    t2 = int(df_timestamps.iloc[idx2].value // 10**9)
    t_target = int(target_timestamp.value // 10**9)
    if t2 == t1:
        return price2
    slope = (price2 - price1) / (t2 - t1)
    proj_price = price1 + slope * (t_target - t1)
    return float(proj_price)

# ---------- Main class ----------
class XAUUSDMultiStepLSTM:
    def __init__(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15,
                 from_date=datetime(2025, 1, 1), to_date=datetime(2025, 9, 30),
                 lookback=3, tolerance=0.0025, future_window=10,
                 seq_length=20, n_steps=3, recent_candles=2000,
                 price_range=0.05, max_pool_features=MAX_POOL_FEATURES,
                 train_model=True, train_epochs=5, train_batch=32):
        self.symbol = symbol
        self.timeframe = timeframe
        self.from_date = from_date
        self.to_date = to_date
        self.lookback = lookback
        self.tolerance = tolerance
        self.future_window = future_window
        self.seq_length = seq_length
        self.n_steps = n_steps
        self.recent_candles = recent_candles
        self.price_range = price_range

        # control memory: how many pools to include as features (others still stored)
        self.max_pool_features = max_pool_features

        # model training control
        self.train_model_flag = train_model
        self.train_epochs = train_epochs
        self.train_batch = train_batch

        # data / results
        self.df = None
        self.swings = []
        self.snr_levels = []
        self.trendlines = []
        self.liquidity_pools_all = []
        self.liquidity_pools = []  # potentially filtered for training
        self.feature_cols = []
        self.scaler = MinMaxScaler()
        self.X = None
        self.y = None
        self.model = None

    # ------------------------
    # MT5 load
    # ------------------------
    def load_data(self):
        if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, timeout=15000, portable=False):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            mt5.shutdown()
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

        rates = mt5.copy_rates_range(self.symbol, self.timeframe, self.from_date, self.to_date)
        mt5.shutdown()
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_range failed or returned empty: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']].reset_index(drop=True)
        df.rename(columns={'tick_volume': 'tick_volume'}, inplace=True)
        self.df = df
        print(f"[load_data] Loaded {len(self.df)} bars for {self.symbol}")
        return self.df

    # ------------------------
    # Swings detection (local highs / lows)
    # ------------------------
    def detect_swings(self):
        if self.df is None:
            raise RuntimeError("Load data first")
        highs_mask, lows_mask = find_local_high_low(self.df, self.lookback)
        swings = []
        for i in range(len(self.df)):
            if highs_mask[i]:
                swings.append({'index': i, 'price': float(self.df['high'].iloc[i]), 'type': 'swing_high'})
            if lows_mask[i]:
                swings.append({'index': i, 'price': float(self.df['low'].iloc[i]), 'type': 'swing_low'})
        self.swings = swings
        print(f"[detect_swings] Detected {len(swings)} swings")
        return swings

    # ------------------------
    # SNR detection via DBSCAN (supports & resistances)
    # ------------------------
    def detect_snr(self, use_optimize_eps=True, min_samples=2):
        if self.df is None:
            raise RuntimeError("Load data first")
        # choose subset for eps optimization (recent portion)
        df_sub = self.df[-min(500, len(self.df)):]
        if use_optimize_eps and len(df_sub) >= 10:
            eps, min_s = optimize_dbscan_eps(df_sub, atr_period=14, default_eps=0.002, min_samples_range=[min_samples, min_samples+1])
        else:
            eps = calculate_atr(self.df, period=14)  # sensible default
            min_s = min_samples
        supports, resistances = calculate_snr_line_dbscan(self.df, initial_candles=len(self.df), eps=eps, min_samples=min_s)
        self.snr_levels = [{'price': p, 'type': 'support'} for p in supports] + [{'price': p, 'type': 'resistance'} for p in resistances]
        print(f"[detect_snr] Supports:{len(supports)} Resistances:{len(resistances)} (eps={eps:.6g}, min_samples={min_s})")
        return self.snr_levels

    # ------------------------
    # Trendlines from swings (connect consecutive highs / lows and project to last timestamp)
    # ------------------------
    def detect_trendlines_and_project(self):
        if not self.swings:
            self.detect_swings()
        df = self.df
        highs = [s for s in self.swings if s['type'] == 'swing_high']
        lows = [s for s in self.swings if s['type'] == 'swing_low']
        trendlines = []
        # connect consecutive highs -> downtrend projections
        for i in range(len(highs) - 1):
            p1, p2 = highs[i], highs[i + 1]
            if p2['index'] <= p1['index']:
                continue
            proj_price = project_trendline_to_timestamp(p1['index'], p1['price'], p2['index'], p2['price'], df['timestamp'], df['timestamp'].iloc[-1])
            trendlines.append({
                'type': 'downtrend',
                'anchor_indices': (p1['index'], p2['index']),
                'anchor_prices': (p1['price'], p2['price']),
                'projected_price_at_last': proj_price,
                'color': 'red',
                'median_index': int((p1['index'] + p2['index']) // 2)
            })
        # connect consecutive lows -> uptrend projections
        for i in range(len(lows) - 1):
            p1, p2 = lows[i], lows[i + 1]
            if p2['index'] <= p1['index']:
                continue
            proj_price = project_trendline_to_timestamp(p1['index'], p1['price'], p2['index'], p2['price'], df['timestamp'], df['timestamp'].iloc[-1])
            trendlines.append({
                'type': 'uptrend',
                'anchor_indices': (p1['index'], p2['index']),
                'anchor_prices': (p1['price'], p2['price']),
                'projected_price_at_last': proj_price,
                'color': 'blue',
                'median_index': int((p1['index'] + p2['index']) // 2)
            })
        self.trendlines = trendlines
        print(f"[detect_trendlines_and_project] Detected {len(trendlines)} trendlines")
        return trendlines

    # ------------------------
    # Form liquidity pools (merge SNR, trend projections, swings)
    # ------------------------
    def form_liquidity_pools(self):
        if self.df is None:
            raise RuntimeError("Load data first")
        if not self.snr_levels:
            self.detect_snr()
        if not self.trendlines:
            self.detect_trendlines_and_project()
        if not self.swings:
            self.detect_swings()

        pools = []
        tol_price = self.tolerance * self.df['close'].iloc[-1]
        # 1) SNR levels
        for snr in self.snr_levels:
            pools.append({
                'pool_id': len(pools),
                'type': f"snr_{snr['type']}",
                'price': float(snr['price']),
                'indices': [len(self.df)-1],
                'count': 1,
                'median_index': len(self.df)-1,
                'weight': 2.0
            })
        # 2) Trendline projections
        for tr in self.trendlines:
            pools.append({
                'pool_id': len(pools),
                'type': f"trend_{tr['type']}",
                'price': float(tr['projected_price_at_last']),
                'indices': [tr['median_index']],
                'count': 1,
                'median_index': tr['median_index'],
                'weight': 1.8
            })
        # 3) Swings merge if close
        for s in self.swings:
            s_price = s['price']
            merged = False
            for p in pools:
                same_side = (('high' in s['type'] and ('high' in p['type'] or 'resistance' in p['type'])) or
                             ('low' in s['type'] and ('low' in p['type'] or 'support' in p['type'])))
                if abs(p['price'] - s_price) <= tol_price and same_side:
                    total_w = p.get('weight', 1.0) + 1.0
                    p['price'] = (p['price'] * p.get('weight', 1.0) + s_price * 1.0) / total_w
                    p['weight'] = total_w
                    p['count'] += 1
                    p['indices'].append(s['index'])
                    p['median_index'] = int(np.median(p['indices']))
                    merged = True
                    break
            if not merged:
                pools.append({
                    'pool_id': len(pools),
                    'type': s['type'],
                    'price': float(s_price),
                    'indices': [s['index']],
                    'count': 1,
                    'median_index': s['index'],
                    'weight': 1.0
                })
        pools = sorted(pools, key=lambda x: x['price'])
        # store
        self.liquidity_pools_all = pools[:]
        # choose pools used for features/training but limit count to avoid memory blow-up
        self.liquidity_pools = pools[:self.max_pool_features]
        print(f"[form_liquidity_pools] Formed {len(pools)} pools (kept {len(self.liquidity_pools)} for features)")
        return pools

    # ------------------------
    # Prepare features + labels
    # ------------------------
    def prepare_features_and_labels(self):
        if self.df is None:
            raise RuntimeError("Load data first")
        df = self.df.copy()
        # compute base features
        df['return'] = df['close'].pct_change().fillna(0)
        df['volatility'] = (df['high'] - df['low']).fillna(0)
        df['momentum'] = (df['close'] - df['close'].shift(5)).fillna(0)
        df['hour'] = df['timestamp'].dt.hour
        df['session'] = df['hour'].apply(lambda x: 0 if x < 8 else 1 if x < 16 else 2)

        # build pool distance columns, but collect into a dict then concat to avoid fragmentation
        extra_cols = {}
        if self.liquidity_pools:
            for i, pool in enumerate(self.liquidity_pools):
                price = pool['price']
                col = f'dist_pool_{i}'
                # for horizontal/resistance define price - close (positive if pool above price)
                if ('high' in pool['type'] or 'resistance' in pool['type'] or 'trend_down' in pool['type']):
                    extra_cols[col] = price - df['close']
                elif ('low' in pool['type'] or 'support' in pool['type'] or 'trend_up' in pool['type']):
                    extra_cols[col] = df['close'] - price
                else:
                    extra_cols[col] = (df['close'] - price).abs()
        else:
            extra_cols['dist_pool_0'] = pd.Series(np.zeros(len(df)), index=df.index)

        # concat extra cols at once
        df = pd.concat([df, pd.DataFrame(extra_cols, index=df.index)], axis=1)

        # feature columns
        self.feature_cols = ['return', 'volatility', 'momentum', 'session'] + [c for c in df.columns if c.startswith('dist_pool_')]
        features = df[self.feature_cols].fillna(0).astype(float).reset_index(drop=True)

        num_pools = max(1, len(self.liquidity_pools))
        total_future = self.future_window * self.n_steps
        min_end_idx = self.seq_length - 1
        max_end_idx = len(df) - 1 - total_future
        if max_end_idx < min_end_idx:
            print("[prepare_features_and_labels] Not enough data to build sequences/labels.")
            return None, None

        # Fit scaler on training portion only (80%) to avoid leakage
        train_size = max(1, int(len(features) * 0.8))
        self.scaler.fit(features.iloc[:train_size].values)
        features_scaled_all = self.scaler.transform(features.values)

        X, y = [], []
        for end_idx in range(min_end_idx, max_end_idx + 1):
            start_idx = end_idx - self.seq_length + 1
            seq = features_scaled_all[start_idx:end_idx + 1]
            step_labels = []
            for step in range(self.n_steps):
                start_f = end_idx + 1 + step * self.future_window
                end_f = start_f + self.future_window - 1
                future_high = df['high'].iloc[start_f:end_f + 1].max()
                future_low = df['low'].iloc[start_f:end_f + 1].min()
                step_label = np.zeros(num_pools, dtype=int)
                for i, pool in enumerate(self.liquidity_pools):
                    pool_price = pool['price']
                    if ('high' in pool['type'] or 'resistance' in pool['type']) and future_high >= pool_price:
                        step_label[i] = 1
                    elif ('low' in pool['type'] or 'support' in pool['type']) and future_low <= pool_price:
                        step_label[i] = 1
                step_labels.append(step_label)
            X.append(seq)
            y.append(step_labels)

        self.X = np.array(X)
        self.y = np.array(y)
        print(f"[prepare_features_and_labels] Prepared X={self.X.shape if self.X is not None else None}, y={self.y.shape if self.y is not None else None}")
        return self.X, self.y

    # ------------------------
    # Model build/train/predict
    # ------------------------
    def build_model(self):
        num_pools = max(1, len(self.liquidity_pools))
        if self.X is None:
            raise RuntimeError("Call prepare_features_and_labels() before build_model()")
        model = Sequential()
        model.add(Input(shape=(self.seq_length, self.X.shape[2])))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64))
        model.add(RepeatVector(self.n_steps))
        model.add(LSTM(64, return_sequences=True))
        model.add(TimeDistributed(Dense(num_pools, activation='sigmoid')))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[BinaryAccuracy(name='binary_acc')])
        self.model = model
        print("[build_model] LSTM model built")
        return model

    def train(self, epochs=None, batch_size=None):
        if not self.train_model_flag:
            print("[train] Training skipped (train_model_flag=False)")
            return
        if self.model is None:
            raise RuntimeError("Call build_model() before train()")
        if self.X is None or self.y is None:
            raise RuntimeError("Call prepare_features_and_labels() before train()")
        epochs = epochs or self.train_epochs
        batch_size = batch_size or self.train_batch
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        print("[train] Training finished")

    def predict_next_steps(self):
        if self.model is None or self.X is None:
            raise RuntimeError("Need model and prepared data")
        latest_seq = self.X[-1].reshape(1, self.seq_length, self.X.shape[2])
        probs = self.model.predict(latest_seq)[0]  # shape (n_steps, num_pools)
        ranked_steps = []
        for step_probs in probs:
            ranked = sorted(enumerate(step_probs), key=lambda x: x[1], reverse=True)
            ranked_steps.append(ranked)
        return ranked_steps

    # ------------------------
    # Plotting (Plotly) - keeps candles visible always
    # ------------------------
    def plot_candlestick_with_pools(self, max_pools=MAX_PLOTS_POOLS, price_window=0.05):
        if self.df is None:
            raise RuntimeError("Load data first")
        df = self.df.copy()
        last_price = df['close'].iloc[-1]
        pools_all = getattr(self, 'liquidity_pools_all', []) or []
        pools_all = pools_all[:max_pools]   # limit to avoid heavy chart

        if not pools_all:
            print("[plot_candlestick_with_pools] No pools available to plot.")
            return

        # subsets
        recent_threshold_index = max(0, len(df) - (self.recent_candles or len(df)))
        pools_recent = [p for p in pools_all if p['median_index'] >= recent_threshold_index]
        pools_near = [p for p in pools_all if abs(p['price'] - last_price) / last_price <= price_window]

        # 🔑 ensure some pools always show
        if not pools_recent:
            pools_recent = sorted(pools_all, key=lambda p: abs(p['price'] - last_price))[:10]
        if not pools_near:
            pools_near = sorted(pools_all, key=lambda p: abs(p['price'] - last_price))[:10]

        fig = go.Figure()

        # --- candlesticks always visible
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=self.symbol,
            increasing_line_color='green',
            decreasing_line_color='red',
            visible=True
        ))

        # helper to add pools
        def add_pool_group(pool_list, suffix, visible):
            added_vis = []
            for p in pool_list:
                color = 'red' if ('high' in p['type'] or 'resistance' in p['type'] or 'trend_down' in p['type']) else 'blue'
                # horizontal line
                fig.add_trace(go.Scatter(
                    x=[df['timestamp'].iloc[0], df['timestamp'].iloc[-1]],
                    y=[p['price'], p['price']],
                    mode='lines',
                    line=dict(color=color, dash='dash'),
                    name=f"{p['type']} line {suffix}",
                    visible=visible,
                    showlegend=False
                ))
                # marker
                ts = df['timestamp'].iloc[p['median_index']]
                fig.add_trace(go.Scatter(
                    x=[ts], y=[p['price']], mode='markers',
                    marker=dict(size=8 + int(p.get('weight', 1.0) * 3), color=color),
                    name=f"{p['type']} marker {suffix}",
                    visible=visible,
                    hovertemplate=f"{p['type']} {p['price']:.4f}<extra></extra>",
                    showlegend=False
                ))
                added_vis.extend([visible, visible])
            return added_vis

        vis_all = add_pool_group(pools_all, "(all)", visible=False)
        vis_recent = add_pool_group(pools_recent, "(recent)", visible=True)
        vis_near = add_pool_group(pools_near, "(near)", visible=False)

        # legend helpers
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="red", dash="dash"), name="High/Resistance"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="blue", dash="dash"), name="Low/Support"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="black", dash="dot"), name="Trendlines"))

        total_traces = len(fig.data)
        vis_all_mask = [False] * total_traces
        vis_recent_mask = [False] * total_traces
        vis_near_mask = [False] * total_traces
        vis_hide_mask = [False] * total_traces

        # candlestick + legend always on
        for i, tr in enumerate(fig.data):
            if i == 0 or "high/resistance" in (tr.name or "").lower() or "low/support" in (tr.name or "").lower() or "trendlines" in (tr.name or "").lower():
                vis_all_mask[i] = vis_recent_mask[i] = vis_near_mask[i] = vis_hide_mask[i] = True

        idx = 1
        for j, v in enumerate(vis_all):
            if idx + j < total_traces:
                vis_all_mask[idx + j] = v
        idx += len(vis_all)
        for j, v in enumerate(vis_recent):
            if idx + j < total_traces:
                vis_recent_mask[idx + j] = v
                vis_all_mask[idx + j] |= v
        idx += len(vis_recent)
        for j, v in enumerate(vis_near):
            if idx + j < total_traces:
                vis_near_mask[idx + j] = v
                vis_all_mask[idx + j] |= v

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=1.0, xanchor="right",
                y=1.12, yanchor="top",
                buttons=[
                    dict(label="Show All Pools", method="update", args=[{"visible": vis_all_mask}]),
                    dict(label="Show Recent Pools", method="update", args=[{"visible": vis_recent_mask}]),
                    dict(label="Show Near Pools", method="update", args=[{"visible": vis_near_mask}]),
                    dict(label="Hide Pools", method="update", args=[{"visible": vis_hide_mask}])
                ]
            )],
            title=f"{self.symbol} Liquidity Pools (Interactive)",
            yaxis_title="Price",
            xaxis_title="Time",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700
        )

        try:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        except Exception:
            pass

        fig.show()




obj = XAUUSDMultiStepLSTM(
    symbol="XAUUSD",
    timeframe=mt5.TIMEFRAME_M15,
    from_date=datetime(2025, 1, 1),
    to_date=datetime(2025, 9, 30),
    lookback=3,
    tolerance=0.0025,
    future_window=10,
    seq_length=20,
    n_steps=3,
    recent_candles=2000,
    price_range=0.05,
    max_pool_features=MAX_POOL_FEATURES,
    train_model=True,           # set False while debugging / plotting to avoid memory errors
    train_epochs=5,
    train_batch=32
)
# Load, detect, form pools, prepare (optional train)
obj.load_data()
obj.detect_swings()
obj.detect_snr()
obj.detect_trendlines_and_project()
obj.form_liquidity_pools()
X, y = obj.prepare_features_and_labels()
if X is not None and y is not None and obj.train_model_flag:
    obj.build_model()
    obj.train()
    preds = obj.predict_next_steps()
    print("Predicted ranking (per step):", preds)
# Plot: candlesticks remain visible and pools toggle correctly
obj.plot_candlestick_with_pools()










1. 2% loss
2. 1% loss 
3. 0.5% loss
4. 0.5% win
5. 0.5% win (50% of 1% loss recovered)
6. 1% win (25% of 2% loss recovered)
7. 1% win (75% of 2% loss recovered)
8. 2% win 