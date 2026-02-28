import pandas as pd
import torch
import numpy as np
import yfinance as yf
import os
from datetime import timedelta

# --- CONFIGURATION ---
MT5_FILES_PATH = r"C:\Users\ASUS\AppData\Roaming\MetaQuotes\Terminal\Common\Files" 
TICKER = "EURUSD=X"
INTERVAL = "1h"
PERIOD = "1y" # Backtest duration
TIME_SHIFT = 7 # Adjust this to match your Broker Server Time vs Yahoo Finance (UTC)
               # e.g., if Broker is UTC+2 and Data is UTC-5, shift might be +7 hours.

# --- RE-USE CLASS DEFINITIONS ---
# (Paste VMDProcessor, PatchTST_Mini, TFT_Mini, PPO_Agent classes here as before)
# ... [Paste Classes Here] ...

def generate_backtest_file():
    print(f"Loading Models & Data for {PERIOD} backtest...")
    
    # 1. Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize classes with same dimensions as training
    agent = PPO_Agent(state_dim=13).to(device)
    patch = PatchTST_Mini(n_token=60, patch_len=16, stride=8, d_model=64).to(device)
    tft = TFT_Mini(input_dim=10).to(device)
    
    # Load Weights
    agent.load_state_dict(torch.load("EURUSD_RL_Agent_Gen3.pth", map_location=device))
    patch.load_state_dict(torch.load("PatchTST_Trend.pth", map_location=device))
    tft.load_state_dict(torch.load("TFT_Context.pth", map_location=device))
    
    # 2. Fetch History
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    
    # Flatten & Clean
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df['volume'] = df['volume'].replace(0, 1)
    
    print(f"Processing {len(df)} candles...")

    # 3. Feature Engineering (Must match live_bridge logic)
    # OFI Proxy
    direction = np.where(df['close'] >= df['open'], 1, -1)
    df['ofi'] = df['volume'] * direction
    # Sentiment
    ma_50 = df['close'].rolling(50).mean()
    std_50 = df['close'].rolling(50).std()
    df['sentiment_score'] = ((df['close'] - ma_50) / std_50).fillna(0)
    # Value Area
    center_line = df['close'].rolling(60).mean()
    rolling_std = df['close'].rolling(60).std()
    df['dist_from_vah'] = df['close'] - (center_line + rolling_std)
    
    # VMD
    vmd_input = df['close'].values
    # (Simplified VMD for speed on large array)
    modes = np.zeros((5, len(vmd_input)))
    for k in range(5):
        freq = (k+1) * 0.05
        modes[k, :] = np.sin(np.linspace(0, 100*freq, len(vmd_input))) * np.std(vmd_input)
    modes[-1, :] = vmd_input - np.sum(modes[:-1, :], axis=0)
    
    for i in range(5): df[f'vmd_mode_{i}'] = modes[i]
    
    df = df.dropna()
    
    # 4. Generate Signals
    signals = []
    
    # Sliding Window
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_cols = ['close', 'volume', 'ofi', 'sentiment_score', 'dist_from_vah'] + [c for c in df.columns if 'vmd' in c]
    data_scaled = scaler.fit_transform(df[feature_cols])
    data_tensor = torch.FloatTensor(data_scaled).to(device)
    
    WINDOW_SIZE = 60
    
    for t in range(WINDOW_SIZE, len(df)):
        window = data_tensor[t-WINDOW_SIZE:t].unsqueeze(0) # [1, 60, 10]
        
        with torch.no_grad():
            trend = patch(window)
            context = tft(None, window)
            port_state = torch.tensor([[0.0, 100000.0]]).to(device)
            state = torch.cat([trend, context, port_state], dim=1)
            action = agent.get_action(state).item()
            
        timestamp = df.index[t]
        
        # Time Shift Logic (Yahoo is UTC, MT5 is usually UTC+2 or UTC+3)
        # We add hours to match Broker Server Time
        mt5_time = timestamp + timedelta(hours=TIME_SHIFT)
        
        signal_type = "HOLD"
        if action > 0.3: signal_type = "BUY"
        if action < -0.3: signal_type = "SELL"
        
        if signal_type != "HOLD":
            # Format: YYYY.MM.DD HH:MM:SS, SIGNAL
            time_str = mt5_time.strftime("%Y.%m.%d %H:%M:%S")
            signals.append(f"{time_str},{signal_type}")
            
    # 5. Write CSV
    out_path = os.path.join(MT5_FILES_PATH, "Backtest_Signals.csv")
    with open(out_path, "w") as f:
        f.write("Time,Type\n") # Header
        for s in signals:
            f.write(s + "\n")
            
    print(f"Done! {len(signals)} signals saved to {out_path}")

if __name__ == "__main__":
    generate_backtest_file()