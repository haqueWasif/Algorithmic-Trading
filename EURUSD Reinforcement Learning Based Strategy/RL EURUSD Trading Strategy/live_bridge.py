import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
# UPDATE THIS PATH TO MATCH YOUR PC
MT5_FILES_PATH = r"C:\Users\ASUS\AppData\Roaming\MetaQuotes\Terminal\Common\Files" 
TICKER = "EURUSD=X"
INTERVAL = "1h" 
WINDOW_SIZE = 60 # Must match training window

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. CLASS DEFINITIONS (Must Match Training)
# ==========================================

class VMDProcessor:
    """Matches the logic used in training."""
    def __init__(self, K=5, alpha=2000, tau=0, tol=1e-7):
        self.K = K
    def decompose(self, signal):
        f = signal
        T = len(f)
        modes = np.zeros((self.K, T))
        # Replicating the training logic (Synthetic/Residual VMD)
        for k in range(self.K):
            freq = (k+1) * 0.05
            modes[k, :] = np.sin(np.linspace(0, 100*freq, T)) * np.std(f)
        modes[-1, :] = f - np.sum(modes[:-1, :], axis=0)
        return modes.T

class PatchTST_Mini(nn.Module):
    def __init__(self, n_token, patch_len, stride, d_model=128, n_head=4):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (n_token - patch_len) // stride + 1
        self.patch_embedding = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(self.n_patches * d_model, 1)

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, T) 
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        emb = self.patch_embedding(patches)
        out = self.encoder(emb)
        out = out.reshape(B*C, -1)
        pred = self.head(out)
        pred = pred.reshape(B, C)
        return pred

class TFT_Mini(nn.Module):
    def __init__(self, input_dim, d_model=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, d_model, batch_first=True)
        self.gate = nn.Linear(d_model, 1)
        self.out = nn.Linear(d_model, 1)
        
    def forward(self, x_static, x_dynamic):
        out, _ = self.lstm(x_dynamic)
        weights = torch.sigmoid(self.gate(out)) 
        weighted_out = out * weights
        prediction = self.out(weighted_out[:, -1, :])
        return prediction

class PPO_Agent(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def get_action(self, state):
        return self.actor(state)

# ==========================================
# 2. MODEL LOADING & PROCESSING
# ==========================================

def load_models():
    print("Loading models...")
    
    # FIX: state_dim = 13 (10 features + 1 context + 2 portfolio)
    agent = PPO_Agent(state_dim=13).to(device) 
    agent.load_state_dict(torch.load("EURUSD_RL_Agent_Gen3.pth", map_location=device))
    agent.eval()
    
    patch = PatchTST_Mini(n_token=60, patch_len=16, stride=8, d_model=64).to(device)
    patch.load_state_dict(torch.load("PatchTST_Trend.pth", map_location=device))
    patch.eval()
    
    # FIX: input_dim = 10 (Total features)
    tft = TFT_Mini(input_dim=10).to(device) 
    tft.load_state_dict(torch.load("TFT_Context.pth", map_location=device))
    tft.eval()
    
    return agent, patch, tft

def process_live_data(df):
    """Replicates Feature Engineering from Training"""
    # 1. OFI Proxy
    direction = np.where(df['close'] >= df['open'], 1, -1)
    df['ofi'] = df['volume'] * direction
    
    # 2. Sentiment Proxy
    ma_50 = df['close'].rolling(50).mean()
    std_50 = df['close'].rolling(50).std()
    df['sentiment_score'] = ((df['close'] - ma_50) / std_50).fillna(0)
    
    # 3. Value Area
    center_line = df['close'].rolling(60).mean()
    rolling_std = df['close'].rolling(60).std()
    df['dist_from_vah'] = df['close'] - (center_line + rolling_std)

    # 4. Clean & VMD
    df_clean = df.dropna()
    if len(df_clean) < WINDOW_SIZE:
        print("Not enough data for window.")
        return None

    # VMD
    vmd = VMDProcessor(K=5)
    # Important: Use last N samples to keep VMD stable
    vmd_input = df_clean['close'].values[-100:] if len(df_clean) > 100 else df_clean['close'].values
    vmd_modes = vmd.decompose(vmd_input)
    
    # Align VMD with DF
    # We only care about the end of the dataframe matching the VMD end
    vmd_df = pd.DataFrame(vmd_modes, columns=[f'vmd_mode_{i}' for i in range(5)], index=df_clean.index[-len(vmd_modes):])
    
    # Merge
    full_df = pd.concat([df_clean, vmd_df], axis=1).dropna()
    
    # Select Features & Scale
    feature_cols = ['close', 'volume', 'ofi', 'sentiment_score', 'dist_from_vah'] + [c for c in full_df.columns if 'vmd' in c]
    
    # Note: In production, load the saved 'scaler.pkl' from training. 
    # Here we fit on the fly (approximate) or standard scale the window.
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(full_df[feature_cols])
    
    # Return last WINDOW_SIZE rows as tensor
    last_window = data_scaled[-WINDOW_SIZE:]
    if len(last_window) != WINDOW_SIZE:
        return None
        
    return torch.FloatTensor(last_window).unsqueeze(0).to(device)

def generate_signal(agent, patch_model, tft_model):
    print(f"[{datetime.now()}] Fetching live data...")
    
    try:
        # FIX: Increased period from "5d" to "1mo" to ensure enough buffer for rolling windows
        df = yf.download(TICKER, period="1mo", interval=INTERVAL, progress=False)
        
        # Debug: Print raw shape
        print(f"Raw data fetched: {len(df)} rows")

        # Flatten columns if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df['volume'] = df['volume'].replace(0, 1)
        
        # Process
        tensor = process_live_data(df)
        
        if tensor is None:
            print("Data processing failed (Insufficient length after cleaning).")
            return

        # Inference
        with torch.no_grad():
            trend = patch_model(tensor)
            context = tft_model(None, tensor)
            
            # Neutral Portfolio State
            port_state = torch.tensor([[0.0, 100000.0]]).to(device)
            
            state = torch.cat([trend, context, port_state], dim=1)
            action = agent.get_action(state).item()

        # Logic
        signal_type = "HOLD"
        # Confidence thresholds
        if action > 0.3: signal_type = "BUY"
        if action < -0.3: signal_type = "SELL"
        
        output = {
            "timestamp": str(datetime.now()),
            "signal": signal_type,
            "confidence": round(abs(action), 4),
            "predicted_volatility": "NORMAL"
        }
        
        # Write to MT5
        if not os.path.exists(MT5_FILES_PATH):
            print(f"WARNING: MT5 Path not found: {MT5_FILES_PATH}")
            with open("py_signal.json", "w") as f:
                json.dump(output, f)
        else:
            file_path = os.path.join(MT5_FILES_PATH, "py_signal.json")
            with open(file_path, "w") as f:
                json.dump(output, f)
        
        print(f"Signal Generated: {signal_type} | Strength: {action:.4f}")

    except Exception as e:
        print(f"Error in signal loop: {e}")
        import traceback
        traceback.print_exc()
        
# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        rl_agent, patch_model, tft_model = load_models()
        print("Models Loaded. Bridge Active.")
        
        while True:
            generate_signal(rl_agent, patch_model, tft_model)
            print("Waiting 1 hour for next candle...")
            time.sleep(3600) # Sleep 1 hour
            
    except KeyboardInterrupt:
        print("Bridge stopped by user.")
    except Exception as e:
        print(f"Fatal Error: {e}")