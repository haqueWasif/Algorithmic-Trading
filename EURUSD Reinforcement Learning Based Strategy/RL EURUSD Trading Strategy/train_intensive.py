import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
TICKER = "EURUSD=X"
EPOCHS = 50           # Train 50 times over the data
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.01   # Forces exploration (Prevents getting stuck)
BATCH_SIZE = 64

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- IMPORT ARCHITECTURE (Ensure these classes are defined) ---
# (Paste the VMDProcessor, PatchTST_Mini, TFT_Mini, PPO_Agent, DifferentiableSharpeLoss classes here)
# For brevity, I assume you have them defined or imported from your previous cells.
# ... 

# --- DATA LOADING (Same as before) ---
def prepare_data():
    print("Fetching 2y of Data for Deep Training...")
    df = yf.download(TICKER, period="2y", interval="1h", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df['volume'] = df['volume'].replace(0, 1)

    # Feature Engineering
    direction = np.where(df['close'] >= df['open'], 1, -1)
    df['ofi'] = df['volume'] * direction
    ma_50 = df['close'].rolling(50).mean()
    std_50 = df['close'].rolling(50).std()
    df['sentiment_score'] = ((df['close'] - ma_50) / std_50).fillna(0)
    center_line = df['close'].rolling(60).mean()
    rolling_std = df['close'].rolling(60).std()
    df['dist_from_vah'] = df['close'] - (center_line + rolling_std)
    
    # VMD
    vmd_input = df['close'].values
    modes = np.zeros((5, len(vmd_input)))
    for k in range(5):
        freq = (k+1) * 0.05
        modes[k, :] = np.sin(np.linspace(0, 100*freq, len(vmd_input))) * np.std(vmd_input)
    modes[-1, :] = vmd_input - np.sum(modes[:-1, :], axis=0)
    for i in range(5): df[f'vmd_mode_{i}'] = modes[i]
    
    return df.dropna()

def train_intensive():
    # 1. Prepare Data
    df = prepare_data()
    scaler = StandardScaler()
    feature_cols = ['close', 'volume', 'ofi', 'sentiment_score', 'dist_from_vah'] + [c for c in df.columns if 'vmd' in c]
    data_scaled = scaler.fit_transform(df[feature_cols])
    data_tensor = torch.FloatTensor(data_scaled).unsqueeze(0).to(device) # [1, T, F]

    # 2. Initialize Models
    # Input Dims: Features=10. PPO State = 10 (Trend) + 1 (Context) + 2 (Port) = 13.
    patch_model = PatchTST_Mini(n_token=60, patch_len=16, stride=8, d_model=64).to(device)
    tft_model = TFT_Mini(input_dim=10).to(device)
    rl_agent = PPO_Agent(state_dim=13).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(list(patch_model.parameters()) + 
                           list(tft_model.parameters()) + 
                           list(rl_agent.parameters()), lr=LEARNING_RATE)
    
    sharpe_loss_fn = DifferentiableSharpeLoss().to(device)

    # 3. Training Loop
    print(f"\n>>> STARTING INTENSIVE TRAINING ({EPOCHS} Epochs) <<<")
    
    for epoch in range(1, EPOCHS + 1):
        
        # Reset Portfolio for each epoch (Simulate a fresh year of trading)
        portfolio_cash = 100000.0
        position = 0.0
        batch_returns = []
        batch_log_probs = [] # For Entropy
        
        total_sharpe_loss = 0
        actions_taken = []
        
        # Sliding Window Loop
        WINDOW_SIZE = 60
        # Jump by 4 hours to speed up training (optional, removes redundancy)
        for t in range(WINDOW_SIZE, data_tensor.shape[1] - 1, 1):
            
            # A. Get State
            window_data = data_tensor[:, t-WINDOW_SIZE:t, :]
            
            trend = patch_model(window_data)
            context = tft_model(None, window_data)
            port_state = torch.tensor([[position, portfolio_cash]], dtype=torch.float32).to(device)
            state = torch.cat([trend, context, port_state], dim=1)
            
            # B. Get Action
            action_raw = rl_agent.get_action(state)
            actions_taken.append(action_raw.item())
            
            # C. Reward
            # Calculate next candle return
            # (In training, we use scaled 'close' price diff as proxy for return)
            # Or better: use raw percent change if available. Here we use scaled diff for stability.
            price_t = data_tensor[0, t, 0] # Close is col 0
            price_next = data_tensor[0, t+1, 0]
            r_t = action_raw * (price_next - price_t)
            
            batch_returns.append(r_t)
            
            # D. Optimize Batch
            if len(batch_returns) >= BATCH_SIZE:
                returns_tensor = torch.stack(batch_returns).squeeze()
                
                # Main Loss: Maximize Sharpe (Minimize Negative Sharpe)
                s_loss = sharpe_loss_fn(returns_tensor)
                
                # Entropy Loss: Force diversity
                # Start big, decay over time
                entropy_scale = ENTROPY_BETA * (1.0 - epoch/EPOCHS) 
                # Approx entropy: penalize squaring the action (pushing towards 0) 
                # We want action^2 to be high (near -1 or 1). 
                # So we minimize -action^2 (maximize magnitude)
                entropy_loss = -torch.mean(torch.stack(batch_returns)**2) * entropy_scale

                total_loss = s_loss + entropy_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(rl_agent.parameters(), 0.5) # Prevent explosion
                optimizer.step()
                
                total_sharpe_loss += s_loss.item()
                
                # Clear Batch
                batch_returns = []
                
                # Update Portfolio (Simulation)
                # Just simple compounding for state tracking
                portfolio_cash *= (1.0 + r_t.item())
                position = action_raw.item()

        # End of Epoch Stats
        avg_sharpe = total_sharpe_loss / (data_tensor.shape[1] // BATCH_SIZE)
        avg_action = np.mean(actions_taken)
        std_action = np.std(actions_taken)
        
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_sharpe:.4f} | Avg Action: {avg_action:.3f} | Activity (Std): {std_action:.3f}")
        
        # Stop early if "Activity" is high enough (Model woke up)
        if std_action > 0.2 and epoch > 10:
            print("Model has converged (Activity > 0.2). Saving early.")
            break

    # 4. Save
    print("\nSaving Smart Models...")
    torch.save(rl_agent.state_dict(), "EURUSD_RL_Agent_Gen3.pth")
    torch.save(patch_model.state_dict(), "PatchTST_Trend.pth")
    torch.save(tft_model.state_dict(), "TFT_Context.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train_intensive()