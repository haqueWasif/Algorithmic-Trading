# FVG Trading Strategy Bot

A Python-based trading bot that identifies Fair Value Gaps (FVGs) in financial markets using MetaTrader 5 (MT5) and sends trading signals via Telegram. The bot supports multi-timeframe analysis and interactive visualization of FVGs with a candlestick chart.

## Features
- **FVG Detection**: Identifies bullish and bearish FVGs across multiple timeframes (M1, M5, M15, H1).
- **Signal Generation**: Generates Buy/Sell signals after the FVG-forming candle closes, with stop-loss and take-profit levels.
- **Real-Time Monitoring**: Fetches live OHLCV data from MT5 and updates signals periodically.
- **Interactive Plotting**: Displays candlestick charts with FVG zones for multiple timeframes, navigable via a slider.
- **Telegram Notifications**: Sends trading signals to a specified Telegram chat.
- **Secure Configuration**: Stores sensitive credentials (MT5 login, Telegram token) in environment variables.

## Prerequisites
- **Python 3.8+**
- **MetaTrader 5** installed with a valid account (demo or live).
- **Telegram Bot**: A bot token and chat ID for notifications.
- **Git** (optional, for cloning the repository).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/haqueWasif/FVG-Trading-Strategy-Bot.git
   cd FVG-Trading-Strategy-Bot
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` File**:
   - In the project root, create a `.env` file and add your credentials:
     ```env
     MT5_LOGIN=your_mt5_login
     MT5_PASSWORD=your_mt5_password
     MT5_SERVER=your_mt5_server
     TELEGRAM_BOT_TOKEN=your_telegram_bot_token
     TELEGRAM_CHAT_ID=your_telegram_chat_id
     ```
   - Replace `your_...` with your actual MT5 and Telegram credentials.

5. **Ensure MetaTrader 5 Path**:
   - Update `MT5_TERMINAL_PATH` in `config/config.py` if your MT5 terminal is installed in a different location:
     ```python
     MT5_TERMINAL_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
     ```

## Project Structure
```
fvg-trading-bot/
├── src/                    # Core source code
│   ├── __init__.py         # Empty, marks src as a package
│   ├── main.py             # Main script to run the bot
│   ├── analysis/           # Trading strategy logic
│   │   ├── __init__.py     # Empty, marks analysis as a package
│   │   ├── fvg_analysis.py # FVG detection and signal generation
│   ├── plotting/           # Plotting-related code
│   │   ├── __init__.py     # Empty, marks plotting as a package
│   │   ├── plotting.py     # Candlestick and FVG plotting
│   ├── utils/              # Utility functions
│   │   ├── __init__.py     # Empty, marks utils as a package
│   │   ├── mt5_utils.py    # MT5 connection and data fetching
│   │   ├── telegram_utils.py # Telegram notifications
├── config/                 # Configuration files
│   ├── __init__.py         # Empty, marks config as a package
│   ├── config.py           # Settings (symbol, timeframes, etc.)
├── docs/                   # Documentation and assets
│   ├── README.md           # This file
├── tests/                  # Unit tests (optional)
│   ├── __init__.py         # Empty, marks tests as a package
├── .env                    # Sensitive credentials (not tracked)
├── .gitignore              # Git ignore file
├── requirements.txt        # Python dependencies
├── LICENSE                 # License file
```

## Usage

1. **Run the Bot**:
   ```bash
   python -m src.main
   ```
   - Run from the project root (`FVG-Trading-Strategy-Bot/`) to ensure the `src` package is found.
   - The bot initializes MT5, fetches data, detects FVGs, and generates signals.
   - A plot window opens, showing candlestick charts with FVG zones for M1, M5, M15, and H1 timeframes.
   - Use the slider to navigate historical candles.
   - Signals are sent to your Telegram chat when conditions are met.

2. **Monitor Logs**:
   - Check the console for logs on data fetching, FVG detection, and signal generation.
   - Example log:
     ```
     2025-04-13 10:00:01 - INFO - Detected 3 FVG zones for M5
     2025-04-13 10:00:01 - INFO - Bullish signal generated at 2025-04-13 10:00:00 for FVG 1.0800-1.0805
     ```

3. **Stop the Bot**:
   - Press `Ctrl+C` to stop the bot gracefully.

## Configuration
Edit `config/config.py` to customize:
- **SYMBOL**: Trading instrument (default: `BTCUSD`).
- **LIMIT**: Number of historical candles to fetch (default: `1000`).
- **DISPLAY_CANDLES**: Candles shown in plots (default: `100`).
- **PLOT_TIMEFRAMES**: Timeframes to analyze (default: `['M1', 'M5', 'M15', 'H1']`).

## Requirements
Create a `requirements.txt` with:
```
MetaTrader5
pandas
matplotlib
mplfinance
python-telegram-bot
python-dotenv
```

Install with:
```bash
pip install -r requirements.txt
```

## Notes
- **Security**: Never commit `.env` to version control. It’s ignored in `.gitignore`.
- **Data**: Ensure MT5 is running or provides data for the chosen `SYMBOL` and timeframes.
- **Plotting**: Plots are non-blocking, allowing real-time signal generation.
- **Signals**: Signals are generated only after the FVG candle closes, within a 10-candle window.
- **Logging**: Debug logs help diagnose issues (e.g., no signals, MT5 connection errors).

## Troubleshooting
- **ModuleNotFoundError: No module named 'src'**:
  - Ensure you’re running `python -m src.main` from the project root (`FVG-Trading-Strategy-Bot/`).
  - Verify `__init__.py` files exist in `src/`, `src/analysis/`, `src/plotting/`, `src/utils/`, and `config/`.
  - Check that `src/utils/mt5_utils.py` and `src/utils/telegram_utils.py` exist.
- **ImportError: cannot import name 'MT5_LOGIN' from 'config'**:
  - Ensure `src/utils/mt5_utils.py` and `src/utils/telegram_utils.py` import from `config.config` (e.g., `from config.config import MT5_LOGIN`).
  - Verify `config/config.py` defines all required variables (`MT5_LOGIN`, `TELEGRAM_BOT_TOKEN`, etc.).
- **No Signals**:
  - Check logs for `Empty DataFrame` or `No FVG zones`.
  - Verify MT5 connection and data for `SYMBOL`.
  - Ensure `src/analysis/fvg_analysis.py` conditions match market behavior.
- **Plot Issues**:
  - Confirm `matplotlib` backend supports interactive mode (e.g., `Qt5Agg`).
  - Try adding `import matplotlib; matplotlib.use('Qt5Agg')` at the top of `src/main.py` if plots don’t appear.
- **Telegram Errors**:
  - Validate `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.
  - Check network connectivity.
- **MT5 Connection Errors**:
  - Ensure MetaTrader 5 is installed and running.
  - Verify `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` in `.env` are correct.
  - Check `MT5_TERMINAL_PATH` in `config/config.py`.

## Contributing
Feel free to open issues or submit pull requests for improvements, such as:
- Additional signal conditions.
- Support for more symbols or timeframes.
- Enhanced plotting features.

## License
MIT License

Copyright (c) 2025 Wasiful Haque

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Disclaimer
This bot is for educational purposes only. Trading involves risk, and past performance is not indicative of future results. Use at your own discretion.
```
