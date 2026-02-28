import ccxt
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def test_binance_api():
    binance_api_key = os.getenv("BINANCE_API_KEY")
    binance_secret = os.getenv("BINANCE_SECRET")
    binance_sandbox = os.getenv("BINANCE_SANDBOX", "false").lower() == "true"

    print(f"Testing Binance API with sandbox mode: {binance_sandbox}")
    print(f"API Key: {binance_api_key}")
    print(f"Secret: {binance_secret}")

    exchange = ccxt.binance({
        'apiKey': binance_api_key,
        'secret': binance_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True
        }
    })

    if binance_sandbox:
        try:
            exchange.set_sandbox_mode(True)
            print("Binance set to SANDBOX mode.")
        except Exception as e:
            print(f"Error setting sandbox mode for Binance: {e}")
            return

    try:
        print("Fetching Binance balance...")
        balance = await exchange.fetch_balance()
        print("Binance balance fetched successfully:")
        for currency, bal_data in balance['free'].items():
            if bal_data > 0:
                print(f"  {currency}: {bal_data}")
    except Exception as e:
        print(f"Failed to fetch Binance balance: {e}")


if __name__ == "__main__":
    asyncio.run(test_binance_api())


