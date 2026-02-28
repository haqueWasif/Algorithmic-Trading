import asyncio
import logging
from exchange_manager import ExchangeManager

logging.basicConfig(level=logging.INFO)

async def test_balance_fetching():
    print("Starting balance fetching test...")

    # Dummy configuration for testing
    exchanges_config = {
        "kraken": {
            "api_key": "JzqCwT5AZbD7+dyyBLEeDJiiDNQ3W7ci++8uTUtmnAyIfdATPcRLo2+5",
            "secret": "f+9yUE7U/ML6KbekpX0HXaygw7nTX26n0kOp2WLgbkVMRygO/DjdQl6YV7nXHwvb2bIB3kyVLXZb1cwfFZJ/jw==",
            "sandbox": True
        },
        "bybit": {
            "api_key": "IG3Gxr8rl94IV1SZeu",
            "secret": "NdIT4EW5DZbqpBtaveyubPM1nWCRbRvKmzGb",
            "sandbox": True
        }
    }

    exchange_manager = ExchangeManager(exchanges_config)
    await exchange_manager.initialize_exchanges()

    print("\nTesting get_balance for USDT on Kraken...")
    kraken_usdt_balance = await exchange_manager.get_balance("kraken", "USDT")
    print(f"Kraken USDT Balance: {kraken_usdt_balance}")
    assert isinstance(kraken_usdt_balance, float), "Kraken USDT balance should be a float"

    print("\nTesting get_balance for BTC on Bybit...")
    bybit_btc_balance = await exchange_manager.get_balance("bybit", "BTC")
    print(f"Bybit BTC Balance: {bybit_btc_balance}")
    assert isinstance(bybit_btc_balance, float), "Bybit BTC balance should be a float"

    print("\nTesting get_all_balances...")
    all_balances = await exchange_manager.get_all_balances()
    print(f"All Balances: {all_balances}")
    assert isinstance(all_balances, dict), "All balances should be a dictionary"
    assert "kraken" in all_balances, "Kraken should be in all_balances"
    assert "bybit" in all_balances, "Bybit should be in all_balances"

    print("\nBalance fetching tests completed.")

if __name__ == "__main__":
    asyncio.run(test_balance_fetching())

