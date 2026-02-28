import asyncio
import websockets

async def test():
    try:
        async with websockets.connect("wss://stream.binance.com:9443/ws/btcusdt@trade") as ws:
            print("Connected!")
            print(await ws.recv())
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
