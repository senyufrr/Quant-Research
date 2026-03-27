import alpaca_trade_api as tradeapi

API_KEY =  -------------------
SECRET_KEY = -----------------

BASE_URL = 'https://paper-api.alpaca.markets'

print("Attempting to connect to the stock market...")

try:
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

    account = api.get_account()
    print(f"SUCCESS! Connection established.")
    print(f"Account Status: {account.status}")
    print(f"My Fake Buying Power: ${account.buying_power}")
    
    ticker = 'AAPL'
    quote = api.get_latest_quote(ticker)
    print(f"\nLive Data for {ticker}:")
    print(f"Asking Price right now: ${quote.ask_price}")
    
except Exception as e:
    print(f"Uh oh, something went wrong: {e}")
