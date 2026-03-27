import alpaca_trade_api as tradeapi

# 1. Put your Alpaca keys here (Keep the single quote marks around them!)
API_KEY = PKLY6ITX3U2C5FA4ZXDOKDFPNA
SECRET_KEY = HBhqhAT8ZBLz6RMprmWVn4rgsXaSPiJzo6LNuVeens7y

# 2. Tell the bot to use the Paper Trading (fake money) URL
BASE_URL = 'https://paper-api.alpaca.markets'

print("Attempting to connect to the stock market...")

try:
    # 3. Establish the connection
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

    # 4. Check our bank account
    account = api.get_account()
    print(f"SUCCESS! Connection established.")
    print(f"Account Status: {account.status}")
    print(f"My Fake Buying Power: ${account.buying_power}")
    
    # Let's pull the live price of Apple stock just to prove it works
    ticker = 'AAPL'
    quote = api.get_latest_quote(ticker)
    print(f"\nLive Data for {ticker}:")
    print(f"Asking Price right now: ${quote.ask_price}")
    
except Exception as e:
    print(f"Uh oh, something went wrong: {e}")