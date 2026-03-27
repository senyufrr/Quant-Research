import yfinance as yf
import pandas as pd

def calculate_rsi(ticker, window=14):
    # Fix: Added 'multi_level_index=False' to flatten the columns immediately
    data = yf.download(ticker, period="3mo", interval="1d", multi_level_index=False)
    
    if data.empty:
        return 0.0

    # Ensure we are looking at a simple 'Close' column
    close_prices = data['Close']
    
    # Calculate Price Change
    delta = close_prices.diff()
    
    # Separate Gains and Losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Calculate RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # .item() converts the single-value Series into a regular number (float)
    return rsi.iloc[-1].item()

# List of stocks you are analyzing at Chola
stocks = ['CHOLAHLDNG.NS', 'SHRIRAMFIN.NS'] # Added Chola's holding co too

print("\n--- LIVE MOMENTUM SCANNER ---")
for s in stocks:
    current_rsi = calculate_rsi(s)
    status = "OVERBOUGHT (Sell?)" if current_rsi > 70 else "OVERSOLD (Buy?)" if current_rsi < 30 else "NEUTRAL"
    print(f"{s}: RSI is {current_rsi:.2f} -> {status}")


import yfinance as yf
import pandas as pd

def get_signals(ticker):
    # Download 1 year of data to get a solid 200-day average
    data = yf.download(ticker, period="1y", interval="1d", multi_level_index=False)
    
    if data.empty:
        return None

    # Calculate Moving Averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Get the latest values
    current_price = data['Close'].iloc[-1].item()
    ma50 = data['MA50'].iloc[-1].item()
    ma200 = data['MA200'].iloc[-1].item()
    
    # Logic for Trend
    if ma50 > ma200:
        trend = "BULLISH (Golden Cross Zone)"
    else:
        trend = "BEARISH (Death Cross Zone)"
        
    return {
        "price": current_price,
        "ma50": ma50,
        "ma200": ma200,
        "trend": trend
    }

stocks = ['SHRIRAMFIN.NS', 'CHOLAHLDNG.NS']

print("\n--- QUANT TREND ANALYSIS ---")
for s in stocks:
    stats = get_signals(s)
    if stats:
        print(f"\n{s}:")
        print(f"  Current Price: {stats['price']:.2f}")
        print(f"  50-Day MA: {stats['ma50']:.2f}")
        print(f"  200-Day MA: {stats['ma200']:.2f}")
        print(f"  Status: {stats['trend']}")
    percent_below_ma200 = ((stats['ma200'] - stats['price']) / stats['ma200']) * 100
    print(f"  Distance from MA200: {percent_below_ma200:.2f}% Below")
