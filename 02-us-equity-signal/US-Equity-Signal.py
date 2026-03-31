import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd

# 1. Credentials
API_KEY = ''
SECRET_KEY = ''
BASE_URL = 'https://paper-api.alpaca.markets'

ticker = input("Enter ticker symbol: ")

try:
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
    
    # 2. Get Data (Need at least 1 year for SMA200)
    df = yf.download(ticker, period='1y', interval='1d', multi_level_index=False)
    
    # --- Moving Average Calculations ---
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # --- RSI Calculation ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + (gain/loss)))
    
    curr_p = df['Close'].iloc[-1].item()
    sma50 = df['SMA50'].iloc[-1].item()
    sma200 = df['SMA200'].iloc[-1].item()
    curr_rsi = rsi.iloc[-1].item()

    print(f"\n--- {ticker} Analysis ---")
    print(f"Price: ${curr_p:.2f} | RSI: {curr_rsi:.2f}")
    print(f"SMA50: ${sma50:.2f} | SMA200: ${sma200:.2f}")

    # 3. ADVANCED DIRECTIONAL LOGIC
    # LONG: Price is cheap (RSI < 35) AND we are in a long-term uptrend (SMA50 > SMA200)
    if curr_rsi < 35 and sma50 > sma200:
        print("🟢 SIGNAL: Buy the Dip in an Uptrend. Going LONG.")
        api.submit_order(symbol=ticker, qty=10, side='buy', type='market', time_in_force='gtc')

    # SHORT: Price is over-extended (RSI > 65) AND we are in a long-term downtrend (SMA50 < SMA200)
    elif curr_rsi > 65 and sma50 < sma200:
        print("🔴 SIGNAL: Short the Rip in a Downtrend. Going SHORT.")
        api.submit_order(symbol=ticker, qty=10, side='sell', type='market', time_in_force='gtc')

    else:
        print("⚪ NO SIGNAL: Trend and Momentum are not aligned.")

except Exception as e:
    print(f"Error: {e}")
