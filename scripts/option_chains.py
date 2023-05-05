#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np

def get_option_chain(ticker):
    stock = yf.Ticker(ticker)
    try:
        option_chain = stock.option_chain(stock.options[0])
        return option_chain.calls, option_chain.puts
    except IndexError:
        print(f"No options data available for {ticker}")
        return None, None

# Get S&P 500 components
table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = table['Symbol'].tolist()
for ticker in tickers:
    print(get_option_chain(ticker))

