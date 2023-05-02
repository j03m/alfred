#!/usr/bin/env python3
import yfinance as yf
import time
import pandas as pd
import pandas_datareader.data as web

# Download the symbols for the Nasdaq exchange
nasdaq_symbols = web.get_nasdaq_symbols()

# Download the symbols for the AMEX exchange
amex_symbols = web.get_amex_symbols()

# Download the symbols for the NYSE exchange
nyse_symbols = web.get_nyse_symbols()

# Concatenate the symbols from each exchange into a single DataFrame
all_tickers = pd.concat([nasdaq_symbols, amex_symbols, nyse_symbols])

def get_dividend_stats(ticker):
    # Retrieve the historical data for the ticker
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="5y")

    # Calculate the dividend yield for each year
    dividend_yield = hist_data['Dividends'] / hist_data['Close']

    # Calculate the statistics for the dividend yield
    stats = {
        "ticker": ticker,
        "mean": dividend_yield.mean(),
        "max": dividend_yield.max(),
        "min": dividend_yield.min(),
        "median": dividend_yield.median(),
        "avg": dividend_yield.sum() / len(dividend_yield),
        "std": dividend_yield.std()
    }

    return stats


# Loop through the tickers and get the dividend statistics
stats_list = []
total_tickers = len(all_tickers)
for index, ticker in enumerate(all_tickers):
    try:
        # Add a delay to avoid overloading the yfinance API
        time.sleep(0.25)

        # Get the dividend statistics for the ticker
        stats = get_dividend_stats(ticker)
        stats_list.append(stats)

        # Print status update
        print(f'[{index + 1}/{total_tickers}] Processed {ticker}')

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Create a DataFrame with the dividend statistics
df = pd.DataFrame(stats_list)

# Save the DataFrame to a CSV file
df.to_csv("./data/dividend_stats.csv", index=False)
