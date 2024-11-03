import pandas as pd
from datetime import datetime, timedelta
from alfred.metadata import TickerCategories

def load_symbols_from_file(file):
    tickers = TickerCategories(file)
    return tickers.get(["training", "evaluation"])

def calculate_30_day_returns(ticker):
    # Load the price data for the ticker
    df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'], index_col='Date')
    df = df.sort_index()  # Ensure it's sorted by date
    df['30d_return'] = df['Close'].pct_change(30)  # Calculate 30-day return
    return df[['30d_return']]

# Load tickers and create ticker-to-ID mapping
tickers = load_symbols_from_file("metadata/ticker-categorization.json")
ticker_to_id = {ticker: idx for idx, ticker in enumerate(tickers, start=1)}

# Define the date range for two years
end_date = datetime.today()
start_date = end_date - timedelta(days=2*365)

# Calculate returns for each ticker
returns = {}
for ticker in tickers:
    df = calculate_30_day_returns(ticker)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    returns[ticker] = df

# Combine all tickers' returns into a single DataFrame
combined_df = pd.concat(returns, axis=1)
combined_df.columns = tickers  # Adjust to handle multi-level columns if needed

# Sort tickers by return for each date and replace tickers with IDs
ranked_by_date = {}
for date in combined_df.index:
    ranked_tickers = combined_df.loc[date].dropna().sort_values(ascending=False)
    ranked_ids = [ticker_to_id[ticker] for ticker in ranked_tickers.index]  # Map tickers to IDs
    ranked_by_date[date] = ranked_ids  # Store list of IDs ordered by 30-day return

# Convert to DataFrame and save
ranked_df = pd.DataFrame.from_dict(ranked_by_date, orient='index')
ranked_df.to_csv("results/returns-ranked.csv")

# Save ticker-to-ID mapping to CSV
ticker_id_df = pd.DataFrame(list(ticker_to_id.items()), columns=["Ticker", "ID"])
ticker_id_df.to_csv("results/ticker_ids.csv", index=False)
