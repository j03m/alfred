from logging import exception

import pandas as pd
import os
import argparse
from alfred.data import AlphaDownloader
from alfred.metadata import TickerCategories
from pathlib import Path
import json

def create_news_indicators(symbol, news_dir="./news", required_rel=0.7):
    ticker_dir = Path(news_dir) / symbol
    data = []
    # Check if the ticker directory exists
    if not ticker_dir.exists() or not ticker_dir.is_dir():
        return None

    # Loop through each date directory in the ticker directory
    for date_dir in ticker_dir.iterdir():
        if date_dir.is_dir():  # Check if it's a directory (date in YYYYMMDD format)
            date = date_dir.name  # e.g., '20241031'

            # Initialize lists to collect sentiment and outlook for the date
            sentiments = []
            outlooks = []

            # Loop through each JSON file in the date directory
            for file_path in date_dir.glob("*.json"):
                with open(file_path, 'r') as file:
                    data_json = json.load(file)

                    # Only consider files with relevance >= 0.7
                    if data_json.get("relevance", 0) >= required_rel:
                        sentiments.append(data_json.get("sentiment", 0))
                        outlooks.append(data_json.get("outlook", 0))

            # Calculate the mean sentiment and outlook if we have relevant data
            if sentiments and outlooks:
                mean_sentiment = sum(sentiments) / len(sentiments)
                mean_outlook = sum(outlooks) / len(outlooks)

                # Append the data as a row in the list
                data.append({
                    "Date": date,
                    "mean_sentiment": mean_sentiment,
                    "mean_outlook": mean_outlook
                })

    if len(data) == 0:
        return None

    # Convert the list to a DataFrame
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

PROCESSED_TICKERS_FILE = 'data/processed_tickers.json'

def load_processed_tickers():
    if os.path.exists(PROCESSED_TICKERS_FILE):
        with open(PROCESSED_TICKERS_FILE, 'r') as f:
            return json.load(f)
    return []  # Return an empty list if the file doesn't exist

def save_processed_tickers(processed_tickers):
    with open(PROCESSED_TICKERS_FILE, 'w') as f:
        json.dump(processed_tickers, f)

def main(symbols_file, data_dir, test_symbol):
    if test_symbol is not None:
        symbols = [test_symbol]
    else:
        ticker_categories = TickerCategories(symbols_file)
        symbols = ticker_categories.get(["training", "evaluation"])

    alpha = AlphaDownloader()

    processed_already = load_processed_tickers()

    for symbol in symbols:
        if symbol in processed_already:
            print(f"skipping {symbol}")
            continue

        print(f"Processing {symbol}")
        _, quarterly_earnings = alpha.earnings(symbol)
        margins = alpha.margins(symbol)
        insiders = alpha.get_normalized_insider_transactions(symbol)
        df_news = create_news_indicators(symbol)
        price_file_path = os.path.join(data_dir, f"{symbol}.csv")

        if os.path.exists(price_file_path):
            df_prices = pd.read_csv(price_file_path)
            quarterly_earnings.index = pd.to_datetime(quarterly_earnings["Date"])
            quarterly_earnings = quarterly_earnings.drop(columns=['Date'])
            df_prices.index = pd.to_datetime(df_prices['Date'])
            df_prices = df_prices.drop(columns=['Date'])

            margins.index = pd.to_datetime(margins['Date'])
            margins = margins.drop(columns=['Date'])

            # Merging data
            df_combined = df_prices.join(quarterly_earnings, how='outer')
            df_combined = df_combined.join(margins, how='outer')

            # forward will earnings - prevents lookahead
            df_combined.ffill(inplace=True)

            # after forward fill, we may still have na if price goes back further than earnings, treat these as 0
            df_combined.fillna(0, inplace=True)

            if insiders is not None:
                df_combined = df_combined.join(insiders, how='outer')
                df_combined.fillna(0, inplace=True)
            else:
                df_combined["insider_acquisition"] = 0
                df_combined["insider_disposal"] = 0

            if df_news is not None:
                df_combined = df_combined.join(df_news, how='outer')
                df_combined.fillna(0, inplace=True)
            else:
                df_combined["mean_sentiment"] = 0
                df_combined["mean_outlook"] = 0

            # Writing to CSV
            output_path = os.path.join(data_dir, f"{symbol}_fundamentals.csv")
            df_combined.to_csv(output_path)
            print(f"Written combined data to {output_path}")
        else:
            raise exception(f"{symbol} file not found. Did you cache prices?")

        processed_already.append(symbol)
        save_processed_tickers(processed_already)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process stock fundamentals and pricing data.")
    parser.add_argument("--symbol-file", type=str, help="Path to the CSV file containing stock symbols")
    parser.add_argument("--symbol", type=str, help="one symbol for testing")
    parser.add_argument("--data-dir", default="./data", type=str,
                        help="Directory to look for pricing data and save output")

    args = parser.parse_args()
    main(args.symbol_file, args.data_dir, args.symbol)
