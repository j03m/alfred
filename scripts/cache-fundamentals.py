from logging import exception

import os
import argparse

import json

import pandas as pd

from alfred.data import AlphaDownloader, NewsDb, EdgarDb
from alfred.metadata import TickerCategories

PROCESSED_TICKERS_FILE = 'data/processed_tickers.json'

news_db = NewsDb()
edgar_db = EdgarDb()

def create_news_indicators(symbol, required_rel=0.7):
   return news_db.get_summary(symbol, required_rel)

def get_institutional_ownership(symbol):
   return edgar_db.get_filings(symbol)

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
    bad_tickers = []
    for symbol in symbols:
        if symbol in processed_already and not args.force:
            print(f"skipping {symbol}")
            continue

        print(f"Processing {symbol}")
        quarterly_earnings = alpha.earnings(symbol)
        margins = alpha.margins(symbol)
        insiders = alpha.get_normalized_insider_transactions(symbol)
        df_news = create_news_indicators(symbol)
        df_inst_ownership = get_institutional_ownership(symbol)

        price_file_path = os.path.join(data_dir, f"{symbol}.csv")


        # TODO: could all these joins use merge_asof instead?
        if os.path.exists(price_file_path):
            if quarterly_earnings is None:
                bad_tickers.append(symbol)
                continue

            df_prices = pd.read_csv(price_file_path)
            if len(df_prices) == 0:
                bad_tickers.append(symbol)
                continue

            close_zeros = len(df_prices[df_prices['Close'] == 0])
            assert (close_zeros == 0)

            quarterly_earnings.index = pd.to_datetime(quarterly_earnings["Date"])
            quarterly_earnings = quarterly_earnings.drop(columns=['Date'])
            quarterly_earnings.sort_index(inplace=True)
            quarterly_earnings = quarterly_earnings[~quarterly_earnings.index.duplicated(keep='first')]

            df_prices.index = pd.to_datetime(df_prices['Date'])
            df_prices = df_prices.drop(columns=['Date'])
            df_prices.sort_index(inplace=True)

            margins.index = pd.to_datetime(margins['Date'])
            margins = margins.drop(columns=['Date'])
            margins = margins[~margins.index.duplicated(keep='first')]
            margins.sort_index(inplace=True)

            # before we combine anything, prices is the main date range, we need to trim everything
            # else to align with prices otherwise combine will give us 0's for Close price etc
            # Determine date range from prices
            # quarterly_earnings, margins = reindex_dataframes(df_prices, quarterly_earnings, margins)

            # Merging data
            df_combined = pd.merge_asof(df_prices, quarterly_earnings, left_index=True, right_index=True)
            df_combined = pd.merge_asof(df_combined, margins, left_index=True, right_index=True)

            if insiders is not None:
                df_combined = pd.merge_asof(df_combined, insiders, left_index=True, right_index=True)
            else:
                df_combined["insider_acquisition"] = 0
                df_combined["insider_disposal"] = 0

            if df_news is not None:
                df_combined = pd.merge_asof(df_combined, df_news,left_index=True, right_index=True)
                df_combined.fillna(0, inplace=True)
            else:
                df_combined["mean_sentiment"] = 0
                df_combined["mean_outlook"] = 0

            # institutional ownership
            df_combined = pd.merge_asof(df_combined, df_inst_ownership, left_index=True, right_index=True)

            close_zeros = len(df_combined[df_combined['Close'] == 0])
            assert (close_zeros == 0)

            # Writing to CSV
            output_path = os.path.join(data_dir, f"{symbol}_fundamentals.csv")
            df_combined.to_csv(output_path)
            print(f"Written combined data to {output_path}")
        else:
            raise exception(f"{symbol} file not found. Did you cache prices?")

        processed_already.append(symbol)
        save_processed_tickers(processed_already)

    if test_symbol is None:
        ticker_categories.purge(bad_tickers)
        ticker_categories.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process stock fundamentals and pricing data.")
    parser.add_argument("--symbol-file", type=str, help="Path to the CSV file containing stock symbols")
    parser.add_argument("--symbol", type=str, help="one symbol for testing")
    parser.add_argument("--data-dir", default="./data", type=str,
                        help="Directory to look for pricing data and save output")
    parser.add_argument("--force", action="store_true", help="force")

    args = parser.parse_args()
    main(args.symbol_file, args.data_dir, args.symbol)
