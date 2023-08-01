#!/usr/bin/env python3
import yfinance as yf
import time
import pandas as pd
import pandas_datareader.data as web
import random
import argparse
from machine_learning_finance import download_ticker_list

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbols", help="Symbols to use (default: SPY), separated by comma")
parser.add_argument("-f", "--symbol-file", help="Load symbols from a file")
parser.add_argument("-o", "--output-dir", default="./data", help="Output directory (default: ./data)")
parser.add_argument("-rs", "--random-spys", type=int, default=None, help="Number of random stocks to select from SPY")
parser.add_argument("-sgl", "--save-good-list", action="store_true", help="Save the symbol list")

args = parser.parse_args()


def rando_spys(num):
    sp_assets = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    assets = sp_assets['Symbol'].str.replace('.', '-').tolist()

    # Select num random symbols from SPY
    random_symbols = random.sample(assets, num)

    return random_symbols


def load_symbols_from_file(file):
    return pd.read_csv(file)["Symbols"].tolist()


symbols = []
if args.symbols is not None:
    symbols += args.symbols.split(',')

if args.symbol_file is not None:
    symbols += load_symbols_from_file(args.symbol_file)

if args.random_spys is not None:
    symbols += rando_spys(args.random_spys)

symbols = list(set(symbols))
bad_symbols = download_ticker_list(symbols)

if bad_symbols is None:
    final_symbols = set(symbols)
else:
    final_symbols = set(symbols) - set(bad_symbols)

# Create a pandas DataFrame with symbols as data and "Symbols" as column name
df = pd.DataFrame({'Symbols': list(final_symbols)})

for symbol in symbols:
    ticker_obj = yf.download(tickers=symbol, interval="1d")
    symbol_df = pd.DataFrame(ticker_obj)
    symbol_df.to_csv(f"./data/{symbol}.csv")

# Save the DataFrame to a CSV file named "symbols.csv"
df.to_csv('./data/symbols.csv', index=False)
