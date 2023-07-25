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
parser.add_argument("-t", "--tail", type=int, default=2500, help="Tail size (default: 2500)")
parser.add_argument("-o", "--output-dir", default="./data", help="Output directory (default: ./data)")
parser.add_argument("-rs", "--random-spys", type=int, default=None, help="Number of random stocks to select from SPY")
parser.add_argument("-rt", "--random-tickers", type=int, default=None,
                    help="Number of random stocks to select from ticker lists")

args = parser.parse_args()

def rando_tickers(num):
    dfs = [pd.read_csv(f"./data/training_tickers{i}.csv") for i in range(1, 4)]
    df = pd.concat(dfs)
    tickers = df['TICKERS'].tolist()

    # Select num random tickers from the list
    random_tickers = random.sample(tickers, num)

    return random_tickers

def rando_spys(num):

    sp_assets = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    assets = sp_assets['Symbol'].str.replace('.', '-').tolist()

    # Select num random symbols from SPY
    random_symbols = random.sample(assets, num)

    return random_symbols

symbols = []
if args.symbols is not None:
    symbols += args.symbols.split(',')

if args.random_spys is not None:
    symbols += rando_spys(args.random_spys)

if args.random_tickers is not None:
    symbols += rando_tickers(args.random_tickers)
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
