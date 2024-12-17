#!/usr/bin/env python3
import argparse
from alfred.metadata import TickerCategories
import os
import yfinance as yf

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbols",
                    help="Symbols to use (default: SPY), separated by comma")
parser.add_argument("-f", "--symbol-file", default="./metadata/ticker-categorization.json",
                    help="Load symbols from a file")
parser.add_argument("-i", "--interval", choices=['1d', '1wk', '1mo'], default='1d', help="Load symbols from a file")
parser.add_argument("-o", "--output-dir", default="./data", help="Output directory (default: ./data)")

args = parser.parse_args()

tickers = TickerCategories(args.symbol_file)
metadata_tickers = tickers.get(["training", "evaluation", "data"])

symbols = []
bad_symbols = []
if args.symbols is not None:
    symbols += args.symbols.split(',')

if args.symbol_file is not None:
    symbols += metadata_tickers

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

symbols = list(set(symbols))
for symbol in symbols:
    try:
        print("dividends for: ", symbol)
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends
        dividends_df = dividends.reset_index()
        dividends_df.columns = ['Date', 'Dividend']  # Rename columns for clarity
        dividends_df.to_csv(os.path.join(args.output_dir, f"{symbol}_dividends.csv"), index=False)
    except Exception as e:
        print(f"failed {symbol} with {e}")
        bad_symbols.append(symbol)

tickers.purge(bad_symbols)
tickers.save()
