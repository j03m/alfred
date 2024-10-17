#!/usr/bin/env python3
import pandas as pd
import argparse
from alfred.data import download_ticker_list
import os

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbols", help="Symbols to use (default: SPY), separated by comma")
parser.add_argument("-f", "--symbol-file", help="Load symbols from a file")
parser.add_argument("-i", "--interval", choices=['1d', '1wk', '1mo'], default='1d', help="Load symbols from a file")
parser.add_argument("-fo", "--symbol-file-out", default="./lists/symbols.csv",
                    help="Output file - all bad tickers trimmed")
parser.add_argument("-o", "--output-dir", default="./data", help="Output directory (default: ./data)")


args = parser.parse_args()

def load_symbols_from_file(file):
    return pd.read_csv(file)["Symbols"].tolist()


symbols = []
if args.symbols is not None:
    symbols += args.symbols.split(',')

if args.symbol_file is not None:
    symbols += load_symbols_from_file(args.symbol_file)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

symbols = list(set(symbols))
bad_symbols = download_ticker_list(symbols, args.output_dir, interval=args.interval)

if bad_symbols is None:
    final_symbols = set(symbols)
else:
    final_symbols = set(symbols) - set(bad_symbols)

# Create a pandas DataFrame with symbols as data and "Symbols" as column name
df = pd.DataFrame({'Symbols': list(final_symbols)})

# Save the DataFrame to a CSV file named "symbols.csv"
df.to_csv(args.symbol_file_out, index=False)
