#!/usr/bin/env python3
import pandas as pd
import argparse
from alfred.data import download_ticker_list
from alfred.metadata import TickerCategories
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbols",
                    help="Symbols to use (default: SPY), separated by comma")
parser.add_argument("-f", "--symbol-file", default="./metadata/ticker-categorization.json",
                    help="Load symbols from a file")
parser.add_argument("-i", "--interval", choices=['1d', '1wk', '1mo'], default='1d', help="Load symbols from a file")
parser.add_argument("-o", "--output-dir", default="./data", help="Output directory (default: ./data)")
parser.add_argument("-t", "--types", nargs='+', choices=['training', 'evaluation', 'data'],
                    default=['training', 'evaluation', 'data'], help="portion of the file to download")

args = parser.parse_args()

tickers = TickerCategories(args.symbol_file)
metadata_tickers = tickers.get(args.types)

symbols = []
if args.symbols is not None:
    symbols += args.symbols.split(',')

if args.symbol_file is not None:
    symbols += metadata_tickers

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

symbols = list(set(symbols))
bad_symbols = download_ticker_list(symbols, args.output_dir, interval=args.interval)

tickers.purge(bad_symbols)

tickers.save()
