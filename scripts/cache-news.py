#!/usr/bin/env python3
import pandas as pd
import os
import argparse
from alfred.data import ArticleDownloader
from alfred.metadata import TickerCategories
from alfred.data import choose_train_range
from datetime import datetime, timedelta

def load_symbols_from_file(file):
    tickers = TickerCategories(file)
    return tickers.get(["training", "evaluation", "data"])

def main(args):
    tickers = load_symbols_from_file(args.symbol_file)
    dl = ArticleDownloader(cache_dir=args.cache)
    for ticker in tickers:

        if args.use_seed:
            # first get a range of time
            df = pd.read_csv(f"{args.data}/{ticker}_unscaled.csv")
            df[args.date_column] = pd.to_datetime(df[args.date_column])
            df = df.set_index(args.date_column)
            total_length = len(df)
            start = choose_train_range(ticker, args.seed, total_length, args.period)
            end = start + args.period
            start_date = df.index[start]
            end_date = df.index[end]
        else:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=730)

        if args.update_only and os.path.isdir(os.path.join(args.cache, ticker)):
            print(f"skipping {ticker} cache exists")
            continue

        # cache the data
        dl.download_and_cache_article(ticker, start_date, end_date)



        #todo: run and test me

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol-file', type=str, help="List of symbols in a file")
    parser.add_argument('--cache', type=str, default="./news", help="news cache dir (./news)")
    parser.add_argument('--data', type=str, default="./data", help="market data dir (./data)")
    parser.add_argument('--date-column', type=str, default="Unnamed: 0", help="date column for cache csvs")
    parser.add_argument("--use-seed", action="store_true")
    parser.add_argument("--update-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed is combined with a ticker to produce a consistent random training and eval period")
    parser.add_argument('--period', type=int, default=120, help="window of time")
    args = parser.parse_args()
    main(args)





