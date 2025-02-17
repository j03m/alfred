from alfred.easy import trainer
from alfred.metadata import TickerCategories
import pandas as pd
import argparse

def trim_timerange(df, min_date=None, max_date=None):
    if min_date is None:
        start_date = df.index.min()
    else:
        start_date = pd.Timestamp(min_date, tz='UTC')

    if max_date is None:
        # If max_date is not provided, we are not trimming the upper bound
        return df[df.index >= start_date]
    else:
        end_date = pd.Timestamp(max_date, tz='UTC')
        return df[(df.index >= start_date) & (df.index <= end_date)]

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run trainer with specified model and timerange.")
    parser.add_argument('--model', type=str, default='vanilla.small', help='Name of the model to use')
    parser.add_argument('--size', type=int, default=256, help='The size of the model to use')
    parser.add_argument('--min_date', type=str, default="2004-03-31", help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default=None, help='Maximum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, default="./metadata/basic-tickers.json", help='Tickers to train on')

    args = parser.parse_args()

    ticker_metadata = TickerCategories(args.tickers)
    tickers = ticker_metadata.get(["training"])
    files = []
    for ticker in tickers:
        files.append(f"data/{ticker}_quarterly_directional.csv")


    print("Starting easy trainer")
    trainer(
        augment_func=lambda df: trim_timerange(df, min_date=args.min_date, max_date=args.max_date),
        files=files,
        verbose=True,
        model_size=args.size,
        model_name=args.model
    )

if __name__ == "__main__":
    main()