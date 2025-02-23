from alfred.easy import evaler
from alfred.metadata import TickerCategories
from alfred.utils import trim_timerange, set_deterministic

import argparse

set_deterministic(0)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run trainer with specified model.")
    parser.add_argument('--model', type=str, default='vanilla.small', help='Name of the model to use')
    parser.add_argument('--size', type=int, default=256, help='The size of the model to use')
    parser.add_argument('--tickers', type=str, default="./metadata/basic-tickers.json", help='Tickers to evaluate on')
    parser.add_argument('--min_date', type=str, default="2004-03-31", help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default=None, help='Maximum date for timerange trimming (YYYY-MM-DD)')

    args = parser.parse_args()

    ticker_metadata = TickerCategories(args.tickers)
    tickers = ticker_metadata.get(["evaluation"])
    files = []
    for ticker in tickers:
        files.append(f"data/{ticker}_quarterly_directional.csv")

    print("Starting easy evaler")
    evaler(augment_func=lambda df: trim_timerange(df, min_date=args.min_date, max_date=args.max_date),
           model_size=args.size,
           model_name=args.model,
           files=files)

if __name__ == "__main__":
    main()
