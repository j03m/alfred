from alfred.easy import evaler
from alfred.metadata import TickerCategories
from alfred.model_metrics import BCEAccumulator, MSEAccumulator
from alfred.utils import trim_timerange, set_deterministic

import argparse
import torch.nn as nn

set_deterministic(0)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run trainer with specified model.")
    parser.add_argument('--model', type=str, default='vanilla.small', help='Name of the model to use')
    parser.add_argument('--category', type=str, default="easy_model",
                        help='category can be used to put the same models into different buckets')
    parser.add_argument('--size', type=int, default=256, help='The size of the model to use')
    parser.add_argument('--tickers', type=str, default="./metadata/basic-tickers.json", help='Tickers to evaluate on')
    parser.add_argument('--min_date', type=str, default="2004-03-31",
                        help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default=None, help='Maximum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--file-post-fix', type=str, default="_quarter_directional",
                        help='assumes data/[ticker][args.file_post_fix].csv as data to use')
    parser.add_argument('--label', type=str, default="PQ",
                        help='label column')
    parser.add_argument('--loss', choices=["bce", "mse"], default="bce", help='loss function')

    args = parser.parse_args()

    ticker_metadata = TickerCategories(args.tickers)
    tickers = ticker_metadata.get(["evaluation"])
    files = []
    for ticker in tickers:
        files.append(f"data/{ticker}{args.file_post_fix}.csv")

    print("Starting easy evaler")

    evaler(category=args.category,
           augment_func=lambda df: trim_timerange(df, min_date=args.min_date, max_date=args.max_date),
           model_size=args.size,
           model_name=args.model,
           files=files,
           labels=[args.label],
           loss_function = nn.BCELoss() if args.loss == "bce" else nn.MSELoss(),
           stat_accumulator = BCEAccumulator() if args.loss == "bce" else MSEAccumulator())



if __name__ == "__main__":
    main()
