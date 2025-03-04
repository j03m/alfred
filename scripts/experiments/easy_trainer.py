from alfred.easy import trainer
from alfred.metadata import TickerCategories
from alfred.utils import trim_timerange, set_deterministic
from alfred.model_metrics import BCEAccumulator, RegressionAccumulator, HuberWithSignPenalty, MSEWithSignPenalty, SignErrorRatioLoss

import torch.nn as nn

import argparse

set_deterministic(0)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run trainer with specified model and timerange.")
    parser.add_argument('--category', type=str, default="easy_model",
                        help='category can be used to put the same models into different buckets')
    parser.add_argument('--model', type=str, default='vanilla.small', help='Name of the model to use')
    parser.add_argument('--size', type=int, default=256, help='The size of the model to use')
    parser.add_argument('--seq-len', type=int, default=None, help='Sequence length, supply when using lstms')
    parser.add_argument('--min_date', type=str, default="2004-03-31",
                        help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default=None, help='Maximum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, default="./metadata/basic-tickers.json", help='Tickers to train on')
    parser.add_argument('--file-post-fix', type=str, default="_quarterly_directional",
                        help='assumes data/[ticker][args.file_post_fix].csv as data to use')
    parser.add_argument('--label', type=str, default="PQ",
                        help='label column')
    parser.add_argument('--loss', choices=["bce", "mse", "huber-sign", "mse-sign", "sign-ratio"], default="bce", help='loss function')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--patience', type=int, default=500, help='number of epochs to allow without loss decrease')

    args = parser.parse_args()

    ticker_metadata = TickerCategories(args.tickers)
    tickers = ticker_metadata.get(["training"])
    files = []
    for ticker in tickers:
        files.append(f"data/{ticker}{args.file_post_fix}.csv")

    loss_function = None
    if args.loss == "bce":
        loss_function = nn.BCELoss()
    elif args.loss == "mse":
        loss_function = nn.MSELoss()
    elif args.loss == "huber-sign":
        loss_function = HuberWithSignPenalty()
    elif args.loss == "mse-sign":
        loss_function = MSEWithSignPenalty()
    elif args.loss == "sign-ratio":
        loss_function = SignErrorRatioLoss()

    print("Starting easy trainer")
    trainer(
        category=args.category,
        augment_func=lambda df: trim_timerange(df, min_date=args.min_date, max_date=args.max_date),
        files=files,
        patience=args.patience,
        verbose=True,
        model_size=args.size,
        model_name=args.model,
        labels=[args.label],
        epochs=args.epochs,
        loss_function=loss_function,
        seq_len=args.seq_len,
        stat_accumulator=BCEAccumulator() if args.loss == "bce" else RegressionAccumulator())

if __name__ == "__main__":
    main()
