import pandas as pd

from alfred.easy import evaler
from alfred.metadata import TickerCategories
from alfred.utils import trim_timerange, set_deterministic, read_time_series_file

import argparse

set_deterministic(0)


def ablate(df, min_date, max_date, column):
    df = trim_timerange(df, min_date, max_date).copy()
    df[column] = df[column].mean()
    return df


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run trainer with specified model.")
    parser.add_argument('--model', type=str, default='vanilla.medium', help='Name of the model to use')
    parser.add_argument('--size', type=int, default=1024, help='The size of the model to use')
    parser.add_argument('--tickers', type=str, default="./metadata/basic-tickers.json", help='Tickers to evaluate on')
    parser.add_argument('--min_date', type=str, default="2004-03-31",
                        help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default=None, help='Maximum date for timerange trimming (YYYY-MM-DD)')

    args = parser.parse_args()

    ticker_metadata = TickerCategories(args.tickers)
    tickers = ticker_metadata.get(["evaluation"])
    files = []
    once = False
    columns = []
    for ticker in tickers:
        file = f"data/{ticker}_quarterly_directional.csv"
        if not once:
            once = True
            df = trim_timerange(read_time_series_file(file), args.min_date, args.max_date)
            columns = df.columns.tolist()
            columns.remove("PQ")
        files.append(file)

    print("Evaluating once with all columns:")
    main_loss, main_stats = evaler(
        augment_func=lambda df: trim_timerange(df, min_date=args.min_date, max_date=args.max_date),
        model_size=args.size,
        model_name=args.model,
        files=files)

    ablation_results = {}
    for column in columns:
        print("Evaluating with {column} ablated:")
        loss, stats = evaler(
            augment_func=lambda df: ablate(df, min_date=args.min_date, max_date=args.max_date, column=column),
            model_size=args.size,
            model_name=args.model,
            files=files)
        ablation_results[column] = {
            "loss": loss,
            "stats": stats,
        }

    # Build a results DataFrame
    results = []
    for column, result in ablation_results.items():
        row = {'Feature': column, 'Loss Change': result['loss'] - main_loss,
               f'f1 Change': result['stats']['f1'].item() - main_stats['f1'].item()}
        results.append(row)

    results_df = pd.DataFrame(results)
    filtered_df = results_df[(results_df['Loss Change'] > 0) & (results_df['f1 Change'] < 0)]
    filtered_df = filtered_df.sort_values(by='Loss Change', ascending=False)

    # Print results
    print("\nAblation Study Results:")
    print(filtered_df)
    filtered_df.to_csv("data/ablation_results.csv", index=False)


if __name__ == "__main__":
    main()
