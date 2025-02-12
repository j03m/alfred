import argparse
import os
import pandas as pd
import numpy as np

def has_min(df, start_date):
    min_date = df.iloc[:, 0].min()
    return pd.notna(min_date) and pd.Timestamp(min_date) and min_date <= start_date


def has_max(df, end_date):
    max_date = df.iloc[:, 0].max()
    return pd.notna(max_date) and pd.Timestamp(max_date) and max_date >= end_date


def build_correlations(directory, matcher, input_start_date, input_end_date, date_col):
    start_date = pd.Timestamp(input_start_date)
    all_frames = []
    end_date = pd.Timestamp(input_end_date)
    for filename in os.listdir(directory):
        if filename.endswith(matcher):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, parse_dates=[0])
            if has_min(df, start_date) and has_max(df, end_date):
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                ticker = filename.replace(matcher, "")
                df.set_index(date_col, inplace=True)
                df = df[["Close"]]
                df.rename(columns={"Close": ticker}, inplace=True)
                all_frames.append(df)

    # concat on columns
    correlations = pd.concat(all_frames, axis=1).sort_index()
    return correlations.corr()

def greedy_anti_correlation(starting_ticker, correlations, num_stocks=10):
    selected_stocks = [starting_ticker]
    remaining_stocks = list(correlations.columns)
    remaining_stocks.remove(starting_ticker)  # Remove the starting stock

    for _ in range(num_stocks):
        min_avg_corr = 1.0  # Initialize with maximum possible correlation
        next_stock = None

        for stock in remaining_stocks:
            # Calculate average correlation with the *already selected* stocks
            stock_correlations = correlations.loc[stock, selected_stocks].values
            avg_corr = np.mean(
                np.abs(stock_correlations))  # Use absolute correlation to find stocks uncorrelated in either direction

            if avg_corr < min_avg_corr:
                min_avg_corr = avg_corr
                next_stock = stock

        selected_stocks.append(next_stock)
        remaining_stocks.remove(next_stock)

    return selected_stocks

def main():
    parser = argparse.ArgumentParser(description="Find CSV files with data from year 2000 or earlier.")
    parser.add_argument("--directory", default="./data", help="Directory containing the CSV files")
    parser.add_argument("--matcher", default="_unscaled.csv", help="End of file pattern to match")
    parser.add_argument("--start_date", default="2001-01-01", help="YYYY-MM-DD min requirement")
    parser.add_argument("--end_date", default="2024-12-17", help="YYYY-MM-DD max requirement")
    parser.add_argument("--date_column", default="Unnamed: 0", help="My date columns are anon maybe yours aren't")
    parser.add_argument("--starting_ticker", default="AAPL", help="Greedily choose uncorrelated stocks starting with")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        return

    correlations = build_correlations(args.directory, args.matcher, args.start_date, args.end_date, args.date_column)

    print(greedy_anti_correlation("AAPL", correlations))


if __name__ == "__main__":
    main()
