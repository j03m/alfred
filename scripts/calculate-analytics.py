#!/usr/bin/env python3
from machine_learning_finance import calculate_trend_metrics_full, generate_max_profit_actions, read_symbol_file
import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-set', type=str, default=None)
    parser.add_argument('--data-path', type=str, default="./data")
    args = parser.parse_args()
    if args.train_set is None:
        print("Provide a training set!")
        return -1
    if not os.path.isfile(args.train_set):
        print(f"I can't find {args.train_set}!")
        return -2

    symbols_df = pd.read_csv(args.train_set)
    for symbol in symbols_df["Symbols"].values:
        print("pre-processing: ", symbol)
        df = read_symbol_file(args.data_path, symbol)
        df, columns = calculate_trend_metrics_full(df)
        df["actions"] = generate_max_profit_actions(df["Close"], [5, 15, 30, 60], 5, 10)
        processed_file = os.path.join(args.data_path, f"{symbol}_processed.csv")
        df.to_csv(processed_file)


if __name__ == "__main__":
    main()
