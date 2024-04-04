#!/usr/bin/env python3
from machine_learning_finance import read_symbol_file
from next_gen import attach_moving_average_diffs, scale_relevant_training_columns
import argparse
import os
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help="Symbols to use separated by comma")
    parser.add_argument('--symbol-file', type=str, help="List of symbols in a file")
    parser.add_argument('--data', type=str, default="./data", help="data dir (./data)")

    args = parser.parse_args()
    symbols = []
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols += pd.read_csv(args.symbol_file)["Symbols"].tolist()

    for symbol in symbols:
        print("pre-processing: ", symbol)
        df = read_symbol_file(args.data, symbol)
        if df is None:
            continue

        df, columns = attach_moving_average_diffs(df)
        columns.extend(["Close"])
        df, scaler = scale_relevant_training_columns(df, columns)
        processed_file = os.path.join(args.data, f"{symbol}_diffs.csv")
        df = df[columns].dropna()
        df.to_csv(processed_file)
        joblib.dump(scaler, os.path.join(args.data, f'{symbol}_scaler.save'))


if __name__ == "__main__":
    main()
