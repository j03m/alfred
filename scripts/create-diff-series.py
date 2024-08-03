#!/usr/bin/env python3
from alfred.data import read_symbol_file
from alfred.data import attach_moving_average_diffs, scale_relevant_training_columns, attach_profits
import argparse
import os
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help="Symbols to use separated by comma")
    parser.add_argument('--symbol-file', type=str, default="./lists/training_list.csv", help="List of symbols in a file")
    parser.add_argument('--data', type=str, default="./data", help="data dir (./data)")
    parser.add_argument('--windows', type=str, default="8,12,24", help="profit windows , separated")
    parser.add_argument('--bars', type=str, default="weekly", help="bar type")


    args = parser.parse_args()
    symbols = []
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols += pd.read_csv(args.symbol_file)["Symbols"].tolist()

    windows = args.windows.split(',')
    windows = [int(window) for window in windows]

    for symbol in symbols:
        print("pre-processing: ", symbol)
        df = read_symbol_file(args.data, symbol)
        if df is None:
            continue

        df, columns = attach_moving_average_diffs(df)

        columns.extend(["Close"])
        df, scaler = scale_relevant_training_columns(df, columns)
        df = df[columns].dropna()
        df = attach_profits(windows, args.bars, df)
        processed_file = os.path.join(args.data, f"{symbol}_diffs.csv")
        df.to_csv(processed_file)
        # todo: graph the diffs - make sure they look the way they should
        joblib.dump(scaler, os.path.join(args.data, f'{symbol}_scaler.save'))


if __name__ == "__main__":
    main()
