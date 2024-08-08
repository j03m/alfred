#!/usr/bin/env python3
from alfred.data import read_symbol_file
from alfred.data import attach_moving_average_diffs, scale_relevant_training_columns
import argparse
import os
import joblib
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help="Symbols to use separated by comma")
    parser.add_argument('--symbol-file', type=str, help="List of symbols in a file")
    parser.add_argument('--data', type=str, default="./data", help="data dir (./data)")
    parser.add_argument('--pred', type=int, nargs="+", help="A space separated list of prediction periods in days")

    args = parser.parse_args()
    symbols = []
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols += pd.read_csv(args.symbol_file)["Symbols"].tolist()

    ticker_data_frames = []
    for symbol in symbols:
        print("pre-processing: ", symbol)
        df = read_symbol_file(args.data, symbol)
        if df is None:
            continue

        # fix dates
        df['Date'] = df['Date'].apply(lambda x: str(x))
        df['Date'] = pd.to_datetime(df['Date'])

        # attach moving averages
        df, columns = attach_moving_average_diffs(df)

        # attach the labels for price movement
        for pred in args.pred:
            label = f'label_change_term_{pred}'
            df[label] = df['close'].pct_change(periods=pred).shift(
                periods=(-1 * pred))
            columns.append(label)

        # capture the close price
        columns.append("Close")

        # drop columns we don't want. We need columns untouched for later
        temp_columns = columns + ["Date"]
        df = df[temp_columns]

        # index by symbol
        df["Symbol"] = symbol

        # prepare to merge all
        ticker_data_frames.append(df)

    final_df = pd.concat(ticker_data_frames)

    # sort by date, then ticker 1,msft, 1,aapl | 2,msft etc
    final_df = final_df.sort_values(by=['Date', 'Symbol'])

    # fill na
    final_df = final_df.ffill().bfill()

    # save unscaled interim path
    directory = os.path.dirname(args.symbol_file)
    base_name = os.path.basename(args.symbol_file)
    file_name, file_extension = os.path.splitext(base_name)
    new_file_name = f"{file_name}_processed_unscaled{file_extension}"
    new_file_path = os.path.join(directory, new_file_name)
    final_df.to_csv(new_file_path)

    # continue scaling
    final_df, scaler = scale_relevant_training_columns(final_df, columns)

    # save the scaled data file (final)
    new_file_name = f"{file_name}_processed_scaled{file_extension}"
    new_file_path = os.path.join(directory, new_file_name)
    df.to_csv(new_file_path)

    # save the scaler
    new_file_name = f"{file_name}_scaler.save"
    new_file_path = os.path.join(directory, new_file_name)
    joblib.dump(scaler, os.path.join(args.data, new_file_path))


if __name__ == "__main__":
    main()
