#!/usr/bin/env python3

from alfred.data import attach_moving_average_diffs, scale_relevant_training_columns, read_file
from alfred.utils import CustomScaler
import argparse
import os
import joblib
import pandas as pd



initial_columns_to_keep = [
    "Symbol",
    "Close",
    "Volume",
    "reportedEPS",
    "estimatedEPS",
    "surprise",
    "surprisePercentage"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help="Symbols to use separated by comma")
    parser.add_argument('--symbol-file', type=str, help="List of symbols in a file")
    parser.add_argument('--data', type=str, default="./data", help="data dir (./data)")
    parser.add_argument('--pred', type=int, nargs="+", help="A space separated list of prediction periods in days")
    parser.add_argument('--debug', type=bool, default=True, help="write debug to console")

    args = parser.parse_args()
    symbols = []
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols += pd.read_csv(args.symbol_file)["Symbols"].tolist()

    ticker_data_frames = []
    for symbol in symbols:
        print("pre-processing: ", symbol)
        df = read_file(args.data, f"{symbol}_fundamentals.csv")
        if df is None:
            continue

        # attach moving averages
        df, columns = attach_moving_average_diffs(df)

        # attach the labels for price movement
        for pred in args.pred:
            label = f'price_change_term_{pred}'
            df[label] = df['Close'].pct_change(periods=pred).shift(
                periods=(-1 * pred))
            columns.append(label)

        df["Symbol"] = symbol

        columns.extend(initial_columns_to_keep)

        # drop columns we don't want. We need columns untouched for later
        df = df[columns]

        # prepare to merge all
        ticker_data_frames.append(df)

    final_df = pd.concat(ticker_data_frames)

    treasuries = pd.read_csv(f"{args.data}/treasuries.csv")
    treasuries.index = pd.to_datetime(treasuries['date'])
    treasuries = treasuries.drop(columns=['date'])

    # First, sort by index
    final_df = final_df.sort_index()

    # Then, sort by the 'Symbol' column, maintaining the order of the index
    final_df = final_df.sort_values(by='Symbol', kind='mergesort')

    earliest_date = final_df.index.min()

    treasuries_filtered = treasuries[treasuries.index >= earliest_date]

    # Step 3: Perform the join
    final_df = final_df.join(treasuries_filtered, how='outer')

    # forward fill - prevents lookahead
    final_df.fillna(method='ffill', inplace=True)

    # backfill anything left over
    final_df.fillna(method='bfill', inplace=True)

    # save unscaled interim path
    base_name = os.path.basename(args.symbol_file)
    file_name, file_extension = os.path.splitext(base_name)
    new_file_name = f"{file_name}_processed_unscaled{file_extension}"
    new_file_path = os.path.join(args.data, new_file_name)
    final_df.to_csv(new_file_path)

    # continue scaling
    scaler = CustomScaler([
        {'regex': r'^Close.*', 'type': 'standard', 'augment': 'log'},
        {'regex': r'^pricing_change_term_.+', 'type': 'standard', 'augment': 'log'},
        {'regex': r'Volume.*', 'type': 'standard', 'augment': 'log'},
        {'columns': ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage'], 'type': 'standard'},
        {'regex': r'\d+year', 'type': 'standard'}
    ], final_df)

    scaled_df = scaler.fit_transform(final_df)

    if args.debug:
        for column in final_df.columns:
            print("column: ", column,
                  "min value: ", final_df[column].min(),
                  "max value: ", final_df[column].max(),
                  "min scaled: ", scaled_df[column].min(),
                  "max scaled: ", scaled_df[column].max())

    # save the scaled data file (final)
    new_file_name = f"{file_name}_processed_scaled{file_extension}"
    new_file_path = os.path.join(args.data, new_file_name)
    scaled_df.to_csv(new_file_path)

    scaler.serialize(os.path.join(args.data, f"{file_name}_scaler.joblib"))



if __name__ == "__main__":
    main()
