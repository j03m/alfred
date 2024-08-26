#!/usr/bin/env python3

from alfred.data import attach_moving_average_diffs, scale_relevant_training_columns, read_file
from alfred.utils import CustomScaler
import argparse
import os
import joblib
import pandas as pd
from sklearn.decomposition import PCA

initial_columns_to_keep = [
    "Symbol",
    "Close",
    "Volume",
    "reportedEPS",
    "estimatedEPS",
    "surprise",
    "surprisePercentage",
    'Margin_Gross',
    'Margin_Operating',
    'Margin_Net_Profit'
]

def add_vix(final_df, args):
    vix = pd.read_csv(f"{args.data}/^VIX.csv")
    vix.index = pd.to_datetime(vix['Date'])
    vix = vix[["Close"]]
    vix.rename(columns={'Close': 'VIX'}, inplace=True)
    earliest_date = final_df.index.min()
    vix = vix[vix.index >= earliest_date]
    final_df = final_df.join(vix, how='outer')
    final_df.fillna(method='ffill', inplace=True)
    final_df.fillna(method='bfill', inplace=True)
    return final_df

def add_treasuries(final_df, args):
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
    return final_df


import pandas as pd

# Assuming 'final_df' has a DatetimeIndex and a column called 'Symbol'
import pandas as pd


def align_date_range(final_df):
    # Step 1: Group by 'Symbol' and find the start date for each symbol
    start_dates = final_df.groupby('Symbol').apply(lambda x: x.index.min())

    # Step 2: Find the symbol with the latest start date (this limits the date range)
    latest_start_date = start_dates.max()
    symbol_with_latest_start_date = start_dates.idxmax()

    # Step 3: Filter out rows where the date is earlier than the latest start date
    filtered_df = final_df[final_df.index >= latest_start_date]

    # Step 4: Get the final min and max dates in the filtered DataFrame
    final_min_date = filtered_df.index.min()
    final_max_date = filtered_df.index.max()

    # Emit the final date range and the symbol causing the limitation
    print(f"Final Date Range: {final_min_date} to {final_max_date} limited by: {symbol_with_latest_start_date}")

    return filtered_df, final_min_date, final_max_date


def attach_price_prediction_labels(args, columns, df):
    # attach the labels for price movement
    for pred in args.pred:
        label = f'price_change_term_{pred}'
        df[label] = df['Close'].pct_change(periods=pred).shift(
            periods=(-1 * pred))
        columns.append(label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help="Symbols to use separated by comma")
    parser.add_argument('--symbol-file', type=str, help="List of symbols in a file")
    parser.add_argument('--data', type=str, default="./data", help="data dir (./data)")
    parser.add_argument('--pred', type=int, nargs="+", default=[7, 30, 120, 240], help="A space separated list of prediction periods in days")
    parser.add_argument('--debug', type=bool, default=True, help="write debug to console")

    args = parser.parse_args()
    symbols = []
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols += pd.read_csv(args.symbol_file)["Symbols"].tolist()

    ticker_data_frames = []
    for symbol in symbols:
        if symbol == "^VIX":
            continue

        print("pre-processing: ", symbol)

        df = read_file(args.data, f"{symbol}_fundamentals.csv")
        assert(df is not None)

        # attach moving averages
        df, columns = attach_moving_average_diffs(df)

        # attach labels
        attach_price_prediction_labels(args, columns, df)

        # both of these functions will cause NaN to get introduced. We can't predict for labels we don't have with missing data, so we'll trim it all out
        df.dropna(inplace=True)

        min_date = df.index.min()
        max_date = df.index.max()
        print(f"Min date for {symbol}: {min_date}")
        print(f"Max date for {symbol}: {max_date}")

        df["Symbol"] = symbol

        columns.extend(initial_columns_to_keep)

        # drop columns we don't want. We need columns untouched for later
        df = df[columns]

        # prepare to merge all
        ticker_data_frames.append(df)

    final_df = pd.concat(ticker_data_frames)
    final_df = add_vix(final_df, args)
    final_df = add_treasuries(final_df, args)

    # todo: what is up here? final date range is 2021 start...should go further back
    final_df, _, _ = align_date_range(final_df)

    assert not final_df.isnull().any().any(), f"unscaled df has null after transform"

    # save unscaled interim path
    base_name = os.path.basename(args.symbol_file)
    file_name, file_extension = os.path.splitext(base_name)
    new_file_name = f"{file_name}_processed_unscaled{file_extension}"
    new_file_path = os.path.join(args.data, new_file_name)
    final_df.to_csv(new_file_path)

    # continue scaling
    scaler = CustomScaler([
        {'regex': r'^Close.*', 'type': 'standard'},
        {'regex': r'^VIX.*', 'type': 'standard'},
        {'regex': r'^Margin.*', 'type': 'standard'},
        {'regex': r'^pricing_change_term_.+', 'type': 'standard'},
        {'regex': r'Volume.*', 'type': 'standard', 'augment': 'log'},
        {'columns': ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage'], 'type': 'standard'},
        {'regex': r'\d+year', 'type': 'standard'}
    ], final_df)

    scaled_df = scaler.fit_transform(final_df)

    assert not scaled_df.isnull().any().any(), f"scaled df has null after transform"

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
    print("start")
    main()
    print("end")
