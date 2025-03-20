import os
import pandas as pd
import argparse
from alfred.metadata import TickerCategories, ColumnSelector
from alfred.utils import make_datetime_index, print_in_place
from alfred.data import attach_moving_average_diffs

# add something like BTC as a column
def add_data_column(final_df, args, ticker):
    data = pd.read_csv(f"{args.data}/{ticker}.csv")
    data = make_datetime_index(data, "Date")
    data = data[["Close"]]
    # rename the cose price to the name of this column
    data.rename(columns={'Close': ticker}, inplace=True)

    # align dates
    earliest_date = final_df.index.min()
    data = data[data.index >= earliest_date]

    # merge
    final_df = pd.merge_asof(final_df, data, left_index=True, right_index=True)
    return final_df

def add_treasuries(final_df, args):
    treasuries = pd.read_csv(f"{args.data}/treasuries.csv")
    treasuries = make_datetime_index(treasuries, 'date')

    earliest_date = final_df.index.min()

    treasuries_filtered = treasuries[treasuries.index >= earliest_date]

    final_df = pd.merge_asof(final_df, treasuries_filtered, left_index=True, right_index=True)

    return final_df

def main():
    parser = argparse.ArgumentParser(description="Basic direction test")
    parser.add_argument('--ticker-file', type=str, default="./metadata/nasdaq.json", help='ticker file')
    parser.add_argument('--column-file', type=str, default="./metadata/column-descriptors.json", help='ticker file')
    parser.add_argument('--operation', choices=['direction', 'magnitude'], default="magnitude", help='tells us what the label column should be')
    parser.add_argument('--data', type=str, default="./data/", help='data dir')
    args = parser.parse_args()

    categories = TickerCategories(args.ticker_file)
    column_selector = ColumnSelector(args.column_file)
    agg_config = column_selector.get_aggregation_config()
    tickers = categories.get(["training", "evaluation"])
    data_tickers = categories.get(["data"])
    total = len(tickers)
    count = 0
    for ticker in tickers:
        count+=1
        print_in_place(f"Processing {ticker}: {count} of {total}")
        df = pd.read_csv(os.path.join(args.data, f"{ticker}_fundamentals.csv"))
        df = make_datetime_index(df, "Unnamed: 0")
        # for each data ticker, merge into the main set as a column
        for data_ticker in data_tickers:
            df = add_data_column(df, args, data_ticker)

        df, _ = attach_moving_average_diffs(df)

        # add treasuries
        df = add_treasuries(df, args)

        # aggregate the data to quarter
        quarterly_data = df.resample('QE').agg(agg_config)

        # if diff isn't already in the column name, create a diff column of the column to represent the change in
        # quarter. I did this before I was working with lstms and transformers to give historical context and haven't
        # removed it. Todo: ablation study on lstm
        for column in quarterly_data.columns:
            if "diff" not in column.lower():  # Check if "diff" is not in the column name (case-insensitive)
                new_column_name = f"delta_{column}"
                quarterly_data[new_column_name] = quarterly_data[column] - quarterly_data[column].shift(1)

        # Initially the work I did was just on direction, but we'll mostly use magnitude moving forward
        if args.operation == "direction":
            comparison_result = quarterly_data["Close"].shift(-1) > quarterly_data["Close"]
            quarterly_data["PQ"] = comparison_result.astype(int)
            quarterly_data.dropna(inplace=True)
            quarterly_data.to_csv(os.path.join(args.data, f"{ticker}_quarterly_directional.csv"))
        elif args.operation == "magnitude":
            next_day_close = quarterly_data["Close"].shift(-1)
            quarterly_data["PM"] = (next_day_close - quarterly_data["Close"] ) / quarterly_data["Close"]
            quarterly_data.dropna(inplace=True)
            quarterly_data.to_csv(os.path.join(args.data, f"{ticker}_quarterly_magnitude.csv"))


if __name__ == "__main__":
    main()

# todo tomorrow:
# make sure this works
# run and make the files
# scaffold a basic set of NNs to experiment
# write a new experimenter to run them