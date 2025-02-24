import os
import pandas as pd
import argparse
from alfred.metadata import TickerCategories, ColumnSelector
from alfred.utils import make_datetime_index

def main():
    parser = argparse.ArgumentParser(description="Basic direction test")
    parser.add_argument('--ticker-file', type=str, default="./metadata/basic-tickers.json", help='ticker file')
    parser.add_argument('--column-file', type=str, default="./metadata/column-descriptors.json", help='ticker file')
    parser.add_argument('--operation', choices=['direction', 'magnitude'], default="direction", help='tells us what the label column PQ should be')
    parser.add_argument('--data', type=str, default="./data/", help='data dir')
    args = parser.parse_args()

    categories = TickerCategories(args.ticker_file)
    column_selector = ColumnSelector(args.column_file)
    agg_config = column_selector.get_aggregation_config()
    tickers = categories.get(["training", "evaluation"])
    for ticker in tickers:
        df = pd.read_csv(os.path.join(args.data, f"{ticker}_unscaled.csv"))
        df = make_datetime_index(df, "Unnamed: 0")
        quarterly_data = df.resample('QE').agg(agg_config)
        for column in quarterly_data.columns:
            if "diff" not in column.lower():  # Check if "diff" is not in the column name (case-insensitive)
                new_column_name = f"delta_{column}"
                quarterly_data[new_column_name] = quarterly_data[column] - quarterly_data[column].shift(1)
        # our boolean checks to see if this ROW predicts a future price increase
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