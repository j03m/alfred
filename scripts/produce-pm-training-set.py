import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from alfred.data import CachedStockDataSet, ANALYST_SCALER_CONFIG
from alfred.metadata import TickerCategories, ColumnSelector, ExperimentSelector
from alfred.model_evaluation import evaluate_model
from alfred.model_persistence import eval_model_selector
from alfred.utils import make_datetime_index

# probably shouldn't hard code this, but it's been a while since we've changed the
# analysts
# todo there are new columns in the PM set that we're not giving to the analysts, we
# should fix this?
column_selector = ColumnSelector("./metadata/column-descriptors.json")
agg_config = column_selector.get_aggregation_config()

# this isn't an experiment selector, but it does describe the analyst models we
# want to use so we use that class here:
model_descriptors = ExperimentSelector("./metadata/analyst-config.json").get(include_ranges="", exclude_ranges="")
analysts = {}
for _descriptor in model_descriptors:
    model, token, columns = eval_model_selector(_descriptor, column_selector)
    analysts[token] = (model, columns, _descriptor)


def load_symbols_from_file(file):
    return TickerCategories(file)


def calculate_analyst_projections(df, ticker, batch_size, start, end, sequence_length=24):
    # why? we're going to lose the first sequence_length records of the df during prediction
    # this will be the dataset we return
    df_trimmed = df.iloc[sequence_length:].copy()
    for model_token, value in analysts.items():
        model, columns, descriptor = value
        assert sequence_length == descriptor["sequence_length"], "model sequence lengths must be uniform"

        # todo: we need to fix this function and make its parameter set sensible
        # it doesn't need agg_config if a df is passed etc
        dataset = CachedStockDataSet(symbol=ticker,
                                     scaler_config=ANALYST_SCALER_CONFIG,
                                     sequence_length=sequence_length,
                                     feature_columns=columns,
                                     bar_type=descriptor["bar_type"],
                                     target_columns=["Close"],
                                     column_aggregation_config=agg_config,
                                     df=df)

        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        predictions, actuals = evaluate_model(model, eval_loader)
        print(f"Processed analyst: {ticker} {model_token} ({len(predictions)} predictions)")
        assert len(predictions) == len(df_trimmed), "bad dala length assumption!"

        unscaled_productions = dataset.scaler.inverse_transform_column("Close", np.array(predictions).reshape(-1, 1))
        df_trimmed[f"analyst_{model_token}"] = unscaled_productions

    return df_trimmed


def calculate_30_day_returns(ticker, data_dir, date_column="Unnamed: 0"):
    # Load the price data for the ticker
    df = pd.read_csv(f'{data_dir}/{ticker}_unscaled.csv')
    df = make_datetime_index(df, date_column)
    agg_df = df.resample('ME').agg(agg_config)
    agg_df.index = agg_df.index.to_period('M').to_timestamp()
    agg_df['30d_return'] = agg_df['Close'].pct_change(1)  # Calculate 30-day return
    return agg_df


def main(args):
    # Load tickers and create ticker-to-ID mapping
    tickers_categories = load_symbols_from_file(args.ticker_file)
    tickers = tickers_categories.get(["training", "evaluation"])
    max_rank = len(tickers)
    ticker_to_id = {ticker: idx for idx, ticker in enumerate(tickers)}

    # Define the date range for two years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=12 * 365)

    # Calculate returns for each ticker
    returns = []
    filtered = []
    date_column = "Unnamed: 0"
    for ticker in tickers:
        df = calculate_30_day_returns(ticker, args.data_dir,  date_column =  date_column)
        try:
            divs = pd.read_csv(os.path.join(args.data_dir, f"{ticker}_dividends.csv"))
        except FileNotFoundError as fne:
            print(f"no dividend file for: {ticker}, {fne}")
            filtered.append(ticker)
            continue

        divs = make_datetime_index(divs, date_column="Date")
        divs.index = pd.to_datetime(divs.index).tz_localize(None)
        if df.index.min() > start_date or len(df) == 0:
            print("Not enough data to include: ", ticker, " in training")
            continue
        # quick note, I added dividends here because I didn't want to retrain the analysts on and additional data column
        # especially considering the best mse's seem to be price only
        # that could change
        df_time_range = df[(df.index >= start_date) & (df.index <= end_date)]
        div_time_range = divs[(divs.index >= start_date) & (divs.index <= end_date)]
        div_time_range.index = div_time_range.index.to_period('M').to_timestamp()
        df_time_range.index = df_time_range.index.to_period('M').to_timestamp()
        if len(df_time_range) != 144:
            filtered.append(ticker)
            pass
        df_final = calculate_analyst_projections(df_time_range, ticker, 256, start_date, end_date)
        df_final["id"] = ticker_to_id[ticker]
        df_with_dividend = pd.merge(div_time_range, df_final, left_index=True, right_index=True)
        returns.append(df_with_dividend)

    tickers_categories.purge(filtered)
    tickers_categories.save()

    # Combine all tickers' returns into a single DataFrame
    combined_df = pd.concat(returns)
    sorted_df = combined_df.sort_index(ascending=True)

    # load and cleanup inst owners
    inst_owners = pd.read_csv(args.institutional_ownership)
    inst_owners = make_datetime_index(inst_owners, date_column="date")
    inst_owners.index = inst_owners.index.to_period('M').to_timestamp()
    inst_owners.rename(columns={'value': 'Institutional'}, inplace=True)
    inst_owners.drop(columns=['Unnamed: 0'], inplace=True)
    inst_owners.reset_index(inplace=True)

    # map tickers to id
    inst_owners['id'] = inst_owners['ticker'].map(ticker_to_id).fillna(-1).astype(int)
    inst_owners.drop(columns=['ticker'], inplace=True)

    sorted_df.reset_index(inplace=True)
    sorted_df.rename(columns={'index': 'date'}, inplace=True)
    pm_final = pd.merge(sorted_df, inst_owners, on=["date", "id"], how="left")
    pm_final = make_datetime_index(pm_final, date_column="date")
    pm_final["Institutional"].fillna(0, inplace=True)

    # see educational/ranked.py to recall our example for this
    pm_final["rank"] = pm_final.groupby(pm_final.index)["30d_return"].rank(ascending=False).astype(int)
    pm_final = pm_final.sort_index()
    pm_final.to_csv(args.training_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process stock ticker data for portfolio manager training.")
    parser.add_argument("--ticker-file", type=str, default="metadata/spy-ticker-categorization.json",
                        help="Path to the ticker categorization file.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing ticker data CSV files.")
    parser.add_argument("--rank-output-file", type=str, default="results/ranked-output.csv",
                        help="Path to save ranked returns CSV.")
    parser.add_argument("--ticker-id-output-file", type=str, default="results/ticker-id-output.csv",
                        help="Path to save ticker-to-ID mapping CSV.")
    parser.add_argument("--column-descriptor-file", type=str, default="metadata/column-descriptors.json",
                        help="Path to column descriptor JSON file.")
    parser.add_argument("--training-output-file", type=str, default="results/pm-training-final.csv",
                        help="Path to save training dataset CSV.")
    parser.add_argument("--institutional-ownership", type=str, default="data/institutional_ownership.csv",
                        help="Data indicating institutional ownership.")
    args = parser.parse_args()

    main(args)
