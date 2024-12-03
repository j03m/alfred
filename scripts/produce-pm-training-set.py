import argparse
import pandas as pd
from datetime import datetime, timedelta
from alfred.metadata import TickerCategories, ColumnSelector, ExperimentSelector
from alfred.model_evaluation import evaluate_model
from alfred.model_persistence import eval_model_selector
from alfred.data import CachedStockDataSet, ANALYST_SCALER_CONFIG

from torch.utils.data import DataLoader

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
    tickers = TickerCategories(file)
    return tickers.get(["training", "evaluation"])

# todo - this isn't going to work because we don't want a random seed now
# we want the last N bars and we want 1 prediction off that to put into a column
# with df (analyst 1, 2, 3, 4)
def calculate_analyst_projections(pm_df, ticker, batch_size, start, end):
    for key, value in model_descriptors.items():
        model, columns, descriptor = value
        dataset = CachedStockDataSet(symbol=ticker,
                                     scaler_config=ANALYST_SCALER_CONFIG,
                                     sequence_length=descriptor["sequence_length"],
                                     feature_columns=columns,
                                     bar_type=descriptor["bar_type"],
                                     target_columns=["Close"],
                                     column_aggregation_config=agg_config,
                                     start_date=start,
                                     end_date=end)
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        predictions, actuals = evaluate_model(model, eval_loader)

        # augment pf_df with predictions for each bar


def calculate_30_day_returns(ticker, data_dir):
    # Load the price data for the ticker
    df = pd.read_csv(f'{data_dir}/{ticker}.csv', parse_dates=['Date'], index_col='Date')
    df = df.sort_index()  # Ensure it's sorted by date
    df = df.resample('ME').agg({"Close": "last"})
    df['30d_return'] = df['Close'].pct_change(1)  # Calculate 30-day return
    return df[['30d_return']]


def main(args):
    # Load tickers and create ticker-to-ID mapping
    tickers = load_symbols_from_file(args.ticker_file)
    max_rank = len(tickers)
    ticker_to_id = {ticker: idx for idx, ticker in enumerate(tickers)}

    # Define the date range for two years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10 * 365)

    # Calculate returns for each ticker
    returns = {}
    for ticker in tickers:
        df = calculate_30_day_returns(ticker, args.data_dir)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        print("calculting anlyst predictions - this could take a bit....")
        # this reloads the same prices from disk :( however, diskio isn't the bottleneck
        # my time is the bottle neck. Todo: fix this at some future date
        df = calculate_analyst_projections(df, ticker, start_date, end_date)
        returns[ticker] = df

    # Combine all tickers' returns into a single DataFrame
    combined_df = pd.concat(returns, axis=1)
    combined_df.columns = tickers  # Adjust to handle multi-level columns if needed

    # Sort tickers by return for each date and replace tickers with IDs
    ranked_by_date = {}
    for date in combined_df.index:
        ranked_tickers = combined_df.loc[date].dropna().sort_values(ascending=False)
        ranked_ids = [ticker_to_id[ticker] for ticker in ranked_tickers.index]  # Map tickers to IDs
        ranked_by_date[date] = ranked_ids  # Store list of IDs ordered by 30-day return

    # Convert to DataFrame and save
    ranked_df = pd.DataFrame.from_dict(ranked_by_date, orient='index')
    ranked_df.to_csv(args.rank_output_file)

    # Save ticker-to-ID mapping to CSV
    ticker_id_df = pd.DataFrame(list(ticker_to_id.items()), columns=["Ticker", "ID"])
    ticker_id_df.to_csv(args.ticker_id_output_file, index=False)

    column_descriptor = ColumnSelector(file_name=args.column_descriptor_file)
    agg_config = column_descriptor.get_aggregation_config()

    dfs = []
    for ticker in tickers:
        print("Processing:", ticker)
        date_column = 'Unnamed: 0'
        df = pd.read_csv(f"{args.data_dir}/{ticker}_unscaled.csv")
        if len(df) == 0:
            continue
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Resample monthly
        df = df.resample('ME').agg(agg_config)
        # Filter to range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        df["ID"] = ticker_to_id[ticker]
        # Assert that all dates in df exist in ranked_df before assigning ranks
        missing_dates = [date for date in df.index if date not in ranked_df.index]
        assert not missing_dates, f"Missing dates in ranked_df for ticker {ticker}: {missing_dates}"

        # Map ranks from ranked_df to df's Rank column
        df["Rank"] = df.index.map(lambda date: ranked_df.loc[date, ticker_to_id[ticker]])

        dfs.append(df)

    all_data = pd.concat(dfs, axis=0).sort_index()

    # Reindex the DataFrame to include all combinations, filling missing values with fill_value
    # This makes sure that ids are included in all dates
    all_ids = all_data["ID"].unique()
    all_dates = all_data.index.unique()
    full_index = pd.MultiIndex.from_product([all_dates, all_ids], names=['Date', 'ID'])
    final_df = all_data.set_index(['ID'], append=True).reindex(full_index, fill_value=0).reset_index()

    final_df['Rank'] = final_df.apply(lambda row: row['Rank'] if not pd.isna(row['Rank']) else max_rank, axis=1)
    final_df['Rank'] = final_df.apply(lambda row: row['Rank'] if row['Rank'] != 0 else max_rank, axis=1)
    final_df = final_df.set_index("Date")
    final_df.to_csv(args.training_output_file)


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
    args = parser.parse_args()

    main(args)
