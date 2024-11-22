import argparse
import pandas as pd
from datetime import datetime, timedelta
from alfred.metadata import TickerCategories, ColumnSelector


def load_symbols_from_file(file):
    tickers = TickerCategories(file)
    return tickers.get(["training", "evaluation"])


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
