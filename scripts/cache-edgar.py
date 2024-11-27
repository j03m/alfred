from alfred.data import download_13f_filings
import os
import pandas as pd
from alfred.metadata import TickerCategories
from alfred.data import download_13f_filings

def main(symbol_file, data_folder, output_folder="./filings"):
    # Load tickers from the provided symbol file
    tickers = TickerCategories(symbol_file)
    metadata_tickers = tickers.get(["training", "evaluation"])  # Filter relevant tickers

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for ticker in metadata_tickers:
        ticker_file = os.path.join(data_folder, f"{ticker}.csv")
        if not os.path.exists(ticker_file):
            print(f"Data file not found for ticker: {ticker}")
            continue

        # Load the data file for the ticker
        df = pd.read_csv(ticker_file)

        # Ensure "Date" column exists
        if "Date" not in df.columns:
            print(f"'Date' column not found in file: {ticker_file}")
            continue

        # Extract unique years from the "Date" column
        try:
            df['Year'] = pd.to_datetime(df['Date']).dt.year
            unique_years = sorted(df['Year'].unique())
        except Exception as e:
            print(f"Error processing dates for {ticker}: {e}")
            continue

        # Download 13F filings for the extracted years
        print(f"Processing ticker: {ticker}")
        print(f"Years found: {unique_years}")
        try:
            download_13f_filings(ticker, unique_years, output_folder)
        except Exception as e:
            print(f"Error downloading filings for {ticker}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download 13F filings for tickers.")
    parser.add_argument("--symbol-file", default="./metadata/spy-ticker-categorization.json", help="Path to the symbol file.")
    parser.add_argument("--data", default="./data", help="Path to the folder containing ticker data files.")
    parser.add_argument("--output-folder", default="./filings", help="Output folder for filings.")

    args = parser.parse_args()

    main(args.symbol_file, args.data, args.output_folder)