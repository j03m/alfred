import pandas as pd
import os
import argparse
from alfred.data import downloaders

def main(symbols_file, data_dir):
    df_symbols = pd.read_csv(symbols_file)
    alpha = downloaders.AlphaDownloader()

    for symbol in df_symbols['Symbols']:
        print(f"Processing {symbol}")
        # skip the fix
        if symbol == '^VIX':
            continue

        _, quarterly_earnings = alpha.earnings(symbol)
        margins = alpha.margins(symbol)

        price_file_path = os.path.join(data_dir, f"{symbol}.csv")

        if os.path.exists(price_file_path):
            df_prices = pd.read_csv(price_file_path)
            quarterly_earnings.index = pd.to_datetime(quarterly_earnings["Date"])
            quarterly_earnings = quarterly_earnings.drop(columns=['Date'])
            df_prices.index = pd.to_datetime(df_prices['Date'])
            df_prices = df_prices.drop(columns=['Date'])

            margins.index = pd.to_datetime(margins['Date'])
            margins = margins.drop(columns=['Date'])

            # Merging data
            df_combined = df_prices.join(quarterly_earnings, how='outer')
            df_combined = df_combined.join(margins, how='outer')

            # forward will earnings - prevents lookahead
            df_combined.fillna(method='ffill', inplace=True)

            # after forward fill, we may still have na if price goes back further than earnings, treat these as 0
            df_combined.fillna(0, inplace=True)

            min_date = df_combined.index.min()
            max_date = df_combined.index.max()
            print(f"Min date for {symbol}: {min_date}")
            print(f"Max date for {symbol}: {max_date}")

            # Writing to CSV
            output_path = os.path.join(data_dir, f"{symbol}_fundamentals.csv")
            df_combined.to_csv(output_path)
            print(f"Written combined data to {output_path}")
        else:
            print(f"Pricing data file not found for {symbol}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process stock fundamentals and pricing data.")
    parser.add_argument("--symbol-file", type=str, help="Path to the CSV file containing stock symbols")
    parser.add_argument("--data-dir", default="./data", type=str,
                        help="Directory to look for pricing data and save output")

    args = parser.parse_args()
    main(args.symbol_file, args.data_dir)
