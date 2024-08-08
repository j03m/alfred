import pandas as pd
import yfinance as yf
import os
import argparse


def fetch_fundamentals(symbol):
    stock = yf.Ticker(symbol)
    df_financials = stock.quarterly_financials.transpose()
    df_balance_sheet = stock.quarterly_balance_sheet.transpose()
    df_cash_flow = stock.quarterly_cash_flow.transpose()
    df_fundamentals = pd.concat([df_financials, df_balance_sheet, df_cash_flow], axis=1)

    return df_fundamentals


def main(symbols_file, data_dir):
    df_symbols = pd.read_csv(symbols_file)
    for symbol in df_symbols['Symbols']:
        print(f"Processing {symbol}")
        df_fundamentals = fetch_fundamentals(symbol)

        price_file_path = os.path.join(data_dir, f"{symbol}.csv")

        if os.path.exists(price_file_path):
            df_prices = pd.read_csv(price_file_path)
            df_fundamentals.index = pd.to_datetime(df_fundamentals.index)
            df_prices.index = pd.to_datetime(df_prices['Date'])

            # Merging data
            df_combined = df_prices.join(df_fundamentals, how='outer')
            df_combined.fillna(method='ffill', inplace=True)

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
