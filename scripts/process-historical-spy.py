import csv
from datetime import datetime, timedelta


def process_tickers(input_file, output_file, lookback_years):
    """
    Reads a CSV file line by line, filters rows based on the lookback parameter, and writes unique tickers to an output file.

    :param input_file: Path to the input CSV file
    :param output_file: Path to the output CSV file
    :param lookback_years: Number of years to look back from today's date
    """
    # Calculate the cutoff date
    cutoff_date = datetime.now() - timedelta(days=lookback_years * 365)

    unique_tickers = set()

    with open(input_file, mode='r') as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            # Parse the date
            try:
                row_date = datetime.strptime(row['date'], '%Y-%m-%d')
            except ValueError:
                print(f"Skipping invalid date: {row['date']}")
                continue

            # Check if the date is within the lookback range
            if row_date >= cutoff_date:
                tickers = row['tickers'].split(',')
                unique_tickers.update(tickers)

    # Write unique tickers to the output file
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['ticker'])  # Header
        for ticker in sorted(unique_tickers):
            writer.writerow([ticker])


if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Process tickers from CSV.")
    parser.add_argument('--input_file', type=str, default="/Users/jmordetsky/sp500/spy-historical.csv",
                        help="Path to the input CSV file.")
    parser.add_argument('--output_file', type=str, default="./data/spy-historical-members.csv",
                        help="Path to the output CSV file.")
    parser.add_argument('--lookback', type=int, default=10, help="Number of years to look back.")

    args = parser.parse_args()


    process_tickers(args.input_file, args.output_file, args.lookback)
