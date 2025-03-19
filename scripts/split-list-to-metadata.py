import argparse
import json
import random
import pandas as pd

def main(input_file, output_file, train_percent):
    # Load tickers from CSV
    tickers_df = pd.read_csv(input_file)
    tickers = tickers_df.iloc[:, 0].tolist()

    # Shuffle and split tickers
    random.shuffle(tickers)
    split_index = int(len(tickers) * train_percent / 100)
    training_set = tickers[:split_index]
    evaluation_set = tickers[split_index:]

    # Metadata output structure
    metadata = {
        "training": training_set,
        "evaluation": evaluation_set,
        "data": ["VIX", "SPY", "CL", "BZ", "BTC"]
    }

    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split tickers for training and evaluation")
    # note to my future self or future readers - symbols.csv is special in that when cache-prices.py is done running
    # any delisted tickers in a list it received are filtered out and the final list is written to lists/symbols.csv
    # so, prior to running additional analysis on the metadata use this or you will hit errors where you have missing
    # pricing data files
    parser.add_argument("--input_file", default="lists/symbols.csv", help="Path to input CSV file containing tickers")
    parser.add_argument("--output_file", default="metadata/spy-ticker-categorization.json", help="Path to output JSON file")
    parser.add_argument("--train_percent", type=int, default=75, help="Percentage of tickers for training set (default: 75)")

    args = parser.parse_args()
    eval_percent = 100 - args.train_percent
    if eval_percent <= 0 or eval_percent >= 100:
        raise ValueError("Training percentage should be between 1 and 99")

    main(args.input_file, args.output_file, args.train_percent)
