#!/usr/bin/env python3
import pandas as pd
from machine_learning_finance import analyze_trades, metrics_to_dataframe
import argparse
import glob
import os

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbol",  default="SPY", help="Symbol to use (default: SPY)")
parser.add_argument("-p", "--period", type=int, default=365, help="Timespan of the ledger")
args = parser.parse_args()

# Find all backtest files
backtest_files = glob.glob(f"backtests/backtest*_{args.symbol}_{args.period}*.csv")

# Create an empty DataFrame to store the results
results = pd.DataFrame()

# Process each backtest file
for file in backtest_files:
    print(f"Processing file: {file}")
    df = pd.read_csv(file)
    metrics = analyze_trades(df, args.symbol, args.period)

    # Add file name to the metrics
    metrics["file"] = os.path.basename(file)

    # Append the metrics to the results DataFrame
    results = pd.concat([results, metrics_to_dataframe(metrics)], ignore_index=True)

# Save the results to a CSV file
results.to_csv("backtests/analysis_results.csv", index=False)

