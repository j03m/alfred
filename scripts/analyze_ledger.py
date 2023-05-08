#!/usr/bin/env python3
import pandas as pd
from machine_learning_finance import analyze_trades
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbol",  default="SPY", help="Symbol to use (default: SPY)")
parser.add_argument("-p", "--period", type=int, default=365, help="Timespan of the ledger")
parser.add_argument("-f", "--file", type=str, help="location of the ledger file")

args = parser.parse_args()

print("Analyzing: ", args.f)

# Analyze the trades
df = pd.read_csv(args.file)
metrics = analyze_trades(df, args.symbol, args.period)

# Print the metrics
for key, value in metrics.items():
    print(f"{key}: {value}")
