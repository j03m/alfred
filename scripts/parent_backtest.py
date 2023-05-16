#!/usr/bin/env python3

import subprocess
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbols",  default="SPY", help="Symbol to use (default: SPY)")
parser.add_argument("-p", "--period", type=int, default=365, help="Timespan of the ledger")
parser.add_argument("-c", "--cash", type=int, default=85000, help="Timespan of the ledger")
parser.add_argument("-u", "--curriculum", type=int, default=1, help="curriculum (used for ai)")

args = parser.parse_args()

print("Testing: ", args)

# Variable parameters
#high_probability_values = [0.5, 0.6, 0.7, 0.8, 0.9]
#low_probability_values = [0.5, 0.4, 0.3, 0.2, 0.1]
high_probability_values = [0.8, 0.9]
low_probability_values = [0.2, 0.1]

# Env types to execute
env_types = ["long-short", "inverse", "buy-sell"]

symbols = []
symbols += args.symbols.split(',')

# Iterate through env_types, high_probability_values, and low_probability_values
for env_type in env_types:
    for high_prob in high_probability_values:
        for low_prob in low_probability_values:
            for symbol in symbols:
                # Execute the expert_backtest.py script with the specified parameters
                subprocess.run([
                    "python3", "./scripts/expert_backtest.py",
                    "-s", symbol,
                    "-u", str(args.curriculum),
                    "-t", str(args.period),
                    "-e", env_type,
                    "-c", str(args.cash),
                    "-hp", str(high_prob),
                    "-lp", str(low_prob)
                ])
