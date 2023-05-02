#!/usr/bin/env python3
import argparse
from machine_learning_finance import back_test_expert

import warnings

# filter out UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("-s", "--symbol",  default="SPY", help="Symbol to use (default: SPY)")
parser.add_argument("-u", "--curriculum", type=int, choices=[1, 2, 3], default=2, help="Curriculum level (default: 2)")
parser.add_argument("-t", "--tail", type=int, default=2500, help="Tail size (default: 2500)")
args = parser.parse_args()

print(args)
env = back_test_expert(args.symbol, args.curriculum, args.tail)
env.ledger.to_csv(f"./data/backtest_{args.symbol}_{args.tail}.csv")


