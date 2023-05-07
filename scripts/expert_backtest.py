#!/usr/bin/env python3
import argparse
from machine_learning_finance import BuySellEnv, back_test_expert, make_env_for, make_inverse_env_for
import pandas as pd
import warnings


# filter out UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("-s", "--symbol",  default="SPY", help="Symbol to use (default: SPY)")
parser.add_argument("-u", "--curriculum", type=int, choices=[1, 2, 3], default=2, help="Curriculum level (default: 2)")
parser.add_argument("-t", "--tail", type=int, default=2500, help="Tail size (default: 2500)")
parser.add_argument("-e", "--env-type", type=str, default="long-short",
                    help="Environment to use: 'long-short, options, inverse, buy-sell")

args = parser.parse_args()

print(args)

if args.env_type == "long-short":
    env = make_env_for(args.symbol, args.curriculum, args.tail)
    env = back_test_expert(env)
    env.ledger.to_csv(f"./data/backtest_longer_short_{args.symbol}_{args.tail}.csv")

if args.env_type == "inverse":
    inverse_file = "./data/inverse_pairs.csv"
    df = pd.read_csv(inverse_file)
    df = df.set_index('Main')
    if args.symbol in df.index:
        inverse_value = df.at[args.symbol, 'Inverse']
    else:
        print(f"{args.symbol} not found in {inverse_file}")
        exit(-1)

    env = make_inverse_env_for(args.symbol, inverse_value, args.curriculum, args.tail)
    env = back_test_expert(env)
    env.ledger.to_csv(f"./data/backtest_inverse_{args.symbol}_{args.tail}.csv")


if args.env_type == "buy-sell":
    env = make_env_for(args.symbol, args.curriculum, args.tail, EnvClass=BuySellEnv)
    env = back_test_expert(env)
    env.ledger.to_csv(f"./data/backtest_buy_sell_{args.symbol}_{args.tail}.csv")

if args.env_type == "options":
    raise Exception("Implement me")

