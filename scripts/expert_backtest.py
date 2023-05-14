#!/usr/bin/env python3
import argparse
from machine_learning_finance import BuySellEnv, back_test_expert, make_env_for, make_inverse_env_for
import pandas as pd
import warnings

# filter out UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("-s", "--symbol", default="SPY", help="Symbol to use (default: SPY)")
parser.add_argument("-u", "--curriculum", type=int, choices=[1, 2, 3], default=2, help="Curriculum level (default: 2)")
parser.add_argument("-t", "--tail", type=int, default=365, help="Tail size (default: 365)")
parser.add_argument("-e", "--env-type", type=str, default="long-short",
                    help="Environment to use: 'long-short, options, inverse, buy-sell")
parser.add_argument("-c", "--cash", type=int, default=5000,
                    help="how much cash to trade with")
parser.add_argument("-hp", "--high-probability", type=float, default=0.8,
                    help="high probability marker for trades")
parser.add_argument("-lp", "--low-probability", type=float, default=0.2,
                    help="low probability marker for trades")
args = parser.parse_args()

print(args)


def gen_file_id(args):
    return f"{args.symbol}_{args.tail}_h{args.high_probability}_l{args.low_probability}"


if args.env_type == "long-short":
    env = make_env_for(args.symbol,
                       args.curriculum,
                       args.tail,
                       cash=args.cash,
                       prob_high=args.high_probability,
                       prob_low=args.low_probability)
    env = back_test_expert(env)
    env.ledger.to_csv(f"./backtests/backtest_long_short_{gen_file_id(args)}.csv")

if args.env_type == "inverse":
    inverse_file = "./data/inverse_pairs.csv"
    df = pd.read_csv(inverse_file)
    df = df.set_index('Main')
    if args.symbol in df.index:
        inverse_value = df.at[args.symbol, 'Inverse']
    else:
        print(f"{args.symbol} not found in {inverse_file}")
        exit(-1)

    env = make_inverse_env_for(args.symbol,
                               inverse_value,
                               args.curriculum,
                               args.tail,
                               cash=args.cash,
                               prob_high=args.high_probability,
                               prob_low=args.low_probability)
    env = back_test_expert(env)
    env.ledger.to_csv(f"./backtests/backtest_inverse_{gen_file_id(args)}.csv")

if args.env_type == "buy-sell":
    env = make_env_for(args.symbol,
                       args.curriculum,
                       args.tail,
                       cash=args.cash,
                       prob_high=args.high_probability,
                       prob_low=args.low_probability,
                       env_class=BuySellEnv)
    env = back_test_expert(env)
    env.ledger.to_csv(f"./backtests/backtest_buy_sell_{gen_file_id(args)}.csv")

if args.env_type == "options":
    raise Exception("Implement me")
