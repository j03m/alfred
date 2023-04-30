#!/usr/bin/env python3
import argparse
import warnings
import random
import time
import pandas as pd

from machine_learning_finance import guided_training, make_env_for, partial_test, partial_train

# filter out UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbols", help="Symbols to use (default: SPY), separated by comma")
parser.add_argument("-t", "--tail", type=int, default=2500, help="Tail size (default: 2500)")
parser.add_argument("-g", "--guide", action="store_true", help="Enable guided training")
parser.add_argument("-r", "--train", action="store_true", help="Enable partial training")
parser.add_argument("-e", "--test", action="store_true", help="Enable partial testing")
parser.add_argument("-p", "--steps", type=int, default=25000, help="Number of steps (default: 25000)")
parser.add_argument("-c", "--create", action="store_true", help="Create new environment")
parser.add_argument("-o", "--output-dir", default="./data", help="Output directory (default: ./data)")
parser.add_argument("-l", "--random-stocks", type=int, default=None, help="Number of random stocks to select from SPY")
parser.add_argument("-u", "--curriculum", type=int, choices=[1, 2, 3], default=2, help="Curriculum level (default: 2)")

args = parser.parse_args()


def rando_spys(num):

    sp_assets = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    assets = sp_assets['Symbol'].str.replace('.', '-').tolist()

    # Select 25 random symbols from SPY
    random_symbols = random.sample(assets, num)

    return random_symbols


if args.random_stocks is None:
    if args.symbols is None:
        parser.error("Must specify either --symbols or --random-stocks")

if args.random_stocks is not None:
    if args.symbols is not None:
        parser.error("Cannot specify both --symbols and --random-stocks")

if args.random_stocks is not None:
    symbols = rando_spys(args.random_stocks)

if args.symbols is not None:
    symbols = args.symbols.split(',')

if args.guide:
    for symbol in symbols:
        time.sleep(0.25)
        env = make_env_for(symbol, args.curriculum, args.tail)
        env = guided_training(symbol, args.create, args.steps, args.tail)
        env.ledger.to_csv(f"{args.output_dir}/env_{symbol}_guided.csv")

if args.train:
    for symbol in symbols:
        time.sleep(0.25)
        env = make_env_for(symbol, args.curriculum, args.tail)
        partial_train(env, args.steps, args.create)
        env.ledger.to_csv(f"{args.output_dir}/env_{symbol}_train.csv")

if args.test:
    for symbol in symbols:
        time.sleep(0.25)
        env = make_env_for(symbol, args.curriculum, args.tail)
        partial_test(env)
        env.ledger.to_csv(f"{args.output_dir}/env_{symbol}_test.csv")
