#!/usr/bin/env python3
import argparse
import warnings
import random
import time
import pandas as pd
import os

from machine_learning_finance import make_env_for, partial_test, partial_train, download_ticker_list

# filter out UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PPO = 0
MODEL_RECURRENT = 1

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("-t", "--tail", type=int, default=2500, help="Tail size (default: 2500)")

parser.add_argument("-r", "--train", action="store_true", help="Enable partial training")

parser.add_argument("-e", "--test", action="store_true", help="Enable partial testing")

parser.add_argument("-p", "--steps", type=int, default=100000, help="Number of steps (default: 100000)")

# todo make this backup the old agent
parser.add_argument("-c", "--create", action="store_true", help="Create new agent")

# todo broken into output for ledgers vs model vs data
parser.add_argument("-o", "--output-dir", default="./backtests", help="Output directory (default: ./backtests)")

parser.add_argument("-m", "--model", type=int, choices=[0, 1], default=1, help="What model to use")

parser.add_argument("-s", "--symbols_path", default="./data/symbols.csv", type=str, help="The location of a csv file of symbols")

args = parser.parse_args()

if not os.path.isfile(args.symbols_path):
    raise Exception(f"{args.symbols_path} does not exist. Please check that a valid set of symbols to use exists.")

# assumes data is pre cached with cache_data.py
symbols = pd.read_csv(args.symbols_path)["Symbols"]

if args.train:
    for symbol in symbols:
        print("training:", symbol)
        try:
            time.sleep(0.25)
            env = make_env_for(symbol, args.curriculum, args.tail, "file")
            partial_train(env, args.steps, args.create, args.model)
            env.ledger.to_csv(f"{args.output_dir}/env_{symbol}_train.csv")
        except Exception as e:
            print(e)

if args.test:
    for symbol in symbols:
        print("testing:", symbol)
        try:
            time.sleep(0.25)
            env = make_env_for(symbol, args.curriculum, args.tail, "file")
            partial_test(env)
            env.ledger.to_csv(f"{args.output_dir}/env_{symbol}_test.csv")
        except Exception as e:
            print(e)
