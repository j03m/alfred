#!/usr/bin/env python3
import argparse
import warnings
from machine_learning_finance import guided_training, make_env_for, partial_test, partial_train

# filter out UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--symbol", default="SPY", help="Symbol to use (default: SPY)")
parser.add_argument("--tail", type=int, default=2500, help="Tail size (default: 2500)")
parser.add_argument("--guide", action="store_true", help="Enable guided training")
parser.add_argument("--train", action="store_true", help="Enable partial training")
parser.add_argument("--test", action="store_true", help="Enable partial testing")
parser.add_argument("--steps", type=int, default=25000, help="Number of steps (default: 25000)")
parser.add_argument("--create", action="store_true", help="Create new environment")
parser.add_argument("--output-dir", default="./data", help="Output directory (default: ./data)")

args = parser.parse_args()

print(args)

if args.guide:
    env1 = make_env_for(args.symbol, 1, args.tail)
    env1 = guided_training(args.symbol, args.create, args.steps, args.tail)
    env1.ledger.to_csv(f"{args.output_dir}/env1.csv")

if args.train:
    env2 = make_env_for(args.symbol, 1, args.tail)
    partial_train(env2, args.steps, args.create)
    env2.ledger.to_csv(f"{args.output_dir}/env2.csv")

if args.test:
    env3 = make_env_for(args.symbol, 1, args.tail)
    partial_test(env3)
    env3.ledger.to_csv(f"{args.output_dir}/env3.csv")
