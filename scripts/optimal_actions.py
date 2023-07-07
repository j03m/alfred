#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import argparse
from machine_learning_finance import generate_max_profit_actions

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbol", default="SPY", help="Symbol to use (default: SPY)")
parser.add_argument("-t", "--tail", type=int, default=365, help="Tail size (default: 365)")
parser.add_argument("-f", "--file", type=str, default="./backtests/actions.csv",
                    help="Out file default data/actions.csv")
args = parser.parse_args()

ticker_obj = yf.download(tickers=args.symbol)
df = pd.DataFrame(ticker_obj).tail(args.tail)
final_actions = generate_max_profit_actions(df["Close"], [10, 15, 20, 25, 30], 5, 5)

df["actions"] = final_actions
df[["actions", "Close"]].to_csv(args.file)
