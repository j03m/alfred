#!/usr/bin/env python3

from machine_learning_finance import OptionTraderEnv
import yfinance as yf
import datetime
import pandas as pd


def make_env_for(symbol, code, tail=-1, head=-1, data_source="yahoo"):
    if data_source == "yahoo":
        ticker_obj = yf.download(tickers=symbol)
        df = pd.DataFrame(ticker_obj)
    elif data_source == "file":
        df = pd.read_csv(f"./data/{symbol}.csv")
    else:
        raise Exception("Implement me")
    if tail != -1:
        df = df.tail(tail)
    if head != -1:
        df = df.head(head)
    env = OptionTraderEnv(symbol, df, code)
    return env


env = make_env_for("SPY", 2, 365)

env.timeseries.to_csv("./check_options.csv")
print(env.timeseries)