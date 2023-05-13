#!/usr/bin/env python3

from machine_learning_finance import OptionTraderEnv, make_env_for
import yfinance as yf
import datetime
import pandas as pd

_env = make_env_for("SPY", 2, env_class=OptionTraderEnv)

_env.timeseries.to_csv("./check_options.csv")
print(_env.timeseries)