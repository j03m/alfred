#!/usr/bin/env python3

import pandas as pd
from stable_baselines3 import PPO
import yfinance as yf
import argparse
from machine_learning_finance import SimpleEnv, create_train_test_windows
from stable_baselines3.common.monitor import Monitor

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--symbol", help="Symbols to use")

args = parser.parse_args()

symbol = args.symbol

ticker_obj = yf.download(tickers=symbol, interval="1d")
df = pd.DataFrame(ticker_obj)
hist_df, test_df = create_train_test_windows(df, None, 365 * 4, None, 365)
env = SimpleEnv(test_df, hist_df)
env = Monitor(env)
save_path = "models/ppo_mlp_policy_simple_env"
model = PPO.load(save_path)

obs, info = env.reset()
states = None
done = False
while not done:
    action, states = model.predict(
        obs,  # type: ignore[arg-type]
        state=states
    )
    obs, reward, done, _, _ = env.step(action)
    print("reward is:", reward)

print(env.profit)