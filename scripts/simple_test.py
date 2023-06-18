#!/usr/bin/env python3

import pandas as pd
from stable_baselines3 import PPO
import yfinance as yf

from machine_learning_finance import SimpleEnv, create_train_test_windows

# handles UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
from stable_baselines3.common.monitor import Monitor

# todo add args for tickers and time frames

ticker_obj = yf.download(tickers="SPY", interval="1d")
df = pd.DataFrame(ticker_obj)
hist_df, test_df = create_train_test_windows(df, None, 365 * 4, None, 365)
env = SimpleEnv(test_df, hist_df)
env = Monitor(env)
save_path = "../models/ppo_mlp_policy_simple_env"
model = PPO.load(save_path)

obs, info = env.reset()
states = None
done = False
while not done:
    actions, states = model.predict(
        obs,  # type: ignore[arg-type]
        state=states
    )
    obs, reward, done, _, _ = env.step(actions)
    print("reward is:", reward)

print(env.profit)