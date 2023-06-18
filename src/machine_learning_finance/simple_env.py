import gymnasium as gym
from gymnasium import logger, spaces
import pandas as pd
import numpy as np

from .timeseries_analytics import calc_probabilties_without_lookahead
from sklearn.preprocessing import MinMaxScaler


class SimpleEnv(gym.Env):
    def __init__(self,
                 test_period_df: pd.DataFrame,
                 historical_period_df: pd.DataFrame):
        self.test_period_df = test_period_df
        self.historical_period_df = historical_period_df

        logger.info(f"test period len: {len(test_period_df)}")
        logger.info(f"hist period len: {len(historical_period_df)}")

        self.expanded_df = calc_probabilties_without_lookahead(test_period_df, historical_period_df)

        # This is the dataframe the AI will see as our environment we scale numbers to avoid huge prices diffs
        self.expanded_df = self.scale(self.expanded_df[['prob_above_trend', 'trend', 'trend-diff']])

        # Todo - we need to have a think here. We read some stuff about how the AI
        # could be improved if it had an understanding of history. We need to maybe
        # supply previous frames?
        high = np.array(
            [
                1,  # max probability
                np.finfo(np.float32).max,  # any possible value
                np.finfo(np.float32).max,  # any possible value
            ],
            dtype=np.float32,
        )

        low = np.array(
            [
                0,  # min probability
                0,  # min price is 0
                -np.finfo(np.float32).max,  # any possible negative value
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # simple 0 or 1 action space

        self.index = 0
        self.state = -1
        self.price = 0
        self.cash = 50000
        self.profit = 0

    def scale(self, timeseries):
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(timeseries)
        scaled_df = pd.DataFrame(scaled_data, columns=timeseries.columns, index=timeseries.index)
        return scaled_df

    def step(self, action):
        logger.info(f"step: {self.index}, action: {action}")
        env_row = self.expanded_df.iloc[self.index]
        real_row = self.test_period_df.iloc[self.index]
        price = real_row["Close"]
        prob_above_trend = env_row["prob_above_trend"]
        reward = 0
        if action == 0 and prob_above_trend >= 0.9:
            reward = 1
        elif action == 1 and prob_above_trend <= 0.1:
            reward = 1

        # selling a buy
        if self.state == 0 and action == 1:
            diff = price - self.price
            if diff >= 0:
                reward = 2
            else:
                reward = 0
            self.profit += self.cash * (diff / self.price)
            self.state = action
            self.price = price

        # buying back a short
        if self.state == 1 and action == 0:
            diff = self.price - price
            if diff >= 0:
                reward = 2
            else:
                reward = 0
            profit = self.cash * (diff / self.price)
            self.profit += profit
            self.state = action
            self.price = price

        self.index += 1
        if self.index >= len(self.expanded_df):
            terminated = True
            next_row = []
        else:
            next_row = self.expanded_df.iloc[self.index].values
            terminated = False

        if self.state == -1:
            self.state = action
            self.price = price

        return np.array(next_row, dtype=np.float64), reward, terminated, False, {}

    def reset(self):
        self.index = 0
        self.state = -1
        return np.array(self.expanded_df.iloc[0].values, dtype=np.float32), {}
