import gym
from gym import logger, spaces
import pandas as pd
import numpy as np

from .timeseries_analytics import calc_probabilties_without_lookahead
from sklearn.preprocessing import MinMaxScaler


class SimpleEnv(gym.Env):
    def __init__(self,
                 product: str,
                 test_period_df: pd.DataFrame,
                 historical_period_df: pd.DataFrame,
                 cash: int = 5000):
        self.starting_cash = cash
        self.test_period_df = test_period_df
        self.historical_period_df = historical_period_df
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

    def scale(self, timeseries):
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(timeseries)
        scaled_df = pd.DataFrame(scaled_data, columns=timeseries.columns, index=timeseries.index)
        return scaled_df
