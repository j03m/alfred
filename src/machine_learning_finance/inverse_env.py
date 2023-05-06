import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from statsmodels.tsa.seasonal import seasonal_decompose
from .logger import info, debug, error, verbose
from sklearn.preprocessing import MinMaxScaler
import math
from scipy.stats import norm
import pandas as pd
from machine_learning_finance import generate_probability, attach_markers, calculate_duration_probabilities, \
    calc_durations_with_extremes
from .trader_env import TraderEnv


class InverseEnv(TraderEnv):

    def __init__(self, product, df, inverse_product, inverse_df):
        super(InverseEnv, self).__init__(product, df, 1)
        self.inverse_product = inverse_product
        self.inverse_df = inverse_df

    def _apply_action(self, action):
        # AI says hold
        if action == 0:
            info("holding")
            pass
        elif action == 1 and self.in_long:
            info("holding long")
            pass
        elif action == 1 and not self.in_position:
            info("opening long.")
            self.open_position()
        # AI says long, but we're short. Close the short, open a long.
        elif action == 1 and self.in_short:
            info("close short, open long")
            self.close_inverse()
            self.open_position()
            pass
        # AI says short, but we're already short
        elif action == 2 and self.in_short:
            info("hold short")
            pass
        # AI says short, we're not in a position so exit
        elif action == 2 and not self.in_position:
            info("opening short")
            self.open_inverse()
            pass
        # AI says short but we're long, close it
        elif action == 2 and self.in_long:
            info("closing long, opening short")
            self.closed_position()
        else:  # assume hold
            info("unknown - hold")
            pass
        self.update_position_value()

    def open_inverse(self):
        # opening a short position, is opening a long position on the inverse
        self._open_position(self.inverse_df, self.inverse_product)

    def close_inverse(self):
        pass

    def update_position_value(self):
        pass

    def total_value(self):
      pass

    def update_position_value(self):
        pass

    def total_value(self):
        self.update_position_value()
        return self.position_value + self.cash
