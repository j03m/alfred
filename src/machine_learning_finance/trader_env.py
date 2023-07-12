import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from .logger import info, debug, error, verbose
from .timeseries_analytics import detect_change_points, compute_derivatives_between_change_points, \
    generate_max_profit_actions
from sklearn.preprocessing import MinMaxScaler
import math

import pandas as pd
from .defaults import DEFAULT_CASH, \
    DEFAULT_TOP_PERCENT, \
    DEFAULT_BOTTOM_PERCENT


class TraderEnv(gym.Env):

    def __init__(self,
                 product,
                 test_period_df,
                 historical_period_df,
                 curriculum_code=1,
                 cash=DEFAULT_CASH,
                 prob_high=DEFAULT_TOP_PERCENT,
                 prob_low=DEFAULT_BOTTOM_PERCENT):

        self.max_cash = cash
        self.benchmark_value = None
        self.position_value = None
        self.cash_from_short = None
        self.shares_owed = None
        self.last_profit = None
        self.last_percent_gain_loss = None
        self.cash = cash
        self.in_short = None
        self.position_shares = None
        self.in_long = None
        self.last_action = None
        self.scaler = None
        self.benchmark_position_shares = None
        self.env_version = "2"
        # Define the observation space
        # "trend", chng_pt_column, "polynomial_derivative", "trend-diff"
        high = np.array(
            [
                np.finfo(np.float32).max,  # any possible value
                1,  # boolean change point
                np.finfo(np.float32).max,  # any possible value
                np.finfo(np.float32).max,  # any possible value
            ],
            dtype=np.float32,
        )

        low = np.array(
            [
                0,  # min value of trend is 0
                0,  # boolean change point
                -np.finfo(np.float32).max,  # any possible negative value
                -np.finfo(np.float32).max,  # any possible negative value
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # We have 3 actions, hold (0), long (1), short (2)
        self.action_space = spaces.Discrete(3)

        temp_df = test_period_df.copy()

        # Get a moving average for the whole series, but tail it just to our test period and call it trend
        temp_df["trend"] = pd.concat([historical_period_df, temp_df])["Close"].rolling(30).mean().tail(
            len(test_period_df))

        # get a trend diff
        temp_df["trend-diff"] = temp_df["Close"] - temp_df["trend"]

        # detect change points
        df_change_points, chng_pt_column = detect_change_points(temp_df, moving_avg=temp_df["trend"])

        # detect the derivative of a polynomial between the points, this should indicate trend direction
        self.timeseries = compute_derivatives_between_change_points(df_change_points, chng_pt_column)

        # This is the dataframe the AI will see as our environment we scale numbers to avoid huge prices diffs
        self.timeseries = self.scale(self.timeseries[["trend", chng_pt_column, "polynomial_derivative", "trend-diff"]])

        # This is the dataframe we will use to calculate profits and generate curriculum guidance
        self.orig_timeseries = test_period_df
        self._reset_vars()

        self.product = product
        self.final = len(test_period_df)

        self.curriculum_code = curriculum_code
        self.rolling_score = 0
        self.prob_high = prob_high
        self.prob_low = prob_low
        self._expert_actions = []

        # we could apply other expert/proven strategies here? turtles etc
        self.generate_expert_opinion()

    def generate_expert_opinion(self):
        df = self.orig_timeseries
        self._expert_actions = generate_max_profit_actions(df["Close"], [5, 15, 30, 60], 5, 10)

    def calculate_benchmark_metrics(self):
        df = self.orig_timeseries
        row = df.iloc[0, :]
        price = self.get_price_with_slippage(row["Close"])
        self.benchmark_position_shares = math.floor(self.cash / price)

    def scale(self, timeseries):
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(timeseries)
        scaled_df = pd.DataFrame(scaled_data, columns=timeseries.columns, index=timeseries.index)
        return scaled_df

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value):
        assert (value >= 0)
        self._cash = value

    def _reset_vars(self):
        self._episode_ended = False
        self.ledger = self.make_ledger_row()
        self.slippage = .01
        self.fee = .0025
        self.current_index = 0
        self.cash = self.max_cash
        self.position_shares = 0
        self.cash_from_short = 0
        self.position_value = 0
        self.closed_position = False
        self.shares_owed = 0
        self.in_long = False
        self.in_short = False
        self.last_action = -1
        self.long_profit = []
        self.short_profit = []
        self.long_entry = -1
        self.short_entry = -1
        self.rolling_score = 0
        self.last_action = None
        self.last_profit = 0
        self.last_percent_gain_loss = 0
        self.calculate_benchmark_metrics()

    def reset(self):
        # Reset the environment and return the initial time step
        self._reset_vars()
        return self._get_next_state(), {}
        # return self._get_next_state()

    def reset_test(self):
        self._reset_vars()
        return self._get_next_state()

    def step(self, action):

        action = int(action)

        if len(self._expert_actions) > 0:
            info("_step:", self.current_index, " action: ", action, " expert action is: ",
                 self._expert_actions[self.current_index])
        else:
            info("_step:", self.current_index, " action: ", action)

        self.last_action = action
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        # Advance the environment by one time step and return the observation, reward, and done flag
        verbose("step:", "index:", self.current_index, " of: ", self.final - 1, "action: ", int(action))
        self.update_position_value()
        if self.current_index >= self.final - 1 or self.should_stop():
            info("********MARKING DONE", "index:", self.current_index, " of: ", self.final, " cash: ", self.cash,
                  " value: ", self.position_value)
            self.clear_trades()
            self._episode_ended = True
        else:
            self._episode_ended = False
            # Apply the action and update the environment state
            self._apply_action(action)

        if self._is_episode_ended():
            reward = self.get_reward(action)
            info("final reward:", reward)
            return self._get_next_state(), reward, False, True, {}
            # return self._get_next_state(), reward, True, {}
        else:
            reward = self.get_reward(action)
            info("current reward:", reward)
            self.current_index += 1
            return self._get_next_state(), reward, False, False, {}
            # return self._get_next_state(), reward, False, {}

    def clear_trades(self):
        if self.position_shares != 0:
            verbose("done so closing position")
            self.close_position()
        if self.shares_owed != 0:
            self.close_short()

    def _get_initial_state(self):
        # Return the initial state of the environment
        self.current_index = 0
        return self.env_block()

    def should_stop(self):
        # if cash is negative
        if self.total_value() <= 0:
            error("Bankrupt.")
            return True
        return False

    def _apply_action(self, action):

        # AI says hold
        if action == 0:
            info("holding action = 0.")

        # AI says long but we're already in a position
        elif action == 1 and self.in_long:
            info("holding long.")

        # AI says long, we're not in a position, so buy
        elif action == 1 and not self.in_position:
            info("opening long.")
            self.open_position()

        # AI says long, but we're short. Close the short, open a long.
        elif action == 1 and self.in_short:
            info("closing short to open long.")

            self.close_short()
            self.open_position()
        # AI says short, but we're already short
        elif action == 2 and self.in_short:
            info("holding short.")

        # AI says short, we're not in a position so exit
        elif action == 2 and not self.in_position:
            info("opening short.")
            self.open_short()

        # AI says short but we're long, close it
        elif action == 2 and self.in_long:
            info("closing long to open short")
            self.close_position()
            info("opening short.")
            self.open_short()

        else:  # assume hold
            error("unknown state! holding:", action, self.in_position, self.in_long, self.in_short)

        self.update_position_value()

    def _get_next_state(self):
        # Calculate and return the next state based on the current state and action taken
        block = self.env_block()
        return block

    def get_price_with_slippage(self, price):
        return price + (price * self.slippage)

    def make_ledger_row(self):
        ledger = pd.DataFrame()
        ledger["Date"] = []
        ledger["Product"] = []
        ledger["Side"] = []
        ledger["Action"] = []
        ledger["Profit_Percent"] = []
        ledger["Profit_Actual"] = []
        ledger["Fee"] = []
        ledger["Value"] = []

        return ledger

    @property
    def in_position(self):
        return self.in_long or self.in_short

    def open_position(self):
        df = self.orig_timeseries
        self._open_position(df, self.product)

    def _open_position(self, df, product):
        # row
        row = df.iloc[self.current_index, :]
        price = self.get_price_with_slippage(row["Close"])

        # fee and cash management
        fee = self.cash * self.fee
        self.cash -= fee
        self.position_shares = shares = math.floor(self.cash / price)
        cost = shares * price
        self.cash -= cost

        # state and tracking
        self.last_profit = 0
        self.long_entry = price
        self.in_long = True
        self.add_ledger_row(fee, price, row, "long", "enter", product, 0, 0, shares, cost)

    def close_position(self):

        self.in_long = False
        df = self.orig_timeseries
        self._close_position(df, self.product)

    def _close_position(self, df, product):
        row = df.iloc[self.current_index, :]
        price = self.get_price_with_slippage(row["Close"])

        # transaction size and fee
        value = price * self.position_shares
        fee = value * self.fee

        # cash management

        self.cash = self.cash + (value - fee)

        # state and tracking
        self.in_long = False
        self.last_profit = (self.position_shares * price) - ((self.position_shares * self.long_entry) + fee)
        self.position_shares = 0
        self.last_percent_gain_loss = (price - self.long_entry) / self.long_entry * 100
        self.long_entry = -1
        self.add_ledger_row(fee, price, row, "long", "exit", product, self.last_profit, self.last_percent_gain_loss,
                            self.position_shares, 0)

    def add_ledger_row(self,
                       fee,
                       price,
                       row,
                       side,
                       action,
                       product,
                       profit,
                       percent,
                       shares,
                       cost):
        ledger_row = self.make_ledger_row()
        ledger_row["Date"] = [row.name]
        ledger_row["Product"] = [product]
        ledger_row["Side"] = [side]
        ledger_row["Action"] = [action]
        ledger_row["Price"] = [price]
        ledger_row["Shares"] = [shares]
        ledger_row["Fee"] = [fee]
        ledger_row["Cost"] = [cost]
        ledger_row["Profit_Percent"] = [percent]
        ledger_row["Profit_Actual"] = [profit]
        ledger_row["Value"] = [self.total_value()]
        verbose(ledger_row)
        self.ledger = pd.concat([self.ledger, ledger_row])

    def open_short(self):
        df = self.orig_timeseries
        self._open_short(df, self.product)

    def _open_short(self, df, product):
        row = df.iloc[self.current_index, :]
        # reduce cash by fee
        fee = self.cash * self.fee
        self.cash -= fee

        # get max short for current cash
        price = self.get_price_with_slippage(row["Close"])
        max_short_pos = math.floor(self.cash / price)
        self.shares_owed = max_short_pos

        # track cash from short
        self.cash_from_short = self.shares_owed * price

        # pay oursleves
        self.cash = self.cash + self.cash_from_short

        # state
        self.short_entry = price
        self.in_short = True
        self.last_profit = 0
        verbose("Added cash on short: ", self.shares_owed * price, " total: ", self.cash, " took share debt:",
                self.shares_owed)

        # ledger
        self.add_ledger_row(fee, price, row, "short", "enter", product, 0, 0, 0, 0)

    def close_short(self):
        df = self.orig_timeseries
        self._close_short(df, self.product)

    def _close_short(self, df, product):
        # get price
        row = df.iloc[self.current_index, :]
        price = self.get_price_with_slippage(row["Close"])

        # assets transaction size and cost
        value = price * self.shares_owed
        fee = value * self.fee
        cost = value + fee

        # profit is our short cash minus cost
        self.last_profit = self.cash_from_short - cost
        self.shares_owed = 0
        self.cash_from_short = 0

        # cash on hand lowers from cost
        self.cash = self.cash - cost

        # profit
        self.last_percent_gain_loss = ((self.short_entry - price) / self.short_entry) * 100

        # state
        self.short_entry = -1
        self.in_short = False
        self.add_ledger_row(fee, price, row, "short", "exit", product, self.last_profit, self.last_percent_gain_loss,
                            self.shares_owed, cost)

    def _is_episode_ended(self):
        return self._episode_ended

    def update_position_value(self):
        df = self.orig_timeseries
        self.position_value = self.get_position_value(df, self.current_index)
        self.benchmark_value = self.get_bench_mark_value()

    def get_bench_mark_value(self):
        row = self.orig_timeseries.iloc[self.current_index, :]
        return row["Close"] * self.benchmark_position_shares

    def get_position_value(self, df, index):
        close = self.get_current_close(df, index)
        return (close * self.position_shares) - (close * self.shares_owed)

    def get_current_close(self, df, index):
        row = df.iloc[index, :]
        return row["Close"]

    def get_reward(self, action):
        index = self.current_index
        correct_action = self._expert_actions[index]
        if action == correct_action:
            return 1
        else:
            return 0

    def total_value(self):
        self.update_position_value()
        return self.position_value + self.cash

    def env_block(self):
        start_index = self.current_index
        df = self.timeseries
        block = df.iloc[start_index].to_numpy()
        return block

    def __str__(self):
        return f"\nposition value: {self.position_value}" \
               f"\nlong shares: {self.position_shares}" \
               f"\nshort shares:  {self.shares_owed}" \
               f"\ncash value: {self.cash}" \
               f"\nbenchmark value: {self.benchmark_value}"
