import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from statsmodels.tsa.seasonal import seasonal_decompose
from .logger import info, debug, error, verbose
from sklearn.preprocessing import MinMaxScaler
import math
from scipy.stats import norm
import pandas as pd

class TraderEnv(gym.Env):

    def __init__(self, product, df, curriculum_code=1):
        super(TraderEnv, self).__init__()

        # Define the bounds for each column
        close_min, close_max = 0, np.inf
        volume_min, volume_max = 0, np.inf
        trend_min, trend_max = 0, np.inf
        percentile_min, percentile_max = 0, 1

        # Define the observation space
        self.observation_space = spaces.Box(low=np.array([close_min, volume_min, trend_min, percentile_min]),
                                            high=np.array([close_max, volume_max, trend_max, percentile_max]),
                                            dtype=np.float32)

        # We have 3 actions, hold (0), long (1), short (2)
        self.action_space = spaces.Discrete(3)

        # Initialize environment state
        df = self.expand(df.copy())

        self.orig_timeseries = df
        self.timeseries = self.scale(df[["Date", "Close", "weighted-volume", "trend", "prob_above_trend"]])

        self._reset_vars()
        self.calculate_benchmark_metrics()

        # Don't add this here. Process the tickers into data files - you already have some.
        # self.lstm_internal = keras.models.load_model(model_path + "/lstm_cv-4.h15")
        # predictions = self.predict_trade(self.lstm_internal, self.timeseries[["Close", "weighted-volume"]])
        # self.timeseries["predicted-price"] = predictions

        self.product = product
        self.final = len(df)

        self.curriculum_code = curriculum_code
        self.rolling_score = 0
        self.prob_high = 0.8
        self.prob_low = 0.2
        self._expert_actions = []

    def expert_opinion(self):
        df = self.timeseries

        def assign_action(prob):
            if prob >= self.prob_high:
                return 1
            elif prob <= self.prob_low:
                return 2
            else:
                return 0

        state_action_data = []

        for index, row in df.iterrows():
            state = row.values  # Keep all columns, including 'prob_above_trend', as part of the state
            action = assign_action(row["prob_above_trend"])
            state_action_data.append((state, action))
            self._expert_actions.append(action)
        return state_action_data

    def expert_opinion_df(self):
        df = self.timeseries

        # iterate over the time series
        def assign_action(prob):
            if prob >= self.prob_high:
                return 1
            elif prob <= self.prob_low:
                return 2
            else:
                return 0

        # Create the new column 'action' based on the values in 'prob_above_trend'
        df['action'] = df['prob_above_trend'].apply(assign_action)
        self._expert_actions = df['action'].values

    def calculate_benchmark_metrics(self):
        df = self.orig_timeseries
        row = df.iloc[0, :]
        price = self.get_price_with_slippage(row["Close"])
        self.benchmark_position_shares = math.floor(self.cash / price)

    def predict_trade(self, model, X):
        predicted = model.predict(X, verbose=0).flatten()
        return predicted

    def expand(self, df):
        # Perform seasonal decomposition
        result = seasonal_decompose(df['Close'], model='additive', period=90, extrapolate_trend='freq')

        # Add trend back to original time series
        df["trend"] = result.trend

        # Compute the residuals by subtracting the trend from the original time series
        residuals = result.resid

        # Fit a Gaussian distribution to the residuals
        mu, std = norm.fit(residuals)

        # Compute the probability of a value being above or below the trend line
        # for each point in the time series
        z_scores = residuals / std
        df["prob_above_trend"] = 1 - norm.cdf(z_scores)
        df["weighted-volume"] = df["Close"] * df["Volume"]
        return df

    def scale(self, timeseries):
        df = timeseries.reset_index()  # Reset the index of the DataFrame
        dates = df['Date']
        data_to_scale = df.drop(['Date', 'index'], axis=1)
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data_to_scale)
        scaled_df = pd.concat([dates, pd.DataFrame(scaled_data, columns=data_to_scale.columns)], axis=1)
        return scaled_df.set_index('Date')  # Set the index back to 'Date'

    def _reset_vars(self):

        self._episode_ended = False
        self.ledger = self.make_ledger_row()
        self.slippage = .01
        self.fee = .0025
        self.current_index = 0
        self.cash = 5000
        self.position_shares = 0
        self.cash_from_short = 0
        self.position_value = 0
        self.shares_owed = 0
        self.in_position = False
        self.in_long = False
        self.in_short = False
        self.last_action = -1
        self.long_profit = []
        self.short_profit = []
        self.long_entry = -1
        self.short_entry = -1
        self.rolling_score = 0

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

        # Apply the action and update the environment state
        self._apply_action(action)

        if self._is_episode_ended():
            reward = self._get_reward2()
            info("final reward:", reward)
            return self._get_next_state(), reward, False, True, {}
            # return self._get_next_state(), reward, True, {}
        else:
            reward = self._get_reward2()
            info("current reward:", reward)
            self.current_index += 1
            return self._get_next_state(), reward, False, False, {}
            # return self._get_next_state(), reward, False, {}

    def _get_initial_state(self):
        # Return the initial state of the environment
        self.current_index = 0
        return self.env_block()

    def should_stop(self):
        # if cash is negative
        if (self.total_value() <= 0):
            error("Bankrupt.")
            return True
        return False

    def _apply_action(self, action):
        # Advance the environment by one time step and return the observation, reward, and done flag
        verbose("step:", "index:", self.current_index, " of: ", self.final - 1, "action: ", int(action))
        self.update_position_value()
        if (self.current_index >= self.final - 1):  # or  self.should_stop()
            error("********MARKING DONE", "index:", self.current_index, " of: ", self.final - 1, " cash: ", self.cash,
                  " value: ", self.position_value)
            if (self.position_shares != 0):
                verbose("done so closing position")
                self.close_position()
            self._episode_ended = True
        else:
            self._episode_ended = False

        # AI says hold
        if (action == 0):
            info("holding action = 0.")
            self.closed_position = False
        # AI says long but we're already in a position
        elif (action == 1 and self.in_long):
            info("holding long.")
            self.closed_position = False
        # AI says long, we're not in a position, so buy
        elif (action == 1 and not self.in_position):
            info("opening long.")
            self.open_position()
            self.closed_position = False
        # AI says long, but we're short. Close the short, open a long.
        elif (action == 1 and self.in_short):
            info("closing short to open long.")
            self.closed_position = True
            self.close_short()
            self.open_position()
        # AI says short, but we're already short
        elif (action == 2 and self.in_short):
            info("holding short.")
            self.closed_position = False
        # AI says short and we're not in a position so exit
        elif (action == 2 and not self.in_position):
            info("opening short.")
            self.open_short()
            self.closed_position = False
        # AI says short but we're long, close it
        elif (action == 2 and self.in_long):
            info("closing long to open short")
            self.close_position()
            info("opening short.")
            self.open_short()
            self.closed_position = True
        else:  # assume hold
            error("unknown state! holding:", action, self.in_position, self.in_long, self.in_short)
            self.closed_position = False
        self.update_position_value()

    def _get_next_state(self):
        # Calculate and return the next state based on the current state and action taken
        return self.env_block()

    def get_price_with_slippage(self, price):
        return price + (price * self.slippage)

    def make_ledger_row(self):
        ledger = pd.DataFrame()
        ledger["Date"] = []
        ledger["Side"] = []
        ledger["Action"] = []
        ledger["Price"] = []
        ledger["Profit"] = []
        ledger["Fee"] = []
        ledger["Value"] = []

        return ledger

    # todo you need to subtract fee from cash
    def open_position(self):
        self.in_position = True
        self.in_long = True
        df = self.orig_timeseries
        row = df.iloc[self.current_index, :]
        price = self.get_price_with_slippage(row["Close"])
        self.position_shares = math.floor(self.cash / price)
        fee = self.cash * self.fee
        self.cash = 0
        self.long_entry = price
        ledger_row = self.make_ledger_row()
        ledger_row["Date"] = [row["Date"]]
        ledger_row["Side"] = ["long"]
        ledger_row["Action"] = ["enter"]
        ledger_row["Price"] = [price]
        ledger_row["Fee"] = [fee]
        ledger_row["Profit"] = [0]
        ledger_row["Value"] = [self.total_value()]

        self.ledger = pd.concat([self.ledger, ledger_row])

    def open_short(self):
        self.in_position = True
        self.in_short = True
        df = self.orig_timeseries
        row = df.iloc[self.current_index, :]
        price = self.get_price_with_slippage(row["Close"])
        max_short_pos = math.floor(self.cash / price)
        self.shares_owed = max_short_pos
        self.cash_from_short = (self.shares_owed * price)
        fee = self.cash * self.fee
        self.cash = self.cash + self.cash_from_short
        self.short_entry = price
        verbose("Added cash on short: ", self.shares_owed * price, " total: ", self.cash, " took share debt:",
                self.shares_owed)
        ledger_row = self.make_ledger_row()
        ledger_row["Date"] = [row["Date"]]
        ledger_row["Side"] = ["short"]
        ledger_row["Action"] = ["enter"]
        ledger_row["Price"] = [price]
        ledger_row["Fee"] = [fee]
        ledger_row["Profit"] = [0]
        ledger_row["Value"] = [self.total_value()]
        self.ledger = pd.concat([self.ledger, ledger_row])

    def close_position(self):
        self.in_position = False
        self.in_long = False
        df = self.orig_timeseries
        row = df.iloc[self.current_index, :]
        price = self.get_price_with_slippage(row["Close"])
        value = price * self.position_shares
        fee = value * self.fee
        self.position_shares = 0
        self.cash = self.cash + value
        self.last_profit = (price - self.long_entry) / self.long_entry
        self.long_profit.append(self.last_profit)
        ledger_row = self.make_ledger_row()
        ledger_row["Date"] = [row["Date"]]
        ledger_row["Side"] = ["long"]
        ledger_row["Action"] = ["exit"]
        ledger_row["Price"] = [price]
        ledger_row["Fee"] = [fee]
        ledger_row["Profit"] = [self.last_profit]
        ledger_row["Value"] = [self.total_value()]
        self.ledger = pd.concat([self.ledger, ledger_row])

    def close_short(self):
        self.in_position = False
        self.in_short = False
        df = self.orig_timeseries
        row = df.iloc[self.current_index, :]
        price = self.get_price_with_slippage(row["Close"])
        value = price * self.shares_owed
        fee = value * self.fee
        self.shares_owed = 0
        self.cash_from_short = 0
        self.cash = self.cash - value
        self.last_profit = (self.short_entry - price) / self.short_entry
        self.short_profit.append(self.last_profit)
        ledger_row = self.make_ledger_row()
        ledger_row["Date"] = [row["Date"]]
        ledger_row["Side"] = ["short"]
        ledger_row["Action"] = ["exit"]
        ledger_row["Price"] = [price]
        ledger_row["Fee"] = [fee]
        ledger_row["Profit"] = [self.short_entry - price]
        ledger_row["Value"] = [self.total_value()]
        self.ledger = pd.concat([self.ledger, ledger_row])

    def _is_episode_ended(self):
        return self._episode_ended

    def update_position_value(self):
        df = self.orig_timeseries
        row = df.iloc[self.current_index, :]
        self.position_value = (row["Close"] * self.position_shares) - (row["Close"] * self.shares_owed)
        self.benchmark_value = row["Close"] * self.benchmark_position_shares

    def _get_reward2(self):
        '''
          What components should go into the reward?
          We can encode some of our desired behavior here, for example if we are at 99% chance of above trend we could reward long and vice versa
          We can remove cash from the reward calculation and simply treat it well we initially trade from. Any profits go into cash from trades, initial cash is penalized
          Consistency - making lots of profitable trades is better then big bets or not betting
        '''
        score = 0

        component1 = self.is_probable_set_up()

        return component1

    def calculate_consistency_bonus(self):
        profitable_longs = list(filter(lambda x: x >= 0, self.long_profit))
        profitable_shorts = list(filter(lambda x: x >= 0, self.short_profit))
        profitable_trades = len(profitable_longs) + len(profitable_shorts)
        max = len(self.long_profit) + len(self.short_profit)
        if max == 0:
            return 0
        return profitable_trades / max * 100

    def calculate_close_bonus(self):
        if self.closed_position and self.last_profit > 0:
            return self.last_profit * 10
        else:
            return 0

    def is_probable_set_up(self):
        df = self.orig_timeseries
        row = df.iloc[self.current_index, :]
        prob = row["prob_above_trend"]
        action = self.last_action

        print("action:", action, " prob:", prob)
        if action == 1 and prob >= self.prob_high:
            return 100  # reward a highly probable long
        elif action == 2 and prob <= self.prob_low:
            return 100  # reward a highly probable short
        elif action == 0 and (self.prob_low > prob or prob < self.prob_high):
            return 0.01  # reward holding when probability is moderate
        else:
            # raise Exception(f"WHY YOU WRONG? {self.current_index} {action} {prob}")
            return -10  # penalize for taking actions against the criteria

    def _get_reward(self):
        df = self.orig_timeseries
        row = df.iloc[self.current_index, :]
        current_portfolio_value = self.total_value()

        # compare to a bench mark
        percentage_change = ((current_portfolio_value - self.benchmark_value) / self.benchmark_value) * 100

        verbose("reward states: ",
                "\nposition value: ", self.position_value,
                "\nlong shares: ", self.position_shares,
                "\nshort shares: ", self.shares_owed,
                "\nlong value:", row["Close"] * self.position_shares,
                "\nshort debt:", row["Close"] * self.shares_owed,
                "\ncash value:", self.cash,
                "\ntoal value:", current_portfolio_value,
                "\nbenchmark value:", self.benchmark_value,
                "\nreal percentage_change: ", percentage_change
                )

        return percentage_change

    def total_value(self):
        self.update_position_value()
        return self.position_value + self.cash

    def env_block(self):
        start_index = self.current_index
        end_index = self.current_index
        df = self.timeseries.reset_index().drop(['Date'], axis=1)
        block = df.iloc[start_index].to_numpy()
        return block