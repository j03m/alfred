import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import joblib
import re


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LogReturnScaler(BaseEstimator, TransformerMixin):
    '''
    Experimental, I've not done this before

    Why Use Log Returns?
    	1.	Stability: Log returns remove the dependency on the absolute price and focus on the percentage change, which helps reduce the effect of large price differences.
    	2.	Stationarity: Log returns often make a time series more stationary (a necessary condition for many statistical models), as the distribution of returns tends to be more stable over time compared to raw prices.
    	3.	Normalization: It eliminates the scale effects from stocks with different price ranges (e.g., a $5 stock versus a $500 stock).

    Why Use cumsum and amplifier:
        1. One common scenario where you would see .cumsum() used is when dealing with log returns (or percentage returns). In financial modeling, log returns are often used as features because they are more stationary and easier to model. If you’re converting back to prices (for interpretability or prediction purposes), applying .cumsum() makes sense:
	    2. Log returns are additive, so summing them over time gives you the log of the cumulative price change. This allows you to recover the actual price trajectory.
	    3. In this case, .cumsum() is a well-known technique to revert the log returns back to a price-like series, and this is common in finance.

    '''

    def __init__(self, cumsum: bool = True, amplifier: int = 2):
        self.do_cumsum = cumsum
        self.amplifier = amplifier
        # used to capture interim steps so they can be graphed
        self.cumsum = None
        self.log_returns = None
        self.amplified = None
        self.original = None

    def fit(self, X, y=None):
        return self  # This scaler doesn't require fitting

    def transform(self, input, y=None):
        # zeros negative values and nulls will break this
        assert(input.where((input<0)), "this transform only support data that is >=0 no negative numbers")
        scrub_condition = (input == 0) | (input.isnull())
        cleaned = input.where(~scrub_condition).ffill()
        X = np.asarray(cleaned)
        self.original = X
        # todo you still have the off by 1 issue here, append the last value to the end
        X = self.log_returns = np.diff(np.log(X), axis=0)  # Log returns
        if self.do_cumsum:
            X = self.cumsum = X.cumsum()
        if self.amplifier != 0:
            X = self.amplified = self.amplifier * X
        X = np.append(X, X[-1])
        return np.expand_dims(X, axis=1)  # to match what minmax scaler produces

    def inverse_transform(self, X, initial_price=None):
        if initial_price is None:
            raise ValueError("initial_price is required for inverse_transform to recover prices.")
        X = np.asarray(X)
        prices = np.exp(np.cumsum(X, axis=0))  # Reverse log returns by cumulative sum
        return np.vstack([initial_price, initial_price * prices])  # Recover prices by starting with the initial price


class SignedLog1pMinMaxScaler(BaseEstimator, TransformerMixin):
    '''
    Why use sign log1p min max? I think log returns might be better, but this was meant to reduce vol numbers
    '''

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.minmax_scaler = MinMaxScaler(feature_range=feature_range)

    def fit(self, X, y=None):
        # Apply signed log1p transformation and then fit the MinMaxScaler
        X_transformed = signed_log1p(X)
        self.minmax_scaler.fit(X_transformed)
        return self

    def transform(self, X, y=None):
        # Apply signed log1p transformation and then MinMax scaling
        X_transformed = signed_log1p(X)
        return self.minmax_scaler.transform(X_transformed)

    def inverse_transform(self, X, y=None):
        # First inverse the MinMax scaling, then apply the inverse of the signed log1p
        X_inverse_scaled = self.minmax_scaler.inverse_transform(X)
        return np.sign(X_inverse_scaled) * (np.expm1(np.abs(X_inverse_scaled)))


class CustomScaler:
    def __init__(self, config, df):
        self.config = config
        self.scaler_mapping = {}
        self.scalers = {}
        self._process_config(df)

    def _process_config(self, df):
        for entry in self.config:
            scaler_type = entry.get('type', 'standard')
            columns = entry.get('columns', [])
            regex_pattern = entry.get('regex', None)

            if regex_pattern:
                pattern = re.compile(regex_pattern)
                matched_columns = [col for col in df.columns if pattern.match(col)]
                columns.extend(matched_columns)

            for column in columns:
                if scaler_type == "standard":
                    self.scaler_mapping[column] = StandardScaler()
                elif scaler_type == "minmax":
                    self.scaler_mapping[column] = MinMaxScaler()
                elif scaler_type == "robust":
                    self.scaler_mapping[column] = RobustScaler()
                elif scaler_type == "log_returns":
                    self.scaler_mapping[column] = LogReturnScaler()
                elif scaler_type == "log1p":
                    self.scaler_mapping[column] = SignedLog1pMinMaxScaler()
                else:
                    raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def fit(self, df):

        for column, scaler in self.scaler_mapping.items():
            scaler.fit(df[[column]])

    def transform(self, df, in_place=False):
        if not in_place:
            df = df.copy()

        for column, scaler in self.scaler_mapping.items():
            assert not df[column].isnull().any(), f"{column} has null before transform"
            temp_col = scaler.transform(df[[column]])
            df[column] = temp_col
            assert not df[column].isnull().any(), f"{column} has null after transform"
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_scaler(self, column_name):
        if column_name in self.scaler_mapping:
            return self.scaler_mapping[column_name]
        else:
            raise ValueError(f"No scaler found for column: {column_name}")

    def inverse_transform(self, df):
        for column, scaler in self.scaler_mapping.items():
            df[[column]] = scaler.inverse_transform(df[[column]])
        return df

    def serialize(self, path):
        # Save the actual scalers along with the mappings
        joblib.dump({
            "config": self.config,
            "scaler_mapping": self.scaler_mapping,
            "scalers": {col: scaler for col, scaler in self.scaler_mapping.items()}
        }, path)

    @staticmethod
    def load(path):
        data = joblib.load(path)
        custom_scaler = CustomScaler(data['config'])
        custom_scaler.scaler_mapping = data['scalers']
        return custom_scaler

    def get_scaled(self, df):
        return self.transform(df.copy())

# # Example usage
# config = [
#     {'columns': ['a', 'b', 'c'], 'type': 'standard', 'augment': None},
#     {'columns': ['d', 'e'], 'type': 'minmax', 'augment': 'log'},
#     {'regex': r'^f.*', 'type': 'standard'}
# ]
#
# # Assuming df is your DataFrame
# scaler = CustomScaler(config)
# scaled_df = scaler.fit_transform(df)
# scaler.serialize('scalers.pkl')
