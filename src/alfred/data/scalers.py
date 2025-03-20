import joblib
import re

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd

PM_SCALER_CONFIG = [
    {'regex': r'^Close$', 'type': 'yeo-johnson'},
    {'regex': r'.+_diff_.+', 'type': 'standard'},
    {'regex': r'^analyst_.+', 'type': 'yeo-johnson'},
    {'regex': r'^delta_.+', 'type': 'yeo-johnson'},
    {'columns': ['Institutional'], 'type': 'yeo-johnson'},
    {'columns': ['^VIX'], 'type': 'standard'},
    {'columns': ['SPY', 'CL=F', 'BZ=F'], 'type': 'yeo-johnson'},
    {'columns': ["BTC=F", "PM"], 'type': 'standard'},
    # We only have btc prices back to 2017 which leads to segments with no variation, which blows up yeo-johnson
    {'regex': r'^Margin.*', 'type': 'standard'},
    {'regex': r'^Volume$', 'type': 'yeo-johnson'},
    {'columns': ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage', 'insider_acquisition',
                 'insider_disposal', 'mean_outlook', 'mean_sentiment'], 'type': 'standard'},
    {'regex': r'\d+year', 'type': 'standard'},
    {'columns': ['ID', 'Rank', "PQ"], 'type': 'none'}
]

ANALYST_SCALER_CONFIG = [
    {'regex': r'^Close$', 'type': 'yeo-johnson'},
    {'columns': ['^VIX'], 'type': 'standard'},
    {'columns': ['SPY', 'CL=F', 'BZ=F'], 'type': 'yeo-johnson'},
    {'columns': ["BTC=F"], 'type': 'standard'},
    # We only have btc prices back to 2017 which leads to segments with no variation, which blows up yeo-johnson
    {'regex': r'^Margin.*', 'type': 'standard'},
    {'regex': r'^Volume$', 'type': 'yeo-johnson'},
    {'columns': ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage', 'insider_acquisition',
                 'insider_disposal', 'mean_outlook', 'mean_sentiment'], 'type': 'standard'},
    {'regex': r'\d+year', 'type': 'standard'}
]

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
                if scaler_type == "none":
                    self.scaler_mapping[column] = NoOpScaler()
                elif scaler_type == "standard":
                    self.scaler_mapping[column] = StandardScaler()
                elif scaler_type == "minmax":
                    self.scaler_mapping[column] = MinMaxScaler()
                elif scaler_type == "robust":
                    self.scaler_mapping[column] = RobustScaler()
                elif scaler_type == "log_returns":
                    self.scaler_mapping[column] = LogReturnScaler()
                elif scaler_type == "yeo-johnson":
                    self.scaler_mapping[column] = PowerTransformer(method='yeo-johnson')
                elif scaler_type == "log1p":
                    self.scaler_mapping[column] = SignedLog1pMinMaxScaler()
                else:
                    raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def fit(self, df):
        # verify no dataframe column lacks representation in the scaler
        for column in df.columns:
            if column not in self.scaler_mapping:
                raise Exception(f"Column {column} is not represented in the scaler config. Please specify all columns "
                                f"or (as we have painfully experienced in the past) columns will slip through unscaled"
                                f"and lead to much harder to deal with gradient issues.")

        for column, scaler in self.scaler_mapping.items():
            # only scale columns that are preset:
            if column in df.columns:
                scaler.fit(df[[column]].values)

    def transform(self, df, in_place=False):
        for column in df.columns:
            if column not in self.scaler_mapping:
                raise Exception(
                    f"(2) Column {column} is not represented in the scaler config. Please specify all columns "
                    f"or (as we have painfully experienced in the past) columns will slip through unscaled"
                    f"and lead to much harder to deal with gradient issues.")

        if not in_place:
            df = df.copy()

        for column, scaler in self.scaler_mapping.items():
            if column in df.columns:
                assert not df[column].isnull().any(), f"{column} has null before transform"
                temp_col = scaler.transform(df[[column]].values)
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
            df[[column]] = scaler.inverse_transform(df[[column]].values)
        return df

    def inverse_transform_column(self, column, data):
        scaler = self.scaler_mapping[column]
        return scaler.inverse_transform(data)

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


class NoOpScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        return np.asarray(X)  # Ensures it's still an array

    def inverse_transform(self, X):
        return np.asarray(X)  # Returns original data unchanged


class LogReturnScaler(BaseEstimator, TransformerMixin):
    '''
    LogReturnScaler - A scaler for converting prices to log returns and vice versa.

    Why Use Log Returns?
    - Stability, stationarity, and normalization, as described in financial modeling.
    '''

    def __init__(self, cumsum: bool = True, amplifier: int = 2):
        self.do_cumsum = cumsum
        self.amplifier = amplifier
        self.initial_price = None

    def fit(self, X, y=None):
        return self  # This scaler doesn't require fitting

    def transform(self, input_data):
        if isinstance(input_data, np.ndarray):
            input = pd.Series(input_data)
        else:
            input = input_data

        # Ensure input contains no negative or zero values
        assert not (input <= 0).any(), "This transform only supports data that is > 0, no negative or zero values"

        # Forward fill any NaNs for continuity
        cleaned = input.ffill()
        X = np.asarray(cleaned)
        self.initial_price = X[0]  # Store the initial price

        # Calculate log returns
        log_returns = np.diff(np.log(X), axis=0)  # Log returns
        if self.do_cumsum:
            log_returns = log_returns.cumsum()  # Apply cumulative sum if required
        if self.amplifier != 0:
            log_returns = self.amplifier * log_returns  # Apply amplification
        # todo: This keeps the length the same but is a problem for inverse. I didn't get the working though
        log_returns = np.append(log_returns, log_returns[-1])
        return np.expand_dims(log_returns, axis=1)  # Return transformed data

    def inverse_transform(self, X):
        raise NotImplementedError("doesn't work")


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


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
