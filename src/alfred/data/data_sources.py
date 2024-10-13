from sympy import false
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
# from .features_and_labels import feature_columns, label_columns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from alfred.utils.custom_scaler import LogReturnScaler
from alfred.utils import CustomScaler
from .range_selection import load_csv_files_and_apply_range

# added this flag to go live (yahoo) or cache (file) due to network issues
LIVE = false
TICKER = "AAPL"


def filter_by_date_range(df, start_date, end_date):
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Convert start_date and end_date to pd.Timestamp for comparison
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Check if the start and end dates are within the DataFrame's index range
    if not (start in df.index or df.index.min() <= start <= df.index.max()):
        raise ValueError(f"Start date {start_date} is out of range.")
    if not (end in df.index or df.index.min() <= end <= df.index.max()):
        raise ValueError(f"End date {end_date} is out of range.")

    # Filter the DataFrame within the date range
    filtered_df = df.loc[start:end]

    # Raise an exception if the resulting DataFrame is empty
    if filtered_df.empty:
        raise ValueError(f"No data available between {start_date} and {end_date}.")

    return filtered_df


class YahooNextCloseWindowDataSet(Dataset):
    def __init__(self, stock, start, end, seq_length, change, log_return_scaler=False):
        self.df = None
        self.change = change
        self.seq_length = seq_length
        self.scaler = None
        self.log_return_scaler = log_return_scaler
        self.data = self.fetch_data(stock, start, end)
        n_row = self.data.shape[0] - self.seq_length + 1
        x = np.lib.stride_tricks.as_strided(self.data, shape=(n_row, self.seq_length),
                                            strides=(self.data.strides[0], self.data.strides[0]))
        self.x = np.expand_dims(x[:-1], 2)

        self.y = self.data[seq_length + change - 1:]

    def fetch_data(self, ticker, start, end):
        if LIVE:
            self.df = yf.download(ticker, start=start, end=end)
        else:
            df = pd.read_csv(f"./data/{TICKER}.csv")
            date_column = "Date"
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
            self.df = filter_by_date_range(df, start, end)
        data = self.produce_data()
        return self.scale_data(data)

    def scale_data(self, data):
        if self.log_return_scaler:
            scaler = LogReturnScaler()
        else:
            scaler = MinMaxScaler()
        self.scaler = scaler  # Store the scaler if you need to inverse transform later
        return scaler.fit_transform(data).reshape(-1, 1)

    def produce_data(self):
        data = self.df["Close"].values
        # perform windowing
        return data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class CachedStockDataSet(Dataset):
    def __init__(self, symbol, seed, length, sequence_length, feature_columns, target_columns,
                 scaler_config, range_provider, change=1,
                 date_column="Unnamed: 0", data_path="./data"):


        # i wrote this to get many files, but then decided I would only train one series at a time
        # so we just take 
        training_sets = load_csv_files_and_apply_range([symbol], data_path, length, seed, date_column).values()[0]


            self.scaler = CustomScaler(scaler_config, training_set)
        self.df = self.scaler.fit_transform(self.orig_df)
        assert not self.df.isnull().any().any(), f"scaled df has null after transform"

        self.seq_length = sequence_length
        features = self.df[feature_columns].values
        targets = self.df[target_columns].values
        n_row = features.shape[0] - self.seq_length + 1
        x = np.lib.stride_tricks.as_strided(features,
                                            shape=(n_row, self.seq_length, len(feature_columns)),
                                            strides=(features.strides[0], features.strides[0], features.strides[1]))
        self.x = x[:-1]
        self.y = targets[self.seq_length + change - 1:]
        self.data = features

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]  # Get the input sequence
        y = self.y[index]  # Get the target value
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
