from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from .features_and_labels import feature_columns, label_columns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from alfred.utils.custom_scaler import LogReturnScaler



class BaseYahooDataSet(Dataset):
    def __init__(self, stock, start, end, seq_length, change, log_return_scaler=False):
        self.df = None
        self.change = change
        self.seq_length = seq_length
        self.scaler = None
        self.log_return_scaler = log_return_scaler
        self.data = self.fetch_data(stock, start, end)

    def fetch_data(self, ticker, start, end):
        self.df = yf.download(ticker, start=start, end=end)
        data = self.produce_data()
        return self.scale_data(data)

    def scale_data(self, data):
        if self.log_return_scaler:
            scaler = LogReturnScaler()
        else:
            scaler = MinMaxScaler()
        self.scaler = scaler  # Store the scaler if you need to inverse transform later
        return scaler.fit_transform(data)

    def produce_data(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

class YahooNextCloseWindowDataSet(BaseYahooDataSet):
    def __init__(self, stock, start, end, seq_length, change, log_return_scaler=False):
        super().__init__(stock, start, end, seq_length, change)
        n_row = self.data.shape[0] - self.seq_length + 1
        x = np.lib.stride_tricks.as_strided(self.data, shape=(n_row, self.seq_length),
                                            strides=(self.data.strides[0], self.data.strides[0]))
        self.x = np.expand_dims(x[:-1], 2)

        self.y = self.data[seq_length + change - 1:]

    def produce_data(self):
        data = self.df["Close"].values
        # perform windowing
        return data.reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class YahooChangeSeriesWindowDataSet(BaseYahooDataSet):
    def __init__(self, stock, start, end, seq_length, change):
        super().__init__(stock, start, end, seq_length, change)
        n_row = self.data.shape[0] - self.seq_length + 1
        x = np.lib.stride_tricks.as_strided(self.data, shape=(n_row, self.seq_length),
                                            strides=(self.data.strides[0], self.data.strides[0]))
        self.x = np.expand_dims(x[:-1], 2)

        self.y = self.data[seq_length + change - 1:]

    def scale_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Scaling (-1, 1)
        return self.scaler.fit_transform(data)

    def produce_data(self):
        self.df["Change"] = self.df['Close'].pct_change(periods=self.change).shift(
            periods=(-1 * self.change))
        self.df.dropna(inplace=True)
        data = self.df["Change"].values
        # perform windowing
        return data.reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class YahooChangeWindowDataSet(YahooNextCloseWindowDataSet):
    def __init__(self, stock, start, end, seq_length, change, log_return_scale=False):
        if log_return_scale:
            self.close_scaler = LogReturnScaler()  # Standard min-max scaling (0, 1)
        else:
            self.close_scaler = MinMaxScaler()
        self.change_scaler = MinMaxScaler(feature_range=(-1, 1))  # Scaling (-1, 1)
        super().__init__(stock, start, end, seq_length, change)
        n_row = self.data.shape[0] - self.seq_length + 1
        x = self.data[:, 0]
        y = self.data[:, 1]
        x = np.lib.stride_tricks.as_strided(x, shape=(n_row, self.seq_length),
                                            strides=(self.data.strides[0], self.data.strides[0]))
        self.x = np.expand_dims(x[:-1], 2)
        self.y = np.expand_dims(y, 1)

    def scale_data(self, data):
        close_scaled = self.close_scaler.fit_transform(data[:, [0]])
        change_scaled = self.change_scaler.fit_transform(data[:, [1]])
        scaled_data = np.hstack((close_scaled, change_scaled))
        return scaled_data

    def produce_data(self):
        self.df["Change"] = self.df['Close'].pct_change(periods=self.change).shift(
            periods=(-1 * self.change))
        self.df.dropna(inplace=True)
        data = self.df[['Close', 'Change']].values
        return data

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class YahooDirectionWindowDataSet(YahooNextCloseWindowDataSet):
    def __init__(self, stock, start, end, seq_length, change, log_return_scale=False):
        if log_return_scale:
            self.close_scaler = LogReturnScaler()  # Standard min-max scaling (0, 1)
        else:
            self.close_scaler = MinMaxScaler()
        self.dir_scaler = MinMaxScaler()
        super().__init__(stock, start, end, seq_length, change)
        n_row = self.data.shape[0] - self.seq_length + 1
        x = self.data[:, 0]
        y = self.data[:, 1]
        x = np.lib.stride_tricks.as_strided(x, shape=(n_row, self.seq_length),
                                            strides=(self.data.strides[0], self.data.strides[0]))
        self.x = np.expand_dims(x[:-1], 2)
        self.y = np.expand_dims(y, 1)

    def scale_data(self, data):
        close_scaled = self.close_scaler.fit_transform(data[:, [0]])
        dir_scaled = self.dir_scaler.fit_transform(data[:, [1]])
        scaled_data = np.hstack((close_scaled, dir_scaled))
        return scaled_data

    def produce_data(self):
        self.df["Change"] = self.df['Close'].pct_change(periods=self.change).shift(
            periods=(-1 * self.change))
        self.df["Direction"] = self.df["Change"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        self.df.dropna(inplace=True)
        return self.df[['Close', 'Direction']].values

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class YahooSeriesAsFeaturesWindowDataSet(YahooNextCloseWindowDataSet):
    def __getitem__(self, index):
        x = self.x[index].flatten()
        y = self.y[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)