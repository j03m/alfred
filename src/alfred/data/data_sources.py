from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from .features_and_labels import feature_columns, label_columns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class BaseYahooDataSet(Dataset):
    def __init__(self, stock, start, end, seq_length, change):
        self.df = None
        self.change = change
        self.seq_length = seq_length
        self.scaler = None
        self.data = self.fetch_data(stock, start, end)

    def fetch_data(self, ticker, start, end):
        self.df = yf.download(ticker, start=start, end=end)
        data = self.produce_data()
        return self.scale_data(data)

    def scale_data(self, data):
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
    def __init__(self, stock, start, end, seq_length, change):
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

        self.y = self.data[self.seq_length:]

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
    def __init__(self, stock, start, end, seq_length, change):
        self.close_scaler = MinMaxScaler()  # Standard min-max scaling (0, 1)
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
    def __init__(self, stock, start, end, seq_length, change):
        self.close_scaler = MinMaxScaler()  # Standard min-max scaling (0, 1)
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


class DatasetStocks(Dataset):
    def __init__(self, input_file, sequence_length):
        super().__init__()
        self.df = pd.read_csv(input_file)
        self.num_symbols = self.df["Symbol"].nunique()
        self.drop = ["Date", "Symbol"]
        self.feature_columns = feature_columns
        self.features = len(self.feature_columns)
        self.label_columns = label_columns
        self.labels = len(self.label_columns)
        self.sequence_length = sequence_length
        self.symbols = self.df['Symbol'].values

    def filter(self, symbol: str):
        self.df = self.df[(self.df['Symbol'] == symbol)].reset_index(drop=True)
        self.num_symbols = 1

    def trim_to_size(self, size: int):
        if len(self.df) > size:
            self.df = self.df.iloc[:size]
        else:
            raise ValueError(f"dataframe {len(self.df)} too small to be trimmed to {size}")

    def get_symbols(self, batch_idx, batch_size):
        # Calculate the start and end indices for the batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        # Retrieve the symbols for this batch
        return self.symbols[start_idx:end_idx]

    def __getitem__(self, index):
        # Calculate the starting and ending indices for the sequence
        start_idx = index * self.num_symbols
        end_idx = start_idx + self.sequence_length * self.num_symbols

        # Ensure we do not go out of bounds
        if end_idx > len(self.df):
            raise IndexError("Index range is out of bounds")

        # Extract the sequence of features (X) across all stocks for the given date range
        X_seq = self.df.iloc[start_idx:end_idx][self.feature_columns].values

        # Extract the sequence of labels (Y) across all stocks for the given date range
        Y_seq = self.df.iloc[start_idx:end_idx][self.label_columns].values

        # Reshape the sequences to have shape (seq_length, num_stocks, feature_dim)
        X_seq = X_seq.reshape(self.sequence_length, self.num_symbols, len(self.feature_columns))
        Y_seq = Y_seq.reshape(self.sequence_length, self.num_symbols, len(self.label_columns))

        # Convert to PyTorch tensors
        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        Y_seq = torch.tensor(Y_seq, dtype=torch.float32)

        return X_seq, Y_seq

    def __len__(self):
        return len(self.df) // (self.sequence_length * self.num_symbols)
