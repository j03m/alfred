from torch.utils.data import Dataset
import torch
import pandas as pd

from .features_and_labels import feature_columns, label_columns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class SimpleYahooCloseChangeDataset(Dataset):
    def __init__(self, stock, start, end, seq_length):
        self.df = None
        self.data = self.fetch_data(stock, start, end)
        self.seq_length = seq_length

    def fetch_data(self, ticker, start, end):
        self.df = yf.download(ticker, start=start, end=end)
        self.df["Change"] = self.df['Close'].pct_change(periods=30).shift(
            periods=(-1 * 30))
        self.df.dropna(inplace=True)
        data = self.df[['Close', 'Change']].values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = scaler  # Store the scaler if you need to inverse transform later
        return scaler.fit_transform(data)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        sequence = self.data[index:index + self.seq_length]
        x = sequence[:, 0]  # Select only the first column (Close prices)
        y = sequence[-1, 1]  # Select the second column (Change) of the last item in the sequence
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


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
