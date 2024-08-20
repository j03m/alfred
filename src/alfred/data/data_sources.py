from torch.utils.data import Dataset
import torch
import pandas as pd


class DatasetStocks(Dataset):
    def __init__(self, input_file, sequence_length):
        super().__init__()
        self.df = pd.read_csv(input_file)
        self.num_symbols = self.df["Symbol"].nunique()
        self.drop = ["Date", "Symbol"]
        self.feature_columns = ["Close_diff_MA_7", "Volume_diff_MA_7", "Close_diff_MA_30", "Volume_diff_MA_30",
                                "Close_diff_MA_90", "Volume_diff_MA_90", "Close_diff_MA_180", "Volume_diff_MA_180",
                                "Close_diff_MA_360", "Volume_diff_MA_360", "Close", "Volume", "reportedEPS",
                                "estimatedEPS", "surprise", "surprisePercentage", "10year", "5year", "3year", "2year"]
        self.features = len(self.feature_columns)
        self.label_columns = ["price_change_term_4", "price_change_term_8", "price_change_term_12",
                              "price_change_term_24"]
        self.labels = len(self.label_columns)
        self.sequence_length = sequence_length

    # todo validate me
    def __getitem__(self, index):
        # Calculate the starting and ending indices for the sequence
        start_idx = index * self.num_symbols
        end_idx = start_idx + self.sequence_length * self.num_symbols

        # Extract the sequence of features (X) across all stocks for the given date range
        X_seq = self.df.loc[start_idx:end_idx - 1, self.feature_columns].values

        # Extract the sequence of labels (Y) across all stocks for the given date range
        Y_seq = self.df.loc[start_idx:end_idx - 1, self.label_columns].values

        # Reshape the sequences to have shape (seq_length, num_stocks, feature_dim)
        X_seq = X_seq.reshape(self.sequence_length, self.num_symbols, len(self.feature_columns))
        Y_seq = Y_seq.reshape(self.sequence_length, self.num_symbols, len(self.label_columns))

        # Convert to PyTorch tensors
        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        Y_seq = torch.tensor(Y_seq, dtype=torch.float32)

        return X_seq, Y_seq


    def __len__(self):
        return len(self.df)
