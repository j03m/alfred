from torch.utils.data import Dataset
import pandas as pd


class DatasetStocks(Dataset):
    def __init__(self, input_file):
        super().__init__()
        self.df = pd.read_csv(input_file)
        self.num_symbols = self.df["Symbol"].nunique()
        self.drop = ["Date", "Symbol"]
        self.feature_columns = ["Close_diff_MA_7", "Volume_diff_MA_7", "Close_diff_MA_30", "Volume_diff_MA_30",
                                "Close_diff_MA_90", "Volume_diff_MA_90", "Close_diff_MA_180", "Volume_diff_MA_180",
                                "Close_diff_MA_360", "Volume_diff_MA_360", "Close", "Volume", "reportedEPS",
                                "estimatedEPS", "surprise", "surprisePercentage", "10year", "5year", "3year", "2year"]
        self.label_columns = ["price_change_term_4", "price_change_term_8", "price_change_term_12",
                              "price_change_term_24"]

    def __getitem__(self, index):
        # Extract features (X) for the given index
        X = self.df.loc[index, self.feature_columns].values

        # Extract labels (Y) for the given index
        Y = self.df.loc[index, self.label_columns].values

        # Optionally convert X and Y to torch tensors if needed
        # X = torch.tensor(X, dtype=torch.float32)
        # Y = torch.tensor(Y, dtype=torch.float32)

        return X, Y

    def __len__(self):
        return len(self.df)
