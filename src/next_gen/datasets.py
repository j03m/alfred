import torch
from torch.utils.data import Dataset

class BasicPandasDataset(Dataset):
    """Dataset wrapping data and target variables in a pandas dataframe."""

    def __init__(self, dataframe, feature_columns, target_column):
        """
        Args:
            dataframe (DataFrame): A pandas dataframe.
            feature_columns (list of str): The names of the columns in the dataframe to be used as features.
            target_column (str): The name of the column in the dataframe to be used as the target variable.
        """
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.target_column = target_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.dataframe.iloc[idx][self.feature_columns].values.astype('float32')
        target = self.dataframe.iloc[idx][self.target_column]
        sample = {'features': torch.from_numpy(features), 'target': torch.tensor(target, dtype=torch.float32)}

        return sample


def sliding_time_window(time_series, input_window, output_window):
    # length of time series - input - output windows + 1
    max_index = len(time_series) - (input_window - output_window + 1)

    # create a sliding window of sequences each predicting the next point.
    # if we have 4000 points, and we want 100 point predictions, we can have 3900 of these total
    # thus time_series.size(0) (the length) - iw
    # temp would end up being 3900 groups of (2, input_window)
    # first of the 2 groups is the input input_window and the 2nd would be the expected output window
    # even tho we may predict a subset of input_window in output_window we assume it's larger here
    # the last input are the number of features time_series.size(1)
    temp = torch.empty(max_index, 2, input_window, time_series.size(1))
    for i in range(max_index):
        temp[i][0] = time_series[i:i + input_window]
        temp[i][1] = time_series[i + output_window:i + input_window + output_window]

    return temp


class SlidingWindowPandasDataset(Dataset):

    def __init__(self, dataframe, feature_columns, input_window, output_window):
        """
        Args:
            dataframe (DataFrame): A pandas dataframe.
            feature_columns (list of str): The names of the columns in the dataframe to be used as features.
            input_window (int):  the size of the input window
            output_window (int): the size of the output window

        """
        features = dataframe[feature_columns].values.astype('float32')
        self.sliding_windows = sliding_time_window(torch.tensor(features), input_window, output_window)

    def __len__(self):
        return len(self.sliding_windows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        window = self.sliding_windows[idx][0], self.sliding_windows[idx][1]
        return window

class SlidingWindowDerivedOutputDataset(SlidingWindowPandasDataset):
    def __init__(self, dataframe, feature_columns, input_window, derivations):
        output_window = max(derivations)
        super().__init__(dataframe, feature_columns, input_window, output_window)

        # use sliding windows to build derivations
        self.derivations = []
        for window in derivations:



    def __len__(self):
        return len(self.sliding_windows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        window = self.sliding_windows[idx][0], self.sliding_windows[idx][1]
        return window