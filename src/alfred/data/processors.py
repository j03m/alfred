import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def attach_moving_average_diffs(data: pd.DataFrame, moving_averages_days=[7, 30, 90, 180, 360]) -> pd.DataFrame:
    # Initialize an empty list to store the new column names
    new_columns = []
    for ma in moving_averages_days:
        close_col_name = f'Close_diff_MA_{ma}'
        volume_col_name = f'Volume_diff_MA_{ma}'

        data[close_col_name] = data['Close'] - data['Close'].rolling(window=ma).mean()
        data[volume_col_name] = data['Volume'] - data['Volume'].rolling(window=ma).mean()

        # Add the new column names to the list
        new_columns.extend([close_col_name, volume_col_name])

    return data, new_columns


def scale_relevant_training_columns(data: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    scaled_df = data.copy()

    scaled_df = scaled_df[columns_to_scale]

    # Initialize the scaler
    scaler = StandardScaler()

    # Scale each specified column independently
    for col in columns_to_scale:
        scaled_df[col] = scaler.fit_transform(scaled_df[col].values.reshape(-1, 1))

    return scaled_df, scaler