import pandas as pd
import numpy as np


def choose_train_eval_range(seed, total_length, train_length, eval_length):
    np.random.seed(seed)

    # Ensure there's enough space for both training and evaluation periods with no overlap
    if train_length + eval_length > total_length:
        raise ValueError("Total length of the dataset is too short for the given train and eval lengths.")

    # Randomly choose a start point for the training period
    max_train_start = total_length - train_length - eval_length  # Reserve space for eval period after training
    train_start = np.random.randint(0, max_train_start)

    # This is not what we want
    eval_start = train_start + train_length

    return train_start, eval_start


def load_csv_files_and_apply_range(csv_files, train_length, eval_length, seed, date_column):
    train_eval_data = {}

    # Iterate over all CSV files
    for file_path in csv_files:
        # Load each CSV file into a DataFrame
        df = pd.read_csv(file_path)
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        total_length = len(df)

        # Choose consistent train and eval start dates based on the seed and lengths
        train_start, eval_start = choose_train_eval_range(seed, total_length, train_length, eval_length)

        # Subset the DataFrame to the train and eval ranges
        train_df = df.iloc[train_start:train_start + train_length]
        eval_df = df.iloc[eval_start:eval_start + eval_length]

        train_eval_data[file_path] = {
            'train': train_df,
            'eval': eval_df
        }

    return train_eval_data


# # Example usage
# csv_files = ['file1.csv', 'file2.csv']  # Replace with actual file paths
# train_length = 100  # Example: 100 days for training
# eval_length = 50  # Example: 50 days for evaluation
# seed = 42  # Fixed seed for reproducibility
#
# train_eval_data = load_csv_files_and_apply_range(csv_files, train_length, eval_length, seed)
#
# # Access the training and evaluation data for each file
# for file, data in train_eval_data.items():
#     print(f"Training data for {file}:", data['train'].head())
#     print(f"Evaluation data for {file}:", data['eval'].head())