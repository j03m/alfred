import pandas as pd
import numpy as np
import random
def consistent_rand_for_symbol(symbol, seed, start, end):
    combined = str(seed) + symbol
    rnd = random.Random(combined)
    return rnd.randint(start, end)

# probably never going to use this actually :/ keeping cuz I'm a code hoarder
def choose_eval_range(symbol, seed, data_length, training_length, training_start, eval_length):
    np.random.seed(seed)
    # select a window AFTER the training window
    if training_start + training_length + eval_length < data_length:
        return consistent_rand_for_symbol(symbol, seed, training_start + training_length, data_length)
    # otherwise select a window before the training window
    elif eval_length < training_start:
        return consistent_rand_for_symbol(symbol, seed, 0, training_start - eval_length)
    else:
        raise Exception("No suitable range outside the training window")

def choose_train_range(symbol, seed, data_length, training_length):
    return consistent_rand_for_symbol(symbol, seed, 0, data_length - training_length)


def load_csv_files_and_apply_range(symbols, data_path, period_length, seed, date_column):
    train_data = {}

    # Iterate over all CSV files
    for symbol in symbols:
        # Load each CSV file into a DataFrame
        df = pd.read_csv(f"{data_path}/{symbol}_unscaled.csv")
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        total_length = len(df)

        # Choose consistent train and eval start dates based on the seed and lengths
        start = choose_train_range(symbol, seed, total_length, period_length)

        # Subset the DataFrame to the train and eval ranges
        train_df = df.iloc[start:start + period_length]

        train_data[symbol] = train_df

    return train_data
