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

# todo, implement this into sacred
def load_csv_files_and_apply_range(symbols, data_path, period_length, seed, bar_type, aggregation_config, date_column):
    train_data = {}

    # Iterate over all CSV files
    for symbol in symbols:
        # Load each CSV file into a DataFrame
        df = pd.read_csv(f"{data_path}/{symbol}_unscaled.csv")
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        # prior to range selection, if we have a value that isn't d (the assumption) we need to
        # the bars into groups for w or m.
        if bar_type == "w":
            training_set = df.resample('W-FRI').agg(aggregation_config)
        elif bar_type == "m":
            training_set = df.resample('ME').agg(aggregation_config)
        elif bar_type == "d":
            pass  # no aggregate
        else:
            raise Exception(f"{bar_type} is not supported.")

        # once we have the total length post aggregation we can proceed with range selection
        total_length = len(df)

        assert period_length <= total_length, f"Total available data for {symbol} is too short."



        # Choose consistent train and eval start dates based on the seed and lengths
        start = choose_train_range(symbol, seed, total_length, period_length)

        assert (total_length > start + period_length)

        # Subset the DataFrame to the train and eval ranges
        train_df = df.iloc[start:start + period_length]

        train_data[symbol] = train_df

    return train_data


def load_csv_file(symbol, data_path, bar_type, aggregation_config, date_column):

    # Load each CSV file into a DataFrame
    df = pd.read_csv(f"{data_path}/{symbol}_unscaled.csv")
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)

    # prior to range selection, if we have a value that isn't d (the assumption) we need to
    # the bars into groups for w or m.
    if bar_type == "w":
        training_set = df.resample('W-FRI').agg(aggregation_config)
    elif bar_type == "m":
        training_set = df.resample('ME').agg(aggregation_config)
    elif bar_type == "d":
        pass  # no aggregate
    else:
        raise Exception(f"{bar_type} is not supported.")

    return df
