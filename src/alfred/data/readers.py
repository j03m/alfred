import pandas as pd
import os


def read_processed_file(data_path_, symbol, fail_on_missing=False):
    return read_file(data_path_, f"{symbol}_processed.csv", fail_on_missing)


def read_symbol_file(data_path_, symbol, fail_on_missing=False):
    return read_file(data_path_, f"{symbol}.csv", fail_on_missing)


def read_file(data_path_, file, symbol, fail_on_missing=False):
    symbol_file = os.path.join(data_path_, file)
    data_df = None
    try:
        data_df = pd.read_csv(symbol_file)
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df.set_index('Date', inplace=True)
    except FileNotFoundError as fnfe:
        print(f"The file {symbol_file} was not found.")
        if fail_on_missing:
            raise fnfe
    except pd.errors.ParserError as pe:
        print(f"The file {symbol_file} could not be parsed as a CSV. Continuing")
        if fail_on_missing:
            raise pe
    return data_df
