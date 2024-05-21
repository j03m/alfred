from next_gen import train_tr_forecaster
import pandas as pd
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help="Symbols to use separated by comma")
    parser.add_argument('--symbol-file', type=str, default="./lists/training_list.csv", help="List of symbols in a file")
    parser.add_argument('--data', type=str, default="./data", help="data dir (./data)")

    args = parser.parse_args()
    symbols = []
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols += pd.read_csv(args.symbol_file)["Symbols"].tolist()

    random.shuffle(symbols)
    for symbol in symbols:
        print(f"training: {symbol}")
        # todo we're going to pass a single dataframe in here
        # consisting of multiple symbols but we have to make sure each

        train_tr_forecaster(model_path="./models",
                         model_prefix="tranformer_forecaster_",
                         training_data_path=f"./data/{symbol}_diffs.csv",
                         token=symbol)

if __name__ == "__main__":
    main()