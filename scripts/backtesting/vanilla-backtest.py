import pandas as pd
import torch
import argparse

from datetime import datetime

from alfred.utils import set_deterministic, read_time_series_file, trim_timerange
from alfred.easy import prepare_data_and_model, ModelPrepConfig
from alfred.devices import set_device
from alfred.model_backtesting import SimpleBacktester, NuancedBacktester, MagnitudeChoiceBacktester
from alfred.metadata import TickerCategories

device = set_device()


def main(args):
    set_deterministic(args.seed)  # Set seed for reproducibility
    if args.type == "simple" or args.type == "nuanced":
        if args.test_ticker is None:
            raise Exception(f"Please specify a ticker for backtest type {args.type}")
        basic_back_tests(args)
    elif args.type == "portfolio":
        portfolio_back_test(args)


def portfolio_back_test(args):
    # Validate dates
    start_date, end_date = extract_dates(args)
    tc = TickerCategories(args.ticker_file)
    tickers = tc.get(["evaluation"])
    files = []

    dfs = []
    for ticker in tickers:
        file = f"./data/{ticker}_quarterly_magnitude.csv"
        files.append(file)
        df = trim_timerange(read_time_series_file(file), start_date, end_date)
        df["ticker"] = ticker
        df = df[["ticker", "Close"]]
        # if we're operating on a sequence, The model will make a prediction from seq_len on, so
        # for the back test data that has to be removed.
        if args.seq_len is not None:
            df = df.iloc[args.seq_len:]
        dfs.append(df)

    df_final = pd.concat(dfs)


    config = ModelPrepConfig(model_name=args.model_name,
                             model_size=args.model_size,
                             shuffle=False,
                             files=files,
                             labels=["PM"],
                             category=args.category,
                             batch_size=1,
                             seq_len=args.seq_len,
                             augment_func=lambda df: trim_timerange(df,
                                                                    min_date=start_date,
                                                                    max_date=end_date))

    result = prepare_data_and_model(config)

    model = result.model
    loader = result.loader
    outputs = []
    model.eval()
    for features, _ in loader:
        features = features.to(device)
        with torch.no_grad():
            output = model(features).squeeze().detach().item()
            outputs.append(output)

    df_final["predictions"] = outputs
    df_final.sort_index(inplace=True)

    pb = MagnitudeChoiceBacktester()
    ledger = pb.run_test(df_final)

    pb.print_ledger_metrics(df_final, ledger)


def basic_back_tests(args):
    # Validate dates
    start_date, end_date = extract_dates(args)

    file = f"data/{args.test_ticker}_quarterly_directional.csv"
    # back test wants a pandas dataframe. We need to give it one, but we also need to get our
    # loader to the model. We'll reread the file here and capture the loader as a global
    df = trim_timerange(read_time_series_file(file), args.min_date, args.max_date)
    df = df[["Close"]]


    config = ModelPrepConfig(model_name=args.model_name,
                             model_size=args.model_size,
                             shuffle=False,
                             files=[file],
                             batch_size=1,
                             seq_len=args.seq_len,
                             category=args.category,
                             augment_func=lambda df: trim_timerange(df,
                                                                    min_date=start_date,
                                                                    max_date=end_date))

    result = prepare_data_and_model(config)
    model = result.model
    loader = result.loader
    was_loaded = result.was_loaded
    if not was_loaded:
        print("WARNING: you're backtesting with a brand new model. ")
    operations = []
    readings = []
    model.eval()
    if args.type == "simple":
        for features, _ in loader:
            features = features.to(device)
            with torch.no_grad():
                output = model(features).squeeze()
            value = output.item()
            readings.append(value)
            if value >= args.buy_confidence:
                operations.append(1)
            else:
                operations.append(0)
        bt = SimpleBacktester()
        ledger = bt.run_test(df, operations, readings)
        bt.print_ledger_metrics(df, ledger)

    elif args.type == "nuanced":
        for features, _ in loader:
            features = features.to(device)
            with torch.no_grad():
                output = model(features).squeeze()
            value = output.item()
            readings.append(value)
            if value >= args.buy_confidence:
                operations.append(2)
            elif args.buy_confidence > value >= args.close_buy_confidence:
                operations.append(1)
            elif args.close_short_confidence < value >= args.short_confidence:
                operations.append(-1)
            elif value < args.short_confidence:
                operations.append(-2)
            else:
                operations.append(0)
        bt = NuancedBacktester()
        ledger = bt.run_test(df, operations, readings)
        bt.print_ledger_metrics(df, ledger)
    else:
        assert False, "I'm broken"


def extract_dates(args):
    try:
        start_date = datetime.strptime(args.min_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.max_date, '%Y-%m-%d')
        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")
    except ValueError as e:
        raise ValueError(f"Invalid date format or range: {e}")
    return start_date, end_date


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest with a model strategy, benchmark vs buy/hold.")
    parser.add_argument('--min_date', type=str, default="2004-03-31",
                        help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default="2024-12-31",
                        help='Maximum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument("--test_ticker", type=str, default="AAPL", help="Ticker symbol for the test asset")
    parser.add_argument("--ticker_file", type=str, default="metadata/basic-tickers.json",
                        help="Path to the tickers metadata file")
    parser.add_argument("--type", choices=["simple", "nuanced", "portfolio"], default="simple",
                        help="The type of backtest to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--model_name", type=str, default="vanilla.medium", help="model to back test")
    parser.add_argument("--model_size", type=int, default=1024, help="model size")
    parser.add_argument("--seq_len", type=int,  help="sequence length, optional supply for lstm/transformers")
    parser.add_argument("--buy_confidence", type=float, default=0.7, help="score >= needed to initiate a buy")
    parser.add_argument("--close_buy_confidence", type=float, default=0.5, help="score >= needed to initiate a buy")
    parser.add_argument("--close_short_confidence", type=float, default=0.5, help="score <= needed to initiate a buy")
    parser.add_argument("--short_confidence", type=float, default=0.3, help="score <= needed to initiate a buy")
    parser.add_argument("--category", type=str, help="Category label for the model")

    args = parser.parse_args()
    main(args)
