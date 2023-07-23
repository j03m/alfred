#!/usr/bin/env python3
import os.path
import argparse
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
import yfinance as yf
from datetime import datetime, timedelta

from machine_learning_finance import TraderEnv, get_or_create_model, RangeTrainingWindowUtil

# handles UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
from stable_baselines3.common.monitor import Monitor


def download_symbol(symbol):
    ticker_obj = yf.download(tickers=symbol, interval="1d")
    return pd.DataFrame(ticker_obj)


def train_model(symbol, df, args):

    training_window = RangeTrainingWindowUtil(df, args.start, args.end)

    env = TraderEnv(symbol, training_window.test_df, training_window.hist_df)

    env = Monitor(env)

    model, model_path, save_path = get_or_create_model(args.model_name, env, args.tensorboard_log_path)

    eval_callback = EvalCallback(env, best_model_save_path=model_path,
                                 log_path=model_path, eval_freq=500)

    print("Train the agent for N steps")
    for i in range(0, args.learning_runs):
        model.learn(total_timesteps=args.training_intervals, tb_log_name=f"{args.learning_run_prefix}_{i}",
                    callback=eval_callback, reset_num_timesteps=True if i == 0 else False)


def main():
    now = datetime.now()
    start_default = now - timedelta(days=365)
    start_default_str = start_default.strftime('%Y-%m-%d')
    end_default_str = now.strftime('%Y-%m-%d')

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--bulk', type=str, default=None)
    parser.add_argument('--data-path', type=str, default="./data/")
    parser.add_argument('--benchmark-intervals', type=int, default=100)
    parser.add_argument('--training-intervals', type=int, default=1000)
    parser.add_argument('--tensorboard-log-path', type=str, default="./tensorboard-logs")
    parser.add_argument('--saving-intervals', type=int, default=1000)
    parser.add_argument('--model-name', type=str, default="ppo_mlp_policy_simple_env")
    parser.add_argument('--learning-run-prefix', type=str, default="run_number")
    parser.add_argument('--learning-runs', type=int, default=3)
    parser.add_argument('--start', type=str, default=start_default_str)
    parser.add_argument('--end', type=str, default=end_default_str)

    args = parser.parse_args()

    if args.symbol is not None:
        df = download_symbol(args.symbol)
        train_model(args.symbol, df, args)
    elif args.bulk is not None:
        if os.path.isfile(args.bulk):
            symbols_df = pd.read_csv(args.bulk)
            for symbol in symbols_df["Symbols"].values:
                symbol_file = os.path.join(args.data_path, f"{symbol}.csv")
                try:
                    data_df = pd.read_csv(symbol_file)
                except FileNotFoundError:
                    print(f"The file {symbol_file} was not found. Continuing")
                except pd.errors.ParserError:
                    print(f"The file {symbol_file} could not be parsed as a CSV. Continuing")

                train_model(symbol, data_df, args)
        else:
            print(f"Could not find file {args.bulk}.")
    else:
        print("--symbol or --bulk must be supplied!")
        return -1


if __name__ == "__main__":
    main()
