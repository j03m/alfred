#!/usr/bin/env python3
import os.path
import time
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import yfinance as yf
import datetime

from machine_learning_finance import TraderEnv, create_train_test_windows, get_or_create_model

# handles UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
from stable_baselines3.common.monitor import Monitor


def estimate_time(_model: PPO):
    start_time = time.time()
    _model.learn(total_timesteps=100)
    end_time = time.time()
    _time_per_episode = (end_time - start_time) * 100
    return _time_per_episode


def convert_seconds_to_time(estimated_time: int):
    # Get a timedelta object representing the duration
    timedelta = datetime.timedelta(seconds=estimated_time)
    return str(timedelta)


def download_symbol(symbol):
    ticker_obj = yf.download(tickers=symbol, interval="1d")
    return pd.DataFrame(ticker_obj)


def train_model(symbol, df, args):
    # todo replace with TrainingWindowUtil
    hist_df, test_df = create_train_test_windows(df, None, 365 * 4, None, 365)
    env = TraderEnv(symbol, test_df, hist_df)

    env = Monitor(env)

    model, model_path, save_path = get_or_create_model(args.model_name, env, args.tensorboard_log_path)

    eval_callback = EvalCallback(env, best_model_save_path=model_path,
                                 log_path=model_path, eval_freq=500)

    print("Train the agent for N steps")
    for i in range(0, args.learning_runs):
        model.learn(total_timesteps=args.training_intervals, tb_log_name=f"{args.learning_run_prefix}_{i}",
                    callback=eval_callback, reset_num_timesteps=True if i == 0 else False)


def main():
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
