#!/usr/bin/env python3
import os.path
import argparse
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import yfinance as yf
from datetime import datetime, timedelta
from typing import Callable
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

from machine_learning_finance import TraderEnv, get_or_create_model, RangeTrainingWindowUtil, TailTrainingWindowUtil

# handles UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
from stable_baselines3.common.monitor import Monitor


def main():
    now = datetime.now()
    start_default = now - timedelta(days=365)
    start_default_str = start_default.strftime('%Y-%m-%d')
    end_default_str = now.strftime('%Y-%m-%d')

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--train-set', type=str, default=None)
    parser.add_argument('--eval-set', type=str, default=None)
    parser.add_argument('--data-path', type=str, default="./data/")
    parser.add_argument('--training-intervals', type=int, default=1000)
    parser.add_argument('--tensorboard-log-path', type=str, default="./tensorboard-logs")
    parser.add_argument('--eval-frequency', type=int, default=1000)
    parser.add_argument('--model-name', type=str, default="ppo_mlp_policy_simple_env")
    parser.add_argument('--learning-run-prefix', type=str, default="run_")
    parser.add_argument('--learning-runs', type=int, default=3)
    parser.add_argument('--start', type=str, default=start_default_str)
    parser.add_argument('--end', type=str, default=end_default_str)
    parser.add_argument('--tail', type=int, default=None)
    parser.add_argument('--file', action="store_true")

    args = parser.parse_args()

    if args.symbol is not None:
        if not args.file:
            df = download_symbol(args.symbol)
        else:
            df = read_symbol_file(args, args.symbol, fail_on_missing=True)
        train_model(args.symbol, df, args)
    elif args.train_set is not None:
        if os.path.isfile(args.train_set):
            # training vector env
            train_env = symbols_to_vec_env(args.train_set, args)
            if args.eval_set is not None:
                if os.path.isfile(args.train_set):
                    # eval vector env
                    eval_env = symbols_to_vec_env(args.eval_set, args)
                else:
                    print(f"Could not find file {args.train_set}.")
            else:
                eval_env = train_env
            train_model(train_env, eval_env, args)
        else:
            print(f"Could not find file {args.train_set}.")
    else:
        print("--symbol or --training-set must be supplied!")
        return -1


def symbols_to_vec_env(file: str, args):
    symbols_df = pd.read_csv(file)
    train_env = get_vector_env(symbols_df["Symbols"].values, args)
    return train_env


def read_symbol_file(args, symbol, fail_on_missing=False):
    symbol_file = os.path.join(args.data_path, f"{symbol}.csv")
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


def download_symbol(symbol):
    ticker_obj = yf.download(tickers=symbol, interval="1d")
    return pd.DataFrame(ticker_obj)


def get_environment_factory(symbol: str, df: pd.DataFrame, args: any) -> Callable[[], TraderEnv]:
    def generate_environment() -> TraderEnv:
        if args.tail is not None:
            training_window = TailTrainingWindowUtil(df, args.tail)
        else:
            training_window = RangeTrainingWindowUtil(df, args.start, args.end)

        return TraderEnv(symbol, training_window.test_df, training_window.full_hist_df)

    return generate_environment


def generate_evaluation_vec_env(symbols: [str], args):
    for symbol in symbols:
        data_df = read_symbol_file(args, symbol)
        get_environment_factory(symbol, data_df, args)


def get_vector_env(symbols: [str], args: any) -> DummyVecEnv:
    environments = []
    for symbol in symbols:
        data_df = read_symbol_file(args, symbol)
        environments.append(get_environment_factory(symbol, data_df, args))
    return DummyVecEnv(environments)


def train_model(train_env: DummyVecEnv, eval_env: DummyVecEnv, args: any):
    #train_env = Monitor(train_env)
    #eval_env = Monitor(eval_env)

    model, model_path, save_path = get_or_create_model(args.model_name, train_env, args.tensorboard_log_path)

    eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
                                 log_path=model_path, eval_freq=args.eval_frequency)

    print("Train the agent for N steps")
    model.learn(total_timesteps=args.training_intervals, tb_log_name=f"{args.learning_run_prefix}",
                callback=eval_callback, reset_num_timesteps=True)


if __name__ == "__main__":
    main()
