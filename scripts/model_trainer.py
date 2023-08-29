#!/usr/bin/env python3
import os.path
import argparse
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import yfinance as yf
from datetime import datetime, timedelta
from typing import Callable
import warnings
import time

warnings.filterwarnings('ignore', category=RuntimeWarning)

from machine_learning_finance import (TraderEnv, get_or_create_model, RandomTrainingWindowUtil, RangeTrainingWindowUtil,
                                      TailTrainingWindowUtil, mylogger, read_symbol_file, read_processed_file)


# handles UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
# from stable_baselines3.common.monitor import Monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--train-set', type=str, default=None)
    parser.add_argument('--eval-set', type=str, default=None)
    parser.add_argument('--data-path', type=str, default="./data/")
    parser.add_argument('--training-intervals', type=int, default=1000)
    parser.add_argument('--tensorboard-log-path', type=str, default="./tensorboard-logs")
    parser.add_argument('--eval-frequency', type=int, default=500)
    parser.add_argument('--model-name', type=str, default="ppo_mlp_policy_simple_env")
    parser.add_argument('--learning-run-prefix', type=str, default="run_")
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--tail', type=int, default=730)
    parser.add_argument('--random', type=int, default=None)
    parser.add_argument('--windows', type=int, default=1)
    parser.add_argument('--file', action="store_true")
    parser.add_argument('--time', action="store_true", default=False)
    parser.add_argument('--multi-proc', action="store_true", default=False)

    args = parser.parse_args()

    if args.symbol is not None:
        env = make_env(args.symbol, args)
        train_model(env, env, args, 0)

    elif args.train_set is not None and args.eval_set is not None:
        if not os.path.isfile(args.train_set):
            print(f"Could not find file {args.train_set}.")
            return -1
        if not os.path.isfile(args.eval_set):
            print(f"Could not find file {args.eval_set}.")
            return -2

        for window in range(0, args.windows):
            train_env = symbols_to_vec_env(args.train_set, args)
            eval_env = symbols_to_vec_env(args.eval_set, args)
            train_model(train_env, eval_env, args, window)


    else:
        print("--symbol or --training-set and --eval-set must be supplied!")
        return -1


def symbols_to_vec_env(file: str, args):
    symbols_df = pd.read_csv(file)
    train_env = get_vector_env(symbols_df["Symbols"].values, args)
    return train_env





def download_symbol(symbol):
    ticker_obj = yf.download(tickers=symbol, interval="1d")
    return pd.DataFrame(ticker_obj)


def make_env(symbol, args, throw_on_process=False):

    process_data = False
    if args.file and not throw_on_process:
        df = read_symbol_file(args.data_path, args.symbol, fail_on_missing=True)
        process_data = True
    elif os.path.isfile(os.path.join(args.data_path, f"{symbol}_processed.csv")):
        df = read_processed_file(args.data_path, symbol)
        process_data = False
    elif os.path.isfile(os.path.join(args.data_path, f"{symbol}.csv")) and not throw_on_process:
        df = read_symbol_file(args.data_path, symbol)
        process_data = True
    elif not throw_on_process:
        df = download_symbol(symbol)
    else:
        raise Exception(f"{symbol} has not be preprocessed. ")

    if args.random is not None:
        training_window = RandomTrainingWindowUtil(df, args.random)
    elif args.tail is not None:
        training_window = TailTrainingWindowUtil(df, args.tail)
    else:
        training_window = RangeTrainingWindowUtil(df, args.start, args.end)

    if process_data:
        return TraderEnv(symbol, training_window.test_df, training_window.full_hist_df, process_data=process_data)
    else:
        return TraderEnv(symbol, training_window.test_df, training_window.test_df, process_data=process_data)


def get_environment_factory(symbol: str, args: any) -> Callable[[], TraderEnv]:
    def generate_environment() -> TraderEnv:
        return make_env(symbol, args, throw_on_process=True)

    return generate_environment

def get_vector_env(symbols: [str], args: any) -> DummyVecEnv:
    environments = []
    for symbol in symbols:
        mylogger.info("Generating factory for symbol: ", symbol)
        environments.append(get_environment_factory(symbol, args))
    if args.multi_proc:
        return SubprocVecEnv(environments)
    else:
        return DummyVecEnv(environments)


def convert_seconds_to_time(estimated_time: int):
    # Get a timedelta object representing the duration
    delta = timedelta(seconds=estimated_time)
    return str(delta)


def train_model(train_env: DummyVecEnv, eval_env: DummyVecEnv, args: any, window: int):
    # train_env = Monitor(train_env)
    # eval_env = Monitor(eval_env)

    model, model_path, save_path = get_or_create_model(args.model_name, train_env,
                                                       f"{args.tensorboard_log_path}_{window}")

    eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
                                 log_path=model_path, eval_freq=args.eval_frequency)

    print("Train the agent for N steps")

    start_time = time.time()
    model.learn(total_timesteps=args.training_intervals, tb_log_name=f"{args.learning_run_prefix}",
                callback=eval_callback, reset_num_timesteps=True)
    end_time = time.time()
    if args.time:
        final_time = end_time - start_time
        time_per_episode = final_time / args.training_intervals
        print(f"Estimated time for {args.training_intervals} episodes: {convert_seconds_to_time(final_time)}")
        print(f"Time per episode: {convert_seconds_to_time(time_per_episode)}")


if __name__ == "__main__":
    main()
