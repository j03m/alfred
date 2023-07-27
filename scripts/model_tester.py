#!/usr/bin/env python3
import os.path
import argparse
import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy
import yfinance as yf
from datetime import datetime, timedelta
from machine_learning_finance import TraderEnv, get_or_create_model, RangeTrainingWindowUtil, TailTrainingWindowUtil
# handles UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
from stable_baselines3.common.monitor import Monitor


def download_symbol(symbol):
    ticker_obj = yf.download(tickers=symbol, interval="1d")
    return pd.DataFrame(ticker_obj)


def main():
    now = datetime.now()
    start_default = now - timedelta(days=365)
    start_default_str = start_default.strftime('%Y-%m-%d')
    end_default_str = now.strftime('%Y-%m-%d')

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default="SPY")
    parser.add_argument('--model-name', type=str, default="ppo_mlp_policy_simple_env")
    parser.add_argument('--tensorboard-log-path', type=str, default="./tensorboard-logs")
    parser.add_argument('--benchmark-intervals', type=int, default=100)
    parser.add_argument('--start', type=str, default=start_default_str)
    parser.add_argument('--end', type=str, default=end_default_str)
    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--tail', type=int, default=None)

    args = parser.parse_args()

    if args.tail is not None:
        training_window = TailTrainingWindowUtil(download_symbol(args.symbol), args.tail)
    else:
        training_window = RangeTrainingWindowUtil(download_symbol(args.symbol), args.start, args.end)

    env = TraderEnv(args.symbol, training_window.test_df, training_window.full_hist_df)
    env = Monitor(env)
    model, model_path, save_path = get_or_create_model(args.model_name, env, args.tensorboard_log_path)

    if args.eval:
        print("Agent status:")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.benchmark_intervals)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    elif args.test:
        obs, data = env.reset()
        done = False
        state = None
        while not done:
            action, state = model.predict(obs, state, episode_start=False)
            # take the action and observe the next state and reward
            obs, reward, _, done, info_ = env.step(action)
        env.ledger.to_csv(f"./backtests/{args.symbol}-model-back-test.csv")
        print(f"(post test profit) {env}")
    else:
        print("please supply --eval or --test")


if __name__ == "__main__":
    main()
