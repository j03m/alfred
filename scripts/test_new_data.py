#!/usr/bin/env python3
import os
import time
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import yfinance as yf
import datetime

from machine_learning_finance import TraderEnv, create_train_test_windows, compute_derivatives_between_change_points

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


parser = argparse.ArgumentParser()
parser.add_argument('--symbol', type=str, default="SPY")
parser.add_argument('--eval-train', action="store_true", default=False)
parser.add_argument('--eval-only', action="store_true", default=False)
parser.add_argument('--test', action="store_true", default=False)
parser.add_argument('--benchmark-intervals', type=int, default=100)
parser.add_argument('--training-intervals', type=int, default=1000)
parser.add_argument('--tensorboard-log-path', type=str, default="./tensorboard-logs")
parser.add_argument('--saving-intervals', type=int, default=1000)
parser.add_argument('--model-name', type=str, default="ppo_mlp_policy_simple_env")
parser.add_argument('--estimate', action="store_true")
parser.add_argument('--learning-run-prefix', type=str, default="run_number")
parser.add_argument('--learning-runs', type=int, default=3)

args = parser.parse_args()

model_path = f"./models/{args.model_name}"
ticker_obj = yf.download(tickers=args.symbol, interval="1d")
df = pd.DataFrame(ticker_obj)
hist_df, test_df = create_train_test_windows(df, None, 365 * 4, None, 365)

env = TraderEnv(args.symbol, test_df, hist_df)
env = Monitor(env)
save_path = f"./models/{args.model_name}/best_model.zip"
if not os.path.isfile(save_path):
    print("Creating a new model")
    model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=args.tensorboard_log_path)
else:
    print(f"Loading the model from {save_path}")
    model = PPO.load(save_path, env=env)

if args.test:
    obs, data = env.reset()
    done = False
    state = None
    while not done:
        action, state = model.predict(obs, state, episode_start=False)
        # take the action and observe the next state and reward
        obs, reward, _, done, info_ = env.step(action)
    env.ledger.to_csv(f"./backtests/{args.symbol}-model-back-test.csv")
    print(f"(post test profit) {env}")

elif args.eval_train:
    print("Agent status, before training")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.benchmark_intervals)
    print(f"(pre train) mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"(pre train profit) {env}")

    eval_callback = EvalCallback(env, best_model_save_path=model_path,
                                 log_path=model_path, eval_freq=500)

    print("Train the agent for N steps")
    for i in range(0, args.learning_runs):
        model.learn(total_timesteps=args.training_intervals, tb_log_name=f"{args.learning_run_prefix}_{i}",
                    callback=eval_callback, reset_num_timesteps=True if i == 0 else False)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.benchmark_intervals)
    print(f"(post train) mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"(post train profit) {env}")

    model.save(save_path)  # final save
elif args.eval_only:
    print("Agent status, no training")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.benchmark_intervals)
    print(f"(pre train) mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"(pre train profit) {env}")
elif args.estimate is not None:
    time_per_episode = estimate_time(model)  # Time for 100 episodes
    final_time = time_per_episode * args.training_intervals
    print(f"Estimated time for {args.training_intervals} episodes: {convert_seconds_to_time(final_time)}")
else:
    raise Exception("Supply --test or --eval-train")
