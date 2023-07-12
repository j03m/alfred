import os
import time
import argparse
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import yfinance as yf
import datetime


from machine_learning_finance import TraderEnv, create_train_test_windows, compute_derivatives_between_change_points

# handles UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
from stable_baselines3.common.monitor import Monitor


class SaveOnInterval(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveOnInterval, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.steps = 0

    def _on_step(self) -> bool:
        if self.steps % self.check_freq == 0:
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {self.save_path}")
        self.steps += 1
        return True


def estimate_time(num_episodes: int, model: PPO, env):
    start_time = time.time()
    model.learn(total_timesteps=num_episodes)
    end_time = time.time()
    time_per_episode = (end_time - start_time) / num_episodes
    return time_per_episode


def convert_seconds_to_time(estimated_time):
    # Get a timedelta object representing the duration
    timedelta = datetime.timedelta(seconds=estimated_time)
    return str(timedelta)


parser = argparse.ArgumentParser()
parser.add_argument('--benchmark-intervals', type=int, default=100)
parser.add_argument('--training-intervals', type=int, default=1000)
parser.add_argument('--saving-intervals', type=int, default=1000)
parser.add_argument('--model-name', type=str, default="models/ppo_mlp_policy_simple_env")
parser.add_argument('--estimate', type=int)
args = parser.parse_args()

ticker_obj = yf.download(tickers="SPY", interval="1d")
df = pd.DataFrame(ticker_obj)
hist_df, test_df = create_train_test_windows(df, None, 365 * 4, None, 365)

env = TraderEnv("SPY", test_df, hist_df)
env = Monitor(env)

save_path = args.model_name
model = PPO(MlpPolicy, env, verbose=1) if not os.path.isfile(save_path + ".zip") else PPO.load(save_path, env=env)

if args.estimate is not None:
    time_per_episode = estimate_time(100, model, env)  # Time for 100 episodes
    estimated_time = time_per_episode * args.estimate
    print(f"Estimated time for {args.estimate} episodes: {convert_seconds_to_time(estimated_time)}")
else:
    callback = SaveOnInterval(check_freq=args.saving_intervals, save_path=save_path)

    print("Agent status, before training")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.benchmark_intervals)
    print(f"(pre train) mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"(pre train profit) {env}")

    print("Train the agent for N steps")
    model.learn(total_timesteps=args.training_intervals, log_interval=100, callback=callback)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.benchmark_intervals)
    print(f"(post train) mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"(post train profit) {env}")

    model.save(save_path)  # final save
