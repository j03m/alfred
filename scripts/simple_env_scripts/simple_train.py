#!/usr/bin/env python3
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import yfinance as yf

from machine_learning_finance import SimpleEnv, create_train_test_windows

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


ticker_obj = yf.download(tickers="SPY", interval="1d")
df = pd.DataFrame(ticker_obj)
hist_df, test_df = create_train_test_windows(df, None, 365 * 4, None, 365)
env = SimpleEnv(test_df, hist_df)
env = Monitor(env)

model = PPO(MlpPolicy, env, verbose=1)

save_path = "models/ppo_mlp_policy_simple_env"
callback = SaveOnInterval(check_freq=1000, save_path=save_path)

print("Random Agent, before training")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"(pre train) mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print(f"(pre train profit) {env.profit}")

print("Train the agent for N steps")
model.learn(total_timesteps=10_000_000, log_interval=100, callback=callback)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"(post train) mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print(f"(post train profit) {env.profit}")

model.save(save_path)  # final save
