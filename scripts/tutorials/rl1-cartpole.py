#!/usr/bin/env python3
import gymnasium as gym
from cartpole import CartPoleEnv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# handles UserWarning: Evaluation environment is not wrapped with a ``Monitor`` wrapper.
from stable_baselines3.common.monitor import Monitor



env = CartPoleEnv()
env = Monitor(env)

model = PPO(MlpPolicy, env, verbose=0)

#record_video("CartPole-v1", model, video_length=1000, prefix="ppo-cartpole-untrained")


print("Agent, before training")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

print("Train the agent for 10000 steps")
model.learn(total_timesteps=10_000)

print("Agent, after training")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward post train:{mean_reward:.2f} +/- {std_reward:.2f}")

print("Adjust gravity!")
high_grav_env = CartPoleEnv()
high_grav_env.x_threshold = 8
high_grav_env.gravity = 18
high_grav_env = Monitor(high_grav_env)
mean_reward, std_reward = evaluate_policy(model, high_grav_env, n_eval_episodes=100)
print(f"mean_reward post grav:{mean_reward:.2f} +/- {std_reward:.2f}")

print("Train again with new model!")
model.set_env(high_grav_env)
model.learn(total_timesteps=10_000)
mean_reward, std_reward = evaluate_policy(model, high_grav_env, n_eval_episodes=100)
print(f"mean_reward post train 2:{mean_reward:.2f} +/- {std_reward:.2f}")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward back to ogs:{mean_reward:.2f} +/- {std_reward:.2f}")