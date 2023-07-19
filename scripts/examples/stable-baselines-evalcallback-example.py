import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Separate evaluation env
eval_env = gym.make("Pendulum-v1")

eval_env = Monitor(eval_env)

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./example-eval-callback/",
                             log_path="./example-eval-callback/", eval_freq=500,
                             deterministic=True, render=False)

model = SAC("MlpPolicy", "Pendulum-v1")
model.learn(5000, callback=eval_callback)