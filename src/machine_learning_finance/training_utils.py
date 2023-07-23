from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import os


def back_test_expert(env):
    obs, _ = env.reset()
    count = 0
    action = env._expert_actions[count]
    done = False
    while not done:
        print("Action:", action)
        state, reward, _, done, _ = env.step(int(action))
        print("Reward:", reward, " for action: ", action)
        if not done:
            count += 1
            action = env._expert_actions[count]
    return env


def get_or_create_model(model_name, env, tensorboard_log_path):
    model_path = f"./models/{model_name}"
    save_path = f"./{model_path}/best_model.zip"
    if not os.path.isfile(save_path):
        print("Creating a new model")
        model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log_path)
    else:
        print(f"Loading the model from {save_path}")
        model = PPO.load(save_path, env=env)
    return model, model_path, save_path
