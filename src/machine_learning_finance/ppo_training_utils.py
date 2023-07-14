import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from .trader_env import TraderEnv
from .data_utils import model_path
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from .defaults import DEFAULT_TEST_LENGTH, \
    DEFAULT_BOTTOM_PERCENT, \
    DEFAULT_TOP_PERCENT, \
    DEFAULT_CASH
import os
from .data_utils import get_coin_data_frames, create_train_test_windows

ppo_model_name = "models/ppo_mlp_policy_simple_env"

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


def make_env_for(symbol,
                 code,
                 tail=DEFAULT_TEST_LENGTH,  # test timeframe is 1 years default
                 data_source="yahoo",
                 path=None,
                 cash=DEFAULT_CASH,
                 prob_high=DEFAULT_TOP_PERCENT,
                 prob_low=DEFAULT_BOTTOM_PERCENT,
                 env_class=TraderEnv,
                 start=None,
                 end=None,
                 hist_tail=None):  # historical timeframe is 4 years default
    if data_source == "yahoo":
        ticker_obj = yf.download(tickers=symbol, interval="1d")
        df = pd.DataFrame(ticker_obj)
    elif data_source == "file":
        df = pd.read_csv(f"./data/{symbol}.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif data_source == "direct":
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif data_source == "ku_coin":
        df = get_coin_data_frames(hist_tail, symbol)
    else:
        raise Exception("Implement me")

    hist_df, test_df = create_train_test_windows(df, end, hist_tail, start, tail)
    env = env_class(symbol, test_df, hist_df, code, cash=cash, prob_high=prob_high, prob_low=prob_low)
    return env


def timestr():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def partial_test(env):
    ppo_agent = get_or_create_ppo(env)
    ppo_agent.policy.custom_actions = None
    obs = env.reset_test()
    done = False
    state = None
    while not done:
        action, state = ppo_agent.predict(obs, state, episode_start=False, deterministic=False)
        # take the action and observe the next state and reward
        obs, reward, _, done, info_ = env.step(action)


def partial_train(env, steps=500, create=False):
    ppo_agent = get_or_create_ppo(env)
    ppo_agent.learn(total_timesteps=steps)
    save_path = os.path.join(model_path, ppo_model_name)
    callback = SaveOnInterval(check_freq=1000, save_path=save_path)
    ppo_agent.learn(total_timesteps=steps, log_interval=100, callback=callback)
    ppo_agent.save(save_path)



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


def capture_exper_trajectories(env):
    def wrap(input):
        return np.array(input, dtype=np.float32)

    env.expert_opinion_df()
    first_obs, _ = env.reset()
    action = first_obs[-1]
    done = False
    obs = [first_obs]
    rews = []
    terminal = []
    infos = []
    acts = []
    while not done:
        print("Action:", action)
        state, reward, _, done, _ = env.step(int(action))
        action = state[-1]
        print("Reward:", reward, " for action: ", action, "on probability: ", state[3])
        obs.append(state)
        acts.append(action)
        infos.append({})
        rews.append(reward)
        terminal.append(done)

    return wrap(obs), wrap(acts), terminal, infos, wrap(rews)


def get_or_create_ppo(env):
    return PPO(MlpPolicy, env, verbose=1) if not os.path.isfile(ppo_model_name + ".zip") else PPO.load(ppo_model_name, env=env)
