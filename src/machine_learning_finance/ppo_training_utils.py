import yfinance as yf
import datetime
import pandas as pd
from .trader_env import TraderEnv
from .curriculum_policy_support import CustomActorCriticPolicy
from .data_utils import get_data_for_training, model_path, data_path
from sb3_contrib import RecurrentPPO
import numpy as np
import os

env_count = 0

def make_env_for(symbol, code, tail=-1, head=-1, data_source="yahoo"):
    if data_source == "yahoo":
        tickerObj = yf.download(tickers=symbol, interval="1d")
        df = pd.DataFrame(tickerObj)
    elif data_source == "file":
        df = pd.read_csv(f"./data/{symbol}.csv")
    else:
        raise Exception("Implement me")
    if tail != -1:
        df = df.tail(tail)
    if head != -1:
        df = df.head(head)
    df = df.reset_index()
    env = TraderEnv(symbol, df, code)
    return env

def create_env(product_data, code=1):
    global env_count
    products = list(product_data.keys())
    product = products[env_count]
    env = TraderEnv(product, product_data[product], code)
    env_count += 1
    return env


def timestr():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def full_train(curriculum_code, num_stocks, create=True):
    ppo_agent, sorted_envs = get_or_create_recurrent_ppo(create, num_stocks)
    for env in sorted_envs:
        ppo_agent.set_env(env)
        ppo_agent.learn(total_timesteps=35000)
        ppo_agent.save(os.path.join(model_path, "baseline-recurrent-ppo"))


def full_test(symbol, tail=1, head=-1):
    tickerObj = yf.download(tickers=symbol, interval="1d")
    df = pd.DataFrame(tickerObj)
    if tail != -1:
        df = df.tail(tail)
    if head != -1:
        df = df.head(head)

    df = df.reset_index()
    env = TraderEnv(symbol, df)
    model = RecurrentPPO.load(os.path.join(model_path, "baseline-recurrent-ppo"))
    obs, _ = env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    done = False
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, _, done, extra = env.step(action)
        episode_starts = done


def partial_test(env):
    model = RecurrentPPO.load(os.path.join(model_path, "baseline-recurrent-ppo"))
    model.set_env(env)
    model.policy.custom_actions = None
    obs = env.reset_test()
    done = False
    state = None
    while not done:
        action, state = model.predict(obs, state, episode_start=False, deterministic=False)
        # take the action and observe the next state and reward
        obs, reward, _, done, info_ = env.step(action)


def partial_train(env, steps=500, create=False):
    if create:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            ent_coef=0.1,
            clip_range=0.3,
            verbose=1,
        )
    else:
        model = RecurrentPPO.load(os.path.join(model_path, "baseline-recurrent-ppo"))
        model.set_env(env)
        model.policy.custom_actions = None

    model.learn(total_timesteps=steps)
    model.save(os.path.join(model_path, "baseline-recurrent-ppo"))


def back_test_expert(symbol, curriculum, days):
    env = make_env_for(symbol, curriculum, days)
    env.expert_opinion_df()
    obs, _ = env.reset()

    action = obs[-1]
    done = False
    while not done:
        print("Action:", action)
        state, reward, _, done, _ = env.step(int(action))
        action = state[-1]
        print("Reward:", reward, " for action: ", action, "on probability: ", state[3])
    return env


def guided_training(env, create, steps=250000):

    if create:
        ppo_agent = RecurrentPPO(
            CustomActorCriticPolicy,
            env,
            ent_coef=0.1,
            clip_range=0.3,
            verbose=1,
        )
    else:
        ppo_agent = RecurrentPPO.load(os.path.join(model_path, "baseline-recurrent-ppo"))
        ppo_agent.set_env(env)

    state_action_data = env.expert_opinion()
    custom_actions = [action for _, action in state_action_data]
    ppo_agent.policy.custom_actions = custom_actions
    ppo_agent.learn(total_timesteps=steps)
    ppo_agent.save(os.path.join(model_path, "baseline-recurrent-ppo"))
    return env


def full_guided_train(num_stocks, create=True):
    ppo_agent, sorted_envs = get_or_create_recurrent_ppo(create, num_stocks)
    for env in sorted_envs:
        ppo_agent.set_env(env)
        state_action_data = env.expert_opinion()
        custom_actions = [action for _, action in state_action_data]
        ppo_agent.policy.custom_actions = custom_actions
        ppo_agent.learn(total_timesteps=35000)
        ppo_agent.save(os.path.join(model_path, "baseline-recurrent-ppo"))


def get_or_create_recurrent_ppo(create, num_stocks):
    product_data = get_data_for_training(num_stocks)
    all_envs = [create_env(product_data) for _ in range(len(product_data))]
    sorted_envs = sorted(all_envs, key=lambda env: env.product)
    if create:
        ppo_agent = RecurrentPPO(
            "MlpLstmPolicy",
            sorted_envs[0],
            ent_coef=0.1,
            clip_range=0.3,
            verbose=1,
        )
    else:
        ppo_agent = RecurrentPPO.load(os.path.join(model_path, "baseline-recurrent-ppo"))
    return ppo_agent, sorted_envs
