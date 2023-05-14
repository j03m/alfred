import yfinance as yf
import datetime
import pandas as pd
from .trader_env import TraderEnv
from .data_utils import model_path
from sb3_contrib import RecurrentPPO
from .defaults import DEFAULT_TEST_LENGTH, \
    DEFAULT_HISTORICAL_MULT, \
    DEFAULT_BOTTOM_PERCENT, \
    DEFAULT_TOP_PERCENT, \
    DEFAULT_CASH
import os
from .data_utils import get_coin_data_frames


def make_env_for(symbol,
                 code,
                 tail=DEFAULT_TEST_LENGTH,  # test timeframe is 1 years default
                 data_source="yahoo",
                 path=None,
                 cash=DEFAULT_CASH,
                 prob_high=DEFAULT_TOP_PERCENT,
                 prob_low=DEFAULT_BOTTOM_PERCENT,
                 env_class=TraderEnv,
                 hist_tail=None):  # historical timeframe is 4 years default
    if hist_tail is None:
        hist_tail = tail * DEFAULT_HISTORICAL_MULT
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

    # history dataframe has to be data that isn't in the test, we trim it to all the data but the
    # data under test and then apply the desired timeframe
    hist_df = df.head(len(df)-tail)
    hist_df = hist_df.tail(hist_tail)

    # The test df is applied against the lastest data.
    # todo: I guess is to provide some way to window both of these
    test_df = df.tail(tail)
    env = env_class(symbol, test_df, hist_df, code, cash=cash, prob_high=prob_high, prob_low=prob_low)
    return env


def timestr():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def partial_test(env):
    ppo_agent = get_or_create_recurrent_ppo(False, env)
    ppo_agent.policy.custom_actions = None
    obs = env.reset_test()
    done = False
    state = None
    while not done:
        action, state = ppo_agent.predict(obs, state, episode_start=False, deterministic=False)
        # take the action and observe the next state and reward
        obs, reward, _, done, info_ = env.step(action)


def partial_train(env, steps=500, create=False):
    ppo_agent = get_or_create_recurrent_ppo(False, env)
    ppo_agent.policy.custom_actions = None
    ppo_agent.learn(total_timesteps=steps)
    ppo_agent.save(os.path.join(model_path, "baseline-recurrent-ppo"))


def back_test_expert(env):
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

    ppo_agent = get_or_create_recurrent_ppo(create, env)

    state_action_data = env.expert_opinion()
    custom_actions = [action for _, action in state_action_data]
    ppo_agent.policy.custom_actions = custom_actions
    ppo_agent.learn(total_timesteps=steps)
    ppo_agent.save(os.path.join(model_path, "baseline-recurrent-ppo"))
    return env


def get_or_create_recurrent_ppo(create, env):
    if create:
        ppo_agent = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            ent_coef=0.1,
            clip_range=0.3,
            verbose=1,
        )
    else:
        ppo_agent = RecurrentPPO.load(os.path.join(model_path, "baseline-recurrent-ppo"))
        ppo_agent.set_env(env)
    return ppo_agent
