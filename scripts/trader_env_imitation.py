#!/usr/bin/env python3
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data.types import TrajectoryWithRew

from machine_learning_finance import make_env_for, capture_exper_trajectories

rng = np.random.default_rng(0)

# capture our expert curriculum
env = make_env_for("SPY", 1)
obs, acts, terminal, infos, rews = capture_exper_trajectories(env)

# convert to the class imitation is expecting
demonstrations = []

trajectory_with_rew = TrajectoryWithRew(obs, acts, infos, terminal, rews)
demonstrations.append(trajectory_with_rew)

# train
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=demonstrations,
    rng=rng,
)

bc_trainer.train(n_epochs=1)

reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)
