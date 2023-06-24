import numpy as np
from imitation.data.types import TrajectoryWithRew

# Create a random observation. In the CartPole environment, an observation is a vector of 4 floats.
obs = np.random.rand(4)

# Create a random action. In the CartPole environment, an action is an integer (either 0 or 1).
act = np.random.choice([0, 1])

# Create a random reward. In most environments, a reward is a single float.
rew = np.random.rand()

# Create a random done flag. This is a boolean that indicates whether the episode has ended.
done = np.random.choice([True, False])

# Create a random next observation.
next_obs = np.random.rand(4)

# Create a transition with the random data
transition = TrajectoryWithRew(obs=[obs, obs], acts=[act], rews=np.array([rew]), terminal=[done], infos=[{}])

# Print the transition
print(transition)