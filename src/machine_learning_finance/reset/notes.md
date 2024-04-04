# agent

* Agent picks from two actions, buy or sell (not unlike left right for cartpole) but the agent scores points based on profitability of the system

* Need to decide its model (probably lstm)
* Maybe leverages other lstm price predictions? (1 yr, 1 mnt, 1 dy) and directional indicators
  * maybe price is too unstable, and there is more accuracy on up down
  * could it derive a plan on duration based on what those duration indicators said?
* DQN vs PPO understand differences?

Note on re-inforce reverse iteration of rewards:

>So, when we say the 0th reward (the first in the episode) is given the most weight, we mean it in the sense that it has the most comprehensive view of the future, incorporating all subsequent rewards. However, each of those future rewards is indeed discounted to reflect their reduced value from the perspective of the starting point. This method ensures that the agent values immediate rewards more than distant ones but still considers the long-term consequences of its actions.