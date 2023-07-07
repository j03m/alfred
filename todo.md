# General:
   
* Implement human advisor with: 
  * We won';t use: https://github.com/HumanCompatibleAI/imitation until gym vs gymnasium is ironed out
    * Instead, it looks like this is built into sb3: 
    * https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html
    * https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pretraining.ipynb#scrollTo=lIdT-zMV8aot
* How to predict the amount of time training is going to take in advance?
* Read: https://github.com/optuna/optuna
  * In the hyper parameter tuning tutorial they show how different net_arch for the SAC agent have different
  * results. Optuna is supposed to help

# Reading List

## Open AI Five

* https://openai.com/research/openai-five
* https://arxiv.org/pdf/1912.06680.pdf

## Bayes online change detection: 
* https://arxiv.org/abs/0710.3742

# An Option Strategy

Next steps - now that we'll have a buy/sell indicator and some dates with probability
Lets create a new env that overrides our position functions and instead:
* looks at contracts available 
* selects a contract based on strike and expiration where strike is profitable given a mean regression and expiration is after our estimated mean regression
* implement pricing based on whether or not we reach price before expiration
* verify price decay (write a unit test?)
* Finalize the option strategy. Make a call option vs short?

# A Full Market Strat -  ppo
  * Train the PPO to think in this manner across all timeseries?
  * Given what we know of prob + mean reversion can we make an LSTM more accurate than the base?
  * Given what we know of prob + mean reversion can we validate, check, cross reference the LSTMs

# Full Market Strat

  * Train LSTMs to do price prediction on all stocks but use monthly granularity
  * Review the entire market for predict increases in price and low MSE in the last N month (maybe weeks?)
  * Examine the move upward or downward
  * For downward moves scan the market for the strongest inverse?
  * Take positions

# Dividend Strategy

* Scan market for highest yield dividend portfolio/ETFS?
* Consider risk management
* Consider dividend payout periods vs interest

# Tracking

# Risk Management

