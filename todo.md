# General:
* Replace your current Save callback with EvalCallback (built in) and try out tensorboard
  * https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
  * https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#event-callback
    * One thing I didn't see above was how to supply a tensorboard path when loading a model
    * Check the docs quick
  * Once successful, migrate all of this learning into ppo_train.py
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

