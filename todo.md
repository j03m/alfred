# Now:
  * Make separate scripts:
    * Bulk trainer - trains multiple symbols based on list uses offline cached data
    * Simple training - trains agains a single symbol online
    * test/evaluator - Uses stablebaselines reward evaluation and also prints backtest results
    * cacher - caches timeseries data


# Next:
  * Add to the model the value per share determined by a DCF model - the current price - might be interesting?
  * Learn how to train the agent on multiple environments at the sample time
  * check out obtuna
  * How do we not only compare to bench but compare to the absolute trading policy?
      * We should do a study of the optimal vs bench
  * We need to retest this against inverse?
* Read: https://github.com/optuna/optuna
  * In the hyper parameter tuning tutorial they show how different net_arch for the SAC agent have different
  * results. Optuna is supposed to help

# Icebox

* Kill all old notebooks - archive the tensorflow stuff

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

