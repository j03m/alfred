# Now
  * Test model_training.py to see if the traderenv works on preprocessed data, then implement the random windows testing
  * Bug report and PR to SB3 for /Users/jmordetsky/machine_learning_finance/venv/lib/python3.11/site-packages/stable_baselines3/common/evaluation.py:67
    * If we supply a monitor, we wrap it in a vecenv and then we have a vecenv wrapped by a vecenv and we get reset signature issues
  * Add fundamental data, see if that correlates to trends and helps the model 
    * eps
    * analyst recommendations
  * Can we subclass EvalCallback to benchmark gainst profit vs score? Do we want to do that or would it better to tie profit TO score
    * Either way we might need to update our benchmark/eval script

# Next:
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

