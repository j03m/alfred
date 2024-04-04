# Now
  * Smarter:
    * Give it a really long training time
    * Understand why we plateau - we seems to get to 2/3rds accuracy
    * Understand is 2/3 accuracy profitable?
    * Add fundamental data, see if that correlates to trends and helps the model 
    * eps
    * analyst recommendations
  * Training:
    * Can we make a YahooFinance pytorch datasource? Get a lot for free
    * What does multi-core CPU testing look like on a really beefy linux box?
    * What does BDSP give us?
    * We can give it more data using multiple random windows per time series :)
    * Lets drop stable baselines, so we're much closer to the metal and can understand the LSTM better
    * The reinforcement learning can be working off of lstm predictions?
    * What does granularity changes do to lstm predictions, like is it more or less accurate given daily vs weekly data?
    * Given what we know about lstm and gru memory now, is there a way to remember less granular data (monthly) while processing daily?
      * Maybe training against the same bar repeatedly where each day feeds into the current month bar
    * confidence?
    * Rendering?

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

