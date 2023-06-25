# Revisting basics

* ~~finish open ai tutorials~~
* ~~read cartpole v0 source~~
  * ~~Consider our action space and environment setup~~
  * ~~build a new env that doesn't use price or volume, just distance from mean and historical percent divergence~~
  * ~~Test simple env against various timeseries and tickers~~
    * ~~Figure out how to integrate it into all the other scripts~~
    * ~~Consider changing inverse_env to be more like simple_env rather than trader_env~~
* Probability tasks: 
  * ~~What do when the historical probabilities change and you discover you might not want to be in a position?~~
    * We had big bug around look ahead here, made significant changes, the strategy is now not nearly as good :(
  * todo: The test moving average needs to be calculated using window_size + test period or you end up with a huge flat period in the test
  * With our new found naivety, we need to consider:
    * "patience" - when reaching a probability, should we go in right away or wait to see if there more to come?
      * 
    * "trend" - On April 29th QQQ we get an oversold indicator but the overall trend is egregiously downward - we need to consider this wait for trends to flatten
      * Can we generate a linear regression through the 180 exponential moving average and generate buy/sell signal based on a positive/negative slope
    * "different entry/exit moving averages"
      * should we use two different moving average (time) to indicate long/short as shorts seem to be more risky (maybe, prove it)
      * Consider the above, rather than long/short consider enter/exit, for example entry for long or short can use the 90 day probabilities but exit
        * would be faster, using the 30 or 60 day probabilities
      * Explore the relationship between the sensitivity of the probabilities thresholds vs the window/speed of the moving average
* Implement human advisor with: 
  * https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html
  * https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pretraining.ipynb#scrollTo=lIdT-zMV8aot
* How to predict the amount of time training is going to take in advance?
* Read: https://github.com/optuna/optuna
  * In the hyper parameter tuning tutorial they show how different net_arch for the SAC agent have different
  * results. Optuna is supposed to help
* try to implement guided learning on a simpler problem (Reward shaping, Behavior Cloning)
  * What would guided learning on cartpole look like?
  * Read: https://github.com/HumanCompatibleAI/imitation
    * This is what stable baselines recommends 
* Read the open AI five paper

# Open AI Five

https://openai.com/research/openai-five
https://arxiv.org/pdf/1912.06680.pdf

# Option Strategy

Next steps - now that we'll have a buy/sell indicator and some dates with probability
Lets create a new env that overrides our position functions and instead:
* looks at contracts available 
* selects a contract based on strike and expiration where strike is profitable given a mean regression and expiration is after our estimated mean regression
* implement pricing based on whether or not we reach price before expiration
* verify price decay (write a unit test?)
* Finalize the option strategy. Make a call option vs short?

# AI

* Base Strat - PPO
  * I trained the PPO for 2 days using a guide and it is kinda terrible :/ it always selects action 2 and runs out of money after a single pass through the data
my thoughts are something is wrong.
  * Starting over, reading docs.
    * rl-tutorial-jnrr19/1_getting_started.ipynb <- going through these
    * How is the cartpole env implemented? What can it tell us about our envs?
  * May try to implement cart-pole. 
  * Todos:
    * change action space akin to guidance: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    * Train on more than 1 time series
    * Courses: 
      * https://stable-baselines3.readthedocs.io/en/master/guide/rl.html
        * https://sites.google.com/view/deep-rl-bootcamp/lectures
        * https://spinningup.openai.com/en/latest/

* Full Market Strat -  ppo
  * Train the PPO to think in this manner across all timeseries?
  * Given what we know of prob + mean reversion can we make an LSTM more accurate than the base?
  * Given what we know of prob + mean reversion can we validate, check, cross reference the LSTMs

# Base Strategy
*done*

# Full Market Strat

  * Train LSTMs to do price prediction on all stocks but use monthly granularity
  * Review the entire market for predict increases in price and low MSE in the last N month (maybe weeks?)
  * Examine the move upward or downward
  * For downward moves scan the market for the strongest inverse?
  * Take positions


# Tracking


# Risk Management



# Dividend Strategy

* Scan market for highest yield dividend portfolio/ETFS?
* Consider risk management
* Consider dividend payout periods vs interest