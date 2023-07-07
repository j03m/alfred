# Strategies for augmented agent data

* !!!! Consider simply marking the absolute optimum trade points in a timeseries, rather than finding addition points to trade on and give the agent guidance
* Probability tasks: 
  * TODO: Idea
    * Run bocd on the timeseries iteratively, capture change points in the training data
    * Run polynomial regression between change points, when a change point is detected
    * Run derivative in the center of the regression and detect trend via slope - provide this to the training model
    * GET BACK TO TRAINING AIs
  * Bayes online change detection: 
    * Readme: https://arxiv.org/abs/0710.3742
    * we used this as a look: https://github.com/y-bar/bocd
    * TODO: Pick up here, visualize this on a financial instrument
            
          

    * Ruptures papers 
      * https://centre-borelli.github.io/ruptures-docs/
      * http://www.laurentoudre.fr/publis/TOG-SP-19.pdf
      * https://charles.doffy.net/files/sp-review-2020.pdf
      
* Implement human advisor with: 
  * We won';t use: https://github.com/HumanCompatibleAI/imitation until gym vs gymnasium is ironed out
    * Instead it looks like this is built into sb3: 
    * https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html
    * https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pretraining.ipynb#scrollTo=lIdT-zMV8aot
* How to predict the amount of time training is going to take in advance?
* Read: https://github.com/optuna/optuna
  * In the hyper parameter tuning tutorial they show how different net_arch for the SAC agent have different
  * results. Optuna is supposed to help
* try to implement guided learning on a simpler problem (Reward shaping, Behavior Cloning)
  * What would guided learning on cartpole look like?

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