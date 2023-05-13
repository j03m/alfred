
# Option Strategy

Next steps - now that we'll have a buy/sell indicator and some dates with probability
Lets create a new env that overrides our position functions and instead:
* looks at contracts available 
* selects a contract based on strike and expiration where strike is profitable given a mean regression and expiration is after our estimated mean regression
* implement pricing based on whether or not we reach price before expiration
* verify price decay (write a unit test?)
* Finalize the option strategy. Make a call option vs short?

# AI

* Todo: kick off guided training vs curriculum two, let it run all day.

# Base Strategy

* The "Let's study this data" plan:
  * Run probability on 365 days of data leading to the date in question during a back test. This is different backtest
    * You will have to change the backtester to have excess data and run probability at each step
  * Wire up a daemon that scans the pairs isolates opps and takes positions
    * SMS on entry/exit detected
      * current price, probability, image?
    * Track positions
    
    * Research: Should probability needs to be generated against MAX history?
      * It seems like this woulld be more confident, but it would trade less. 
      * Our year numbers are great, might just want to stick with what we have

* The "Maybe we don't need AI" plan:
    * scan all markets
    * filter by high probabilities
    
* create a process that can just run forever, detects if there is no internet and waits
* Another env - if we can't short or option due to restrictions is to find every possible 3x inverse etf and trade their relationships
  * Measure the inverse correlations

# Tracking

We need a notebook that is going to track/visualize the following off a strategy ledger:
(see if there is something off the shelf that will track/visualize this for you)
* mean/max/min/std duration of a hold
* mean/max/min/std duration between trades
* graph strategies against each other in terms of returns
  * graph strategies again base: buy/hold, momentum, simple mr
* Common benchmarks: max draw down etc
* 

# Risk Management




# Dividend Strategy

* Scan market for highest yield dividend portfolio/ETFS?
* Consider risk management
* Consider dividend payout periods vs interest