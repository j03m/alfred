
# Option Strategy

Next steps - now that we'll have a buy/sell indicator and some dates with probability
Lets create a new env that overrides our position functions and instead:
* looks at contracts available 
* selects a contract based on strike and expiration where strike is profitable given a mean regression and expiration is after our estimated mean regression
* implement pricing based on whether or not we reach price before expiration
* verify price decay (write a unit test?)

# Todo

* Todo: kick off guided training vs curriculum two, let it run all day.
* Finalize the option strategy. Make a call option vs short?
* The "Maybe we don't need AI" plan:
    * scan all markets
    * filter by high probabilities
    * apply LSTM prediction against high probabilities
    * take a position
    * track results
* create a process that can just run forever, detects if there is no internet and waits
* Another env - if we can't short or option due to restrictions is to find every possible 3x inverse etf and trade their relationships
  * Measure the inverse correlations

# Dividend Strategy

Scan market for highest yield dividend portfolio/ETFS?

Consider risk management

Consider dividend payout periods vs interest