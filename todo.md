
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
  * Still doesn't run behind proxy - yahoo fails
  * Is there a way to measure the relationship between the amount of divergence in either direction and the profitability of the trade
    * As such we could look at the divergence and get a sense for which ETF to take a position in
    * In addition, are any additional signals we could take into account we which had a relationship to profitibility
  * Build the ability to backtest different year periods
    * Run a few scenarios
      * covid
      * 2008
      * run up to and after trump
      * 2008 recovery etc
      * Running behind proxy?
  

    

# Tracking


# Risk Management



# Dividend Strategy

* Scan market for highest yield dividend portfolio/ETFS?
* Consider risk management
* Consider dividend payout periods vs interest