
# Option Strategy

Let's assume historically there are puts and calls that are 1-7 weeks away from the set fridays (expireys)
Let's assume the option contracts are set at prices corresponding to our model (even tho they really aren't, maybe they are for 1-7 weeks)
Let's also assume implied volatility is volatility for the 7 days leading to today
Let's assume then the price of the option can be calculated with black scholes (see if there is a more modern calculation in a python lib)

The environment needs to be able to look up contracts and select a contract based on the highest probably mean regression date
Profit/Loss is then calculated off of that option price, instead of the underlying asset.

We need to verify pricing. We can graph our theoretical price movement against the underlying historical movement.


* Todo: kick off guided training vs curriculum two, let it run all day.
* Finalize the option strategy. Make a call option vs short?
* The "Maybe we don't need AI" plan:
    * scan all markets
    * filter by high probabilities
    * apply LSTM prediction against high probabilities
    * take a position
    * track results
* create a process that can just run forever, detects if there is no internet and waits

# Dividend Strategy

Scan market for highest yield dividend portfolio/ETFS?

Consider risk management

Consider dividend payout periods vs interest