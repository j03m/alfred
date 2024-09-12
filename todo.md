* ~~finalize linear series experiment~~
  * linear does okay, but it doesn't get close to lstm accuracy on something like, though it trains much faster
  * 2ndly, lstm does not generalize at all, receiving a series of prices and then trying to extrapolate what a 10 day (or even next day)
  change might be its pretty terrible. However, the linear network extrapolates pretty well.
  * How does it do on 5 day price vs lstm?
    * Way better than
* ~~Try projecting 2, 3, 4, 5 bars~~
* Try Conv1d projections
* series translation to trend
* more complicated models - they are still broken advanced lstm and transformer
* more datapoints
  * you can calculate expected earnings using alphavantage surprise - might be interesting to look at projections of price reflected in expected earnings
* How ususable are the tools in https://github.com/AI4Finance-Foundation/FinRL?