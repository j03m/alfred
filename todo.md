Next:
* add gold and futures to fundamental data
* add insider events to data as (has_disposal: [score: 0 -> some max], has_acquisition: etc)
  * https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol=IBM&apikey=demo
* you can calculate expected earnings using alphavantage surprise - might be interesting to look at projections of price reflected in expected earnings


The manager:

The manager's environment will be 2years of the snp all trickers indexed:

TICKER_INDEX, [price], [price prediction columns], [fundemental columns], [aggregate news sentiment], [last q earnings], [expected q earnings]

There will be 500+ TICKERs

The goal of SAC will be to pick 10 sacs internal model will be modified to from linear to timeseries

The reward will be how close SAC got picking the 10 highest increasing stocks (we will calculate this before hand)

* We know the optimal policy, we can calculate it in advance
* So we should train the inner models against that? (we maybe don't need SAC)


Research:
* How ususable are the tools in https://github.com/AI4Finance-Foundation/FinRL?