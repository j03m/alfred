Next:
* add gold and futures to fundamental data
* add insider events to data as (has_disposal: [score: 0 -> some max], has_acquisition: etc)
  * https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol=IBM&apikey=demo
* you can calculate expected earnings using alphavantage surprise - might be interesting to look at projections of price reflected in expected earnings
* Add analyst predictions to the dataset
* Add news sentiment aggregate to the dataset

Manager:

* Once the above are added, we can 
  * rerun the scripts for generating management training data on a 4 year period
  * split that 3 years train 1 year eval
* Create another set of sacred experiments that look at the timeseries but tries to predict rank 
* train those models

* Calculate performance SPY hold vs top 5 rank monthly

Research:
* How ususable are the tools in https://github.com/AI4Finance-Foundation/FinRL?