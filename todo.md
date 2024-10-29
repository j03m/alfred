Next:
* find research feed
  * push research into llm with constructed prompts to break down research into some sort of score
* For timeseries - we need to add something that tracks if more training had been better?
  * ie, did we run out of patience or out of epochs? Can we make epochs infinite and only exit on patience? 
* add gold and futures to fundamental data
* add insider events to data as (has_disposal: [score: 0 -> some max], has_acquisition: etc)
  * https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol=IBM&apikey=demo
* you can calculate expected earnings using alphavantage surprise - might be interesting to look at projections of price reflected in expected earnings
* 

Analyst Ideas
* series translation to trend
* Bigger models?
* fold in more datapoints
* Can we project earnings instead of price?
* Research + news feed analyst 
  * should be able digest research and news into datapoints that the portfolio manager can leverage
Research:
* How ususable are the tools in https://github.com/AI4Finance-Foundation/FinRL?