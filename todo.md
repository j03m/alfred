* Modify the dataloaders and trainers to use the symbol meta data
* Create an experiment descriptor
  * models
  * sizes
  * column permuations
* Implement a 3 tier loop in Sacred
  * Iterate over model type
  * Size
  * Column permuations (from another meta file)
  * Train against training symbols
  * Eval against eval symbols

Analyst Ideas
* series translation to trend
* Bigger models?
* fold in more datapoints
  * you can calculate expected earnings using alphavantage surprise - might be interesting to look at projections of price reflected in expected earnings
* Can we project earnings instead of price?
* Research + news feed analyst 
  * should be able digest research and news into datapoints that the portfolio manager can leverage
Research:
* How ususable are the tools in https://github.com/AI4Finance-Foundation/FinRL?