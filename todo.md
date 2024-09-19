* ~~finalize linear series experiment~~
* ~~Try projecting 2, 3, 4, 5 bars~~
* ~~Try Conv1d projections~~
* Fix advanced lstm, informer, transformer models
  * advanced works but seems to suck
  * ~trans-am is busted~
  * Stockformer (transformer model only)
  * Informer (transformer model only)
  * Pytorch timeseries model
  
* Set up a generalization test with a full basic of test stocks
  * Experiment config and manager
    * pull + cache data for experiment period
    * Configure all variations of models, epochs for each
    * run all experiments
    * Capture results
    * Eval must be on a fresh model loaded
    * Consider a json format we can feed to an llm for comment

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