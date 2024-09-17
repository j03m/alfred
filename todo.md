* ~~finalize linear series experiment~~
* ~~Try projecting 2, 3, 4, 5 bars~~
* ~~Try Conv1d projections~~
* Fix advanced lstm, informer, transformer models
  * advanced works but seems to suck
  * trans-am is busted
    * Test scaled cumsum - maybe graph the trans-am data vs our data?
      * Retest, graph
    * Positional Encoding has the wrong dims? It is not set up for batch_first
      * replace with a learnable embedding?
      * Time for a experiment config
        * script reads the config, runs all experiments
  * Stockformer + Informer - maybe next
* Set up a generalization test with a full basic of test stocks

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