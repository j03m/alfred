* ~~finalize linear series experiment~~
* ~~Try projecting 2, 3, 4, 5 bars~~
* ~~Try Conv1d projections~~
* ~~Fix advanced lstm, stock-former, transformer models~~
  * Most are done, but I removed informer
    * The TODO here is to implement temporal fusion tranformers (TFTs later)

* Set up a generalization test with a full basic of test stocks
  * Experiment config and manager
    * ~~Review: https://github.com/IDSIA/sacred~~
    * Requirements:
      * ~~pull + cache data for experiment period~~
        * ~~finish individual tickers~~
          ~~* price is nan :(~~
        * finalize a new dataset that works for single files - retest it
        * Consider experiments that leverage new training data columns
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