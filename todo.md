* ~~finalize linear series experiment~~
* ~~Try projecting 2, 3, 4, 5 bars~~
* ~~Try Conv1d projections~~
* ~~Fix advanced lstm, stock-former, transformer models~~
  

* Set up a generalization test with a full basic of test stocks
  * Experiment config and manager
    * ~~Review: https://github.com/IDSIA/sacred~~
    * Requirements:
      * ~~pull + cache data for experiment period~~
        * ~~finish individual tickers~~
          ~~* price is nan :(~~
        * finalize a new dataset that works for single files - retest it
          * TODO We're hitting a wall pre-scaling the data and then trying to
            train based on a subset range. Change the caching to NOT scale, rather you
            should only scale for the range you intend to train on.
          
        * Consider experiments that leverage new training data columns
      * Configure all variations of models, epochs for each
      * run all experiments
      * Capture results
      * Eval must be on a fresh model loaded
      * Consider a json format we can feed to an llm for comment
* Implement some of the more advanced models if we need - temporal fusion tranformers (TFTs later)

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