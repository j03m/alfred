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
          * ~~finalize a new dataset that works for single files - retest it~~ 
            * Consider experiments that leverage new training data columns
                * We essentially want to run experiments across N model types
                * With some combination of training columns to see who gets the most accurate predictions
                  * https://github.com/IDSIA/sacred
            * ```
              # given a set of instruments (stocks)
              # create a deterministic start and end based on a seed, based on available start times
                  # allow seed to be passed into the experiment
              # given ALL instruments
                  # train on period a
                  # eval on period b
              # How to quantify overall performance?
                  # min, max, mean, std?
                  # mean might not be the best, best mean plus lowest std might be best
                  # we should judge each analyst on the most accurate 5, 10, 15, 30 day projections?
              # Then permute data (columns), model type, model size, training time
                  # how do we approach this? Do we just run all permutations? This could take forever.
              
              # Can lstm be trained on multiple symbols instead of one?
              # we can make a multi equity dataset where each close is a column
              # can we train the lstm to make 1, 5, 15, 30 day predictions?
            ```
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