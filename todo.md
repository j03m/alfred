Next Plans:

* Can we actually predict rank? Using a small set of stocks conduct a study to see if its even possible to train on the parms we have
and get matching rank value. Even if we overfit.
  * It's looking grim. Our MSE on the eval set is a whole number lol
  * Gonna review the columns and see if they can be simplified in some way and that they are all scaled correctly
  * Can try to train on a larger set

Other Experiments:

* Ablation Study
  * Drop columns? 
  * Using the small model to determine impact if any?
  * lstm only?
  
* We never validated that the N day diff from moving avg was helpful - how do we validate that?
* Back to reinforcement learning - what can we do here?
  



Longer Term:

* Long term we should think about how to measure optimal with risk
  * Ie, ranks right now don't take into account risk, just return

* how to find neglected stocks? Low volume, lack of index and etf inclusion
  * Get our data feed to include a list of ALL tickers on US indexes
  * Devise algo for "overlooked"
* alfred could look at full universes soon

* Looking at "the spreadsheet from our course" can we use that input as training data and come up with optimal quadrant? 

Research:
* Finish book
* How ususable are the tools in https://github.com/AI4Finance-Foundation/FinRL?