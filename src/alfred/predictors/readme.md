Based on the StockFormer project we will roll a series of prediction focused Transformers.

We will predict 5 days, 5 weeks and 5 months into the time series

* Download and cache - can we use finrl or qlibs dataloaders - they seem pretty thorough
* normalize (no prices, just diffs against moving avgs)
* Train Predict
* Integrate and test predictions against real prices


RL:

Given a basket of stocks
The Env should use the predictors to predict an outcome
Train SAC (spinning up impl) using StockFormers transformer blending technique on predictions and relationships
Understand the action space of buy/sell assets in the basket
Unerstand how the impls for portfolio optimization track history and then benchmark it



