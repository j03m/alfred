Based on the StockFormer project we will roll a series of prediction focused Transformers. The code in this dir is taken from: https://github.com/gsyyysg/StockFormer

We will predict 4 weeks, 8 weeks, 12 weeks and 24 weeks

* Download and cache - can we use finrl or qlibs dataloaders - they seem pretty thorough
* normalize (no prices, just diffs against moving avgs)
* Train Predict
* Integrate and test predictions against real prices


RL:

Given a basket of stocks
The Env should use the predictors to predict an outcome
Train SAC (spinning up impl) using StockFormers transformer blending technique on predictions and relationships
Understand the action space of buy/sell assets in the basket
Understand how the impls for portfolio optimization track history and then benchmark it



