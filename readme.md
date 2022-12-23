# LSTM stock price predictor

This notebook is meant to run in google collab (otherwise, you might need to change some of the file paths).

It does the following:

* Trains a model against the sp500
* Lets you visually back test the price predictions visually
* Note: the model is fairly accurate on daily direction, but not magnitude. I'm not a financial advisor, trade at your own risk.
* Using the model it will then interogate coinbase for pricing and provide you with a table of possible daily positive moves on coinbase listed cryptos
* It will also "unscale" price and predicted price points so you can leverage these as possibly entry and exit points on a daily position.

Again: Trade at your own risk. I am a total amatuer at this. 


