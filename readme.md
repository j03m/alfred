Alfred is an experimental set of machine learning for markets and portfolio management. 

The plan for Alfred is to have 3 parts:

* A series of technical and fundamental analyst models that work on available data to try and project prices and future fundamentals
* One or maybe more language model analysts that will function as news and research digesters
* A reinforcement learning model portfolio manager that will ingest data from the ensemble of analysts and try to predict optimal portfolio makeup for some time period N. 

I'm currently in the stage of weighing effectiveness vs size for various technical analysts.  


# Experiment 1 - Projecting a quarterly velocity score

For a basic experiment we will examine whether a basic neural network without glancing back at detailed historical data can look at a roll up of quarterly features
and project an equity looks positive or negative for the following quarter. 

## Features

TODO - outline columns and the data that produces them

## Testing Data Approach

To get a solid experiment we want to:

1) Select a time range of reasonable length, since we'll be working across quarters. We'll only consider stocks exist and have existed since 2004
2) Select a handful of test stocks to train on that meet the time criteria and are relatively uncorrelated
3) Select another, again uncorrelated set of stocks and evaluate

### Candidates with enough history
To grab these stocks we'll uses `scripts/scanner.py`. It will check out data directory output a list that enough history for the experiment


### Lack of Correlation

Next up we'll need to find some stocks that are uncorrelated. My approach here is to create a correlation matrix then pick a stock (AAPL for test), (F for eval) then select the stock least correlated to it.
Then pick a stock least correlatd to the previous two so on and so forth. We'll do this for a test and eval set. 


