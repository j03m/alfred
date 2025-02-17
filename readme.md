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
To grab these stocks we'll use `scripts/scanner.py`. It will check out data directory output a list that enough history for the experiment


### Lack of Correlation

Next up we'll need to find some stocks that are uncorrelated. My approach here is to create a correlation matrix then pick a stock (AAPL for test), (F for eval) then select the stock least correlated to it.
Then pick a stock least correlated to the previous two so on and so forth. We'll do this for a test and eval set.

```
(venv) jmordetsky in ~/alfred (main) > python scripts/scanner.py --starting_ticker=AAPL
['AAPL', 'LNC', 'ATI', 'PRGO', 'JNPR', 'THC', 'MO', 'MAT', 'ALB', 'ROST', 'NEM']
(venv) jmordetsky in ~/alfred (main) > python scripts/scanner.py --starting_ticker=F
['F', 'AVB', 'MAC', 'ZION', 'FCX', 'ROST', 'SWK', 'PDCO', 'JNPR', 'T', 'HIG']
```

### The Label

We'll run `./scripts/create-basic-direction-data.py` against these tickers. Our label indicates "Is the future price higher than todays?"

As such we shift the future price to the past and check if it is > than the current price ala:

```
# our boolean checks to see if this ROW predicts a future price increase
comparison_result = quarterly_data["Close"].shift(-1) > quarterly_data["Close"]
```

### Correlations?

Next up we should probably take a look at which of our columns have naive correlation to our labels. We can do this with `scripts/feature-correlation.py`:

That generates an interested set of data, which is slightly depressing but not completely surprising (predicting the future is hard :) ). There are no really strong correlators
to quarterly price increase:

````text

PQ: 0.9999999999999999 - Very Strong
Close_diff_MA_90: 0.12892171826831145 - Very Weak
Margin_Gross: 0.1142569938393819 - Very Weak
Margin_Net_Profit: 0.10581040452589217 - Very Weak
Margin_Operating: 0.10279929789032917 - Very Weak
Close_diff_MA_180: 0.10091601601995985 - Very Weak
surprise: 0.08847178902031437 - Very Weak
^VIX: 0.0772748721272654 - Very Weak
surprisePercentage: 0.0746242823372321 - Very Weak
delta_surprisePercentage: 0.06686143047333952 - Very Weak
CL=F: 0.06139709088093088 - Very Weak
reportedEPS: 0.06131443260157919 - Very Weak
SPY: 0.060237998259389094 - Very Weak
estimatedEPS: 0.05514866597700653 - Very Weak
delta_^VIX: 0.05323173806282268 - Very Weak
insider_acquisition: 0.0398854615692598 - Very Weak
insider_disposal: 0.03408291420884296 - Very Weak
delta_insider_acquisition: 0.027500301568763978 - Very Weak
Close_diff_MA_30: 0.019272988736387695 - Very Weak
Close_diff_MA_7: 0.01756933471949962 - Very Weak
delta_insider_disposal: 0.014982149264578538 - Very Weak
delta_surprise: 0.010419266205410624 - Very Weak
Close: 0.010067864423516168 - Very Weak
Volume: 0.001043278963139337 - Very Weak
delta_SPY: -0.020424790826131415 - Very Weak
delta_CL=F: -0.03241308412432996 - Very Weak
delta_reportedEPS: -0.04088861812992127 - Very Weak
delta_estimatedEPS: -0.0466051589440676 - Very Weak
delta_mean_sentiment: -0.05141544424189076 - Very Weak
BTC=F: -0.05172377221830401 - Very Weak
delta_mean_outlook: -0.05399907814650073 - Very Weak
mean_sentiment: -0.06829949796739071 - Very Weak
mean_outlook: -0.0694016695074835 - Very Weak
delta_2year: -0.07574503914903966 - Very Weak
delta_Volume: -0.0775137319976775 - Very Weak
delta_Close: -0.08216866581240505 - Very Weak
delta_BZ=F: -0.08218229083223447 - Very Weak
delta_BTC=F: -0.08647960943079801 - Very Weak
2year: -0.08971257073156266 - Very Weak
10year: -0.09101558180056132 - Very Weak
3year: -0.09101558180056132 - Very Weak
5year: -0.0919723430909219 - Very Weak
delta_5year: -0.09339736617097226 - Very Weak
delta_10year: -0.09405986175375887 - Very Weak
delta_3year: -0.09405986175375887 - Very Weak
BZ=F: -0.13956943885419681 - Very Weak
delta_Margin_Gross: -0.1435668195894457 - Very Weak
Volume_diff_MA_90: -0.14944746202121892 - Very Weak
Volume_diff_MA_180: -0.15110832793311033 - Very Weak
Volume_diff_MA_30: -0.1574657530343537 - Very Weak
delta_Margin_Operating: -0.16011068939526674 - Very Weak
delta_Margin_Net_Profit: -0.17011037553899938 - Very Weak
Volume_diff_MA_7: -0.19504551648570898 - Weak
````
While all of our columns are weak our top performers were kinda surprising:

```
delta_5year: -0.09339736617097226 - Very Weak
delta_10year: -0.09405986175375887 - Very Weak
delta_3year: -0.09405986175375887 - Very Weak
BZ=F: -0.13956943885419681 - Very Weak
delta_Margin_Gross: -0.1435668195894457 - Very Weak
Volume_diff_MA_90: -0.14944746202121892 - Very Weak
Volume_diff_MA_180: -0.15110832793311033 - Very Weak
Volume_diff_MA_30: -0.1574657530343537 - Very Weak
delta_Margin_Operating: -0.16011068939526674 - Very Weak
delta_Margin_Net_Profit: -0.17011037553899938 - Very Weak
Volume_diff_MA_7: -0.19504551648570898 - Weak
```

These can be categorized as changes in interest rates, volume patterns and some fundemental data. 

### The model

So I guess the question is - are all of our weak or very weak correlations enough to allow for a well trained model to make a prediction?

To do so we need N features to go into a model that is of size Y to make a prediction from 0 (false) to 1 (true) and to clamp round
to the closest answer.

For my first crack at training I purposefully avoided LSTMs, GANs or Transformers that would be better at analyzing the historical aspects of 
the time series and used a plan old Neural Network which you can find [here](src/alfred/models/vanilla.py). Our first run
using 10 layers of size 256 didn't go so well. Trainer script is [here](scripts/experiments/easy_test.py) which leverages an "easy"
training wrapper I used for alfred to make this sort of thing faster, capped out with a pretty abysmal mse of 0.43 and ran out of 
patience cycling around numbers in this area. 

I decided to try a larger model, this time using 100 layers and 1024 params. All models served up via alfred can be easily
served up via `alfred.model_persistence.model_from_config` function. Using the framework models are stashed in and can be restored from 
mongodb. I don't really love mongo, but it was easy to use and I have a bunch of gpus floating around the network at my house and
this was a clever hack for being able to run multiple experiments at once on different computers.

Given `(size^2 * layers-1) + size * layers` we move from `(256**2 * 9) + 256 *10=592,384` to
`(1024**2 * 99) + (1024 * 100)=103,911,424`. A sizeable shift. These models can be found as `vanilla.small` and
`vanilla.large`.

However, even at this size the model plateaued at roughly at the same rate :(. 

I then also noticed, the `CustomScaler` I wrote had a major oversight in it. When I looked at the first level of activations
in our model, I noticed the values were exploding in one iteration. With some help from an AI, I spotted a massive value
in column 51 of my input data. Turns out, the scaler configuration supplied to the scaler was missing columns and there was
nothing in alfred's `CustomScaler` to catch it. I added that code, fixed the input and the subsequent values looked
much more reasonable.

> Hard lesson: LOOK CLOSELY AT YOUR INPUT before you look at other stuff in the case of exploding gradients! 

### Training and Performance on a Single Equity

The very simple training script using `alfred's` easy wrapper is [here](scripts/experiments/easy_test.py)). This only trains against 
one ticker (APPL) the results of which are below. We'll expand on this, including other tickers for training and evaluating against
tickers we never trained on later.

After running a quick test at 10 layers and size 256 I achieved decent performance on the training set, but increasing the size of the model
to 100 layers at 1024 I achieved event better:

| Metric    | Previous Run (Epoch 2300) | Current Run (Epoch 1190) |
|-----------|---------------------------|--------------------------|
| Mean Loss | 0.2085                    | 0.0577                   |
| Accuracy  | 0.7024                    | 0.9582                   |
| Precision | 0.7024                    | 0.9449                   |
| Recall    | 0.9999                    | 0.9987                   |
| F1-Score  | 0.8252                    | 0.9711                   |


First I should while I was happy with this outcome, the real test is on data the model hasn't seen so there is no real reason to get excited here. But after a day of abysmal scoring
due to scaling issues I'm happy to have landed here. 

Because we're trying to predict if next quarter will be a good trade, we need to think a bit about the game theory of outcomes. While 
we likely cannot trade off binary classes, the thought exercise about which of these metrics is most valuable to trading environment is a useful one. For simplicty, 
I'll assume we will only issue buy orders or hold on True. On False we will either avoid buying or sell our position.

*Accuracy* - gives us our overall performance at 95.8%. This is our total number of correct predictions. Ie, we want to be mostly correct in either direction
with our positions so we make and don't lose money. 

*Precision* - This is a measure of our true positives. Ie, if we make a trade thinking the market will go up how often are we right? Ie, we would be wrong around 5.5% of the time. 
That may not seem like a lot but depending on how wrong we are that could hurt.

*Recall* - Recall lets us know, of the universe of good trades, how many are we catching? So this being almost 100% (99.87) mixed with our precision score indicates that maybe we lean
toward predicting positive trades more often than not. Ie, we're getting all the good trades, but 5% of the time we're off.

*F1* - Gives us the balance of precision and recall. 


### Evaluating on a single equity

Earlier we spent some time picking uncorrelated stocks with `LNC` being the least correlated to `AAPL` so we'll run an evaluation against that.

Just as a sanity check we'll use `alfred's` easy evaler to check how our model performs 
against an uncorrelated stock. The results are unsurprisingly pretty abysmal:

```
Evaluation: Loss: 0.323896328608195 stats: {'accuracy': tensor(0.5357, device='mps:0'), 'precision': tensor(0.5417, device='mps:0'), 'recall': tensor(0.8667, device='mps:0'), 'f1': tensor(0.6667, device='mps:0')}
```

So that leaves us with the question, if we train against our corpus of uncorrelated tickers
can we do better against a 2nd corpus of uncorrelated tickers?

### Expanding Training




## LSTMs

## Convolutions

## Transformer





