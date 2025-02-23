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

### Expanding Our Equity Set

#### Training vanilla.large and vanilla.small v1

Now we have to consider how to train against additional equities. For this experiment since we're not taking into account the timeseries aspect and treating it all
as a tabular classification problem one possibility is concatenating all the files together. This would let us train against the largest set of available data. Another consideration
though is if we want to expand our list of equities using this approach we would have to retrain on all the available data again. Another possibility is to train the model across each file independently. This is computationally cheaper if we add coverage, but runs the risk of catastrophic forgetting 
and possibly as some risk of order dependence. 

I decided to go with initially concatenating with the assumption we could fine tune on the original dataset later. We'll have to make some changes to the `easy` module to save scalers
to be re-used for fine-tuning later. For both initial training and fine-tuning, we will want to use consistent feature scaling. The best approach is usually to calculate scaling parameters 
(e.g., mean, standard deviation) from your initial training dataset (e.g., the first 100 equities) and apply the same scaling transformation to all subsequent data, including new equities. 
This ensures consistency in feature ranges across all data seen by the model. For an illustration of why this is needed check out [scripts/educational/data-distrubtion-shift.py](scripts/educational/data-distrubtion-shift.py).

Running `vanilla.large` (100 layers) with size 1024 against the concatenated list of test tickers gives us decent results, but not as solid as testing a single ticker:

```
BCE:  {'accuracy': tensor(0.9231, device='mps:0'), 'precision': tensor(0.9124, device='mps:0'), 'recall': tensor(0.9425, device='mps:0'), 'f1': tensor(0.9272, device='mps:0')}
```

Oddly, a smart model `vanilla.small` (10 layers) with size 256 yielded: 

```
{'accuracy': tensor(0.9965, device='mps:0'), 'precision': tensor(0.9959, device='mps:0'), 'recall': tensor(0.9973, device='mps:0'), 'f1': tensor(0.9966, device='mps:0')}
```

Which made me pause. However, after some research I found that he larger net might be dealing with vanishing gradients due to its size.

#### Evaluating vanilla.large and vanilla.small v1

Running `scripts/easy_evaler.py` we get a sense of how both larger and small models will perform on other equities.

Small: `Evaluation: Loss: 0.30814009917951446 stats: {'accuracy': tensor(0.6591, device='mps:0'), 'precision': tensor(0.6602, device='mps:0'), 'recall': tensor(0.6616, device='mps:0'), 'f1': tensor(0.6609, device='mps:0')}`

Large: `Evaluation: Loss: 0.27523269157151437 stats: {'accuracy': tensor(0.6645, device='mps:0'), 'precision': tensor(0.6631, device='mps:0'), 'recall': tensor(0.6746, device='mps:0'), 'f1': tensor(0.6688, device='mps:0')}`

Even though the smaller model seemed to fit the training data much better

#### Minor change to activation

During a code review (from an AI lol) I got the following advice:

>  Batch normalization is most effective when applied before the activation function. By applying it after, you're normalizing the output of the Tanh, which is already bounded and might have regions of very small gradients. By applying batch norm before the Tanh, you ensure that the input to the Tanh has a good distribution (mean 0, variance 1), preventing it from saturating and improving gradient flow. The layers_pairs line is changed to iterate over linear layers and batchnorm layers in the right order, and the activation is moved after the batch norm in the forward method.

I made this change and retrained, and revaluated the models. For the large model, that didn't give much if any lift, I wonder about the soundness of the advice.

Training ended a little later:

```text
Epoch 2020 - patience 1000/1000 - mean loss: 0.003793424164980714 vs best loss: 0.001796873403785365 - Stats: 
BCE:  {'accuracy': tensor(0.9166, device='mps:0'), 'precision': tensor(0.9035, device='mps:0'), 'recall': tensor(0.9398, device='mps:0'), 'f1': tensor(0.9213, device='mps:0')}
```

Eval ended about the same:

```text
Evaluation: Loss: 0.2811918371461898 stats: {'accuracy': tensor(0.6602, device='mps:0'), 'precision': tensor(0.6645, device='mps:0'), 'recall': tensor(0.6530, device='mps:0'), 'f1': tensor(0.6587, device='mps:0')}
```

That said we may still be suffering from disappearing gradients due to the depth of the network. We'll look at that next. 

For the small model we were again, highly tuned to the training set, but our eval performance was roughly the same:

```text
BCE:  {'accuracy': tensor(0.9964, device='mps:0'), 'precision': tensor(0.9958, device='mps:0'), 'recall': tensor(0.9973, device='mps:0'), 'f1': tensor(0.9965, device='mps:0')}

Evaluation: Loss: 0.31217595087277394 stats: {'accuracy': tensor(0.6602, device='mps:0'), 'precision': tensor(0.6524, device='mps:0'), 'recall': tensor(0.6918, device='mps:0'), 'f1': tensor(0.6715, device='mps:0')}
```


#### More size, less depth

I decided to train the model again to test the layers as the problem theory. This time I would test the 1024 sized model, but only with 10 layers. The result there mimicked the 
small model in that it was able to get very low error rate on the training data, but not so much during evaluation on the second set:

Training:
```text
Epoch 4990 - patience 1/1000 - mean loss: 1.8080596676220705e-11 vs best loss: 1.803881936065778e-11 - Stats: 
BCE:  {'accuracy': tensor(0.9970, device='mps:0'), 'precision': tensor(0.9965, device='mps:0'), 'recall': tensor(0.9978, device='mps:0'), 'f1': tensor(0.9971, device='mps:0')}
last learning rate: [0.001]
```

Eval:
```
Evaluation: Loss: 0.3111821541515449 stats: {'accuracy': tensor(0.6634, device='mps:0'), 'precision': tensor(0.6577, device='mps:0'), 'recall': tensor(0.6875, device='mps:0'), 'f1': tensor(0.6723, device='mps:0')}
```

So what we've established here is for this particular case:

* More isn't always better - specifically more layers
* We may be overfitting to the training set

I don't fully grok the ins and outs of resnet, but the idea that it helps solve deep network issues is covered here:
https://medium.com/@ibtedaazeem/understanding-resnet-architecture-a-deep-dive-into-residual-neural-network-2c792e6537a9

To be fair, I don't know that 66% accuracy is a bad score for our use case. In theory, anything that gave us a > 50 chance would be an advantage. We'll look at how to 
backtest the predictions later in the article.

### Dropout - Vanilla.small and Vanilla.large v2

To help deal with the issue of overfitting, I introduced dropout to the vanilla network which I had previous left out
on purpose to make sure we were using the simplest possible example.

We add the dropout after activation. Because dropout zeros out a fraction of neuron activations during training, applying it
before batch normalization would lead to inconsistent statistics being calculated in that layer. Putting the dropout after 
activation also isn't ideal. For example if we were using ReLU we could end up with many activations being zero-ed before
dropout is even applied.

TODO: review the script we added educational regarding order

#### Retraining and retesting 

Notably, because the 100 layer experiment seems to be such a wash and is much more expensive to train I will initially run another test of `vanilla.small` and our 10 layer, 1024 size
model (now dubbed `vanilla.medium`) first. If either of those look like had a performance improvement on the eval set, I'll rerun `vanilla.large`. 

I should probably note if you are following along here that its important we wipe out the models prior to retesting! Use the [clear_models.py](scripts/experiments/clear_models.py)

Drop out didn't significantly move the needle with our evaluation still being roughly 66% accuracy:

```text
Evaluation: Loss: 0.3080481131546376 stats: {'accuracy': tensor(0.6677, device='mps:0'), 'precision': tensor(0.6619, device='mps:0'), 'recall': tensor(0.6918, device='mps:0'), 'f1': tensor(0.6765, device='mps:0')}
```


## Back Testing Directional Signals

There are a few options available to us here moving forward, we could potentially play around with different model architectures and types and we can potentially increase the number of 
equities we're training against. Given the blessing that we're working with quarterly data, we can likely load up entire universes of equities for training and not hit any physical limits
on modern hardware. That said, I've mentioned before the directional indicator is a bit ham-fisted. If we're going to spend an enormous amount of time or hardware training on more data,
we should migrate the current network to predict something like velocity of the move.  

That said before we do that, given our results, what we can do is get an understanding of how impactful a 66% projection is from a profitability perspective against single equities vs buy and hold of a benchmark.

To do that we can run a basic experiment. We can choose two uncorrelated equities from our evaluation set and for given period of time trade them long/short quarter over quarter in a back
test and compare the result to holding an index over the same time period. This won't be a sophisticated backtest, but it will let us know in theory what our model might perform like as
directional analyst. 

To do this I introduced `alfred.model_backtesting`. There were a few backtesters out in the world already, for example `backtesting.py` and `backtrader` but for various reasons I was jumping 
through hoops implementing what I wanted here which was not a fully featured backtesting solution, but a smoke test of sorts. For example backtrader demands trades be executed on the next bar open
which is probably realistic, but not what needed for my quarter over quarter testing. When I did finally get it to work, something was going wrong with the broker execution framework and 
my buy orders were never executing. I opted for two of my own models. `SimpleBackTester` and `NuancedBackTester` which can see in [scripts/backtesting/vanilla-backtest.py](scripts/backtesting/vanilla-backtest.py).

The former simply looks at 1 or 0 signals and either shorts or longs based on the signal and uses the opposite direction to take the other side. Nuanced backtester moves away from our boolean model toward
a place where we use a score to determine how long or short we want to be. (More on that later).

Testing against our tickers our results are very mixed. This set of executions:

```shell
python scripts/backtesting/vanilla-backtest.py --test_ticker=F &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=AVB &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=MAC &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=ZION &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=FCX &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=ROST &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=SWK &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=PDCO &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=JNPR &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=T &&
python scripts/backtesting/vanilla-backtest.py --test_ticker=HIG &&
echo "done"
```
Ended up with a mix of positive vs negative win rates and profit levels vs buy hold. Nothing I would want to trade against but an interesting experiment. 

## Re-Examining Feature Importance

### Poor Man's Ablation
Earlier we looked at correlation of various features to our class of buy or sell. Now that we have a trained model I'd like to revisit that idea by doing a sort of rough sketch
sensitivity analysis or feature ablation and then looking at SHAP to see if it gives us different outcome. The idea here being if there are features we can fully eliminate when we 
try to train on more data or do a more complex prediction set we can perhaps be more efficient. [scripts/experiments/poor-mans-feature-ablation.py](scripts/experiments/poor-mans-feature-ablation.py)
implement a loop where we ablate each feature column and re-evaluate the model. 

The idea here is that if the model gets worse when the feature is ablated, then the feature is needed. If the model gets better when the feature is ablated, then the feature is perhaps noise. 
We also calculate differences for our other BCELoss stats. 

My theory is that we can drop anything that has both a positive impact on loss and negative impact on F1 (Ie, loss goes up, f1 goes down)

````text
                   Feature  Loss Change  f1 Change
25                   2year     0.012671  -0.009751
46             delta_3year     0.006012  -0.002079
21                   BTC=F     0.005888  -0.000988
24                   3year     0.005429  -0.000544
51      delta_mean_outlook     0.004501  -0.002548
37  delta_Margin_Operating     0.003540  -0.003140
34          delta_surprise     0.002971  -0.000512
0          Close_diff_MA_7     0.002962  -0.004616
29            mean_outlook     0.002173  -0.002079
12                surprise     0.001501  -0.003140
16       Margin_Net_Profit     0.001170  -0.002630
18                     SPY     0.000709  -0.014790
50    delta_mean_sentiment     0.000290  -0.001568
20                    BZ=F     0.000120  -0.000544
````
These columns are likely on the chopping block.

### SHAP






## Predicting Magnitudes

Now that we have a decent signal, prior to spending more money on training I wanted to mature our model. In this manner I want to try a few things:

1) Have the model predict a score representing the percent increase of the stock and perhaps some confidence measure
2) Have the model operate on a universe of evaluation tickers in its backtest. Ie, rather than trading against one ticker, look at all tickers and chose the one with the best score
3) Consider reinforcement learning for the above process

### Changing up our model and training data




## LSTMs

## Convolutions

## Transformer  





