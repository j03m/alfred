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

### Training and Performance

The very simple training script using `alfred's` `easy` wrapper is [here](scripts/experiments/easy_trainer.py)). This only trains against 
some subset of tickers. For now I used one, AAPL - the results of which are below. We'll expand on this, including other tickers for training and evaluating against
tickers we never trained on later.

I ran a quick test at 10 layers and size 256 I achieved decent performance on the training set (vanilla.small/256 is the default model):

> (venv) jmordetsky in ~/alfred (main) > python scripts/experiments/easy_trainer.py --tickers=metadata/basic-tickers.json

| Metric    | Value          |
|-----------|----------------|
| Best Loss | 7.9142e-07     |
| Accuracy  | 1.0000 (mps:0) |
| Precision | 1.0000 (mps:0) |
| Recall    | 1.0000 (mps:0) |
| F1 Score  | 1.0000 (mps:0) |



I then tried `vanilla.medium` which is 10 layers at 1024:

> (venv) jmordetsky in ~/alfred (main) > python scripts/experiments/easy_trainer.py --tickers=metadata/basic-tickers.json --model vanilla.medium --size 1024

| Metric    | Value          |
|-----------|----------------|
| Best Loss | 5.7641e-08     |
| Accuracy  | 1.0000 (mps:0) |
| Precision | 1.0000 (mps:0) |
| Recall    | 1.0000 (mps:0) |
| F1 Score  | 1.0000 (mps:0) |

First I should while I was happy with this outcome, the real test is on data the model hasn't seen so there is no real reason to get excited here. But after a day of abysmal scoring
due to scaling issues I'm happy to have landed here. Seeing 1's across the board for our metrics is nice, but also worrying about the possibily
of over fitting against our training stocks.

Because we're trying to predict if next quarter will be a good trade, we need to think a bit about the game theory of outcomes. While 
we likely cannot trade off binary classes, the thought exercise about which of these metrics is most valuable to trading environment is a useful one. For simplicty, 
I'll assume we will only issue buy orders or hold on True. On False we will either avoid buying or sell our position.

To get a sense on how we perform against data we've never seen, we'll use alfred's `easy_evaler` against our evaluation stocks. We'll test
`vanilla.small` and `vanilla.medium`

Medium: 

| Metric    | Value              |
|-----------|--------------------|
| Loss      | 6.602123483308945  |
| Accuracy  | 0.6287878751754761 |
| Precision | 0.6306695342063904 |
| Recall    | 0.6293103694915771 |
| F1        | 0.6299892067909241 |

Small:

| Metric    | Value          |
|-----------|----------------|
| Loss      | 13.9524        |
| Accuracy  | 0.5357 (mps:0) |
| Precision | 0.5313 (mps:0) |
| Recall    | 0.6401 (mps:0) |
| F1 Score  | 0.5806 (mps:0) |


*Accuracy* - gives us our overall performance at 95.8%. This is our total number of correct predictions. Ie, we want to be mostly correct in either direction
with our positions so we make and don't lose money. 

*Precision* - This is a measure of our true positives. Ie, if we make a trade thinking the market will go up how often are we right? Ie, we would be wrong around 5.5% of the time. 
That may not seem like a lot but depending on how wrong we are that could hurt.

*Recall* - Recall lets us know, of the universe of good trades, how many are we catching? So this being almost 100% (99.87) mixed with our precision score indicates that maybe we lean
toward predicting positive trades more often than not. Ie, we're getting all the good trades, but 5% of the time we're off.

*F1* - Gives us the balance of precision and recall. 

From a purely theoretical point of view, an F1 score of >  50 should allow us to make money at least on the
basket of evaluation stocks in question. However, there is that age old xkcd commic we should keep in mind:

![img.png](img.png)

There is more to a profitable trading strategy to be considered :).



## Back Testing Directional Signals

There are a few options available to us here moving forward, we could potentially play around with different model architectures and types and we can potentially increase the number of 
equities we're training against. Given the blessing that we're working with quarterly data, we can likely load up entire universes of equities for training and not hit any physical limits
on modern hardware. That said, I've mentioned before the directional indicator is a bit ham-fisted. If we're going to spend an enormous amount of time or hardware training on more data,
we should migrate the current network to predict something like the velocity or magnitude of the move.  

That said before we do that, given our results, what we can do is get an understanding of how impactful a 66% projection is from a profitability perspective against single equities vs buy and hold of a benchmark.

To do that we can run a basic experiment. We can choose take the uncorrelated equities from our evaluation set and for given period of time trade them long/short quarter over quarter in a back
test and compare the result to holding an index over the same time period. This won't be a sophisticated backtest, but it will let us know in theory what our model might perform like as
directional analyst. 

To do this I introduced `alfred.model_backtesting`. There were a few backtesters out in the world already, for example `backtesting.py` and `backtrader` but for various reasons I was jumping 
through hoops implementing what I wanted here, which was not a fully featured backtesting solution, but a smoke test of sorts. For example backtrader demands trades be executed on the next bar open
which is probably realistic, but not what needed for my quarter over quarter testing. When I did finally get it to work, something was going wrong with the broker execution framework and 
my buy orders were never executing. I opted for two of my own models. `SimpleBackTester` and `NuancedBackTester` which can see in [scripts/backtesting/vanilla-backtest.py](scripts/backtesting/vanilla-backtest.py).

The former simply checks if the prediction > 0.5. If so it takes a long posisiton, otherwise it takes a short position. Opposite signals are exits. The latter introduces a confidence level to the score. If the signal is > 0.7 or < 0.3 it will consider that a strong indicator to go long or short. A lesser score of >= 0.5 or < 0.5 is a signal to get out of a trade if we're in one. 

Seasoned traders will probably giggle at my lack of sophistication here, but again, I just wanted to get a sense of how our prediction scores would play out.

I ran two tests, both using `vanilla.medium`:

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

Results were mixed for the simple back tests with roughly equal winners and losers and a mixed bag from a win rate perspective.

TODO: build a summary 



## Re-Examining Feature Importance

### Poor Man's Ablation
Earlier we looked at correlation of various features to our class of buy or sell. Now that we have a trained model I'd like to revisit that idea by doing a sort of rough sketch
sensitivity analysis or feature ablation and then looking at SHAP to see if it gives us different outcome. The idea here being if there are features we can fully eliminate when we 
try to train on more data or do a more complex prediction set we can perhaps be more efficient. [scripts/experiments/poor-mans-feature-ablation.py](scripts/experiments/poor-mans-feature-ablation.py)
implement a loop where we ablate each feature column and re-evaluate the model. 

The idea here is that if the model gets worse when the feature is ablated, then the feature is needed. If the model gets better when the feature is ablated, then the feature is perhaps noise. 
We also calculate differences for our other BCELoss stats. 

My theory is that we could drop anything that has both a positive impact on loss and negative impact on F1 (Ie, loss goes up, f1 goes down). But there wasn't much that fit this criteria:

````text
Ablation Study Results:
           Feature  Loss Change  f1 Change
0  Close_diff_MA_7     0.697478  -0.003172
````

Only `Close_diff_MA_7` ended up on the chopping block.

### SHAP

TODO: Revisit shap, I kept getting an error with the vanilla model: 

```
AssertionError: The SHAP explanations do not sum up to the model's output! This is either because of a rounding error or because an operator in 
your computation graph was not fully supported. If the sum difference of %f is significant compared to the scale of your model outputs, 
please post as a github issue, with a reproducible example so we can debug it. Used framework: pytorch - Max. diff: [some value] - Tolerance: 0.01
```

## Predicting Magnitudes

Now that we have a decent signal, prior to spending more money on training I wanted to mature our model. In this manner I want to try a few things:

1) Have the model predict a score representing the percent increase of the stock and perhaps some confidence measure
2) Have the model operate on a universe of evaluation tickers in its backtest. Ie, rather than trading against one ticker, look at all tickers and chose the one with the best score
3) Consider reinforcement learning for the above process

To do this, we'll need to recreate our training data. We'll add an `--operation` parameter to `scripts/create-basic-direction-data.py` to either be `direction` (bool) or `magnitude` 
(+/- percent change).

Changing the model slightly to not use sigmoid as its final activation, I retrained the model to look at its overall mse on the eval set. 

From here, we'll try something different. Rather than taking a position in the same stock based on teh signal, we'll run each of our evaluation stocks
through the backtester, but we'll only take a position in the stock with the highest predicted magnitude. 

From there based on the results we'll make a  call to train the model on a massive set of data (all stocks).

#### Try 1 - Same model, nn.Identity and MSE

I tweaked the model just slightly before running this test. I replaced the `nn.Sigmoid` activation with `nn.Identity` as the final
activation to `vanilla.medium` and gave it `nn.MSE` as our loss function. I also rolled a new accumulator `RegressionAccumulator` which
track some metrics in the way our `BCEAccumulator` did, the most important of which was `sign accuracy`. However, when running this test
I could not really get a decent level of loss and sign accuracy was never much higher than 50%. I'll explain why this was important to me below.

#### Try 2 - Same model, nn.Identity and custom Loss

After doing some thinking and some research I decided to try crafting my own loss function. The idea being I wanted to punish the model 
harder for missing sign. MSE measure a broad error in either direction. But in our case, it is actually much worse to get the direction wrong than it is for the magnitude to be off. We also know with our classifier runs that we should realy be able to do a good job on direction. Magnitude is essentially the ability for use to decide which of things headed in a direction is going there the fastest so we can certainly tolerate more error there than we can on direction. Ie, prediction that a stock will be +3% and it actually ends up going +1% is must better than prediction a stock will go up +1% and it really goes up -1%. Both are off by 2% but one lead us to a loss.

Initially I constructed `MSEWithSignPenalty` which you can see in `src/alfred/model_metrics/loss.py`. The idea here is based on some weight, we can apply the sign error to the mse and use that as the loss. I also rolled a version of this that uses Huber `HuberWithSignPenalty` and `SignErrorRatioLoss` which uses a score to either amplify or reduce mse and the sign error at some ratio. For example the default params for the ration are 0.5 to 2. This in theory dampens the value error and amplifies the sign error.

I also changed the final activation to Tanh. I broke the rule of tweaking too many thing here, but decided to take a shotgun approach because my first try was so lack luster in its ability to memorize the training data.

Huberloss and an increased model size (2024) gave us during training:

```text
Out of patience at epoch 1837. Patience count: 500/501. Limit: 500
Best loss: 0.1627537259613244, Best Stats: {'mae': 0.05751076340675354, 'r2': 0.07053571939468384, 'pearson_corr': 0.3521348536014557, 'sign_accuracy': 0.9211391018619934}
```

And for eval:
```text
Evaluation: Loss: 0.9553842600552626 stats: {'mae': 0.10528900474309921, 'r2': 0.05812478065490723, 'pearson_corr': 0.40794986486434937, 'sign_accuracy': 0.6911281489594743}
```

I decided to try that again with `SignErrorRatioLoss`. 

Evaluating with no training:

```text
(venv) jmordetsky in ~/alfred (main) > python scripts/experiments/easy_evaler.py --tickers ./metadata/basic-tickers.json --model vanilla.small.tanh --size 2048 --loss sign-ratio --file-post-fix=_quarterly_magnitude --label PM

WARNING: You are evaluating an empty model. This model was unknown to the system.

Evaluation: Loss: 1.4829210649276603 stats: {'mae': 0.36226388812065125, 'r2': -0.1713106632232666, 'pearson_corr': -0.021615857258439064, 'sign_accuracy': 0.547645125958379}
```

Best loss on the training data:
```text
(venv) jmordetsky in ~/alfred (main) > python scripts/experiments/easy_trainer.py --tickers ./metadata/basic-tickers.json --model vanilla.small.tanh --size 2048 --loss sign-ratio --file-post-fix=_quarterly_magnitude --label PM

Best loss: 0.6709100175367925, Best Stats: {'mae': 0.06811121851205826, 'r2': 0.07829374074935913, 'pearson_corr': 0.3725942075252533, 'sign_accuracy': 0.864184008762322}

```
Which did slightly worse against from a loss and sign accuracy perspective, which I found suprising.

Eval post training did worse on loss, but slightly better on sign accuracy.
```text
Evaluation: Loss: 1.0630815586506475 stats: {'mae': 0.10861995071172714, 'r2': 0.06449377536773682, 'pearson_corr': 0.40425240993499756, 'sign_accuracy': 0.7020810514786419}
```

I also wrote a script that looks at our label data and given a target accuracy tries to tell us what our mse should be `scripts/analyze-loss.py`. I was having a hard time knowing based on the range of my data if a "loss" was good or bad (though arguably most of our results are bad lol). This script will take a target r2 calculate mean, variance and std and based on that estimate an mse.
If we want our model to explain 80% of the data’s variance it aims for an MSE that’s 20% of the variance (0.2 × variance) and we would supply 0.8 (the default) to the script. 

```text
TRAINING

Analysis for 'PM' (Desired R² = 0.8):
  Variance (Baseline MSE): 1.4121
  Range: 49.4893
  Mean: 0.0630, Std: 1.1883
  Target MSE (R² = 0.8): 0.2824
  Target MSE (10% Range): 24.4919
  Suggested Target MSE: 24.4919
  Suggested Target RMSE: 4.9489 (avg error)
  Target RMSE as % of range: 10.0%
EVAL

Analysis for 'PM' (Desired R² = 0.8):
  Variance (Baseline MSE): 1.5578
  Range: 49.4874
  Mean: 0.0593, Std: 1.2481
  Target MSE (R² = 0.8): 0.3116
  Target MSE (10% Range): 24.4900
  Suggested Target MSE: 24.4900
  Suggested Target RMSE: 4.9487 (avg error)
  Target RMSE as % of range: 10.0%

```
Looking at our target MSE of roughly `.3` makes me feel okay about the Huber loss result on training but not great about any of our other
results.

### Trying out another network architecture

I wired up a new model called `vanilla.compression.tahn`. This is basically the same setup but rather than N layers of a fixed size,
the model uses 3 layers each decreasing the initial network size by 50%. I tested this at 2048 and landed with a post training loss of:

```text
Best loss: 0.6880370694501646, Best Stats: {'mae': 0.07100587338209152, 'r2': 0.0779222846031189, 'pearson_corr': 0.37335172295570374, 'sign_accuracy': 0.8521358159912377}
```

The eval loss was not a ton better than its previous version:
```text
Evaluation: Loss: 1.0694481803534943 stats: {'mae': 0.10864986479282379, 'r2': 0.0647653341293335, 'pearson_corr': 0.4073215425014496, 'sign_accuracy': 0.6987951807228916}
```

As a last check before looking at a back test and studying our outcomes more closely I trained another model of this kind at twice the size as well as our original model at twice the size 4096.

`vanilla.compress.tahn.4096 huber-sign` training/eval: 
```text
Pre train eval: 

Evaluation: Loss: 1.044123684537822 stats: {'mae': 0.2633186876773834, 'r2': -0.07346522808074951, 'pearson_corr': -0.00986915547400713, 'sign_accuracy': 0.5169769989047097}

Training:

Best loss: 0.1796942362862896, Best Stats: {'mae': 0.060097835958004, 'r2': 0.07199329137802124, 'pearson_corr': 0.3599674701690674, 'sign_accuracy': 0.9014238773274917}


Post train eval:
Evaluation: Loss: 0.6630501580473657 stats: {'mae': 0.11169926077127457, 'r2': 0.057153403759002686, 'pearson_corr': 0.35252252221107483, 'sign_accuracy': 0.6911281489594743}
```

Which, if I'm not mistaken is roughly our best result

`vanilla.small.tahn.4096 huber-sign` training/eval: 
```text
Pre train:
Evaluation: Loss: 1.1708936259664338 stats: {'mae': 0.3010151982307434, 'r2': -0.08084475994110107, 'pearson_corr': 0.02140171267092228, 'sign_accuracy': 0.45454545454545453}

Training:
Best loss: 0.18610617210325817, Best Stats: {'mae': 0.062222350388765335, 'r2': 0.07174217700958252, 'pearson_corr': 0.357318252325058, 'sign_accuracy': 0.8860898138006572}

Post train:
Evaluation: Loss: 0.7189666780402055 stats: {'mae': 0.11250916868448257, 'r2': 0.05626195669174194, 'pearson_corr': 0.36998236179351807, 'sign_accuracy': 0.6637458926615553}
```

The compressed architecture did slightly better, and our 4096 compressed did significantly better than our 2048. Given there was plenty of RAM available
on my M4 I decided to really go for it and train at max size but still against our small dataset. I used size 24000 this time:  

```text
Pre train:
Evaluation: Loss: 1.0985138457396935 stats: {'mae': 0.2700347304344177, 'r2': -0.09530162811279297, 'pearson_corr': -0.036143042147159576, 'sign_accuracy': 0.4939759036144578}


Training:
Out of patience at epoch 627. Patience count: 500/501. Limit: 500
Best loss: 1.3837785391971982, Best Stats: {'mae': 1.0133318901062012, 'r2': -0.900772213935852, 'pearson_corr': 0.04850158095359802, 'sign_accuracy': 0.5355969331872946}

Post train:
Evaluation: Loss: 1.5134698645821933 stats: {'mae': 1.027387261390686, 'r2': -0.9289259910583496, 'pearson_corr': 0.03586142882704735, 'sign_accuracy': 0.5060240963855421}

```
Which was interesting, because it just couldn't seem to learn at all. I couldn't help but wonder if that had something to do with our learning rate or optimizer setup.

Nonetheless, I went a little less aggressive afterward with 16384. It was equally unimpressive. I started to worry that the model was predicting
a range that always evaluated to 1 and -1. To deal with this, I removed the inner Tanh activation layer fearing it might be unnecessary with
the batch normalization right before it. I reran that with a 4096 model:

```text
Best loss: 0.18610617210325817, Best Stats: {'mae': 0.062222350388765335, 'r2': 0.07174217700958252, 'pearson_corr': 0.357318252325058, 'sign_accuracy': 0.8860898138006572}
```
But it had a harder time memorizing the training data than its predecessors. 

I then also modified easy to be more patient, but in the training loop and in LR reduction. I reran this with a 8192 sized model:

I then made some changes, I moved away from the compressed model and instead increased our number of layers AND added a normalization
layer to our model. This lead to the first ever result that didn't end in a patience loss, but rather ran out of epochs:

```
Best loss: 0.113925356641191, Best Stats: {'mae': 0.06269007176160812, 'r2': 0.04430222511291504, 'pearson_corr': 0.247194305062294, 'sign_accuracy': 0.8992332968236583}
```

The loss was our best yet, but our sign accuracy was lower. Emboldened, I doubled the size again and left it overnight but got
roughly the same result:

```text
Training:
Out of patience at epoch 3945. Patience count: 1500/1501. Limit: 1500
Best loss: 0.11361983619281091, Best Stats: {'mae': 0.05934416502714157, 'r2': 0.04976963996887207, 'pearson_corr': 0.26973167061805725, 'sign_accuracy': 0.9112814895947426}

Eval:
Evaluation: Loss: 0.5659032241256086 stats: {'mae': 0.10116428881883621, 'r2': 0.04964590072631836, 'pearson_corr': 0.41185587644577026, 'sign_accuracy': 0.7371303395399781}
```

Our MAE on the eval set means that we're going to roughly off by 10% when predicting magnitudes. Our pearson coeficient measures if the data 
moves linearly together which is weak, and our R^2 is quite low meaning there may be some additional variance we're missing. Our sign accuracy is high though,
higher than our previous accuracy on directionality. 


### Another small backtester

I wanted to see if given variability if we could still build a winning mini portfolio out of our eval set. If for example, if we were go 
long the top 2 positive predictions and short the lowest negative positions, would we outperform the whole basket?




### Changing up our model and training data


## LSTMs vs NNs

## LSTMs with Convolutions

## Transformer?

## Reinforcement Learning





