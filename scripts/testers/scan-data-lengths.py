from alfred.metadata import TickerCategories, ColumnSelector
from alfred.data import CachedStockDataSet, ANALYST_SCALER_CONFIG

ticker_categories = TickerCategories("metadata/spy-ticker-categorization.json")
training = ticker_categories.get(["training", "eval"])

column_selector = ColumnSelector("metadata/column-descriptors.json")
agg_config = column_selector.get_aggregation_config()

seed = 42

# train on all training tickers
for ticker in training:
    print("training against: ", ticker)

    try:
        dataset = CachedStockDataSet(symbol=ticker,
                                     seed=seed,
                                     column_aggregation_config=agg_config,
                                     scaler_config=ANALYST_SCALER_CONFIG,
                                     period_length=365 * 2,
                                     sequence_length=60,
                                     feature_columns=sorted(column_selector.get([
                                         "core",
                                         "fundamentals",
                                         "technicals",
                                         "macro"
                                     ])),
                                     target_columns=["Close"])
    except Exception as e:
        print("Ticker: ", ticker, " failed cuz: ", e)
