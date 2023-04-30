import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .data_utils import model_path, sort_date

sc = MinMaxScaler()
model_version_token = "4"
models = {}
all_stock_dfs = []
all_coin_dfs = []
gbl_all_features = [
    "Open", "High",
    "Low",
    "Close",
    "Volume",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "RSI_14",
    "STOCHk_14_3_3",
    "STOCHd_14_3_3",
    "BBL_5_2.0",
    "BBM_5_2.0",
    "BBU_5_2.0",
    "BBB_5_2.0",
    "BBP_5_2.0",
    "OBV",
    "AD",
    "MFI_14",
    "WILLR_14",
    "RVI", "VWAP", "VWAPD"]

gbl_target_column = ["Target"]
gbl_all_columns = gbl_all_features + gbl_target_column

all_model_names = ["lstm_cv",
                   #        "lstm_coins_cv",
                   "lstm_att_cv",
                   "lstm_att_ohlcv",
                   "lstm_cv_rvi",
                   "lstm_cv_vwap",
                   "lstm_ohlc",
                   #        "svm_cv",
                   #        "svm_cv_vwap",
                   "lstm_xgb_cols"
                   ]

model_config = {
    "day_bar_models": [
        "lstm_cv",
        #       "lstm_coins_cv",
        #       "lstm_att_cv",
        #       "lstm_att_ohlcv",
        #       "lstm_cv_rvi",
        #       "lstm_cv_vwap",
        #       "lstm_ohlc",
        #       "svm_cv",
        #       "svm_cv_vwap",
        #       "lstm_xgb_cols"
    ],

    "training_filter": [],
    "backtest_filter": [],
    "15m_bars": ["lstm_15m"],
    "training_types": {
        "lstm_coins_cv": "all"
    },
    "column_sets": {
        "lstm_cv": ["Close", "Volume"],
        "lstm_15m": ["Close", "Volume"],
        "lstm_coins_cv": ["Close", "Volume"],
        "lstm_ohlc": ["Open", "High", "Low", "Close", "Volume"],
        "lstm_att_cv": ["Close", "Volume"],
        "lstm_att_ohlcv": ["Open", "High", "Low", "Close", "Volume"],
        "lstm_cv_rvi": ["Close", "Volume", "RVI"],
        "lstm_cv_vwap": ["Close", "Volume", "VWAP", "VWAPD"],
        "svm_cv": ["Close", "Volume"],
        "svm_cv_vwap": ["Close", "Volume", "VWAP", "VWAPD"],
        "lstm_xgb_cols": [
            'High',
            'Low',
            'Close',
            'MACDh_12_26_9',
            'Open',
            'BBL_5_2.0',
            'AD',
            'MACDs_12_26_9',
            'MACD_12_26_9']
    },
    "build_type": {
        "lstm_att_cv": "att",
        "lstm_att_ohlcv": "att",
        "lstm_xgb_cols": "att"
    },
    "load_type": {
        "svm_cv": "joblib",
        "svm_cv_vwap": "joblib"
    }
}


def pluck(nparray, all_columns, desired_columns):
    df = pd.DataFrame(nparray, columns=all_columns)
    return df[desired_columns].values


def load_all_models():
    training_filter = model_config["training_filter"]
    for name in model_config["day_bar_models"]:
        if name in training_filter:  # skip training models
            print("skipping: ", name)
            continue
        print("loading:", name)
        if name in model_config["load_type"] and model_config["load_type"][name] == "joblib":
            models[name] = joblib.load(model_path + "/" + name + "-" + model_version_token + ".joblib")
        else:
            models[name] = keras.models.load_model(model_path + "/" + name + "-" + model_version_token + ".h15")
    lstm_15m = keras.models.load_model(model_path + "/lstm_15m.h15")
    print("models loaded")
    models_loaded = True


def predict_trade(model, X, columns):
    predicted = model.predict(X, verbose=0).flatten()
    df_pred = pd.DataFrame(predicted, columns=["Close"])
    for column in gbl_all_columns:
        if column != "Close":
            df_pred[column] = 0
    return [predicted, sc.inverse_transform(df_pred)[:, [0]].flatten()]


def predict_config_model_for_product_raw_sup(df_raw, name, product, model, columns):
    [scaled_features, X, y, normal_features] = convert_to_training_dataset(df_raw)
    x = pluck(X, gbl_all_features, columns)
    [predictions_scaled, predictions] = predict_trade(model, x, columns)

    scaled_close = pluck(X, gbl_all_features, ["Close"])
    mse = mean_squared_error(scaled_close, predictions_scaled)

    return [predictions_scaled, predictions, mse, normal_features, scaled_features]

def consensus_prediction(df):
  return mse_weighted_average(df, "Predicted")

def predict_config_model_for_product(df_raw, name, product):
    columns = model_config["column_sets"][name]

    [scaled_features, X, y, normal_features] = convert_to_training_dataset(df_raw)
    x = pluck(X, gbl_all_features, columns)
    [predictions_scaled, predictions] = predict_trade(models[name], x, columns)

    scaled_close = pluck(X, gbl_all_features, ["Close"])
    mse = mean_squared_error(scaled_close, predictions_scaled)

    prediction = predictions[-1]

    return build_trade_model(prediction, predictions_scaled, normal_features, product, name, mse)


def build_trade_model(predicted, predicted_scaled, df, product, name, mse):
    # add predicted
    df_trade = df
    df_trade["Predicted_Scaled"] = predicted_scaled
    df_trade = df_trade.tail(1)

    # add the product, derive a move and percent
    df_trade["Predicted"] = [predicted]
    df_trade["Product"] = product
    df_trade["Model Name"] = name
    df_trade["Move"] = df_trade["Predicted"] - df_trade["Close"]
    df_trade["MSE"] = mse
    df_trade["Percent"] = (df_trade["Move"] / df_trade["Close"]) * 100
    df_trade["RawPercent"] = df_trade["Move"] / df_trade["Close"]
    df_trade["250Fees"] = (250 * 0.004) * 2
    df_trade["5kFees"] = (5000 * 0.004) * 2
    df_trade["10kFees"] = (10000 * 0.0025) * 2
    df_trade["250Profit"] = (250 * df_trade["RawPercent"]) - df_trade["250Fees"]
    df_trade["5kProfit"] = (5000 * df_trade["RawPercent"]) - df_trade["5kFees"]
    df_trade["10k0Profit"] = (10000 * df_trade["RawPercent"]) - df_trade["10kFees"]

    return df_trade


def attachVWAPS(df, length=30):
    vwaps = df
    vwaps.set_index(pd.DatetimeIndex(vwaps["Date"]), inplace=True)
    vwaps["VWAP"] = df.ta.vwap(length=length)
    vwaps = vwaps.dropna(subset=["VWAP"])
    vwaps['VWAPD'] = vwaps['Close'] - vwaps['VWAP']
    return vwaps


def attachRVI(df):
    vol_df = df
    vol_df["RVI"] = df.ta.rvi()
    return vol_df.fillna(0)


# attach candle patterns
def attach_technicals(df_raw):
    df_raw = attachVWAPS(df_raw)
    df_raw = attachRVI(df_raw)
    # df_candles = df_raw.ta.cdl_pattern(name=["engulfing", "harami", "haramicross", "piercing", "darkcloudcover", "hammer", "invertedhammer"])
    macds = df_raw.ta.macd()
    rsi = df_raw.ta.rsi()
    stoch = df_raw.ta.stoch()
    bbands = df_raw.ta.bbands()
    obv = df_raw.ta.obv()
    ad = df_raw.ta.ad()
    mfi = df_raw.ta.mfi()
    willr = df_raw.ta.willr()
    # df_final = pd.concat([df_raw, df_candles, macds, rsi, stoch, bbands, obv, ad, mfi, willr], axis=1)
    df_final = pd.concat([df_raw, macds, rsi, stoch, bbands, obv, ad, mfi, willr], axis=1)
    df_final = df_final.fillna(0)
    return df_final


def append_price_dif(df):
    df['Target'] = df['Close'].shift(-1)
    return df


def convert_to_training_dataset(df):
    if (len(df) < 30):
        raise Exception("Training sets must be atleast 30 bars")
    # TODO FIX ME IM SORRY FAST HACK:
    if 'VWAPD' not in df:
        df = sort_date(df)
        target_df = attach_technicals(df)
        target_df = append_price_dif(target_df)
    else:
        target_df = df

    # The last row of the frame will have an NaN for Target. When training, we pop this.
    target_df.loc[target_df.index[-1], "Target"] = target_df.loc[target_df.index[-1], "Close"]

    features = target_df[gbl_all_columns]
    scaled_features = scale_data(features)

    X = pluck(scaled_features, gbl_all_columns, gbl_all_features)
    y = pluck(scaled_features, gbl_all_columns, gbl_target_column)

    return [scaled_features, X, y, features]


def scale_data(data):
    # Scale the data
    scaled_data = sc.fit_transform(data)
    return scaled_data

def consensus_percent(df):
  return mse_weighted_average(df, "Percent")


def consensus_prediction(df):
  return mse_weighted_average(df, "Predicted")

def mse_weighted_average(df, column):
    mse = df["MSE"]
    predictions = df[column]
    weights = np.array([1/(np.sqrt(mse)) for mse in mse])
    weights = weights / np.sum(weights)
    weighted_average = np.dot(predictions, weights)
    return weighted_average

def consensus_overall(df):
  return (1 - (df['Predicted'].std()/df['Predicted'].mean())) * 100