import joblib
from tensorflow import keras
from .data_utils import model_path

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


def load_all_models():
    training_filter = model_config["training_filter"]
    for name in model_config["day_bar_models"]:
        if (name in training_filter):  # skip training models
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
