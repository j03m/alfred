import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, confusion_matrix, classification_report, accuracy_score, f1_score
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import requests
from requests.exceptions import HTTPError
import json as js
from datetime import datetime, timedelta
import time
from os.path import exists
from decimal import *
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from sklearn.model_selection import KFold, ParameterGrid
from keras.layers import Input, LSTM, Attention, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error


pd.options.display.float_format = '{:f}'.format
np.set_printoptions(formatter={'float': '{:f}'.format})

# Function to plot interactive plots using Plotly Express
sc = MinMaxScaler()
num_features = 2 #3
candle_features = 5 #6
coin_base = False
ku_coin = True
load_models = True
extra_training = False
COINBASE_REST_API = 'https://api.pro.coinbase.com'
COINBASE_PRODUCTS = COINBASE_REST_API+'/products'
KUCOIN_REST_API = "https://api.kucoin.com"
KUCOIN_PRODUCTS = KUCOIN_REST_API+ "/api/v1/market/allTickers"
KUCOIN_CANDLES = KUCOIN_REST_API+ "/api/v1/market/candles"

data_path = '/content/drive/My Drive/ml-trde-data'
model_path = "/content/drive/My Drive/ml-trde-models"

def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()

def get_single_stock(price_df, vol_df, name):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name], 'Volume': vol_df[name]})

def scale_data(data):
  # Scale the data
  scaled_data = sc.fit_transform(data)
  return scaled_data

def sort_date(pric_df):
  pric_df = pric_df.sort_values(by = ['Date'])
  return pric_df

def append_price_dif(df):
  df['Target'] = df['Close'].shift(-1)
  df['Diff'] = df['Target'] - df['Close']
  df = df[:-1]
  return df

def append_price_dif_(df):
  df['Target'] = df['Close'].shift(-1)
  df['Diff'] = df['Target'] - df['Close']
  return df

def append_15d_slope(df):
  df['15Close'] = df['Close'].shift(15)
  df['15Date'] = df['Date'].shift(15)
  df['Trend'] = (df['Close'] - df['15Close']) / 15
  df = df[15:]
  return df

def show_plot(data, title):
  plt.figure(figsize = (13, 5))
  plt.plot(data, linewidth = 3)
  plt.title(title)
  plt.grid()

def build_model(features, outcomes):
  # Create the model
  inputs = keras.layers.Input(shape=(features,outcomes))
  x = keras.layers.LSTM(150, return_sequences= True)(inputs)
  x = keras.layers.Dropout(0.3)(x)
  x = keras.layers.LSTM(150, return_sequences=True)(x)
  x = keras.layers.Dropout(0.3)(x)
  x = keras.layers.LSTM(150)(x)
  outputs = keras.layers.Dense(1, activation='linear')(x)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer='adam', loss="mse")
  return model

def build_attention_model(features, outcomes):
  # Create the model
  inputs = keras.layers.Input(shape=(features,outcomes))
  x = keras.layers.LSTM(150, return_sequences= True)(inputs)
  x = keras.layers.Dropout(0.3)(x)
  x = keras.layers.LSTM(150, return_sequences=True)(x)
  x = keras.layers.Dropout(0.3)(x)
  x = keras.layers.LSTM(150)(x)
  attention_layer = Attention()([x, x])
  outputs = keras.layers.Dense(1, activation='linear')(x)
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer='adam', loss="mse")
  return model

def connect(url, params):
  response = requests.get(url,params)
  response.raise_for_status()
  return response

def coinbase_json_to_df(delta, product, granularity='86400'):
  start_date = (datetime.today() - timedelta(seconds=delta*int(granularity))).isoformat()
  end_date = datetime.now().isoformat()
  # Please refer to the coinbase documentation on the expected parameters
  params = {'start':start_date, 'end':end_date, 'granularity':granularity}
  response = connect(COINBASE_PRODUCTS+'/' + product + '/candles', params)
  response_text = response.text
  df_history = pd.read_json(response_text)
  # Add column names in line with the Coinbase Pro documentation
  df_history.columns = ['time','low','high','open','close','volume']
  df_history['time'] = [datetime.fromtimestamp(x) for x in df_history['time']]
  return df_history

def ku_coin_json_to_df(delta, product, granularity='86400'):
  granularity = int(granularity)
  start_date = (datetime.today() - timedelta(seconds=delta*granularity))
  end_date = datetime.now()

  # Please refer to the kucoin documentation on the expected parameters
  params = {'startAt':int(start_date.timestamp()), 'endAt':int(end_date.timestamp()), 'type':gran_to_string(granularity), 'symbol':product}
  response = connect(KUCOIN_CANDLES, params)
  response_text = response.text
  response_data = js.loads(response_text);
  if (response_data["code"] != "200000"):
    raise Exception("Illegal response: " + response_text)

  df_history = pd.DataFrame(response_data["data"])

  # kucoin is weird in that they don't have candles for everything. IF we don't have the requested
  # number of bars here, it throws off the whole algo. I don't want to try and project so we
  # just won't trade those instruments
  got_bars = len(df_history)
  if ( got_bars < delta-1):
    raise Exception("Requested:" + str(delta) + " bars " + " but only got:" + str(got_bars))

  df_history.columns = ['time','open','close','high','low','volume', 'amount']
  df_history['time'] = [datetime.fromtimestamp(int(x)) for x in df_history['time']]
  df_history['open'] = [float(x) for x in df_history['open']]
  df_history['close'] = [float(x) for x in df_history['close']]
  df_history['high'] = [float(x) for x in df_history['high']]
  df_history['low'] = [float(x) for x in df_history['low']]
  df_history['low'] = [float(x) for x in df_history['low']]
  df_history['volume'] = [float(x) for x in df_history['volume']]
  df_history['amount'] = [float(x) for x in df_history['amount']]
  return df_history

def gran_to_string(granularity):
  #todo implement this actually
  if granularity == 86400:
    return "1day"
  if granularity == 900:
    return "15min"
  raise Exception("Joe didn't implement a proper granularity to string. Lazy, lazy.")

#def get_coin_data_frames(time, product, granularity='86400', feature_set = ["Close", "Volume", "Trend"]):
def get_coin_data_frames(time, product, granularity='86400', feature_set = ["Close", "Volume"]):
  if coin_base:
    df_raw = coinbase_json_to_df(time, product, granularity)
  else:
    df_raw = ku_coin_json_to_df(time, product, granularity)

  df_btc_history = df_raw
  if len(df_btc_history.index) == 0:
    print("No data for ", product)

  df_btc_history = df_btc_history.rename(columns={"time":"Date", "open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"})
  df_btc_history = sort_date(df_btc_history)
  df_btc_history = append_price_dif_(df_btc_history)
  #df_btc_history = append_15d_slope(df_btc_history)
  df_btc_features = df_btc_history[feature_set]
  df_history_scaled = sc.fit_transform(df_btc_features)
  return [df_btc_history, df_btc_features, df_history_scaled, df_raw]

def build_profit_estimate(predicted, df_btc_history):
  df_predicted_chart = pd.DataFrame();
  df_predicted_chart["Date"] = df_btc_history["Date"]
  df_predicted_chart["Predicted"] = predicted
  df_predicted_chart["Predicted-Target"] = df_predicted_chart["Predicted"].shift(-1)
  df_predicted_chart["Predicted-Diff"] = df_predicted_chart["Predicted-Target"] - df_predicted_chart["Predicted"]
  df_predicted_chart["Should-Trade"] = np.where(df_predicted_chart["Predicted-Diff"] > 0, True, False)
  df_predicted_chart["RealDiff"] = df_btc_history["Diff"]
  df_predicted_chart["Percent"] = df_predicted_chart["RealDiff"] / df_btc_history["Close"]
  df_predicted_chart["Profit"] = np.where(df_predicted_chart["Should-Trade"] > 0, df_predicted_chart["Percent"] * budget, 0)
  profit = df_predicted_chart["Profit"].sum()
  return [df_predicted_chart, profit]

def debug_prediction_frame(predicted, df_history, df_history_scaled):
  df_predicted_chart = pd.DataFrame();
  df_predicted_chart["Date"] = df_history["Date"]
  df_predicted_chart["Predicted"] = predicted
  df_predicted_chart["Original"] = df_history_scaled[:,0]
  #Trend
  #df_predicted_chart["Original-Target"] = df_history_scaled[:,2]
  df_predicted_chart["Original-Target"] = df_history_scaled[:,1]
  df_predicted_chart["Target-Date"] = df_predicted_chart["Date"].shift(-1)
  df_predicted_chart["Predicted-Diff"] = df_predicted_chart["Predicted"] - df_predicted_chart["Original"]
  df_predicted_chart["Actual-Diff"] = df_predicted_chart["Original-Target"] - df_predicted_chart["Original"]
  df_predicted_chart["Should-Trade"] = np.where(df_predicted_chart["Predicted-Diff"] > 0, True, False)
  df_predicted_chart["Close"] = df_history["Close"]
  df_predicted_chart["Target"] = df_history["Target"]
  df_predicted_chart["RealDiff"] = df_history["Diff"]
  df_predicted_chart["Percent"] = df_predicted_chart["RealDiff"] / df_predicted_chart["Close"]
  df_predicted_chart["Profit"] = np.where(df_predicted_chart["Should-Trade"] > 0, df_predicted_chart["Percent"] * budget, 0)
  return df_predicted_chart

def get_all_products():
  if coin_base:
    return get_all_coinbase_products()

  if ku_coin:
    return get_all_kucoin_products()

def get_all_kucoin_products():
  response = connect(KUCOIN_PRODUCTS, {})
  products = js.loads(response.text)
  df_products = pd.DataFrame(products["data"]["ticker"])
  df_products = df_products.rename(columns={"symbol":"id"})
  return df_products

def get_all_coinbase_products():
  response = connect(COINBASE_PRODUCTS, {})
  response_text = response.text
  df_products = pd.read_json(response_text)
  return df_products

def predict_trade(model, product, bars, npa_scaled=[]):

  if len(npa_scaled) == 0:
    print("downloading...")
    [df_full, df_features, npa_scaled, df_raw] = get_coin_data_frames(bars, product)

  predicted = model.predict(npa_scaled).flatten()

  #convert to data frames that have the correct shape for being unscaled
  #df_scaled = pd.DataFrame(npa_scaled, columns = ["Close", "Volume", "Trend"])
  df_scaled = pd.DataFrame(npa_scaled, columns = ["Close", "Volume"])

  # I want to believe that scaling happens on a per column basis, we only care about
  # price here so we will dummy out volume and trend and use the scaler on it
  # this kinda sucks, if we add features we'll need to add them here for unscaling
  df_temp = pd.DataFrame(predicted, columns = ["Close"])
  df_temp["Volume"] = 0
  #df_temp["Trend"] = 0

  # unscale them both
  #df_temp = pd.DataFrame(sc.inverse_transform(df_temp), columns = ["Close", "Volume", "Trend"])
  #df_trade = pd.DataFrame(sc.inverse_transform(df_scaled), columns = ["Close", "Volume", "Trend"])
  df_temp = pd.DataFrame(sc.inverse_transform(df_temp), columns = ["Close", "Volume"])
  df_trade = pd.DataFrame(sc.inverse_transform(df_scaled), columns = ["Close", "Volume"])


  # add predicted
  df_trade["Predicted"] = df_temp["Close"]
  df_trade = df_trade.tail(1)

  # add the product, derive a move and percent
  df_trade["Product"] = row.id;
  df_trade["Move"] = df_trade["Predicted"] - df_trade["Close"]
  df_trade["Percent"] = (df_trade["Move"] / df_trade["Close"]) * 100
  df_trade["RawPercent"] = df_trade["Move"] / df_trade["Close"]
  df_trade["250Fees"] = (250 * 0.004) * 2
  df_trade["5kFees"] = (5000 * 0.004) * 2
  df_trade["10kFees"] = (10000 * 0.0025) * 2
  df_trade["250Profit"] = (250 * df_trade["RawPercent"]) - df_trade["250Fees"]
  df_trade["5kProfit"] = (5000 * df_trade["RawPercent"]) - df_trade["5kFees"]
  df_trade["10k0Profit"] = (10000 * df_trade["RawPercent"]) - df_trade["10kFees"]
  return df_trade

def get_yf_training_set_for(df):
  target_df = append_price_dif(df)
  #target_df = append_15d_slope(target_df)
  #features = target_df[["Open", "High", "Low", "Close", "Volume", "Trend", "Target"]]
  features = target_df[["Open", "High", "Low", "Close", "Volume", "Target"]]
  scaled_features = scale_data(features)
  return extract_training(scaled_features, len(target_df),len(features.columns)-1)

def extract_training(scaled_features, length, num_features):
  X = []
  y = []

  for i in range(0, length):
    X.append(scaled_features [i][0:num_features])
    y.append(scaled_features [i][num_features])
  X = np.asarray(X)
  y = np.asarray(y)
  return [scaled_features, X, y]

def get_training_set_for(ticker):
  target_df = get_single_stock(all_stocks_price_df, all_stocks_vol_df, ticker)
  target_df = append_price_dif(target_df)
  #target_df = append_15d_slope(target_df)
  #features = target_df[["Close", "Volume", "Trend", "Target"]]
  features = target_df[["Close", "Volume", "Target"]]
  scaled_features = scale_data(features)
  return extract_training(scaled_features, len(target_df), len(features.columns)-1)

def train_model(model, X, y):

  # One day we might need test, but for now we don't we can use another
  # time series, we have so many
  # Split the data
  #split = int(0.7 * len(X))
  #X_train = X[:split]
  #y_train = y[:split]
  #X_test = X[split:]
  #y_test = y[split:]

  # Reshape the 1D arrays to 3D arrays to feed in the model
  X_train = np.reshape(X, (X.shape[0], X.shape[1], 1))

  # Create an early stopping callback
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)

  history = model.fit(
      X_train, y,
      epochs = 20,
      batch_size = 32,
      validation_split = 0.2,
      callbacks=[early_stopping]
  )
  return [model, history]

def get_group_bars(df):
  df = pd.DataFrame(sc.fit_transform(df[["Close", "Volume"]]), columns=["Close","Volume"])
  # Split into input sequences and target values
  n_steps = 4*4  # 4 hours of data at 15 minute intervals
  X = []
  Y = []
  for i in range(0, len(df), n_steps):
    df_group = df.iloc[i:i+n_steps]
    if len(df_group) != n_steps:
      continue
    X.append(np.array(df_group.values))
    Y.append(df_group.values[-1,0])

  # Convert the lists to NumPy arrays
  X = np.array(X)
  Y = np.array(Y)
  return [X, Y]

def build_15m_model():
  # the 15 min bar model
  # Build the model
  group_size = 4*4
  features = 2
  model15 = build_model(group_size, features)

  # Compile the model
  model15.compile(loss='mean_squared_error', optimizer='adam')

  # Train it
  tickers = ["SPY", "IBM", "TSLA", "CAT", "XOM", "B", "F", "AAPL", "AMZN"]

  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  for ticker in tickers:
    file_path = data_path + "/" + ticker + "-15.csv"
    df = sort_date(pd.read_csv(file_path).rename(columns={"Datetime":"Date"}))
    df['Date'] = pd.to_datetime(df['Date'])
    [X,Y] = get_group_bars(df)
    model15.fit(X, Y,
      epochs = 20,
      batch_size = 32,
      validation_split = 0.2,callbacks=[early_stopping])
  return model15

def fetch_and_predict_short_term(model, product):
  if coin_base:
    df_raw = coinbase_json_to_df(16, product, 900)
  else:
    df_raw = ku_coin_json_to_df(16, product, 900)
  df_raw = df_raw.rename(columns={"close":"Close", "volume": "Volume"})
  [X, Y] = get_group_bars(df_raw[["Close", "Volume"]])
  predicted = model.predict(X)
  df_pred = pd.DataFrame(predicted, columns = ["Close"])
  df_pred["Volume"] = 0
  return [predicted.flatten()[0], sc.inverse_transform(df_pred).flatten()[0]]




#pull training data
all_stocks_price_df = sort_date(pd.read_csv(data_path+'/stock.csv'))
all_stocks_vol_df = sort_date(pd.read_csv(data_path+"/stock_volume.csv"))
spy_df = sort_date(pd.read_csv(data_path+'/SPY.csv'))
cat_df = sort_date(pd.read_csv(data_path+'/CAT.csv'))
f_df = sort_date(pd.read_csv(data_path+'/F.csv'))
xom_df = sort_date(pd.read_csv(data_path+'/XOM.csv'))
ibm_df = sort_date(pd.read_csv(data_path+'/IBM.csv'))
spy_15_df = sort_date(pd.read_csv(data_path+'/SPY-15.csv').rename(columns={"Datetime":"Date"}))