import os
import pandas as pd
import decimal
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests
import json as js
from requests.exceptions import HTTPError

coin_base = False
ku_coin = True
model_version_token = "4"
models_loaded = False

COINBASE_REST_API = 'https://api.exchange.coinbase.com'
COINBASE_PRODUCTS = COINBASE_REST_API + '/products'
KUCOIN_REST_API = "https://api.kucoin.com"
KUCOIN_PRODUCTS = KUCOIN_REST_API + "/api/v1/market/allTickers"
KUCOIN_CANDLES = KUCOIN_REST_API + "/api/v1/market/candles"

data_path = '/Users/jmordetsky/machine_learning_finance/data'
model_path = "/Users/jmordetsky/machine_learning_finance/models"

def connect(url, params):
    response = requests.get(url, params)
    response.raise_for_status()
    return response

def download_ticker_list(ticker_list, tail=-1, head=-1):
    for ticker in ticker_list:
        time.sleep(0.25)
        print("ticker: ", ticker)
        try:
            tickerObj = yf.download(tickers=ticker, interval="1d")
            df = pd.DataFrame(tickerObj)
            if tail != -1:
                df = df.tail(tail)
            if head != -1:
                df = df.head(head)
            df.to_csv(f"./data/{ticker}.csv")
        except Exception as e:
            print("Failed to download:", ticker, " with ", e)
def get_all_products():
    if coin_base:
        return get_all_coinbase_products()

    if ku_coin:
        return get_all_kucoin_products()


def get_all_kucoin_products():
    response = connect(KUCOIN_PRODUCTS, {})
    products = js.loads(response.text)
    df_products = pd.DataFrame(products["data"]["ticker"])
    df_products = df_products.rename(columns={"symbol": "id"})
    return df_products


def get_all_coinbase_products():
    response = connect(COINBASE_PRODUCTS, {})
    response_text = response.text
    df_products = pd.read_json(response_text)
    return df_products


def save_dict(product_data, crypto=False):
    # Write each dataframe to disk
    for key, df in product_data.items():
        df.to_csv(data_path + "/dqn-preds-" + key + ".csv", index=False)

    # Convert the keys of the dictionary to a dataframe
    keys_df = pd.DataFrame({"keys": list(product_data.keys())})

    # Save the keys dataframe to disk as a CSV file
    if crypto:
        keys_df.to_csv(data_path + "/dqn-preds-keys-crypto.csv", index=False)
    else:
        keys_df.to_csv(data_path + "/dqn-preds-keys.csv", index=False)


def load_dict(crypto=False):
    if crypto:
        keys = pd.read_csv(data_path + "/dqn-preds-keys-crypto.csv")
    else:
        keys = pd.read_csv(data_path + "/dqn-preds-keys.csv")
    product_data = {}
    # Iterate over the keys
    for index, row in keys.iterrows():
        key = row["keys"]
        product_data[key] = pd.read_csv(data_path + "/dqn-preds-" + key + ".csv")
    return product_data


def sort_date(pric_df):
    pric_df = pric_df.sort_values(by=['Date'])
    return pric_df


def get_all_product_timeseries(cutoff=-1, length=320, limit=-1, raise_error=False):
    df_products = get_all_products()
    df_products = df_products[df_products.id.str.endswith('USDT')]
    df_products = df_products[~df_products.id.str.endswith('3L-USDT')]
    df_products = df_products[~df_products.id.str.endswith('3S-USDT')]
    df_products = df_products.sort_values(by="id")
    product_data = {}
    tries = 3
    i = 0

    for index, row in df_products.iterrows():
        if cutoff != -1 and i > cutoff:
            return product_data
        else:
            i += 1
        loop = True
        count = 0
        while (loop):
            try:
                time.sleep(0.25)
                print("fetch for: ", row.id)
                df_raw = sort_date(get_coin_data_frames(length, row.id))
                product_data[row.id] = df_raw
                loop = False
            except Exception as inst:
                if raise_error:
                    raise inst
                print("Error: ", inst)
                if "but only got" in str(inst) or "Illegal response" in str(inst):
                    loop = False
                else:
                    time.sleep(1)
                    count = count + 1
                if count > tries:
                    loop = False

    if limit != -1:
        print("Trimming to: ", limit)
        df_products["volValue"] = df_products["volValue"].apply(decimal.Decimal)
        df_products = df_products.sort_values(by=['volValue'], ascending=False)
        df_products = df_products.head(limit)
        assert (len(df_products) == limit)
        # kill keys not present anymore. this sucks because we spent time downloading
        # these, a better way might be to add symbols when we fail to download one
        # but otherwise we're short symbols and the dqn model wants a specific sized list
        # (which also sucks and needs to be fixed)
        # loop through the keys of the dictionary

        if len(product_data.keys()) > limit:
            print(f"Purging low volume to {limit}")
            for key in list(product_data.keys()):
                # check if the key is in the dataframe
                if key not in df_products['id'].values:
                    # if the key is not in the dataframe, remove it from the dictionary
                    del product_data[key]
                if len(product_data.keys()) == limit:
                    return product_data
    return product_data


def get_all_stock_timerseries_for_csv(csv_name, bars, cutoff=-1, raise_error=False):
    product_data = {}
    tickers_df = pd.read_csv(data_path + "/" + csv_name)
    total = len(tickers_df)
    i = 0
    for ticker in tickers_df.iloc[:, 0]:
        print("loading ticker: ", ticker, " count:", i, "of:", total)
        if cutoff != -1 and i > cutoff:
            return product_data
        else:
            i += 1
        try:
            tickerObj = yf.download(tickers=ticker, interval="1d")
            df = pd.DataFrame(tickerObj)
            if len(df) == 0 or len(df) < bars:
                continue
            product_data[ticker] = df.tail(bars)
            time.sleep(0.25)
        except Exception as inst:
            if raise_error:
                raise inst

            print("Error: ", inst)
    return product_data


def coinbase_json_to_df(delta, product, granularity='86400'):
    start_date = (datetime.today() - timedelta(seconds=delta * int(granularity))).isoformat()
    end_date = datetime.now().isoformat()
    # Please refer to the coinbase documentation on the expected parameters
    params = {'start': start_date, 'end': end_date, 'granularity': granularity}
    response = connect(COINBASE_PRODUCTS + '/' + product + '/candles', params)
    response_text = response.text
    df_history = pd.read_json(response_text)
    # Add column names in line with the Coinbase Pro documentation
    df_history.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
    df_history['time'] = [datetime.fromtimestamp(x) for x in df_history['time']]
    return df_history


def ku_coin_json_to_df(delta, product, granularity='86400'):
    granularity = int(granularity)
    start_date = (datetime.today() - timedelta(seconds=delta * granularity))
    end_date = datetime.now()

    # Please refer to the kucoin documentation on the expected parameters
    params = {'startAt': int(start_date.timestamp()), 'endAt': int(end_date.timestamp()),
              'type': gran_to_string(granularity), 'symbol': product}
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
    if (got_bars < delta - 1):
        raise Exception("Requested:" + str(delta) + " bars " + " but only got:" + str(got_bars))

    df_history.columns = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount']
    df_history['time'] = [datetime.fromtimestamp(int(x)) for x in df_history['time']]
    df_history['open'] = [float(x) for x in df_history['open']]
    df_history['close'] = [float(x) for x in df_history['close']]
    df_history['high'] = [float(x) for x in df_history['high']]
    df_history['low'] = [float(x) for x in df_history['low']]
    df_history['volume'] = [float(x) for x in df_history['volume']]
    df_history['amount'] = [float(x) for x in df_history['amount']]
    return df_history


def gran_to_string(granularity):
    # todo implement this actually
    if granularity == 86400:
        return "1day"
    if granularity == 900:
        return "15min"
    raise Exception("Joe didn't implement a proper granularity to string. Lazy, lazy.")


def get_all_products():
    if coin_base:
        return get_all_coinbase_products()

    if ku_coin:
        return get_all_kucoin_products()


def get_all_coinbase_products():
    response = connect(COINBASE_PRODUCTS, {})
    response_text = response.text
    df_products = pd.read_json(response_text)
    return df_products


def get_coin_data_frames(time, product, granularity='86400', feature_set=["Close", "Volume"]):
    if coin_base:
        df_raw = coinbase_json_to_df(time, product, granularity)
    else:
        df_raw = ku_coin_json_to_df(time, product, granularity)

    if len(df_raw.index) == 0:
        raise Exception("No data for " + product)

    df_raw = df_raw.rename(
        columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df_raw


def download_stocks(total):
    # We don't want to train against crypto
    return get_all_stock_timerseries_for_csv("training_tickers3.csv", 3500, total)


LENGTH_OF_STOCK_TRAINGING_DATA = 145  # I might need to fix this, but the model is tied to the number of symbols we trained on


def download_crypto():
    return get_all_product_timeseries(-1, 180, LENGTH_OF_STOCK_TRAINGING_DATA)


def get_data_for_training(num):
    from_disk = False
    use_crypto = False

    if from_disk:
        product_data = load_dict(use_crypto)
        first_key = list(product_data.keys())[0]
        length = len(product_data[first_key])
    elif use_crypto:
        product_data = download_crypto()
        save_dict(product_data, True)
    else:
        product_data = download_stocks(num)
        save_dict(product_data)

    min_len = min(len(df) for df in product_data.values())
    print("symbols:", len(product_data.keys()))
    print("min length:", min_len)
    for name, df in product_data.items():
        product_data[name] = df.head(min_len)

    product_data = dict(sorted(product_data.items(), reverse=True))

    return product_data
