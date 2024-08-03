import os
import pandas as pd
import yfinance as yf
import time
import requests

def download_ticker_list(ticker_list, output_dir="./data/", interval="1d", tail=-1, head=-1):
    bad_tickers = []
    for ticker in ticker_list:
        time.sleep(0.25)
        print("ticker: ", ticker)
        try:
            ticker_obj = yf.download(tickers=ticker, interval=interval)
            df = pd.DataFrame(ticker_obj)
            if tail != -1:
                df = df.tail(tail)
            if head != -1:
                df = df.head(head)
            if len(df) == 0:
                bad_tickers.append(ticker)
            else:
                df.to_csv(os.path.join(output_dir, f"{ticker}.csv"))
        except (requests.exceptions.HTTPError, ValueError) as e:
            print(f"Failed to download {ticker} due to an HTTP or Value error: {e}")
            bad_tickers.append(ticker)
    return bad_tickers
