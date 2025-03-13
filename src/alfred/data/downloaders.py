import os
import pandas as pd
import time
import ssl
from datetime import datetime
import binascii
from io import StringIO

from .news_db import NewsDb
from .openai_query import OpenAiQuery
from fuzzywuzzy import process
import re

from fake_useragent import UserAgent


ssl.create_default_https_context = ssl._create_unverified_context

def download_ticker_list(ticker_list, output_dir="./data/", interval="1d", tail=-1, head=-1):
    bad_tickers = []
    downloader = AlphaDownloader()
    for ticker in ticker_list:
        time.sleep(0.25)
        print("ticker: ", ticker)
        data_file = os.path.join(output_dir, f"{ticker}.csv")
        if os.path.exists(data_file):
            try:
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['Date'])
                today = datetime.today().strftime('%m-%d-%Y')
                latest_date = df['Date'].max().strftime('%m-%d-%Y')
                if latest_date == today:
                    print(f"{ticker} prices are up2date")
                    continue
            except:
                print(f"Bad on disk file time in {ticker} file. Re-downloading")
        try:
            df = downloader.prices(ticker, interval=interval)
            if len(df) == 0:
                print(f"no data for {ticker}")
                bad_tickers.append(ticker)
                continue

            df.to_csv(data_file)
        except (requests.exceptions.HTTPError, ValueError) as e:
            print(f"Failed to download {ticker} due to an HTTP or Value error: {e}")
            bad_tickers.append(ticker)
        except Exception as e:
            print(f"General exception {e}")
            bad_tickers.append(ticker)

    return bad_tickers


import json
import requests
from time import sleep


class OpenFigiDownloader:
    def __init__(self, key_file="./keys/openfigi.txt", rate_limit=0.25, cache_file="./data/cusip_cache.json"):
        with open(key_file, 'r') as file:
            self.api_key = file.readline().strip()

        self.rate_limit = rate_limit
        self.cache_file = cache_file
        if os.path.exists(cache_file):
            with open(cache_file, "r") as file:
                self.cache = json.load(file)
        else:
            self.cache = {}

    def fetch(self, url, data):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                sleep(self.rate_limit)
                headers = {
                    "Content-Type": "application/json",
                    "X-OPENFIGI-APIKEY": self.api_key
                }
                response = requests.post(url, json=data, headers=headers, verify=False)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as err:
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = int(response.headers.get("Retry-After", 0))
                    if retry_after == 0:
                        retry_after = self.rate_limit * attempt * 2
                    print(f"Error {response.status_code}. Retrying after {retry_after} seconds...")
                    sleep(retry_after)
                else:
                    raise
            except ConnectionError as err:
                retry_after = self.rate_limit * attempt * 60
                print(f"ConnectionError. Retrying after {retry_after} seconds...")
                sleep(retry_after)
        else:
            raise RuntimeError(f"Failed to fetch data after {max_retries} attempts")

    def get_data_for_cusip(self, cusip, exchange="US"):
        """
        Fetches data from OpenFIGI for the given CUSIP and exchange code.
        """
        mapping_request = [
            {"idType": "ID_CUSIP", "idValue": cusip, "exchCode": exchange},
        ]
        result = self.fetch("https://api.openfigi.com/v3/mapping", mapping_request)
        if result[0].get("warning", None) is not None:
            return None
        return result[0]["data"][0]

    # caches
    def get_ticker_for_cusip(self, cusip):
        if cusip in self.cache:
            return self.cache[cusip]

        data = self.get_data_for_cusip(cusip)
        if data is not None and "ticker" in data:
            self.cache[cusip] = data["ticker"]
            return data["ticker"]
        else:
            self.cache[cusip] = "Unknown"
            return None

    def dump_cache(self):
        with open(self.cache_file, "w") as file:
            json.dump(self.cache, file)


class AlphaDownloader:
    def __init__(self, key_file='./keys/alpha.txt', rate_limit=0.2):
        # Read the API key from the specified file
        with open(key_file, 'r') as file:
            self.api_key = file.readline().strip()
        self.rate_limit = rate_limit

    def get(self, url):
        sleep(self.rate_limit)
        return requests.get(url, verify=False)

    def prices(self, symbol, interval="1d"):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&interval={interval}&apikey={self.api_key}&datatype=csv'
        response = self.get(url)
        df = pd.read_csv(StringIO(response.text))
        df.rename(columns={"timestamp": "Date", "open":"Open", "high":"High", "low":"Low", "close":"Close", "volume": "Volume"}, inplace=True)
        return df

    def earnings(self, symbol):
        # Construct the URL for the API request
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={self.api_key}'
        # Send the request and get the JSON response
        response = self.get(url)
        data = response.json()

        # Convert the earnings data to a DataFrame
        quarterly_earnings = data.get('quarterlyEarnings', [])
        if len(quarterly_earnings) == 0:
            return None
        quarterly_earnings = pd.DataFrame(quarterly_earnings)
        quarterly_earnings = quarterly_earnings.drop(columns=['reportTime'])
        quarterly_earnings = quarterly_earnings.drop(columns=['fiscalDateEnding'])
        quarterly_earnings = quarterly_earnings.rename(columns={'reportedDate': 'Date'})

        quarterly_earnings = quarterly_earnings.fillna(0)

        return quarterly_earnings

    def earnings_to_csv(self, symbol, annual_csv_file='annual_earnings.csv',
                        quarterly_csv_file='quarterly_earnings.csv'):
        # Fetch the earnings data
        annual_earnings, quarterly_earnings = self.earnings(symbol)

        # Save the earnings data to CSV files
        annual_earnings.to_csv(annual_csv_file, index=False)
        quarterly_earnings.to_csv(quarterly_csv_file, index=False)

    def margins(self, symbol):
        # Construct the URL for the API request
        url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={self.api_key}'
        response = self.get(url)
        data = response.json()

        # Extract relevant margin data
        quarterly_reports = data.get('quarterlyReports', [])

        # Create lists to store the extracted data
        dates = []
        gross_margins = []
        operating_margins = []
        net_profit_margins = []

        # scrub - javascript makes the world bad
        for report in quarterly_reports:
            for key, value in report.items():
                if value in ['None', 'N/A', '0']:
                    report[key] = 0
                else:
                    try:
                        if key not in ["reportedCurrency", "fiscalDateEnding"]:
                            report[key] = float(value)
                    except ValueError:
                        print(f"Don't convert {key} / {value} to float")

        for report in quarterly_reports:
            dates.append(report['fiscalDateEnding'])
            report['totalRevenue'] = report['grossProfit'] + report['costofGoodsAndServicesSold']


            total_revenue = report['totalRevenue']
            gross_margin = report['grossProfit'] / total_revenue if total_revenue != 0 else 0

            gross_margins.append(gross_margin)

            # Calculate Operating Margin
            operating_margin = report['operatingIncome'] / total_revenue if total_revenue != 0 else 0
            operating_margins.append(operating_margin)

            # Calculate Net Profit Margin
            net_profit_margin = report['netIncome'] / total_revenue if total_revenue != 0 else 0
            net_profit_margins.append(net_profit_margin)

        # Create a DataFrame with the margin data
        df_margins = pd.DataFrame({
            'Date': dates,
            'Margin_Gross': gross_margins,
            'Margin_Operating': operating_margins,
            'Margin_Net_Profit': net_profit_margins
        })

        return df_margins

    def treasury_yields(self, maturities=['10year', '5year', '3year', '2year']):
        # Initialize a DataFrame to hold all the yield data
        yield_data = pd.DataFrame()

        for maturity in maturities:
            # Construct the URL for the Treasury Yield API request
            url = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity={maturity}&apikey={self.api_key}'
            response = self.get(url)
            data = response.json()

            # Extract the data and create a DataFrame
            maturity_df = pd.DataFrame(data['data'])
            maturity_df = maturity_df.rename(columns={'value': maturity})
            maturity_df['date'] = pd.to_datetime(maturity_df['date'])
            maturity_df.set_index('date', inplace=True)

            # Merge with the existing data
            if yield_data.empty:
                yield_data = maturity_df
            else:
                yield_data = yield_data.join(maturity_df, how='outer')

        yield_data = yield_data.sort_index()

        return yield_data

    def treasury_yields_to_csv(self, maturities=['10year', '5year', '3year', '2year'], csv_file='treasury_yields.csv'):
        # Fetch the treasury yield data
        yield_data = self.treasury_yields(maturities)
        yield_data.fillna(0)
        # Save the yield data to a CSV file
        yield_data.to_csv(csv_file)

    def balance_sheet(self, symbol):
        url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={self.api_key}'
        response = self.get(url)
        data = response.json()

        # Convert balance sheet data to DataFrame
        quarterly_balance_sheet = pd.DataFrame(data['quarterlyReports'])
        quarterly_balance_sheet['fiscalDateEnding'] = pd.to_datetime(quarterly_balance_sheet['fiscalDateEnding'])
        quarterly_balance_sheet.set_index('fiscalDateEnding', inplace=True)

        return quarterly_balance_sheet

    def cash_flow(self, symbol):
        url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={self.api_key}'
        response = self.get(url)
        data = response.json()

        # Convert cash flow data to DataFrame
        quarterly_cash_flow = pd.DataFrame(data['quarterlyReports'])
        quarterly_cash_flow['fiscalDateEnding'] = pd.to_datetime(quarterly_cash_flow['fiscalDateEnding'])
        quarterly_cash_flow.set_index('fiscalDateEnding', inplace=True)

        return quarterly_cash_flow

    def historical_prices(self, symbol):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={self.api_key}'
        response = self.get(url)
        data = response.json()

        # Convert the time series data to a DataFrame
        prices_df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.rename(columns={'5. adjusted close': 'close'})

        return prices_df[['close']].astype(float)

    def news_sentiment(self, symbol, time_from, time_to):
        # transform dates for alphavantage
        formatted_from = time_from.strftime('%Y%m%dT%H%M')
        formatted_to = time_to.strftime('%Y%m%dT%H%M')
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&time_from={formatted_from}&time_to={formatted_to}&sort=RELEVANCE&limit=1000&apikey={self.api_key}"
        response = self.get(url)
        return response.json()

    def news_sentiment_for_window_and_symbol(self, symbol, time_from, time_to):
        # Fetch the news sentiment data
        data = self.news_sentiment(symbol, time_from, time_to)

        # Prepare the output list
        filtered_articles = []

        # Iterate through each article in the feed
        if data.get("items", 0) == 0:
            print(f"No articles for {symbol}")

        for article in data.get("feed", []):
            # Convert time_published to datetime for comparison
            time_published_str = article.get("time_published", None)
            if time_published_str is not None:
                time_published = datetime.strptime(time_published_str, "%Y%m%dT%H%M%S").date()

            # Check if the article's published time is within the specified range
            if time_published_str is None or time_from <= time_published <= time_to:
                # Iterate through each ticker sentiment in the article
                for ticker_sentiment in article.get("ticker_sentiment", []):
                    # Check if the ticker matches the specified symbol
                    if ticker_sentiment["ticker"] == symbol:
                        # Extract and format the desired information
                        article_info = {
                            "title": article["title"],
                            "url": article["url"],
                            "summary": article["summary"],
                            "time_published": time_published,
                            "relevance_score": float(ticker_sentiment["relevance_score"]),
                            "ticker_sentiment_score": float(ticker_sentiment["ticker_sentiment_score"]),
                            "ticker_sentiment_label": ticker_sentiment["ticker_sentiment_label"]
                        }
                        # Add the formatted information to the results list
                        filtered_articles.append(article_info)
                        break  # Stop further checking once the relevant ticker is found

        return filtered_articles

    def fetch_insider_transactions(self, ticker):
        url = f"https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol={ticker}&apikey={self.api_key}"
        return self.get(url).json()

    def get_normalized_insider_transactions(self, ticker):
        response = self.fetch_insider_transactions(ticker)
        transactions = response.get("data", [])
        security_weights = {
            "common": 1.0,
            "preferred": 0.8,
            "phantom": 0.5,
            "class": 0.55,
            "option": 0.6,
            "options": 0.6,
            "series": 0.75,
            "forward purchase contract": 0.6,
            "ltip": 0.65,
            "restricted": 0.7,
            "subscription": 0.4,
            "contingent": 0.3,
            "remainder": 0.2,
            "stock appreciation right": 0.5,
            "sar": 0.5,
            "dividend": 0.3,
            "award": 0.7,
            "obligation": 0.4,
            "right": 0.6,
            "ordinary": 1.0,
            "deferred": 0.5,
            "performance": 0.7,
            "warrants": 0.7,
            "debentures": 0.5,
            "units": 0.5,
            "rsu": 0.7,
            "psu": 0.7,
            "rsus": 0.7,
            "psus": 0.7,
            "dsu": 0.7,
            "dsus": 0.7,
            "convertible": 0.5
        }

        words = list(security_weights.keys())

        def get_security_score(security_type):
            corrected_text = security_type.lower()
            corrected_text = re.sub(r"\W+", " ", corrected_text).strip()
            tokens = corrected_text.split()

            # any of our hits in the string?
            for word in words:
                if word in corrected_text:
                    return security_weights[word]

            # any of their tokens in our strings?
            for word in tokens:
                if word in words:
                    return security_weights[word]

                # anything mispelled but maybe matching?
                matched, score = process.extractOne(word, words)
                threshold = 80
                if score >= threshold:
                    return security_weights[matched]

            print("unknown security type:", security_type)
            return 0.1  # Default weight if no match is found

        data = []
        share_sizes = [1]
        for t in transactions:
            if t["shares"] == "":
                t["shares"] = 1
            t["shares"] = float(t["shares"])
            share_sizes.append(t["shares"])

        biggest_distro = max(share_sizes)
        for event in transactions:
            # Parse and extract details
            try:
                date = pd.to_datetime(event["transaction_date"])
            except:
                continue

            action_type = event["acquisition_or_disposal"]
            security_type = event["security_type"]
            shares = event["shares"]

            # Calculate scores based on security type, shares, and executive rank

            security_score = get_security_score(security_type)

            normalized_shares = float(shares) / biggest_distro  # Normalize shares based on max shares in data

            # Total score combines security, rank, and normalized shares scores
            score = security_score * normalized_shares

            # Append data for the DataFrame
            if action_type == "A":
                data.append({"Date": date, "insider_acquisition": score, "insider_disposal": 0})
            else:
                data.append({"Date": date, "insider_acquisition": 0, "insider_disposal": score})

        if len(data) == 0:
            return None

        # Create DataFrame, set index, and group by date to aggregate scores
        df = pd.DataFrame(data)
        df = df.groupby("Date").sum()
        return df

    def get_ticker_from_name(self, company_name):
        """Retrieve ticker using Alpha Vantage."""
        url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        if "bestMatches" in data and len(data["bestMatches"]) > 0:
            return data["bestMatches"][0]["1. symbol"]
        return None


# Example usage:
# downloader = AlphaDownloader(key_file='./keys/alpha.txt')
# treasury_df = downloader.treasury_yields()
# downloader.treasury_yields_to_csv(csv_file='treasury_yields.csv')


class ArticleDownloader:
    def __init__(self, rate_limit=0.5):
        self.api = AlphaDownloader()
        self.openai = OpenAiQuery()
        self.news_db = NewsDb()
        self.rate_limit = rate_limit
        self.ua = UserAgent()

    def get(self, url):
        sleep(self.rate_limit)
        return requests.get(url, verify=False)

    def fetch_article_body(self, url):
        """Fetch the article body from the given URL. Note we don't rate limit here since we're not hitting AA"""
        headers = {
            'User-Agent': self.ua.random
        }
        response = requests.get(url, headers=headers, timeout=1)
        response.raise_for_status()  # Ensure we handle errors
        return response.text

    def generate_article_id(self, url):
        """Generate a unique article ID based on the CRC32 hash of the URL."""
        return format(binascii.crc32(url.encode()), '08x')

    def cache_article_metadata(self, ticker, time_from, time_to):
        # Fetch articles using `news_sentiment_for_window`
        articles = self.api.news_sentiment_for_window_and_symbol(ticker, time_from, time_to)

        for i, article in enumerate(articles):
            # Convert time_published to a date string for directory naming
            published = article.get('time_published', None)
            if published is None:
                published = datetime.today().date().strftime('%Y-%m-%d')
            else:
                published = datetime.strftime(published, '%Y-%m-%d')
            try:
                if not self.news_db.has_article(ticker, article["url"]):
                    print(f"Fetching article: {article['title']}")
                    body = self.fetch_article_body(article["url"])
                    metadata = self.get_metadata(ticker, body)
                    self.news_db.save(published, ticker, article["url"], metadata["relevance"], metadata["sentiment"],
                                      metadata["outlook"])
            except Exception as e:
                print(f"Failed to fetch article or get metadata {article['title']}: {e} caching to avoid it next time")
                self.news_db.save(published, ticker, article["url"], article['relevance_score'],
                                  article['ticker_sentiment_score'],
                                  1 if article['ticker_sentiment_label'] == 'Bullish' else 0)


    def get_metadata(self, ticker, body):
        return self.openai.news_query(body, ticker)
