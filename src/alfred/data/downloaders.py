import os
import pandas as pd
import yfinance as yf
import time
import requests
import ssl
from datetime import datetime
import json
import binascii
from .openai_query import OpenAiQuery

ssl.create_default_https_context = ssl._create_unverified_context

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
                min_date = df.index.min()
                max_date = df.index.max()
                print(f"Min date for {ticker}: {min_date}")
                print(f"Max date for {ticker}: {max_date}")
                df.to_csv(os.path.join(output_dir, f"{ticker}.csv"))
        except (requests.exceptions.HTTPError, ValueError) as e:
            print(f"Failed to download {ticker} due to an HTTP or Value error: {e}")
            bad_tickers.append(ticker)
    return bad_tickers


class AlphaDownloader:
    def __init__(self, key_file='./keys/alpha.txt'):
        # Read the API key from the specified file
        with open(key_file, 'r') as file:
            self.api_key = file.readline().strip()

    def earnings(self, symbol):
        # Construct the URL for the API request
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={self.api_key}'
        # Send the request and get the JSON response
        response = requests.get(url, verify=False)
        data = response.json()

        # Convert the earnings data to a DataFrame
        annual_earnings = pd.DataFrame(data['annualEarnings'])
        quarterly_earnings = pd.DataFrame(data['quarterlyEarnings'])

        quarterly_earnings = quarterly_earnings.drop(columns=['reportTime'])
        quarterly_earnings = quarterly_earnings.drop(columns=['fiscalDateEnding'])
        quarterly_earnings = quarterly_earnings.rename(columns={'reportedDate': 'Date'})

        annual_earnings = annual_earnings.fillna(0)
        quarterly_earnings = quarterly_earnings.fillna(0)

        return annual_earnings, quarterly_earnings

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
        response = requests.get(url)
        data = response.json()

        # Extract relevant margin data
        quarterly_reports = data['quarterlyReports']

        # Create lists to store the extracted data
        dates = []
        gross_margins = []
        operating_margins = []
        net_profit_margins = []

        for report in quarterly_reports:
            dates.append(report['fiscalDateEnding'])
            # handle the case where totalRev is missing (happens for AAPL for example)
            if report['totalRevenue'] in ['None', 'N/A', '0']:
                if report['grossProfit'] != 'None' and report['costofGoodsAndServicesSold'] != 'None':
                    report['totalRevenue'] = str(
                        float(report['grossProfit']) + float(report['costofGoodsAndServicesSold']))
                elif report['grossProfit'] != 'None' and report['costOfRevenue'] != 'None':
                    report['totalRevenue'] = str(float(report['grossProfit']) + float(report['costOfRevenue']))
                else:
                    report['totalRevenue'] = '0'

            if report['operatingIncome'] == 'None':
                report['operatingIncome'] = '0'

            # Calculate Gross Margin

            if report['grossProfit'] in [None, 'None', '']:
                gross_margin = 0
            else:
                # Ensure totalRevenue is a valid float
                total_revenue = float(report['totalRevenue']) if report['totalRevenue'] not in [None, 'None', '',0] else 0
                gross_margin = (float(report['grossProfit']) / total_revenue) if total_revenue != 0 else 0

            gross_margins.append(gross_margin)

            # Calculate Operating Margin
            operating_margin = (float(report['operatingIncome']) / float(report['totalRevenue'])) if report[
                                                                                                         'totalRevenue'] != 0 else 0
            operating_margins.append(operating_margin)

            # Calculate Net Profit Margin
            net_profit_margin = (float(report['netIncome']) / float(report['totalRevenue'])) if report[
                                                                                                    'totalRevenue'] != 0 else 0
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
            response = requests.get(url)
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
        response = requests.get(url)
        data = response.json()

        # Convert balance sheet data to DataFrame
        quarterly_balance_sheet = pd.DataFrame(data['quarterlyReports'])
        quarterly_balance_sheet['fiscalDateEnding'] = pd.to_datetime(quarterly_balance_sheet['fiscalDateEnding'])
        quarterly_balance_sheet.set_index('fiscalDateEnding', inplace=True)

        return quarterly_balance_sheet

    def cash_flow(self, symbol):
        url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={self.api_key}'
        response = requests.get(url)
        data = response.json()

        # Convert cash flow data to DataFrame
        quarterly_cash_flow = pd.DataFrame(data['quarterlyReports'])
        quarterly_cash_flow['fiscalDateEnding'] = pd.to_datetime(quarterly_cash_flow['fiscalDateEnding'])
        quarterly_cash_flow.set_index('fiscalDateEnding', inplace=True)

        return quarterly_cash_flow

    def historical_prices(self, symbol):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={self.api_key}'
        response = requests.get(url)
        data = response.json()

        # Convert the time series data to a DataFrame
        prices_df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.rename(columns={'5. adjusted close': 'close'})

        return prices_df[['close']].astype(float)

    def news_sentiment(self, symbol, time_from, time_to):
        formatted_from = time_from.strftime('%Y%m%dT%H%M')
        formatted_to = time_to.strftime('%Y%m%dT%H%M')
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&time_from={formatted_from}&time_to={formatted_to}&limit=10000&apikey={self.api_key}"
        response = requests.get(url)
        return response.json()

    def news_sentiment_for_window_and_symbol(self, symbol, time_from, time_to):
        # Fetch the news sentiment data
        data = self.news_sentiment(symbol, time_from, time_to)

        # Prepare the output list
        filtered_articles = []

        # Iterate through each article in the feed
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
                            "relevance_score": float(ticker_sentiment["relevance_score"]),
                            "ticker_sentiment_score": float(ticker_sentiment["ticker_sentiment_score"]),
                            "ticker_sentiment_label": ticker_sentiment["ticker_sentiment_label"]
                        }
                        # Add the formatted information to the results list
                        filtered_articles.append(article_info)
                        break  # Stop further checking once the relevant ticker is found

        return filtered_articles

# Example usage:
# downloader = AlphaDownloader(key_file='./keys/alpha.txt')
# treasury_df = downloader.treasury_yields()
# downloader.treasury_yields_to_csv(csv_file='treasury_yields.csv')


class ArticleDownloader:
    def __init__(self, cache_dir='./news'):
        self.cache_dir = cache_dir
        self.api = AlphaDownloader()
        self.openai = OpenAiQuery()
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def fetch_article_body(self, url):
        """Fetch the article body from the given URL."""
        response = requests.get(url)
        response.raise_for_status()  # Ensure we handle errors
        return response.text

    def generate_article_id(self, url):
        """Generate a unique article ID based on the CRC32 hash of the URL."""
        return format(binascii.crc32(url.encode()), '08x')

    def cache_article(self, ticker, date, article_id, metadata, body):
        """Cache article metadata and body to disk."""
        # Directory structure based on ticker and date
        ticker_dir = os.path.join(self.cache_dir, ticker)
        date_dir = os.path.join(ticker_dir, date)
        os.makedirs(date_dir, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(date_dir, f"{article_id}.json")
        with open(metadata_path, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=2)

        # Save article body
        body_path = os.path.join(date_dir, f"{article_id}.txt")
        with open(body_path, 'w') as body_file:
            body_file.write(body)

    def download_and_cache_article(self, ticker, time_from, time_to):
        """Fetch articles for a ticker and cache the metadata and bodies."""

        # Fetch articles using `news_sentiment_for_window`
        articles = self.api.news_sentiment_for_window_and_symbol(ticker, time_from, time_to)

        for i, article in enumerate(articles):
            # Convert time_published to a date string for directory naming
            publish_str = article.get('time_published', None)
            if publish_str is None:
                date = datetime.today().date()
                publish_str = date.strftime("%Y%m%d")

            # Check if article is already cached by URL
            article_id = self.generate_article_id(article["url"])
            ticker_dir = os.path.join(self.cache_dir, ticker)
            date_dir = os.path.join(ticker_dir, publish_str)
            body_path = os.path.join(date_dir, f"{article_id}.txt")

            # Skip fetching if already cached
            if os.path.exists(body_path):
                print(f"Article already cached: {ticker} - {publish_str} - {article_id}")
                continue




            # Fetch and cache the article body
            try:
                print(f"Fetching article: {article['title']}")
                body = self.fetch_article_body(article["url"])
                article = self.enrich_metadata(ticker, article, body)
                self.cache_article(ticker, publish_str, article_id, article, body)
            except requests.RequestException as e:
                print(f"Failed to fetch article {article['title']}: {e}")

    def enrich_metadata(self, ticker, article, body):
        response = self.openai.news_query(body, ticker)
        article.update(response)
        return article