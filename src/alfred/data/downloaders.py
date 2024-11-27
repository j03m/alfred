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
from time import sleep
import re
from fuzzywuzzy import process
from edgar import Company

ssl.create_default_https_context = ssl._create_unverified_context


def download_ticker_list(ticker_list, output_dir="./data/", interval="1d", tail=-1, head=-1):
    bad_tickers = []
    for ticker in ticker_list:
        time.sleep(0.25)
        print("ticker: ", ticker)
        data_file = os.path.join(output_dir, f"{ticker}.csv")
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['Date'])
            today = datetime.today().strftime('%m-%d-%Y')
            latest_date = df['Date'].max().strftime('%m-%d-%Y')
            if latest_date == today:
                print(f"{ticker} prices are up2date")
                continue

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
                df.to_csv(data_file)
        except (requests.exceptions.HTTPError, ValueError) as e:
            print(f"Failed to download {ticker} due to an HTTP or Value error: {e}")
            bad_tickers.append(ticker)
    return bad_tickers


class AlphaDownloader:
    def __init__(self, key_file='./keys/alpha.txt', rate_limit=0.5):
        # Read the API key from the specified file
        with open(key_file, 'r') as file:
            self.api_key = file.readline().strip()
        self.rate_limit = rate_limit

    def get(self, url):
        sleep(self.rate_limit)
        return requests.get(url, verify=False)

    def earnings(self, symbol):
        # Construct the URL for the API request
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={self.api_key}'
        # Send the request and get the JSON response
        response = self.get(url)
        data = response.json()

        # Convert the earnings data to a DataFrame
        quarterly_earnings = data.get('quarterlyEarnings', None)
        if quarterly_earnings is None:
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

            total_revenue = float(report['totalRevenue']) if report['totalRevenue'] not in [None, 'None', '', 0] else 0
            if report['grossProfit'] in [None, 'None', '']:
                gross_margin = 0
            else:
                # Ensure totalRevenue is a valid float
                gross_margin = (float(report['grossProfit']) / total_revenue) if total_revenue != 0 else 0

            gross_margins.append(gross_margin)

            # Calculate Operating Margin
            operating_margin = (float(report['operatingIncome']) / total_revenue) if total_revenue != 0 else 0
            operating_margins.append(operating_margin)

            # Calculate Net Profit Margin
            net_profit_margin = (float(report['netIncome']) / total_revenue) if total_revenue != 0 else 0
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


# Example usage:
# downloader = AlphaDownloader(key_file='./keys/alpha.txt')
# treasury_df = downloader.treasury_yields()
# downloader.treasury_yields_to_csv(csv_file='treasury_yields.csv')


class ArticleDownloader:
    def __init__(self, cache_dir='./news', rate_limit=0.5):
        self.cache_dir = cache_dir
        self.api = AlphaDownloader()
        self.openai = OpenAiQuery()
        self.rate_limit = rate_limit
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get(self, url):
        sleep(self.rate_limit)
        return requests.get(url, verify=False)

    def fetch_article_body(self, url):
        """Fetch the article body from the given URL. Note we don't rate limit here since we're not hitting AA"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
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
            publish = article.get('time_published', None)
            if publish is None:
                date = datetime.today().date()
                publish_str = date.strftime("%Y%m%d")
            else:
                publish_str = publish.strftime('%Y%m%d')
            del article['time_published']

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

def make_sec_header(key_file="./keys/id.txt"):
    with open(key_file, 'r') as file:
        id = file.readline().strip()

    return {
        "User-Agent": id
    }

def get_company_info(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = make_sec_header()
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        ticker = ticker.upper()
        for item in data.values():
            if item['ticker'] == ticker:
                cik = str(item['cik_str']).zfill(10)
                return item['title'], cik
    else:
        raise Exception(f"Failed to retrieve company info: {response.status_code}")


def dump_filing(filing, output_path):
    headers = make_sec_header()
    response = requests.get(filing.full_url, headers=headers)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"Failed to download filing: {filing.full_url} (status: {response.status_code})")
        return False

def download_13f_filings(ticker, years, output_folder="./filings"):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Look up the company name and CIK
    company_name, cik = get_company_info(ticker)
    if not cik:
        raise ValueError(f"Unable to find CIK for ticker {ticker}")

    print(f"Found company: {company_name} (CIK: {cik})")

    # Initialize the EDGAR Company object
    company = Company(company_name, cik)

    # Manifest to keep track of saved files
    manifest = []

    for year in years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"Fetching filings for {ticker} ({start_date} to {end_date})...")

        try:
            filings = company.get_filings(filing_type="13F-HR", datea=start_date, dateb=end_date)
            if not filings:
                print(f"No filings found for {ticker} in {year}.")
                continue

            for filing in filings:
                # Build file name and output path
                date = filing.date_filed.replace("-", "")
                file_type = filing.file_type.lower()  # e.g., 'html', 'xml', etc.
                filename = f"{ticker}-{date}-13F-HR.{file_type}"
                output_path = os.path.join(output_folder, filename)

                # Download and save the filing
                if dump_filing(filing, output_path):
                    manifest.append({"filename": filename, "url": filing.full_url, "date": filing.date_filed})
        except Exception as e:
            print(f"Error fetching filings for {ticker} in {year}: {e}")

    # Save manifest to a JSON file
    manifest_path = os.path.join(output_folder, f"{ticker}-manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Manifest saved: {manifest_path}")
