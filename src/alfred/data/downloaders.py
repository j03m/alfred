import os
import pandas as pd
import yfinance as yf
import time
import requests
import ssl

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
            gross_margin = (float(report['grossProfit']) / float(report['totalRevenue'])) if report['totalRevenue'] != 0 else 0
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

# Example usage:
# downloader = AlphaDownloader(key_file='./keys/alpha.txt')
# treasury_df = downloader.treasury_yields()
# downloader.treasury_yields_to_csv(csv_file='treasury_yields.csv')
