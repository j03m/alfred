import requests
import pandas as pd

class AlphaDownloader:
    def __init__(self, key_file='./keys/alpha.txt'):
        # Read the API key from the specified file
        with open(key_file, 'r') as file:
            self.api_key = file.readline().strip()

    def earnings(self, symbol):
        # Construct the URL for the API request
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={self.api_key}'
        # Send the request and get the JSON response
        response = requests.get(url)
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
