# get api keys at: https://site.financialmodelingprep.com/developer/docs/dashboard

from financetoolkit import Toolkit

with open("../keys/fmp_api_key.txt", "r") as file:
    content = file.read()

API_KEY = content

companies = Toolkit(['AAPL'], api_key=API_KEY, start_date='2017-12-31')

# a Historical example
historical_data = companies.get_historical_data()

# a Financial Statement example
balance_sheet_statement = companies.get_balance_sheet_statement()

# a Ratios example
profitability_ratios = companies.ratios.collect_profitability_ratios()

# a Models example
extended_dupont_analysis = companies.models.get_extended_dupont_analysis()

# a Technical example
bollinger_bands = companies.technicals.get_bollinger_bands()