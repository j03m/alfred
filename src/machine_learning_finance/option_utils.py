import pandas as pd
from datetime import datetime, timedelta
import QuantLib as ql
from fredapi import Fred
import yfinance as yf

def check_strike_price_rule(price):
    if price < 25:
        return 2.5 if price >= 10 else 1
    elif price < 200:
        return 5
    else:
        return 10


def generate_strike_prices(historical_price):
    if historical_price < 25:
        itm_below = 2
        atm = 3
        itm_above = 2
    elif historical_price < 50:
        itm_below = 2
        atm = 1
        itm_above = 2
    elif historical_price < 100:
        itm_below = 3
        atm = 1
        itm_above = 1
    else:
        itm_below = 4
        atm = 1
        itm_above = 0

    interval = check_strike_price_rule(historical_price)
    strike_prices = []
    for i in range(-itm_below, atm + itm_above + 1):
        strike_prices.append(historical_price + (i * interval))

    return strike_prices


def generate_option_chain(equity, historical_date, historical_price, weeks=8):
    historical_date = datetime.strptime(historical_date, '%Y-%m-%d')
    expiries = [historical_date + timedelta(weeks=i) for i in range(1, weeks + 1)]

    columns = ['Equity', 'Option Type', 'Expiry', 'Strike Price']
    option_chain_df = pd.DataFrame(columns=columns)

    strike_prices = generate_strike_prices(historical_price)

    data_to_concat = []
    for expiry in expiries:
        for strike_price in strike_prices:
            data_to_concat.append(pd.DataFrame({'Equity': [equity], 'Option Type': ['Call'], 'Expiry': [expiry], 'Strike Price': [strike_price]}))
            data_to_concat.append(pd.DataFrame({'Equity': [equity], 'Option Type': ['Put'], 'Expiry': [expiry], 'Strike Price': [strike_price]}))

    option_chain_df = pd.concat(data_to_concat, ignore_index=True)

    return option_chain_df


def american_option_price(valuation_date, option_type, strike_price, expiry_date, underlying_price, risk_free_rate,
                          dividend_yield, volatility, num_time_steps=100):
    ql.Settings.instance().evaluationDate = valuation_date

    # Option parameters
    exercise = ql.AmericanExercise(valuation_date, expiry_date)
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    option = ql.VanillaOption(payoff, exercise)

    # Market data
    quote = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
    risk_free_curve = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, ql.Actual365Fixed()))
    dividend_curve = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, dividend_yield, ql.Actual365Fixed()))
    volatility_curve = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(valuation_date, ql.NullCalendar(), volatility, ql.Actual365Fixed()))
    process = ql.BlackScholesMertonProcess(quote, dividend_curve, risk_free_curve, volatility_curve)

    # Price the option using the binomial tree model
    binomial_engine = ql.BinomialVanillaEngine(process, 'crr', num_time_steps)
    option.setPricingEngine(binomial_engine)
    binomial_price = option.NPV()

    return binomial_price


def read_api_key_from_file(file_path):
    with open(file_path, 'r') as file:
        api_key = file.readline().strip()
    return api_key


def get_risk_free_rate_data(start_date, end_date):
    fred_api_key = read_api_key_from_file('./keys/fred.txt')
    fred = Fred(api_key=fred_api_key)

    risk_free_rate_data = fred.get_series('GS3M', start_date, end_date)

    return risk_free_rate_data


def calculate_volatility(price_data, start_date, end_date):
    # Filter the price data for the specified date range
    window_price_data = price_data.loc[start_date:end_date]['Close']

    # Calculate daily returns
    daily_returns = window_price_data.pct_change().dropna()

    # Calculate the standard deviation of daily returns (volatility)
    volatility = daily_returns.std()

    return volatility

def get_expected_dividends_for_option_window(ticker, option_start_date, option_end_date):
    # Download historical dividend data
    dividend_data = yf.Ticker(ticker).dividends

    # Convert option start and end dates to Pandas Timestamps
    option_start_date = pd.to_datetime(option_start_date)
    option_end_date = pd.to_datetime(option_end_date)

    # Filter dividend payment dates within the option's life
    dividends_within_window = dividend_data[(dividend_data.index >= option_start_date) & (dividend_data.index <= option_end_date)]

    # Check if there are any dividend payment dates within the option's life
    if len(dividends_within_window) == 0:
        return 0

    # Sum the expected dividends
    expected_dividends = dividends_within_window.sum().values[0]

    return expected_dividends


def get_risk_free_rate_for_date(risk_free_data, target_date):
    target_date = pd.to_datetime(target_date)

    # Find the closest date in the index before or equal to the target_date
    closest_date = risk_free_data.index.asof(target_date)

    # If there's no available data before the target_date, use the first available data point
    if pd.isna(closest_date):
        closest_date = risk_free_data.index[0]

    return risk_free_data[closest_date]

'''
# Example usage
valuation_date = ql.Date(1, 1, 2021)
option_type = ql.Option.Call
strike_price = 100
expiry_date = ql.Date(1, 4, 2021)
underlying_price = 100
risk_free_rate = 0.01
dividend_yield = 0.0
volatility = 0.20
num_time_steps = 100

price = american_option_price(valuation_date, option_type, strike_price, expiry_date, underlying_price, risk_free_rate,
                              dividend_yield, volatility, num_time_steps)
print("Binomial tree price:", price)



# Example usage:
equity = 'AAPL'
historical_date = '2021-01-04'
historical_price = 129  # Example price for AAPL on 2021-01-04
option_chain = generate_option_chain(equity, historical_date, historical_price)
print(option_chain)

# Calculate the start date as 2500 days ago
start_date = (datetime.now() - timedelta(days=2500)).strftime('%Y-%m-%d')
# Get the current date
end_date = datetime.now().strftime('%Y-%m-%d')
'''

