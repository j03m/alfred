import pandas as pd
import numpy as np

# Simulate some stock prices for multiple stocks over multiple days
dates = pd.date_range(start='2023-01-01', periods=10)
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
data = []

np.random.seed(42)  # For reproducibility

for date in dates:
    for symbol in symbols:
        # Generate random close prices
        close_price = np.random.uniform(100, 200)
        data.append([date, symbol, close_price])

# Create a DataFrame
df_prices = pd.DataFrame(data, columns=['Date', 'Symbol', 'Close'])

# Calculate covariance matrix for each date
def calculate_covariance(df_prices):
    price_lookback = df_prices.pivot_table(index='Date', columns='Symbol', values='Close')
    return_lookback = price_lookback.pct_change().dropna()
    cov_matrix = return_lookback.cov()
    return cov_matrix

# Calculate the covariance matrix using data from all previous dates (lookback window)
cov_matrices = {}
lookback = 5  # Example lookback period (in days)

for i in range(lookback, len(dates)):
    date_range = dates[i-lookback:i]
    df_lookback = df_prices[df_prices['Date'].isin(date_range)]
    cov_matrix = calculate_covariance(df_lookback)
    cov_matrices[dates[i]] = cov_matrix

# Flatten the covariance matrix and join with the original data
df_cov_list = []

for date in cov_matrices:
    cov_matrix = cov_matrices[date].values.flatten()
    df_cov = pd.DataFrame([cov_matrix], columns=[f'cov_{i}' for i in range(cov_matrix.size)])
    df_cov['Date'] = date
    df_cov_list.append(df_cov)

# Combine all covariance data into one DataFrame
df_cov_flattened = pd.concat(df_cov_list, ignore_index=True)

# Join the flattened covariance row with the features dataframe
df_features = df_prices.merge(df_cov_flattened, on='Date', how='left')

print(df_features)
