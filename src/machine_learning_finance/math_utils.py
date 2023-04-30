from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy.stats import norm
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression


def scale_to_price(series, df):
    price_range = [df['Close'].min(), df['Close'].max()]
    scaled_series = (series - series.min()) / (series.max() - series.min()) * (price_range[1] - price_range[0]) + \
                    price_range[0]
    return scaled_series


def generate_day_probability(df):
    [trend, prob_above_trend, prob_below_trend, volatility, model] = generate_probability(df)
    return trend.iloc[-1], prob_above_trend[-1], prob_below_trend[-1], volatility.iloc[-1]


def generate_probability(df):
    # Perform seasonal decomposition
    # period = int(len(df)/3.3)
    result = seasonal_decompose(df['Close'], model='additive', period=90, extrapolate_trend='freq')

    # Add trend back to original time series
    trend = result.trend

    # Compute the residuals by subtracting the trend from the original time series
    residuals = result.resid

    # Fit a Gaussian distribution to the residuals
    mu, std = norm.fit(residuals)

    # Compute the probability of a value being above or below the trend line
    # for each point in the time series
    z_scores = residuals / std
    prob_above_trend = 1 - norm.cdf(z_scores)
    prob_below_trend = norm.cdf(z_scores)

    volatility = df['Close'].pct_change()

    # Get the last 30 days of the trend
    last_30_days_trend = trend[-30:]

    # Create an array of integers representing the days
    x = np.arange(1, 31).reshape((-1, 1))

    # Create the linear regression model
    model = LinearRegression()

    # Fit the model on the last 30 days of the trend
    model.fit(x, last_30_days_trend)

    return trend, prob_above_trend, prob_below_trend, volatility, model


def generate_day_probability(df):
    [trend, prob_above_trend, prob_below_trend, volatility, model] = generate_probability(df)
    return trend.iloc[-1], prob_above_trend[-1], prob_below_trend[-1], volatility.iloc[-1]

