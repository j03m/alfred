import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import poisson
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


from .plotly_utils import prob_chart, graph_pdf_bar, bar_chart

pd.set_option('mode.chained_assignment', None)


def calculate_polynomial_regression(df):
    # Assume x and y are your data
    df['DateNumber'] = [i for i in range(len(df))]

    # Prepare input features
    x = df[['DateNumber']]
    y = df['Close']

    # Transform the x data into polynomial features
    degree = 2
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)

    # Now fit a Linear Regression model on the transformed data
    model = LinearRegression()
    model.fit(x_poly, y)

    # Now the model can predict a curve rather than a straight line
    y_pred = model.predict(x_poly)

    return y_pred


# This is broken, doesn't work see git history for details
def calc_probabilties_without_lookahead(test, hist, window_size=90):
    # Initialize new columns in test
    test['trend'] = None
    test['sd_trend'] = None
    test["prob_above_trend"] = None
    test["weighted-volume"] = None
    test["trend-diff"] = None

    # Iterate over test
    for i in range(len(test)):
        # Update hist with data up to current index in test
        hist_updated = pd.concat([hist, test.iloc[:i]])

        # Calculate the normal distribution on the updated historical period
        # hist_updated['trend'] = hist_updated['Close'].ewm(span=180, adjust=False).mean()
        # hist_updated['trend'].fillna(method='bfill', inplace=True)
        # hist_residuals = hist_updated['Close'] - hist_updated['trend']
        # hist_percentage_deviations = hist_residuals / hist_updated['trend'] * 100
        # mu, std = norm.fit(hist_percentage_deviations)

        result = seasonal_decompose(hist_updated['Close'], period=90, extrapolate_trend='freq')
        residuals = result.resid
        percentage_deviations = residuals / result.trend * 100
        mu, std = norm.fit(percentage_deviations)

        # Calculate the trailing moving average for the current point in test
        test['trend'].iloc[i] = test['Close'].iloc[:i + 1].ewm(span=180, adjust=False).mean().iloc[-1]

        # Calculate the deviation for the current point in test
        test_residuals = test['Close'].iloc[i] - test['trend'].iloc[i]
        test_percentage_deviations = test_residuals / test['trend'].iloc[i] * 100
        z_scores = test_percentage_deviations / std

        # Calculate the probability of the value being above the trend line for the current point in test
        test["prob_above_trend"].iloc[i] = 1 - norm.cdf(z_scores)
        test["weighted-volume"].iloc[i] = test["Close"].iloc[i] * test["Volume"].iloc[i]
        test["trend-diff"].iloc[i] = test_residuals

    # Calculate the seasonal decomposition trend for the current point in test
    result = seasonal_decompose(test['Close'], model='additive', period=90, extrapolate_trend='freq')
    test['sd_trend'] = result.trend

    return test


def calc_durations_with_extremes(df_raw):
    # get last index
    last_index = df_raw.iloc[-1].name

    # get the first index that is the beginning of a high probability window
    start_index = df_raw['high_prob_start'].first_valid_index()
    df_durations = pd.DataFrame(columns=['start', 'end', 'duration', 'extreme'])

    # loop through all high probability windows
    while start_index < last_index:
        start_pos = df_raw.index.get_loc(start_index)

        # loop through all indexes after the high probability window starts, searching for a cross to mark its end
        for index in df_raw.index[start_pos + 1:]:
            cross1 = df_raw.loc[index, 'cross_over_positive']
            cross2 = df_raw.loc[index, 'cross_over_negative']

            # continue until one of these is not nan
            if np.isnan(cross1) and np.isnan(cross2):
                continue

            # we found a cross, calculate how far it was from the probability start
            duration = (index - start_index).days

            # get the extreme value in the duration
            if (np.isnan(cross1)):
                extreme_value = df_raw.loc[start_index:index, "Close"].max()
                extreme_index = df_raw.loc[start_index:index, "Close"].idxmax()
            else:
                extreme_value = df_raw.loc[start_index:index, "Close"].min()
                extreme_index = df_raw.loc[start_index:index, "Close"].idxmin()

            # Create a new row using a dictionary
            row = {'start': start_index, 'end': index, 'duration': duration, 'extreme': extreme_value,
                   'extreme_index': extreme_index}
            df_durations = pd.concat([df_durations, pd.DataFrame([row])], ignore_index=True)

            # once we find a cross, we need to exit. Get the position of the exit.
            start_pos = df_raw.index.get_loc(index)

            break

        # find the next high probability window start AFTER the exit
        start_index = df_raw['high_prob_start'].iloc[start_pos + 1:].first_valid_index()

        if start_index is None:
            break

    # Create a box plot of the duration data
    return df_durations


def attach_markers(df_raw, trend, prob_above_trend):
    threshold = 0.85
    threshold_low = 0.15
    prob_above_trend = pd.Series(prob_above_trend, index=df_raw.index)
    high_prob_zones = (prob_above_trend > threshold) | (prob_above_trend < threshold_low)
    high_prob_starts = high_prob_zones[high_prob_zones == 1].index

    df_raw['high_prob_start'] = np.nan
    # Iterate over the high probability start dates
    for i, start_date in enumerate(high_prob_starts):
        df_raw.loc[start_date, 'high_prob_start'] = df_raw.loc[start_date, 'Close']

    # Calculate the sign of the difference between Close and trend at each point in time
    diff_sign = np.sign(trend - df_raw["Close"])

    # Take the difference of the sign values to detect when the sign changes
    cross_over = diff_sign.diff().fillna(0)

    # Detect when the sign changes from positive to negative or negative to positive
    cross_over_positive = (cross_over == -2).astype(int).diff().fillna(0)
    cross_over_negative = (cross_over == 2).astype(int).diff().fillna(0)

    # Create empty columns in df_raw
    df_raw['cross_over_positive'] = np.nan
    df_raw['cross_over_negative'] = np.nan

    # Set the values of the new columns based on cross_over_positive and cross_over_negative
    df_raw.loc[cross_over_positive == 1, 'cross_over_positive'] = df_raw.loc[cross_over_positive == 1, 'Close']
    df_raw.loc[cross_over_negative == 1, 'cross_over_negative'] = df_raw.loc[cross_over_negative == 1, 'Close']

    return df_raw


def calculate_and_graph_price_probabilities(percentage_differences):
    # Fit percentage differences to a normal distribution
    mean, std = norm.fit(percentage_differences)

    # Define the percentage deviation range
    min_percentage = int(np.floor(percentage_differences.min()))
    max_percentage = int(np.ceil(percentage_differences.max()))
    num_points = max_percentage - min_percentage + 1
    percentage_range = np.linspace(min_percentage, max_percentage, num_points)

    # Calculate the PDF of the normal distribution for the range of percentage deviations
    pdf_values = norm.pdf(percentage_range, mean, std)

    # Create a DataFrame with the percentage deviations and their corresponding PDF values
    pdf_df = pd.DataFrame({"Percentage Deviation": percentage_range, "PDF Value": pdf_values})

    graph_pdf_bar(pdf_df)
    print("Current price diff:", percentage_differences[-1])


def calculate_duration_probabilities(start_date, df_raw, df_durations):
    # seed 60 days from the start of when we want to predict when the
    # mean regression will happen
    n_periods = 60
    dates = [start_date + pd.DateOffset(days=i) for i in range(n_periods)]
    df = pd.DataFrame({'date': dates})

    durations = df_durations['duration'].values.tolist()

    # Fit a Poisson distribution to the durations
    # Then figure out the probability of a cross in n days
    rate = np.mean(durations)
    poisson_dist = poisson(rate)
    numbers = np.arange(1, n_periods + 1)
    cdf_values = poisson_dist.cdf(numbers)

    # Calculate the probabilities for each duration window
    window_probabilities = np.diff(cdf_values, prepend=0)

    # Graph as bars so we can predict when the price will
    total_probability = np.sum(window_probabilities)
    df['probability'] = window_probabilities
    df = df.set_index("date")
    return df


def calculate_and_graph_duration_probabilities(start_date, df_raw, df_durations):
    df = calculate_duration_probabilities(start_date, df_raw, df_durations)
    bar_chart(df, False)


def calc_extreme_percentage_deviations(df_durations, trend):
    extreme_percentage_deviations = []

    for index, row in df_durations.iterrows():
        start_date = row['start']
        end_date = row['end']
        extreme_price = row['extreme']
        extreme_index = row['extreme_index']

        trend_value = trend.loc[extreme_index]

        deviation_percentage = (extreme_price - trend_value) / trend_value * 100
        extreme_percentage_deviations.append(deviation_percentage)

    return extreme_percentage_deviations
