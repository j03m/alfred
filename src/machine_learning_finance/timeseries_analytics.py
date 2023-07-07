import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import bocd
from .plotly_utils import prob_chart, graph_pdf_bar, bar_chart

pd.set_option('mode.chained_assignment', None)


def make_change_point_column_name(prefix):
    return f"{prefix}_change_point"


def detect_change_points(df, period=30, hazard=30, mu=0, kappa=1, alpha=1, beta=1, moving_avg = None):
    if moving_avg is None:
        moving_avg = df["Close"].rolling(period).mean()
    data = moving_avg.dropna().values
    bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(hazard),
                                                 bocd.StudentT(mu=mu, kappa=kappa, alpha=alpha, beta=beta))
    rt_mle = np.empty(data.shape)
    for i, d in enumerate(data):
        bc.update(d)
        rt_mle[i] = bc.rt
    rt_mle_padded = np.insert(rt_mle, 0, np.full(len(df) - len(rt_mle) + 1, 0))
    column = "moving_avg_{period}"
    df[make_change_point_column_name(column)] = np.where(np.diff(rt_mle_padded) < 0, True, False)
    return df, column


def compute_derivatives_between_change_points(df, prefix):
    # Compute change points and prepend a dummy change point at the start
    change_points = [df.index[0]] + list(df.loc[df[make_change_point_column_name(prefix)]].index)

    derivatives = pd.Series(index=df.index)

    # Loop through pairs of change points
    for i in range(len(change_points) - 1):
        # Extract data between change points
        segment = df.loc[change_points[i]:change_points[i + 1]]

        y_pred, model, x, y = calculate_polynomial_regression(segment)

        ridge = model.named_steps['linearregression']
        deriv = np.polyder(ridge.coef_[::-1])
        yd_plot = np.polyval(deriv, x)

        # Assign all values of yd_plot to corresponding positions in derivatives
        derivatives[segment.index] = yd_plot

    df["polynomial_derivative"] = derivatives
    return df


def calculate_polynomial_regression(df):
    # Assume x and y are your data
    df['DateNumber'] = [i for i in range(len(df))]

    # Prepare input features
    x = df[['DateNumber']]
    y = df['Close']

    # Transform the x data into polynomial features
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(x, y)
    y_pred = model.predict(x)
    return y_pred, model, x, y


# This is broken, doesn't work see git history for details
def calc_probabilties_without_lookahead(test, hist):
    # calculate the normal distribution on the historical period. This avoids look ahead bias when trying
    # to apply this to the test set.
    result = seasonal_decompose(hist['Close'], model='additive', period=90, extrapolate_trend='freq')
    residuals = result.resid
    percentage_deviations = residuals / result.trend * 100
    mu, std = norm.fit(percentage_deviations)

    # calculate the deviation for the test set, but apply to it the historical percentages
    test_result = seasonal_decompose(test['Close'], model='additive', period=90, extrapolate_trend='freq')
    test_residuals = test_result.resid
    test_percentage_deviations = test_residuals / test_result.trend * 100
    z_scores = test_percentage_deviations / std

    # Calculate the probability of a value being above the trend line for each point in the time series
    test["prob_above_trend"] = 1 - norm.cdf(z_scores)
    test["weighted-volume"] = test["Close"] * test["Volume"]
    test["trend"] = test_result.trend
    test["trend-diff"] = test["Close"] - test["trend"]
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


def calculate_profit_and_drawdown(price_series, start_index, window):
    if start_index + window > len(price_series):
        return None, None, None, None

    window_prices = price_series[start_index: start_index + window]
    max_price = max(window_prices)
    min_price = min(window_prices)

    long_profit = window_prices[-1] - window_prices[0]
    long_drawdown = window_prices[0] - min_price
    short_profit = window_prices[0] - window_prices[-1]
    short_drawdown = max_price - window_prices[0]

    return long_profit, long_drawdown, short_profit, short_drawdown


def generate_max_profit_actions(price_series,
                                window_sizes,
                                profit_threshold,
                                drawdown_threshold):
    actions = []
    for i in range(len(price_series)):
        max_long_profit = float('-inf')
        max_short_profit = float('-inf')
        corresponding_long_drawdown = 0
        corresponding_short_drawdown = 0
        for window in window_sizes:
            long_profit, long_drawdown, short_profit, short_drawdown = calculate_profit_and_drawdown(price_series, i,
                                                                                                     window)
            if long_profit is None:
                continue

            if long_profit > max_long_profit:
                max_long_profit = long_profit
                corresponding_long_drawdown = long_drawdown
            if short_profit > max_short_profit:
                max_short_profit = short_profit
                corresponding_short_drawdown = short_drawdown
        if max_long_profit > profit_threshold and corresponding_long_drawdown < drawdown_threshold:
            actions.append(1)  # go long
        elif max_short_profit > profit_threshold and corresponding_short_drawdown < drawdown_threshold:
            actions.append(-1)  # go short
        else:
            actions.append(0)  # hold the previous position
    return actions
