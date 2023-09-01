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
from .actions import BUY, SHORT

pd.set_option('mode.chained_assignment', None)


def make_change_point_column_name(prefix):
    return f"{prefix}_change_point"


def detect_change_points(df, data_column, hazard=30, mu=0, kappa=1, alpha=1, beta=1):
    data = df[data_column].dropna().values
    bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(hazard),
                                                 bocd.StudentT(mu=mu, kappa=kappa, alpha=alpha, beta=beta))
    rt_mle = np.empty(data.shape)
    for i, d in enumerate(data):
        bc.update(d)
        rt_mle[i] = bc.rt
    rt_mle_padded = np.insert(rt_mle, 0, np.full(len(df) - len(rt_mle) + 1, 0))
    return np.where(np.diff(rt_mle_padded) < 0, True, False)


def make_price_marker_from_boolean(df, bool_col, price_col, final_col):
    df[final_col] = np.where(df[bool_col], df[price_col], np.nan)
    return df


def calculate_trend_metrics_full(df, periods=[30, 60, 90]):
    # Get a moving average for the whole series, but tail it just to our test period and call it trend
    column_list = []
    for period in periods:
        # window of trends
        trend_col = f"trend-{period}"
        df[trend_col] = df["Close"].rolling(period).mean()
        df = df.dropna()
        column_list.append(trend_col)

        # get a trend diff
        diff = f"trend-diff-{period}"
        df[diff] = df["Close"] - df[trend_col]
        column_list.append(diff)

        # detect change points
        # detect cp on each period
        cp = f"change-point-{period}"
        df[cp] = detect_change_points(df, data_column=trend_col, hazard=period)
        column_list.append(cp)

        # detect the derivative of a polynomial between the points, this should indicate trend direction
        poly = f"polynomial_derivative-{period}"
        df[poly] = compute_derivatives_between_change_points(df, cp, trend_col)
        column_list.append(poly)
    return df, column_list


def generate_ai_columns(periods=[30, 60, 90]):
    column_list = []
    for period in periods:
        column_list.append(f"trend-{period}")
        column_list.append(f"trend-diff-{period}")
        column_list.append(f"change-point-{period}")
        column_list.append(f"polynomial_derivative-{period}")
    return column_list


def calculate_trend_metrics_for_ai(full_series_df, test_period_df, periods=[30, 60, 90]):
    # Get a moving average for the whole series, but tail it just to our test period and call it trend
    column_list = []
    for period in periods:
        # window of trends
        trend_col = f"trend-{period}"
        concat_df = pd.concat([full_series_df, test_period_df])
        test_period_df[trend_col] = concat_df["Close"].rolling(period).mean().tail(len(test_period_df))
        column_list.append(trend_col)

        # get a trend diff
        diff = f"trend-diff-{period}"
        test_period_df[diff] = test_period_df["Close"] - test_period_df[trend_col]
        column_list.append(diff)

        # detect change points
        # detect cp on each period
        cp = f"change-point-{period}"
        test_period_df[cp] = detect_change_points(test_period_df, data_column=trend_col, hazard=period)
        column_list.append(cp)

        # detect the derivative of a polynomial between the points, this should indicate trend direction
        poly = f"polynomial_derivative-{period}"
        test_period_df[poly] = compute_derivatives_between_change_points(test_period_df, cp, trend_col)
        column_list.append(poly)
    return test_period_df, column_list


def compute_derivatives_between_change_points(df, cp_column, data_column):
    raise Exception("Fix me, I have lookahead bias")

    # Todo, you have to change this such that you are
    # graphing from the last change point you've seen to the
    # end of data, not change point to change point

    # Compute change points and prepend a dummy change point at the start
    change_points = [df.index[0]] + list(df.loc[df[cp_column]].index)

    derivatives = pd.Series(index=df.index)

    # Loop through pairs of change points
    for i in range(len(change_points) - 1):
        # Extract data between change points
        segment = df.loc[change_points[i]:change_points[i + 1]]

        model, x, y = calculate_polynomial_regression(segment, data_column)

        ridge = model.named_steps['linearregression']
        deriv = np.polyder(ridge.coef_[::-1])
        yd_plot = np.polyval(deriv, x)

        # Assign all values of yd_plot to corresponding positions in derivatives
        derivatives.loc[segment.index] = yd_plot.flatten()

    return derivatives.fillna(0)


def calculate_polynomial_regression(df, column):
    # Assume x and y are your data
    df['DateNumber'] = [i for i in range(len(df))]

    # Prepare input features
    x = df[['DateNumber']]
    y = df[column]

    # Transform the x data into polynomial features
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(x, y)
    return model, x, y


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
            actions.append(BUY)  # go long
        elif max_short_profit > profit_threshold and corresponding_short_drawdown < drawdown_threshold:
            actions.append(SHORT)  # go short
        else:
            if len(actions) != 0:
                actions.append(actions[-1])  # hold the previous action
            else:
                actions.append(BUY)  # assume long (can happen)

    return actions
