#!/usr/bin/env python3
import pandas as pd
from machine_learning_finance import get_coin_data_frames, attach_markers, generate_probability, \
    calc_durations_with_extremes, prob_chart, plot, calculate_and_graph_duration_probabilities, \
    calculate_and_graph_price_probabilities, calculate_and_graph_price_probabilities, analyze_extreme_deviations, \
    plot_full_analysis

windows = [300, 600, 900, 1500]

# for window in windows:
window = 300
coin_base = False
ku_coin = True
df_raw = get_coin_data_frames(window, "ICP-USDT")

# tickerObj = yf.download(tickers = "SPY", interval = "1d")
# df_raw = pd.DataFrame(tickerObj).tail(365)
# df_raw = df_raw.reset_index()

# [results, data, features, fig] = renderPredictions(df_raw, models, [], False)
# features = features.set_index("Date")
df_raw = df_raw.set_index("Date")
df_raw = df_raw.sort_index()
trend, prob_above_trend, prob_below_trend, volatility, model = generate_probability(df_raw)

df_raw = attach_markers(df_raw, trend, prob_above_trend)
df_durations = calc_durations_with_extremes(df_raw)
plot_full_analysis(df_raw, trend, prob_above_trend, prob_below_trend, model, df_durations)
prob_chart(df_raw, prob_above_trend)

start_date = pd.to_datetime('2023-04-23')
calculate_and_graph_duration_probabilities(start_date, df_raw, df_durations)

percent_diff_from_trend = ((df_raw["Close"] - trend) / trend) * 100
calculate_and_graph_price_probabilities(percent_diff_from_trend)

analyze_extreme_deviations(df_durations, trend)
