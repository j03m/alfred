#!/usr/bin/env python3
import argparse
import pandas as pd
from machine_learning_finance import get_coin_data_frames, attach_markers, generate_probability, \
    calc_durations_with_extremes, prob_chart, plot_full_analysis, calculate_and_graph_duration_probabilities, \
    calculate_and_graph_price_probabilities, analyze_extreme_deviations, \
    plot_full_analysis

windows = [300, 600, 900, 1500]

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data-source", default="ku_coin", choices=["ku_coin", "coin_base"], help="Data source to use (default: ku_coin)")
parser.add_argument("--symbol", default="ICP-USDT", help="Symbol to use (default: ICP-USDT)")
parser.add_argument("--window", type=int, default=300, help="Window size (default: 300)")
parser.add_argument("--start-date", default=pd.Timestamp.now().strftime("%Y-%m-%d"), help="Start date (default: today's date)")
args = parser.parse_args()

# Print command line arguments
print(args)

coin_base = args.data_source == "coin_base"
ku_coin = args.data_source == "ku_coin"
df_raw = get_coin_data_frames(args.window, args.symbol)

df_raw = df_raw.set_index("Date")
df_raw = df_raw.sort_index()
trend, prob_above_trend, prob_below_trend, volatility, model = generate_probability(df_raw)

df_raw = attach_markers(df_raw, trend, prob_above_trend)
df_durations = calc_durations_with_extremes(df_raw)
plot_full_analysis(df_raw, trend, prob_above_trend, prob_below_trend, model, df_durations)
prob_chart(df_raw, prob_above_trend)

start_date = pd.to_datetime(args.start_date)
calculate_and_graph_duration_probabilities(start_date, df_raw, df_durations)

percent_diff_from_trend = ((df_raw["Close"] - trend) / trend) * 100
calculate_and_graph_price_probabilities(percent_diff_from_trend)

analyze_extreme_deviations(df_durations, trend)
