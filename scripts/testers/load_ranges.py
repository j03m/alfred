from alfred.data import load_csv_files_and_apply_range
import pandas as pd

df_symbols = pd.read_csv("./lists/baby_training_list.csv")
seed = 43

symbols = df_symbols["Symbols"].values
filtered_symbols = [symbol for symbol in symbols if symbol not in ["^VIX", "SPY"]]
results = load_csv_files_and_apply_range(filtered_symbols, "./data", 30, seed, "Unnamed: 0")

# Iterate over the dictionary and print start and end dates for each DataFrame
for symbol, df in results.items():
    start_date = df.index.min() if not df.empty else "No data"
    end_date = df.index.max() if not df.empty else "No data"
    print(f"Symbol: {symbol}, Start Date: {start_date}, End Date: {end_date}")
