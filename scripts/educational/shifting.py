import pandas as pd

# Create a DataFrame with one row of numbers 1 to 10
data = {
    'Numbers': list(range(1, 11))
}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

back_df = df.copy()
front_df = df.copy()

# Create a new column with the shifted data
back_df['Shifted'] = df['Numbers'].shift(1)
front_df['Shifted'] = df['Numbers'].shift(-1)

# Display the DataFrame with the new column
print("\nDataFrame with FORWARD Shifted Column ie, for us past to present:")
print(back_df)

print("\nDataFrame with BACK Shifted Column ie, for us future to present:")
print(front_df)