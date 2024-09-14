import pandas as pd

# Create a sample DataFrame
data = {
    'A': [10, 20, 30, 40, 50],
}

df = pd.DataFrame(data)

df["B"] = df["A"]

# Display original DataFrame
print("Original DataFrame:")
print(df)

# Shift the DataFrame down by 1 row
df["B"] = df["B"].shift(-1)
df["C"] = df["B"] - df["A"]
print("Shifted DataFrame:")
print(df)