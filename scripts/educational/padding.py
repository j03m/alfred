import numpy as np
import pandas as pd

# Example df_trimmed and predictions
df_trimmed = pd.DataFrame({'existing_col': range(100)})  # 100 rows
predictions = [0.1, 0.2, 0.3, 0.4, 0.5]  # Only 5 predictions

# Ensure predictions length matches df_trimmed
if len(predictions) < len(df_trimmed):
    diff = len(df_trimmed) - len(predictions)
    predictions = np.pad(predictions, (diff, 0), constant_values=np.nan)

print(predictions)

# Add the predictions column
model_token = 'example'
df_trimmed[f"analyst_{model_token}"] = predictions

# Forward-fill the NaN values at the front with the first available prediction
first_value = df_trimmed[f"analyst_{model_token}"].dropna().iloc[0]
df_trimmed[f"analyst_{model_token}"].fillna(first_value, inplace=True)

# Display the result
print(df_trimmed)