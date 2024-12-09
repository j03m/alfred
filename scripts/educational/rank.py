import pandas as pd
import numpy as np
from alfred.utils import make_datetime_index
# Generate sample data
np.random.seed(42)  # For reproducibility

# Create a DataFrame with 10 tickers and 5 days of data
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "FB", "TSLA", "NFLX", "NVDA", "BABA", "V"]
dates = pd.date_range("2024-01-01", periods=5)
returns = np.random.rand(50) - 0.5  # Random returns between -0.5 and 0.5

data = {
    "date": np.tile(dates, len(tickers)),
    "id": np.repeat(tickers, len(dates)),
    "return": returns,
}

df = pd.DataFrame(data)
df = make_datetime_index(df, date_column="date")
df = df.sort_index()

df["rank"] = df.groupby(df.index)["return"].rank(ascending=False).astype(int)
print(df)