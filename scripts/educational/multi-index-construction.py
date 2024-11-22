import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define some IDs
ids = [1, 2, 3]

def ensure_all_date_id_pairs(df, all_dates, all_ids, fill_value=0):
    """
    Ensures that the DataFrame contains all combinations of dates and IDs.

    Args:
        df (pd.DataFrame): The input DataFrame with a DatetimeIndex and an 'ID' column.
        all_dates (list): List of all unique dates that should be present.
        all_ids (list): List of all unique IDs that should be present.
        fill_value: The value to fill for missing combinations (default is 0).

    Returns:
        pd.DataFrame: A DataFrame with all date|ID combinations filled.
    """
    # Create a MultiIndex of all combinations of dates and IDs
    full_index = pd.MultiIndex.from_product([all_dates, all_ids], names=['Date', 'ID'])

    # Reindex the DataFrame to include all combinations, filling missing values with fill_value
    df = df.set_index(['ID'], append=True).reindex(full_index, fill_value=fill_value).reset_index()

    return df


# Generate sample data for debugging
def generate_sample_data():
    # Define a date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')



    # Create random data for some of the date-ID pairs
    data = []
    for date in dates:
        for _id in ids:
            if np.random.rand() > 0.3:  # Randomly skip some date-ID pairs
                data.append({
                    "Date": date,
                    "ID": _id,
                    "Rank": np.random.randint(1, 10),
                    "Feature1": np.random.random(),
                    "Feature2": np.random.random()
                })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Generate sample data
sample_data = generate_sample_data()


all_dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='M')
df = ensure_all_date_id_pairs(sample_data, all_dates, ids, fill_value=0)
print(df.head())