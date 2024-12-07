import pandas as pd
def make_datetime_index(df, date_column="Unnamed: 0"):
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    df = df.sort_index()  # Ensure it's sorted by date
    return df