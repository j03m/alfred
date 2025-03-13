import pandas as pd
def make_datetime_index(df, date_column="Unnamed: 0"):
    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    df = df.set_index(date_column)
    df = df.sort_index()  # Ensure it's sorted by date
    return df

def read_time_series_file(file, date_column="Unnamed: 0"):
    df = pd.read_csv(file)
    df = make_datetime_index(df, date_column)
    return df

def trim_timerange(df, min_date=None, max_date=None):
    if min_date is None:
        start_date = df.index.min()
    else:
        start_date = pd.Timestamp(min_date, tz='UTC')

    if max_date is None:
        # If max_date is not provided, we are not trimming the upper bound
        return df[df.index >= start_date]
    else:
        end_date = pd.Timestamp(max_date, tz='UTC')
        return df[(df.index >= start_date) & (df.index <= end_date)]

# don't use this use pd.merge_asof
def reindex_dataframes(main_df, *target_dfs):
    '''
    This will reindex all parameters following main to main's index
    '''
    for i, df in enumerate(target_dfs):
        if not isinstance(df, pd.DataFrame):
            raise f"parameter {i} was not a pd.DataFrame, Got: {type(df).__name__}"

    min_date, max_date = main_df.index.min(), main_df.index.max()
    freq = main_df.index.freq or "D"  # Use main_df's frequency if available
    date_range = pd.date_range(min_date, max_date, freq=freq, tz=main_df.index.tz)

    reindexed_dfs = tuple(
                        df.reindex(date_range, method='ffill')
                        for df in target_dfs)

    return reindexed_dfs