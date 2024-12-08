import pandas as pd
def make_datetime_index(df, date_column="Unnamed: 0"):
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    df = df.sort_index()  # Ensure it's sorted by date
    return df

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