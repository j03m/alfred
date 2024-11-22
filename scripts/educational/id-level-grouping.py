import pandas as pd
import numpy as np


def generate_and_process_data(num_ids=10, num_dates=365, window_size=60):
    # Generate a date range
    dates = pd.date_range(start='2021-01-01', periods=num_dates, freq='D')

    # Generate IDs
    ids = np.arange(1, num_ids + 1)

    # Create a DataFrame with all combinations of dates and IDs
    df = pd.DataFrame({
        'Date': np.repeat(dates, num_ids),
        'ID': np.tile(ids, num_dates)
    })

    # Generate random data for Price, Volume, EPS, and Rank
    np.random.seed(0)  # For reproducibility
    df['Price'] = np.random.uniform(100, 200, size=len(df))
    df['Volume'] = np.random.randint(1000, 5000, size=len(df))
    df['EPS'] = np.random.uniform(1, 5, size=len(df))
    df['Rank'] = np.random.randint(0, num_ids, size=len(df))

    # Ensure the data is sorted by Date and ID
    df.sort_values(by=['Date', 'ID'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create sequences without pivoting
    unique_dates = df['Date'].unique()
    num_windows = len(unique_dates) - window_size + 1
    sequences = []

    for i in range(num_windows):
        # Get dates for the current window
        window_dates = unique_dates[i:i + window_size]
        # Get data for these dates
        window_data = df[df['Date'].isin(window_dates)].copy()
        # Ensure data is sorted by Date and ID
        window_data.sort_values(by=['Date', 'ID'], inplace=True)
        window_data.reset_index(drop=True, inplace=True)
        # Append the sequence
        sequences.append(window_data)

    # Each sequence is a DataFrame containing data for window_size dates and all IDs
    # Optionally, convert each DataFrame to a NumPy array
    sequences_array = [seq[['ID', 'Price', 'Volume', 'EPS', 'Rank']].values for seq in sequences]

    # Output some information for review
    print(f"Generated DataFrame shape: {df.shape}")
    print(f"Number of sequences generated: {len(sequences)}")
    print(f"Each sequence shape: {sequences_array[0].shape} (should be {window_size * num_ids} x 5)")

    # Return the sequences for further inspection if needed
    return sequences_array, df


# Run the function
sequences_array, df = generate_and_process_data()

# Optionally, inspect the first sequence
print("\nFirst sequence:")
print(sequences_array[0])
