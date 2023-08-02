#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta


def check_data(symbol, years):
    filename = f'./data/{symbol}.csv'
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Data file for {symbol} not found.")
        return False

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Get the minimum and maximum dates in the DataFrame
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    # Calculate the difference between the max and min dates
    diff_years = (max_date - min_date).days / 365.25

    # Check if there are at least 'years' years of data
    if diff_years < years:
        print(f'{symbol} has less than {years} years of data.')
        return False
    else:
        print(f'{symbol} has {years} or more years of data.')
        return True


def main(years):
    # Read the training list file
    df = pd.read_csv('./lists/training_list.csv')

    # Check each symbol's data and remove the symbol from the list if necessary
    df = df[df['Symbols'].apply(check_data, years=years)]

    # Write the updated DataFrame back to the CSV file
    df.to_csv('./lists/training_list.csv', index=False)


if __name__ == "__main__":
    # Create the parser and add the 'years' argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, help='The minimum number of years of data required')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the 'years' argument
    main(args.years)
