import pandas as pd
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate correlations with a specified column.")
    parser.add_argument('--file', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--column', type=str, required=True, help='Column name to correlate with')
    parser.add_argument('--drop', nargs='+', default=[], help='Columns to drop from the DataFrame')
    args = parser.parse_args()

    # Read the CSV file
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: The file {args.file} does not exist.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {args.file} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: Unable to parse the file {args.file}. Check if it's a valid CSV.")
        return

    # Drop specified columns
    df = df.drop(args.drop, axis=1, errors='ignore')  # 'ignore' prevents error if columns don't exist

    # Check if the specified column exists in the DataFrame
    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found in the DataFrame.")
        return

    # Calculate correlations
    correlation_with_label = df.corrwith(df[args.column])

    # Sort correlations to see highest first
    correlation_with_label = correlation_with_label.sort_values(ascending=False)

    # Print correlations with strength assessment
    for feature, correlation in correlation_with_label.items():
        cat = None
        if abs(correlation) < 0.19:
            cat = "Very Weak"
        elif abs(correlation) < 0.39:
            cat = "Weak"
        elif abs(correlation) < 0.59:
            cat = "Moderate"
        elif abs(correlation) < 0.79:
            cat = "Strong"
        elif abs(correlation) < 1.00:
            cat = "Very Strong"
        print(f"{feature}: {correlation} - {cat}")
if __name__ == "__main__":
    main()