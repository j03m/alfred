import pandas as pd
import numpy as np
import argparse

from alfred.metadata import TickerCategories
from alfred.easy import dfs_from_files

def analyze_labels(df, label_column, desired_r2=0.8):
    labels = df[label_column].values
    variance = np.var(labels)
    label_range = labels.max() - labels.min()
    mean = np.mean(labels)
    std = np.std(labels)

    # Variance-based target
    target_mse_variance = (1 - desired_r2) * variance if variance > 0 else 0.0

    # Range-based target (10% of range)
    target_rmse_range = 0.1 * label_range if label_range > 0 else 0.0
    target_mse_range = target_rmse_range ** 2

    # Conservative target
    target_mse = max(target_mse_variance, target_mse_range)
    target_rmse = np.sqrt(target_mse)

    return {
        "Variance (Baseline MSE)": variance,
        "Range": label_range,
        "Mean": mean,
        "Std": std,
        "Target MSE (R²)": target_mse_variance,
        "Target MSE (10% Range)": target_mse_range,
        "Suggested Target MSE": target_mse,
        "Suggested Target RMSE": target_rmse,
        "Target RMSE (% of range)": (target_rmse / label_range) * 100 if label_range > 0 else float('nan'),
        "Desired R²": desired_r2
    }


def print_results(label_col, stats):

    print(f"\nAnalysis for '{label_col}' (Desired R² = {stats['Desired R²']}):")
    print(f"  Variance (Baseline MSE): {stats['Variance (Baseline MSE)']:.4f}")
    print(f"  Range: {stats['Range']:.4f}")
    print(f"  Mean: {stats['Mean']:.4f}, Std: {stats['Std']:.4f}")
    print(f"  Target MSE (R² = {stats['Desired R²']:.1f}): {stats['Target MSE (R²)']:.4f}")
    print(f"  Target MSE (10% Range): {stats['Target MSE (10% Range)']:.4f}")
    print(f"  Suggested Target MSE: {stats['Suggested Target MSE']:.4f}")
    print(f"  Suggested Target RMSE: {stats['Suggested Target RMSE']:.4f} (avg error)")
    print(f"  Target RMSE as % of range: {stats['Target RMSE (% of range)']:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suggest a target MSE based on label columns in a CSV.")
    parser.add_argument("--label_column", type=str, default="PM", help="One or more label column names")
    parser.add_argument("--r2", type=float, default=0.8, help="Desired R² value (0 to 1), default 0.8")
    parser.add_argument('--tickers', type=str, default="./metadata/basic-tickers.json", help='Tickers to train on')
    parser.add_argument('--file-post-fix', type=str, default="_quarterly_magnitude",
                        help='assumes data/[ticker][args.file_post_fix].csv as data to use')

    args = parser.parse_args()

    tc = TickerCategories(args.tickers)
    training = tc.get(["training"])
    evals = tc.get(["evaluation"])

    files = []
    for ticker in training:
        files.append(f"./data/{ticker}{args.file_post_fix}.csv")

    dfs = dfs_from_files(files)
    training_df = pd.concat(dfs)

    print("TRAINING")
    results = analyze_labels(training_df, args.label_column, desired_r2=args.r2)
    print_results(args.label_column, results)

    print("EVAL")
    files = []
    for ticker in evals:
        files.append(f"./data/{ticker}{args.file_post_fix}.csv")

    dfs = dfs_from_files(files)
    eval_df = pd.concat(dfs)
    results = analyze_labels(eval_df, args.label_column, desired_r2=args.r2)
    print_results(args.label_column, results)