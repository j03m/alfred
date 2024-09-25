import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from alfred.data import feature_columns, label_columns
import sys
import io

def load_data(file_path):
    return pd.read_csv(file_path)


def compute_correlation(data, method):
    return data.corr(method=method)


def plot_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


def granger_causality(data, columns, max_lag):
    sys.stdout = io.StringIO()
    value = grangercausalitytests(data[columns], maxlag=max_lag)
    sys.stdout = sys.__stdout__
    return value

def granger_causality_score(granger_result):

    def p_value_to_score(p_value):
        # Map p-value to a score, where lower p-value gets a higher score
        if p_value >= 0.1:  # weak evidence
            return 0
        elif p_value >= 0.05:  # marginal evidence
            return 2
        elif p_value >= 0.01:  # moderate evidence
            return 4
        elif p_value >= 0.001:  # strong evidence
            return 7
        else:  # very strong evidence
            return 10

    scores = {}

    for lag, results in granger_result.items():
        # Extract the p-value for the F-test
        p_value = results[0]['ssr_ftest'][1]
        # Convert the p-value to a score
        score = p_value_to_score(p_value)
        scores[lag] = score

    return scores


def causal_inference(data, treatment, outcome):
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=data.columns.difference([treatment, outcome]).tolist()
    )
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(identified_estimand)
    return causal_estimate


def main(args):
    # Load data
    data = load_data(args.data_file)
    data.index = pd.to_datetime(data['Date'])
    data = data.drop(columns=['Date'])
    data = data.drop(columns=['Symbol'])
    # Correlation analysis
    print(f"Computing {args.correlation_method} correlation matrix...")
    correlation_matrix = compute_correlation(data, method=args.correlation_method)
    correlation_matrix = correlation_matrix[label_columns]
    print(correlation_matrix[label_columns])

    if args.plot:
        plot_correlation_matrix(correlation_matrix)

    # Granger causality analysis (only if columns are specified)
    print(f"Running Granger Causality Test on column pairs")
    for feature in feature_columns:
        for label in label_columns:
            granger_result = granger_causality(data, [feature, label], max_lag=args.max_lag)
            print(f"result {feature} -> {label}: ", granger_causality_score(granger_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation and Causation Analysis Script")
    parser.add_argument('--data-file', type=str, required=True, help="Path to the CSV data file")
    parser.add_argument('--correlation-method', type=str, default='pearson', choices=['pearson', 'spearman', 'kendall'],
                        help="Correlation method to use (default: pearson)")
    parser.add_argument('--plot', action='store_true', help="Plot the correlation matrix heatmap")
    parser.add_argument('--max-lag', type=int, default=30,
                        help="Maximum lag to use for Granger Causality test (default: 30)")

    args = parser.parse_args()
    main(args)
