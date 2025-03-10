import shap
import numpy as np
import argparse

import torch

from alfred.metadata import TickerCategories
from alfred.utils import trim_timerange, set_deterministic
from alfred.devices import set_device
from alfred.easy import prepare_data_and_model_raw

device = set_device()
set_deterministic(0)


def get_data_for_ticker_set(tickers,args):
    files = []
    for ticker in tickers:
        file = f"data/{ticker}_quarterly_directional.csv"
        files.append(file)

    (features_train,
     labels_train,
     model,
     optimizer,
     real_model_token,
     scaler,
     scheduler,
     was_loaded) = prepare_data_and_model_raw(
            augment_func=lambda df: trim_timerange(df, min_date=args.min_date, max_date=args.max_date),
            model_size=args.size,
            model_name=args.model,
            files=files)
    x_train_tensor = torch.tensor(features_train.values, dtype=torch.float32)
    return features_train, x_train_tensor, model

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run trainer with specified model.")
    parser.add_argument('--model', type=str, default='vanilla.medium', help='Name of the model to use')
    parser.add_argument('--size', type=int, default=1024, help='The size of the model to use')
    parser.add_argument('--tickers', type=str, default="./metadata/basic-tickers.json", help='Tickers to evaluate on')
    parser.add_argument('--min_date', type=str, default="2004-03-31",
                        help='Minimum date for timerange trimming (YYYY-MM-DD)')
    parser.add_argument('--max_date', type=str, default=None, help='Maximum date for timerange trimming (YYYY-MM-DD)')

    args = parser.parse_args()

    ticker_metadata = TickerCategories(args.tickers)
    train_tickers = ticker_metadata.get(["training"])
    eval_tickers = ticker_metadata.get(["evaluation"])

    train_df, x_train_tensor, model = get_data_for_ticker_set(train_tickers, args)
    eval_df, x_eval_tensor, _ = get_data_for_ticker_set(eval_tickers, args)

    model.eval()
    x_train_tensor = x_train_tensor.to(device)
    x_eval_tensor = x_eval_tensor.to(device)
    explainer = shap.DeepExplainer(model, x_train_tensor)

    shap_values = explainer.shap_values(x_train_tensor)

    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_ranking = sorted(zip(train_df.columns, shap_importance), key=lambda x: x[1], reverse=True)
    for feature, importance in feature_ranking:
        print(f"{feature}: {importance}")

if __name__ == "__main__":
    main()
