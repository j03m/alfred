import sys

# Mock the 'this' module to prevent it from executing
sys.modules['this'] = None

from alfred.models import Stockformer, Transformer
from alfred.devices import set_device
from alfred.model_persistence import get_latest_model
from alfred.data import DatasetStocks
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import argparse
import warnings

# Make all UserWarnings throw exceptions
warnings.simplefilter("error", UserWarning)


def plot_predictions_vs_labels(df, prediction_columns, label_columns):
    for pred_col, label_col in zip(prediction_columns, label_columns):
        plt.figure(figsize=(10, 6))

        plt.scatter(df.index, df[pred_col], color='blue', alpha=0.5, label=f'Predicted {pred_col}')

        plt.scatter(df.index, df[label_col], color='orange', alpha=0.5, label=f'Actual {label_col}')

        plt.title(f'{pred_col} vs {label_col}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-file', type=str, help="training data")
    parser.add_argument('--symbol', type=str, default="GE", help="symbol to render")
    parser.add_argument("--sequence-length", type=int, default=24, help="sequence length")
    parser.add_argument('--batch-size', type=int, default=32, help="batch size")
    parser.add_argument("--model-path", type=str, default='./models', help="where to store models and best loss data")
    parser.add_argument("--model-token", type=str, choices=['transformer', 'stockformer'], default='transformer',
                        help="prefix used to select model architecture, also used as a persistence token to store and load models")
    parser.add_argument("--action", type=str, choices=['train', 'assess'], default='train',
                        help="train to train, assess to check the prediction value")

    args = parser.parse_args()

    # fitler down to one symbol and test that way
    data_set = DatasetStocks(args.training_file, args.sequence_length)
    data_set.filter(args.symbol)

    device = set_device()

    if args.model_token == 'stockformer':
        model = Stockformer(
            d_model=512,
            enc_in=data_set.features,
            c_out=data_set.labels
        ).to(device)
    elif args.model_token == 'transformer':
        model = Transformer(model_dim=512, input_dim=data_set.features, output_dim=data_set.labels).to(device)

    model_data = get_latest_model(args.model_path, args.model_token)
    if model_data is not None:
        model.load_state_dict(torch.load(model_data))
    else:
        raise Exception("No model found, must render a previously trained model")

    all_outputs = []

    # Define DataLoader
    train_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )

    criterion = nn.MSELoss()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        batch_x1 = batch_x.reshape(-1, batch_x.shape[-2], batch_x.shape[-1]).float().to(device)
        batch_y1 = batch_y.reshape(-1, batch_y.shape[-2], batch_y.shape[-1]).float().to(device)
        outputs = model(batch_x1)

        # Accumulate the data
        all_outputs.append(outputs.cpu().detach().numpy().reshape(-1, 4))

    flattened_outputs = np.concatenate(all_outputs, axis=0)
    data_set.trim_to_size(len(flattened_outputs))
    data_set.df[['Prediction_1', 'Prediction_2', 'Prediction_3', 'Prediction_4']] = flattened_outputs

    plot_predictions_vs_labels(data_set.df,
                               prediction_columns=['Prediction_1', 'Prediction_2', 'Prediction_3', 'Prediction_4'],
                               label_columns=data_set.label_columns)


if __name__ == "__main__":
    main()
