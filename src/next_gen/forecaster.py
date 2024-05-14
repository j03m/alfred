import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import joblib
import os
import numpy as np
import argparse
import random
from torch.utils.tensorboard import SummaryWriter
from devices import set_device
from model_persistence import (get_latest_model, maybe_save_model)
g_tensor_board_writer = None

g_num_features = 11  # Number of input features
g_hidden_dim = 64  # Number of LSTM units
g_output_dim = 1  # Number of output values
g_num_layers = 5  # Number of LSTM layers
g_dropout = 0.2  # Dropout rate
g_num_epochs = 2000  # Epochs
g_update_interval = 10
g_eval_save_interval = 100


class Forecaster(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, num_layers, dropout):
        super(Forecaster, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)

        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = self.batchnorm1(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x.permute(0, 2, 1)
        x = self.batchnorm2(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm3(x)

        # Apply attention
        x = self.attention(x)

        # Apply the output layer
        x = self.output(x)

        return x


def create_dataset(df, lookback):
    X, y = [], []
    for i in range(len(df) - lookback):
        # previous N bars
        feature = df[i:i + lookback].values
        # next bar
        target = df["Close"][i + lookback:i + lookback + 1].values
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X), dtype=torch.float), torch.tensor(np.array(y), dtype=torch.float)




def eval_forecaster_model(model,
                          training_data_path):
    df = pd.read_csv(training_data_path, index_col='Date', parse_dates=True)
    features, labels = create_dataset(df, 30)
    loader = data.DataLoader(data.TensorDataset(features, labels), batch_size=32)
    device = set_device()
    model.to(device)
    total_loss = 0
    total_batches = 0
    loss_function = torch.nn.MSELoss()
    with torch.no_grad():  # No gradients needed for evaluation
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            loss = loss_function(predictions, labels)
            total_loss += loss.item()
            total_batches += 1

    average_loss = total_loss / total_batches
    print(f'Eval: average loss: {average_loss}')

    return average_loss

def get_evaluator(model_path, model_prefix, eval_data_path):

    def eval_forecaster():
        current_model = get_latest_model(model_path, model_prefix)
        model = Forecaster(g_num_features, g_hidden_dim, g_output_dim, g_num_layers, g_dropout)
        if current_model is not None:
            print("found model, loading previous state.")
            model.load_state_dict(torch.load(current_model))
        else:
            raise Exception(f"No model to test found at: {model_path} with {model_prefix}!")

        return eval_forecaster_model(model, model_path, model_prefix, eval_data_path)

    return eval_forecaster

def train_forecaster(model_path,
                     model_prefix,
                     training_data_path,
                     eval_data_path=None):
    device = set_device()
    try:
        df = pd.read_csv(training_data_path, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("No data found, skipping")
        return

    eval_save = False
    if eval_data_path is not None:
        eval_save = True

    features, labels = create_dataset(df, 30)

    current_model = get_latest_model(model_path, model_prefix)
    model = Forecaster(g_num_features, g_hidden_dim, g_output_dim, g_num_layers, g_dropout)
    evaluator = get_evaluator(model_path, model_prefix, eval_data_path)
    if current_model is not None:
        print("found model, loading previous state.")
        model.load_state_dict(torch.load(current_model))

    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = data.DataLoader(data.TensorDataset(features, labels), batch_size=32)
    model.train()
    last_loss = float('inf')
    for epoch in range(g_num_epochs):
        if epoch % g_update_interval == 0:
            print("epoch: ", epoch, "last_loss: ", last_loss)
            g_tensor_board_writer.add_scalar("Loss/train", last_loss, epoch)
        if epoch % g_eval_save_interval == 0 and epoch >= g_eval_save_interval:
            maybe_save_model(epoch, evaluator, eval_save, model, model_path, model_prefix)
            g_tensor_board_writer.flush()

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

    maybe_save_model(epoch, evaluator, eval_save, model, model_path, model_prefix)





# todo:
# Fix tensorboard
# Learn about attention MultiheadAttention and the Transformer
# We want to change forecaster to take an input of 30 days and then predict an optimal exit (peak or trough) price for the
#   the following 30 days
#   This will require preprocessing to extract these days, you'll need to test that and graph
# how to shard on cpu
# how to shard on gpu
# get rudimentary forecaster working
# get a classifier working
# load both into an RL model that simulates trading
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./models/forecaster", help="Path to where the model should be stored ")
    parser.add_argument("--model-prefix", default="forecaster", help="The model name [prefix]{iteration}.pth")
    parser.add_argument("--data-path", default="./data", help="path to data and scaler files")
    parser.add_argument("--ticker", default="aapl", help="Ticker, used to find training csv and scaler")
    parser.add_argument('--symbol-file', type=str, help="List of symbols in a file")
    parser.add_argument("--action", default="train", help="train, train-list or eval")
    parser.add_argument("--eval-set", default=None,
                        help="if supplied, training will eval and only save the model when perf improves")
    args = parser.parse_args()
    if args.eval_set:
        eval_data_path = os.path.join(args.data_path, f"{args.ticker}_diffs.csv")
    else:
        eval_data_path = None

    if args.action == "train":
        training_data_path = os.path.join(args.data_path, f"{args.ticker}_diffs.csv")
        g_tensor_board_writer = SummaryWriter(f"train")
        train_forecaster(
            model_path=args.model_path,
            model_prefix=args.model_prefix,
            training_data_path=training_data_path,
            eval_data_path=eval_data_path
        )
        # todo reminder we have the scaler for evaluation
        scaler = joblib.load(os.path.join(args.data_path, f"{args.ticker}_scaler.save"))
        g_tensor_board_writer.close()
    elif args.action == "train-list":
        symbol_file = args.symbol_file
        if symbol_file is None:
            print("Action train-list requires --symbol-file")
            exit(-1)
        else:
            symbols = pd.read_csv(args.symbol_file)["Symbols"].tolist()
            random.shuffle(symbols)
            for ticker in symbols:
                g_tensor_board_writer = SummaryWriter(f"train")
                training_data_path = os.path.join(args.data_path, f"{ticker}_diffs.csv")
                train_forecaster(
                    model_path=args.model_path,
                    model_prefix=args.model_prefix,
                    training_data_path=training_data_path,
                    eval_data_path=eval_data_path
                )
                # todo reminder we have the scaler for evaluation
                scaler = joblib.load(os.path.join(args.data_path, f"{args.ticker}_scaler.save"))
                g_tensor_board_writer.close()
    elif args.action == "eval":
        eval_forecaster(
            model_path=args.model_path,
            model_prefix=args.model_prefix,
            eval_data_path=os.path.join(args.data_path, f"{args.ticker}_diffs.csv"))
    else:
        print("Unknown action: ", args.action)
