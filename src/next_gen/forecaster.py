import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import joblib
import os
import numpy as np
import argparse
import glob
import json
import random
from torch.utils.tensorboard import SummaryWriter
g_tensor_board_writer = None

g_num_features = 11  # Number of input features
g_hidden_dim = 64  # Number of LSTM units
g_output_dim = 1  # Number of output values
g_num_layers = 5  # Number of LSTM layers
g_dropout = 0.2  # Dropout rate
g_num_epochs = 2000  # Epochs
g_update_interval = 10
g_eval_save_interval = 10


class Forecaster(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, num_layers, dropout):
        super(Forecaster, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Attention Layer: Implement a simple attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)

        # Output Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attn_applied = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(attn_applied)

        return output


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


def get_latest_model(model_path, model_prefix):
    search_pattern = os.path.join(model_path, f"{model_prefix}*.pth")
    model_files = glob.glob(search_pattern)
    if not model_files:
        print("No previous model.")
        return None
    model_files.sort(key=lambda x: int(os.path.basename(x)[len(model_prefix):-4]), reverse=True)
    print(f"Found {model_files[0]} for previous model.")
    return model_files[0]


def save_next_model(model, model_path, model_prefix):
    search_pattern = os.path.join(model_path, f"{model_prefix}*.pth")

    model_files = glob.glob(search_pattern)

    max_counter = -1
    for model_file in model_files:
        basename = os.path.basename(model_file)
        counter = basename[len(model_prefix):-4]
        try:
            counter = int(counter)
            if counter > max_counter:
                max_counter = counter
        except ValueError:
            continue  # Skip files that do not end with a number

    next_counter = max_counter + 1
    next_model_filename = f"{model_prefix}{next_counter}.pth"
    next_model_path = os.path.join(model_path, next_model_filename)

    # Save the model
    torch.save(model.state_dict(), next_model_path)
    print(f"Model saved to {next_model_path}")
    return next_model_path


def is_debugger_active():
    try:
        import pydevd  # Module used by PyCharm's debugger
        return True
    except ImportError:
        return False


def set_device():
    if torch.cuda.is_available() and not is_debugger_active():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and not is_debugger_active():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def get_best_lost(model_path, model_prefix):
    try:
        with open(f"{model_path}/{model_prefix}-metrics.json", 'r') as f:
            metrics = json.load(f)
        best_loss = metrics['best_loss']
    except (FileNotFoundError, KeyError):
        best_loss = float('inf')
    return best_loss


def set_best_loss(model_path, model_prefix, loss):
    with open(f"{model_path}/{model_prefix}-metrics.json", 'w') as f:
        json.dump({'best_loss': loss}, f)


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


def eval_forecaster(model_path,
                    model_prefix,
                    eval_data_path):
    current_model = get_latest_model(model_path, model_prefix)
    model = Forecaster(g_num_features, g_hidden_dim, g_output_dim, g_num_layers, g_dropout)
    if current_model is not None:
        print("found model, loading previous state.")
        model.load_state_dict(torch.load(current_model))
    else:
        raise Exception(f"No model to test found at: {model_path} with {model_prefix}!")

    return eval_forecaster_model(model, model_path, model_prefix, eval_data_path)


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
            maybe_save_model(epoch, eval_data_path, eval_save, model, model_path, model_prefix)
            g_tensor_board_writer.flush()

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()



    maybe_save_model(epoch, eval_data_path, eval_save, model, model_path, model_prefix)


def maybe_save_model(epoch, eval_data_path, eval_save, model, model_path, model_prefix):
    if eval_save:
        eval_loss = eval_forecaster_model(model, eval_data_path)
        g_tensor_board_writer.add_scalar("Loss/eval", eval_loss, epoch)
        best_loss = get_best_lost(model_path, model_prefix)
        if eval_loss < best_loss:
            print(f"New best model: {eval_loss} vs {best_loss}: saving")
            save_next_model(model, model_path, model_prefix)
            set_best_loss(model_path, model_prefix, eval_loss)
        else:
            print(f"{eval_loss} vs {best_loss}: declining save")
    else:
        print("saving model at: ", epoch)
        save_next_model(model, model_path, model_prefix)


# todo:
# add tensor board: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html?highlight=tensorboard
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
        train_forecaster(
            model_path=args.model_path,
            model_prefix=args.model_prefix,
            training_data_path=training_data_path,
            eval_data_path=eval_data_path
        )
        # todo reminder we have the scaler for evaluation
        scaler = joblib.load(os.path.join(args.data_path, f"{args.ticker}_scaler.save"))
    elif args.action == "train-list":
        symbol_file = args.symbol_file
        if symbol_file is None:
            print("Action train-list requires --symbol-file")
            exit(-1)
        else:
            symbols = pd.read_csv(args.symbol_file)["Symbols"].tolist()
            random.shuffle(symbols)
            for ticker in symbols:
                g_tensor_board_writer = SummaryWriter(f"train_{ticker}")
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
