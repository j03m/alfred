import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from alfred.models import LSTMModel, Stockformer, AdvancedLSTM, LinearSeries, LinearConv1dSeries, LSTMConv1d, TransAm
from alfred.data import YahooNextCloseWindowDataSet, YahooChangeWindowDataSet, YahooDirectionWindowDataSet, \
    YahooChangeSeriesWindowDataSet
from alfred.model_persistence import maybe_save_model, get_latest_model
from statistics import mean
from sklearn.metrics import mean_squared_error
import argparse
import warnings
from alfred.devices import set_device, build_model_token
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import faulthandler
faulthandler.enable()

np.random.seed(42)

device = set_device()

# Make all UserWarnings throw exceptions
warnings.simplefilter("error", UserWarning)

BATCH_SIZE = 64
SIZE = 32


def get_simple_yahoo_data_loader(ticker, start, end, seq_length, predict_type, window=1):
    if predict_type == "change":
        dataset = YahooChangeWindowDataSet(ticker, start, end, seq_length, change=window)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True), dataset
    elif predict_type == "change-series":
        dataset = YahooChangeSeriesWindowDataSet(ticker, start, end, seq_length, change=window)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True), dataset
    elif predict_type == "direction":
        dataset = YahooDirectionWindowDataSet(ticker, start, end, seq_length, change=window)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True), dataset
    elif predict_type == "price":
        dataset = YahooNextCloseWindowDataSet(ticker, start, end, seq_length, change=window, log_return_scaler=True)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True), dataset
    else:
        raise NotImplementedError(f"Data type: {predict_type} not implemented")


# Step 4: Training Loop
def train_model(model, optimizer, scheduler, train_loader, patience, model_path, model_token, epochs=20,
                loss_function=nn.MSELoss()):
    model.train()

    single_loss = None
    patience_count = 0
    last_mean_loss = None
    for epoch in range(epochs):
        epoch_losses = []
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            if torch.isnan(single_loss):
                raise Exception("Found NaN!")

            single_loss.backward()
            optimizer.step()
            loss_value = single_loss.item()
            epoch_losses.append(loss_value)

        loss_mean = mean(epoch_losses)

        print(f'Epoch {epoch} loss: {loss_mean}, patience: {patience_count}')
        # todo: maybe save model really needs to take the optimizer and scheduler as well if its going to resume at an optimzied state
        # otherwise we lose like a 100 epochs prior to it getting to the right place again
        maybe_save_model(model, optimizer, scheduler, loss_mean, model_path, model_token)

        if last_mean_loss != None:
            if loss_mean >= last_mean_loss:
                patience_count += 1
            else:
                patience_count = 0
        last_mean_loss = loss_mean
        if patience_count > patience:
            print(f'Out of patience at epoch {epoch}. Patience count: {patience_count}. Limit: {patience}')
            return
        scheduler.step(loss_mean)


# Step 5: Evaluation and Prediction
def evaluate_model(model, loader):
    model.eval()
    predictions = []
    actuals = []
    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(seq).squeeze(-1)
            predictions.extend(output.cpu().tolist())
            actuals.extend(labels.cpu().tolist())

    return predictions, actuals


def plot(index, x):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(index, x, color="blue")

    plt.title("Daily close price")
    plt.grid(which='major', axis='y', linestyle='--')
    plt.show(block=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-token", type=str,
                        choices=['stockformer', 'lstm', 'advanced-lstm', "linear", "linear-conv1d", "lstm-conv1d",
                                 "trans-am"],
                        default='lstm',
                        help="prefix used to select model architecture, also used as a persistence token to store and load models")
    parser.add_argument("--model-path", type=str, default='./models', help="where to store models and best loss data")
    parser.add_argument("--patience", type=int, default=250,
                        help="when to stop training after patience epochs of no improvements")
    parser.add_argument("--action", type=str, choices=['train', 'eval', 'both'], default='both',
                        help="when to stop training after patience epochs of no improvements")
    parser.add_argument("--predict-type", type=str, choices=['change', 'change-series', 'direction', 'price'],
                        default='price',
                        help="type of data prediction to make")
    parser.add_argument("--window", type=int, default=1,
                        help="some datasets need a window, for example when predict the next change")
    parser.add_argument("--make-plots", action='store_true',
                        help="plot all data")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    args = parser.parse_args()

    ticker = 'SPY'
    start_date = '1999-01-01'
    end_date = '2021-01-01'
    seq_length = 30
    num_features = 1
    output = 1
    layers = 2
    # Initialize the model
    loss_function = nn.MSELoss()  # default
    if args.model_token == 'lstm':
        model = LSTMModel(features=num_features, hidden_dim=SIZE, output_size=output,
                          num_layers=layers).to(device)

    elif args.model_token == 'stockformer':
        model = Stockformer(1,1)
    elif args.model_token == 'advanced-lstm':
        model = AdvancedLSTM(features=num_features, hidden_dim=SIZE, output_dim=output)
    elif args.model_token == 'linear' and args.predict_type != 'direction':
        model = LinearSeries(seq_len=seq_length, hidden_dim=SIZE, output_size=output)
    elif args.model_token == 'linear' and args.predict_type == 'direction':
        loss_function = nn.BCELoss()
        model = LinearSeries(seq_len=seq_length, hidden_dim=SIZE, output_size=output, activation=nn.Sigmoid())
    elif args.model_token == 'linear-conv1d':
        # size 10 kernel should smooth about 2 weeks of data
        model = LinearConv1dSeries(seq_len=seq_length, hidden_dim=SIZE, output_size=output, kernel_size=10)
    elif args.model_token == 'lstm-conv1d':
        # size 10 kernel should smooth about 2 weeks of data
        model = LSTMConv1d(features=1, seq_len=seq_length, hidden_dim=SIZE, output_size=output, kernel_size=10)
    elif args.model_token == 'trans-am':
        model = TransAm(feature_size=250, last_bar=True) # not apples to apples, size needs to be div by heads so larger number from transam exp
    else:
        raise Exception("Model type not supported")

    model = model.to(device)

    # all this does is make a string separated by _ with the device tacked on the end
    model_token = build_model_token(
        [args.model_token, args.predict_type, seq_length, num_features, SIZE, layers, output])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    model_checkpoint = get_latest_model(args.model_path, model_token)
    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])

    if args.action == 'train' or args.action == 'both':
        train_loader, dataset = get_simple_yahoo_data_loader(ticker, start_date, end_date, seq_length,
                                                             args.predict_type, args.window)

        if args.make_plots:
            plot(dataset.df.index, dataset.df["Close"])
            plot(dataset.df.index[:len(dataset.data)], dataset.data)

        # Train the model
        train_model(model, optimizer, scheduler, train_loader, patience=args.patience, model_token=model_token,
                    model_path=args.model_path, epochs=args.epochs, loss_function=loss_function)

    if args.action == 'eval' or args.action == 'both':
        eval_loader, dataset = get_simple_yahoo_data_loader(ticker, end_date, '2023-01-01', seq_length,
                                                            args.predict_type)
        if args.make_plots:
            plot(dataset.df.index, dataset.df["Close"])
            plot(dataset.df.index[:len(dataset.data)], dataset.data)

        predictions, actuals = evaluate_model(model, eval_loader)

        # Calculate Mean Squared Error
        mse = mean_squared_error(actuals, predictions)
        print(f"Mean Squared Error: {mse}")

        # Plotting predictions against actuals
        plt.figure(figsize=(10, 5))
        plt.plot(actuals, label='Actual Values')
        plt.plot(predictions, label='Predictions', alpha=0.7)
        plt.title('Predictions vs Actuals')
        plt.xlabel('Sample Index')
        plt.ylabel('Scaled Price Change')
        plt.legend()
        plt.show(block=False)
        plt.savefig(f"{args.model_path}/{model_token}_eval.png", dpi=600, bbox_inches='tight', transparent=True)


# Main Script
if __name__ == "__main__":
    main()

# todo:
# add transformer
# add stockformer
# add informer
# add nbeats
# add tft
