import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from alfred.models import LSTMModel, Transformer
from alfred.data import SimpleYahooCloseChangeDataset
from alfred.devices import set_device
from alfred.model_persistence import maybe_save_model, get_latest_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Make all UserWarnings throw exceptions
warnings.simplefilter("error", UserWarning)

device = set_device()

BATCH_SIZE = 32

def get_simple_yahoo_data_loader(ticker, start, end, seq_length):
    dataset = SimpleYahooCloseChangeDataset(ticker, start, end, seq_length)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Step 4: Training Loop
def train_model(model, train_loader, epochs=20):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    single_loss = None
    for epoch in range(epochs):
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)

            optimizer.zero_grad()

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels.unsqueeze(1))
            single_loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {single_loss.item()}')
            maybe_save_model(model, single_loss.item(), args.model_path, args.model_token)
            scheduler.step(single_loss)


# Step 5: Evaluation and Prediction
def evaluate_model(model, loader):
    model.eval()
    predictions = []
    actuals = []
    for seq, labels in loader:
        with torch.no_grad():
            predictions.append(model(seq).item())
            actuals.append(labels.item())
    return predictions, actuals


# Main Script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-token", type=str, choices=['transformer', 'lstm'],
                        default='lstm',
                        help="prefix used to select model architecture, also used as a persistence token to store and load models")
    parser.add_argument("--model-path", type=str, default='./models', help="where to store models and best loss data")
    args = parser.parse_args()

    ticker = 'SPY'
    start_date = '2010-01-01'
    end_date = '2021-01-01'
    seq_length = 365
    num_features = 1
    output = 1

    # Initialize the model
    if args.model_token == 'lstm':
        model = LSTMModel(features=num_features, hidden_layer_size=1024, batch_size=BATCH_SIZE, output_size=output, num_layers=4).to(device)
    elif args.model_token == 'transformer':
        model = Transformer(features=1, output_dim=1).to(device)
    else:
        raise Exception("Model type not supported")

    model_data = get_latest_model(args.model_path, args.model_token)
    if model_data is not None:
        model.load_state_dict(torch.load(model_data))

    # todo allow ticker as arg, do caching of files etc
    train_loader = get_simple_yahoo_data_loader(ticker, start_date, end_date, seq_length)

    # Train the model
    train_model(model, train_loader, epochs=1000)

    eval_loader = get_simple_yahoo_data_loader(ticker, end_date, '2023-01-01', seq_length)

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
    plt.show()
    plt.savefig(f"{args.model_path}/{args.model_token}_eval.png", dpi=600, bbox_inches='tight', transparent=True)

# todo:
# add transformer
# add stockformer
# add informer
# add nbeats
# add tft