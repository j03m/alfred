import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from alfred.models import LSTMModel
from alfred.data import SimpleYahooCloseChangeDataset
from alfred.devices import set_device
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

device = set_device()

def get_simple_yahoo_data_loader(ticker, start, end, seq_length):
    dataset = SimpleYahooCloseChangeDataset(ticker, start, end, seq_length)
    return DataLoader(dataset, batch_size=32, shuffle=False)

# Step 4: Training Loop
def train_model(model, train_loader, epochs=20):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)

            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(model.num_layers, 1, model.hidden_layer_size),
                                 torch.zeros(model.num_layers, 1, model.hidden_layer_size))

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print(f'Epoch {epoch} loss: {single_loss.item()}')


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
    ticker = 'SPY'
    start_date = '2010-01-01'
    end_date = '2021-01-01'
    seq_length = 365
    train_loader = get_simple_yahoo_data_loader(ticker, start_date, end_date, seq_length)

    # Initialize the model
    model = LSTMModel(input_size=seq_length, hidden_layer_size=512, output_size=1, num_layers=4).to(device)

    # Train the model
    train_model(model, train_loader, epochs=1000)

    eval_loader = get_simple_yahoo_data_loader(ticker, end_date, '2023-01-01', seq_length)

    # Evaluate the model
    predictions, labels = evaluate_model(model, eval_loader)

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

# todo:
# add transformer
# add stockformer
# add informer
# add nbeats
# add tft