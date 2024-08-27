import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


# Step 3: Define the LSTM Model


# Step 4: Training Loop
def train_model(model, train_loader, epochs=20):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print(f'Epoch {epoch} loss: {single_loss.item()}')


# Step 5: Evaluation and Prediction
def evaluate_model(model, data, scaler, seq_length):
    model.eval()
    predictions = []

    for i in range(len(data) - seq_length):
        seq = torch.tensor(data[i:i + seq_length], dtype=torch.float32)
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            predictions.append(model(seq).item())

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions


# Main Script
if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    seq_length = 60

    # Prepare the data
    train_loader, scaler = prepare_data(ticker, start_date, end_date, seq_length)

    # Initialize the model
    model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1, num_layers=2)

    # Train the model
    train_model(model, train_loader, epochs=20)

    # Fetch the data for prediction (newer data)
    prediction_data = fetch_data(ticker, start_date, '2021-01-01')
    prediction_data_normalized = scaler.transform(prediction_data.reshape(-1, 1)).flatten()

    # Evaluate the model
    predictions = evaluate_model(model, prediction_data_normalized, scaler, seq_length)

    # Plotting (optional)
    import matplotlib.pyplot as plt

    plt.plot(prediction_data[seq_length:], label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    plt.show()