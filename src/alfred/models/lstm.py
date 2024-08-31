import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, features, batch_size, hidden_layer_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(features, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.batch_size = batch_size
        self.hidden_cell = (torch.zeros(num_layers, batch_size, self.hidden_layer_size),
                            torch.zeros(num_layers, batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
