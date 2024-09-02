import torch
import torch.nn as nn
from alfred.devices import set_device
device = set_device()

class LSTMModel(nn.Module):
    def __init__(self, features, batch_size, seq_len, hidden_dim, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(features, hidden_dim, num_layers, batch_first=True)
        self.flatten = nn.Flatten()  # Flatten the output of LSTM
        self.linear = nn.Linear(hidden_dim * seq_len, output_size)
        self.batch_size = batch_size
        self.num_layers = num_layers

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        flattened_output = self.flatten(lstm_out)
        predictions = self.linear(flattened_output)
        return predictions
