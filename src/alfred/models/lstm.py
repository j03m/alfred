import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, features, batch_size, hidden_dim, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(features, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)
        self.batch_size = batch_size

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
