import torch.nn as nn

# lstm model borrowed from: https://github.com/jinglescode/time-series-forecasting-pytorch.git
class LSTMModel(nn.Module):
    def __init__(self, features, hidden_dim, output_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(features, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_size=self.hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_dim, output_size)

        self.init_weights()

    # can use defaults
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batch_size = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batch_size, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions

class LSTMConv1d(nn.Module):
    def __init__(self, features, seq_len, hidden_dim, kernel_size, output_size, num_layers=1, padding=15, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

        lstm_size = int((seq_len + (2 * padding) - kernel_size) / 1) + 1
        self.lstm = nn.LSTM(lstm_size, hidden_size=self.hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_dim, output_size)


    def forward(self, input):
        batch_size = input.shape[0]

        # conv1d
        input_seq = input.permute(0, 2, 1)  # Transpose to [batch, features, seq_length]
        x = self.conv1(input_seq)
        x = self.relu(x)

        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batch_size, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions