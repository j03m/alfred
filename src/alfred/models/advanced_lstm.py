import torch
import torch.nn as nn
from alfred.devices import set_device

device = set_device()

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, lstm_output):
        # Applying the attention layer on the LSTM output
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum of LSTM output based on attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

class AdvancedLSTM(nn.Module):
    def __init__(self, features=1, hidden_dim=1024, output_dim=1, num_layers=2):
        super(AdvancedLSTM, self).__init__()
        self.lstm1 = nn.LSTM(features, hidden_dim, num_layers=num_layers, batch_first=True).to(device)
        self.dropout1 = nn.Dropout(0.3).to(device)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim).to(device)

        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True).to(device)
        self.dropout2 = nn.Dropout(0.3).to(device)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim).to(device)

        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True).to(device)

        self.attention = Attention(hidden_dim).to(device)
        self.fc = nn.Linear(hidden_dim*2, output_dim).to(device)

    def forward(self, x):
        # LSTM Layer 1
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = self.batch_norm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        # LSTM Layer 2
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.batch_norm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # LSTM Layer 3
        lstm_out, (h_n, _) = self.lstm3(x)

        # Attention Layer
        attention_vector = self.attention(lstm_out)

        # Concatenate the attention vector with the hidden state
        context_vector = torch.cat([attention_vector, h_n[-1]], dim=1)

        # Fully connected layer
        output = self.fc(context_vector)
        return output
