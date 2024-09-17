import torch
import torch.nn as nn
from alfred.devices import set_device

device = set_device()

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # Applying the attention layer on the LSTM output
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum of LSTM output based on attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

class AdvancedLSTM(nn.Module):
    def __init__(self, features=1, hidden_dim=1024, output_dim=1, num_layers=2):
        super(AdvancedLSTM, self).__init__()
        self.lstm1 = nn.LSTM(features, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # LSTM Layer 1
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = self.layer_norm1(x)

        # LSTM Layer 2
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.layer_norm2(x)

        # LSTM Layer 3
        lstm_out, (h_n, _) = self.lstm3(x)

        # Attention Layer
        attention_vector = self.attention(lstm_out)

        # Concatenate the attention vector with the hidden state
        context_vector = torch.cat([attention_vector, h_n[-1]], dim=1)
        #h_n_flat = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        #context_vector = torch.cat([attention_vector, h_n_flat], dim=1)

        # Fully connected layer
        output = self.fc(context_vector)
        return output