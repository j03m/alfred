import torch.nn as nn
import torch.nn.functional as F

# feed a seq_len as 30 features
class LinearSeries(nn.Module):
    def __init__(self, seq_len, hidden_dim, output_size, activation=None):
        super().__init__()
        self.activation = activation
        self.expand = nn.Linear(seq_len, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim, output_size)

    def forward(self, input_seq):
        input_seq = input_seq.view(input_seq.size(0), -1)
        x = self.expand(input_seq)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.final(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class LinearConv1dSeries(nn.Module):
    def __init__(self, seq_len, kernel_size, hidden_dim, output_size, activation=None, padding=15):
        super().__init__()
        self.activation = activation
        self.padding = padding
        self.seq_len =seq_len
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.hidden1 = nn.Linear(hidden_dim * (seq_len +1), hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim, output_size)

    def forward(self, input_seq):

        # Reshape to [batch_size, in_channels, seq_len] for Conv1d
        input_seq = input_seq.permute(0, 2, 1)  # Transpose to [64, 1, 30]
        x = self.conv1(input_seq)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.final(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


