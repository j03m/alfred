import torch
import torch.nn as nn


# Enum for extractor types (assumed defined elsewhere)
class ExtractorType:
    NONE = "none"
    LSTM = "lstm"
    CONVOLUTION = "convolution"
    ATTENTION = "attention"


# LSTM Extractor: Outputs the final hidden state
class LSTMExtractor(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch_size, hidden_size)
        return h_n[-1]  # (batch_size, hidden_size)


# Convolutional Extractor: Applies convolution and pooling
class ConvExtractor(nn.Module):
    def __init__(self, num_features, hidden_size, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(num_features, hidden_size, kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len)
        x = self.conv(x)  # (batch_size, hidden_size, seq_len)
        x = self.pool(x)  # (batch_size, hidden_size, 1)
        return x.squeeze(-1)  # (batch_size, hidden_size)


# Attention Extractor: Computes a weighted context vector
class AttentionExtractor(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.attention = nn.Linear(num_features, 1)
        self.projection = nn.Linear(num_features, hidden_size)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * x, dim=1)  # (batch_size, num_features)
        return self.projection(context)  # (batch_size, hidden_size)