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
    def __init__(self, num_features, hidden_size, num_layers=1):
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


# Convolutional Layer: Permutes, applies convolutions and restores, no pooling
class ConvLayer(nn.Module):
    def __init__(self, num_features, hidden_size, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(num_features, hidden_size, kernel_size, padding=kernel_size // 2)


    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len)
        x = self.conv(x)  # (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1) # (batch, seq, hidden
        return x


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

# todo here make this keep batch, seq, hidden
class AttentionLayer(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.attention = nn.Linear(num_features, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * x, dim=1)  # (batch_size, num_features)
        return context ## ??? should be the correct shape I think? or maybe we just want the weights?

class TransformerExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Embedding(seq_len, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout),
            num_layers
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len).to(x.device)
        x = self.embedding(x) + self.positional_encoding(positions)
        x = self.transformer(x)  # (batch_size, seq_len, hidden_size)
        x = x.mean(dim=1)  # Average to get (batch_size, hidden_size)
        return x

class TimeStepAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)  # Learn a weight for each time step

    def forward(self, x):
        # Input: (batch_size, seq_len, hidden_size)
        weights = torch.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * x, dim=1)  # (batch_size, hidden_size)
        return context  # (batch_size, hidden_size)

class FeatureAttention(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.attention = nn.Linear(num_features, num_features)  # Learn weights for features

    def forward(self, x):
        # Input: (batch_size, seq_len, num_features)
        weights = torch.softmax(self.attention(x), dim=2)  # (batch_size, seq_len, num_features)
        weighted_features = weights * x  # Element-wise multiplication
        return weighted_features  # (batch_size, seq_len, num_features)