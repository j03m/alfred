import torch.nn as nn


# feed a seq_len as 30 features
class SeriesAsLinear(nn.Module):
    def __init__(self, seq_len, hidden_dim, output_size):
        super().__init__()
        self.expand = nn.Linear(seq_len, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim, output_size)

    def forward(self, input_seq):
        input_seq = input_seq.view(input_seq.size(0), -1)
        x = self.expand(input_seq)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.final(x)