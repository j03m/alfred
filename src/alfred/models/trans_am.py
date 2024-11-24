# Transformer model taken from https://github.com/ctxj/Time-Series-Transformer-Pytorch.git
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Keep batch dimension as 1
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to have shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)  # Get sequence length from batch-first input
        return x + self.pe[:, :seq_len, :]  # Add positional encodings based on sequence length


class TransAm(nn.Module):
    '''
    Notes on TransAm - feature_size is actually BATCH
    It receives input: 10 (seq_length), 250 (Batch), 1 (features)
    Pos encoding projects

    '''
    def __init__(self, features=1, output=1, model_size=250, heads=10, num_layers=1, dropout=0.1, last_bar=False):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.heads = int(heads)
        self.embed = nn.Linear(features, model_size)
        self.pos_encoder = PositionalEncoding(model_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=self.heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(model_size, output)
        self.last_bar = last_bar # return the last bar or the whole sequence
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            seq_length = int(src.shape[1])
            batch = src.shape[0] * self.heads
            mask = nn.Transformer.generate_square_subsequent_mask(seq_length)
            mask = mask.unsqueeze(0)  # Shape: [1, seq_length, seq_length]
            mask = mask.expand(batch, seq_length, seq_length).to(device)

            self.src_mask = mask
        x = self.embed(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        output = self.decoder(output)
        if self.last_bar:
            return output[:,-1,:]
        else:
            return output

