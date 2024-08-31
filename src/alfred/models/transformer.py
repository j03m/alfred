import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, features, output_dim=4, model_dim=1024, nhead=8, num_encoder_layers=2):
        super(Transformer, self).__init__()

        # Input embedding layer
        self.embedding = nn.Linear(features, model_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Linear layer to compress transformer output to the required output dimension
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        # Embed the input (linear projection)
        embedded_src = self.embedding(src)

         # Apply transformer encoding with causal masking
        transformer_out = self.transformer_encoder(embedded_src)

        # We take the output of the final step in the sequence
        out = self.fc_out(transformer_out)  # Compress to (batch_size, stocks, output_dim)

        return out


