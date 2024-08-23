import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim=4, model_dim=256, nhead=8, num_encoder_layers=6):
        super(Transformer, self).__init__()
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, model_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Linear layer to compress transformer output to the required output dimension
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        # src shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_length, _ = src.shape

        # Embed the input (linear projection)
        embedded_src = self.embedding(src)

        # Generate a causal mask (upper triangular)
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=src.device)).unsqueeze(0)

        # Expand the mask to match the number of heads and batch size
        # Shape should be (batch_size * num_heads, sequence_length, sequence_length)
        causal_mask = causal_mask.expand(batch_size * self.transformer_encoder.layers[0].self_attn.num_heads, -1, -1)

        # Apply transformer encoding with causal masking
        transformer_out = self.transformer_encoder(embedded_src, mask=causal_mask)

        # We take the output of the final step in the sequence
        out = self.fc_out(transformer_out)  # Compress to (batch_size, stocks, output_dim)

        return out


