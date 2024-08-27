import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import sys

sys.path.append('Transformer/')
from .stockformer_layer import EncoderLayer, Encoder
from .attn import FullAttention, AttentionLayer
from .embed import DataEmbedding

class Stockformer(nn.Module):
    def __init__(self, enc_in, c_out,
                d_model=128, n_heads=4, e_layers=2,
                dropout=0.0, activation='gelu', output_attention=False):
        super(Stockformer, self).__init__()

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        d_ff = d_model * 2
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection_decoder = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        output = self.projection_decoder(enc_out)
        return output

