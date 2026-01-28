import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import PixelEmbedding
from decoder.decoder import AxialDecoder


class AxialTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.H = config.H
        self.W = config.W

        self.embedding = PixelEmbedding(
            config.vocab_size,
            config.dim
        )

        self.decoder = AxialDecoder(
            dim=config.dim,
            num_heads=config.num_heads,
            ff_hidden_dim=config.ff_hidden_dim,
            num_upper_layers=config.num_upper_layers,
            num_row_layers=config.num_row_layers,
            H=config.H,
            W=config.W
        )

        self.to_logits = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)     
        h = self.decoder(x)
        logits = self.to_logits(h)
        return logits
