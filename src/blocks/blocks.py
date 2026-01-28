import torch
import torch.nn as nn

from attention.attention import AxialAttention
from layers.layers import FeedForward

class AttentionBlock(nn.Module):
    def __init__(self, dim, axis, num_heads, masked=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = AxialAttention(
            dim=dim,
            axis=axis,
            num_heads=num_heads,
            masked=masked
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, axis, num_heads, ff_hidden_dim, masked=False):
        super().__init__()

        self.attn_block = AttentionBlock(
            dim=dim,
            axis=axis,
            num_heads=num_heads,
            masked=masked
        )

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidden_dim)

    def forward(self, x):
        x = self.attn_block(x)
        x = x + self.ff(self.ff_norm(x))
        return x
