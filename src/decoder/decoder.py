import torch
import torch.nn as nn

from layers.layers import shift_down, shift_right
from blocks.blocks import TransformerBlock


class AxialDecoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ff_hidden_dim,
        num_upper_layers,
        num_row_layers,
        H,
        W
    ):
        super().__init__()

        self.upper_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                axis="col",
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                masked=False
            )
            for _ in range(num_upper_layers)
        ])

        self.row_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                axis="row",
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                masked=True
            )
            for _ in range(num_row_layers)
        ])

        self.row_pos = nn.Parameter(torch.randn(1, H, 1, dim))
        self.col_pos = nn.Parameter(torch.randn(1, 1, W, dim))

        self.output = nn.Linear(dim, dim)

    def forward(self, x):

        u = shift_down(x)
        for block in self.upper_blocks:
            u = block(u)

        h = shift_right(x)
        for block in self.row_blocks:
            h = block(h)

        h = h + u + self.row_pos + self.col_pos

        return self.output(h)
