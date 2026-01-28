import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masks import row_causal_mask, column_causal_mask


class AxialAttention(nn.Module):
    def __init__(self, dim, axis, num_heads, masked=False):
        super().__init__()
        assert axis in ["row", "col"]
        self.axis = axis
        self.masked = masked
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, D = x.shape

        if self.axis == "row":
            x = x.view(B * H, W, D)
            causal_mask = row_causal_mask(W, x.device) if self.masked else None
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(B * W, H, D)
            causal_mask = column_causal_mask(H, x.device) if self.masked else None

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], D)
        out = self.proj(out)

        if self.axis == "row":
            out = out.view(B, H, W, D)
        else:
            out = out.view(B, W, H, D).permute(0, 2, 1, 3)

        return out
