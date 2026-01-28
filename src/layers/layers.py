import torch
import torch.nn as nn
import torch.nn.functional as F

def shift_right(x):
    B, H, W, D = x.shape
    zero = torch.zeros(B, H, 1, D, device=x.device)
    return torch.cat([zero, x[:, :, :-1, :]], dim=2)


def shift_down(x):
    B, H, W, D = x.shape
    zero = torch.zeros(B, 1, W, D, device=x.device)
    return torch.cat([zero, x[:, :-1, :, :]], dim=1)

class PixelEmbedding(nn.Module):
    def __init__(self, num_embeddings, dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, dim)

    def forward(self, x):
        return self.emb(x)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)
