import torch


def row_causal_mask(W, device=None):
    mask = torch.tril(torch.ones(W, W, device=device))
    return mask


def column_causal_mask(H, device=None):
    mask = torch.tril(torch.ones(H, H, device=device))
    return mask


def axial_causal_masks(H, W, device=None):
    row_mask = row_causal_mask(W, device)
    col_mask = column_causal_mask(H, device)
    return row_mask, col_mask
