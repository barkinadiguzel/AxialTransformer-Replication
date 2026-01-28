import matplotlib.pyplot as plt
import numpy as np


def visualize_mask(mask, title="Causal Mask"):
    mask_np = mask.detach().cpu().numpy()
    plt.imshow(mask_np, cmap="gray")
    plt.title(title)
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    plt.colorbar()
    plt.show()

