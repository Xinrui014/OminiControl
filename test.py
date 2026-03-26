import numpy as np
import matplotlib.pyplot as plt

def plot_naive_mask(n_text, n_image):
    total_tokens = n_text + n_image
    mask = np.ones((total_tokens, total_tokens), dtype=int)

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(mask, cmap='Greens')
    ax.set_xticks(range(total_tokens))
    ax.set_yticks(range(total_tokens))
    ax.set_xticklabels(
        [f"T{i}" for i in range(n_text)] +
        [f"I{j}" for j in range(n_image)],
        rotation=90)
    ax.set_yticklabels(
        [f"T{i}" for i in range(n_text)] +
        [f"I{j}" for j in range(n_image)])
    ax.set_title("Naive (No-Bridge) Attention Mask — All Tokens Attend")
    fig.colorbar(cax, orientation='vertical', label='Allowed Attn=1')
    plt.tight_layout()
    plt.show()

# Example usage
plot_naive_mask(n_text=3, n_image=4)