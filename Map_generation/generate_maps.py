import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from train_maps import Generator, quantize

# --------------------
# GLOBÁLNÍ PARAMETRY
# --------------------
LATENT_DIM = 256
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "weights/generator_latest-256_128_0.001.pth"
OUTPUT_DIR = "output_maps"
NUM_MAPS = 8  # počet generovaných map
DATASET_DIR = "dataset" # reálné data

CMAP = {
    0: (1.0, 1.0, 1.0),  # prázdno – bílá
    1: (0.75, 0.0, 0.0),  # budovy – červená
    2: (0.0, 0.5, 0.0),  # vegetace – zelená
    3: (0.0, 0.4, 1.0),  # voda – modrá
    4: (0.4, 0.4, 0.4),  # silnice – šedá
}


@torch.no_grad()
def plot_mask(mask, ax):
    """Vykreslí 2D masku pomocí CMAP (0–4) na jeden ax."""
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=float)
    for cls, col in CMAP.items():
        img[mask == cls] = col
    ax.imshow(img)
    ax.axis('off')

# --------------------
# NAČTENÍ GENERÁTORU
# --------------------
G = Generator().to(DEVICE)
G.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
G.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# GENEROVÁNÍ
# --------------------
with torch.no_grad():
    z = torch.randn(NUM_MAPS, LATENT_DIM, device=DEVICE)
    generated = G(z)  # (N, 1, H, W), hodnoty 0–1
    quantized = quantize(generated).cpu().numpy()  # (N, H, W), int 0–4

    # Uložení jako .txt + vykreslení
    for i, mask in enumerate(quantized):
        np.savetxt(f"{OUTPUT_DIR}/map_{i:03d}.txt", mask, fmt="%d")

    # Grid vizualizace
    # n = int(np.ceil(np.sqrt(NUM_MAPS)))
    # fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    # axes = axes.flatten()
    # for i in range(NUM_MAPS):
    #     plot_mask(quantized[i], axes[i])
    # for j in range(NUM_MAPS, len(axes)):
    #     axes[j].axis('off')

# vygenerované mapy
    n = int(np.ceil(np.sqrt(NUM_MAPS)))
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    axes = axes.flatten()
    for i in range(NUM_MAPS):
        plot_mask(quantized[i], axes[i])
    for j in range(NUM_MAPS, len(axes)):
        axes[j].axis('off')

    fig.suptitle("Generované mapy", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/preview_grid.png", bbox_inches='tight')
    plt.close(fig)

# dataset - realné mapy
    real_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".txt")])
    assert len(real_files) >= NUM_MAPS, "V datasetu není dostatek reálných map."

    real_selected = np.random.choice(real_files, NUM_MAPS, replace=False)
    real_maps = [np.loadtxt(os.path.join(DATASET_DIR, fname), dtype=int) for fname in real_selected]

    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    axes = axes.flatten()
    for i in range(NUM_MAPS):
        plot_mask(real_maps[i], axes[i])
        axes[i].set_title(real_selected[i].replace(".txt", ""), fontsize=6)

    for j in range(NUM_MAPS, len(axes)):
        axes[j].axis('off')

    fig.suptitle("Reálné mapy z datasetu", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/real_grid.png", bbox_inches='tight')
    plt.close(fig)

print(f"  {NUM_MAPS} map uloženo jako .txt a preview.")
