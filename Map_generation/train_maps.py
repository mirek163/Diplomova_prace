# ============================================
#  MAP DCGAN
# ============================================
import os, glob, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# --------------------
# GLOBÁLNÍ PARAMETRY
# --------------------
DATA_DIRECTORY       = "dataset"        # složka s *.txt
TXT_GLOB             = "*.txt"          # pattern pro datové soubory
IMG_SIZE             = 32               # velikost kvartálu po splitu
NUM_CLASSES          = 5                # 0: prázdno, 1: budovy, 2: vegetace, 3: voda, 4: silnice
LATENT_DIM           = 256              # dimenze šumu z
BATCH_SIZE           = 128
EPOCHS               = 1000
SAVE_EVERY           = 10               # perioda checkpointu
LR_G                 = 1e-3             # learning-rate generátoru
LR_D                 = 1e-3              # learning-rate diskriminátoru
BETAS                = (0.5, 0.999)
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paleta (index → RGB)
CMAP = {
    0: (1.0, 1.0, 1.0),   # prázdno – bílá
    1: (0.75, 0.0, 0.0),  # budovy – červená
    2: (0.0, 0.5, 0.0),   # vegetace – zelená
    3: (0.0, 0.4, 1.0),   # voda – modrá
    4: (0.4, 0.4, 0.4),   # silnice – šedá
}

# --------------------
#  DATASET
# --------------------
class MapDataset(Dataset):
    """Čte *.txt a vrací tensor (1, H, W) s hodnotami v <0,1>."""
    def __init__(self, root: str, pattern: str = TXT_GLOB):
        self.paths = sorted(pathlib.Path(root).glob(pattern))
        if not self.paths:
            raise RuntimeError(f"V {root} jsem nenašel žádná .txt data.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.loadtxt(self.paths[idx], dtype=np.float32)
        arr /= (NUM_CLASSES - 1)              # normalizace
        return torch.from_numpy(arr).unsqueeze(0)

# --------------------
#  GENERÁTOR
# --------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(True),

            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

# --------------------
#  DISKRIMINÁTOR
# --------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --------------------
#  UTILITKY
# --------------------
@torch.no_grad()
def quantize(x):
    """Převede <0,1> mapu na int hodnoty 0–4."""
    return torch.round(x * (NUM_CLASSES - 1)).long().squeeze(1)

@torch.no_grad()
def plot_mask(mask, ax):
    """Vykreslí 2D masku pomocí CMAP (0–4) na jeden ax."""
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=float)
    for cls, col in CMAP.items():
        img[mask==cls] = col
    ax.imshow(img)
    ax.axis('off')

# --------------------
#  TRÉNINK
# --------------------
def train():
    # loader
    ds = MapDataset(DATA_DIRECTORY)
    dl = DataLoader(ds, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=4, pin_memory=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

    for epoch in range(1, EPOCHS + 1):
        for real in dl:
            real = real.to(DEVICE)
            b = real.size(0)

            valid = torch.ones(b, 1, device=DEVICE)
            fake_lbl = torch.zeros(b, 1, device=DEVICE)

            # --- Discriminator ---
            z = torch.randn(b, LATENT_DIM, device=DEVICE)
            fake = G(z).detach()

            loss_D = criterion(D(real), valid) + criterion(D(fake), fake_lbl)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # --- Generator ---
            for _ in range(2):
                z = torch.randn(b, LATENT_DIM, device=DEVICE)
                gen = G(z)
                loss_G = criterion(D(gen), valid)
                opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        print(f"Epoch {epoch:4d}/{EPOCHS} │ D: {loss_D.item():.3f} │ G: {loss_G.item():.3f}")

        # checkpoint + vizualizace
        if epoch % SAVE_EVERY == 0:
            os.makedirs("weights", exist_ok=True)
            torch.save(G.state_dict(),
                       f"weights/generator_{epoch:05d}-{LATENT_DIM}_{BATCH_SIZE}_{LR_G}.pth")
            with torch.no_grad():
                z = torch.randn(16, LATENT_DIM, device=DEVICE)
                maps = quantize(G(z).cpu()).numpy()  # tvar [16, H, W]

            # vykreslení 4×4 mřížky
            fig, axes = plt.subplots(2, 4, figsize=(8, 4))
            axes = axes.flatten()
            for i, mask in enumerate(maps[:8]):  # jen prvních 8
                plot_mask(mask, axes[i])
            fig.suptitle(f"Epocha {epoch}", fontsize=16)
            os.makedirs("vizualizace", exist_ok=True)
            fig.savefig(f"vizualizace/epoch_{epoch:05d}.png", bbox_inches='tight')
            plt.close(fig)

    print("Konec tréninku – poslední váhy uloženy.")
    torch.save(G.state_dict(),
               f"weights/generator_latest-{LATENT_DIM}_{BATCH_SIZE}_{LR_G}.pth")
    torch.save(D.state_dict(),
               f"weights/discriminator_latest-{LATENT_DIM}_{BATCH_SIZE}_{LR_D}.pth")

# --------------------
#  MAIN
# --------------------
if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f" Během tréninku se to sesypalo: {e}")
