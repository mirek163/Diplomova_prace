"""
3D WGAN-GP na voxelových datech zpracované z OBJ souborů
===================================================================
Potřeba:
  – Složka s tréninkovými .obj soubory: DATA_DIRECTORY (např. "dataset/BuildingB-rotation_and_floors")
  – Závislosti: processing_obj.py

===================================================================
Zpracování dat:
  – Načtou se všechny .obj soubory a převedou se na voxel grid 64x64x64
  – První běh vytváří cache soubor "voxel_cache.pt" pro zrychlení opakovaného tréninku

GAN Architektura:
  – Wasserstein GAN s Gradient Penalty
  – Discriminator:
      – Používá spectral normalization a Dropout3D
      – Trénován několikanásobně častěji než generátor (CRITIC_ITERS = 5)
  – Generator:
      – Latentní vektor je rozvinut na mřížku 64³ pomocí ConvTranspose3D
      – Výstup: voxel grid s hodnotami v rozsahu <-1, 1> (Tanh)

Loss funkce:
  – Discriminator: Wasserstein loss + LAMBDA·GradientPenalty (LAMBDA = 20)
  – Generator: záporné skóre z D (snaží se zvýšit výstup diskriminátoru)

===================================================================
Trénink:
  – Optimizer: Adam (betas = 0.5, 0.9)
  – LR_G a LR_D = 1e-3 (nastaveno rychlejší konvergenci)
  – Inicializace vah pomocí N(0, 0.02)

===================================================================
  – Každých EPOCHS_WEIGHT epoch:
      – Uloží váhy generátoru a diskriminátoru do složky SAVE_DIR/
      – Vykreslí a uloží náhled voxel objektu do složky "vizualizace/"
  – Po skončení:
      – Vytvoří jeden nový objekt a uloží ho jako OBJ soubor přes voxel_to_obj()

===================================================================
Globální parametry pro zvolený trénink u diplomové práce:
  LATENT_DIM = 128
  EPOCHS = 10000
  BATCH_SIZE = 64
  CRITIC_ITERS = 5
  LAMBDA_GP = 20
  EPOCHS_WEIGHT = 100
===================================================================
Výstupy:
  – Váhy:    weights/generator_{epoch}-128_64_{LR_G}.pth
             weights/discriminator_{epoch}-128_64_{LR_D}.pth
  – Vizualizace: visualization/epoch_000XX.png
  – Výsledný 3D objekt: generated_object.obj
===================================================================
"""
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import processing_obj as prc
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.utils import spectral_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------------
# GLOBÁLNÍ PARAMETRY
# --------------------

LATENT_DIM = 128  # latentní dimenze->výstup generátoru
EPOCHS = 10000
BATCH_SIZE = 64  # velikost dávky
LR_G = 1e-3
LR_D = 1e-3
betas = (0.5, 0.9)
DATA_DIRECTORY = r".\dataset\BuildingB-rotation_and_floors"
CACHE_PATH = os.path.join(DATA_DIRECTORY, "voxel_cache.pt")

SAVE_DIR = "weights"  # složka pro uložení vah
EPOCHS_WEIGHT = 100
EPOCHS_VISUALIZATION = True
CRITIC_ITERS=5
LAMBDA_GP=20

obj_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.obj"))
os.makedirs(SAVE_DIR, exist_ok=True)

if not obj_files:
    raise FileNotFoundError(f"Nenašel se žadny obj file: {DATA_DIRECTORY}")


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=None):
        super().__init__()
        self.init_channels = 512                    # počáteční počet kanálů
        self.init_size = 4                          # mřížka 4x4x4 po projekci

        # projekce latentního vektoru na ten “seed” 4×4×4×512
        self.project = nn.Linear(latent_dim,
                                 self.init_channels * self.init_size ** 3)

        # postupné zdvojování rozměrů pomocí ConvTranspose3d
        self.net = nn.Sequential(
            nn.BatchNorm3d(self.init_channels),
            nn.ReLU(True),

            nn.ConvTranspose3d(self.init_channels, self.init_channels // 2,
                               kernel_size=4, stride=2, padding=1),  # 8³
            nn.BatchNorm3d(self.init_channels // 2),
            nn.ReLU(True),

            nn.ConvTranspose3d(self.init_channels // 2, self.init_channels // 4,
                               kernel_size=4, stride=2, padding=1),  # 16³
            nn.BatchNorm3d(self.init_channels // 4),
            nn.ReLU(True),

            nn.ConvTranspose3d(self.init_channels // 4, self.init_channels // 8,
                               kernel_size=4, stride=2, padding=1),  # 32³
            nn.BatchNorm3d(self.init_channels // 8),
            nn.ReLU(True),

            nn.ConvTranspose3d(self.init_channels // 8, 1,
                               kernel_size=4, stride=2, padding=1),  # 64³
            nn.Tanh()
        )

    def forward(self, z):
        x = self.project(z) \
               .view(-1, self.init_channels,
                     self.init_size, self.init_size, self.init_size)
        x = self.net(x)          # (B, 1, 64, 64, 64)
        return x.squeeze(1)      # (B, 64, 64, 64)


class Discriminator(nn.Module):
    def __init__(self, input_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1)),   # 32³
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),

            spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)), # 16³
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.3),

            spectral_norm(nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)),# 8³
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)),# 4³
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0)),  # 1³
            #nn.Sigmoid()
        )

    def forward(self, voxel):
        x = voxel.unsqueeze(1)
        return self.net(x).view(-1, 1)


def gradient_penalty(D, real, fake):
    batch_size = real.size(0)

    eps_shape = (batch_size,) + (1,) * (real.dim() - 1)
    ε = torch.rand(eps_shape, device=real.device).expand_as(real)

    interpolated = (ε * real + (1 - ε) * fake).requires_grad_(True)
    d_interpolated = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_gan(generator, discriminator, data_loader,
              num_epochs=EPOCHS_WEIGHT, latent_dim=LATENT_DIM, LAMBDA_gp=LAMBDA_GP, critic_iters=CRITIC_ITERS):

    optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=betas)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=betas)

    for epoch in range(num_epochs):
        for real_voxels in data_loader:
            real_voxels = real_voxels.to(device)


            for _ in range(critic_iters):
                z = torch.randn(real_voxels.size(0), latent_dim, device=device)
                fake_voxels = generator(z).detach()

                optimizer_d.zero_grad()

                real_score = discriminator(real_voxels).mean()
                fake_score = discriminator(fake_voxels).mean()

                gp = gradient_penalty(discriminator, real_voxels, fake_voxels)
                d_loss = fake_score - real_score + LAMBDA_gp * gp

                d_loss.backward()
                optimizer_d.step()

            z = torch.randn(real_voxels.size(0), latent_dim, device=device)
            fake_voxels = generator(z)

            optimizer_g.zero_grad()
            g_loss = -discriminator(fake_voxels).mean()
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

        if (epoch + 1) % EPOCHS_WEIGHT == 0:
            current_epoch = epoch + 1
            save_weights(version=str(current_epoch))
            print(f"Váhy pro generátor a diskriminátor při epoše {current_epoch} byly uloženy.")

            if EPOCHS_VISUALIZATION:
                print(f"Vizualizace mřížky pro epochu {current_epoch}")

                generator.eval()
                with torch.no_grad():
                    z = torch.randn(1, latent_dim, device=device)
                    vox = generator(z).cpu().numpy().squeeze()
                generator.train()

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.voxels(vox > 0.5, edgecolor='k')

                save_dir = "visualization"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"epoch_{current_epoch:05d}.png")
                plt.savefig(save_path, bbox_inches="tight")
                print(f"Obrázek uložen do {save_path}")
                plt.show()
                plt.close(fig)


def save_weights(version="latest"):
    torch.save(generator.state_dict(),
               os.path.join(SAVE_DIR, f"generator_{version}-{LATENT_DIM}_{BATCH_SIZE}_{LR_G}.pth"))
    torch.save(discriminator.state_dict(),
               os.path.join(SAVE_DIR, f"discriminator_{version}-{LATENT_DIM}_{BATCH_SIZE}_{LR_D}.pth"))
    if version == "latest":
        print("Váhy byly uloženy při ukončení programu.")


if __name__ == "__main__":
    #  nahraj z cache
    if os.path.exists(CACHE_PATH):
        print("Načítám voxel data z cache...")
        voxel_data = torch.load(CACHE_PATH, weights_only=False)
    else:
        print("Převádím .obj soubory na voxel mřížku (poprvé)...")
        voxel_data = []
        for filepath in tqdm(obj_files, desc="Načtení .obj souborů"):
            voxel_data.append(prc.obj_to_voxel(filepath ,show=True))
        # cache pro příště
        torch.save(voxel_data, CACHE_PATH)

    data_loader = torch.utils.data.DataLoader(voxel_data, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(latent_dim=LATENT_DIM, output_dim=64 * 64 * 64).to(device)
    discriminator = Discriminator(input_dim=64 * 64 * 64).to(device)

# váhy neodstartujou na divných hodnotách
    def init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    # ----- ověření, zda se sít trenuje na gpu: -----
    #print(f"zařízení: {device}")
    # print(f"generator: {next(generator.parameters()).device}")
    # print(f"diskriminator: {next(discriminator.parameters()).device}")
    try:
        train_gan(generator, discriminator, data_loader, num_epochs=EPOCHS, latent_dim=LATENT_DIM)
    except Exception as e:
        print(f"\nDošlo k chybě během trénování: {e}")
    finally:
        save_weights()

    # Generuj objekt
    z = torch.randn(1, LATENT_DIM).to(device)
    generated_voxel = generator(z).detach().cpu().numpy().squeeze()
    prc.voxel_to_obj(generated_voxel, "generated_object.obj")