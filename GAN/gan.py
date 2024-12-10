import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import processing_obj as prc
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------
# GLOBAL PARAMETERS
# --------------------
LATENT_DIM = 100  # latentní dimenze -> výstup generátoru
EPOCHS = 100  # počet epoch pro trénování
BATCH_SIZE = 64  # velikost pro trénování
LR = 0.0002  # rychlost pro generátor a diskriminátor
DATA_DIRECTORY = r"..\blender\object\small_buildingA\output\window_move"
SAVE_DIR = "weights"  # váhy vygenerované při tréninku
EPOCHS_WEIGHT = 10
EPOCHS_VISUALIZATION = True

obj_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.obj"))
os.makedirs(SAVE_DIR, exist_ok=True)

if not obj_files:
    raise FileNotFoundError(f"Nenašel se žádný obj soubor: {DATA_DIRECTORY}")


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 32, 32, 32)  # úpravy do voxel mřížky


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, voxel):
        return self.model(voxel.view(voxel.size(0), -1))


def train_gan(generator, discriminator, data_loader, num_epochs=100, latent_dim=100):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=LR)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR * 0.5)

    for epoch in range(num_epochs):
        for real_voxels in tqdm(data_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):
            # Train Discriminator
            optimizer_d.zero_grad()
            real_voxels = real_voxels.to(device)
            real_labels = torch.ones(real_voxels.size(0), 1).to(device)
            fake_labels = torch.zeros(real_voxels.size(0), 1).to(device)

            real_loss = criterion(discriminator(real_voxels), real_labels)
            z = torch.randn(real_voxels.size(0), latent_dim).to(device)
            fake_voxels = generator(z)
            fake_loss = criterion(discriminator(fake_voxels.detach()), fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(fake_voxels), real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        if (epoch + 1) % EPOCHS_WEIGHT == 0:
            save_weights(generator, discriminator, version=str(epoch + 1))
            if EPOCHS_VISUALIZATION:
                visualize(generator, latent_dim, epoch + 1)


def save_weights(generator, discriminator, version="latest"):
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_{version}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f"discriminator_{version}.pth"))
    print(f"Weights saved for epoch {version}")


def visualize(generator, latent_dim, epoch):
    z = torch.randn(1, latent_dim).to(device)
    generated_voxel = generator(z).detach().cpu().numpy().squeeze()

    generated_voxel[generated_voxel < 0.5] = 0
    #generated_voxel[generated_voxel >= 0.5] = 1

    if np.sum(generated_voxel) == 0:
        print(f"Generovaná voxel mřížka je nulová na epoše:  {epoch}")
    else:
        print(f"Suma na Voxel mřížce pro epochu {epoch}: {np.sum(generated_voxel)}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(generated_voxel, edgecolor='k')
    plt.title(f"Voxel mřížka (Epocha: {epoch})")
    plt.show()


if __name__ == "__main__":
    # testovací
    voxel_grid = prc.obj_to_voxel(os.path.join(DATA_DIRECTORY, "variant_1.obj"), show=True)

    # Načtení všech .obj souborů a jejich převod na voxel mřížky
    voxel_data = []
    for filepath in tqdm(obj_files, desc="Načítání .obj souborů"):
        try:
            voxel = prc.obj_to_voxel(filepath, grid_size=64)
            slices = prc.slice_voxel_grid(voxel, slice_size=32, overlap=8)
            voxel_data.extend(slices)
        except Exception as e:
            print(f"Chyba při zpracování souboru {filepath}: {e}")

    if not voxel_data:
        raise ValueError("Žádné voxelové mřížky nebyly úspěšně načteny.")

    # Konverze do PyTorch tensoru
    dataset = torch.tensor(np.array(voxel_data)).unsqueeze(1).float()
    print(f"Dataset Shape: {dataset.shape}")  # Očekávaný tvar: (počet_sliců, 1, 32, 32, 32)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Inicializace modelů
    generator = Generator(latent_dim=LATENT_DIM, output_dim=32 * 32 * 32).to(device)
    discriminator = Discriminator(input_dim=32 * 32 * 32).to(device)

    # Trénování GAN
    train_gan(generator, discriminator, data_loader, num_epochs=EPOCHS, latent_dim=LATENT_DIM)
