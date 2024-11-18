import atexit
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import processing_obj as prc
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"zařízení: {device}")
# globální parametry

LATENT_DIM = 100  # latentní dimenze->výstup generátoru
EPOCHS = 10000  # počet epoch pro trénování
BATCH_SIZE = 32  # velikost pro trénování
LR = 0.0002  # rychlost pro generator a diskriminator
DATA_DIRECTORY = r"..\blender\object\small_buildingA\output\window_move"
SAVE_DIR = "weights"  # váhy vygenerované při tréninku
EPOCHS_WEIGHT = 1000
EPOCHS_VISUALIZATION = False

obj_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.obj"))
os.makedirs(SAVE_DIR, exist_ok=True)

if not obj_files:
    raise FileNotFoundError(f"Nenašel se žadny obj file: {DATA_DIRECTORY}")


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
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
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
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR)

    for epoch in range(num_epochs):
        for real_voxels in data_loader:
            # Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(real_voxels.size(0), 1).to(device)
            fake_labels = torch.zeros(real_voxels.size(0), 1).to(device)
            real_voxels = real_voxels.to(device)  # pojištění, že to pojedu přes gpu

            real_loss = criterion(discriminator(real_voxels), real_labels)

            z = torch.randn(real_voxels.size(0), latent_dim).to(device)
            fake_voxels = generator(z)
            fake_loss = criterion(discriminator(fake_voxels.detach()), fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Generator
            # for _ in range(2):  # mam nizky hodnoty u generatoru
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(fake_voxels), real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        if (epoch + 1) % EPOCHS_WEIGHT == 0:
            current_epoch = epoch + 1
            save_weights(version=str(current_epoch))
            print(f"Váhy pro generátor a diskriminátor při epoše : {current_epoch} byly uloženy.")

            # Lehká rychlá vizualizace
            if EPOCHS_VISUALIZATION:
                print(f"Vizualizace mřížky pro epochu {current_epoch}")
                z = torch.randn(1, latent_dim).to(device)
                generated_voxel = generator(
                    z).detach().cpu().numpy().squeeze()  # použití cpu misto gpu - pro použití numpy nezbytný
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.voxels(generated_voxel > 0, edgecolor='k')
                plt.show()


def save_weights(version="latest"):
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_{version}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, f"discriminator_{version}.pth"))
    if version == "latest":
        print("Váhy byly uloženy při ukončení programu.")


if __name__ == "__main__":
    # zobraz objekt z datasetu pyplotu jen pro kontrolu
    voxel_grid = prc.obj_to_voxel(os.path.join(DATA_DIRECTORY, "variant_1.obj"), show=True)

    # fancy bar pro načtení obj souborů.
    print("Převod .obj souborů na voxel mřížku:")
    voxel_data = []
    for filepath in tqdm(obj_files, desc="Načtení .obj souborů"):
        voxel_data.append(prc.obj_to_voxel(filepath))

    # Preproces
    data_loader = torch.utils.data.DataLoader(voxel_data, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(latent_dim=LATENT_DIM, output_dim=32 * 32 * 32).to(device)
    discriminator = Discriminator(input_dim=32 * 32 * 32).to(device)
    #----- ověření, zda se sít trenuje na gpu: -----
    # print(f"generator: {next(generator.parameters()).device}")
    # print(f"diskriminator: {next(discriminator.parameters()).device}")

    #atexit.register(lambda: save_weights())  # uložení vah při "stop" programu , načtení až po modelu ->lambda
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
