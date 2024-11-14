import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import processing_obj as prc
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Snaha pro využití grafiky

#načtení objektů
data_directory = r"C:\Users\Lenovo\Desktop\Diplomova_prace\blender\object\small_buildingA\output\window_move"
#data_directory = r"C:\Users\Lenovo\Desktop\Diplomova_prace\blender\object\small_buildingA\output\test3"
obj_files = glob.glob(os.path.join(data_directory, "*.obj"))

if not obj_files:
    raise FileNotFoundError(f"No .obj files found in directory: {data_directory}")


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 32, 32, 32)  # úpravy do voxel mřížky


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, voxel):
        return self.model(voxel.view(voxel.size(0), -1))

def train_gan(generator, discriminator, data_loader, num_epochs=100, latent_dim=100):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for real_voxels in data_loader:
            # Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(real_voxels.size(0), 1)
            fake_labels = torch.zeros(real_voxels.size(0), 1)

            real_loss = criterion(discriminator(real_voxels), real_labels)

            z = torch.randn(real_voxels.size(0), latent_dim)
            fake_voxels = generator(z)
            fake_loss = criterion(discriminator(fake_voxels.detach()), fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Generator
            #for _ in range(2):  # mam nizky hodnoty u generatoru
            optimizer_g.zero_grad()
            # Generate new fake data for each iteration (new computation graph)
            #z = torch.randn(real_voxels.size(0), latent_dim).to(device)
            #fake_voxels = generator(z)
            g_loss = criterion(discriminator(fake_voxels), real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # Lehká rychlá vizualizace
        if (epoch + 1) % 10000 == 0:
            print(f"Visualizing voxel grid at epoch {epoch + 1}")
            z = torch.randn(1, latent_dim)
            generated_voxel = generator(z).detach().numpy().squeeze()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.voxels(generated_voxel > 0, edgecolor='k')
            plt.show()

if __name__ == "__main__":
    # zobraz objekt pyplotu jen pro kontrolu
    voxel_grid = prc.obj_to_voxel2("C:\\Users\\Lenovo\\Desktop\\Diplomova_prace\\blender\\object\\small_buildingA"
                                  "\\output\\window_move\\variant_1.obj")
    #voxel_grid=prc.obj_to_voxel2("C:\\Users\\Lenovo\\Desktop\\Diplomova_prace\\blender\\object\\small_buildingA\\output\\test3\\0_032.obj")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, edgecolor='k')
    plt.show()

    # Preprocess
    voxel_data = [prc.obj_to_voxel(filepath) for filepath in obj_files]
    data_loader = torch.utils.data.DataLoader(voxel_data, batch_size=32, shuffle=True)

    latent_dim = 100
    generator = Generator(latent_dim=latent_dim, output_dim=32 * 32 * 32)
    discriminator = Discriminator(input_dim=32 * 32 * 32)

    train_gan(generator, discriminator, data_loader, num_epochs=10000, latent_dim=latent_dim)

    # Generuj objekt
    z = torch.randn(1, latent_dim)
    generated_voxel = generator(z).detach().numpy().squeeze()
    prc.voxel_to_obj(generated_voxel, "generated_object.obj")
