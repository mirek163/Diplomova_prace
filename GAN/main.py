import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh

class OBJDataset(Dataset):
    def __init__(self, obj_folder):
        self.obj_files = [os.path.join(obj_folder, f) for f in os.listdir(obj_folder) if f.endswith('.obj')]

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        # Load .obj file
        mesh = trimesh.load(self.obj_files[idx])
        vertices = mesh.vertices

        # Normalize vertex data
        vertices = (vertices - np.mean(vertices, axis=0)) / np.max(np.abs(vertices))
        return torch.tensor(vertices, dtype=torch.float32)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(z_dim, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(generator, discriminator, data_loader, epochs, z_dim, device):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images in data_loader:
            real_images = real_images.unsqueeze(1).to(device)  # Přidání dimenze pro kanály
            batch_size = real_images.size(0)

            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Trénink diskriminátoru
            optimizer_d.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, z_dim, 1, 1, 1).to(device)  # Vytvoření šumu pro generátor
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Trénink generátoru
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

if __name__ == "__main__":
    obj_folder = 'path/to/your/obj/files'  # Update path for .obj files
    dataset = OBJDataset(obj_folder)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    z_dim = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator(z_dim).to(device)
    discriminator = Discriminator().to(device)

    train_gan(generator, discriminator, data_loader, epochs=100, z_dim=z_dim, device=device)
