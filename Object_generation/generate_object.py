import os
import numpy as np
import torch
from matplotlib import pyplot as plt

import processing_obj as prc
from DCGAN import Generator, LATENT_DIM

# Výstupní složka
SAVE_DIR = "generated"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_object(weight_file, output_file):
    # Inicializace generátoru a načtení vah
    generator = Generator(latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(weight_file, map_location=device))
    generator.eval()

    # Náhodný latentní vektor
    z = torch.randn(1, LATENT_DIM).to(device)
    with torch.no_grad():
        generated_voxel = generator(z).cpu().numpy().squeeze()

    # Binarizace voxelové mřížky
    generated_voxel = generated_voxel > 0.5

    # Přidání hraničních bodů
    generated_voxel[0, 0, 0] = True
    generated_voxel[-1, -1, -1] = True

    for i in range(generated_voxel.shape[0]):
        print(f"Vrstva {i}: {np.sum(generated_voxel[i, :, :])}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(generated_voxel, edgecolor='k')
    plt.show()

    prc.voxel_to_obj(generated_voxel, output_file)
    print(f"Objekt uložen na: {output_file}")


if __name__ == "__main__":
    weight_path = "weights/generator_3600-128_64_0.001.pth"
    ITERATIONS = 1
    for i in range(1, ITERATIONS + 1):
        output_file = os.path.join(SAVE_DIR, f"gen{i}.obj")
        generate_object(weight_path, output_file)
