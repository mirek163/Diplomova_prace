import os
import numpy as np
import torch
from matplotlib import pyplot as plt

import processing_obj as prc
from gan import Conv3DGenerator, LATENT_DIM

LATENT_DIM = 100
SLICE_SIZE = 32  # Slices size during generation
OVERLAP = 8      # Overlap between slices
ORIGINAL_SIZE = 64  # Target voxel grid size
SAVE_DIR = "generated"
os.makedirs(SAVE_DIR, exist_ok=True)


def generate_slices(generator, num_slices, slice_size=SLICE_SIZE):
    """Generate slices from the generator."""
    slices = []
    for _ in range(num_slices):
        z = torch.randn(1, LATENT_DIM)
        generated_slice = generator(z).detach().numpy().squeeze()
        generated_slice[generated_slice < 0.5] = 0  # Thresholding
        slices.append(generated_slice)
    return slices


def generate_full_voxel_grid(weight_file, output_file):
    """Generate a full voxel grid by combining slices."""
    # Load generator and weights
    generator = Conv3DGenerator(latent_dim=LATENT_DIM).eval()
    generator.load_state_dict(torch.load(weight_file))

    # Calculate number of slices for the full grid
    num_slices = ((ORIGINAL_SIZE - SLICE_SIZE) // (SLICE_SIZE - OVERLAP) + 1) ** 3
    print(f"Generating {num_slices} slices...")

    # Generate slices
    slices = generate_slices(generator, num_slices, slice_size=SLICE_SIZE)

    # Reconstruct full voxel grid
    full_voxel_grid = prc.reconstruct_from_slices(
        slices, original_size=ORIGINAL_SIZE, slice_size=SLICE_SIZE, overlap=OVERLAP
    )

    # Visualize the generated grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(full_voxel_grid, edgecolor='k')
    plt.show()

    # Save to OBJ
    prc.voxel_to_obj(full_voxel_grid, output_file)
    print(f"Generated object saved at: {output_file}")


if __name__ == "__main__":
    weight_path = "weights/generator_1000-100_32_0.0002.pth"  # Path to weights
    output_file = os.path.join(SAVE_DIR, "generated.obj")
    generate_full_voxel_grid(weight_path, output_file)
