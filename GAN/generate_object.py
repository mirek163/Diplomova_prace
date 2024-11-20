import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import processing_obj as prc
from gan import Generator, LATENT_DIM

LATENT_DIM = 200
SAVE_DIR = "generated"
os.makedirs(SAVE_DIR, exist_ok=True)


def generate_object(weight_file, output_file=os.path.join(SAVE_DIR, "generated_obj.obj")):
    generator = Generator(latent_dim=LATENT_DIM, output_dim=32 * 32 * 32)
    generator.load_state_dict(torch.load(weight_file))
    generator.eval()

    z = torch.randn(1, LATENT_DIM)
    generated_voxel = generator(z).detach().numpy().squeeze()
    #print(generated_voxel)
    #generated_voxel[0,0,0]=False
    generated_voxel[generated_voxel<0.5]=0
    #print(np.sum(generated_voxel>0))
    #print(np.sum(generated_voxel==0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(generated_voxel, edgecolor='k')
    plt.show()

    prc.voxel_to_obj(generated_voxel, output_file)
    print(f"Objekt uložen na adrese: {output_file}")


if __name__ == "__main__":
    weight_path = "weights/generator_1000-200_64_0.0002.pth"
    generate_object(weight_path)
