import os
import torch
import processing_obj as prc
from main import Generator, LATENT_DIM

#LATENT_DIM=100
SAVE_DIR = "obj_generated"
os.makedirs(SAVE_DIR, exist_ok=True)
def generate_object(weight_file, output_file=os.path.join(SAVE_DIR, "generated_obj.obj")):
    generator = Generator(latent_dim=LATENT_DIM, output_dim=32 * 32 * 32)
    generator.load_state_dict(torch.load(weight_file))
    generator.eval()

    z = torch.randn(1, LATENT_DIM)
    generated_voxel = generator(z).detach().numpy().squeeze()

    prc.voxel_to_obj(generated_voxel, output_file)
    print(f"Objekt uložen na adrese: {output_file}")

if __name__ == "__main__":
    weight_path = "weights/generator_10.pth"
    generate_object(weight_path)
