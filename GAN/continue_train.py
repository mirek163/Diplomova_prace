import glob
import torch
import main as mn
import processing_obj as prc

#LATENT_DIM = 100
#BATCH_SIZE = 32
ADD_EPOCHS = 5000  # Další epochy na trénování

def continue_training(generator_weight, discriminator_weight, data_loader, num_epochs=ADD_EPOCHS):
    generator = mn.Generator(latent_dim=mn.LATENT_DIM, output_dim=32 * 32 * 32).to(mn.device)
    discriminator = mn.Discriminator(input_dim=32 * 32 * 32).to(mn.device)

    generator.load_state_dict(torch.load(generator_weight))
    discriminator.load_state_dict(torch.load(discriminator_weight))

    mn.train_gan(generator, discriminator, data_loader, num_epochs=num_epochs, latent_dim=mn.LATENT_DIM)

if __name__ == "__main__":
    generator_path = "weights/generator_10.pth"
    discriminator_path = "weights/discriminator_10.pth"
    obj_files = glob.glob(r"..\blender\object\small_buildingA\output\window_move\*.obj")
    voxel_data = []
    for filepath in mn.tqdm(obj_files, desc="Načtení .obj souborů"):
        voxel_data.append(prc.obj_to_voxel(filepath))

    data_loader = torch.utils.data.DataLoader(voxel_data, batch_size=mn.BATCH_SIZE, shuffle=True)

    continue_training(generator_path, discriminator_path, data_loader)
