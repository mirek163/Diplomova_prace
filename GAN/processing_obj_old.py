import trimesh
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import numpy as np

def obj_to_voxel(filepath, grid_size=32, show=False):

    mesh = trimesh.load(filepath, force='mesh')
    # scale za pomocí pythonu (mohu nechat v blenderu default hodnoty)
    bounds = mesh.bounds  #[min, max]

    dimensions = bounds[1] - bounds[0]
    pitch = max(dimensions) / (grid_size - 1)  # -1 protože to zakruhluje nahoru a může to přesáhnout
    #mesh.apply_scale(scale_factor) # neni potřeba, očividně pitch už dělá tenhle scale
    voxels = mesh.voxelized(pitch=pitch)
    voxel_matrix = voxels.matrix
    new_matrix = pad_to_grid(voxel_matrix, grid_size)
    #print(new_matrix)

    # vizualizace
    if show:
        print(f"Voxel dimenze: {voxel_matrix.shape}")
        print(f"Dimenze matice s padding: {new_matrix.shape}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(new_matrix, edgecolor='k')
        plt.show()
    return new_matrix.astype(np.float32)

def pad_to_grid(voxel_matrix, grid_size):
    current_shape = voxel_matrix.shape
    if current_shape[0] > grid_size or current_shape[1] > grid_size or current_shape[2] > grid_size:
        raise ValueError("Velikost matice je větší -> sniž pod 32x32x32")

    #print(current_shape[0])
    pad_x = grid_size - current_shape[0]
    #print(pad_x)
    pad_y = grid_size - current_shape[1]
    pad_z = grid_size - current_shape[2]

    #print(padding)
## test s posunem
    random_x_offset = np.random.randint(0, pad_x+1)
    print(random_x_offset)
    random_y_offset = np.random.randint(0, pad_y+1)
    padding = [
        ((pad_x + random_x_offset)//2, pad_x - (pad_x + random_x_offset)//2),
        ((pad_y + random_y_offset)//2, pad_y - (pad_y + random_y_offset)//2),
        (pad_z // 2, pad_z - pad_z // 2),
    ]

#    padding = [
#        (pad_x // 2, pad_x - pad_x // 2),
#        (pad_y // 2, pad_y - pad_y // 2),
#        (pad_z // 2, pad_z - pad_z // 2),
#    ]
    padded_matrix = np.pad(voxel_matrix, pad_width=padding, mode='constant', constant_values=0)
    return padded_matrix

def voxel_to_obj(voxel_grid, output_filepath):
    # vxytvoř mesh z voxelu
    #voxel_grid[0,0,0] = False
    #print(np.sum(voxel_grid==False))
    voxels = trimesh.voxel.VoxelGrid(voxel_grid)
    #print(np.sum(voxels.matrix==False))
    mesh = voxels.marching_cubes
    mesh.export(output_filepath)

    #--------------------------------------
    #TESTING - NOT USED CURRENTLY FREQUENTLY
    #--------------------------------------
def obj_to_voxel_zoom(filepath, grid_size=32):
    # snaha udelaní linearní interpolace a nastavení objektu na 32x32x32
    # -> později jsem zvolil nynější metodu, kde doplnuju chybějící mezery 0.
    pitch = 1/(grid_size)
    mesh = trimesh.load(filepath, force='mesh')
    voxels = mesh.voxelized(pitch=pitch)
    voxel_matrix = voxels.matrix
    print(f"1.{voxel_matrix.shape}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_matrix, edgecolor='k')
    plt.show()
    # Uprav velikosti
    scale_factors = (
        grid_size / voxel_matrix.shape[0],
        grid_size / voxel_matrix.shape[1],
        grid_size / voxel_matrix.shape[2]
    )
    resized_voxel = zoom(voxel_matrix, scale_factors, order=1, grid_mode=False, cval=0.0, mode='constant')  # linearní interpolace - resample velikosti stejnou velikost voxel (32x32x32)
    print(f"2.{resized_voxel.shape}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(resized_voxel, edgecolor='k')
    plt.show()

    return resized_voxel.astype(np.float32)


def obj_to_voxel_default(filepath, grid_size=32):
    #převedení bez úprav, zde pitch funguje, avšak nedostavam požadovanou velikost pro mou sít
    mesh = trimesh.load(filepath, force='mesh')

    pitch = 1 / (grid_size) # čím menší pitch, tím větší hustota voxelů (64 voxelu za jednotku)
    voxels = mesh.voxelized(pitch=pitch)
    voxel_matrix = voxels.matrix
    print(voxel_matrix.shape)
    return voxel_matrix.astype(np.float32)

if __name__ == "__main__":

    Test = obj_to_voxel(r"..\blender\object\small_buildingA\output\test_only_one\variant_1.obj", show=True)
    voxel_to_obj(Test,r"generated\generated2.obj")
