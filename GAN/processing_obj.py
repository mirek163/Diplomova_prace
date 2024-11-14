import trimesh
from scipy.ndimage import zoom
import numpy as np


def obj_to_voxel(filepath, grid_size=32):
    mesh = trimesh.load(filepath, force='mesh')
    voxels = mesh.voxelized(pitch=1 / grid_size)
    voxel_matrix = voxels.matrix

    # Uprav velikosti
    scale_factors = (
        grid_size / voxel_matrix.shape[0],
        grid_size / voxel_matrix.shape[1],
        grid_size / voxel_matrix.shape[2]
    )
    resized_voxel = zoom(voxel_matrix, scale_factors, order=1)  # linearní interpolace - resample velikosti stejnou velikost voxel (32x32x32)

    return resized_voxel.astype(np.float32)

def obj_to_voxel2(filepath, grid_size=16):
    mesh = trimesh.load(filepath, force='mesh')

    pitch = 1 / (grid_size) # čím menší pitch, tím větší hustota voxelů (64 voxelu za jednotku)
    voxels = mesh.voxelized(pitch=pitch)
    voxel_matrix = voxels.matrix
    return voxel_matrix.astype(np.float32)


def voxel_to_obj(voxel_grid, output_filepath):
    # vxytvoř mesh z voxelu
    voxels = trimesh.voxel.VoxelGrid(voxel_grid)
    mesh = voxels.marching_cubes
    mesh.export(output_filepath)

