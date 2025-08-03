"""
Konverze 3D OBJ modelu na voxel grid (a naopak) + padding
================================================================================
Potřeba:
  - Vstupní .obj soubor (filepath). -> vse volany ve WGAN

Funkce:
  obj_to_voxel(...):
      – Načte OBJ jako mesh (trimesh).
      – Spočítá voxelizaci podle největšího rozměru (pitch = max_dim/(grid_size-1)).
      – Volitelně vyplní vnitřek (fill_internal).
      – Změří (zoom) a automaticky "padne" do přesné velikosti grid_size
            (pad_to_grid, viz níže) s náhodným offsetem v X/Y pro augmentaci.
      – Pokud show=True, vytiskne info o surovém mesh_i, rozměrech a zobrazí 3D voxel preview v matplotlibu.
      – Vrací voxelovou matici tvaru (grid_size,grid_size,grid_size).

  pad_to_grid(voxel_matrix, grid_size):
      – Pokud je vstup větší než cílový grid, změní velikost poměrem (nejbližší
        soused, order=0) bez zkreslení proporcí.
      – Vypočte, o kolik je menší, a obalí nulami (np.pad) s náhodným X/Y posunem (augmentace) a středovým Z-paddingem.
            (náhodný offset se aplikuje jen když je kolem objektu místo (tj. když po resize zbývá padding).
      – Výsledkem je matice přesně tvaru (grid_size³).

  voxel_to_obj(voxel_grid, output_filepath):
      – Z voxelové matice vytvoří VoxelGrid (trimesh) a extrahuje mesh - exportuje do OBJ.

================================================================================
"""
import os
import mesh
import trimesh
from matplotlib import pyplot as plt
import scipy.ndimage
import numpy as np
import scipy.ndimage as ndi

import os, numpy as np

CACHE_DIR = "voxel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def obj_to_voxel(filepath, grid_size=64, show=False,debug=False, fill_internal=True):
    # Převod relativní cesty na absolutní
    abs_filepath = os.path.abspath(filepath)
    if not os.path.exists(abs_filepath):
        raise FileNotFoundError(f"Soubor nebyl nalezen: {abs_filepath}")

    mesh = trimesh.load(abs_filepath, force='mesh')
    if show:
        print("=== Raw Mesh Information ===")
        print("Vertices (shape {}):".format(mesh.vertices.shape))
        print(mesh.vertices)
        print("Faces (shape {}):".format(mesh.faces.shape))
        print(mesh.faces)
        print("Bounds:")
        print(mesh.bounds)
        print("============================")

    bounds = mesh.bounds  #[min, max]

    dimensions = bounds[1] - bounds[0]
    pitch = max(dimensions) / (grid_size - 1)  # -1 protože to zakruhluje nahoru a může to přesáhnout
    #mesh.apply_scale(scale_factor) # neni potřeba, očividně pitch už dělá tenhle scale
    voxels = mesh.voxelized(pitch=pitch)
    voxel_matrix = voxels.matrix
    if fill_internal:
        voxel_matrix = fill_voxel_grid(voxel_matrix)

    new_matrix = pad_to_grid(voxel_matrix, grid_size)
    #print(new_matrix)

    # vizualizace
    if show:
        print(f"Voxel dimensions (raw): {voxel_matrix.shape}")
        print(f"Voxel dimensions (after padding): {new_matrix.shape}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(new_matrix, edgecolor='k')
        plt.show()
    return new_matrix.astype(np.float32)


def pad_to_grid(voxel_matrix, grid_size):
    current_shape = voxel_matrix.shape

    # Nějaká dimenze větší než grid_size -> změním velikost matice
    if current_shape[0] > grid_size or current_shape[1] > grid_size or current_shape[2] > grid_size:
        # Vypočtu měřítko pro každou dimenzi
        scale_factors = [min(1.0, grid_size / dim) for dim in current_shape]
        min_scale = min(scale_factors)  # nejmenší měřítko pro zachování proporcí

        new_shape = tuple(int(dim * min_scale) for dim in current_shape)

        # zoom pro změnu velikosti voxelové mřížky
        resized_matrix = scipy.ndimage.zoom(voxel_matrix,
                                            (new_shape[0] / current_shape[0],
                                             new_shape[1] / current_shape[1],
                                             new_shape[2] / current_shape[2]),
                                            order=0,  # Použijeme nejbližší soused pro zachování binární povahy
                                            mode='constant',
                                            cval=0)
        voxel_matrix = resized_matrix
        current_shape = new_shape

    pad_x = grid_size - current_shape[0]
    pad_y = grid_size - current_shape[1]
    pad_z = grid_size - current_shape[2]

    # Náhodný offset pro augmentaci dat
    random_x_offset = np.random.randint(0, pad_x + 1) if pad_x > 0 else 0
    random_y_offset = np.random.randint(0, pad_y + 1) if pad_y > 0 else 0

    padding = [
        ((pad_x + random_x_offset) // 2, pad_x - (pad_x + random_x_offset) // 2),
        ((pad_y + random_y_offset) // 2, pad_y - (pad_y + random_y_offset) // 2),
        (pad_z // 2, pad_z - pad_z // 2),
    ]

    padded_matrix = np.pad(voxel_matrix, pad_width=padding, mode='constant', constant_values=0)
    return padded_matrix


def voxel_to_obj(voxel_grid, output_filepath):
    # vytvoř mesh z voxelu
    #voxel_grid[0,0,0] = False
    #print(np.sum(voxel_grid==False))
    voxels = trimesh.voxel.VoxelGrid(voxel_grid)
    #print(np.sum(voxels.matrix==False))
    mesh = voxels.marching_cubes
    mesh.export(output_filepath)

def fill_voxel_grid(vox):
    # scipy očekává bool
    solid = vox.astype(bool)
    filled = ndi.binary_fill_holes(solid)
    return filled.astype(vox.dtype)



if __name__ == "__main__":
    Test = obj_to_voxel(r"./dataset/BuildingB-rotation_and_floors/variant_1.obj", show=True)
    voxel_to_obj(Test, r"../generated/generated.obj")