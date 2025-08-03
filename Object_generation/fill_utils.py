import numpy as np
import torch
import sys, site, torch

# print(f"CUDA je dostupná: {torch.cuda.is_available()}")
# print(f"PyTorch verze: {torch.__version__}")
# if torch.cuda.is_available():
#     print(f"Počet GPU: {torch.cuda.device_count()}")
#     print(f"Aktuální GPU: {torch.cuda.get_device_name(0)}")
#     print(f"CUDA verze: {torch.version.cuda}")
#     # Ověřte paměť GPU
#     print(f"Celková paměť GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
# else:
#     print("CUDA není dostupná!")
import scipy.ndimage as ndi

def fill_voxel_grid(vox):
    """
Vytvoří z dutého objektu pevný objekt vyplněním všech vnitřních dutin.
    Každý 0-voxel, který není připojen k hranici mřížky, se změní na 1.
    """
    # scipy expects bool
    solid = vox.astype(bool)
    filled = ndi.binary_fill_holes(solid)         # N-dimensional flood-fill
    return filled.astype(vox.dtype)
