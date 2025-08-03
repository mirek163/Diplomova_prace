# Vyhodnocení, jak je vyplněná voxelová mřížka objektu 

import glob, os, numpy as np, matplotlib.pyplot as plt
import processing_obj as prc               # <- your converter

OBJ_DIR   = "test"              #složka s .obj soubory
GRID_SIZE = 64

ratios = []
obj_paths = glob.glob(os.path.join(OBJ_DIR, "*.obj"))
assert obj_paths, f"Nenašel jsem žádny .obj soubory v  {OBJ_DIR}"

for p in obj_paths:
    vox = prc.obj_to_voxel(p, grid_size=GRID_SIZE, show=False)   # (D,H,W) float32
    occ_ratio = (vox > 0).mean() # poměr obsazených voxelů
    ratios.append(occ_ratio)

ratios = np.array(ratios)
print(f"Zpracovaných modelů : {len(ratios)} ks")
print(f"Průměrně zaplněno   : {ratios.mean():.1%}")
print(f"Medián              : {np.median(ratios):.1%}")
print(f"Rozptyl hodnot      : {ratios.min():.1%} až {ratios.max():.1%}")

plt.hist(ratios, bins=20)
plt.xlabel("Poměr zaplněných voxelů")
plt.ylabel("Počet 3D modelů")
plt.title(f"Distribuce zaplněnosti voxelové mřížky ({GRID_SIZE}³)")
plt.show()
