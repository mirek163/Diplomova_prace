
"""
obarvovač OBJ: buď celistvá barva, nebo vertikální textura.
"""
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

# === GLOBÁLNÍ PARAMETRY ======================================================
# 'solid' = jednobarevné, 'texture' = gradientní textura
MODE = 'texture'
# Složka s .obj soubory
data_dir = Path('./generated')

# barva pro solid režim
SOLID_RGB = (0.80, 0.25, 0.20)
# Červená:        (0.80, 0.25, 0.20)
# Modrá:          (0.20, 0.40, 0.80)
# Žlutá:          (1.00, 0.90, 0.10)
# Tmavě zelená:   (0.10, 0.45, 0.15)
# Fialová:        (0.60, 0.20, 0.70)
# Hnědá:          (0.45, 0.30, 0.10)
# Šedivá:         (0.50, 0.50, 0.50)
# Oranžová:       (1.00, 0.55, 0.10)

# Parametry pro texture režim (hnědá dole, zelená nahoře)
BROWN_RGB     = np.array([0.55, 0.38, 0.18])
GREEN_RGB     = np.array([0.15, 0.40, 0.18])
TEX_NAME      = 'gradient.png'
TEX_SIZE      = 256
BROWN_PORTION = 0.10  # kolik (0–1) má být plně hnědé
GAMMA         = 2.0   #  poměr přechodu

# =========================================================

def create_gradient_texture(out_path: Path,
                            size: int = TEX_SIZE,
                            brown_portion: float = BROWN_PORTION,
                            gamma: float = GAMMA):
    """
    Vytvoří PNG se svislým přechodem (dole hnědá, nahoře zelená).
    """
    rows = np.linspace(1.0, 0.0, size)[:, None]
    t = np.clip((rows - brown_portion) / (1 - brown_portion), 0.0, 1.0)
    t **= gamma
    grad = (1 - t) * BROWN_RGB + t * GREEN_RGB
    img = np.tile(grad[:, None, :], (1, size, 1))
    Image.fromarray((img * 255).astype(np.uint8)).save(out_path)

def save_obj_with_mtl(mesh: trimesh.Trimesh,
                      out_stem: str,
                      out_dir: Path,
                      mtl_content: str,
                      obj_body: str):
    """Společné uložení OBJ a MTL souborů."""
    obj_path = out_dir / f"{out_stem}.obj"
    mtl_path = out_dir / f"{out_stem}.mtl"

    mtl_path.write_text(mtl_content, encoding='utf-8')
    obj_header = f"mtllib {mtl_path.name}\nusemtl default\n"
    obj_path.write_text(obj_header + obj_body, encoding='utf-8')
    print(f"Uloženo: {obj_path.name}")

def process_solid(obj_path: Path, rgb: tuple):
    mesh = trimesh.load(obj_path, force='mesh')
    if mesh.is_empty:
        print(f"Prázdný mesh: {obj_path.name}")
        return

    obj_str = trimesh.exchange.obj.export_obj(
        mesh,
        include_color=False,
        include_normals=True,
        write_texture=False
    )
    r, g, b = rgb
    mtl = f"newmtl default\nKd {r} {g} {b}\nKa {r} {g} {b}\n"
    save_obj_with_mtl(mesh, obj_path.stem + '_solid', obj_path.parent, mtl, obj_str)

def process_textured(obj_path: Path, tex_name: str):
    mesh = trimesh.load(obj_path, force='mesh')
    if mesh.is_empty:
        print(f"Prázdný mesh: {obj_path.name}")
        return

    verts = mesh.vertices
    faces = mesh.faces
    norms = mesh.vertex_normals if mesh.vertex_normals is not None else []

    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    v_coord = np.zeros(len(verts)) if (y_max - y_min) < 1e-6 else (verts[:, 1] - y_min) / (y_max - y_min)
    uv = np.column_stack([np.full(len(verts), 0.5), v_coord])

    lines = []
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for u, v in uv:
        lines.append(f"vt {u:.6f} {v:.6f}")
    for n in norms:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    lines.append("")
    for face in faces:
        if len(norms):
            tokens = [f"{i+1}/{i+1}/{i+1}" for i in face]
        else:
            tokens = [f"{i+1}/{i+1}" for i in face]
        lines.append("f " + " ".join(tokens))
    obj_body = "\n".join(lines) + "\n"

    mtl = f"newmtl default\nmap_Kd {tex_name}\n"
    save_obj_with_mtl(mesh, obj_path.stem + '_texture', obj_path.parent, mtl, obj_body)

def main():
    if not data_dir.exists():
        print(f"Složka '{data_dir}' neexistuje nebo neobsahuje .obj soubory.")
        return
    obj_files = glob.glob(str(data_dir / "*.obj"))
    if not obj_files:
        print(f"Ve složce '{data_dir}' není žádné .obj.")
        return

    if MODE == 'texture':
        tex_path = data_dir / TEX_NAME
        if not tex_path.exists():
            create_gradient_texture(tex_path)
            print(f"Vygenerována textura {TEX_NAME}")
        else:
            print(f"Textura {TEX_NAME} už existuje")

    print(f"Zpracovávám {len(obj_files)} objektů v režimu '{MODE}'...")
    for obj in obj_files:
        path = Path(obj)
        if MODE == 'solid':
            process_solid(path, SOLID_RGB)
        else:
            process_textured(path, TEX_NAME)

if __name__ == '__main__':
    main()