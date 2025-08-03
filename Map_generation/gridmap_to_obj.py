"""
Aplikování 3D modelů na matici z textového souboru.
===================================================================
Potřeba vytvořit res složku s podsložkami buildings/, vegetation/, water/, roads/ a features/.
V každé podsložce musí být alespoň jeden .obj soubor.

Potřeba existence souboru map.txt s maticí, kde:
- 0 = prázdno (nevyplněná buňka)
- 1 = budovy (buildings/)
- 2 = vegetace (vegetation/)
- 3 = voda (water/)
- 4 = silnice (roads/)
- 5 = další prvky (features/)
- 6 = základní deska (base/)
===================================================================
Vytvoří se výstupní složka output/ a v ní scene.obj s celou scénou.
===================================================================
"""
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# === GLOBÁLNÍ PARAMETRY ======================================================
TXT_FILE = "output_maps/map_007.txt"           # Vstupní texták s maticí
ASSET_ROOT = "res"             # Kořen složky s podsložkami buildings/, vegetation/…
OUTPUT_DIR = Path("output_obj")    # Všechno cíluje sem
OUTPUT_OBJ = OUTPUT_DIR / "scene.obj"  # kam se zapíše výsledná scéna
CELL_SIZE = 64.0                 # Rozměr jedné mřížkové buňky (v OBJ jednotkách)
BASE_THICKNESS = 1.0          # Tloušťka základní desky dolů
SEED: Optional[int] = 42       # Reproducibilita; None => opravdu náhodné
ASSET_CACHE: dict[str, dict] = {}

# Mapování hodnot v matici na složky s modely a barvičky pro náhled
CATEGORY_MAP = {
    0: {"name": "empty",      "folder": None,         "color": (1.0, 1.0, 1.0)},  # bílá
    1: {"name": "buildings",  "folder": "buildings",  "color": (0.75, 0.0, 0.0)},  # červená
    2: {"name": "vegetation", "folder": "vegetation", "color": (0.0, 0.5, 0.0)},   # zelená
    3: {"name": "water",      "folder": "water",      "color": (0.0, 0.4, 1.0)},   # modrá
    4: {"name": "roads",      "folder": "roads",      "color": (0.4, 0.4, 0.4)},   # šedá
    5: {"name": "features",   "folder": "features",   "color": (0.8, 0.5, 0.2)},   # hnědá (nová kategorie)
    6: {"name": "base",       "folder": "base",       "color": (0.1, 0.1, 0.1)},   # tmavě šedá (základ)
}


# --------------------------------------------------------------------------- #
# Utility: text -> numpy
def read_matrix(path: str | Path) -> np.ndarray:
    """Načti soubor a vrať numpy matici intů."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    #  \n tak ; je možnáý použít
    #text = text.replace(";", "\n")
    #rows = [row.split(',') for row in text.splitlines() if row]
    rows = [row.strip().split() for row in text.splitlines() if row]
    return np.array(rows, dtype=int)

#- --------------------------------------------------------------------------- #
def prepare_asset(
    asset_path: Path
) -> Tuple[
    List[List[float]],  # verts
    List[List[float]],  # texs
    List[List[float]],  # norms
    List[Tuple[
        List[int],          # vertex indices
        Optional[List[int]],# tex‑coord indices
        Optional[List[int]],# normal indices
        Optional[str]       # current_mtl
    ]],                 # faces
    List[str],          # mtllibs
    str,                # prefix
    Dict[str, str]      # mtl_map
]:
    """
    1) Parse .obj
    2) Vygeneruje prefix + nakopíruje .mtl/textury (jen jednou díky cache)
    3) Vykalibruje měřítko (CELL_SIZE)
    Vrací: verts, texs, norms, faces, mtllibs, prefix, mtl_map
    """
    verts, texs, norms, faces, mtllibs = parse_obj(asset_path)

    if asset_path not in ASSET_CACHE:
        # prefix = a0, a1, … podle počtu unikátních assetů
        prefix = f"a{len(ASSET_CACHE)}"
        mtl_map = copy_mtl_and_textures(asset_path, mtllibs, OUTPUT_DIR, prefix)
        ASSET_CACHE[asset_path] = {"prefix": prefix, "mtl_map": mtl_map}
    else:
        prefix   = ASSET_CACHE[asset_path]["prefix"]
        mtl_map  = ASSET_CACHE[asset_path]["mtl_map"]

    scale_and_center(verts, CELL_SIZE)
    return verts, texs, norms, faces, mtllibs, prefix, mtl_map

# --------------------------------------------------------------------------- #
def preview_matrix(mat: np.ndarray):
    """Rychlý 2D náhled"""
    max_val = int(mat.max())
    cmap_colors = [CATEGORY_MAP.get(v, {"color": (1, 0, 1)})["color"]
                   for v in range(max_val + 1)]
    cmap = ListedColormap(cmap_colors, name="scene_map")

    plt.figure(figsize=(6, 6))
    plt.imshow(mat, cmap=cmap, vmin=0, vmax=max_val, interpolation="none", origin="lower")
    plt.title("Náhled matice")
    plt.axis("off")
    plt.show()


# --------------------------------------------------------------------------- #
def list_assets(asset_root: str | Path, folder: str | None):
    if folder is None:
        return []
    path = Path(asset_root) / folder
    files = list(path.rglob("*.obj"))
    #print(f"[DEBUG] hledám .obj v {path!r}, našel jsem {len(files)} souborů:")
    # for p in files:
    #     print("  •", p)
    return [str(p) for p in files if p.is_file()]

# --------------------------------------------------------------------------- #
# OBJ/MTL parser
def parse_obj(path: Path):
    """
    Vrací:
        verts, texs, norms –List[List[float]]
        faces –List[Tuple[List[int], List[int]|None, List[int]|None, str]]
        mtllibs –List[str] (jen názvy .mtl souborů)
    """
    verts, texs, norms = [], [], []
    faces: List[Tuple[List[int], List[int] | None, List[int] | None, str | None]] = []
    mtllibs: List[str] = []
    current_mtl: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("mtllib"):
                mtllibs.extend(line.strip().split(maxsplit=1)[1].split())
            elif line.startswith("usemtl"):
                current_mtl = line.split()[1].strip()
            elif line.startswith("v "):
                _, x, y, z = line.split(maxsplit=3)
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("vt "):
                parts = line.split()
                texs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("vn "):
                parts = line.split()
                norms.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                v_idx, vt_idx, vn_idx = [], [], []
                for token in line.split()[1:]:
                    # formáty: v, v/vt, v//vn, v/vt/vn
                    splits = token.split("/")
                    v_idx.append(int(splits[0]) - 1)
                    if len(splits) >= 2 and splits[1]:
                        vt_idx.append(int(splits[1]) - 1)
                    if len(splits) == 3 and splits[2]:
                        vn_idx.append(int(splits[2]) - 1)
                faces.append((v_idx,
                              vt_idx or None,
                              vn_idx or None,
                              current_mtl))
    return verts, texs, norms, faces, mtllibs


# --------------------------------------------------------------------------- #
def write_obj(path: Path,
              verts, texs, norms,
              faces: List[Tuple[List[int], List[int]|None, List[int]|None, str]],
              mtllibs: List[str]):
    """Zápis výsledného OBJ s materiály."""
    with open(path, "w", encoding="utf-8") as f:
        # všechny použité MTL
        for m in sorted(set(mtllibs)):
            f.write(f"mtllib {m}\n")

        # geometrie
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in texs:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for n in norms:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # faces + materiály
        last_mtl = None
        for v_idx, vt_idx, vn_idx, mtl in faces:
            if mtl != last_mtl:
                if mtl:
                    f.write(f"usemtl {mtl}\n")
                last_mtl = mtl

            tokens = []
            for i in range(len(v_idx)):
                v_part = str(v_idx[i] + 1)
                vt_part = str(vt_idx[i] + 1) if vt_idx else ""
                vn_part = str(vn_idx[i] + 1) if vn_idx else ""
                if vt_part or vn_part:
                    tokens.append(f"{v_part}/{vt_part}/{vn_part}")
                else:
                    tokens.append(v_part)
            f.write("f " + " ".join(tokens) + "\n")


# --------------------------------------------------------------------------- #
def copy_mtl_and_textures(obj_path: Path,
                          mtllibs: List[str],
                          dest_dir: Path,
                          prefix: str) -> Dict[str, str]:
    """
    Překopíruje .mtl + textury do dest_dir a přidá prefix.
    Vrací mapování původní_mtl_jméno -> nové_jméno.
    """
    mtl_map: Dict[str, str] = {}

    for mtl_filename in mtllibs:
        src_mtl = obj_path.parent / mtl_filename
        if not src_mtl.exists():
            print(f"MTL '{src_mtl}' nenalezen. Přeskakuji.")
            continue

        new_mtl_name = f"{prefix}__{mtl_filename}"
        mtl_map[mtl_filename] = new_mtl_name
        dst_mtl = dest_dir / new_mtl_name
        # už jsem tenhle .mtl jednou vyrobil? Tak nepiš znovu
        if dst_mtl.exists():
            mtl_map[mtl_filename] = new_mtl_name
            continue  # přeskoč zbytek for‑cyklu a jdi na další MTL

        with open(src_mtl, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            # řádek typu: map_Kd texture.png
            if line.lower().startswith("map_"):
                tokens = line.split(maxsplit=1)
                if len(tokens) == 2:
                    tex_name = tokens[1].strip()
                    src_tex = obj_path.parent / tex_name
                    new_tex_name = f"{prefix}__{tex_name}"
                    dst_tex = dest_dir / new_tex_name

                    # zkopírovat texturu
                    if src_tex.exists() and not dst_tex.exists():
                        shutil.copy2(src_tex, dst_tex)
                    else:
                        print(f"Textura '{src_tex}' nenalezena.")
                    # přepsat cestu v MTL
                    new_lines.append(f"{tokens[0]} {new_tex_name}\n")
                    continue  # další řádek
            # ostatní řádky:
            new_lines.append(line)

        # přidáme prefix i k názvům materiálů (newmtl ...)
        for i, ln in enumerate(new_lines):
            if ln.startswith("newmtl"):
                parts = ln.split()
                parts[1] = f"{prefix}__{parts[1]}"
                new_lines[i] = " ".join(parts) + "\n"

        # finální zápis
        with open(dst_mtl, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return mtl_map

#---------------------------------------------------------------------------- #
# Utility: scale and center vertices
def scale_and_center(verts: List[List[float]], target_size: float) -> None:
    """
    In‑place změní vertices tak, aby jejich footprint XY zapadl do target_size×target_size
    a byl vystředěný v[0,target_size].
    """
    xs = [v[0] for v in verts]
    zs = [v[2] for v in verts]
    width  = max(xs) - min(xs)
    depth  = max(zs) - min(zs)
    if width == 0 or depth == 0:
        return  # degenerate – nechám být

    # uniform scale tak, aby se vešel obvod‑větší rozměr
    scale = target_size / max(width, depth)
    for v in verts:
        v[0] *= scale
        v[1] *= scale
        v[2] *= scale

    # posun do středu buňky (v místním originu 0,0)
    xs = [v[0] for v in verts]
    zs = [v[2] for v in verts]
    offset_x = (target_size - (max(xs) - min(xs))) / 2 - min(xs)
    offset_z = (target_size - (max(zs) - min(zs))) / 2 - min(zs)
    for v in verts:
        v[0] += offset_x
        v[2] += offset_z
#- --------------------------------------------------------------------------- #
def insert_base_asset(
    mat: np.ndarray,
    r: int,
    c: int,
    verts: List[List[float]],
    texs: List[List[float]],
    norms: List[List[float]],
    faces: List[Tuple[List[int], Optional[List[int]], Optional[List[int]], str]],
    used_mtllibs: List[str],
    v_offset: int,
    vt_offset: int,
    vn_offset: int
) -> Tuple[int, int, int]:
    """
    Vloží "základní podstavec" asset do scény.
    Vrací nové offsety: v_offset, vt_offset, vn_offset.
    """
    base_folder = (
        "base/water" if mat[r, c] == 3 else
        "base/grass" if mat[r, c] == 2 else
        "base/ground"
    )
    base_list = list_assets(ASSET_ROOT, base_folder)
    if not base_list:
        return v_offset, vt_offset, vn_offset

    asset_path = Path(random.choice(base_list))
    a_v, a_t, a_n, a_f, a_mtl, prefix, mtl_map = prepare_asset(asset_path)
    # posun dolu o BASE_THICKNESS
    for v in a_v:
        v[1] -= BASE_THICKNESS
    rotate_y(a_v, a_n, 0)

    # update libů a faces
    used_mtllibs.extend(f"{prefix}__{m}" for m in a_mtl if m in mtl_map)
    for i, f in enumerate(a_f):
        if f[3]:
            a_f[i] = (f[0], f[1], f[2], f"{prefix}__{f[3]}")

    tx, tz = c * CELL_SIZE, r * CELL_SIZE
    for v in a_v:
        verts.append([v[0] + tx, v[1], v[2] + tz])
    texs.extend(a_t)
    norms.extend(a_n)
    for v_idx, vt_idx, vn_idx, mtl in a_f:
        faces.append((
            [i + v_offset for i in v_idx],
            [i + vt_offset for i in vt_idx] if vt_idx else None,
            [i + vn_offset for i in vn_idx] if vn_idx else None,
            mtl
        ))

    return (
        v_offset + len(a_v),
        vt_offset + len(a_t),
        vn_offset + len(a_n)
    )


def insert_asset_for_value(
    mat: np.ndarray,
    r: int,
    c: int,
    val: int,
    ban_B: np.ndarray,
    verts: List[List[float]],
    texs: List[List[float]],
    norms: List[List[float]],
    faces: List[Tuple[List[int], Optional[List[int]], Optional[List[int]], str]],
    used_mtllibs: List[str],
    v_offset: int,
    vt_offset: int,
    vn_offset: int,
    assets_per_value: Dict[int, List[str]]
) -> Tuple[int, int, int]:
    """
    Vloží asset odpovídající val (1..6) do scény.
    Vrací nové offsety.
    """
    if val == 0:
        return v_offset, vt_offset, vn_offset

    if val == 4:
        asset_path, turns = choose_road_asset(mat, r, c, ban_B, mat.shape[0], mat.shape[1])
        if asset_path is None:
            return v_offset, vt_offset, vn_offset
    else:
        turns = 0
        asset_list = assets_per_value.get(val, [])
        if not asset_list:
            return v_offset, vt_offset, vn_offset
        asset_path = Path(random.choice(asset_list))

    a_v, a_t, a_n, a_f, mtllibs, prefix, mtl_map = prepare_asset(asset_path)
    if val in (1, 2, 3): #srovnani objektu , zaorvnani s base tilem
        lift_to_ground(a_v)
    rotate_y(a_v, a_n, turns)

    used_mtllibs.extend(f"{prefix}__{m}" for m in mtllibs if m in mtl_map)
    # remap faces
    for i, (_, _, _, mtl) in enumerate(a_f):
        if mtl:
            a_f[i] = (
                a_f[i][0], a_f[i][1], a_f[i][2], f"{prefix}__{mtl}"
            )

    tx, tz = c * CELL_SIZE, r * CELL_SIZE
    for v in a_v:
        verts.append([v[0] + tx, v[1], v[2] + tz])
    texs.extend(a_t)
    norms.extend(a_n)
    for v_idx, vt_idx, vn_idx, mtl in a_f:
        faces.append((
            [i + v_offset for i in v_idx],
            [i + vt_offset for i in vt_idx] if vt_idx else None,
            [i + vn_offset for i in vn_idx] if vn_idx else None,
            mtl
        ))

    return (
        v_offset + len(a_v),
        vt_offset + len(a_t),
        vn_offset + len(a_n)
    )
# --------------------------------------------------------------------------- #
# Utility: rotate vertices and normals
def rotate_y(verts, norms, turns: int) -> None:
    """In‑place otočí vertexy+ normály o turns×90° kolem středu buňky."""
    turns &= 3
    if turns == 0:
        return
    ang = turns * (np.pi / 2)
    ca, sa = np.cos(ang), np.sin(ang)
    cx = cz = CELL_SIZE / 2
    for v in verts:
        x, z = v[0] - cx, v[2] - cz
        v[0] =  x * ca - z * sa + cx
        v[2] =  x * sa + z * ca + cz
    for n in norms:
        x, z = n[0], n[2]
        n[0] = x * ca - z * sa
        n[2] = x * sa + z * ca
#- --------------------------------------------------------------------------- #
def lift_to_ground(verts: List[List[float]], eps: float = 0.02) -> None:
    """
    Posune model tak, aby jeho nejnižší bod ležel na Y = eps.
    (eps = malá mezera proti z-fightingu.)
    """
    min_y = min(v[1] for v in verts)
    delta = eps - min_y            # kam to celé posunout
    if abs(delta) < 1e-6:          # když už tam v podstatě jsme, neřeš
        return

    for v in verts:
        v[1] += delta
# --------------------------------------------------------------------------- #
# Utility: road variant
def road_variant(mat: np.ndarray, r: int, c: int) -> str:
    """Vrátí název podsložky podle sousedních silnic (0=prázdno, 4=road)."""

    rows, cols = mat.shape
    n = r > 0         and mat[r-1, c] == 4
    s = r < rows-1    and mat[r+1, c] == 4
    w = c > 0         and mat[r, c-1] == 4
    e = c < cols-1    and mat[r, c+1] == 4

    cnt = int(n) + int(s) + int(w) + int(e)

    # variant = (
    #     "junction" if cnt == 4 else
    #     "tsplit" if cnt == 3 else
    #     ("straight" if (n and s) or (w and e) else "turn") if cnt == 2 else
    #     "end" if cnt == 1 else
    #     "isolated"
    # )
    # print(f"[DBG‑ASSET] tile {(r,c)} variant={variant}, chosen={chosen.name}, turns={turns}")

    if cnt == 4:
        return "junction"  # kříž
    if cnt == 3:
        return "tsplit"  # T‑rozdvojka
    if cnt == 2:
        return "straight" if (n and s) or (w and e) else "turn"
    if cnt == 1:
        return "straight"  # slepá
# --------------------------------------------------------------------------- #
def generate_scene(mat: np.ndarray, output_obj: Path):
    """Slepí scénu podle matice a nakopíruje MTL + textury."""
    if SEED is not None:
        random.seed(SEED)
    # Výstupní složka
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, cols = mat.shape
    ban_B = np.zeros((rows, cols), dtype=bool)
    assets_per_value = {
        val: list_assets(ASSET_ROOT, meta["folder"])
        for val, meta in CATEGORY_MAP.items()
    }

    verts: List[List[float]] = []
    texs: List[List[float]] = []
    norms: List[List[float]] = []
    faces: List[Tuple[List[int], List[int]|None, List[int]|None, str]] = []
    used_mtllibs: List[str] = []

    v_offset, vt_offset, vn_offset = 0, 0, 0

    # # === základní deska ======================================================
    # w = cols * CELL_SIZE
    # d = rows * CELL_SIZE
    # base_verts = [
    #     (0, -BASE_THICKNESS, 0),
    #     (w, -BASE_THICKNESS, 0),
    #     (w, -BASE_THICKNESS, d),
    #     (0, -BASE_THICKNESS, d),
    #     (0, 0, 0),
    #     (w, 0, 0),
    #     (w, 0, d),
    #     (0, 0, d),
    # ]
    # base_faces = [
    #     (0, 1, 2, 3),  # spodek
    #     (4, 5, 6, 7),  # vršek
    #     (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),  # boky
    # ]
    # verts.extend(base_verts)
    # for bf in base_faces:
    #     faces.append(([v + v_offset for v in bf], None, None, None))
    # v_offset = len(verts)

    # === vkládání assetů =====================================================
    for r in range(rows):
        for c in range(cols):
            # vložíme základní podložku
            v_offset, vt_offset, vn_offset = insert_base_asset(
                mat, r, c, verts, texs, norms, faces, used_mtllibs,
                v_offset, vt_offset, vn_offset
            )
            # vložíme vlastní asset podle val
            val = int(mat[r, c])
            # ákladní podlaha – všechno kromě silnic
            if val != 4:
                v_offset, vt_offset, vn_offset = insert_base_asset(
                    mat, r, c, verts, texs, norms, faces, used_mtllibs,
                    v_offset, vt_offset, vn_offset
                )

            # samotný asset podle matice
            v_offset, vt_offset, vn_offset = insert_asset_for_value(
                mat, r, c, val, ban_B, verts, texs, norms,
                faces, used_mtllibs, v_offset, vt_offset, vn_offset,
                assets_per_value
            )


    # === zápis =================================================================
    write_obj(output_obj, verts, texs, norms, faces, used_mtllibs)
    print(f"Hotovo: {len(verts)} vrcholů, {len(faces)} ploch -> {output_obj}")


def choose_road_asset(mat: np.ndarray,
                      r: int, c: int,
                      ban_B: np.ndarray,
                      rows: int, cols: int) -> Tuple[Optional[Path], int]:
    """
    Vrátí (Path k.obj souboru, počet otoček o90° kolemY).
    Když chybí asset, vrátí (None,0).
    """
    variant = road_variant(mat, r, c)      # straight / turn / …
    asset_list = list_assets(ASSET_ROOT, f"roads/{variant}")
    if not asset_list:
        return None, 0

    # --- výběr konkrétního modelu ------------------------------------------
    if variant == "straight":
        straight_b = [p for p in asset_list if Path(p).name == "straight_B.obj"] # přechod
        others     = [p for p in asset_list if Path(p).name != "straight_B.obj"]

        want_B = straight_b and random.random() < 0.20   # los
        can_B  = not ban_B[r, c]                         # není zakázáno

        if want_B and can_B:
            # zamkneme okolí (Chebyshev ≤4 buněk)
            rr0, rr1 = max(0, r - 4), min(rows, r + 5)
            cc0, cc1 = max(0, c - 4), min(cols, c + 5)
            ban_B[rr0:rr1, cc0:cc1] = True
            chosen = Path(straight_b[0])
        else:
            chosen = Path(random.choice(others or straight_b))
    else:
        chosen = Path(random.choice(asset_list))

    # --- orientace podle sousedů -------------------------------------------
    n = r > 0      and mat[r-1, c] == 4
    s = r < rows-1 and mat[r+1, c] == 4
    w = c > 0      and mat[r, c-1] == 4
    e = c < cols-1 and mat[r, c+1] == 4

    if variant == "straight":
        turns = 0 if n and s else 1                     # default asset != vodorovný
    elif variant == "turn":                             # default = spoj E+S
        if e and s:   turns = 0
        elif s and w: turns = 1
        elif w and n: turns = 2
        else:         turns = 3                         # n + e
    elif variant == "tsplit":
        if not s:     turns = 3
        elif not w:   turns = 0
        elif not n:   turns = 1
        else:         turns = 2
    elif variant == "junction":
        turns = random.randint(0, 3)                   # kříž je symetrický
    else:  # end
        if s:         turns = 0
        elif w:       turns = 1
        elif n:       turns = 2
        else:         turns = 3                         # e
    print(f"[DBG‑ASSET] tile {(r, c)} variant={variant},"
          f" chosen={chosen.name}, turns={turns}")
    return chosen, turns
# --------------------------------------------------------------------------- #
def main():
    mat = read_matrix(TXT_FILE)
    print(f"Načtena matice {mat.shape[0]}×{mat.shape[1]}")
    preview_matrix(mat)
    generate_scene(mat, OUTPUT_OBJ)


if __name__ == "__main__":
    main()
