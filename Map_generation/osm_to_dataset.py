"""
Generátor datasetu z OpenStreetMap a kontrola obsahu dlaždic
===============================================================================

Program stáhne zvolenou oblast z OSM, rasterizuje mapové vrstvy do matice
a rozseká je na malé 32×32 dlaždice na dataset.

Každá dlaždice se ukládá pouze pokud:
- obsahuje zvolený podíl silnic, vegetace a vody (s tolerancí)
- např. 10 % silnic ±2 %, 5 % vegetace ±2 %, voda 0–10 % (viz `TARGETS`)

Možnosti augmentace:
- rotace: 0°, 90°, 180°, 270° (parametr `ROTATE`)
- flip: žádný / horizontální / vertikální (parametr `FLIP`)

VÝSTUP:
- složka `dataset/` obsahující .txt soubory s dlaždicemi
- každý soubor je jedna 32×32 matice (čísla 0–4 podle třídy objektu)

VSTUPNÍ DATA:
- automaticky stažená z OSM (silnice, budovy, vegetace, voda)
- oblast daná buď předdefinovanou lokací (`LOCATION`), nebo souřadnicemi

- 0 = prázdno
- 1 = budovy
- 2 = vegetace
- 3 = voda
- 4 = silnice

DALŠÍ FUNKCE:
- možnost postprocessingu silnic (propojení A*)
- jednoduché doplnění budov podél cest (`BUILDING_SIMPLE`)
- náhled v obrázcích (`previews/`) při nedělání datasetu + to vygeneruje jednu mapu

U DATASETU:
- pro větší šanci na vhodné dlaždice lze zvýšit `MASTER_PAD`
- program běží, dokud nenajde požadovaný počet validních vzorků (`N_SAMPLES`)
===============================================================================
"""
from __future__ import annotations
import pathlib
import numpy as np
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio import features
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure, binary_dilation
from scipy.spatial.distance import cdist
from shapely.geometry import box
import random
import networkx as nx
from skimage.morphology import skeletonize
ox.settings.use_cache       = True
ox.settings.cache_only_mode = False
OX_CRS = "EPSG:3857"

# === GLOBÁLNÍ PARAMETRY ======================================================
LOCATION = "washington"
AUTOSEARCH = False       #  Automaticky najít oblast se všemi indexi
AS_LAT, AS_LON = 38.9072, -77.0570 # výchozí souřadnice pro automatické hledání

ROAD_CONNECTION = False  # doplnění silnic pomocí A* algoritmu ( při mensí velikosti mapy např. 32x32)
BUILDING_SIMPLE = False  # jedna budova u cesty, jinak prázdno

RESOLUTION = 128          #  64 #32
OUT = "grid.txt"
PREVIEW_DIR = "visulalization/dataset_from_osm"
SIZE_KM = 0.4          # size_km 0.5 #0.25

#DATASET PARAMETRY
DATASET = False          #  True = vytvořit dataset , False = normaální běh
N_SAMPLES = 8000           # počet map, ktere chci pro dataset
ROTATE = [0, 1, 2, 3]         #  násobky 90° (0 = originál, 1 = 90°, …)
FLIP   = ["none", "h", "v"]   # horizontální, vertikální
TILE_SIZE = 32 # velikost mapy pro každý soubor v datasetu ( lepší cesty při vyšší hlavní dimenzi a pak ořezání na několik 32x32 )
OFFSET_RATIO  = 0.05         #  max +-5% velikosti bboxu
MASTER_PAD   = 5.0      #  zvětšení oblasti pro stažení vrstev (aby se vyhnulo ořezávání)
DATASET_DIR   = pathlib.Path("dataset")

# MAPOVÁNÍ
TAG_TABLE = {
    "roads": 4,   # highway wo railway
    "buildings": 1,   # building= vsechny
    "vegetation": 2, # vegetation
    "water": 3        # all water-related features
}

TARGETS = { # procento pokrytí jednotlivých hodnot v % pro dělání datasetu, pokud prolbem tak zvyšít pad nebo snížit tolerance
    TAG_TABLE["roads"]:      (0.10, 0.02),   # 10% silnic ±2%
    TAG_TABLE["vegetation"]: (0.05, 0.02),   # 5% zeleně ±2%
    TAG_TABLE["water"]:      (0.05, 0.05),   # 5% vody ±5% ->Voda tma nemusi ted v tomhle byt -> snažší generace datasetu
}

ROADS_TAGS = {
    "highway": [
        "motorway", "motorway_link", "trunk", "trunk_link",
        "secondary", "secondary_link", "tertiary",
        "residential", "living_street"
    ],
    "railway": False
}
VEGETATION_TAGS = [
    {"landuse": ["forest", "meadow", "grass", "farmland", "orchard", "vineyard"]},
    {"natural": ["wood", "tree"]},
    {"leisure": ["park"]}
]
WATER_TAGS = [
    {"natural": ["water"]},
    {"waterway": ["river", "stream", "canal"]},
    {"landuse": ["reservoir"]}
]
# souřadnice předdefinovaných lokalit na test, washington vypada zatim nejlepe
locations = {
    "bynov": (50.7687, 14.2361),
    "poruba": (49.8311, 18.1647),
    "havířov": (49.7803, 18.4344),
    "jižní_město": (50.0289, 14.5106),
    "most": (50.5046, 13.6444),
    "vinohrady": (50.0760, 14.4548),
    "manhattan": (40.7931, -73.9712),
    "eixample": (41.3874, 2.1686),
    "washington": (38.9075, -77.0558)  # 38.9072, -77.0570 #38.9072, -77.0558
}
# ----------------------------------------------------------------------

# stáhnout jednou velký polygon a sjednotit CRS
def download_master_layers(lat, lon, size_km, pad=2.0):
    big_bbox = create_small_bbox_around_point(lat, lon, size_km * pad)
    master   = download_layers(big_bbox)
    for gdf in master.values():
        if not gdf.empty:
            gdf.to_crs(OX_CRS, inplace=True)
    return master, big_bbox
# --------------------------------------------------------------------------- #
# rychlý lokální clip
def clip_layers(master_layers: dict[str, gpd.GeoDataFrame], bbox):
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326").to_crs(OX_CRS)
    clipped  = {}
    for name, gdf in master_layers.items():
        if gdf.empty:
            clipped[name] = gdf
        else:
            clipped[name] = gpd.clip(gdf, bbox_gdf)
            if not clipped[name].empty:
                clipped[name].to_crs("EPSG:4326", inplace=True)  # zpět kvůli rasteru
    return clipped
# --------------------------------------------------------------------------- #
def create_small_bbox_around_point(center_lat, center_lon, size_km=0.6):
    size_deg = size_km / 111.0
    margin = size_deg / 2
    bbox = box(center_lon - margin, center_lat - margin, center_lon + margin, center_lat + margin)
    return bbox
# --------------------------------------------------------------------------- #
def download_layers(polygon):
    layers = {}
    bounds = polygon.bounds
    print(f"Souradnice: {bounds}")

    try:
        print("Stahuji dopravní prvky...")
        layers["roads"] = ox.features_from_polygon(polygon, tags=ROADS_TAGS)
        print(f"  roads: {len(layers['roads'])} prvků")
    except Exception as e:
        print(f"Žádné dopravní prvky nalezeny: {e}")
        layers["roads"] = gpd.GeoDataFrame()

    try:
        layers["buildings"] = ox.features_from_polygon(polygon, tags={"building": True})
        print(f"  Budovy: {len(layers['buildings'])} prvků")
    except Exception as e:
        print(f"Žádné budovy nalezeny: {e}")
        layers["buildings"] = gpd.GeoDataFrame()

    all_vegetation = []
    for tag in VEGETATION_TAGS:
        try:
            print(f"Zkouším vegetation {tag}...")
            result = ox.features_from_polygon(polygon, tags=tag)
            if not result.empty:
                print(f"  ✓ Nalezeno {len(result)} prvků pro {tag}")
                all_vegetation.append(result)
            else:
                print(f"  ✗ Nic pro {tag}")
        except Exception as e:
            print(f"  ✗ Chyba pro {tag}: {e}")
    if all_vegetation:
        import pandas as pd
        layers["vegetation"] = gpd.GeoDataFrame(pd.concat(all_vegetation, ignore_index=True))
    else:
        layers["vegetation"] = gpd.GeoDataFrame()

    all_water = []
    for tag in WATER_TAGS:
        try:
            print(f"Zkouším vodní tag {tag}...")
            result = ox.features_from_polygon(polygon, tags=tag)
            if not result.empty:
                print(f"  ✓ Nalezeno {len(result)} vodních prvků pro {tag}")
                all_water.append(result)
            else:
                print(f"  ✗ Nic pro {tag}")
        except Exception as e:
            print(f"  ✗ Chyba pro {tag}: {e}")
    if all_water:
        import pandas as pd
        layers["water"] = gpd.GeoDataFrame(pd.concat(all_water, ignore_index=True))
    else:
        layers["water"] = gpd.GeoDataFrame()
    return layers
# --------------------------------------------------------------------------- #
def rasterise(layers, bounds, res):
    west, south, east, north = bounds
    transform = rasterio.transform.from_bounds(west, south, east, north, res, res)
    out = np.zeros((res, res), dtype=np.uint8)
    for name, gdf in layers.items():
        if gdf.empty:
            continue
        shapes = ((geom, TAG_TABLE[name]) for geom in gdf.geometry if geom is not None)
        burned = features.rasterize(shapes, out_shape=out.shape, transform=transform, fill=0, all_touched=True, dtype=np.uint8)
        out = np.where(burned > 0, burned, out)
    return out
# --------------------------------------------------------------------------- #
def find_rich_location(size_km: float, resolution: int, attempts: int = 100):
    for i in range(attempts):
        lat = AS_LAT
        lon = AS_LON

        bbox = create_small_bbox_around_point(lat, lon, size_km)
        print(f"\n[{i+1}/{attempts}] Zkouším bod {lat:.4f}, {lon:.4f}")
        layers = download_layers(bbox)
        if all(not gdf.empty for gdf in layers.values()):
            print(f"Vhodná oblast nalezena: {lat:.4f}, {lon:.4f}")
            return lat, lon, layers, bbox
        else:
            print("✗ Některé vrstvy chybí.")
    print("Nebyla nalezena žádná vhodná oblast.")
    return None, None, None, None
# --------------------------------------------------------------------------- #
def random_bbox(base_bbox, max_ratio=0.25):
    # Posune bbox náhodně v rámci povoleného podílu jeho velikosti.
    west, south, east, north = base_bbox.bounds
    w_span  = east - west
    h_span  = north - south
    dx = (random.uniform(-max_ratio, max_ratio)) * w_span
    dy = (random.uniform(-max_ratio, max_ratio)) * h_span
    return box(west + dx, south + dy, east + dx, north + dy)
# --------------------------------------------------------------------------- #
def save_matrix_txt(mat: np.ndarray, path: pathlib.Path):
    np.savetxt(path, mat, fmt="%d")
# --------------------------------------------------------------------------- #
def generate_dataset(base_bbox, master_layers):
    DATASET_DIR.mkdir(exist_ok=True)
    saved   = 0
    attempt = 0

    while saved < N_SAMPLES:
        attempt += 1
        bbox   = random_bbox(base_bbox, OFFSET_RATIO)
        layers = clip_layers(master_layers, bbox)
        if sum(len(gdf) for gdf in layers.values()) == 0:
            continue  # prázdno, další pokus

        matrix = rasterise(layers, bbox.bounds, RESOLUTION)

        for tile in split_tiles(matrix, TILE_SIZE):
            # ověř všechny definované cíle
            if all(
                abs(pct_class(tile, cls) - target) <= tol
                for cls, (target, tol) in TARGETS.items()
            ):
                # dlaždice sedí, uložíme (včetně otoček)
                base_name = f"sample_{saved:05d}"
                for k in ROTATE:
                    deg = k * 90
                    m_rot = np.rot90(tile, k)

                    for flip_mode in FLIP:
                        if flip_mode == "none":
                            transformed = m_rot
                        elif flip_mode == "h":
                            transformed = np.fliplr(m_rot)
                        elif flip_mode == "v":
                            transformed = np.flipud(m_rot)
                        else:
                            continue

                        fname = f"{base_name}_rot{deg}_flip{flip_mode}.txt"
                        save_matrix_txt(transformed, DATASET_DIR / fname)
                        saved += 1
                        if saved >= N_SAMPLES:
                            break
                    if saved >= N_SAMPLES:
                        break
            if saved >= N_SAMPLES:
                break

        #  průběžný report každých 100 pokusů
        if attempt % 100 == 0:
            print(f"[{attempt} pokusů] Uloženo {saved}/{N_SAMPLES} dlaždic…")

    print(f"✓ Hotovo: {saved} dlaždic uložených v {DATASET_DIR.resolve()}")
# --------------------------------------------------------------------------- #
def split_tiles(mat: np.ndarray, tile=TILE_SIZE):
    #Vytvoření seznamu všech dlaždic tile×tile.
    h, w = mat.shape
    assert h % tile == 0 and w % tile == 0, (
        f"Resolution {h}×{w} není násobkem {tile}"
    )
    tiles = []
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            tiles.append(mat[r:r + tile, c:c + tile])
    return tiles
# --------------------------------------------------------------------------- #
def pct_class(tile: np.ndarray, class_val: int) -> float:
    return np.count_nonzero(tile == class_val) / tile.size
# --------------------------------------------------------------------------- #
def main():
    if AUTOSEARCH:
        lat, lon, layers, small_area = find_rich_location(SIZE_KM, RESOLUTION)
        if lat is None:
            print("Přerušeno – žádná vhodná oblast")
            return
    else:
        if LOCATION in locations:
            lat, lon = locations[LOCATION]
            print(f"Používám předdefinovanou lokaci {LOCATION}: {lat}, {lon}")
        else:
            print(f"Hledám souřadnice pro: {LOCATION}")
            point = ox.geocode(LOCATION)
            lat, lon = point.y, point.x
            print(f"Nalezeno: {lat}, {lon}")

        small_area = create_small_bbox_around_point(lat, lon, SIZE_KM)
        master_layers, _ = download_master_layers(lat, lon, SIZE_KM, MASTER_PAD)
        layers = clip_layers(master_layers, small_area)

    print(f"Oblast: {SIZE_KM} km × {SIZE_KM} km kolem [{lat:.4f}, {lon:.4f}]")

    total_features = sum(len(gdf) for gdf in layers.values())
    for name, gdf in layers.items():
        print(f"{name}: {len(gdf)} prvků")
    if total_features == 0:
        print("Nebyla nalezena žádná OSM data v této oblasti!")

    matrix = rasterise(layers, small_area.bounds, RESOLUTION)
# ----------------------------------------------------------------------
    if ROAD_CONNECTION == True:
        road_val = TAG_TABLE["roads"]
        water_val = TAG_TABLE["water"]

        # Ulož si původní pro vizualizaci
        before = matrix.copy()
        roads = (matrix == road_val).copy()

        # pro propojení silnic, spojíme je A* jen po ortogonále a mimo vodu
        struct4 = generate_binary_structure(2, 1)
        lbl, ncmp = label(roads, structure=struct4)
        if ncmp > 1:
            counts = np.bincount(lbl.flat)[1:]
            main_id = counts.argmax() + 1
            mainpix = list(zip(*np.where(lbl == main_id)))

            G = nx.grid_2d_graph(*matrix.shape)
            for y, x in zip(*np.where(matrix == water_val)):
                if (y, x) in G:  # defensive
                    G.remove_node((y, x))

            for cid in range(1, ncmp + 1):
                if cid == main_id:
                    continue
                comp = list(zip(*np.where(lbl == cid)))
                d = cdist(comp, mainpix, metric="cityblock")
                i, j = divmod(d.argmin(), d.shape[1])
                start, goal = comp[i], mainpix[j]
                try:
                    path = nx.astar_path(G, start, goal,
                                         heuristic=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]))
                    for y, x in path:
                        roads[y, x] = True
                except nx.NetworkXNoPath:
                    print(f" Komponenta {cid} se kvůli vodě nepřipojila.")

            # skeleton, aby nové spojky nezbytně zůstaly 1‑px
            roads = skeletonize(roads)
            matrix[matrix == road_val] = 0
            matrix[roads] = road_val

            # vizualizace před a po
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(before, cmap="viridis", interpolation="nearest", origin="upper")
            ax[0].set_title("Před úpravou")
            ax[0].axis("off")

            ax[1].imshow(matrix, cmap="viridis", interpolation="nearest", origin="upper")
            ax[1].set_title("Po úpravě")
            ax[1].axis("off")
            plt.tight_layout()
            plt.show()
# ----------------------------------------------------------------------
    if BUILDING_SIMPLE:
        road_val = TAG_TABLE["roads"]
        bld_val = TAG_TABLE["buildings"]
        water_val = TAG_TABLE["water"]

        roads = (matrix == road_val)
        water = (matrix == water_val)

        # povolené pixely pro budovy = 4‑sousedství silnic, mimo vodu a mimo samotné silnice
        struct4 = generate_binary_structure(2, 1)
        allowed = binary_dilation(roads, structure=struct4) & (~roads) & (~water)

        # vyčisti všechny stávající budov
        matrix[matrix == bld_val] = 0
        # zapiš jen ty nové
        matrix[allowed] = bld_val
    # ----------------------------------------------------------------------
    if DATASET:
        print("=== GENERUJU DATASET ===")
        generate_dataset(small_area, master_layers)
        print("Hotovo – uloženy výsledky ve složce 'dataset/'")
        return
    # ----------------------------------------------------------------------
    np.savetxt(OUT, matrix, fmt="%d")
    print(f"Uložena matice {matrix.shape} do {OUT}")
    print(f"Unikátní hodnoty v matici: {np.unique(matrix)}")

    bounds = small_area.bounds
    real_width_km = (bounds[2] - bounds[0]) * 111.0
    resolution_m = (real_width_km * 1000) / RESOLUTION
    print(f"Rozlišení: ~{resolution_m:.1f} metrů na pixel")

    prev_dir = pathlib.Path(PREVIEW_DIR)
    prev_dir.mkdir(exist_ok=True)

    if total_features > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        for name, gdf in layers.items():
            if gdf.empty:
                continue
            color = {"roads": "gray", "buildings": "firebrick", "vegetation": "forestgreen", "water": "blue"}.get(name, "black")
            gdf.plot(ax=ax, color=color, linewidth=0.5, alpha=0.8)
        ax.set_title(f"OSM vrstvy – {LOCATION}\n{SIZE_KM}km × {SIZE_KM}km")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(prev_dir / "original_map.png", dpi=180, bbox_inches="tight")
        plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, interpolation="nearest", cmap="viridis", origin="upper")
    plt.title(f"Mapa tříd {RESOLUTION}×{RESOLUTION}\n0=prázdno, 1=budovy, 2=prostředí, 3=voda, 4=cesty")
    plt.colorbar(label="Třída")
    plt.tight_layout()
    plt.savefig(prev_dir / "class_map.png", dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()