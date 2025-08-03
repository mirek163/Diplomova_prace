"""
Metriky pro 3D GAN na voxelových datech + ukládání grafů metrik.
---------------------------------------
• Chamfer, EMD, Jensen-Shannon, Coverage & MMD
• FID  (na vícenásobných 2D renderovaných pohledech)
----------------------------------------------------------
Závislosti navíc (FPD):
    pip install torch                   # >=2.1.0 / odpovídající CUDA
    Pozn: Pro pytorch3d jsem potřeboval downgradovat cudatoolkit na 11.8 a torch na 2.1.x (ne 2.5.0, jako u WGanu)
  pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.1.0cu118
----------------------------------------------------------
"""
import os, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import entropy           #JSD
from matplotlib import pyplot as plt
from pytorch3d.loss import chamfer_distance
from geomloss import SamplesLoss
from pytorch_fid import fid_score
from scipy.cluster.hierarchy import linkage, leaves_list

import processing_obj as prc
from DCGAN import Generator, LATENT_DIM

# === PARAMETRY ===
DATASET_DIR       = 'BuildingB-rotation_and_floors'
WEIGHTS_DIR       = 'weights'
EPOCHS            = [7500]          #vahy, které chci vyhodnotit
NUM_GENERATED     = 1000
VOXEL_GRID_SIZE   = 64
VOXEL_THRESHOLD   = 0.5
PN_NUM_POINTS     = 1024
IMG_VIEWS         = [(30,  30), (30,120), (60, 45), (60,135)]
REAL_IMG_DIR      = 'real_images'
GEN_IMG_DIR       = 'gen_images'                 #složka pro rendery generovaných voxelů
PLOT_DIR          = 'plots'               #složka pro obrázky s grafy
CSV_PATH          = 'selected_metrics.csv'
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECK_SPECIFIC_WEIGHT = True # přepnutí pro heatmapy a histogramy pro konkrétní váhu

emd_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

# ----------------------------------------------------------
#Utility
# ----------------------------------------------------------

def voxel_to_pointcloud(voxel, threshold: float = VOXEL_THRESHOLD):
    coords = np.argwhere(voxel > threshold)
    return torch.from_numpy(coords.astype(np.float32))


def load_dataset_pcs(dataset_dir: str | Path, grid_size: int = VOXEL_GRID_SIZE):
    """Načte všechny *.obj souboru v datasetu -> point‑cloudy + voxely."""
    pcs, voxels = [], []
    obj_files = [f for f in os.listdir(dataset_dir) if f.endswith('.obj')]
    for fname in tqdm(obj_files, desc='Načítání datasetu'):
        vox = prc.obj_to_voxel(os.path.join(dataset_dir, fname), grid_size=grid_size)
        pcs.append(voxel_to_pointcloud(vox))
        voxels.append(vox.astype(np.float32))
    return pcs, voxels


def generate_pcs(generator: torch.nn.Module,
                 num_samples: int = NUM_GENERATED,
                 threshold: float = VOXEL_THRESHOLD):

    pcs, voxels = [], []
    generator.eval().to(DEVICE)
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc='Generování vzorků'):
            z   = torch.randn(1, LATENT_DIM, device=DEVICE)
            vox = generator(z).cpu().numpy().squeeze()
            vox = (vox > threshold).astype(np.float32)
            pcs.append(voxel_to_pointcloud(vox))
            voxels.append(vox)
    return pcs, voxels

#  Chamfer a EMD

def compute_chamfer_matrix(pcs1, pcs2):
    D = np.zeros((len(pcs1), len(pcs2)), dtype=np.float32)
    for i, pc1 in enumerate(tqdm(pcs1, desc='Chamfer: data')):
        p1 = pc1.to(DEVICE).unsqueeze(0)
        for j, pc2 in enumerate(pcs2):
            p2 = pc2.to(DEVICE).unsqueeze(0)
            dist, _ = chamfer_distance(p1, p2)
            D[i, j] = dist.item()
    return D


def sample_points(pc, N: int = PN_NUM_POINTS):
    if pc.size(0) >= N:
        idx = torch.randperm(pc.size(0))[:N]
        return pc[idx]
    pad = pc[-1].unsqueeze(0).repeat(N - pc.size(0), 1)
    return torch.cat([pc, pad], dim=0)


def compute_emd_matrix(pcs1, pcs2):
    E = np.zeros((len(pcs1), len(pcs2)), dtype=np.float32)
    for i, pc1 in enumerate(tqdm(pcs1, desc='EMD: data')):
        pts1 = sample_points(pc1)
        for j, pc2 in enumerate(pcs2):
            pts2 = sample_points(pc2)
            t1 = pts1.unsqueeze(0).to(DEVICE)
            t2 = pts2.unsqueeze(0).to(DEVICE)
            E[i, j] = emd_loss(t1, t2).item()
    return E

# JSD, Coverage/MMD

def compute_jsd(vox_ds, vox_g):
    ds  = np.stack([(v > VOXEL_THRESHOLD).astype(np.float32) for v in vox_ds])
    gen = np.stack([(v > VOXEL_THRESHOLD).astype(np.float32) for v in vox_g])
    p, q = ds.mean(0).flatten(), gen.mean(0).flatten()
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def compute_cov_mmd(ch):
    mmd = ch.min(axis=1).mean()
    nearest = np.argmin(ch, axis=0)
    cov = len(np.unique(nearest)) / ch.shape[0]
    return cov, mmd

# Render pro FID

def render_voxels(voxels, out_dir: str | Path):
    os.makedirs(out_dir, exist_ok=True)
    for idx, vox in enumerate(tqdm(voxels, desc=f'Render -> {out_dir}')):
        fig = plt.figure(figsize=(3, 3))
        ax  = fig.add_subplot(111, projection='3d')
        ax.voxels(vox > VOXEL_THRESHOLD)
        ax.view_init(*IMG_VIEWS[idx % len(IMG_VIEWS)])
        ax.set_axis_off()
        plt.savefig(os.path.join(out_dir, f"{idx:04d}.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def save_metric_plots(df: pd.DataFrame, out_dir: str | Path = PLOT_DIR):
    """Vygeneruje a uloží grafy jednotlivých metrik u epoch"""
    os.makedirs(out_dir, exist_ok=True)

    metrics = ['FID', 'Chamfer', 'EMD', 'JSD', 'MMD', 'overall_score']
    df_sorted = df.sort_values('epoch')

    for metric in metrics:
        plt.figure()
        plt.plot(df_sorted['epoch'], df_sorted[metric], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} přes epochy')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{metric}.png')
        plt.close()

    #rychlé porovnání trendů
    plt.figure(figsize=(8, 5))
    for metric in metrics[:-1]:  #bez overall_score
        plt.plot(df_sorted['epoch'], df_sorted[metric], marker='o', label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Hodnota metriky')
    plt.title('Všechny metriky (nižší = lepší)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'all_metrics.png')
    plt.close()

# rychlé kreslení pro jedinou váhu
def reorder_matrix(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vrátí přeuspořádanou matici + indexy řádků a sloupců.
    """
    row_order = leaves_list(linkage(M,   method="average"))
    col_order = leaves_list(linkage(M.T, method="average"))
    return M[row_order][:, col_order], row_order, col_order


def save_specific_weight_plots(epoch: int):
    Cfile, Efile = f"chamfer_{epoch}.npy", f"emd_{epoch}.npy"
    if not (os.path.isfile(Cfile) and os.path.isfile(Efile)):
        warnings.warn(f"Soubor {Cfile} / {Efile} nenalezen.")
        return

    C, E = np.load(Cfile), np.load(Efile)
    out = Path(PLOT_DIR) / f"specific_weight_{epoch}"
    out.mkdir(parents=True, exist_ok=True)

    for name, M in [("Chamfer", C), ("EMD", E)]:
        # ---------- HEATMAPA (nesetříděná) ----------
        plt.figure()
        plt.imshow(M, cmap="viridis", aspect="auto", origin="lower")
        plt.colorbar(label=f"{name} vzdálenost")
        plt.xlabel("Index generovaného objektu")
        plt.ylabel("Index reálného objektu")
        plt.title(f"Matice {name} – epocha {epoch}")
        plt.tight_layout()
        plt.savefig(out / f"{name}_heatmap.png")
        plt.close()

        # ---------- HEATMAPA (setříděná) ----------
        M_sorted, rows, cols = reorder_matrix(M)
        plt.figure()
        plt.imshow(M_sorted, cmap="viridis", aspect="auto", origin="lower")
        plt.colorbar(label=f"{name} vzdálenost")
        plt.xlabel("Gerovaný index (po řazení)")
        plt.ylabel("Reálný index (po řazení)")
        plt.title(f"{name} – seřazeno clustrem | epocha {epoch}")
        plt.tight_layout()
        plt.savefig(out / f"{name}_heatmap_sorted.png")
        plt.close()

        # ---------- HISTOGRAM ----------
        plt.figure()
        plt.hist(M.mean(axis=1), bins=30, label="reálný → generovaný", alpha=0.7)
        plt.hist(M.mean(axis=0), bins=30, label="generovaný → reálný", alpha=0.7)
        plt.legend()
        plt.grid(True)
        plt.xlabel(f"{name} vzdálenost")
        plt.ylabel("Počet")
        plt.title(f"{name} na objekt – epocha {epoch}")
        plt.tight_layout()
        plt.savefig(out / f"{name}_hist.png")
        plt.close()


# ------------------------------------------------------------------

def main():
    print('Zařízení:', DEVICE)

    if CHECK_SPECIFIC_WEIGHT:
        # vezme první číslo z EPOCHY ( stejnak budu vyhodnocovat jen tu nejlepší vahu)
        epoch = EPOCHS[0]
        save_specific_weight_plots(epoch)
        return

    #Reálný dataset -> voxely a point‑cloudy
    pcs_ds, vox_ds = load_dataset_pcs(DATASET_DIR)

    #Jen jednou vykreslím reálné voxely pro FID
    if not os.path.isdir(REAL_IMG_DIR) or len(os.listdir(REAL_IMG_DIR)) == 0:
        render_voxels(vox_ds, REAL_IMG_DIR)

    results = []
    for epoch in EPOCHS:
        weight_file = os.path.join(WEIGHTS_DIR, f"generator_{epoch}-128_64_0.001.pth")
        if not os.path.isfile(weight_file):
            warnings.warn(f"Checkpoint nenalezen: {weight_file}.")
            continue

        gen = Generator(latent_dim=LATENT_DIM).to(DEVICE)
        gen.load_state_dict(torch.load(weight_file, map_location=DEVICE, weights_only=False))
        pcs_g, vox_g = generate_pcs(gen)

        C   = compute_chamfer_matrix(pcs_ds, pcs_g)
        E   = compute_emd_matrix(pcs_ds, pcs_g)
        jsd = compute_jsd(vox_ds, vox_g)
        cov, mmd = compute_cov_mmd(C)

        #FID
        run_gen_dir = os.path.join(GEN_IMG_DIR, f'e{epoch}')
        render_voxels(vox_g, run_gen_dir)
        fid = fid_score.calculate_fid_given_paths(
            [REAL_IMG_DIR, run_gen_dir], batch_size=32, device=str(DEVICE), dims=2048
        )

        results.append({
            'epoch':    epoch,
            'Chamfer':  float(C.mean()),
            'EMD':      float(E.mean()),
            'JSD':      jsd,
            'Coverage': cov,
            'MMD':      mmd,
            'FID':      float(fid),
        })
        np.save(f"chamfer_{epoch}.npy", C)
        np.save(f"emd_{epoch}.npy", E)

    if os.path.isfile(CSV_PATH):
        old_df = pd.read_csv(CSV_PATH)
        df = pd.concat([old_df, pd.DataFrame(results)]).drop_duplicates('epoch', keep='last')
    else:
        df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)

    #Vyhodnocení zvolených vah
    metrics_to_minimize = ['FID', 'Chamfer', 'EMD', 'JSD', 'MMD']
    for m in metrics_to_minimize:
        df[f'{m}_rank'] = df[m].rank(method='min')

    df['overall_score'] = df[[f'{m}_rank' for m in metrics_to_minimize]].mean(axis=1)

    save_metric_plots(df, PLOT_DIR)

    #Výpis nejlepší váhy
    best = df.sort_values('overall_score').iloc[0]
    print(best[['epoch', *metrics_to_minimize]])
    print(f"\nCelkově nejlepší váha je při zvolených epochách– epocha: {int(best.epoch)}")

if __name__ == '__main__':
    main()
