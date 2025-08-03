# Generování 3D prostředí pomocí GAN

Tento projekt vznikl v rámci diplomové práce a zabývá se generováním trojrozměrného prostředí pomocí metod strojového učení, konkrétně generativních adversariálních sítí (GAN). Cílem je automaticky vygenerovat věrohodné 3D modely objektů určitého stylu (v práci zvolen low-poly styl) a následně z nich sestavit ucelenou 3D scénu představující celé prostředí. Součástí řešení je také generování 2D mapy (rozmístění objektů v ploše), podle které se objekty rozmístí do scény. Projekt rovněž definuje a využívá kvantitativní metriky k hodnocení kvality generovaných modelů a scén, které posuzují věrnost tvaru a rozmanitost oproti reálným datům.

Podrobnější testy různých GAN modelů (DCGAN, ProGAN, 3DGAN) včetně vizualizací a výsledků nejsou součástí tohoto repozitáře kvůli jejich velikosti, ale jsou k dispozici na OneDrivu. 

## Struktura repozitáře

- **Dataset** – Obsahuje datasety pro trénování 3D objektů a map.
- **Objects_and_scripts** – Blender soubor, který obsahuje jednotlivé objekty včetně jejich variací a rozsekání.Je zde i skript na generování datasetu rotací modelů kolem jedné osy.
- **Object_generation** – Skripty pro generování 3D struktury objektů.
- **Map_generation** – Skripty pro generování, trénování 2D map a sestavení 3D scény.
- **Evaluation_metrics** – Skripty pro vyhodnocení kvality generovaných modelů.

## Obsah

### Generování 3D objektů (složka: `Object_generation`)

- `WGAN.py`: Trénování 3D WGAN-GP modelu pro generování voxelových 3D objektů.
- `generate_object.py`: Generování nových 3D objektů z natrénovaného modelu -> tedy výsledných vah a export do `.obj` formátu.
- `processing_obj.py`: Konverze `.obj` modelů na voxelovou reprezentaci a naopak, úpravy velikosti objektů a padding.
- `colorize_obj.py`: Přidání jednoduchých textur/barev na vygenerované objekty.
- `sparsity.py`: Analýza hustoty voxelových modelů.
- `fill_utils.py`: Vyplnění dutin objektů (pokud zvoleno)

### Generování 3D map (složka: `Map_generation`)

- `osm_to_dataset.py`: Stahování a úpravy mapových dat z OpenStreetMap, tvorba datasetu pro trénování.
- `train_maps.py`: Trénování GAN modelu pro generování 2D mapových podkladů.
- `generate_maps.py`: Vygenerování mapových podkladů z vah trénovací sítě.
- `gridmap_to_obj.py`: Sestavení kompletní 3D scény ze získané mapy a připravených 3D objektů.


### Hodnocení kvality (složka: `Evaluation_metrics`)

- `metrics.py`: Skript pro vyhodnocení kvality generovaných modelů pomocí metrik (Chamferova vzdálenost, EMD, JSD, Coverage, MMD, FID) a vygenerování grafů.
