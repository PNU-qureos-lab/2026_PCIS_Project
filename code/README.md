# Processing code

This is the compact **code-only** version of the PCIS 2026 workflow. Raw Sentinel-2 and Planet Dove scenes are intentionally not stored in GitHub.

## Workflow order

```text
01_SENTINEL2_DOWNLOAD
        |
        v
02_SENTINEL2_GLINT_CORRECTION ----+
                                   +--> 04_SEMI_ANALYTICAL_V42
03_PLANET_DOVE_GLINT_CORRECTION ---+
```

## Python environment

Use 64-bit Python 3.12. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib rasterio netCDF4 requests tqdm PyYAML Pillow
```

The separate full delivery package also contains a tested portable Python interpreter and offline wheels.

## Run each workflow

1. [`01_SENTINEL2_DOWNLOAD/README.md`](01_SENTINEL2_DOWNLOAD/README.md) — search and download additional Sentinel-2 L2A scenes.
2. [`02_SENTINEL2_GLINT_CORRECTION/README.md`](02_SENTINEL2_GLINT_CORRECTION/README.md) — prepare Sentinel-2 bands, run ACOLITE-based glint correction, and export Rrs.
3. [`03_PLANET_DOVE_GLINT_CORRECTION/README.md`](03_PLANET_DOVE_GLINT_CORRECTION/README.md) — run Standard Hedley correction on Planet Dove analytic surface reflectance plus UDM2.
4. [`04_SEMI_ANALYTICAL_V42/README.md`](04_SEMI_ANALYTICAL_V42/README.md) — select deep-water Rrs and estimate depth and bottom fractions.

## Required user inputs

For a new area, provide:

- Sentinel-2: a complete L2A SAFE archive or the required aligned BOA bands, scene metadata, masks, GEBCO subset, and the documented ACOLITE/HydroLight supporting resources.
- Planet Dove: eight-band analytic surface reflectance, its matching UDM2 mask, GEBCO subset, and a reviewed offshore training-water area.
- Semi-analytical processing: final corrected Rrs, valid-water/land/cloud masks, GEBCO, sensor configuration, water coefficients, and benthic endmembers.

Do not write generated products inside the source-code folders. Choose a separate result directory for each scene. The full delivery package provides the standard input/result folder structure and complete worked examples.
