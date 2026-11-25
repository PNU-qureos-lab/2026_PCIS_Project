# PCIS 2026 – Data description

This page describes **every dataset used in PCIS 2026**: where it is, what it contains, where it comes from, how the
pipeline uses it, and its terms of use.

The data are in two places:

| Place | What is there | Why |
|---|---|---|
| **GitHub** (this repository) | code, Sentinel-2 example, all support data (look-up tables, model coefficients, bottom spectra), GEBCO subsets, small masks | free to share |
| **Drive** – `PCIS_2026_DATA.zip` (link from the PCIS team) | all raw Sentinel-2 (4 areas) and Planet Dove (4 areas) scenes, shallow-water outlines of all 8 scenes, HydroLight library | too large for GitHub, or Planet Dove licence (no public sharing) |

All names follow the package naming rule `<Sensor>_<Area>_<YYYYMMDD>` (sensor = `Sentinel2` or `PlanetDove`).

---

## Contents

- [A. Satellite images](#a-satellite-images)
  - [A1. Sentinel-2 Level-2A](#a1-sentinel-2-level-2a)
  - [A2. Planet Dove (PlanetScope SuperDove)](#a2-planet-dove-planetscope-superdove)
- [B. Reference data](#b-reference-data)
  - [B1. GEBCO bathymetry](#b1-gebco-bathymetry)
  - [B2. HydroLight reference library](#b2-hydrolight-reference-library)
  - [B3. Land and shallow-water outlines](#b3-land-and-shallow-water-outlines)
  - [B4. Reviewed Hedley training zone (Planet Dove, Tidung)](#b4-reviewed-hedley-training-zone-planet-dove-tidung)
- [C. Support data used by the scripts](#c-support-data-used-by-the-scripts)
  - [C1. ACOLITE look-up tables](#c1-acolite-look-up-tables)
  - [C2. Semi-analytical model: coefficients, IOP spectra, bottom spectra](#c2-semi-analytical-model-coefficients-iop-spectra-bottom-spectra)
- [D. Example outputs](#d-example-outputs)
- [E. Drive zip: folder structure](#e-drive-zip-folder-structure)

---

## A. Satellite images

### A1. Sentinel-2 Level-2A

| | |
|---|---|
| **What** | ESA Copernicus Sentinel-2 MSI, Level-2A (Sen2Cor bottom-of-atmosphere surface reflectance), original `.SAFE` products |
| **Source** | Copernicus Data Space Ecosystem, <https://dataspace.copernicus.eu> (script `01_download_sentinel2.py` downloads new scenes) |
| **Bands used** | B01 443 nm (60 m), B02 490, B03 560, B04 665 nm, B11 1610 and B12 2190 nm (20 m), SCL scene classification, AOT aerosol optical thickness, `MTD_MSIL2A.xml` / `MTD_TL.xml` (offset, scale, sun and view angles) |
| **Format** | SAFE folder: JPEG2000 `uint16` bands (reflectance x 10000 with offset) + XML metadata |
| **Used by** | script 02 (glint correction) → script 04 (model) |
| **Terms** | free and open (Copernicus Sentinel data licence); cite "contains modified Copernicus Sentinel data 2020–2026" |

| Scene name | Site | Date | Satellite | Tile(s) | Where |
|---|---|---|---|---|---|
| `Sentinel2_Tidung_20260808` | Tidung, Indonesia | 2026-08-08 | S2A | T48MXU | **Drive** (full scene); also on **GitHub** `input_example/sentinel2_L2A_scene/` (only the 8 bands + 2 XML files the pipeline reads, 143 MB) |
| `Sentinel2_VitiLevu_20240520` | Viti Levu, Fiji | 2024-05-20 | S2A | T60KWE, T60KWF, T60KWG, T60KXE, T60KXF, T60KXG (6 tiles) | **Drive** |
| `Sentinel2_Tongatapu_20200522` | Tongatapu, Tonga | 2020-05-22 | S2A | T01KFS, T01KGS (2 tiles) | **Drive** |
| `Sentinel2_Funafuti_20220506` | Funafuti, Tuvalu | 2022-05-06 | S2B | T60LYR | **Drive** |

In the Drive zip each tile is one file `<product>.SAFE.zip` (for example
`S2A_MSIL2A_20260808T030151_N0512_R032_T48MXU_20260808T123808.SAFE.zip`), with the original `.SAFE` folder inside.
Script 02 reads it directly: `--safe <product>.SAFE.zip` (no unzipping needed).

### A2. Planet Dove (PlanetScope SuperDove)

| | |
|---|---|
| **What** | PlanetScope `PSScene` deliveries from SuperDove satellites: 8-band surface reflectance, clipped to the order area |
| **Source** | Planet Labs PBC, ordered by the PCIS project (order number in the folder name) |
| **Files per strip** | `*_3B_AnalyticMS_SR_8b_clip.tif` (surface reflectance x 10000, `uint16`, 8 bands 444–866 nm, 3 m), `*_3B_udm2_clip.tif` (usable-data mask: clear, snow, shadow, haze, cloud…, 8 bands `uint8`), `*_metadata.json` (sun angles), `*_metadata_clip.xml` |
| **Format** | one ZIP per site, containing all dates of the order |
| **Used by** | script 03 (all strips of the chosen date are joined into one 3 m mosaic; all dates are used for the stable-land mask) → script 04 |
| **Terms** | **Planet licence: not for public redistribution.** Shared only through the PCIS Drive with project partners. Not on GitHub |
| **Where** | **Drive** `2_raw_PlanetDove_SR_UDM2/` |

| File in the Drive zip | Site | Date processed | Planet order | Strips on that date | Whole delivery |
|---|---|---|---|---|---|
| `PlanetDove_Tidung_20251117_SR_UDM2_delivery.zip` | Tidung, Indonesia | 2025-11-17 | 8420286 | 1 | 4 strips, 4 dates (2025-10-18 to 2026-06-27) |
| `PlanetDove_Tongatapu_20260422_SR_UDM2_delivery.zip` | Tongatapu, Tonga | 2026-04-22 | 8740134 | 3 | 22 strips, 8 dates (2024-01-27 to 2026-05-11) |
| `PlanetDove_Funafuti_20240616_SR_UDM2_delivery.zip` | Funafuti, Tuvalu | 2024-06-16 | 7377012 | 2 | 13 strips, 4 dates (2023-12-04 to 2024-06-16) |
| `PlanetDove_Savave_20240608_SR_UDM2_delivery.zip` | Savave, Tuvalu | 2024-06-08 | 7359850 | 2 | 9 strips, 5 dates (2023-11-21 to 2024-06-08) |

The zips are the original Planet deliveries (only renamed). Script 03 reads them directly: `--planet <file>.zip --date YYYYMMDD`.
The Tidung zip is also the Planet Dove example input of the package (see [D](#d-example-outputs)).

---

## B. Reference data

### B1. GEBCO bathymetry

| | |
|---|---|
| **What** | Gridded sea-floor elevation (m, negative = below sea level), cut to each site |
| **Source** | GEBCO 2026 grid, GEBCO Compilation Group, <https://www.gebco.net>, 15 arc-second (~450 m) |
| **Files** | `GEBCO2026_Tongatapu.tif`, `GEBCO2026_Funafuti.tif`, `GEBCO2026_Savave.tif` (GeoTIFF `int16`, EPSG:4326, elevation in m); `GEBCO_Grid_documentation.pdf`, `GEBCO_Grid_terms_of_use.pdf` |
| **Used by** | **optional.** Script 04 step 1, option `--gebco <file>`: water 0–30 m deep is analysed (`--max-depth-m` to change). Only for a new area without a reviewed shallow-water outline. The grid is coarse (~450 m), so check the mask on the PNG map. The delivered PCIS results did not use GEBCO; they used the reviewed outlines (B3) |
| **Where** | **GitHub** [`reference_data/bathymetry_GEBCO/`](reference_data/bathymetry_GEBCO/). For another area, cut a subset from the global grid at <https://download.gebco.net> |
| **Terms** | free to use with attribution: "GEBCO Compilation Group (2026) GEBCO 2026 Grid" (see the terms-of-use PDF) |

### B2. HydroLight reference library

| | |
|---|---|
| **What** | 40 completed HydroLight 5.1.4 radiative-transfer simulations of open Pacific water: modelled water-leaving Rrs spectra |
| **Coverage** | water cases PW, VC, CLR, LC; sun zenith 20, 25, 30, 35, 40, 45, 50, 55°; wind 5 m/s (plus 2 and 8 m/s for VC and CLR at 30° and 50°); 400–900 nm every 5 nm; Sentinel-2 band-convolved tables for S2A, S2B and S2C |
| **Files** | `01_INPUTS` (HydroLight input files), `02_RAW_HYDROLIGHT_OUTPUTS` (all output files), `03_HYPERSPECTRAL_RRS` (Rrs CSV), `04_SENTINEL2_BAND_LIBRARY` (Rrs per Sentinel-2 band), `05_FIGURES`, `06_CODE` (builder and look-up code), `07_METADATA` (run manifest, variable dictionary) |
| **Used for** | evidence for the Sentinel-2 Cycle 1 residual-glint test (not applied: all delivered scenes kept Cycle 0) and a plausibility check of deep-water spectra. **Not read by the 4 scripts.** It is a synthetic envelope, not ground truth; field or AERONET-OC Rrs is still needed for independent validation |
| **Where** | **Drive** `3_reference/HydroLight_reference_library/` (592 MB, with its own `README.md`). HydroLight itself is licensed software and is not included; the library can be used without it |

### B3. Land and shallow-water outlines

| | |
|---|---|
| **What** | Outlines of land and of water where the sea floor is visible, for all 8 scenes (4 Sentinel-2 at 10 m, 4 Planet Dove at 3 m) |
| **How made** | rules on the full-resolution image, then checked by eye; rules, thresholds and code are in `criteria_and_code/` (criteria PDF + Python) |
| **Folders** | one per scene, named like the scenes above: `Sentinel2_Tidung_20260808`, `Sentinel2_VitiLevu_20240520`, `Sentinel2_Tongatapu_20200522`, `Sentinel2_Funafuti_20220506`, `PlanetDove_Tidung_20251117`, `PlanetDove_Tongatapu_20260422`, `PlanetDove_Funafuti_20240616`, `PlanetDove_Savave_20240608` |
| **Files per scene** | `<scene>_shallow_water_mask_10m.tif` / `_3m.tif` (1 = sea floor visible), `<scene>_land_mask_10m.tif` / `_3m.tif`, shapefiles `<scene>_shallow_water.shp`, `<scene>_land.shp`, `<scene>_small_features.shp`, check pictures `<scene>_*.png` |
| **Used by** | script 04 step 1, option `--shallow-outline <scene>_shallow_water_mask_*.tif`: a model pixel is analysed when at least half of it is inside the outline. Recommended mask; used for all delivered results |
| **Where** | all 8 scenes in **Drive** `3_reference/shallow_water_outlines/`; the two Tidung masks are also on **GitHub** `input_example/shallow_water_outline/` (`Sentinel2_Tidung_20260808_shallow_water_outline.tif`, `PlanetDove_Tidung_20251117_shallow_water_outline.tif`) |
| **Terms** | PCIS project product |

### B4. Reviewed Hedley training zone (Planet Dove, Tidung)

| | |
|---|---|
| **What** | Mask of open deep water chosen by hand to train the Hedley glint correction of the delivered Tidung Planet product (1 = training pixel, 3 m grid) |
| **Used by** | script 03 step 3, optional `--training-zone`. Without it, script 03 picks the training water automatically (difference 0.0005–0.001 sr⁻¹ in Rrs) |
| **Where** | **GitHub** `input_example/planetdove_reviewed_Hedley_training_zone/` |

---

## C. Support data used by the scripts

### C1. ACOLITE look-up tables

| | |
|---|---|
| **What** | ACOLITE atmospheric look-up tables `ACOLITE-LUT-202110-MOD2-1013mb` (maritime aerosol model, 1013 mb) for S2A, S2B and S2C MSI, plus the ACOLITE code |
| **Source** | ACOLITE, RBINS (Vanhellemont 2019), <https://github.com/acolite/acolite>, commit in `VERSION_USED.txt` |
| **Format** | NetCDF (3 files, 145 MB in total) |
| **Used by** | script 02 step 3: two-way transmittance and Fresnel terms → glint transfer factor per band |
| **Where** | **GitHub** `support_02_ACOLITE_code_and_lookup_tables/` |
| **Terms** | GPL-3 |

### C2. Semi-analytical model: coefficients, IOP spectra, bottom spectra

| File | What | Source | Used by |
|---|---|---|---|
| `DATA/coefficients/Sentinel2.json`, `SuperDove.json` | fitted model coefficients (g0, g1) per wavelength for each sensor | PCIS model calibration (Version 42 v2) | script 04 steps 3–4 |
| `DATA/iop_spectra.json` | absorption / scattering spectra of pure water, chlorophyll and mineral particles | model data, unchanged | script 04 steps 3–4 |
| `DATA/bottom_reflectances/coral_sand.txt` | clean coral sand reflectance | Maritorena, Morel & Gentili (1994), *Limnol. Oceanogr.* 39(7), 1689–1703, Fig. 6 | 2 and 3 classes |
| `DATA/bottom_reflectances/avg_seagrass.txt` | average seagrass reflectance | CoBOP experiment, Lee Stocking Island, Bahamas (C. Mazel and M. Lesser) | 2 and 3 classes |
| `DATA/bottom_reflectances/avg_coral.txt` | average coral reflectance | CoBOP experiment, Lee Stocking Island, Bahamas (C. Mazel and M. Lesser) | 3 classes |
| other `*.txt` in `bottom_reflectances/` | further bottom types of the same library (algae, sediment, kelp…) | same library | not used by default |

All in **GitHub** `support_04_semi_analytical_model_V42v2/`.

---

## D. Example outputs

| Output | Where |
|---|---|
| Sentinel-2 Tidung 2026-08-08: scripts 02 and 04, all steps | **GitHub** `output_example/Sentinel2_Tidung_20260808/` |
| Planet Dove Tidung 2025-11-17: scripts 03 and 04, all steps | not stored (contains Planet imagery): made on your computer when you run the Planet example (below) |

**Planet Dove example:** copy `2_raw_PlanetDove_SR_UDM2/PlanetDove_Tidung_20251117_SR_UDM2_delivery.zip` from the Drive zip to
`input_example/planetdove_SR_UDM2_delivery/` of the cloned repository, then run `bash run_example_linux.sh`.
Without it, the example runs Sentinel-2 only.

Every output file, its format and the step that makes it are listed in [`PCIS_pipeline_steps.xlsx`](PCIS_pipeline_steps.xlsx), sheet `3_Steps_input_output`.

---

## E. Drive zip: folder structure

```
PCIS_2026_DATA/
├── README_DATA.md                                       this page
├── 1_raw_Sentinel2_L2A/                                 (A1) one folder per scene, one <product>.SAFE.zip per tile
│   ├── Sentinel2_Tidung_20260808/                       1 tile
│   ├── Sentinel2_VitiLevu_20240520/                     6 tiles
│   ├── Sentinel2_Tongatapu_20200522/                    2 tiles
│   └── Sentinel2_Funafuti_20220506/                     1 tile
├── 2_raw_PlanetDove_SR_UDM2/                            (A2) original Planet deliveries, one zip per area
│   ├── PlanetDove_Tidung_20251117_SR_UDM2_delivery.zip
│   ├── PlanetDove_Tongatapu_20260422_SR_UDM2_delivery.zip
│   ├── PlanetDove_Funafuti_20240616_SR_UDM2_delivery.zip
│   └── PlanetDove_Savave_20240608_SR_UDM2_delivery.zip
└── 3_reference/
    ├── shallow_water_outlines/                          (B3) one folder per scene (8) + criteria_and_code/
    └── HydroLight_reference_library/                    (B2) with its own README.md
```
