# PCIS 2026 – Satellite bathymetry and seabed habitat pipeline (Linux)

From a satellite image to **water depth** and **seabed habitat maps** (sand, seagrass, coral) in shallow coastal water.
Works with **Sentinel-2** (20 m) and **Planet Dove** (3 m). Runs from the command line, with no GUI and no operator,
and every script tells an automatic system whether to continue (**PASS**) or skip the scene (**SKIP**).

> **Full step table:** [`PCIS_pipeline_steps.xlsx`](PCIS_pipeline_steps.xlsx) lists every step from 01 step1 to 04 step6.
> It has one row per input file and one row per output file, with the folder, content and format of each file,
> and where each file comes from or goes next. File names in the table are clickable links to this repository.
> It also holds every rule and threshold.
>
> **All data:** [`DATA.md`](DATA.md) describes every dataset used (Sentinel-2, Planet Dove, GEBCO,
> HydroLight library, shallow-water outlines, ACOLITE tables, model coefficients and bottom spectra):
> what it is, where it comes from, how the pipeline uses it, and whether it is on GitHub or in the PCIS Drive zip.
>
> **Planet Dove imagery is not on GitHub** (Planet licence: no public sharing). The Planet example and all raw
> scenes are in `PCIS_2026_DATA.zip` on the PCIS Drive; ask the PCIS team for the link.

---

## Contents

1. [Quick start](#1-quick-start)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [How the pipeline works](#3-how-the-pipeline-works)
4. [Running a new scene](#4-running-a-new-scene)
5. [Results](#5-results)
6. [Automatic operation: PASS / SKIP](#6-automatic-operation-pass--skip)
7. [Answers to the automation questions](#7-answers-to-the-automation-questions)
8. [Old package → this package](#8-old-package--this-package)
9. [Naming rule](#9-naming-rule)
10. [Things to know](#10-things-to-know)
11. [Sources and licences](#11-sources-and-licences)

---

## 1. Quick start

Needs Linux (tested on Ubuntu 26.04) and [Miniforge](https://github.com/conda-forge/miniforge).

```bash
git clone https://github.com/PNU-qureos-lab/2026_PCIS_Project.git
cd 2026_PCIS_Project
bash setup_linux.sh          # makes the conda environment "pcis" and checks GDAL, ACOLITE and the model
bash run_example_linux.sh    # runs the example: Sentinel-2 02 -> 04 (and Planet Dove 03 -> 04, see below)
```

The example run writes into [`output_example/`](output_example/) (the included files are what you should get).
For your own scene, one command does everything: see [section 4b](#4b-run-everything-with-one-command-glint-correction--final-maps).

**Planet Dove example:** copy `2_raw_PlanetDove_SR_UDM2/PlanetDove_Tidung_20251117_SR_UDM2_delivery.zip` from the
Drive zip into `input_example/planetdove_SR_UDM2_delivery/`, then run `bash run_example_linux.sh` again.
Without it, the script runs Sentinel-2 only and says so.
Tested with Python 3.12, GDAL 3.13 and rasterio 1.5.2.

---

## 2. What is in this repository

| Item | What it is |
|---|---|
| [`01_download_sentinel2.py`](01_download_sentinel2.py) | Search and download Sentinel-2 Level-2A scenes (Copernicus Data Space) |
| [`02_glint_correction_sentinel2.py`](02_glint_correction_sentinel2.py) | Sentinel-2 sun-glint correction (Glint-ACOLITE, Cycle 0) → Rrs, 4 bands, 20 m |
| [`03_glint_correction_planetdove.py`](03_glint_correction_planetdove.py) | Planet Dove: joins all strips of one date, sun-glint correction (Hedley 2005) → Rrs, 6 bands, 3 m |
| [`04_semi_analytical_bathymetry_habitat.py`](04_semi_analytical_bathymetry_habitat.py) | Shallow-water mask → automatic deep-water Rrs → semi-analytical model → habitat maps → PNG maps |
| [`PCIS_pipeline_steps.xlsx`](PCIS_pipeline_steps.xlsx) | Step table (inputs, outputs, formats), rules and thresholds, skip rules, validation |
| [`DATA.md`](DATA.md) | Description of every dataset, and what is on GitHub and what is in the Drive zip |
| [`support_02_ACOLITE_code_and_lookup_tables/`](support_02_ACOLITE_code_and_lookup_tables/) | Used by script 02: ACOLITE code (GPL-3) and the 3 Sentinel-2 look-up tables |
| [`support_04_semi_analytical_model_V42v2/`](support_04_semi_analytical_model_V42v2/) | Used by script 04: semi-analytical model (code, coefficients, bottom spectra), unchanged |
| [`reference_data/bathymetry_GEBCO/`](reference_data/bathymetry_GEBCO/) | Optional for script 04 (`--gebco`): GEBCO 2026 depth for Tongatapu, Funafuti, Savave (not used for the delivered results) |
| [`input_example/`](input_example/) | Example inputs, Tidung, Indonesia: Sentinel-2 2026-08-08 scene, shallow-water outlines, Planet training zone (Planet imagery: Drive zip) |
| [`output_example/`](output_example/) | Everything the scripts produce for the Sentinel-2 example (Planet example outputs: Drive zip) |
| [`run_scene_linux.sh`](run_scene_linux.sh) | Runs one new scene from glint correction to the final maps with one command (section 4b) |
| [`setup_linux.sh`](setup_linux.sh), [`run_example_linux.sh`](run_example_linux.sh), [`environment_linux.yml`](environment_linux.yml) | Setup, example run, conda environment |

### Excel sheets in [`PCIS_pipeline_steps.xlsx`](PCIS_pipeline_steps.xlsx)

| Sheet | Content |
|---|---|
| `0_Naming_rule` | How every folder and file is named |
| `1_Folder_structure` | The folder tree, with what each folder holds |
| `2_Overview` | One line per script |
| `3_Steps_input_output` | Every step, one row per input and output file: folder, file (link), content, format, comes from / used by |
| `4_Rules_and_thresholds` | Every rule and number used, with its source |
| `5_Automatic_skip_rules` | When a scene is skipped (SKIP) |
| `6_Validation` | How the results were checked against the delivered PCIS products |

---

## 3. How the pipeline works

```
 Sentinel-2                                   Planet Dove
 01_download_sentinel2.py                     (Planet delivery .zip: SR + UDM2)
   step1 search  -> step2 download .SAFE            |
          |                                         v
          v                                  03_glint_correction_planetdove.py
 02_glint_correction_sentinel2.py              step1 read all strips of the date (3 m)
   step1 read L2A on one 20 m grid             step2 cloud, land, water masks
   step2 cloud, land, water masks              step3 Hedley training water + glint slopes
   step3 ACOLITE glint factors                 step4 glint correction + decision
   step4 glint correction + RGB before/after   step5 Rrs 6 bands + QC decision
   step5 Rrs 4 bands + QC decision                  |
          |                                         |
          +--------------------+--------------------+
                               v
             04_semi_analytical_bathymetry_habitat.py
               step1 shallow-water analysis mask
               step2 deep-water Rrs reference (automatic)
               step3 model, 2 bottom types (sand, seagrass)
               step4 model, 3 bottom types (sand, seagrass, coral)
               step5 habitat classes (70 % rule and 50 % rule)
               step6 PNG maps + final QC decision
```

Each script writes one folder per step, and every file name says which step made it
(`<scene name>_step<N>_<content>`). The exact files of every step are in sheet `3_Steps_input_output`.

---

## 4. Running a new scene

### 4a. Get the image

* **Sentinel-2:** download with script 01 (Copernicus login only for the download), or use a `.SAFE.zip` from the Drive zip.
  One run = one tile; a scene with several tiles (e.g. Viti Levu, 6 tiles) is run once per tile.
  ```bash
  conda activate pcis
  export CDSE_USER=<your Copernicus user>; read -s CDSE_PASS; export CDSE_PASS
  python 01_download_sentinel2.py --aoi LON_MIN LAT_MIN LON_MAX LAT_MAX --start 2026-01-01 --end 2026-12-31 \
         --name Sentinel2_SITE --out downloads --download 1      # -> downloads/<product>.SAFE
  ```
* **Planet Dove:** the Planet delivery `.zip` (SR + UDM2). No download script (Planet orders are made by hand).

### 4b. Run everything with one command (glint correction → final maps)

```bash
# Sentinel-2: 02 glint correction -> 04 model, habitat, maps
bash run_scene_linux.sh --sensor Sentinel2 --input downloads/<product>.SAFE \
     --aoi LON_MIN LAT_MIN LON_MAX LAT_MAX --name Sentinel2_SITE_YYYYMMDD \
     --shallow-outline <outline.tif> --place "Site, Country"

# Planet Dove: 03 glint correction -> 04 model, habitat, maps
bash run_scene_linux.sh --sensor PlanetDove --input <delivery>.zip --date YYYYMMDD \
     --name PlanetDove_SITE_YYYYMMDD --auto-shallow --place "Site, Country"
```

* Choose **one** shallow-water mask: `--shallow-outline <tif>` (best), `--gebco <tif>` or `--auto-shallow`.
* Results go to `results/<name>/` (change with `--out`). At the end the script lists every step folder it wrote.
* If a script decides **SKIP** (e.g. too cloudy), the run stops there with exit code 2; the reason is in that step's `QC_decision.json`.

### 4c. Or run the scripts one by one

Same result as 4b, useful to re-run only one part:

```bash
python 02_glint_correction_sentinel2.py --safe downloads/<product>.SAFE --aoi LON_MIN LAT_MIN LON_MAX LAT_MAX \
       --name Sentinel2_SITE_YYYYMMDD --out results/Sentinel2_SITE_YYYYMMDD/02_glint_correction_sentinel2
#   or Planet Dove (3 m):
python 03_glint_correction_planetdove.py --planet <delivery>.zip --date YYYYMMDD \
       --name PlanetDove_SITE_YYYYMMDD --out results/PlanetDove_SITE_YYYYMMDD/03_glint_correction_planetdove

python 04_semi_analytical_bathymetry_habitat.py --sensor Sentinel2 \
       --glint-dir results/Sentinel2_SITE_YYYYMMDD/02_glint_correction_sentinel2 --name Sentinel2_SITE_YYYYMMDD \
       --shallow-outline <outline.tif> --place "Site, Country" \
       --out results/Sentinel2_SITE_YYYYMMDD/04_semi_analytical_bathymetry_habitat
```

For Planet Dove, use `--sensor PlanetDove` and the output folder of script 03 as `--glint-dir`.

| Option | Script | Meaning |
|---|---|---|
| `--aoi` | 01, 02 | area: LON_MIN LAT_MIN LON_MAX LAT_MAX (degrees) |
| `--name` | all | scene name `<Sensor>_<Area>_<YYYYMMDD>`; starts every output file name |
| `--max-cloud`, `--max-cloud-percent` | 01, 02 | cloud limit for search / for the scene QC (default 20 %) |
| `--date` | 03 | the Planet date to process (all strips of that date are joined) |
| `--training-zone` | 03 | optional reviewed Hedley training-water mask |
| `--shallow-outline` | 04 | reviewed shallow-water outline (best) |
| `--gebco` | 04 | GEBCO depth grid (examples in `reference_data/bathymetry_GEBCO/`); water 0–30 m deep is analysed (`--max-depth-m` to change) |
| `--auto-shallow` | 04 | no outline: shallow = green reflectance clearly above the deep-water level |
| `--place` | 04 | place name written on the PNG maps |
| `--workers` | 04 | number of CPU cores for the model |

---

## 5. Results

### 5a. What you get after each step

Every step has its own folder, so each intermediate result can be opened and checked (e.g. in QGIS).
Example: [`output_example/Sentinel2_Tidung_20260808/`](output_example/Sentinel2_Tidung_20260808/). File names start with `<name>_`.

```
results/<name>/
├── 02_glint_correction_sentinel2/            (Planet Dove: 03_glint_correction_planetdove/, same 5 steps)
│   ├── step1_read_Sentinel2_L2A_20m/          surface reflectance on one 20 m grid, SCL, AOT, metadata
│   ├── step2_cloud_land_water_masks/          cloud/shadow mask, valid water mask
│   ├── step3_ACOLITE_glint_factors/           glint factors per band (Planet: step3_Hedley_glint_slopes)
│   ├── step4_glint_corrected_reflectance_and_RGB/
│   │                                          corrected reflectance, removed glint,
│   │                                          RGB_before_after_glint_correction.png  <- look here first
│   └── step5_Rrs_input_for_semi_analytical/   Rrs (model input), valid mask, QC_decision.json (PASS/SKIP)
└── 04_semi_analytical_bathymetry_habitat/
    ├── step1_shallow_water_mask/              area that is analysed
    ├── step2_deep_water_Rrs_reference/        deep-water Rrs + the 7x7 windows used (CSV)
    ├── step3_model_2classes_sand_seagrass/    depth, sand/seagrass fraction, fit error, quality flags
    ├── step4_model_3classes_sand_seagrass_coral/  same with coral
    ├── step5_habitat_classes/                 habitat maps (70 % rule and 50 % rule, 2 and 3 classes)
    └── step6_PNG_maps_and_QC/                 11 PNG report maps + final QC_decision.json
```

The full list of every file with its format is in sheet `3_Steps_input_output` of the Excel table.

### 5b. Final results

Final results are in the script 04 output folder:

| File | Content |
|---|---|
| `<name>_step3_2classes_depth_m.tif`, `<name>_step4_3classes_depth_m.tif` | water depth (m) |
| `..._fraction_sand.tif`, `_fraction_seagrass.tif`, `_fraction_coral.tif` | bottom-cover fraction (0–1; values outside 0–1 are kept) |
| `<name>_step5_<k>classes_habitat_class_70pct_rule.tif` | a bottom type wins if its fraction is ≥ 0.70, otherwise mixed |
| `<name>_step5_<k>classes_habitat_class_50pct_rule.tif` | the largest fraction wins |
| `step6_PNG_maps_and_QC/<name>_step6_map_from_step<N>_*.png` | one report map per result (coordinates, place, legend, scale bar, north arrow) |
| `<name>_step6_QC_decision.json` | PASS / SKIP, deep-water reference, list of outputs |

Habitat codes: **1** sand, **2** seagrass, **4** coral, **3** mixed, **9** unclassified (fraction < 0 or > 1), **0** no data.
RGB images before and after glint correction are in step 4 of scripts 02 and 03.

All rasters are GeoTIFF in the scene's UTM projection. The format of every file (data type, bands, pixel size)
is listed in sheet `3_Steps_input_output`.

---

## 6. Automatic operation: PASS / SKIP

Every script ends with a file `<name>_step<N>_QC_decision.json` (in its last step folder, or in the step where it stopped)
and an exit code:

| Exit code | Decision | What the automatic system should do |
|---|---|---|
| `0` | PASS | continue with the next script |
| `2` | SKIP | stop this scene; the reasons are in the QC file |
| other | error | check the log |

A scene is skipped when (details in sheet `5_Automatic_skip_rules`):

* more than 20 % of the area is cloud or cloud shadow;
* Sentinel-2 L2A reflectance is negative over deep water;
* fewer than 1000 Planet pixels are available to train the Hedley glint correction;
* fewer than 3 clean deep-water windows are found;
* the shallow-water mask is empty.

If the Planet Hedley correction is rejected (more than 1 % of water pixels become negative), the scene is not skipped:
the original reflectance is used and a warning is written to the QC file.

---

## 7. Answers to the automation questions

| Question | Answer |
|---|---|
| **Deep-water Rrs without the GUI, and without an operator?** | Yes. Script 04 step 2 picks it automatically: the median of at least 3 clean 7 × 7 pixel windows, far from shallow water and land (2000 m, relaxed step by step to 300 m). The rules for distance, homogeneity and positive values are in sheet `4_Rules_and_thresholds`. Every window used is written to `<name>_step2_deep_water_windows_7x7.csv` for checking. On Tidung Sentinel-2 it matches the delivered reference within 1 %. Fewer than 3 windows → SKIP. |
| **Planet input to the model: Rrs per band (444–666 nm)? 3 m or 21 m?** | Script 03 step 5 writes the model input directly: Rrs = glint-corrected reflectance / π, 6 bands (443, 490, 531, 565, 610, 665 nm), on the native **3 m** grid. No separate preparation script is needed. |
| **Analysis mask: can the standard method and thresholds be used in a new area?** | Script 04 step 1 has three options. Best: a reviewed shallow-water outline (`--shallow-outline`). Without one: `--gebco` (water 0–30 m deep) or `--auto-shallow` (green reflectance above the deep-water level + 0.010). For a new area, the GEBCO or automatic mask should be checked once on the PNG map before it is trusted. |
| **Water type, coefficients and bottom spectra for each area?** | The same for every area, as in all delivered PCIS results: water type case 1, fitted coefficients; 2 classes = coral sand + average seagrass, 3 classes = + average coral. Spectra are in `support_04_semi_analytical_model_V42v2/DATA/bottom_reflectances/`. |
| **Which scenes to skip in automatic operation?** | See [section 6](#6-automatic-operation-pass--skip) and sheet `5_Automatic_skip_rules`. |
| **Source code / GitHub repository?** | This repository. Changes are visible in its commit history. |

---

## 8. Old package → this package

For readers of the earlier `PCIS_2026_Satellite_Bathymetry_Workflow` package (10 November):

| Earlier package | Now |
|---|---|
| Sentinel-2 glint correction (several scripts) | `02_glint_correction_sentinel2.py` |
| Planet Dove glint correction | `03_glint_correction_planetdove.py` (now joins all strips of one date) |
| `prepare_superdove_v42_input.py` | not needed: script 03 step 5 writes the 6-band 3 m Rrs model input |
| `create_shallow_water_only_mask_planet_3m.py`, `create_gebco_shallow_water_mask.py` | script 04 step 1 (`--shallow-outline`, `--gebco`, `--auto-shallow`) |
| Deep-water Rrs from GUI pixel selection or a hand-made `deep_water.csv` | script 04 step 2, automatic (no GUI) |
| Semi-analytical model, habitat maps | script 04 steps 3–6 (same model, unchanged) |
| GUI applications | removed; everything runs from the command line |

---

## 9. Naming rule

| Item | Rule | Example |
|---|---|---|
| Sensor word | `Sentinel2` or `PlanetDove` only | `Sentinel2`, `PlanetDove` |
| Scene name (`--name`) | `<Sensor>_<Area>_<YYYYMMDD>` | `Sentinel2_Tidung_20260808` |
| Script | `<NN>_<what it does>_<sensor>.py` | `03_glint_correction_planetdove.py` |
| Script output folder | `<scene name>/<script name>/` | `PlanetDove_Tidung_20251117/03_glint_correction_planetdove/` |
| Step folder | `step<N>_<what is done>/` | `step2_cloud_land_water_masks/` |
| File | `<scene name>_step<N>_<content>.<ext>` (N = the step folder it is in) | `Sentinel2_Tidung_20260808_step5_Rrs_above_surface_4band_443-665nm_sr-1.tif` |
| Bands in a name | `<n>band_<first>-<last>nm` | `6band_443-665nm` |
| PNG map | `<scene name>_step6_map_from_step<N>_<content>.png` | `..._step6_map_from_step3_2classes_depth_m.png` |

The same product has the same name for both sensors (for example `step5_valid_Rrs_mask.tif`).

---

## 10. Things to know

* **Sentinel-2 Cycle 1** (the optional second, HydroLight-based residual-glint correction) is not included. It is applied only when leftover glint is statistically proven; no delivered PCIS scene passed that test (all decisions `RETAIN_CYCLE0`), so Cycle 0 is the final product everywhere and this pipeline gives the same result without it. It can be added if needed; the HydroLight reference library is in the Drive zip (see [`DATA.md`](DATA.md)).
* **Planet training water.** The delivered Tidung Planet product used a hand-reviewed Hedley training zone. Script 03 uses the published automatic selection and reports a slope-stability check. To reproduce the delivered product exactly, give the reviewed zone with `--training-zone` (included in [`input_example/planetdove_reviewed_Hedley_training_zone/`](input_example/planetdove_reviewed_Hedley_training_zone/)). The difference is 0.0005–0.001 sr⁻¹ in Rrs.
* **Several Planet strips on one date** are joined into one 3 m mosaic in script 03 step 1 (largest strip first; later strips fill gaps and cloudy pixels).
* **Bright reef flats.** Sen2Cor sometimes labels very bright shallow sand as "bare soil" (SCL 5); those pixels are not glint-corrected, as in the delivered results.
* Rrs = reflectance / π is an approximation; results need field validation.

---

## 11. Sources and licences

* Hedley, Harborne & Mumby (2005), *Int. J. Remote Sens.* 26, 2107–2112.
* Vanhellemont (2019), *Remote Sens. Environ.* 225, 175–192 (ACOLITE).
* ACOLITE: <https://github.com/acolite/acolite>, GPL-3; the version used is in [`support_02_ACOLITE_code_and_lookup_tables/VERSION_USED.txt`](support_02_ACOLITE_code_and_lookup_tables/VERSION_USED.txt).
* Semi-analytical model: [`support_04_semi_analytical_model_V42v2/`](support_04_semi_analytical_model_V42v2/).
