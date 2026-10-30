# Sentinel-2 Scene Search and Download

This folder contains the reusable code for finding and downloading additional Sentinel-2 scenes from the Copernicus Data Space Ecosystem (CDSE).

## Included program

`SOURCE/sentinel2_search_download.py`

The default study locations are Tidung, Viti Levu, Tongatapu, Funafuti and Savave. The destination is derived from the package location, so moving the complete package does not require editing the script.

## 1. Prepare the package environment

Open Command Prompt in the package root and run:

```bat
SETUP_PYTHON_ENVIRONMENT.bat
```

Show every command option:

```bat
.venv\Scripts\python.exe 03_CODE\01_SENTINEL2_DOWNLOAD\SOURCE\sentinel2_search_download.py --help
```

## 2. Search for scenes first

This example searches for the best Fiji L2A scenes in 2024 without downloading them:

```bat
.venv\Scripts\python.exe 03_CODE\01_SENTINEL2_DOWNLOAD\SOURCE\sentinel2_search_download.py --site Fiji --level L2A --yearly --year 2024 --max-cloud 20
```

Search results are saved as CSV files. Check the scene names, cloud percentage and estimated total size before downloading.

Available site names are `Tidung`, `Fiji`, `Tonga` and `Tuvalu`. Tuvalu includes the Funafuti and Savave search areas.

## 3. Download the reviewed scenes

Create a free CDSE account at `https://dataspace.copernicus.eu/`. Set the credentials only in the current Command Prompt:

```bat
set "CDSE_USER=your-CDSE-email@example.com"
set "CDSE_PASS=your-CDSE-password"
```

Repeat the reviewed search with `--download`:

```bat
.venv\Scripts\python.exe 03_CODE\01_SENTINEL2_DOWNLOAD\SOURCE\sentinel2_search_download.py --site Fiji --level L2A --yearly --year 2024 --max-cloud 20 --download
```

To save the scenes on another disk:

```bat
.venv\Scripts\python.exe 03_CODE\01_SENTINEL2_DOWNLOAD\SOURCE\sentinel2_search_download.py --site Fiji --level L2A --yearly --year 2024 --max-cloud 20 --download --download-dir "D:\Sentinel2"
```

## Search modes

- `--diagnose`: report how many scenes are available.
- `--survey`: find scenes close to configured survey dates.
- `--monthly`: select the lowest-cloud scene for each month.
- `--yearly`: select the lowest-cloud scene for each year.
- `--all-scenes`: select every matching scene. Search and inspect the estimated size before adding `--download`.

## Output

- Scene-search CSV files: `05_RESULTS/00_SENTINEL2_DOWNLOAD_LOGS`.
- Default downloaded scene location: `04_DATA/01_RAW/01_SENTINEL2_L2A`.
- Each successful download is extracted as a Sentinel-2 SAFE scene.

The download code obtains raw Sentinel-2 data only. ACOLITE scene resources required by the supplied glint-correction workflow are prepared in the next processing stage.
