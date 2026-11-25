#!/usr/bin/env bash
# One-time setup on Linux: creates the conda environment "pcis" and checks every part of the pipeline.
# Needs Miniconda/Miniforge (https://github.com/conda-forge/miniforge). Run:  bash setup_linux.sh
set -e
cd "$(dirname "$0")"
for d in "$(command -v conda >/dev/null && conda info --base)" ~/miniforge3 ~/miniconda3 ~/anaconda3 /opt/conda; do
  [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; break; }
done
command -v conda >/dev/null || { echo "conda not found: install Miniforge first (https://github.com/conda-forge/miniforge)"; exit 1; }
conda env create -y -f environment_linux.yml || conda env update -f environment_linux.yml
conda activate pcis
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pcis_matplotlib"
python - <<'PY'
import sys, rasterio, scipy, numpy, netCDF4, openpyxl, requests
from osgeo import gdal
assert gdal.GetDriverByName("JP2OpenJPEG"), "GDAL cannot read Sentinel-2 .jp2 (install libgdal-jp2openjpeg)"
sys.path.insert(0, "support_02_ACOLITE_code_and_lookup_tables"); import acolite
print("OK  python", sys.version.split()[0], "| rasterio", rasterio.__version__, "| GDAL", gdal.__version__, "| ACOLITE found")
PY
python support_04_semi_analytical_model_V42v2/SOURCE/main.py --help > /dev/null && echo "OK  semi-analytical model (Version 42 v2)"
echo "Setup finished. Next:  bash run_example_linux.sh"
