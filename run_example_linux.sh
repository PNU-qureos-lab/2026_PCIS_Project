#!/usr/bin/env bash
# Runs the whole pipeline on the included example (Tidung, Indonesia):
#   Sentinel-2 2026-08-08 : 02 glint correction -> 04 semi-analytical
#   Planet Dove 2025-11-17: 03 glint correction -> 04 semi-analytical
#     (only if the Planet example from the PCIS Drive zip has been copied in; see DATA.md)
# (01 download is not needed for the example; the scene is in input_example/)
set -e
cd "$(dirname "$0")"
for d in "$(command -v conda >/dev/null && conda info --base)" ~/miniforge3 ~/miniconda3 ~/anaconda3 /opt/conda; do
  [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; break; }
done
command -v conda >/dev/null || { echo "conda not found: install Miniforge first (https://github.com/conda-forge/miniforge)"; exit 1; }
conda activate pcis
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pcis_matplotlib"
W=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
PLACE="Tidung Island, Kepulauan Seribu, Indonesia"

echo "== Sentinel-2 =="
python 02_glint_correction_sentinel2.py \
  --safe input_example/sentinel2_L2A_scene/S2A_MSIL2A_20260808T030151_N0512_R032_T48MXU_20260808T123808.SAFE \
  --aoi 106.4648 -5.8252 106.5401 -5.7748 --name Sentinel2_Tidung_20260808 \
  --out output_example/Sentinel2_Tidung_20260808/02_glint_correction_sentinel2
python 04_semi_analytical_bathymetry_habitat.py --sensor Sentinel2 \
  --glint-dir output_example/Sentinel2_Tidung_20260808/02_glint_correction_sentinel2 --name Sentinel2_Tidung_20260808 \
  --shallow-outline input_example/shallow_water_outline/Sentinel2_Tidung_20260808_shallow_water_outline.tif \
  --place "$PLACE" --workers $W --out output_example/Sentinel2_Tidung_20260808/04_semi_analytical_bathymetry_habitat

PLANET_ZIP=input_example/planetdove_SR_UDM2_delivery/PlanetDove_Tidung_20251117_SR_UDM2_delivery.zip
if [ ! -f "$PLANET_ZIP" ]; then
  echo "== Planet Dove example skipped: $PLANET_ZIP not found."
  echo "   Planet imagery is not on GitHub (Planet licence). Copy 2_raw_PlanetDove_SR_UDM2/PlanetDove_Tidung_20251117_SR_UDM2_delivery.zip"
  echo "   from the PCIS Drive zip (PCIS_2026_DATA.zip) to input_example/planetdove_SR_UDM2_delivery/, then run again. See DATA.md."
  echo "All done. Results: output_example/"; exit 0
fi
echo "== Planet Dove =="
python 03_glint_correction_planetdove.py \
  --planet "$PLANET_ZIP" \
  --date 20251117 --name PlanetDove_Tidung_20251117 \
  --out output_example/PlanetDove_Tidung_20251117/03_glint_correction_planetdove
python 04_semi_analytical_bathymetry_habitat.py --sensor PlanetDove \
  --glint-dir output_example/PlanetDove_Tidung_20251117/03_glint_correction_planetdove --name PlanetDove_Tidung_20251117 \
  --shallow-outline input_example/shallow_water_outline/PlanetDove_Tidung_20251117_shallow_water_outline.tif \
  --place "$PLACE" --workers $W --out output_example/PlanetDove_Tidung_20251117/04_semi_analytical_bathymetry_habitat
echo "All done. Results: output_example/"
