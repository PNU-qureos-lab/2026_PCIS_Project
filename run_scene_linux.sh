#!/usr/bin/env bash
# Runs one scene from glint correction to the final maps, with one command:
#   Sentinel2 : 02 glint correction -> 04 semi-analytical bathymetry and habitat
#   PlanetDove: 03 glint correction -> 04 semi-analytical bathymetry and habitat
# Stops when a script says SKIP (exit code 2). Every step writes its own folder.
#
# Sentinel-2 (one tile: a .SAFE folder or a .SAFE.zip):
#   bash run_scene_linux.sh --sensor Sentinel2 --input S2A_MSIL2A_..._T48MXU_....SAFE.zip \
#        --aoi 106.4648 -5.8252 106.5401 -5.7748 --name Sentinel2_Tidung_20260808 \
#        --shallow-outline outline.tif --place "Tidung Island, Indonesia"
#
# Planet Dove (the delivery .zip; all strips of --date are joined):
#   bash run_scene_linux.sh --sensor PlanetDove --input PlanetDove_Tidung_20251117_SR_UDM2_delivery.zip \
#        --date 20251117 --name PlanetDove_Tidung_20251117 --auto-shallow --place "Tidung Island, Indonesia"
#
# Shallow-water mask for script 04, choose ONE: --shallow-outline <tif> | --gebco <tif> | --auto-shallow
# Optional: --out <folder> (default: results)   --training-zone <tif> (PlanetDove)   --workers <n>
set -e
cd "$(dirname "$0")"

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
SENSOR= INPUT= NAME= DATE= PLACE= OUT=results AOI=() MASK=() EXTRA03=() W=
while [ $# -gt 0 ]; do
  case "$1" in
    --sensor) SENSOR=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --aoi) AOI=("$2" "$3" "$4" "$5"); shift 5 ;;
    --date) DATE=$2; shift 2 ;;
    --place) PLACE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --workers) W=$2; shift 2 ;;
    --shallow-outline|--gebco) MASK=("$1" "$2"); shift 2 ;;
    --auto-shallow) MASK=("$1"); shift ;;
    --training-zone) EXTRA03=("$1" "$2"); shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1"; usage ;;
  esac
done
[ -n "$SENSOR" ] && [ -n "$INPUT" ] && [ -n "$NAME" ] || { echo "need --sensor, --input and --name"; usage; }
[ ${#MASK[@]} -gt 0 ] || { echo "need one mask option: --shallow-outline, --gebco or --auto-shallow"; usage; }
[ -e "$INPUT" ] || { echo "input not found: $INPUT"; exit 1; }

for d in "$(command -v conda >/dev/null && conda info --base)" ~/miniforge3 ~/miniconda3 ~/anaconda3 /opt/conda; do
  [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; break; }
done
command -v conda >/dev/null || { echo "conda not found: run setup_linux.sh first"; exit 1; }
conda activate pcis
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pcis_matplotlib"
[ -n "$W" ] || W=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))

run() {  # run a script; exit code 2 = SKIP -> stop here
  set +e; "$@"; rc=$?; set -e
  if [ $rc -eq 2 ]; then echo "== SKIP: $NAME stopped by $2 (reasons in its QC_decision.json)"; exit 2; fi
  [ $rc -eq 0 ] || { echo "== ERROR in $2 (exit $rc)"; exit $rc; }
}

SCENE="$OUT/$NAME"
if [ "$SENSOR" = Sentinel2 ]; then
  [ ${#AOI[@]} -eq 4 ] || { echo "Sentinel2 needs --aoi LON_MIN LAT_MIN LON_MAX LAT_MAX"; exit 1; }
  GLINT="$SCENE/02_glint_correction_sentinel2"
  echo "== 02 glint correction (Sentinel-2) -> $GLINT"
  run python 02_glint_correction_sentinel2.py --safe "$INPUT" --aoi "${AOI[@]}" --name "$NAME" --out "$GLINT"
elif [ "$SENSOR" = PlanetDove ]; then
  [ -n "$DATE" ] || { echo "PlanetDove needs --date YYYYMMDD"; exit 1; }
  GLINT="$SCENE/03_glint_correction_planetdove"
  echo "== 03 glint correction (Planet Dove) -> $GLINT"
  run python 03_glint_correction_planetdove.py --planet "$INPUT" --date "$DATE" --name "$NAME" --out "$GLINT" "${EXTRA03[@]}"
else
  echo "--sensor must be Sentinel2 or PlanetDove"; exit 1
fi

MODEL="$SCENE/04_semi_analytical_bathymetry_habitat"
echo "== 04 semi-analytical bathymetry and habitat -> $MODEL"
run python 04_semi_analytical_bathymetry_habitat.py --sensor "$SENSOR" --glint-dir "$GLINT" --name "$NAME" \
    "${MASK[@]}" --place "$PLACE" --workers "$W" --out "$MODEL"

echo "== PASS: $NAME done. Results of every step:"
find "$SCENE" -mindepth 2 -maxdepth 2 -type d -name 'step*' | sort | sed 's/^/   /'
echo "   Final maps: $MODEL/step6_PNG_maps_and_QC/"
