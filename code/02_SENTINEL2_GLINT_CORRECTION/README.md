# Sentinel-2 glint correction by code

## Real entry point

`SOURCE/02_RUN_WORKFLOW/run_complete_example.py`

## Required input

Raw examples are stored as one Sentinel-2 L2A ZIP per location/date under `04_DATA/01_RAW/01_SENTINEL2_L2A`. The GUI accepts these ZIP files directly. Code mode can extract and verify one ZIP with `SOURCE/02_RUN_WORKFLOW/extract_sentinel2_zip.py`.

The correction runner uses the prepared scene-resource folder under `04_DATA/02_REFERENCE/03_SENTINEL2_SCENE_RESOURCES`. It uses documented ACOLITE and HydroLight products supplied by this project. It does not replace the external ACOLITE or HydroLight software.

## Run

```bat
python SOURCE\02_RUN_WORKFLOW\run_complete_example.py
```

To inspect a raw ZIP by code, extract it to a short temporary path:

```bat
python SOURCE\02_RUN_WORKFLOW\extract_sentinel2_zip.py "04_DATA\01_RAW\01_SENTINEL2_L2A\01_INDONESIA\01_TIDUNG\20260808_S2A\Sentinel2_Indonesia_Tidung_20260808_L2A.zip" "C:\PCIS_WORK\Tidung"
```

## Produced stages

The real runner writes numbered prepared-input, correction, validation, Rrs and QC outputs. For final delivery, generated products are placed under `05_RESULTS/01_INTERMEDIATE/01_SENTINEL2_20M`; generated output is not retained inside `03_CODE`.
