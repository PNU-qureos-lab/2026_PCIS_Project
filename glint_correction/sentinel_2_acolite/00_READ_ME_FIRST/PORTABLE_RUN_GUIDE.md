# Portable run guide

## Run the included example

The recipient does not need ACOLITE or HydroLight to reproduce the included
downstream example. The necessary ACOLITE-derived support rasters, transfer
factors, and all 675 HydroLight simulations are already inside `01_INPUT`.

On Windows, open `02_RUN_WORKFLOW` and double-click `RUN_ME_WINDOWS.bat`, or run:

```powershell
cd 02_RUN_WORKFLOW
python -m pip install -r requirements.txt
python run_complete_example.py
```

The new results appear in `04_GENERATED_OUTPUT`. Compare them with
`03_EXAMPLE_OUTPUT`.

## What B01 does

B01 begins at native 60 m. Step 00 aligns it to the shared 20 m grid. The active
Cycle-0 code then corrects B01 together with B02–B8A using the supplied B01
ACOLITE transfer factors. There is no second B01 correction in the portable
run. The separate B01 extension scripts are retained only to document how B01
was historically added to an older product archive.

## Run a different prepared scene

Create another input directory with the same four numbered subdirectories and
filenames, then run:

```powershell
python run_complete_example.py --input-dir PATH_TO_INPUT --output-dir PATH_TO_OUTPUT --scene SCENE_ID
```

The replacement input must contain aligned BOA bands, ACOLITE-derived B11/B12
selection and transfer factors, a HydroLight lookup table, and the scene's
validation evidence.

## Start from a new Sentinel-2 SAFE product

That is a production run rather than the included portable demonstration. It
requires Sen2Cor L2A data and a working ACOLITE installation to generate new
scene-specific support products. HydroLight is optional only when the supplied
simulation ranges and assumptions are suitable; otherwise new simulations
require a HydroLight licence. The complete production source is preserved in
`05_FULL_SOURCE_CODE/production_code`.
